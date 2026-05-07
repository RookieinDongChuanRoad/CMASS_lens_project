"""
CMASS production kernels for the Numba backend.

The functions in this module implement the current canonical CMASS semantics:
per-lens likelihood integration over `(gamma, mstar)`, Monte Carlo selection
normalization, optional Fundamental Plane summary statistics, and canonical
`theta_E x gamma` cross-section interpolation.  They do not know about YAML,
run directories, emcee, or output files; those framework concerns stay in the
runner and sampler layers.
"""

from __future__ import annotations

import math

import numba as nb
import numpy as np

from .primitives import (
    LOG10_2PI,
    cmass_gamma_population_mean,
    interp_cross_section_theta_gamma,
    interp_sigma_unit_clip,
    mu_r,
    normal_pdf,
    p_find,
    phi_standard,
    skewnorm_sample,
    theta_dimension_for_gamma_mode,
    theta_ein_arcsec,
    trapezoid_1d,
    truncated_normal_pdf_nonneg,
    truncnorm_sample,
    unpack_cmass_theta,
)


FP_OLS_SUMMARY_SIZE = 6
FP_OLS_COUNT_INDEX = 0
FP_OLS_SUM_X1_INDEX = 1
FP_OLS_SUM_X1X1_INDEX = 2
FP_OLS_SUM_Y_INDEX = 3
FP_OLS_SUM_X1Y_INDEX = 4
FP_OLS_SUM_YY_INDEX = 5


@nb.njit(cache=True, inline="always")
def _draw_population_state(
    nrm: np.ndarray,
    mu_d: float,
    sigma_d: float,
    mass_function_loc: float,
    mass_function_scale: float,
    mass_function_alpha: float,
    mu_r0: float,
    beta_r: float,
    sigma_r: float,
    nu_r: float,
    use_sersic_index: int,
    n_fixed: float,
    mu_n0: float,
    beta_n: float,
    sigma_n: float,
    mu5_0: float,
    beta5: float,
    xi5: float,
    sigma5: float,
    mu_gamma_0: float,
    beta_gamma: float,
    xi_gamma: float,
    beta_sigma_star_gamma: float,
    sigma_gamma: float,
    gamma_trunc_low: float,
    gamma_trunc_high: float,
    gamma_mode_code: int,
    stellar_mass_pivot: float,
) -> tuple[float, float, float, float, float, float, float]:
    """
    Map one fixed normal row into a CMASS parent-population state.

    The same latent draw feeds normalization and FP summary accumulation.  That
    sharing matters scientifically: the FP prior must describe the selected
    parent population implied by exactly the same hyper-parameters as the
    selection normalization.
    """

    zd = mu_d + sigma_d * nrm[0]
    mstar = skewnorm_sample(
        mass_function_loc,
        mass_function_scale,
        mass_function_alpha,
        nrm[2],
        nrm[3],
    )
    mstar_shift = mstar - stellar_mass_pivot

    if use_sersic_index == 1:
        logn = mu_n0 + beta_n * mstar_shift + sigma_n * nrm[4]
        n_value = 10.0**logn
        mu_r_draw = mu_r(mstar, n_value, use_sersic_index, mu_r0, beta_r, nu_r, stellar_mass_pivot)
        re_draw = mu_r_draw + sigma_r * nrm[5]
        mass_noise = nrm[6]
    else:
        n_value = n_fixed
        mu_r_draw = mu_r(mstar, n_value, use_sersic_index, mu_r0, beta_r, nu_r, stellar_mass_pivot)
        re_draw = mu_r_draw + sigma_r * nrm[4]
        mass_noise = nrm[5]

    delta_r = re_draw - mu_r_draw
    log_enclosed_mass = mu5_0 + beta5 * mstar_shift + xi5 * delta_r + sigma5 * mass_noise
    sigma_star_shift9p0 = mstar - LOG10_2PI - 2.0 * re_draw - 9.0
    mu_gamma = cmass_gamma_population_mean(
        mu_gamma_0,
        beta_gamma,
        xi_gamma,
        beta_sigma_star_gamma,
        mstar_shift,
        delta_r,
        sigma_star_shift9p0,
        gamma_mode_code,
    )
    gamma = truncnorm_sample(mu_gamma, sigma_gamma, gamma_trunc_low, gamma_trunc_high, nrm[7])
    return zd, mstar, n_value, re_draw, delta_r, log_enclosed_mass, gamma


@nb.njit(cache=True, inline="always")
def _accumulate_fp_ols_summary(
    fp_summary: np.ndarray,
    mstar: float,
    log_sigma_model: float,
    pivot_mstar: float,
) -> None:
    """Accumulate one row of 1D FP sufficient statistics in-place."""

    x1 = mstar - pivot_mstar
    fp_summary[FP_OLS_COUNT_INDEX] += 1.0
    fp_summary[FP_OLS_SUM_X1_INDEX] += x1
    fp_summary[FP_OLS_SUM_X1X1_INDEX] += x1 * x1
    fp_summary[FP_OLS_SUM_Y_INDEX] += log_sigma_model
    fp_summary[FP_OLS_SUM_X1Y_INDEX] += x1 * log_sigma_model
    fp_summary[FP_OLS_SUM_YY_INDEX] += log_sigma_model * log_sigma_model


@nb.njit(cache=True, parallel=True, fastmath=True)
def normalization_mc_numba(
    theta: np.ndarray,
    base_normals: np.ndarray,
    cs_theta_e_axis: np.ndarray,
    cs_gamma_grid: np.ndarray,
    cs_cross_section_grid: np.ndarray,
    z_grid: np.ndarray,
    chi_kpc_grid: np.ndarray,
    mu_d: float,
    sigma_d: float,
    mass_function_loc: float,
    mass_function_scale: float,
    mass_function_alpha: float,
    mu_r0: float,
    beta_r: float,
    sigma_r: float,
    nu_r: float,
    use_sersic_index: int,
    n_fixed: float,
    mu_n0: float,
    beta_n: float,
    sigma_n: float,
    gamma_trunc_low: float,
    gamma_trunc_high: float,
    mass_radius_kpc: float,
    gamma_mode_code: int,
    stellar_mass_pivot: float,
    mass_log_physical_offset: float,
) -> float:
    """Estimate the CMASS selection normalization for one theta vector."""

    if theta.shape[0] != theta_dimension_for_gamma_mode(gamma_mode_code):
        return 0.0

    (
        mu5_0,
        beta5,
        xi5,
        sigma5,
        mu_gamma_0,
        beta_gamma,
        xi_gamma,
        beta_sigma_star_gamma,
        sigma_gamma,
        mu_zs,
        sigma_zs,
        theta0,
        loga,
    ) = unpack_cmass_theta(theta, gamma_mode_code)

    if sigma5 <= 0.0 or sigma_gamma <= 0.0 or sigma_zs <= 0.0:
        return 0.0

    trunc_den = 1.0 - phi_standard((0.0 - mu_zs) / sigma_zs)
    if trunc_den <= 0.0:
        return 0.0

    n_samples = base_normals.shape[0]
    inv_trunc_den = 1.0 / trunc_den
    total_weight = 0.0
    for sample_index in nb.prange(n_samples):
        nrm = base_normals[sample_index]
        zd, _mstar, _n_value, _re_draw, _delta_r, log_enclosed_mass, gamma = _draw_population_state(
            nrm,
            mu_d,
            sigma_d,
            mass_function_loc,
            mass_function_scale,
            mass_function_alpha,
            mu_r0,
            beta_r,
            sigma_r,
            nu_r,
            use_sersic_index,
            n_fixed,
            mu_n0,
            beta_n,
            sigma_n,
            mu5_0,
            beta5,
            xi5,
            sigma5,
            mu_gamma_0,
            beta_gamma,
            xi_gamma,
            beta_sigma_star_gamma,
            sigma_gamma,
            gamma_trunc_low,
            gamma_trunc_high,
            gamma_mode_code,
            stellar_mass_pivot,
        )
        zs = mu_zs + sigma_zs * nrm[1]
        if zd <= 0.0 or zs <= zd or not math.isfinite(gamma):
            continue
        theta_e = theta_ein_arcsec(
            zd,
            zs,
            log_enclosed_mass,
            gamma,
            z_grid,
            chi_kpc_grid,
            mass_radius_kpc,
            mass_log_physical_offset,
        )
        if theta_e <= 0.0:
            continue
        cross_section = interp_cross_section_theta_gamma(
            theta_e,
            gamma,
            cs_theta_e_axis,
            cs_gamma_grid,
            cs_cross_section_grid,
        )
        if cross_section <= 0.0:
            continue
        total_weight += inv_trunc_den * p_find(theta_e, theta0, loga) * cross_section

    return total_weight / n_samples


@nb.njit(cache=True, parallel=True, fastmath=True)
def population_summary_mc_numba(
    theta: np.ndarray,
    base_normals: np.ndarray,
    cs_theta_e_axis: np.ndarray,
    cs_gamma_grid: np.ndarray,
    cs_cross_section_grid: np.ndarray,
    z_grid: np.ndarray,
    chi_kpc_grid: np.ndarray,
    mu_d: float,
    sigma_d: float,
    mass_function_loc: float,
    mass_function_scale: float,
    mass_function_alpha: float,
    mu_r0: float,
    beta_r: float,
    sigma_r: float,
    nu_r: float,
    use_sersic_index: int,
    n_fixed: float,
    mu_n0: float,
    beta_n: float,
    sigma_n: float,
    gamma_trunc_low: float,
    gamma_trunc_high: float,
    mass_radius_kpc: float,
    gamma_mode_code: int,
    stellar_mass_pivot: float,
    mass_log_physical_offset: float,
    fp_fit_mstar_min: float,
    fp_pivot_mstar: float,
    fp_gamma_axis: np.ndarray,
    fp_zd_axis: np.ndarray,
    fp_log_re_kpc_axis: np.ndarray,
    fp_n_axis: np.ndarray,
    fp_sigma_unit_grid: np.ndarray,
    fp_has_n_axis: int,
) -> tuple[float, np.ndarray]:
    """Estimate CMASS normalization and FP summary statistics in one pass."""

    fp_summary = np.zeros(FP_OLS_SUMMARY_SIZE, dtype=np.float64)
    if theta.shape[0] != theta_dimension_for_gamma_mode(gamma_mode_code):
        return 0.0, fp_summary

    (
        mu5_0,
        beta5,
        xi5,
        sigma5,
        mu_gamma_0,
        beta_gamma,
        xi_gamma,
        beta_sigma_star_gamma,
        sigma_gamma,
        mu_zs,
        sigma_zs,
        theta0,
        loga,
    ) = unpack_cmass_theta(theta, gamma_mode_code)

    if sigma5 <= 0.0 or sigma_gamma <= 0.0 or sigma_zs <= 0.0:
        return 0.0, fp_summary

    trunc_den = 1.0 - phi_standard((0.0 - mu_zs) / sigma_zs)
    if trunc_den <= 0.0:
        return 0.0, fp_summary

    n_samples = base_normals.shape[0]
    inv_trunc_den = 1.0 / trunc_den
    summary_rows = np.zeros((n_samples, FP_OLS_SUMMARY_SIZE), dtype=np.float64)
    total_weight = 0.0

    for sample_index in nb.prange(n_samples):
        row = summary_rows[sample_index]
        nrm = base_normals[sample_index]
        zd, mstar, n_value, re_draw, _delta_r, log_enclosed_mass, gamma = _draw_population_state(
            nrm,
            mu_d,
            sigma_d,
            mass_function_loc,
            mass_function_scale,
            mass_function_alpha,
            mu_r0,
            beta_r,
            sigma_r,
            nu_r,
            use_sersic_index,
            n_fixed,
            mu_n0,
            beta_n,
            sigma_n,
            mu5_0,
            beta5,
            xi5,
            sigma5,
            mu_gamma_0,
            beta_gamma,
            xi_gamma,
            beta_sigma_star_gamma,
            sigma_gamma,
            gamma_trunc_low,
            gamma_trunc_high,
            gamma_mode_code,
            stellar_mass_pivot,
        )
        if zd > 0.0 and math.isfinite(gamma) and mstar > fp_fit_mstar_min:
            sigma_unit = interp_sigma_unit_clip(
                gamma,
                zd,
                re_draw,
                n_value,
                fp_gamma_axis,
                fp_zd_axis,
                fp_log_re_kpc_axis,
                fp_n_axis,
                fp_sigma_unit_grid,
                fp_has_n_axis,
            )
            if sigma_unit > 0.0 and math.isfinite(sigma_unit):
                log_sigma_model = 0.5 * (math.log10(sigma_unit) + log_enclosed_mass)
                if math.isfinite(log_sigma_model):
                    _accumulate_fp_ols_summary(row, mstar, log_sigma_model, fp_pivot_mstar)

        zs = mu_zs + sigma_zs * nrm[1]
        if zd <= 0.0 or zs <= zd or not math.isfinite(gamma):
            continue
        theta_e = theta_ein_arcsec(
            zd,
            zs,
            log_enclosed_mass,
            gamma,
            z_grid,
            chi_kpc_grid,
            mass_radius_kpc,
            mass_log_physical_offset,
        )
        if theta_e <= 0.0:
            continue
        cross_section = interp_cross_section_theta_gamma(
            theta_e,
            gamma,
            cs_theta_e_axis,
            cs_gamma_grid,
            cs_cross_section_grid,
        )
        if cross_section <= 0.0:
            continue
        total_weight += inv_trunc_den * p_find(theta_e, theta0, loga) * cross_section

    for sample_index in range(n_samples):
        for summary_index in range(FP_OLS_SUMMARY_SIZE):
            fp_summary[summary_index] += summary_rows[sample_index, summary_index]

    return total_weight / n_samples, fp_summary


@nb.njit(cache=True, parallel=True, fastmath=True)
def log_likelihood_lenses_numba(
    theta: np.ndarray,
    z_grid: np.ndarray,
    chi_kpc_grid: np.ndarray,
    cs_theta_e_axis: np.ndarray,
    cs_gamma_grid: np.ndarray,
    cs_cross_section_grid: np.ndarray,
    mass_grid_int: np.ndarray,
    dmass_dthetaein_grid_int: np.ndarray,
    s2_grid_int: np.ndarray,
    has_s2: np.ndarray,
    num_sigma: np.ndarray,
    sigma_obs: np.ndarray,
    sigma_err: np.ndarray,
    zd: np.ndarray,
    zs: np.ndarray,
    p_zd_fixed: np.ndarray,
    mstar_grid: np.ndarray,
    mstar_shift11p4: np.ndarray,
    sigma_star_shift9p0_grid: np.ndarray,
    mstar_integrand_base: np.ndarray,
    delta_r_grid: np.ndarray,
    gamma_grid_int: np.ndarray,
    mass_radius_kpc: float,
    gamma_mode_code: int,
    mass_log_physical_offset: float,
) -> float:
    """Evaluate the complete CMASS sample log-likelihood in one kernel."""

    if theta.shape[0] != theta_dimension_for_gamma_mode(gamma_mode_code):
        return -np.inf

    (
        mu5_0,
        beta5,
        xi5,
        sigma5,
        mu_gamma_0,
        beta_gamma,
        xi_gamma,
        beta_sigma_star_gamma,
        sigma_gamma,
        mu_zs,
        sigma_zs,
        theta0,
        loga,
    ) = unpack_cmass_theta(theta, gamma_mode_code)

    if sigma5 <= 0.0 or sigma_gamma <= 0.0 or sigma_zs <= 0.0:
        return -np.inf

    n_lens = zd.shape[0]
    n_gamma = gamma_grid_int.shape[0]
    n_mstar = mstar_grid.shape[1]
    ll_terms = np.zeros(n_lens, dtype=np.float64)
    valid = np.ones(n_lens, dtype=np.int64)

    for lens_index in nb.prange(n_lens):
        p_zd = p_zd_fixed[lens_index]
        p_zs = truncated_normal_pdf_nonneg(zs[lens_index], mu_zs, sigma_zs)
        if p_zd <= 0.0 or p_zs <= 0.0:
            valid[lens_index] = 0
            continue

        gamma_integrand = np.zeros(n_gamma, dtype=np.float64)
        for gamma_index in range(n_gamma):
            gamma = gamma_grid_int[gamma_index]
            log_enclosed_mass = mass_grid_int[lens_index, gamma_index]
            jacobian = abs(dmass_dthetaein_grid_int[lens_index, gamma_index])
            if jacobian <= 0.0:
                continue

            theta_e = theta_ein_arcsec(
                zd[lens_index],
                zs[lens_index],
                log_enclosed_mass,
                gamma,
                z_grid,
                chi_kpc_grid,
                mass_radius_kpc,
                mass_log_physical_offset,
            )
            if theta_e <= 0.0:
                continue

            cross_section = interp_cross_section_theta_gamma(
                theta_e,
                gamma,
                cs_theta_e_axis,
                cs_gamma_grid,
                cs_cross_section_grid,
            )
            find_probability = p_find(theta_e, theta0, loga)
            if cross_section <= 0.0 or find_probability <= 0.0:
                continue

            sigma_probability = 1.0
            if num_sigma[lens_index] > 0:
                if has_s2[lens_index] == 0:
                    sigma_probability = 0.0
                else:
                    sigma_model = math.sqrt(max(s2_grid_int[lens_index, gamma_index] * (10.0**log_enclosed_mass), 1.0e-30))
                    for sigma_index in range(num_sigma[lens_index]):
                        sigma_probability *= normal_pdf(
                            sigma_obs[lens_index, sigma_index],
                            sigma_model,
                            sigma_err[lens_index, sigma_index],
                        )
            if sigma_probability <= 0.0:
                continue

            mstar_integrand = np.zeros(n_mstar, dtype=np.float64)
            for mstar_index in range(n_mstar):
                fixed_base = mstar_integrand_base[lens_index, mstar_index]
                if fixed_base <= 0.0:
                    continue
                mu5 = (
                    mu5_0
                    + beta5 * mstar_shift11p4[lens_index, mstar_index]
                    + xi5 * delta_r_grid[lens_index, mstar_index]
                )
                mu_gamma = cmass_gamma_population_mean(
                    mu_gamma_0,
                    beta_gamma,
                    xi_gamma,
                    beta_sigma_star_gamma,
                    mstar_shift11p4[lens_index, mstar_index],
                    delta_r_grid[lens_index, mstar_index],
                    sigma_star_shift9p0_grid[lens_index, mstar_index],
                    gamma_mode_code,
                )
                mstar_integrand[mstar_index] = (
                    fixed_base
                    * normal_pdf(log_enclosed_mass, mu5, sigma5)
                    * normal_pdf(gamma, mu_gamma, sigma_gamma)
                )

            integrated_mstar = trapezoid_1d(mstar_integrand, mstar_grid[lens_index])
            gamma_integrand[gamma_index] = (
                integrated_mstar
                * p_zd
                * p_zs
                * find_probability
                * cross_section
                * jacobian
                * sigma_probability
            )

        lens_integral = trapezoid_1d(gamma_integrand, gamma_grid_int)
        if lens_integral <= 0.0 or not math.isfinite(lens_integral):
            valid[lens_index] = 0
            continue
        ll_terms[lens_index] = math.log(lens_integral)

    total = 0.0
    for lens_index in range(n_lens):
        if valid[lens_index] == 0:
            return -np.inf
        total += ll_terms[lens_index]
    return total


__all__ = [
    "FP_OLS_COUNT_INDEX",
    "FP_OLS_SUM_X1X1_INDEX",
    "FP_OLS_SUM_X1Y_INDEX",
    "FP_OLS_SUM_X1_INDEX",
    "FP_OLS_SUM_Y_INDEX",
    "FP_OLS_SUM_YY_INDEX",
    "log_likelihood_lenses_numba",
    "normalization_mc_numba",
    "population_summary_mc_numba",
]
