"""
Monolithic Monte Carlo normalization kernel.

This file owns the heavy-weight selection-normalization computation used by
the production `log_prob` path. The implementation stays entirely inside
`numba` so every evaluation can reuse one compiled kernel and one fixed random
basis without falling back to SciPy helpers.
"""

from __future__ import annotations

import math

import numba as nb
import numpy as np

from .primitives import (
    gamma_population_mean,
    interp1d_clip,
    interp_sigma_unit_clip,
    LOG10_2PI,
    mu_r,
    p_find,
    phi_standard,
    skewnorm_sample,
    theta_dimension_for_gamma_mode,
    unpack_model_theta,
    theta_ein_arcsec,
    truncnorm_sample,
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
    Draw one parent-population galaxy from the fixed Monte Carlo basis.

    The normalization kernel and the FP-prior summary kernel need the same
    latent galaxy state: `(zd, m*, n, logRe, delta_r, m_R, gamma)`. Centralizing
    that draw here prevents the two kernels from drifting apart scientifically.
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
        delta_r = re_draw - mu_r_draw
        log_enclosed_mass = mu5_0 + beta5 * mstar_shift + xi5 * delta_r + sigma5 * nrm[6]
    else:
        n_value = n_fixed
        mu_r_draw = mu_r(mstar, n_value, use_sersic_index, mu_r0, beta_r, nu_r, stellar_mass_pivot)
        re_draw = mu_r_draw + sigma_r * nrm[4]
        delta_r = re_draw - mu_r_draw
        log_enclosed_mass = mu5_0 + beta5 * mstar_shift + xi5 * delta_r + sigma5 * nrm[5]

    sigma_star_shift9p0 = mstar - LOG10_2PI - 2.0 * re_draw - 9.0
    mu_gamma = gamma_population_mean(
        mu_gamma_0,
        beta_gamma,
        xi_gamma,
        beta_sigma_star_gamma,
        mstar_shift,
        delta_r,
        sigma_star_shift9p0,
        gamma_mode_code,
    )
    gamma = truncnorm_sample(
        mu_gamma,
        sigma_gamma,
        gamma_trunc_low,
        gamma_trunc_high,
        nrm[7],
    )

    return zd, mstar, n_value, re_draw, delta_r, log_enclosed_mass, gamma


@nb.njit(cache=True, inline="always")
def _accumulate_fp_ols_summary(
    fp_summary: np.ndarray,
    mstar: float,
    log_sigma_model: float,
    pivot_mstar: float,
) -> None:
    """
    Update the sufficient statistics for the 1D sigma-logM* regression.

    The regression model is:
        log10(sigma) = a + b * (mstar - pivot)
    Only the aggregate moments are needed downstream, so the hot kernel never
    stores the full synthetic population in memory.
    """

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
    cs_gamma_grid: np.ndarray,
    cs_over_theta: np.ndarray,
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
    stellar_mass_pivot: float = 11.4,
    mass_log_physical_offset: float = 0.0,
) -> float:
    """
    Estimate the selection normalization for one hyper-parameter vector.

    The kernel mirrors the reference implementation's structure:
    one loop over the fixed random basis, profile-specific draws injected by
    scalar constants, and a final average of discovery-weighted cross-sections.
    """

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
    ) = unpack_model_theta(theta, gamma_mode_code)

    if sigma5 <= 0.0 or sigma_gamma <= 0.0 or sigma_zs <= 0.0:
        return 0.0

    n_samples = base_normals.shape[0]
    z0 = (0.0 - mu_zs) / sigma_zs
    trunc_den = 1.0 - phi_standard(z0)
    if trunc_den <= 0.0:
        return 0.0
    acc = 0.0

    for i in nb.prange(n_samples):
        nrm = base_normals[i]
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
        if zd <= 0.0 or zs <= 0.0 or zs <= zd:
            continue

        if not math.isfinite(gamma):
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

        cs = interp1d_clip(gamma, cs_gamma_grid, cs_over_theta)
        area = math.pi * (cs * theta_e) ** 2
        pf = p_find(theta_e, theta0, loga)
        acc += (1.0 / trunc_den) * pf * area

    return acc / n_samples


@nb.njit(cache=True, fastmath=True)
def _population_summary_mc_serial_reference_numba(
    theta: np.ndarray,
    base_normals: np.ndarray,
    cs_gamma_grid: np.ndarray,
    cs_over_theta: np.ndarray,
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
    fp_fit_mstar_min: float,
    fp_pivot_mstar: float,
    fp_gamma_axis: np.ndarray,
    fp_zd_axis: np.ndarray,
    fp_log_re_kpc_axis: np.ndarray,
    fp_n_axis: np.ndarray,
    fp_sigma_unit_grid: np.ndarray,
    fp_has_n_axis: int,
    stellar_mass_pivot: float = 11.4,
    mass_log_physical_offset: float = 0.0,
) -> tuple[float, np.ndarray]:
    """
    Legacy serial reference for the FP population-summary computation.

    Why this function is intentionally kept:
    - it captures the original, easy-to-read serial implementation used when
      the optional FP prior was first integrated
    - it provides a trusted numerical reference for regression tests while the
      production path evolves toward a parallel implementation
    - it is not part of the production hot path and should therefore remain
      clearly marked as legacy reference code
    """

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
    ) = unpack_model_theta(theta, gamma_mode_code)

    if sigma5 <= 0.0 or sigma_gamma <= 0.0 or sigma_zs <= 0.0:
        return 0.0, fp_summary

    n_samples = base_normals.shape[0]
    z0 = (0.0 - mu_zs) / sigma_zs
    trunc_den = 1.0 - phi_standard(z0)
    if trunc_den <= 0.0:
        return 0.0, fp_summary

    acc = 0.0
    inv_trunc_den = 1.0 / trunc_den
    for i in range(n_samples):
        nrm = base_normals[i]
        zd, mstar, n_value, re_draw, delta_r, log_enclosed_mass, gamma = _draw_population_state(
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
                    _accumulate_fp_ols_summary(
                        fp_summary,
                        mstar,
                        log_sigma_model,
                        fp_pivot_mstar,
                    )

        zs = mu_zs + sigma_zs * nrm[1]
        if zd <= 0.0 or zs <= 0.0 or zs <= zd or (not math.isfinite(gamma)):
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

        cs = interp1d_clip(gamma, cs_gamma_grid, cs_over_theta)
        area = math.pi * (cs * theta_e) ** 2
        pf = p_find(theta_e, theta0, loga)
        acc += inv_trunc_den * pf * area

    return acc / n_samples, fp_summary


@nb.njit(cache=True, parallel=True, fastmath=True)
def population_summary_mc_numba(
    theta: np.ndarray,
    base_normals: np.ndarray,
    cs_gamma_grid: np.ndarray,
    cs_over_theta: np.ndarray,
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
    fp_fit_mstar_min: float,
    fp_pivot_mstar: float,
    fp_gamma_axis: np.ndarray,
    fp_zd_axis: np.ndarray,
    fp_log_re_kpc_axis: np.ndarray,
    fp_n_axis: np.ndarray,
    fp_sigma_unit_grid: np.ndarray,
    fp_has_n_axis: int,
    stellar_mass_pivot: float = 11.4,
    mass_log_physical_offset: float = 0.0,
) -> tuple[float, np.ndarray]:
    """
    Estimate normalization and FP summary statistics in one parallel population pass.

    Design notes:
    - the parent-population draw remains shared between normalization and FP fit
    - each sample writes into its own summary row to avoid cross-thread races
    - a tiny serial reduction at the end collapses the row buffer back to the
      compact OLS sufficient-statistics vector expected by the Python solver
    """

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
    ) = unpack_model_theta(theta, gamma_mode_code)

    if sigma5 <= 0.0 or sigma_gamma <= 0.0 or sigma_zs <= 0.0:
        return 0.0, fp_summary

    n_samples = base_normals.shape[0]
    z0 = (0.0 - mu_zs) / sigma_zs
    trunc_den = 1.0 - phi_standard(z0)
    if trunc_den <= 0.0:
        return 0.0, fp_summary

    inv_trunc_den = 1.0 / trunc_den
    fp_summary_rows = np.zeros((n_samples, FP_OLS_SUMMARY_SIZE), dtype=np.float64)
    acc = 0.0

    for i in nb.prange(n_samples):
        row = fp_summary_rows[i]
        nrm = base_normals[i]
        zd, mstar, n_value, re_draw, delta_r, log_enclosed_mass, gamma = _draw_population_state(
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
                    _accumulate_fp_ols_summary(
                        row,
                        mstar,
                        log_sigma_model,
                        fp_pivot_mstar,
                    )

        zs = mu_zs + sigma_zs * nrm[1]
        if zd <= 0.0 or zs <= 0.0 or zs <= zd or (not math.isfinite(gamma)):
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

        cs = interp1d_clip(gamma, cs_gamma_grid, cs_over_theta)
        area = math.pi * (cs * theta_e) ** 2
        pf = p_find(theta_e, theta0, loga)
        acc += inv_trunc_den * pf * area

    for i in range(n_samples):
        row = fp_summary_rows[i]
        for summary_index in range(FP_OLS_SUMMARY_SIZE):
            fp_summary[summary_index] += row[summary_index]

    return acc / n_samples, fp_summary
