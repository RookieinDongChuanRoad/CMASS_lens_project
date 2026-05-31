"""CMASS posterior assembly and model-owned Numba kernels.

This module is the complete CMASS posterior implementation.  It keeps the
scientific posterior structure and the private fused Numba kernels together so
there is a single model-owned file to audit.  Shared backend code still owns
execution mechanics, diagnostics, and reusable primitive kernels.
"""

from __future__ import annotations

import math
from time import perf_counter

import numba as nb
import numpy as np

from statistical_sl.numerics.numba.kernels.constants import LOG10_2PI, LOG10_4
from statistical_sl.numerics.numba.kernels.distributions import (
    normal_pdf,
    phi_standard,
    skewnorm_sample,
    truncnorm_sample,
)
from statistical_sl.numerics.numba.kernels.fundamental_plane import (
    FP_OLS_COUNT_INDEX,
    FP_OLS_SUM_X1X1_INDEX,
    FP_OLS_SUM_X1Y_INDEX,
    FP_OLS_SUM_X1_INDEX,
    FP_OLS_SUM_Y_INDEX,
    FP_OLS_SUM_YY_INDEX,
    FP_OLS_SUMMARY_SIZE,
    accumulate_fp_ols_summary,
)
from statistical_sl.numerics.numba.kernels.integration import trapezoid_1d
from statistical_sl.numerics.numba.kernels.interpolation import interp_sigma_unit_clip
from statistical_sl.numerics.numba.kernels.lensing import theta_ein_arcsec
from statistical_sl.numerics.numba.kernels.selection_likelihood import (
    observed_sigma_likelihood,
    policy_cross_section_find_weight,
    sigma_model_from_s2,
    truncated_nonnegative_source_redshift_density,
)
from statistical_sl.inference.types import CompiledModel
from statistical_sl.inference.diagnostics import build_reject_result, build_timing_blob


GAMMA_MODE_DEPENDENT_CODE = 0
GAMMA_MODE_INDEPENDENT_CODE = 1
GAMMA_MODE_SIGMA_STAR_DEPENDENT_CODE = 2


@nb.njit(cache=True)
def mu_r(
    mstar: float,
    n_value: float,
    use_sersic_index: int,
    mu_r0: float,
    beta_r: float,
    nu_r: float,
    stellar_mass_pivot: float,
) -> float:
    """Return the active parent size-relation mean."""

    value = mu_r0 + beta_r * (mstar - stellar_mass_pivot)
    if use_sersic_index == 1:
        value += nu_r * (math.log10(max(n_value, 1.0e-12)) - LOG10_4)
    return value


@nb.njit(cache=True, inline="always")
def theta_dimension_for_gamma_mode(gamma_mode_code: int) -> int:
    """Return the sampled dimension for the active density-slope mode."""

    if gamma_mode_code == GAMMA_MODE_DEPENDENT_CODE:
        return 12
    if gamma_mode_code == GAMMA_MODE_INDEPENDENT_CODE:
        return 10
    if gamma_mode_code == GAMMA_MODE_SIGMA_STAR_DEPENDENT_CODE:
        return 11
    return -1


@nb.njit(cache=True, inline="always")
def unpack_cmass_theta(theta: np.ndarray, gamma_mode_code: int) -> tuple[float, ...]:
    """Return one fixed scalar tuple for the active CMASS parameterization."""

    mu5_0 = theta[0]
    beta5 = theta[1]
    xi5 = theta[2]
    sigma5 = theta[3]
    mu_gamma_0 = theta[4]
    if gamma_mode_code == GAMMA_MODE_DEPENDENT_CODE:
        beta_gamma = theta[5]
        xi_gamma = theta[6]
        beta_sigma_star_gamma = 0.0
        sigma_gamma = theta[7]
        mu_zs = theta[8]
        sigma_zs = theta[9]
        theta0 = theta[10]
        loga = theta[11]
    elif gamma_mode_code == GAMMA_MODE_INDEPENDENT_CODE:
        beta_gamma = 0.0
        xi_gamma = 0.0
        beta_sigma_star_gamma = 0.0
        sigma_gamma = theta[5]
        mu_zs = theta[6]
        sigma_zs = theta[7]
        theta0 = theta[8]
        loga = theta[9]
    else:
        beta_gamma = 0.0
        xi_gamma = 0.0
        beta_sigma_star_gamma = theta[5]
        sigma_gamma = theta[6]
        mu_zs = theta[7]
        sigma_zs = theta[8]
        theta0 = theta[9]
        loga = theta[10]

    return (
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
    )


@nb.njit(cache=True, inline="always")
def cmass_gamma_population_mean(
    mu_gamma_0: float,
    beta_gamma: float,
    xi_gamma: float,
    beta_sigma_star_gamma: float,
    mstar_shift11p4: float,
    delta_r: float,
    sigma_star_shift9p0: float,
    gamma_mode_code: int,
) -> float:
    """Return the CMASS conditional density-slope mean for the active mode."""

    if gamma_mode_code == GAMMA_MODE_DEPENDENT_CODE:
        return mu_gamma_0 + beta_gamma * mstar_shift11p4 + xi_gamma * delta_r
    if gamma_mode_code == GAMMA_MODE_SIGMA_STAR_DEPENDENT_CODE:
        return mu_gamma_0 + beta_sigma_star_gamma * sigma_star_shift9p0
    return mu_gamma_0


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


@nb.njit(cache=True, parallel=True, fastmath=True)
def normalization_mc_numba(
    theta: np.ndarray,
    base_normals: np.ndarray,
    cs_theta_e_axis: np.ndarray,
    cs_gamma_grid: np.ndarray,
    cs_cross_section_grid: np.ndarray,
    cs_over_theta_grid: np.ndarray,
    cross_section_mode_code: int,
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
        selection_weight = policy_cross_section_find_weight(
            theta_e,
            gamma,
            theta_e,
            theta0,
            loga,
            cross_section_mode_code,
            cs_theta_e_axis,
            cs_gamma_grid,
            cs_cross_section_grid,
            cs_over_theta_grid,
        )
        if selection_weight <= 0.0:
            continue
        total_weight += inv_trunc_den * selection_weight

    return total_weight / n_samples


@nb.njit(cache=True, parallel=True, fastmath=True)
def population_summary_mc_numba(
    theta: np.ndarray,
    base_normals: np.ndarray,
    cs_theta_e_axis: np.ndarray,
    cs_gamma_grid: np.ndarray,
    cs_cross_section_grid: np.ndarray,
    cs_over_theta_grid: np.ndarray,
    cross_section_mode_code: int,
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
                    accumulate_fp_ols_summary(row, mstar, log_sigma_model, fp_pivot_mstar)

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
        selection_weight = policy_cross_section_find_weight(
            theta_e,
            gamma,
            theta_e,
            theta0,
            loga,
            cross_section_mode_code,
            cs_theta_e_axis,
            cs_gamma_grid,
            cs_cross_section_grid,
            cs_over_theta_grid,
        )
        if selection_weight <= 0.0:
            continue
        total_weight += inv_trunc_den * selection_weight

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
    cs_over_theta_grid: np.ndarray,
    cross_section_mode_code: int,
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
        p_zs = truncated_nonnegative_source_redshift_density(zs[lens_index], mu_zs, sigma_zs)
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

            selection_weight = policy_cross_section_find_weight(
                theta_e,
                gamma,
                theta_e,
                theta0,
                loga,
                cross_section_mode_code,
                cs_theta_e_axis,
                cs_gamma_grid,
                cs_cross_section_grid,
                cs_over_theta_grid,
            )
            if selection_weight <= 0.0:
                continue

            sigma_model = sigma_model_from_s2(s2_grid_int[lens_index, gamma_index], log_enclosed_mass)
            sigma_probability = observed_sigma_likelihood(
                lens_index,
                num_sigma,
                has_s2,
                sigma_obs,
                sigma_err,
                sigma_model,
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
                * selection_weight
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


def solve_fundamental_plane_ols(
    *,
    sample_count: float,
    sum_x1: float,
    sum_x1x1: float,
    sum_y: float,
    sum_x1y: float,
    sum_yy: float,
) -> tuple[float, float, float]:
    """
    Solve the one-predictor FP regression from sufficient statistics.

    This tiny solve stays in Python because it runs once per proposal and is
    easier to audit here than as custom linear algebra inside the MC kernel.
    The kernel's responsibility is only to produce stable aggregate moments.
    """

    if sample_count < 2.0:
        return math.nan, math.nan, math.nan
    xtx = np.asarray(
        [[sample_count, sum_x1], [sum_x1, sum_x1x1]],
        dtype=np.float64,
    )
    xty = np.asarray([sum_y, sum_x1y], dtype=np.float64)
    try:
        coefficients = np.linalg.solve(xtx, xty)
    except np.linalg.LinAlgError:
        return math.nan, math.nan, math.nan

    sse = float(sum_yy - np.dot(coefficients, xty))
    if sse < 0.0 and abs(sse) < 1.0e-12:
        sse = 0.0
    if sse < 0.0:
        return math.nan, math.nan, math.nan
    scatter = math.sqrt(sse / sample_count)
    return float(coefficients[0]), float(coefficients[1]), float(scatter)


def _fit_fundamental_plane_from_summary(fp_summary: np.ndarray) -> tuple[float, float, float]:
    """Convert one FP summary vector into intercept, mass slope, and scatter."""

    return solve_fundamental_plane_ols(
        sample_count=float(fp_summary[FP_OLS_COUNT_INDEX]),
        sum_x1=float(fp_summary[FP_OLS_SUM_X1_INDEX]),
        sum_x1x1=float(fp_summary[FP_OLS_SUM_X1X1_INDEX]),
        sum_y=float(fp_summary[FP_OLS_SUM_Y_INDEX]),
        sum_x1y=float(fp_summary[FP_OLS_SUM_X1Y_INDEX]),
        sum_yy=float(fp_summary[FP_OLS_SUM_YY_INDEX]),
    )


def _gaussian_quadratic_log_penalty(value: float, mean: float, sigma: float) -> float:
    """Return the unnormalized Gaussian quadratic penalty used by the FP prior."""

    if sigma <= 0.0 or not math.isfinite(value):
        return -np.inf
    z = (value - mean) / sigma
    return -0.5 * z * z


def _evaluate_fundamental_plane_prior(
    fp_summary: np.ndarray,
    compiled_model: CompiledModel,
) -> tuple[float, float, float, float, float]:
    """
    Return the optional FP prior value and diagnostic coefficients.

    The historical output schema contains a radius-slope diagnostic slot.  The
    current 1D relation has no radius predictor, so the value remains NaN by
    design and downstream tools can distinguish it from a fitted zero.
    """

    intercept, beta_mass, scatter = _fit_fundamental_plane_from_summary(fp_summary)
    beta_radius = math.nan
    if not all(np.isfinite([intercept, beta_mass, scatter])):
        return -np.inf, intercept, beta_mass, beta_radius, scatter

    context = compiled_model.context
    log_prior = 0.0
    log_prior += _gaussian_quadratic_log_penalty(
        scatter,
        context.fp_fiducial_scatter,
        context.fp_scatter_error,
    )
    log_prior += _gaussian_quadratic_log_penalty(
        intercept,
        context.fp_mu_v_prior,
        context.fp_mu_v_error,
    )
    log_prior += _gaussian_quadratic_log_penalty(
        beta_mass,
        context.fp_beta_v_prior,
        context.fp_beta_v_error,
    )
    return float(log_prior), intercept, beta_mass, beta_radius, scatter


def log_prob(theta: np.ndarray, compiled_model: CompiledModel, total_start: float) -> tuple[float, np.void]:
    """Evaluate the CMASS posterior using CMASS-specific Numba kernels."""

    context = compiled_model.context

    normalization_start = perf_counter()
    if context.fp_enabled == 1:
        z_norm, fp_summary = population_summary_mc_numba(
            theta=theta,
            base_normals=context.base_normals,
            cs_theta_e_axis=context.cs_theta_e_axis,
            cs_gamma_grid=context.cs_gamma_grid,
            cs_cross_section_grid=context.cs_cross_section_grid,
            cs_over_theta_grid=context.cs_over_theta_grid,
            cross_section_mode_code=context.cross_section_mode_code,
            z_grid=context.z_grid,
            chi_kpc_grid=context.chi_kpc_grid,
            mu_d=context.mu_d,
            sigma_d=context.sigma_d,
            mass_function_loc=context.mass_function_loc,
            mass_function_scale=context.mass_function_scale,
            mass_function_alpha=context.mass_function_alpha,
            mu_r0=context.mu_r0,
            beta_r=context.beta_r,
            sigma_r=context.sigma_r,
            nu_r=context.nu_r,
            use_sersic_index=context.use_sersic_index,
            n_fixed=context.n_fixed,
            mu_n0=context.mu_n0,
            beta_n=context.beta_n,
            sigma_n=context.sigma_n,
            gamma_trunc_low=context.gamma_trunc_low,
            gamma_trunc_high=context.gamma_trunc_high,
            mass_radius_kpc=context.mass_radius_kpc,
            gamma_mode_code=context.gamma_mode_code,
            stellar_mass_pivot=context.stellar_mass_pivot,
            mass_log_physical_offset=context.mass_log_physical_offset,
            fp_fit_mstar_min=context.fp_fit_mstar_min,
            fp_pivot_mstar=context.fp_pivot_mstar,
            fp_gamma_axis=context.fp_gamma_axis,
            fp_zd_axis=context.fp_zd_axis,
            fp_log_re_kpc_axis=context.fp_log_re_kpc_axis,
            fp_n_axis=context.fp_n_axis,
            fp_sigma_unit_grid=context.fp_sigma_unit_grid,
            fp_has_n_axis=context.fp_has_n_axis,
        )
    else:
        z_norm = normalization_mc_numba(
            theta=theta,
            base_normals=context.base_normals,
            cs_theta_e_axis=context.cs_theta_e_axis,
            cs_gamma_grid=context.cs_gamma_grid,
            cs_cross_section_grid=context.cs_cross_section_grid,
            cs_over_theta_grid=context.cs_over_theta_grid,
            cross_section_mode_code=context.cross_section_mode_code,
            z_grid=context.z_grid,
            chi_kpc_grid=context.chi_kpc_grid,
            mu_d=context.mu_d,
            sigma_d=context.sigma_d,
            mass_function_loc=context.mass_function_loc,
            mass_function_scale=context.mass_function_scale,
            mass_function_alpha=context.mass_function_alpha,
            mu_r0=context.mu_r0,
            beta_r=context.beta_r,
            sigma_r=context.sigma_r,
            nu_r=context.nu_r,
            use_sersic_index=context.use_sersic_index,
            n_fixed=context.n_fixed,
            mu_n0=context.mu_n0,
            beta_n=context.beta_n,
            sigma_n=context.sigma_n,
            gamma_trunc_low=context.gamma_trunc_low,
            gamma_trunc_high=context.gamma_trunc_high,
            mass_radius_kpc=context.mass_radius_kpc,
            gamma_mode_code=context.gamma_mode_code,
            stellar_mass_pivot=context.stellar_mass_pivot,
            mass_log_physical_offset=context.mass_log_physical_offset,
        )
        fp_summary = None
    normalization_seconds = perf_counter() - normalization_start
    if (not np.isfinite(z_norm)) or z_norm <= context.normalization_min_value:
        return build_reject_result(total_start, compiled_model, "cmass")

    likelihood_start = perf_counter()
    likelihood_value = log_likelihood_lenses_numba(
        theta=theta,
        z_grid=context.z_grid,
        chi_kpc_grid=context.chi_kpc_grid,
        cs_theta_e_axis=context.cs_theta_e_axis,
        cs_gamma_grid=context.cs_gamma_grid,
        cs_cross_section_grid=context.cs_cross_section_grid,
        cs_over_theta_grid=context.cs_over_theta_grid,
        cross_section_mode_code=context.cross_section_mode_code,
        mass_grid_int=context.mass_grid_int,
        dmass_dthetaein_grid_int=context.dmass_dthetaein_grid_int,
        s2_grid_int=context.s2_grid_int,
        has_s2=context.has_s2,
        num_sigma=context.num_sigma,
        sigma_obs=context.sigma_obs,
        sigma_err=context.sigma_err,
        zd=context.zd,
        zs=context.zs,
        p_zd_fixed=context.p_zd_fixed,
        mstar_grid=context.mstar_grid,
        mstar_shift11p4=context.mstar_shift11p4,
        sigma_star_shift9p0_grid=context.sigma_star_shift9p0_grid,
        mstar_integrand_base=context.mstar_integrand_base,
        delta_r_grid=context.delta_r_grid,
        gamma_grid_int=context.gamma_grid_int,
        mass_radius_kpc=context.mass_radius_kpc,
        gamma_mode_code=context.gamma_mode_code,
        mass_log_physical_offset=context.mass_log_physical_offset,
    )
    likelihood_seconds = perf_counter() - likelihood_start

    fp_prior_seconds = 0.0
    fp_prior_log_term = 0.0
    fpfit_mu = math.nan
    fpfit_beta = math.nan
    fpfit_xi = math.nan
    fpfit_scatter = math.nan
    if context.fp_enabled == 1:
        fp_prior_start = perf_counter()
        fp_prior_log_term, fpfit_mu, fpfit_beta, fpfit_xi, fpfit_scatter = (
            _evaluate_fundamental_plane_prior(fp_summary, compiled_model)
        )
        fp_prior_seconds = perf_counter() - fp_prior_start

    total_seconds = perf_counter() - total_start
    blob = build_timing_blob(
        total_log_prob_seconds=total_seconds,
        likelihood_seconds=likelihood_seconds,
        normalization_seconds=normalization_seconds,
        fp_prior_seconds=fp_prior_seconds,
        normalization_value=float(z_norm),
        fp_prior_log_term=float(fp_prior_log_term),
        fpfit_mu=float(fpfit_mu),
        fpfit_beta=float(fpfit_beta),
        fpfit_xi=float(fpfit_xi),
        fpfit_scatter=float(fpfit_scatter),
        kernel="cmass",
        parallel_strategy=compiled_model.parallelism.strategy,
    )

    if (not np.isfinite(likelihood_value)) or (not np.isfinite(fp_prior_log_term)):
        return -np.inf, blob
    total = float(likelihood_value - context.zd.shape[0] * math.log(z_norm) + fp_prior_log_term)
    return total, blob


__all__ = ["log_prob", "solve_fundamental_plane_ols"]
