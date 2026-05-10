"""Sonnenfeld 2024 SLACS sigma-star-gamma posterior and Numba kernels.

This module intentionally owns a separate posterior from
``models.sonnenfeld2024_slacs.posterior``.  The two models share the same
Sonnenfeld data contract, finite-fibre selection, source-redshift treatment,
and FP-prior machinery, but this file changes the density-slope population to
the sigma-star-dependent relation requested for the new peer model.
"""

from __future__ import annotations

import math
from time import perf_counter

import numba as nb
import numpy as np

from ...numba_backend.diagnostics import build_reject_result, build_timing_blob
from ...numba_backend.kernels.constants import LOG10_2PI
from ...numba_backend.kernels.distributions import normal_pdf, truncnorm_sample
from ...numba_backend.kernels.fundamental_plane import (
    FP_OLS2_COUNT_INDEX,
    FP_OLS2_SUMMARY_SIZE,
    FP_OLS2_SUM_X1X1_INDEX,
    FP_OLS2_SUM_X1X2_INDEX,
    FP_OLS2_SUM_X1Y_INDEX,
    FP_OLS2_SUM_X1_INDEX,
    FP_OLS2_SUM_X2X2_INDEX,
    FP_OLS2_SUM_X2Y_INDEX,
    FP_OLS2_SUM_X2_INDEX,
    FP_OLS2_SUM_YY_INDEX,
    FP_OLS2_SUM_Y_INDEX,
    accumulate_fp_ols2_summary,
)
from ...numba_backend.kernels.integration import trapezoid_1d
from ...numba_backend.kernels.interpolation import interp_sigma_unit_clip
from ...numba_backend.kernels.lensing import theta_ein_arcsec
from ...numba_backend.kernels.selection import theta_e_est_from_sigma_proxy
from ...numba_backend.kernels.selection_likelihood import (
    cross_section_find_weight,
    gaussian_source_redshift_density,
    observed_sigma_likelihood,
    sigma_model_from_s2,
)
from ...types import CompiledModel


@nb.njit(cache=True, inline="always")
def _unpack_theta(theta: np.ndarray) -> tuple[float, ...]:
    """Unpack the fixed 11D sigma-star-gamma parameter vector."""

    return (
        theta[0],
        theta[1],
        theta[2],
        theta[3],
        theta[4],
        theta[5],
        theta[6],
        theta[7],
        theta[8],
        theta[9],
        theta[10],
    )

@nb.njit(cache=True, inline="always")
def _draw_population_state(
    theta: np.ndarray,
    nrm: np.ndarray,
    zd: float,
    mstar: float,
    log_re: float,
    delta_r: float,
    mstar_pivot: float,
    n_fixed: float,
    use_sersic_index: int,
    gamma_trunc_low: float,
    gamma_trunc_high: float,
) -> tuple[float, float, float, float, float, float, float, float]:
    """
    Map one reference parent-population row into theta-dependent quantities.

    Preprocessing already drew `(z_d, M_*, log R_e, delta_r)` from the physical
    parent distribution.  The hot path only adds hyper-parameter-dependent
    `m5`, `gamma`, and source-redshift draws.
    """

    (
        mu5_0,
        beta5,
        xi5,
        sigma5,
        mu_gamma_0,
        beta_sigma_star_gamma,
        sigma_gamma,
        mu_zs,
        sigma_zs,
        _theta0,
        _loga,
    ) = _unpack_theta(theta)

    # The reference normalization draws the source population from a Gaussian
    # truncated at z_s=0, then applies the survey/source-window cuts separately.
    # Keeping the truncation in the draw and the window in the mask mirrors
    # `fit_full.py` and avoids reintroducing an unconstrained proposal density.
    zs = truncnorm_sample(mu_zs, sigma_zs, 0.0, math.inf, nrm[5])
    n_value = n_fixed
    if use_sersic_index == 1:
        n_value = max(4.0 + 0.4 * nrm[7], 0.5)

    mstar_shift = mstar - mstar_pivot
    log_enclosed_mass = mu5_0 + beta5 * mstar_shift + xi5 * delta_r + sigma5 * nrm[3]
    sigma_star_shift9p0 = mstar - LOG10_2PI - 2.0 * log_re - 9.0
    mu_gamma = mu_gamma_0 + beta_sigma_star_gamma * sigma_star_shift9p0
    gamma = truncnorm_sample(mu_gamma, sigma_gamma, gamma_trunc_low, gamma_trunc_high, nrm[4])

    return (
        zd,
        zs,
        mstar,
        n_value,
        log_re,
        delta_r,
        log_enclosed_mass,
        gamma,
    )


@nb.njit(cache=True, inline="always")
def _passes_reference_source_redshift_mask(
    zd: float,
    zs: float,
    source_z_min: float,
    source_z_max: float,
    source_lens_redshift_gap: float,
) -> bool:
    """Return the source-redshift part of the reference population `good` mask."""

    if zd <= 0.0 or not math.isfinite(zs):
        return False
    if zs <= zd + source_lens_redshift_gap:
        return False
    if zs <= source_z_min or zs >= source_z_max:
        return False
    return True


@nb.njit(cache=True, parallel=True, fastmath=True)
def normalization_mc_numba(
    theta: np.ndarray,
    base_normals: np.ndarray,
    parent_sample_zd: np.ndarray,
    parent_sample_mstar: np.ndarray,
    parent_sample_log_re: np.ndarray,
    parent_sample_delta_r: np.ndarray,
    z_grid: np.ndarray,
    chi_kpc_grid: np.ndarray,
    cs_theta_e_axis: np.ndarray,
    cs_gamma_axis: np.ndarray,
    cs_cross_section_grid: np.ndarray,
    population_gamma_axis: np.ndarray,
    population_zd_axis: np.ndarray,
    population_log_re_kpc_axis: np.ndarray,
    population_n_axis: np.ndarray,
    population_sigma_unit_grid: np.ndarray,
    mass_radius_kpc: float,
    mass_log_physical_offset: float,
    mstar_pivot: float,
    n_fixed: float,
    use_sersic_index: int,
    gamma_trunc_low: float,
    gamma_trunc_high: float,
    source_z_min: float,
    source_z_max: float,
    source_lens_redshift_gap: float,
    sigma_proxy_fractional_scatter: float,
) -> float:
    """Estimate the Sonnenfeld selection normalization."""

    if theta.shape[0] != 11:
        return 0.0
    (
        _mu5_0,
        _beta5,
        _xi5,
        sigma5,
        _mu_gamma_0,
        _beta_sigma_star_gamma,
        sigma_gamma,
        _mu_zs,
        sigma_zs,
        theta0,
        loga,
    ) = _unpack_theta(theta)
    if sigma5 <= 0.0 or sigma_gamma <= 0.0 or sigma_zs <= 0.0:
        return 0.0

    n_samples = base_normals.shape[0]
    total = 0.0
    for sample_index in nb.prange(n_samples):
        nrm = base_normals[sample_index]
        (
            zd,
            zs,
            mstar,
            n_value,
            log_re,
            _delta_r,
            log_enclosed_mass,
            gamma,
        ) = _draw_population_state(
            theta,
            nrm,
            parent_sample_zd[sample_index],
            parent_sample_mstar[sample_index],
            parent_sample_log_re[sample_index],
            parent_sample_delta_r[sample_index],
            mstar_pivot,
            n_fixed,
            use_sersic_index,
            gamma_trunc_low,
            gamma_trunc_high,
        )
        if (
            not _passes_reference_source_redshift_mask(
                zd,
                zs,
                source_z_min,
                source_z_max,
                source_lens_redshift_gap,
            )
            or not math.isfinite(gamma)
        ):
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
        sigma_unit = interp_sigma_unit_clip(
            gamma,
            zd,
            log_re,
            n_value,
            population_gamma_axis,
            population_zd_axis,
            population_log_re_kpc_axis,
            population_n_axis,
            population_sigma_unit_grid,
            1,
        )
        if theta_e <= 0.0 or sigma_unit <= 0.0:
            continue
        sigma_model = sigma_model_from_s2(sigma_unit, log_enclosed_mass)
        sigma_proxy = sigma_model * (1.0 + sigma_proxy_fractional_scatter * nrm[6])
        theta_est = theta_e_est_from_sigma_proxy(sigma_proxy, zd, zs, z_grid, chi_kpc_grid)
        selection_weight = cross_section_find_weight(
            theta_e,
            gamma,
            theta_est,
            theta0,
            loga,
            cs_theta_e_axis,
            cs_gamma_axis,
            cs_cross_section_grid,
        )
        if selection_weight <= 0.0:
            continue
        total += selection_weight

    return total / n_samples


@nb.njit(cache=True, parallel=True, fastmath=True)
def population_summary_mc_numba(
    theta: np.ndarray,
    base_normals: np.ndarray,
    parent_sample_zd: np.ndarray,
    parent_sample_mstar: np.ndarray,
    parent_sample_log_re: np.ndarray,
    parent_sample_delta_r: np.ndarray,
    z_grid: np.ndarray,
    chi_kpc_grid: np.ndarray,
    cs_theta_e_axis: np.ndarray,
    cs_gamma_axis: np.ndarray,
    cs_cross_section_grid: np.ndarray,
    population_gamma_axis: np.ndarray,
    population_zd_axis: np.ndarray,
    population_log_re_kpc_axis: np.ndarray,
    population_n_axis: np.ndarray,
    population_sigma_unit_grid: np.ndarray,
    mass_radius_kpc: float,
    mass_log_physical_offset: float,
    mstar_pivot: float,
    n_fixed: float,
    use_sersic_index: int,
    gamma_trunc_low: float,
    gamma_trunc_high: float,
    source_z_min: float,
    source_z_max: float,
    source_lens_redshift_gap: float,
    sigma_proxy_fractional_scatter: float,
    fp_fit_mstar_min: float,
    fp_pivot_mstar: float,
) -> tuple[float, np.ndarray]:
    """
    Estimate Sonnenfeld selection normalization and FP fit moments together.

    The reference implementation fits the Fundamental Plane on the parent
    population sample before lens selection is applied, then adds that prior to
    the same posterior that uses the selection normalization.  Sharing this
    Monte Carlo draw stream keeps the local backend deterministic and avoids a
    second population pass when `fp_prior.enabled` is true.
    """

    fp_summary = np.zeros(FP_OLS2_SUMMARY_SIZE, dtype=np.float64)
    if theta.shape[0] != 11:
        return 0.0, fp_summary
    (
        _mu5_0,
        _beta5,
        _xi5,
        sigma5,
        _mu_gamma_0,
        _beta_sigma_star_gamma,
        sigma_gamma,
        _mu_zs,
        sigma_zs,
        theta0,
        loga,
    ) = _unpack_theta(theta)
    if sigma5 <= 0.0 or sigma_gamma <= 0.0 or sigma_zs <= 0.0:
        return 0.0, fp_summary

    n_samples = base_normals.shape[0]
    total = 0.0
    summary_rows = np.zeros((n_samples, FP_OLS2_SUMMARY_SIZE), dtype=np.float64)
    for sample_index in nb.prange(n_samples):
        nrm = base_normals[sample_index]
        row = summary_rows[sample_index]
        (
            zd,
            zs,
            mstar,
            n_value,
            log_re,
            delta_r,
            log_enclosed_mass,
            gamma,
        ) = _draw_population_state(
            theta,
            nrm,
            parent_sample_zd[sample_index],
            parent_sample_mstar[sample_index],
            parent_sample_log_re[sample_index],
            parent_sample_delta_r[sample_index],
            mstar_pivot,
            n_fixed,
            use_sersic_index,
            gamma_trunc_low,
            gamma_trunc_high,
        )

        sigma_unit = 0.0
        if (
            zd > 0.0
            and mstar > fp_fit_mstar_min
            and math.isfinite(gamma)
        ):
            sigma_unit = interp_sigma_unit_clip(
                gamma,
                zd,
                log_re,
                n_value,
                population_gamma_axis,
                population_zd_axis,
                population_log_re_kpc_axis,
                population_n_axis,
                population_sigma_unit_grid,
                1,
            )
            if sigma_unit > 0.0 and math.isfinite(sigma_unit):
                log_sigma_model = 0.5 * (math.log10(sigma_unit) + log_enclosed_mass)
                if math.isfinite(log_sigma_model):
                    accumulate_fp_ols2_summary(
                        row,
                        mstar,
                        delta_r,
                        log_sigma_model,
                        fp_pivot_mstar,
                        1.0,
                    )

        if (
            not _passes_reference_source_redshift_mask(
                zd,
                zs,
                source_z_min,
                source_z_max,
                source_lens_redshift_gap,
            )
            or not math.isfinite(gamma)
        ):
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
        if sigma_unit <= 0.0:
            sigma_unit = interp_sigma_unit_clip(
                gamma,
                zd,
                log_re,
                n_value,
                population_gamma_axis,
                population_zd_axis,
                population_log_re_kpc_axis,
                population_n_axis,
                population_sigma_unit_grid,
                1,
            )
        if theta_e <= 0.0 or sigma_unit <= 0.0:
            continue
        sigma_model = sigma_model_from_s2(sigma_unit, log_enclosed_mass)
        sigma_proxy = sigma_model * (1.0 + sigma_proxy_fractional_scatter * nrm[6])
        theta_est = theta_e_est_from_sigma_proxy(sigma_proxy, zd, zs, z_grid, chi_kpc_grid)
        selection_weight = cross_section_find_weight(
            theta_e,
            gamma,
            theta_est,
            theta0,
            loga,
            cs_theta_e_axis,
            cs_gamma_axis,
            cs_cross_section_grid,
        )
        if selection_weight <= 0.0:
            continue
        total += selection_weight

    for sample_index in range(n_samples):
        for summary_index in range(FP_OLS2_SUMMARY_SIZE):
            fp_summary[summary_index] += summary_rows[sample_index, summary_index]

    return total / n_samples, fp_summary


@nb.njit(cache=True, parallel=True, fastmath=True)
def log_likelihood_lenses_numba(
    theta: np.ndarray,
    z_grid: np.ndarray,
    chi_kpc_grid: np.ndarray,
    cs_theta_e_axis: np.ndarray,
    cs_gamma_axis: np.ndarray,
    cs_cross_section_grid: np.ndarray,
    gamma_grid_int: np.ndarray,
    mass_grid_int: np.ndarray,
    dmass_dthetaein_grid_int: np.ndarray,
    s2_grid_int: np.ndarray,
    has_s2: np.ndarray,
    num_sigma: np.ndarray,
    sigma_obs: np.ndarray,
    sigma_err: np.ndarray,
    zd: np.ndarray,
    zs: np.ndarray,
    log_re_obs: np.ndarray,
    parent_mstar_density_grid: np.ndarray,
    size_density_grid: np.ndarray,
    delta_r_grid: np.ndarray,
    mstar_shift_grid: np.ndarray,
    mstar_grid: np.ndarray,
    mass_radius_kpc: float,
    mass_log_physical_offset: float,
) -> float:
    """Evaluate all Sonnenfeld per-lens likelihood integrals."""

    if theta.shape[0] != 11:
        return -np.inf
    (
        mu5_0,
        beta5,
        xi5,
        sigma5,
        mu_gamma_0,
        beta_sigma_star_gamma,
        sigma_gamma,
        mu_zs,
        sigma_zs,
        theta0,
        loga,
    ) = _unpack_theta(theta)
    if sigma5 <= 0.0 or sigma_gamma <= 0.0 or sigma_zs <= 0.0:
        return -np.inf

    n_lens = zd.shape[0]
    n_gamma = gamma_grid_int.shape[0]
    n_mstar = mstar_grid.shape[1]
    ll_terms = np.zeros(n_lens, dtype=np.float64)
    valid = np.ones(n_lens, dtype=np.int64)

    for lens_index in nb.prange(n_lens):
        p_zs = gaussian_source_redshift_density(zs[lens_index], mu_zs, sigma_zs)
        if p_zs <= 0.0:
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
            sigma_model = sigma_model_from_s2(s2_grid_int[lens_index, gamma_index], log_enclosed_mass)
            sigma_find_proxy = sigma_model
            if num_sigma[lens_index] >= 1:
                sigma_find_proxy = sigma_obs[lens_index, 0]
            theta_est = theta_e_est_from_sigma_proxy(
                sigma_find_proxy,
                zd[lens_index],
                zs[lens_index],
                z_grid,
                chi_kpc_grid,
            )
            selection_weight = cross_section_find_weight(
                theta_e,
                gamma,
                theta_est,
                theta0,
                loga,
                cs_theta_e_axis,
                cs_gamma_axis,
                cs_cross_section_grid,
            )
            sigma_probability = observed_sigma_likelihood(
                lens_index,
                num_sigma,
                has_s2,
                sigma_obs,
                sigma_err,
                sigma_model,
            )
            if selection_weight <= 0.0 or sigma_probability <= 0.0:
                continue

            mstar_integrand = np.zeros(n_mstar, dtype=np.float64)
            for mstar_index in range(n_mstar):
                mu5 = (
                    mu5_0
                    + beta5 * mstar_shift_grid[lens_index, mstar_index]
                    + xi5 * delta_r_grid[lens_index, mstar_index]
                )
                sigma_star_shift9p0 = (
                    mstar_grid[lens_index, mstar_index]
                    - LOG10_2PI
                    - 2.0 * log_re_obs[lens_index]
                    - 9.0
                )
                mu_gamma = mu_gamma_0 + beta_sigma_star_gamma * sigma_star_shift9p0
                mstar_integrand[mstar_index] = (
                    parent_mstar_density_grid[lens_index, mstar_index]
                    * size_density_grid[lens_index, mstar_index]
                    * normal_pdf(log_enclosed_mass, mu5, sigma5)
                    * normal_pdf(gamma, mu_gamma, sigma_gamma)
                )

            integrated_mstar = trapezoid_1d(mstar_integrand, mstar_grid[lens_index])
            gamma_integrand[gamma_index] = (
                integrated_mstar
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


def solve_fundamental_plane_ols2(
    *,
    sample_count: float,
    sum_x1: float,
    sum_x2: float,
    sum_x1x1: float,
    sum_x1x2: float,
    sum_x2x2: float,
    sum_y: float,
    sum_x1y: float,
    sum_x2y: float,
    sum_yy: float,
) -> tuple[float, float, float, float]:
    """
    Solve the Sonnenfeld two-predictor FP regression from summary moments.

    The fitted relation is
    `log10(sigma) = mu + beta * (mstar - pivot) + xi * delta_r`.  The SLACS
    reference prior constrains `mu`, `beta`, and the residual scatter; `xi` is
    still returned as a diagnostic because the reference blobs expose it.
    `sample_count` is the effective sum of weights when the population moments
    come from importance sampling rather than direct parent draws.
    """

    if sample_count <= 0.0:
        return math.nan, math.nan, math.nan, math.nan
    normal_matrix = np.asarray(
        [
            [sample_count, sum_x1, sum_x2],
            [sum_x1, sum_x1x1, sum_x1x2],
            [sum_x2, sum_x1x2, sum_x2x2],
        ],
        dtype=np.float64,
    )
    rhs = np.asarray([sum_y, sum_x1y, sum_x2y], dtype=np.float64)
    try:
        intercept, beta_mass, beta_radius = np.linalg.solve(normal_matrix, rhs)
    except np.linalg.LinAlgError:
        return math.nan, math.nan, math.nan, math.nan
    if not all(np.isfinite([intercept, beta_mass, beta_radius])):
        return math.nan, math.nan, math.nan, math.nan

    residual_sum_squares = float(sum_yy - np.dot(np.asarray([intercept, beta_mass, beta_radius]), rhs))
    if residual_sum_squares < 0.0 and residual_sum_squares > -1.0e-10:
        residual_sum_squares = 0.0
    if residual_sum_squares < 0.0:
        return math.nan, math.nan, math.nan, math.nan
    scatter = math.sqrt(residual_sum_squares / sample_count)
    return float(intercept), float(beta_mass), float(beta_radius), float(scatter)


def _fit_fundamental_plane_from_summary(fp_summary: np.ndarray) -> tuple[float, float, float, float]:
    """Convert one FP summary vector into `mu`, mass slope, radius slope, scatter."""

    return solve_fundamental_plane_ols2(
        sample_count=float(fp_summary[FP_OLS2_COUNT_INDEX]),
        sum_x1=float(fp_summary[FP_OLS2_SUM_X1_INDEX]),
        sum_x2=float(fp_summary[FP_OLS2_SUM_X2_INDEX]),
        sum_x1x1=float(fp_summary[FP_OLS2_SUM_X1X1_INDEX]),
        sum_x1x2=float(fp_summary[FP_OLS2_SUM_X1X2_INDEX]),
        sum_x2x2=float(fp_summary[FP_OLS2_SUM_X2X2_INDEX]),
        sum_y=float(fp_summary[FP_OLS2_SUM_Y_INDEX]),
        sum_x1y=float(fp_summary[FP_OLS2_SUM_X1Y_INDEX]),
        sum_x2y=float(fp_summary[FP_OLS2_SUM_X2Y_INDEX]),
        sum_yy=float(fp_summary[FP_OLS2_SUM_YY_INDEX]),
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
    Return the optional Sonnenfeld FP prior value and diagnostic coefficients.

    The reference code fits `mu`, mass slope `beta`, radius-residual slope `xi`,
    and scatter, but places priors only on scatter, `mu`, and `beta`.  Keeping
    `xi` in the blob makes local output comparable to the reference chain
    diagnostics without inventing an extra prior term.
    """

    intercept, beta_mass, beta_radius, scatter = _fit_fundamental_plane_from_summary(fp_summary)
    if not all(np.isfinite([intercept, beta_mass, beta_radius, scatter])):
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
    """Evaluate the Sonnenfeld posterior using Sonnenfeld-specific kernels."""

    context = compiled_model.context
    normalization_start = perf_counter()
    if context.fp_enabled == 1:
        z_norm, fp_summary = population_summary_mc_numba(
            theta=theta,
            base_normals=context.base_normals,
            parent_sample_zd=context.parent_sample_zd,
            parent_sample_mstar=context.parent_sample_mstar,
            parent_sample_log_re=context.parent_sample_log_re,
            parent_sample_delta_r=context.parent_sample_delta_r,
            z_grid=context.z_grid,
            chi_kpc_grid=context.chi_kpc_grid,
            cs_theta_e_axis=context.cs_theta_e_axis,
            cs_gamma_axis=context.cs_gamma_axis,
            cs_cross_section_grid=context.cs_cross_section_grid,
            population_gamma_axis=context.population_gamma_axis,
            population_zd_axis=context.population_zd_axis,
            population_log_re_kpc_axis=context.population_log_re_kpc_axis,
            population_n_axis=context.population_n_axis,
            population_sigma_unit_grid=context.population_sigma_unit_grid,
            mass_radius_kpc=context.mass_radius_kpc,
            mass_log_physical_offset=context.mass_log_physical_offset,
            mstar_pivot=context.mstar_pivot,
            n_fixed=context.n_fixed,
            use_sersic_index=context.use_sersic_index,
            gamma_trunc_low=context.gamma_trunc_low,
            gamma_trunc_high=context.gamma_trunc_high,
            source_z_min=context.source_z_min,
            source_z_max=context.source_z_max,
            source_lens_redshift_gap=context.source_lens_redshift_gap,
            sigma_proxy_fractional_scatter=context.sigma_proxy_fractional_scatter,
            fp_fit_mstar_min=context.fp_fit_mstar_min,
            fp_pivot_mstar=context.fp_pivot_mstar,
        )
    else:
        z_norm = normalization_mc_numba(
            theta=theta,
            base_normals=context.base_normals,
            parent_sample_zd=context.parent_sample_zd,
            parent_sample_mstar=context.parent_sample_mstar,
            parent_sample_log_re=context.parent_sample_log_re,
            parent_sample_delta_r=context.parent_sample_delta_r,
            z_grid=context.z_grid,
            chi_kpc_grid=context.chi_kpc_grid,
            cs_theta_e_axis=context.cs_theta_e_axis,
            cs_gamma_axis=context.cs_gamma_axis,
            cs_cross_section_grid=context.cs_cross_section_grid,
            population_gamma_axis=context.population_gamma_axis,
            population_zd_axis=context.population_zd_axis,
            population_log_re_kpc_axis=context.population_log_re_kpc_axis,
            population_n_axis=context.population_n_axis,
            population_sigma_unit_grid=context.population_sigma_unit_grid,
            mass_radius_kpc=context.mass_radius_kpc,
            mass_log_physical_offset=context.mass_log_physical_offset,
            mstar_pivot=context.mstar_pivot,
            n_fixed=context.n_fixed,
            use_sersic_index=context.use_sersic_index,
            gamma_trunc_low=context.gamma_trunc_low,
            gamma_trunc_high=context.gamma_trunc_high,
            source_z_min=context.source_z_min,
            source_z_max=context.source_z_max,
            source_lens_redshift_gap=context.source_lens_redshift_gap,
            sigma_proxy_fractional_scatter=context.sigma_proxy_fractional_scatter,
        )
        fp_summary = None
    normalization_seconds = perf_counter() - normalization_start
    if (not np.isfinite(z_norm)) or z_norm <= context.normalization_min_value:
        return build_reject_result(total_start, compiled_model, "sonnenfeld2024_slacs_sigma_star_gamma")

    likelihood_start = perf_counter()
    likelihood_value = log_likelihood_lenses_numba(
        theta=theta,
        z_grid=context.z_grid,
        chi_kpc_grid=context.chi_kpc_grid,
        cs_theta_e_axis=context.cs_theta_e_axis,
        cs_gamma_axis=context.cs_gamma_axis,
        cs_cross_section_grid=context.cs_cross_section_grid,
        gamma_grid_int=context.gamma_grid_int,
        mass_grid_int=context.mass_grid_int,
        dmass_dthetaein_grid_int=context.dmass_dthetaein_grid_int,
        s2_grid_int=context.s2_grid_int,
        has_s2=context.has_s2,
        num_sigma=context.num_sigma,
        sigma_obs=context.sigma_obs,
        sigma_err=context.sigma_err,
        zd=context.zd,
        zs=context.zs,
        log_re_obs=context.log_re_obs,
        parent_mstar_density_grid=context.parent_mstar_density_grid,
        size_density_grid=context.size_density_grid,
        delta_r_grid=context.delta_r_grid,
        mstar_shift_grid=context.mstar_shift_grid,
        mstar_grid=context.mstar_grid,
        mass_radius_kpc=context.mass_radius_kpc,
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
        kernel="sonnenfeld2024_slacs_sigma_star_gamma",
        parallel_strategy=compiled_model.parallelism.strategy,
    )
    if (not np.isfinite(likelihood_value)) or (not np.isfinite(fp_prior_log_term)):
        return -np.inf, blob
    total = float(likelihood_value - context.zd.shape[0] * math.log(z_norm) + fp_prior_log_term)
    return total, blob


__all__ = ["log_prob", "solve_fundamental_plane_ols2"]
