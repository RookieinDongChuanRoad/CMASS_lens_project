"""
Production Numba likelihood engine.

This module is the only bridge between the framework-level sampler and the
model-specific compiled kernels.  It owns box-prior rejection, backend dispatch,
posterior reduction, and the small structured diagnostic blob that emcee stores
alongside log-probability values in `chain.h5`.
"""

from __future__ import annotations

import math
from time import perf_counter

import numpy as np

from ..model_registry import get_model_definition
from ..types import CompiledModel, RuntimeConfig
from .cmass_kernels import (
    FP_OLS_COUNT_INDEX,
    FP_OLS_SUM_X1X1_INDEX,
    FP_OLS_SUM_X1Y_INDEX,
    FP_OLS_SUM_X1_INDEX,
    FP_OLS_SUM_Y_INDEX,
    FP_OLS_SUM_YY_INDEX,
    log_likelihood_lenses_numba,
    normalization_mc_numba,
    population_summary_mc_numba,
)
from .sonnenfeld_kernels import (
    log_likelihood_lenses_numba as sonnenfeld_log_likelihood_lenses_numba,
    normalization_mc_numba as sonnenfeld_normalization_mc_numba,
)


NUMBA_DIAGNOSTIC_BLOB_DTYPE = np.dtype(
    [
        ("total_log_prob_seconds", np.float64),
        ("likelihood_seconds", np.float64),
        ("normalization_seconds", np.float64),
        ("fp_prior_seconds", np.float64),
        ("normalization_value", np.float64),
        ("fp_prior_log_term", np.float64),
        ("fpfit_mu", np.float64),
        ("fpfit_beta", np.float64),
        ("fpfit_xi", np.float64),
        ("fpfit_scatter", np.float64),
        ("backend", "S16"),
        ("kernel", "S32"),
        ("parallel_strategy", "S16"),
    ]
)
SONNENFELD_TRUNCATION_MASS_POLYNOMIAL_COEFFICIENTS = np.asarray(
    [9.388, 7.855, 48.34, -312.5, 535.7, -274.2],
    dtype=np.float64,
)


def build_compiled_model(runtime_config: RuntimeConfig) -> CompiledModel:
    """
    Build the Numba compiled model for the configured registry model.

    Concrete runtime adapters still own canonical dataset loading and
    preprocessing.  This common function keeps production entrypoints from
    importing model-specific context builders directly.
    """

    model_definition = get_model_definition(runtime_config.model.name)
    return model_definition.build_compiled_model(runtime_config)


def _model_definition_from_compiled(compiled_model: CompiledModel):
    """Resolve the registry definition that owns a compiled model."""

    return get_model_definition(compiled_model.config.model.name)


def _build_timing_blob(
    *,
    total_log_prob_seconds: float,
    likelihood_seconds: float,
    normalization_seconds: float,
    fp_prior_seconds: float,
    normalization_value: float,
    fp_prior_log_term: float,
    fpfit_mu: float,
    fpfit_beta: float,
    fpfit_xi: float,
    fpfit_scatter: float,
    kernel: str,
    parallel_strategy: str,
) -> np.void:
    """
    Build one HDF5-safe structured diagnostic record.

    The same dtype is used on accepted and rejected proposals.  That stability
    is required because emcee's HDF backend appends blobs row-by-row and cannot
    tolerate schema drift during a run.
    """

    return np.array(
        (
            float(total_log_prob_seconds),
            float(likelihood_seconds),
            float(normalization_seconds),
            float(fp_prior_seconds),
            float(normalization_value),
            float(fp_prior_log_term),
            float(fpfit_mu),
            float(fpfit_beta),
            float(fpfit_xi),
            float(fpfit_scatter),
            b"numba",
            str(kernel).encode("utf-8"),
            str(parallel_strategy).encode("utf-8"),
        ),
        dtype=NUMBA_DIAGNOSTIC_BLOB_DTYPE,
    )[()]


def _reject_blob(total_start: float, compiled_model: CompiledModel, kernel: str) -> tuple[float, np.void]:
    """Return a consistent rejected log-probability result."""

    return (
        -np.inf,
        _build_timing_blob(
            total_log_prob_seconds=perf_counter() - total_start,
            likelihood_seconds=0.0,
            normalization_seconds=0.0,
            fp_prior_seconds=0.0,
            normalization_value=0.0,
            fp_prior_log_term=0.0,
            fpfit_mu=math.nan,
            fpfit_beta=math.nan,
            fpfit_xi=math.nan,
            fpfit_scatter=math.nan,
            kernel=kernel,
            parallel_strategy=compiled_model.parallelism.strategy,
        ),
    )


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


def _cmass_log_prob(theta: np.ndarray, compiled_model: CompiledModel, total_start: float) -> tuple[float, np.void]:
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
        return _reject_blob(total_start, compiled_model, "cmass")

    likelihood_start = perf_counter()
    likelihood_value = log_likelihood_lenses_numba(
        theta=theta,
        z_grid=context.z_grid,
        chi_kpc_grid=context.chi_kpc_grid,
        cs_theta_e_axis=context.cs_theta_e_axis,
        cs_gamma_grid=context.cs_gamma_grid,
        cs_cross_section_grid=context.cs_cross_section_grid,
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
    blob = _build_timing_blob(
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


def _sonnenfeld_log_prob(
    theta: np.ndarray,
    compiled_model: CompiledModel,
    total_start: float,
) -> tuple[float, np.void]:
    """Evaluate the Sonnenfeld posterior using Sonnenfeld-specific kernels."""

    context = compiled_model.context
    normalization_start = perf_counter()
    z_norm = sonnenfeld_normalization_mc_numba(
        theta=theta,
        base_normals=context.base_normals,
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
        mbar=context.mbar,
        parent_alpha=context.parent_alpha,
        truncation_mass_scatter=context.truncation_mass_scatter,
        truncation_coefficients=SONNENFELD_TRUNCATION_MASS_POLYNOMIAL_COEFFICIENTS,
        size_mu0=context.size_mu0,
        size_mu1=context.size_mu1,
        size_sigma=context.size_sigma,
        size_mu2=context.size_mu2,
        n_fixed=context.n_fixed,
        use_sersic_index=context.use_sersic_index,
        gamma_trunc_low=context.gamma_trunc_low,
        gamma_trunc_high=context.gamma_trunc_high,
        parent_zd_min=context.parent_zd_min,
        parent_zd_max=context.parent_zd_max,
        parent_mstar_min=context.parent_mstar_min,
        parent_mstar_max=context.parent_mstar_max,
        sigma_proxy_fractional_scatter=context.sigma_proxy_fractional_scatter,
    )
    normalization_seconds = perf_counter() - normalization_start
    if (not np.isfinite(z_norm)) or z_norm <= context.normalization_min_value:
        return _reject_blob(total_start, compiled_model, "sonnenfeld2024_slacs")

    likelihood_start = perf_counter()
    likelihood_value = sonnenfeld_log_likelihood_lenses_numba(
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
        parent_mstar_density_grid=context.parent_mstar_density_grid,
        size_density_grid=context.size_density_grid,
        delta_r_grid=context.delta_r_grid,
        mstar_shift_grid=context.mstar_shift_grid,
        mstar_grid=context.mstar_grid,
        mass_radius_kpc=context.mass_radius_kpc,
        mass_log_physical_offset=context.mass_log_physical_offset,
    )
    likelihood_seconds = perf_counter() - likelihood_start
    total_seconds = perf_counter() - total_start
    blob = _build_timing_blob(
        total_log_prob_seconds=total_seconds,
        likelihood_seconds=likelihood_seconds,
        normalization_seconds=normalization_seconds,
        fp_prior_seconds=0.0,
        normalization_value=float(z_norm),
        fp_prior_log_term=0.0,
        fpfit_mu=math.nan,
        fpfit_beta=math.nan,
        fpfit_xi=math.nan,
        fpfit_scatter=math.nan,
        kernel="sonnenfeld2024_slacs",
        parallel_strategy=compiled_model.parallelism.strategy,
    )
    if not np.isfinite(likelihood_value):
        return -np.inf, blob
    return float(likelihood_value - context.zd.shape[0] * math.log(z_norm)), blob


def log_prob(theta: np.ndarray, compiled_model: CompiledModel) -> tuple[float, np.void]:
    """
    Evaluate one model posterior for emcee.

    The function first enforces the model's explicit box prior on the host.
    Only in-bounds proposals enter model-specific kernels, which keeps
    rejection behavior clear and avoids wasting kernel time on obviously
    invalid parameter vectors.
    """

    total_start = perf_counter()
    theta = np.asarray(theta, dtype=np.float64)
    parameter_schema = compiled_model.config.parameter_schema
    parameter_schema.validate_theta_shape(theta)
    model_definition = _model_definition_from_compiled(compiled_model)
    kernel = model_definition.backend_kernel

    for index, (_name, (lower, upper)) in enumerate(
        zip(
            parameter_schema.internal_parameter_names,
            parameter_schema.prior_bounds,
            strict=True,
        )
    ):
        value = float(theta[index])
        if value < lower or value > upper:
            return _reject_blob(total_start, compiled_model, kernel)

    if kernel == "cmass":
        return _cmass_log_prob(theta, compiled_model, total_start)
    if kernel == "sonnenfeld2024_slacs":
        return _sonnenfeld_log_prob(theta, compiled_model, total_start)

    raise NotImplementedError(
        f"Numba production kernel '{kernel}' is not implemented yet."
    )


__all__ = [
    "NUMBA_DIAGNOSTIC_BLOB_DTYPE",
    "build_compiled_model",
    "log_prob",
    "solve_fundamental_plane_ols",
]
