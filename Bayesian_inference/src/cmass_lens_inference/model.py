"""
Production log-probability entrypoint built on the compiled model context.

This module is intentionally the only place where the outer sampler and the
inner kernels meet. Keeping that contract narrow makes it easier to reason
about performance, timing instrumentation, and future regression benchmarks.
"""

from __future__ import annotations

import math
from time import perf_counter

import numpy as np

from .compiled_context import build_compiled_context
from .kernels.likelihood import log_likelihood_lenses_numba
from .kernels.normalization import (
    FP_OLS_COUNT_INDEX,
    FP_OLS_SUM_X1X1_INDEX,
    FP_OLS_SUM_X1Y_INDEX,
    FP_OLS_SUM_X1_INDEX,
    FP_OLS_SUM_Y_INDEX,
    FP_OLS_SUM_YY_INDEX,
    normalization_mc_numba,
    population_summary_mc_numba,
)
from .parallel import resolve_parallelism
from .types import CompiledModel, RuntimeConfig

# `emcee.backends.HDFBackend` can persist structured numeric blobs, but it is
# not a safe place to store arbitrary Python dictionaries. The timing blob is
# therefore defined as one small structured dtype so sampling can stream
# directly into `chain.h5` without any post-processing compatibility layer.
LOG_PROB_BLOB_DTYPE = np.dtype(
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
        ("parallel_strategy", "S16"),
    ]
)

def build_compiled_model(runtime_config: RuntimeConfig) -> CompiledModel:
    """
    Build the single production model object used by the sampler.

    This is intentionally a high-level helper so callers do not need to know
    about the lower-level compiled-context builder or the array layout choices
    inside it.
    """

    context, profile, cross_section_grid, cosmology, _, _ = build_compiled_context(runtime_config)
    parallelism = resolve_parallelism(
        runtime_config.runtime,
        runtime_config.sampling.n_walkers,
    )
    return CompiledModel(
        config=runtime_config,
        profile=profile,
        cross_section_grid=cross_section_grid,
        cosmology=cosmology,
        parallelism=parallelism,
        context=context,
    )


def _build_timing_blob(
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
    parallel_strategy: str,
) -> np.void:
    """
    Build a single HDF5-safe structured blob for stage timing summaries.

    Keeping this constructor centralized ensures the normal path and every
    rejection path produce the exact same schema, which avoids backend shape
    mismatches when `emcee` appends later samples.
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
            parallel_strategy.encode("utf-8"),
        ),
        dtype=LOG_PROB_BLOB_DTYPE,
    )[()]


def _reject_blob(total_start: float, strategy: str) -> tuple[float, np.void]:
    """Build a consistent timing blob for rejected proposals."""

    return (
        -np.inf,
        _build_timing_blob(
            total_log_prob_seconds=float(perf_counter() - total_start),
            likelihood_seconds=0.0,
            normalization_seconds=0.0,
            fp_prior_seconds=0.0,
            normalization_value=0.0,
            fp_prior_log_term=0.0,
            fpfit_mu=math.nan,
            fpfit_beta=math.nan,
            fpfit_xi=math.nan,
            fpfit_scatter=math.nan,
            parallel_strategy=strategy,
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
    Solve the 1D sigma-logM* regression from sufficient statistics.

    Why the solver lives in Python instead of the kernel:
    - the regression is tiny (one 2x2 solve per `log_prob` evaluation)
    - keeping the linear algebra outside numba avoids embedding a second
      optimization layer inside the hot Monte Carlo loop
    - the kernel only needs to export stable moments, which are easy to test
      and easy to persist later in diagnostics blobs
    """

    if sample_count < 2.0:
        return math.nan, math.nan, math.nan

    xtx = np.array(
        [
            [sample_count, sum_x1],
            [sum_x1, sum_x1x1],
        ],
        dtype=np.float64,
    )
    xty = np.array([sum_y, sum_x1y], dtype=np.float64)

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
    return (
        float(coefficients[0]),
        float(coefficients[1]),
        float(scatter),
    )


def _fit_fundamental_plane_from_summary(fp_summary: np.ndarray) -> tuple[float, float, float]:
    """Convert the kernel summary vector into the `(a, b, scatter)` 1D fit."""

    return solve_fundamental_plane_ols(
        sample_count=float(fp_summary[FP_OLS_COUNT_INDEX]),
        sum_x1=float(fp_summary[FP_OLS_SUM_X1_INDEX]),
        sum_x1x1=float(fp_summary[FP_OLS_SUM_X1X1_INDEX]),
        sum_y=float(fp_summary[FP_OLS_SUM_Y_INDEX]),
        sum_x1y=float(fp_summary[FP_OLS_SUM_X1Y_INDEX]),
        sum_yy=float(fp_summary[FP_OLS_SUM_YY_INDEX]),
    )


def _gaussian_quadratic_log_penalty(value: float, mean: float, sigma: float) -> float:
    """
    Return the unnormalized Gaussian log penalty used by the legacy FP prior.

    The reference implementation adds only the quadratic term, not the normal
    distribution's additive normalization constant. Keeping that convention
    preserves the intended scientific weighting.
    """

    if sigma <= 0.0 or (not math.isfinite(value)):
        return -np.inf
    z = (value - mean) / sigma
    return -0.5 * z * z


def _evaluate_fundamental_plane_prior(
    fp_summary: np.ndarray,
    compiled_model: CompiledModel,
) -> tuple[float, float, float, float, float]:
    """
    Fit the synthetic 1D sigma-logM* relation and return the log-prior term.

    Only three fitted quantities contribute to the posterior:
    - intercept `a`
    - stellar-mass slope `b`
    - residual scatter
    The historical `fpfit_xi` blob field is now a compatibility placeholder
    and therefore remains `NaN`.
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


def log_prob(theta: np.ndarray, compiled_model: CompiledModel) -> tuple[float, np.void]:
    """
    Evaluate the full log posterior contribution needed by `emcee`.

    The model uses the same box prior contract as the earlier implementation,
    but the heavy work is delegated to two monolithic kernels:
    one for normalization and one for all-lens likelihood.
    """

    total_start = perf_counter()
    theta = np.asarray(theta, dtype=np.float64)
    parameter_schema = compiled_model.config.parameter_schema
    parameter_schema.validate_theta_shape(theta)

    for index, (name, (lower, upper)) in enumerate(
        zip(
            parameter_schema.internal_parameter_names,
            parameter_schema.prior_bounds,
            strict=True,
        )
    ):
        value = float(theta[index])
        if value < lower or value > upper:
            return _reject_blob(total_start, compiled_model.parallelism.strategy)

    context = compiled_model.context

    normalization_start = perf_counter()
    if context.fp_enabled == 1:
        # The FP-enabled production path uses the parallel summary kernel.
        # A legacy serial reference helper still lives in `normalization.py`,
        # but it is retained only for regression tests and code-reading support.
        z_norm, fp_summary = population_summary_mc_numba(
            theta=theta,
            base_normals=context.base_normals,
            cs_gamma_grid=context.cs_gamma_grid,
            cs_over_theta=context.cs_over_theta_grid,
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
            cs_gamma_grid=context.cs_gamma_grid,
            cs_over_theta=context.cs_over_theta_grid,
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
        )
        fp_summary = None
    normalization_seconds = perf_counter() - normalization_start
    if (not np.isfinite(z_norm)) or z_norm <= context.normalization_min_value:
        return _reject_blob(total_start, compiled_model.parallelism.strategy)

    likelihood_start = perf_counter()
    likelihood_value = log_likelihood_lenses_numba(
        theta=theta,
        z_grid=context.z_grid,
        chi_kpc_grid=context.chi_kpc_grid,
        cs_over_theta_int=context.cs_over_theta_int,
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
    )
    likelihood_seconds = perf_counter() - likelihood_start

    fp_prior_seconds = 0.0
    log_fp_prior = 0.0
    fpfit_mu = math.nan
    fpfit_beta = math.nan
    fpfit_xi = math.nan
    fpfit_scatter = math.nan
    if context.fp_enabled == 1:
        fp_prior_start = perf_counter()
        log_fp_prior, fpfit_mu, fpfit_beta, fpfit_xi, fpfit_scatter = _evaluate_fundamental_plane_prior(
            fp_summary,
            compiled_model,
        )
        fp_prior_seconds = perf_counter() - fp_prior_start
        if not np.isfinite(log_fp_prior):
            total_seconds = perf_counter() - total_start
            blob = _build_timing_blob(
                total_log_prob_seconds=float(total_seconds),
                likelihood_seconds=float(likelihood_seconds),
                normalization_seconds=float(normalization_seconds),
                fp_prior_seconds=float(fp_prior_seconds),
                normalization_value=float(z_norm),
                fp_prior_log_term=float(log_fp_prior),
                fpfit_mu=float(fpfit_mu),
                fpfit_beta=float(fpfit_beta),
                fpfit_xi=float(fpfit_xi),
                fpfit_scatter=float(fpfit_scatter),
                parallel_strategy=compiled_model.parallelism.strategy,
            )
            return -np.inf, blob

    total_seconds = perf_counter() - total_start

    blob = _build_timing_blob(
        total_log_prob_seconds=float(total_seconds),
        likelihood_seconds=float(likelihood_seconds),
        normalization_seconds=float(normalization_seconds),
        fp_prior_seconds=float(fp_prior_seconds),
        normalization_value=float(z_norm),
        fp_prior_log_term=float(log_fp_prior),
        fpfit_mu=float(fpfit_mu),
        fpfit_beta=float(fpfit_beta),
        fpfit_xi=float(fpfit_xi),
        fpfit_scatter=float(fpfit_scatter),
        parallel_strategy=compiled_model.parallelism.strategy,
    )

    if not np.isfinite(likelihood_value):
        return -np.inf, blob

    return float(likelihood_value - context.zd.shape[0] * math.log(z_norm) + log_fp_prior), blob
