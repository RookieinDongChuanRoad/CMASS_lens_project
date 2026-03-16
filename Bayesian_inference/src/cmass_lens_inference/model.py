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
from .kernels.normalization import normalization_mc_numba
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
        ("normalization_value", np.float64),
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
    normalization_value: float,
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
            float(normalization_value),
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
            normalization_value=0.0,
            parallel_strategy=strategy,
        ),
    )


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
        mstar_integrand_base=context.mstar_integrand_base,
        delta_r_grid=context.delta_r_grid,
        gamma_grid_int=context.gamma_grid_int,
        mass_radius_kpc=context.mass_radius_kpc,
        gamma_mode_code=context.gamma_mode_code,
    )
    likelihood_seconds = perf_counter() - likelihood_start
    total_seconds = perf_counter() - total_start

    blob = _build_timing_blob(
        total_log_prob_seconds=float(total_seconds),
        likelihood_seconds=float(likelihood_seconds),
        normalization_seconds=float(normalization_seconds),
        normalization_value=float(z_norm),
        parallel_strategy=compiled_model.parallelism.strategy,
    )

    if not np.isfinite(likelihood_value):
        return -np.inf, blob

    return float(likelihood_value - context.zd.shape[0] * math.log(z_norm)), blob
