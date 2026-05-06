"""Shared one-dimensional fundamental-plane prior utilities.

The JAX backend asks every model for two things related to the optional FP
prior:

1. one per-normalization-draw sufficient-statistics row;
2. one extra prior value and the diagnostic coefficients derived from the
   summed sufficient statistics.

This module owns the model-agnostic algebra for the current hunit-aware
one-predictor relation, `log_sigma = intercept + beta_mass * (logM* - pivot)`.
The old radius-slope diagnostic slot is still returned as NaN because output
files and plotting tools expect a stable five-value tuple.
"""

from __future__ import annotations

import jax.numpy as jnp


FP_OLS_SUMMARY_SIZE = 6
FP_OLS_COUNT_INDEX = 0
FP_OLS_SUM_X1_INDEX = 1
FP_OLS_SUM_X1X1_INDEX = 2
FP_OLS_SUM_Y_INDEX = 3
FP_OLS_SUM_X1Y_INDEX = 4
FP_OLS_SUM_YY_INDEX = 5


def fp_ols_summary_row(x1: jnp.ndarray, y: jnp.ndarray, valid: jnp.ndarray) -> jnp.ndarray:
    """
    Return one sufficient-statistics row for the 1D FP regression.

    Parameters
    ----------
    x1:
        Predictor value, usually `logM* - fp_pivot_mstar`.
    y:
        Response value, currently the model-predicted `log10(sigma)`.
    valid:
        Boolean JAX scalar.  Invalid draws contribute an all-zero row so the
        backend can sum rows with `vmap` without Python-side filtering.
    """

    row = jnp.asarray(
        [
            1.0,
            x1,
            x1 * x1,
            y,
            x1 * y,
            y * y,
        ],
        dtype=jnp.float64,
    )
    return jnp.where(valid, row, jnp.zeros(FP_OLS_SUMMARY_SIZE, dtype=jnp.float64))


def solve_fundamental_plane_ols_jax(fp_summary: jnp.ndarray) -> tuple[jnp.ndarray, ...]:
    """
    Fit the hunit-aware 1D sigma-logM* relation from sufficient statistics.

    The fitted radius coefficient is intentionally absent from this regression.
    We return NaN for that diagnostic slot to preserve the output table shape
    while making the missing coefficient explicit.
    """

    sample_count = fp_summary[FP_OLS_COUNT_INDEX]
    xtx = jnp.asarray(
        [
            [sample_count, fp_summary[FP_OLS_SUM_X1_INDEX]],
            [fp_summary[FP_OLS_SUM_X1_INDEX], fp_summary[FP_OLS_SUM_X1X1_INDEX]],
        ],
        dtype=jnp.float64,
    )
    xty = jnp.asarray(
        [
            fp_summary[FP_OLS_SUM_Y_INDEX],
            fp_summary[FP_OLS_SUM_X1Y_INDEX],
        ],
        dtype=jnp.float64,
    )
    coefficients = jnp.linalg.solve(xtx, xty)
    sse = fp_summary[FP_OLS_SUM_YY_INDEX] - jnp.dot(coefficients, xty)
    sse = jnp.where((sse < 0.0) & (jnp.abs(sse) < 1.0e-12), 0.0, sse)
    scatter = jnp.sqrt(sse / sample_count)
    valid = (sample_count >= 2.0) & (sse >= 0.0) & jnp.all(jnp.isfinite(coefficients)) & jnp.isfinite(scatter)
    nan = jnp.asarray(jnp.nan, dtype=jnp.float64)
    return (
        jnp.where(valid, coefficients[0], nan),
        jnp.where(valid, coefficients[1], nan),
        nan,
        jnp.where(valid, scatter, nan),
    )


def gaussian_quadratic_log_penalty(value: jnp.ndarray, mean: float, sigma: float) -> jnp.ndarray:
    """
    Return the unnormalized Gaussian quadratic log penalty.

    This function deliberately omits the normalization constant because the
    sampler only needs posterior ratios.  Invalid or non-positive error scales
    are converted to `-inf` so the backend can reject the parameter vector.
    """

    z = (value - mean) / sigma
    return jnp.where((sigma > 0.0) & jnp.isfinite(value), -0.5 * z * z, -jnp.inf)


def fp_prior_value(
    fp_summary: jnp.ndarray,
    *,
    fp_enabled: int,
    fp_fiducial_scatter: float,
    fp_scatter_error: float,
    fp_mu_v_prior: float,
    fp_mu_v_error: float,
    fp_beta_v_prior: float,
    fp_beta_v_error: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Return the optional FP prior value and fitted diagnostic coefficients.

    The disabled path is intentionally neutral: it contributes zero to the log
    posterior and NaN diagnostics, matching the backend's historical output
    convention.
    """

    if fp_enabled == 0:
        nan = jnp.asarray(jnp.nan, dtype=jnp.float64)
        return jnp.asarray(0.0, dtype=jnp.float64), nan, nan, nan, nan

    intercept, beta_mass, beta_radius, scatter = solve_fundamental_plane_ols_jax(fp_summary)
    log_prior = (
        gaussian_quadratic_log_penalty(scatter, fp_fiducial_scatter, fp_scatter_error)
        + gaussian_quadratic_log_penalty(intercept, fp_mu_v_prior, fp_mu_v_error)
        + gaussian_quadratic_log_penalty(beta_mass, fp_beta_v_prior, fp_beta_v_error)
    )
    return (
        log_prior,
        intercept,
        beta_mass,
        beta_radius,
        scatter,
    )


__all__ = [
    "FP_OLS_SUMMARY_SIZE",
    "fp_ols_summary_row",
    "fp_prior_value",
    "gaussian_quadratic_log_penalty",
    "solve_fundamental_plane_ols_jax",
]
