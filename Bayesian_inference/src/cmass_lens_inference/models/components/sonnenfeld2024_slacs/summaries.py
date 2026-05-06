"""Diagnostics and extra-prior hooks for the Sonnenfeld 2024 SLACS model."""

from __future__ import annotations

import jax.numpy as jnp

from ..common import fp_prior
from .context import SonnenfeldJaxContext
from .parameters import SonnenfeldTheta
from .population import SonnenfeldPopulationDraw


def summary_row(
    theta_parts: SonnenfeldTheta,
    draw: SonnenfeldPopulationDraw,
    context: SonnenfeldJaxContext,
    static: dict[str, int],
) -> jnp.ndarray:
    """
    Return a neutral diagnostics row for one normalization draw.

    Sonnenfeld v1 does not enable an additional FP prior.  The backend still
    expects every model to return a fixed-width summary row, so this hook
    returns the shared all-zero sufficient-statistics vector.
    """

    del theta_parts, draw, context, static
    return jnp.zeros(fp_prior.FP_OLS_SUMMARY_SIZE, dtype=jnp.float64)


def extra_prior(
    fp_summary: jnp.ndarray,
    context: SonnenfeldJaxContext,
    static: dict[str, int],
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Return a neutral extra prior and NaN diagnostics.

    The hook shape matches CMASS so the common likelihood engine and output
    writer do not need a model-specific branch.  Scientific priors can be added
    later as explicit Sonnenfeld components.
    """

    del fp_summary, context, static
    nan = jnp.asarray(jnp.nan, dtype=jnp.float64)
    return jnp.asarray(0.0, dtype=jnp.float64), nan, nan, nan, nan


__all__ = ["extra_prior", "summary_row"]
