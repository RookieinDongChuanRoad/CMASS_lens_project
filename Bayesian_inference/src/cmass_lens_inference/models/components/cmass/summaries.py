"""FP-prior summaries and extra-prior hook for the default CMASS model."""

from __future__ import annotations

import jax.numpy as jnp

from ....jax_backend.primitives import interp_sigma_unit_clip_scalar as _interp_sigma_unit_clip_scalar
from ..common import fp_prior
from .context import CMASSJaxContext
from .parameters import CMASSTheta
from .population import CMASSPopulationDraw


def summary_row(
    theta_parts: CMASSTheta,
    draw: CMASSPopulationDraw,
    context: CMASSJaxContext,
    static: dict[str, int],
) -> jnp.ndarray:
    """
    Return one FP OLS sufficient-statistics row for one MC draw.

    The theta argument is accepted for the common backend hook signature.  The
    current FP summary depends on the latent population draw and context only.
    """

    del theta_parts
    scalar_context = context.scalar_context
    fp_enabled = static["fp_enabled"]
    fp_has_n_axis = static["fp_has_n_axis"]
    fp_fit_mstar_min = scalar_context[17]
    fp_pivot_mstar = scalar_context[18]
    sigma_unit = _interp_sigma_unit_clip_scalar(
        draw.gamma,
        draw.zd,
        draw.re_draw,
        draw.n_value,
        context.fp_gamma_axis,
        context.fp_zd_axis,
        context.fp_log_re_kpc_axis,
        context.fp_n_axis,
        context.fp_sigma_unit_grid,
        fp_has_n_axis,
    )
    log_sigma_model = 0.5 * (jnp.log10(sigma_unit) + draw.log_enclosed_mass)
    fp_valid = (
        (fp_enabled == 1)
        & (draw.zd > 0.0)
        & jnp.isfinite(draw.gamma)
        & (draw.mstar > fp_fit_mstar_min)
        & (sigma_unit > 0.0)
        & jnp.isfinite(sigma_unit)
        & jnp.isfinite(log_sigma_model)
    )
    return fp_prior.fp_ols_summary_row(
        draw.mstar - fp_pivot_mstar,
        log_sigma_model,
        fp_valid,
    )


def extra_prior(
    fp_summary: jnp.ndarray,
    context: CMASSJaxContext,
    static: dict[str, int],
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return FP-prior contribution and diagnostics."""

    scalar_context = context.scalar_context
    return fp_prior.fp_prior_value(
        fp_summary,
        fp_enabled=static["fp_enabled"],
        fp_fiducial_scatter=scalar_context[19],
        fp_scatter_error=scalar_context[20],
        fp_mu_v_prior=scalar_context[21],
        fp_mu_v_error=scalar_context[22],
        fp_beta_v_prior=scalar_context[23],
        fp_beta_v_error=scalar_context[24],
    )


__all__ = ["extra_prior", "summary_row"]
