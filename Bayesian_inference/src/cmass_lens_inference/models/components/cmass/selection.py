"""Selection weight for the default CMASS model."""

from __future__ import annotations

import jax.numpy as jnp

from ....jax_backend.primitives import (
    interp_cross_section_theta_gamma as _interp_cross_section_theta_gamma,
    phi_standard as _phi_standard,
    theta_ein_arcsec as _theta_ein_arcsec,
)
from ....jax_backend.selection import sigmoid_find_probability as _p_find
from .context import CMASSJaxContext
from .parameters import CMASSTheta
from .population import CMASSPopulationDraw


def selection_weight_for_source(
    theta_parts: CMASSTheta,
    draw: CMASSPopulationDraw,
    context: CMASSJaxContext,
    zs: jnp.ndarray,
    mass_radius_kpc: jnp.ndarray,
    mass_log_physical_offset: jnp.ndarray,
) -> jnp.ndarray:
    """
    Return the CMASS selection contribution for a known source redshift.

    This expression is shared by normalization draws.  Per-lens likelihoods use
    an equivalent grid expression in `likelihood.py` because those terms are
    vectorized over observed lenses and gamma grids rather than over MC rows.
    """

    z0 = (0.0 - theta_parts.mu_zs) / theta_parts.sigma_zs
    inv_trunc_den = 1.0 / (1.0 - _phi_standard(z0))
    theta_e = _theta_ein_arcsec(
        draw.zd,
        zs,
        draw.log_enclosed_mass,
        draw.gamma,
        context.z_grid,
        context.chi_kpc_grid,
        mass_radius_kpc,
        mass_log_physical_offset,
    )
    cross_section = _interp_cross_section_theta_gamma(
        theta_e,
        draw.gamma,
        context.cs_theta_e_axis,
        context.cs_gamma_grid,
        context.cs_cross_section_grid,
    )
    weight = inv_trunc_den * _p_find(theta_e, theta_parts.theta0, theta_parts.loga) * cross_section
    valid = (
        (draw.zd > 0.0)
        & (zs > 0.0)
        & (zs > draw.zd)
        & jnp.isfinite(draw.gamma)
        & (theta_e > 0.0)
        & (cross_section > 0.0)
    )
    return jnp.where(valid, weight, 0.0)


def selection_weight_from_normal(
    theta_parts: CMASSTheta,
    draw: CMASSPopulationDraw,
    nrm: jnp.ndarray,
    context: CMASSJaxContext,
    static: dict[str, int],
) -> jnp.ndarray:
    """Return selection weight using the source-redshift normal in the MC row."""

    del static
    scalar_context = context.scalar_context
    zs = theta_parts.mu_zs + theta_parts.sigma_zs * nrm[1]
    return selection_weight_for_source(
        theta_parts,
        draw,
        context,
        zs,
        scalar_context[0],
        scalar_context[26],
    )


__all__ = ["selection_weight_for_source", "selection_weight_from_normal"]

