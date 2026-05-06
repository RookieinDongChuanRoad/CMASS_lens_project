"""Selection functions for the Sonnenfeld 2024 SLACS model."""

from __future__ import annotations

import math

import jax.numpy as jnp

from ....jax_backend.primitives import (
    C_KM_S,
    interp_cross_section_theta_gamma,
    interp_sigma_unit_clip_scalar,
    theta_ein_arcsec,
)
from ....jax_backend.selection import sigmoid_find_probability
from .context import SonnenfeldJaxContext
from .parameters import SonnenfeldTheta
from .population import SonnenfeldPopulationDraw


def theta_e_est_from_sigma_proxy(
    sigma_proxy: jnp.ndarray,
    zd: jnp.ndarray,
    zs: jnp.ndarray,
    context: SonnenfeldJaxContext,
) -> jnp.ndarray:
    """
    Convert a velocity-dispersion proxy into the estimated Einstein radius.

    This is the Sonnenfeld-specific selection observable.  It intentionally
    differs from CMASS, where ``P_find`` is evaluated on the true lensing
    Einstein radius.  Distances are read from the same cosmology table used by
    the power-law lensing primitive.
    """

    chi_l = jnp.interp(zd, context.z_grid, context.chi_kpc_grid)
    chi_s = jnp.interp(zs, context.z_grid, context.chi_kpc_grid)
    ds = chi_s / (1.0 + zs)
    dls = (chi_s - chi_l) / (1.0 + zs)
    theta_arcsec = 4.0 * math.pi * (sigma_proxy / C_KM_S) ** 2 * dls / ds * 206265.0
    return jnp.where((zs > zd) & (ds > 0.0) & (dls > 0.0), theta_arcsec, 0.0)


def selection_weight_from_normal(
    theta_parts: SonnenfeldTheta,
    draw: SonnenfeldPopulationDraw,
    nrm: jnp.ndarray,
    context: SonnenfeldJaxContext,
    static: dict[str, int],
) -> jnp.ndarray:
    """
    Return the Sonnenfeld selection-normalization weight for one MC draw.

    The weight combines the parent-population density, finite-fibre
    cross-section, velocity-proxy finding probability, and source-redshift
    support.  The random-basis column ``nrm[7]`` supplies the measurement proxy
    scatter used to construct ``theta_E_est``.
    """

    del static
    scalar = context.scalar_context
    theta_e = theta_ein_arcsec(
        draw.zd,
        draw.zs,
        draw.log_enclosed_mass,
        draw.gamma,
        context.z_grid,
        context.chi_kpc_grid,
        scalar[0],
        scalar[1],
    )
    cross_section = interp_cross_section_theta_gamma(
        theta_e,
        draw.gamma,
        context.cs_theta_e_axis,
        context.cs_gamma_axis,
        context.cs_cross_section_grid,
    )
    sigma_unit = interp_sigma_unit_clip_scalar(
        draw.gamma,
        draw.zd,
        draw.log_re,
        draw.n_value,
        context.population_gamma_axis,
        context.population_zd_axis,
        context.population_log_re_kpc_axis,
        context.population_n_axis,
        context.population_sigma_unit_grid,
        1,
    )
    sigma_model = jnp.sqrt(jnp.maximum(sigma_unit * 10.0 ** draw.log_enclosed_mass, 1.0e-30))
    sigma_proxy = sigma_model * jnp.maximum(0.1, 1.0 + scalar[17] * nrm[7])
    theta_e_est = theta_e_est_from_sigma_proxy(sigma_proxy, draw.zd, draw.zs, context)
    p_find = sigmoid_find_probability(theta_e_est, theta_parts.theta0, theta_parts.loga)
    parent_proposal_correction = draw.parent_density / draw.proposal_density
    valid = (
        (draw.zd > 0.0)
        & (draw.zs > draw.zd)
        & (draw.mstar >= scalar[15])
        & (draw.mstar <= scalar[16])
        & (theta_e > 0.0)
        & (cross_section > 0.0)
        & (sigma_unit > 0.0)
        & (theta_e_est > 0.0)
        & jnp.isfinite(p_find)
    )
    return jnp.where(
        valid,
        cross_section * p_find * parent_proposal_correction * draw.source_density_over_proposal,
        0.0,
    )


__all__ = ["selection_weight_from_normal", "theta_e_est_from_sigma_proxy"]
