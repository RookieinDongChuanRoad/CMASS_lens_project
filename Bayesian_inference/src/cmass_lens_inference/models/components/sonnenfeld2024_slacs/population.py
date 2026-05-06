"""Latent population draw for the Sonnenfeld 2024 SLACS model."""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from ....jax_backend.primitives import (
    normal_pdf,
    phi_standard,
    truncnorm_sample,
)
from .context import SonnenfeldJaxContext
from . import parameters, source
from .parameters import SonnenfeldTheta


class SonnenfeldPopulationDraw(NamedTuple):
    """
    One latent draw from the Sonnenfeld foreground/source population.

    The draw is used only on the Monte Carlo normalization side.  Per-lens
    likelihoods condition on observed lens redshift, source redshift, stellar
    mass, and size through quadrature grids.
    """

    zd: jnp.ndarray
    zs: jnp.ndarray
    mstar: jnp.ndarray
    n_value: jnp.ndarray
    log_re: jnp.ndarray
    delta_r: jnp.ndarray
    log_enclosed_mass: jnp.ndarray
    gamma: jnp.ndarray
    parent_density: jnp.ndarray
    proposal_density: jnp.ndarray
    source_density_over_proposal: jnp.ndarray


def size_relation_mean(
    mstar: jnp.ndarray,
    n_value: jnp.ndarray,
    context: SonnenfeldJaxContext,
    static: dict[str, int],
) -> jnp.ndarray:
    """Return Sonnenfeld Equation 29 in the active canonical size coordinate."""

    scalar = context.scalar_context
    del n_value, static
    size_mu0 = scalar[6]
    size_mu1 = scalar[7]
    size_mu2 = scalar[9]
    return size_mu0 + size_mu1 * mstar + size_mu2 * mstar * mstar


def _active_truncation_mass_threshold(
    zd: jnp.ndarray,
    context: SonnenfeldJaxContext,
) -> jnp.ndarray:
    """
    Evaluate Equation 27's mass-threshold polynomial in the active coordinate.

    The paper coefficients are physical stellar-mass locations.  The context's
    active stellar-mass pivot is either the physical pivot or that same pivot
    shifted by ``2 log10(h_ref)``.  The pivot difference therefore gives the
    exact location shift needed by the hunit branch without adding a JAX-time
    unit-convention branch.
    """

    powers = jnp.asarray([zd**index for index in range(6)])
    physical_threshold = jnp.dot(
        jnp.asarray(parameters.TRUNCATION_MASS_POLYNOMIAL_COEFFICIENTS),
        powers,
    )
    active_mass_shift = context.scalar_context[2] - parameters.MSTAR_PIVOT_PHYSICAL
    return physical_threshold + active_mass_shift


def _parent_density_for_draw(
    zd: jnp.ndarray,
    mstar: jnp.ndarray,
    context: SonnenfeldJaxContext,
) -> jnp.ndarray:
    """Return the Table-1 parent density used by normalization draws."""

    scalar = context.scalar_context
    threshold = _active_truncation_mass_threshold(zd, context)
    completeness_argument = (mstar - threshold) / scalar[5]
    completeness = jnp.arctan(completeness_argument) / jnp.pi + 0.5
    schechter_mass = 10.0 ** (mstar - scalar[3])
    schechter = 10.0 ** ((mstar - scalar[3]) * (scalar[4] + 1.0))
    return (jnp.maximum(zd, 1.0e-6) ** 2) * completeness * schechter * jnp.exp(
        -schechter_mass
    )


def _truncated_normal_pdf(
    x: jnp.ndarray,
    loc: jnp.ndarray,
    scale: jnp.ndarray,
    low: jnp.ndarray,
    high: jnp.ndarray,
) -> jnp.ndarray:
    """PDF of the finite-support Gaussian proposal used by MC normalization."""

    denominator = phi_standard((high - loc) / scale) - phi_standard((low - loc) / scale)
    density = normal_pdf(x, loc, scale) / jnp.maximum(denominator, 1.0e-300)
    return jnp.where((x >= low) & (x <= high) & (denominator > 0.0), density, 0.0)


def draw_population(
    theta_parts: SonnenfeldTheta,
    nrm: jnp.ndarray,
    context: SonnenfeldJaxContext,
    static: dict[str, int],
) -> SonnenfeldPopulationDraw:
    """
    Map one fixed normal row into a Sonnenfeld latent population state.

    The first implementation uses deterministic truncated-normal transforms for
    the parent ``z_d`` and ``m*`` support.  The selection correction still uses
    the Table-1 parent density through the normalization weight, so the draw
    distribution itself is only an integration proposal.
    """

    scalar = context.scalar_context
    zd_center = 0.5 * (scalar[13] + scalar[14])
    zd_scale = 0.25
    mstar_center = 0.5 * (scalar[15] + scalar[16])
    mstar_scale = 0.35
    zd = truncnorm_sample(zd_center, zd_scale, scalar[13], scalar[14], nrm[0])
    mstar = truncnorm_sample(mstar_center, mstar_scale, scalar[15], scalar[16], nrm[1])
    zs = theta_parts.mu_zs + theta_parts.sigma_zs * nrm[2]
    n_value = jnp.where(static["use_sersic_index"] == 1, 4.0 + 0.4 * nrm[3], scalar[10])
    n_value = jnp.maximum(n_value, 0.5)
    mu_r = size_relation_mean(mstar, n_value, context, static)
    log_re = mu_r + scalar[8] * nrm[4]
    delta_r = log_re - mu_r
    mstar_shift = mstar - scalar[2]
    log_enclosed_mass = (
        theta_parts.mu5_0
        + theta_parts.beta5 * mstar_shift
        + theta_parts.xi5 * delta_r
        + theta_parts.sigma5 * nrm[5]
    )
    mu_gamma = (
        theta_parts.mu_gamma_0
        + theta_parts.beta_gamma * mstar_shift
        + theta_parts.xi_gamma * delta_r
    )
    gamma = truncnorm_sample(mu_gamma, theta_parts.sigma_gamma, scalar[11], scalar[12], nrm[6])
    parent_density = _parent_density_for_draw(zd, mstar, context)
    proposal_density = _truncated_normal_pdf(
        zd,
        zd_center,
        zd_scale,
        scalar[13],
        scalar[14],
    ) * _truncated_normal_pdf(
        mstar,
        mstar_center,
        mstar_scale,
        scalar[15],
        scalar[16],
    )
    return SonnenfeldPopulationDraw(
        zd=zd,
        zs=zs,
        mstar=mstar,
        n_value=n_value,
        log_re=log_re,
        delta_r=delta_r,
        log_enclosed_mass=log_enclosed_mass,
        gamma=gamma,
        parent_density=parent_density,
        proposal_density=jnp.maximum(proposal_density, 1.0e-300),
        source_density_over_proposal=source.source_density_over_normal_proposal(zs, theta_parts),
    )


__all__ = [
    "SonnenfeldPopulationDraw",
    "draw_population",
    "size_relation_mean",
]
