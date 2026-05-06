"""Effective source-redshift terms for the Sonnenfeld 2024 SLACS model.

Sonnenfeld Equation 33 introduces an *effective* source-redshift distribution
after marginalizing source-light variables into the detectability factor.  The
paper writes this term as an ordinary Gaussian, not as a Gaussian truncated at
``z_s = 0``.  Lensing geometry and selection weights still become zero when a
draw has ``z_s <= z_d``; that physical support cut belongs to the lensing
kernel, not to the source-redshift prior itself.
"""

from __future__ import annotations

import jax.numpy as jnp

from ....jax_backend.primitives import normal_pdf
from .parameters import SonnenfeldTheta


def effective_source_redshift_density(
    zs: jnp.ndarray,
    theta_parts: SonnenfeldTheta,
) -> jnp.ndarray:
    """Return the ordinary Gaussian ``P_s^eff(z_s | eta)`` from Equation 33."""

    return normal_pdf(zs, theta_parts.mu_zs, theta_parts.sigma_zs)


def source_density_over_normal_proposal(
    zs: jnp.ndarray,
    theta_parts: SonnenfeldTheta,
) -> jnp.ndarray:
    """
    Return target/proposal for MC draws sampled from the same Gaussian.

    Normalization draws use ``z_s = mu_zs + sigma_zs * epsilon``.  Because this
    proposal is exactly the paper's effective source distribution, the
    importance ratio is one wherever the Gaussian density is finite.  Keeping
    this helper explicit avoids accidentally reintroducing the older truncated
    source-redshift approximation.
    """

    density = effective_source_redshift_density(zs, theta_parts)
    return jnp.where(density > 0.0, 1.0, 0.0)


__all__ = ["effective_source_redshift_density", "source_density_over_normal_proposal"]
