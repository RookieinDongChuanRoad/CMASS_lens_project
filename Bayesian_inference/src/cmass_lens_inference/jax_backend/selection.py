"""
Reusable JAX selection helpers.

Concrete models decide which selection function is scientifically appropriate.
The helpers here are small probability kernels that models can compose with
their own cross-section, lensing, or survey-finding terms.
"""

from __future__ import annotations

import jax.numpy as jnp


def sigmoid_find_probability(theta_est: jnp.ndarray, theta0: jnp.ndarray, loga: jnp.ndarray) -> jnp.ndarray:
    """
    Numerically stable sigmoid lens-finding probability.

    ``loga`` stores the base-10 logarithm of the slope so the sampled parameter
    stays on a convenient order-unity scale.  Saturating extreme exponent
    values avoids unnecessary ``inf`` values inside JIT-compiled likelihoods.
    """

    a = 10.0**loga
    x = -a * (theta_est - theta0)
    return jnp.where(
        x > 60.0,
        0.0,
        jnp.where(x < -60.0, 1.0, 1.0 / (1.0 + jnp.exp(x))),
    )


__all__ = ["sigmoid_find_probability"]
