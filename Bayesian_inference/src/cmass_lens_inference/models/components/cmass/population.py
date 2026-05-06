"""Latent parent-population draw for the default CMASS model."""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from ....jax_backend.primitives import (
    LOG10_2PI,
    LOG10_4,
    skewnorm_sample as _skewnorm_sample,
    truncnorm_sample as _truncnorm_sample,
)
from .context import CMASSJaxContext
from .parameters import CMASSTheta


class CMASSPopulationDraw(NamedTuple):
    """
    One latent draw from the CMASS foreground population.

    This is the normalization-side latent state used by both selection weights
    and FP-prior summaries.  It intentionally contains only parameter-dependent
    values; deterministic grids stay in `CMASSJaxContext`.
    """

    zd: jnp.ndarray
    mstar: jnp.ndarray
    n_value: jnp.ndarray
    re_draw: jnp.ndarray
    delta_r: jnp.ndarray
    log_enclosed_mass: jnp.ndarray
    gamma: jnp.ndarray


def mu_r(
    mstar: jnp.ndarray,
    n_value: jnp.ndarray,
    use_sersic_index: int,
    mu_r0: float,
    beta_r: float,
    nu_r: float,
    stellar_mass_pivot: float,
) -> jnp.ndarray:
    """Return the mean size relation for the active profile family."""

    sersic_term = nu_r * (jnp.log10(jnp.maximum(n_value, 1.0e-12)) - LOG10_4)
    return mu_r0 + beta_r * (mstar - stellar_mass_pivot) + jnp.where(use_sersic_index == 1, sersic_term, 0.0)


def gamma_population_mean(
    mu_gamma_0: jnp.ndarray,
    beta_sigma_star_gamma: jnp.ndarray,
    sigma_star_shift9p0: jnp.ndarray,
) -> jnp.ndarray:
    """Conditional mean of gamma for the fixed sigma-star CMASS model."""

    return mu_gamma_0 + beta_sigma_star_gamma * sigma_star_shift9p0


def draw_population(
    theta_parts: CMASSTheta,
    nrm: jnp.ndarray,
    context: CMASSJaxContext,
    static: dict[str, int],
) -> CMASSPopulationDraw:
    """
    Draw one latent parent-population state from one fixed normal row.

    The backend supplies deterministic standard-normal rows.  This hook maps
    one row into the CMASS foreground redshift, stellar mass, size, enclosed
    mass, and density-slope variables used by the normalization integral.
    """

    scalar_context = context.scalar_context
    mass_function_loc = scalar_context[5]
    mass_function_scale = scalar_context[6]
    mass_function_alpha = scalar_context[7]
    mu_r0 = scalar_context[8]
    beta_r = scalar_context[9]
    sigma_r = scalar_context[10]
    nu_r = scalar_context[11]
    mu_d = scalar_context[12]
    sigma_d = scalar_context[13]
    gamma_trunc_low = scalar_context[14]
    gamma_trunc_high = scalar_context[15]
    stellar_mass_pivot = scalar_context[25]
    use_sersic_index = static["use_sersic_index"]
    n_fixed = scalar_context[1]
    mu_n0 = scalar_context[2]
    beta_n = scalar_context[3]
    sigma_n = scalar_context[4]
    zd = mu_d + sigma_d * nrm[0]
    mstar = _skewnorm_sample(
        mass_function_loc,
        mass_function_scale,
        mass_function_alpha,
        nrm[2],
        nrm[3],
    )

    mstar_shift = mstar - stellar_mass_pivot
    logn = mu_n0 + beta_n * mstar_shift + sigma_n * nrm[4]
    n_draw = 10.0**logn
    n_value = jnp.where(use_sersic_index == 1, n_draw, n_fixed)
    mu_r_draw = mu_r(mstar, n_value, use_sersic_index, mu_r0, beta_r, nu_r, stellar_mass_pivot)
    re_noise_column = jnp.where(use_sersic_index == 1, nrm[5], nrm[4])
    mass_noise_column = jnp.where(use_sersic_index == 1, nrm[6], nrm[5])
    re_draw = mu_r_draw + sigma_r * re_noise_column
    delta_r = re_draw - mu_r_draw
    log_enclosed_mass = (
        theta_parts.mu5_0
        + theta_parts.beta5 * mstar_shift
        + theta_parts.xi5 * delta_r
        + theta_parts.sigma5 * mass_noise_column
    )

    sigma_star_shift9p0 = mstar - LOG10_2PI - 2.0 * re_draw - 9.0
    mu_gamma = gamma_population_mean(
        theta_parts.mu_gamma_0,
        theta_parts.beta_sigma_star_gamma,
        sigma_star_shift9p0,
    )
    gamma = _truncnorm_sample(
        mu_gamma,
        theta_parts.sigma_gamma,
        gamma_trunc_low,
        gamma_trunc_high,
        nrm[7],
    )
    return CMASSPopulationDraw(
        zd=zd,
        mstar=mstar,
        n_value=n_value,
        re_draw=re_draw,
        delta_r=delta_r,
        log_enclosed_mass=log_enclosed_mass,
        gamma=gamma,
    )


__all__ = [
    "CMASSPopulationDraw",
    "draw_population",
    "gamma_population_mean",
    "mu_r",
]
