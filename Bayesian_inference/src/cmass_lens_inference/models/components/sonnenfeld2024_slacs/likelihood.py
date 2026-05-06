"""Per-lens likelihood integral for the Sonnenfeld 2024 SLACS model."""

from __future__ import annotations

import jax.numpy as jnp

from ....jax_backend.primitives import (
    interp_cross_section_theta_gamma,
    normal_pdf,
    theta_ein_arcsec,
    trapezoid_last_axis,
)
from ....jax_backend.selection import sigmoid_find_probability
from .context import SonnenfeldJaxContext
from .parameters import SonnenfeldTheta
from .selection import theta_e_est_from_sigma_proxy
from .source import effective_source_redshift_density


def lens_integrals(
    theta_parts: SonnenfeldTheta,
    context: SonnenfeldJaxContext,
    static: dict[str, int],
) -> jnp.ndarray:
    """
    Return per-lens Sonnenfeld likelihood integrals before log reduction.

    Each lens integral runs over the observed lens's gamma grid and a latent
    true stellar-mass quadrature grid.  The integrand follows the implementation
    note: parent ``P_g(z_d,m*)``, size/stellar-mass likelihood, mass and gamma
    population terms, effective source-redshift density, velocity-proxy
    ``P_find(theta_E_est)``, finite-fibre cross-section, Jacobian, and optional
    sigma likelihood.
    """

    del static
    scalar = context.scalar_context
    log_enclosed_mass = context.mass_grid_int
    gamma = context.gamma_grid_int[None, :]
    jacobian = jnp.abs(context.dmass_dthetaein_grid_int)
    theta_e = theta_ein_arcsec(
        context.zd[:, None],
        context.zs[:, None],
        log_enclosed_mass,
        gamma,
        context.z_grid,
        context.chi_kpc_grid,
        scalar[0],
        scalar[1],
    )
    cross_section = interp_cross_section_theta_gamma(
        theta_e,
        gamma,
        context.cs_theta_e_axis,
        context.cs_gamma_axis,
        context.cs_cross_section_grid,
    )
    sigma_model = jnp.sqrt(jnp.maximum(context.s2_grid_int * (10.0**log_enclosed_mass), 1.0e-30))
    sigma_find_proxy = jnp.where(
        context.num_sigma[:, None] >= 1,
        context.sigma_obs[:, None, 0],
        sigma_model,
    )
    theta_e_est = theta_e_est_from_sigma_proxy(
        sigma_find_proxy,
        context.zd[:, None],
        context.zs[:, None],
        context,
    )
    p_find = sigmoid_find_probability(theta_e_est, theta_parts.theta0, theta_parts.loga)
    p_zs = effective_source_redshift_density(context.zs, theta_parts)

    p_sigma_1 = normal_pdf(context.sigma_obs[:, None, 0], sigma_model, context.sigma_err[:, None, 0])
    p_sigma_2 = normal_pdf(context.sigma_obs[:, None, 1], sigma_model, context.sigma_err[:, None, 1])
    p_sigma = jnp.where(context.num_sigma[:, None] >= 1, p_sigma_1, 1.0)
    p_sigma = jnp.where(context.num_sigma[:, None] >= 2, p_sigma * p_sigma_2, p_sigma)
    p_sigma = jnp.where((context.num_sigma[:, None] > 0) & (context.has_s2[:, None] == 0), 0.0, p_sigma)

    mu5 = (
        theta_parts.mu5_0
        + theta_parts.beta5 * context.mstar_shift_grid
        + theta_parts.xi5 * context.delta_r_grid
    )
    mu_gamma = (
        theta_parts.mu_gamma_0
        + theta_parts.beta_gamma * context.mstar_shift_grid
        + theta_parts.xi_gamma * context.delta_r_grid
    )
    mstar_density = (
        context.parent_mstar_density_grid[:, None, :]
        * context.size_density_grid[:, None, :]
        * normal_pdf(log_enclosed_mass[:, :, None], mu5[:, None, :], theta_parts.sigma5)
        * normal_pdf(context.gamma_grid_int[None, :, None], mu_gamma[:, None, :], theta_parts.sigma_gamma)
    )
    integrated_mstar = trapezoid_last_axis(mstar_density, context.mstar_grid[:, None, :])
    gamma_integrand = (
        integrated_mstar
        * p_zs[:, None]
        * p_find
        * cross_section
        * jacobian
        * p_sigma
    )
    valid = (
        (jacobian > 0.0)
        & (theta_e > 0.0)
        & (cross_section > 0.0)
        & (p_find > 0.0)
        & (p_sigma > 0.0)
        & jnp.isfinite(gamma_integrand)
    )
    gamma_integrand = jnp.where(valid, gamma_integrand, 0.0)
    return trapezoid_last_axis(gamma_integrand, context.gamma_grid_int[None, :])


__all__ = ["lens_integrals"]
