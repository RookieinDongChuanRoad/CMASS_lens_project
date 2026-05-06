"""Per-lens likelihood integral for the default CMASS model."""

from __future__ import annotations

import jax.numpy as jnp

from ....jax_backend.primitives import (
    interp_cross_section_theta_gamma as _interp_cross_section_theta_gamma,
    normal_pdf as _normal_pdf,
    theta_ein_arcsec as _theta_ein_arcsec,
    trapezoid_last_axis as _trapezoid_last_axis,
    truncated_normal_pdf_nonneg as _truncated_normal_pdf_nonneg,
)
from ....jax_backend.selection import sigmoid_find_probability as _p_find
from .context import CMASSJaxContext
from .parameters import CMASSTheta
from .population import gamma_population_mean


def lens_integrals(
    theta_parts: CMASSTheta,
    context: CMASSJaxContext,
    static: dict[str, int],
) -> jnp.ndarray:
    """
    Return the per-lens CMASS likelihood integrals before log reduction.

    The generic backend owns the final reduction `sum(log(integral))`; this
    hook only computes the model-specific integral over lensing mass, gamma,
    source redshift, observed stellar mass/size, and optional sigma data.
    """

    del static
    scalar_context = context.scalar_context
    log_enclosed_mass = context.mass_grid_int
    gamma = context.gamma_grid_int[None, :]
    jac = jnp.abs(context.dmass_dthetaein_grid_int)
    theta_e = _theta_ein_arcsec(
        context.zd[:, None],
        context.zs[:, None],
        log_enclosed_mass,
        gamma,
        context.z_grid,
        context.chi_kpc_grid,
        scalar_context[0],
        scalar_context[26],
    )
    cross_section = _interp_cross_section_theta_gamma(
        theta_e,
        gamma,
        context.cs_theta_e_axis,
        context.cs_gamma_grid,
        context.cs_cross_section_grid,
    )
    pf = _p_find(theta_e, theta_parts.theta0, theta_parts.loga)
    p_zs = _truncated_normal_pdf_nonneg(context.zs, theta_parts.mu_zs, theta_parts.sigma_zs)

    sigma_model = jnp.sqrt(jnp.maximum(context.s2_grid_int * (10.0**log_enclosed_mass), 1.0e-30))
    p_sigma_1 = _normal_pdf(context.sigma_obs[:, None, 0], sigma_model, context.sigma_err[:, None, 0])
    p_sigma_2 = _normal_pdf(context.sigma_obs[:, None, 1], sigma_model, context.sigma_err[:, None, 1])
    p_sigma = jnp.where(context.num_sigma[:, None] >= 1, p_sigma_1, 1.0)
    p_sigma = jnp.where(context.num_sigma[:, None] >= 2, p_sigma * p_sigma_2, p_sigma)
    p_sigma = jnp.where((context.num_sigma[:, None] > 0) & (context.has_s2[:, None] == 0), 0.0, p_sigma)

    mu5 = (
        theta_parts.mu5_0
        + theta_parts.beta5 * context.mstar_shift11p4
        + theta_parts.xi5 * context.delta_r_grid
    )
    mu_gamma = gamma_population_mean(
        theta_parts.mu_gamma_0,
        theta_parts.beta_sigma_star_gamma,
        context.sigma_star_shift9p0_grid,
    )
    mstar_density = (
        context.mstar_integrand_base[:, None, :]
        * _normal_pdf(log_enclosed_mass[:, :, None], mu5[:, None, :], theta_parts.sigma5)
        * _normal_pdf(context.gamma_grid_int[None, :, None], mu_gamma[:, None, :], theta_parts.sigma_gamma)
    )
    integrated_mstar = _trapezoid_last_axis(mstar_density, context.mstar_grid[:, None, :])
    gamma_integrand = (
        integrated_mstar
        * context.p_zd_fixed[:, None]
        * p_zs[:, None]
        * pf
        * cross_section
        * jac
        * p_sigma
    )
    gamma_valid = (jac > 0.0) & (theta_e > 0.0) & (cross_section > 0.0) & (pf > 0.0) & (p_sigma > 0.0)
    gamma_integrand = jnp.where(gamma_valid, gamma_integrand, 0.0)
    return _trapezoid_last_axis(gamma_integrand, context.gamma_grid_int[None, :])


__all__ = ["lens_integrals"]

