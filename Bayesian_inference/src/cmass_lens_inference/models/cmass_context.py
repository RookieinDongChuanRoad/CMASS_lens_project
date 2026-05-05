"""Array context for the default CMASS model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np


class CMASSJaxContext(NamedTuple):
    """
    JAX pytree context consumed by the CMASS scientific hooks.

    This type is intentionally colocated with `CMASSModelContext`: the NumPy
    dataclass describes the validated source context, while this NamedTuple
    describes the traced JAX view generated from the model's `DataSpec`.
    """

    z_grid: jnp.ndarray
    chi_kpc_grid: jnp.ndarray
    cs_gamma_grid: jnp.ndarray
    cs_over_theta_grid: jnp.ndarray
    cs_theta_e_axis: jnp.ndarray
    cs_cross_section_grid: jnp.ndarray
    cs_over_theta_int: jnp.ndarray
    gamma_grid_int: jnp.ndarray
    mass_grid_int: jnp.ndarray
    dmass_dthetaein_grid_int: jnp.ndarray
    s2_grid_int: jnp.ndarray
    has_s2: jnp.ndarray
    num_sigma: jnp.ndarray
    sigma_obs: jnp.ndarray
    sigma_err: jnp.ndarray
    zd: jnp.ndarray
    zs: jnp.ndarray
    p_zd_fixed: jnp.ndarray
    mstar_grid: jnp.ndarray
    mstar_shift11p4: jnp.ndarray
    sigma_star_shift9p0_grid: jnp.ndarray
    mstar_integrand_base: jnp.ndarray
    delta_r_grid: jnp.ndarray
    base_normals: jnp.ndarray
    scalar_context: jnp.ndarray
    fp_gamma_axis: jnp.ndarray
    fp_zd_axis: jnp.ndarray
    fp_log_re_kpc_axis: jnp.ndarray
    fp_n_axis: jnp.ndarray
    fp_sigma_unit_grid: jnp.ndarray


@dataclass(frozen=True)
class CMASSModelContext:
    """
    Parameter-independent arrays consumed by the CMASS JAX hooks.

    This context is model-specific.  Generic orchestration stores it as an
    opaque object and routes it through the active model definition.
    """

    z_grid: np.ndarray
    chi_kpc_grid: np.ndarray
    cs_gamma_grid: np.ndarray
    cs_over_theta_grid: np.ndarray
    cs_theta_e_axis: np.ndarray
    cs_cross_section_grid: np.ndarray
    cs_over_theta_int: np.ndarray
    gamma_grid_int: np.ndarray
    mass_grid_int: np.ndarray
    dmass_dthetaein_grid_int: np.ndarray
    s2_grid_int: np.ndarray
    has_s2: np.ndarray
    num_sigma: np.ndarray
    sigma_obs: np.ndarray
    sigma_err: np.ndarray
    zd: np.ndarray
    zs: np.ndarray
    p_zd_fixed: np.ndarray
    mstar_grid: np.ndarray
    mstar_shift11p4: np.ndarray
    stellar_mass_pivot: float
    sigma_star_shift9p0_grid: np.ndarray
    mstar_integrand_base: np.ndarray
    delta_r_grid: np.ndarray
    base_normals: np.ndarray
    mass_radius_kpc: float
    mass_log_physical_offset: float
    use_sersic_index: int
    n_fixed: float
    mu_n0: float
    beta_n: float
    sigma_n: float
    mass_function_loc: float
    mass_function_scale: float
    mass_function_alpha: float
    mu_r0: float
    beta_r: float
    sigma_r: float
    nu_r: float
    mu_d: float
    sigma_d: float
    gamma_trunc_low: float
    gamma_trunc_high: float
    normalization_min_value: float
    gamma_mode_code: int
    fp_enabled: int
    fp_fit_mstar_min: float
    fp_pivot_mstar: float
    fp_fiducial_scatter: float
    fp_scatter_error: float
    fp_mu_v_prior: float
    fp_mu_v_error: float
    fp_beta_v_prior: float
    fp_beta_v_error: float
    fp_gamma_axis: np.ndarray
    fp_zd_axis: np.ndarray
    fp_log_re_kpc_axis: np.ndarray
    fp_n_axis: np.ndarray
    fp_sigma_unit_grid: np.ndarray
    fp_has_n_axis: int


__all__ = ["CMASSJaxContext", "CMASSModelContext"]
