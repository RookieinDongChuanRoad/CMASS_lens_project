"""Array context for the Sonnenfeld 2024 SLACS model.

This module defines the two context shapes used by the runtime:

- ``SonnenfeldModelContext`` is the validated NumPy source context created
  from a canonical inference dataset.
- ``SonnenfeldJaxContext`` is the traced JAX pytree generated from the source
  context by the generic ``DataSpec`` builder.

Keeping these structures separate from the scientific formulas makes the model
file readable while still documenting every numerical array consumed by the
hot JAX kernels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np


class SonnenfeldJaxContext(NamedTuple):
    """JAX pytree context consumed by Sonnenfeld scientific hooks."""

    z_grid: jnp.ndarray
    chi_kpc_grid: jnp.ndarray
    cs_theta_e_axis: jnp.ndarray
    cs_gamma_axis: jnp.ndarray
    cs_cross_section_grid: jnp.ndarray
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
    log_mstar_obs: jnp.ndarray
    log_mstar_err: jnp.ndarray
    log_re_obs: jnp.ndarray
    n_obs: jnp.ndarray
    theta_e_obs: jnp.ndarray
    mstar_grid: jnp.ndarray
    parent_mstar_density_grid: jnp.ndarray
    size_density_grid: jnp.ndarray
    delta_r_grid: jnp.ndarray
    mstar_shift_grid: jnp.ndarray
    base_normals: jnp.ndarray
    population_gamma_axis: jnp.ndarray
    population_zd_axis: jnp.ndarray
    population_log_re_kpc_axis: jnp.ndarray
    population_n_axis: jnp.ndarray
    population_sigma_unit_grid: jnp.ndarray
    scalar_context: jnp.ndarray


@dataclass(frozen=True)
class SonnenfeldModelContext:
    """
    Parameter-independent arrays consumed by the Sonnenfeld model.

    The context contains no sampled hyper-parameters.  It stores canonical data
    arrays, deterministic quadrature tables, shifted h-unit constants, and
    normalization controls so JAX hooks can remain pure functions of
    ``theta_parts``, one random-basis row, and this context.
    """

    z_grid: np.ndarray
    chi_kpc_grid: np.ndarray
    cs_theta_e_axis: np.ndarray
    cs_gamma_axis: np.ndarray
    cs_cross_section_grid: np.ndarray
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
    log_mstar_obs: np.ndarray
    log_mstar_err: np.ndarray
    log_re_obs: np.ndarray
    n_obs: np.ndarray
    theta_e_obs: np.ndarray
    mstar_grid: np.ndarray
    parent_mstar_density_grid: np.ndarray
    size_density_grid: np.ndarray
    delta_r_grid: np.ndarray
    mstar_shift_grid: np.ndarray
    base_normals: np.ndarray
    population_gamma_axis: np.ndarray
    population_zd_axis: np.ndarray
    population_log_re_kpc_axis: np.ndarray
    population_n_axis: np.ndarray
    population_sigma_unit_grid: np.ndarray
    mass_radius_kpc: float
    mass_log_physical_offset: float
    mstar_pivot: float
    mbar: float
    parent_alpha: float
    truncation_mass_scatter: float
    size_mu0: float
    size_mu1: float
    size_sigma: float
    size_mu2: float
    n_fixed: float
    use_sersic_index: int
    gamma_trunc_low: float
    gamma_trunc_high: float
    parent_zd_min: float
    parent_zd_max: float
    parent_mstar_min: float
    parent_mstar_max: float
    sigma_proxy_fractional_scatter: float
    normalization_min_value: float


__all__ = ["SonnenfeldJaxContext", "SonnenfeldModelContext"]
