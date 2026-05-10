"""Array context for the Sonnenfeld 2024 SLACS model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SonnenfeldModelContext:
    """
    Parameter-independent arrays consumed by the Sonnenfeld model.

    The context contains no sampled hyper-parameters.  It stores canonical data
    arrays, deterministic quadrature tables, shifted h-unit constants, and
    normalization controls so backend kernels can remain pure functions of one
    random-basis row and this context.
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
    parent_sample_zd: np.ndarray
    parent_sample_mstar: np.ndarray
    parent_sample_log_re: np.ndarray
    parent_sample_delta_r: np.ndarray
    population_gamma_axis: np.ndarray
    population_zd_axis: np.ndarray
    population_log_re_kpc_axis: np.ndarray
    population_n_axis: np.ndarray
    population_sigma_unit_grid: np.ndarray
    fp_enabled: int
    fp_fit_mstar_min: float
    fp_pivot_mstar: float
    fp_fiducial_scatter: float
    fp_scatter_error: float
    fp_mu_v_prior: float
    fp_mu_v_error: float
    fp_beta_v_prior: float
    fp_beta_v_error: float
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
    source_z_min: float
    source_z_max: float
    source_lens_redshift_gap: float
    sigma_proxy_fractional_scatter: float
    normalization_min_value: float


__all__ = ["SonnenfeldModelContext"]
