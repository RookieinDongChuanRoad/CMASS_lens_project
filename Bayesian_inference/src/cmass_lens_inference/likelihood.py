"""
Likelihood evaluation for individual lenses.

The code in this module mirrors the structure of the requirements document:
- interpolate the 17-point observational tracks onto a 200-point gamma grid
- integrate over `(gamma, m*)`
- keep profile-specific behavior outside this module as much as possible
"""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from numba import njit, prange

from .distributions import (
    discovery_probability,
    normal_pdf,
    skew_normal_pdf,
    source_redshift_pdf,
)
from .interpolation import clipped_linear_interp
from .types import HyperParams, ObservationRecord, PreparedObservation, ProfileSpec


SQRT_TWO = math.sqrt(2.0)
SQRT_TWO_PI = math.sqrt(2.0 * math.pi)


@njit(cache=True)
def _clipped_linear_interp_scalar_numba(x: np.ndarray, y: np.ndarray, x_new: float) -> float:
    """Numba-compatible scalar linear interpolation with boundary clipping."""

    if x_new <= x[0]:
        return float(y[0])
    if x_new >= x[-1]:
        return float(y[-1])
    left_index = 0
    right_index = x.shape[0] - 1
    while right_index - left_index > 1:
        middle_index = (left_index + right_index) // 2
        if x[middle_index] <= x_new:
            left_index = middle_index
        else:
            right_index = middle_index
    fraction = (x_new - x[left_index]) / (x[right_index] - x[left_index])
    return float(y[left_index] + fraction * (y[right_index] - y[left_index]))


@njit(cache=True)
def _normal_pdf_scalar_numba(x: float, mean: float, sigma: float) -> float:
    """Scalar Gaussian density for numba kernels."""

    if sigma <= 0.0:
        return 0.0
    normalized = (x - mean) / sigma
    return math.exp(-0.5 * normalized * normalized) / (sigma * SQRT_TWO_PI)


@njit(cache=True)
def _skew_normal_pdf_scalar_numba(x: float, loc: float, scale: float, alpha: float) -> float:
    """Scalar skew-normal density mirroring the public Python helper."""

    z_value = (x - loc) / scale
    phi = math.exp(-0.5 * z_value * z_value) / SQRT_TWO_PI
    cdf_term = 0.5 * (1.0 + math.erf(alpha * z_value / SQRT_TWO))
    return 2.0 / scale * phi * cdf_term


@njit(cache=True)
def _discovery_probability_scalar_numba(theta_ein: float, theta0: float, loga: float) -> float:
    """Scalar sigmoid detection probability for numba kernels."""

    slope = 10.0**loga
    return 1.0 / (1.0 + math.exp(-slope * (theta_ein - theta0)))


@njit(cache=True)
def _comoving_distance_mpc_numba(z_value: float, z_table: np.ndarray, comoving_distance_table_mpc: np.ndarray) -> float:
    """Interpolate the comoving-distance table inside numba kernels."""

    return _clipped_linear_interp_scalar_numba(z_table, comoving_distance_table_mpc, z_value)


@njit(cache=True)
def _angular_diameter_distance_mpc_numba(
    z_value: float,
    z_table: np.ndarray,
    comoving_distance_table_mpc: np.ndarray,
) -> float:
    """Observer-to-redshift angular-diameter distance inside numba kernels."""

    return _comoving_distance_mpc_numba(z_value, z_table, comoving_distance_table_mpc) / (1.0 + z_value)


@njit(cache=True)
def _angular_diameter_distance_between_mpc_numba(
    z_d: float,
    z_s: float,
    z_table: np.ndarray,
    comoving_distance_table_mpc: np.ndarray,
) -> float:
    """Deflector-to-source angular-diameter distance inside numba kernels."""

    if z_s <= z_d:
        return 0.0
    chi_d = _comoving_distance_mpc_numba(z_d, z_table, comoving_distance_table_mpc)
    chi_s = _comoving_distance_mpc_numba(z_s, z_table, comoving_distance_table_mpc)
    return (chi_s - chi_d) / (1.0 + z_s)


@njit(cache=True)
def _theta_ein_from_m5_gamma_numba(
    z_d: float,
    z_s: float,
    m5: float,
    gamma: float,
    z_table: np.ndarray,
    comoving_distance_table_mpc: np.ndarray,
    speed_of_light_km_s: float,
    gravitational_constant: float,
) -> float:
    """Einstein radius calculation expressed in numba-friendly primitives."""

    if z_d >= z_s:
        return 0.0

    dl_kpc = _angular_diameter_distance_mpc_numba(z_d, z_table, comoving_distance_table_mpc) * 1000.0
    ds_kpc = _angular_diameter_distance_mpc_numba(z_s, z_table, comoving_distance_table_mpc) * 1000.0
    dls_kpc = _angular_diameter_distance_between_mpc_numba(
        z_d,
        z_s,
        z_table,
        comoving_distance_table_mpc,
    ) * 1000.0
    if dl_kpc <= 0.0 or ds_kpc <= 0.0 or dls_kpc <= 0.0:
        return 0.0

    sigma_critical = (
        speed_of_light_km_s * speed_of_light_km_s / (4.0 * math.pi * gravitational_constant)
    ) * (ds_kpc / (dl_kpc * dls_kpc))
    r_ein_kpc = (10.0**m5 / (math.pi * sigma_critical * 5.0 ** (3.0 - gamma))) ** (1.0 / (gamma - 1.0))
    return r_ein_kpc / dl_kpc * 206265.0


@njit(cache=True)
def _trapz_uniform_numba(values: np.ndarray, start: float, step: float) -> float:
    """Trapezoid integration on a uniformly spaced grid without Python overhead."""

    if values.shape[0] == 1:
        return values[0]
    total = 0.5 * (values[0] + values[-1])
    for index in range(1, values.shape[0] - 1):
        total += values[index]
    return total * step


@njit(cache=True, parallel=True)
def _single_lens_likelihood_numba(
    hyper_params_vector: np.ndarray,
    z_d: float,
    z_s: float,
    log_stellar_mass_obs: float,
    log_stellar_mass_err: float,
    n_observed: float,
    num_sigma: int,
    sigma_observed: np.ndarray,
    sigma_error: np.ndarray,
    observed_log_effective_radius_kpc: float,
    gamma_dense: np.ndarray,
    m5_dense: np.ndarray,
    jacobian_dense: np.ndarray,
    s2_dense: np.ndarray,
    has_s2_dense: int,
    cross_section_gamma_grid: np.ndarray,
    cross_section_values: np.ndarray,
    mass_function_loc: float,
    mass_function_scale: float,
    mass_function_alpha: float,
    mu_r0: float,
    beta_r: float,
    sigma_r: float,
    nu_r: float,
    has_nu_r: int,
    z_table: np.ndarray,
    comoving_distance_table_mpc: np.ndarray,
    speed_of_light_km_s: float,
    gravitational_constant: float,
    mstar_points: int,
) -> float:
    """
    Evaluate the full single-lens 2D quadrature inside a numba kernel.

    This is the main hotspot for real inference. Keeping it in one kernel
    removes most Python overhead from the nested `(gamma, m*)` loops.
    """

    mu5_0 = hyper_params_vector[0]
    beta5 = hyper_params_vector[1]
    xi5 = hyper_params_vector[2]
    sigma5 = hyper_params_vector[3]
    mu_gamma_0 = hyper_params_vector[4]
    beta_gamma = hyper_params_vector[5]
    xi_gamma = hyper_params_vector[6]
    sigma_gamma = hyper_params_vector[7]
    mu_zs = hyper_params_vector[8]
    sigma_zs = hyper_params_vector[9]
    theta0 = hyper_params_vector[10]
    loga = hyper_params_vector[11]

    m_star_start = log_stellar_mass_obs - 5.0 * log_stellar_mass_err
    m_star_stop = log_stellar_mass_obs + 5.0 * log_stellar_mass_err
    if mstar_points <= 1:
        m_star_step = 1.0
    else:
        m_star_step = (m_star_stop - m_star_start) / (mstar_points - 1)
    gamma_step = 1.0 if gamma_dense.shape[0] <= 1 else gamma_dense[1] - gamma_dense[0]

    zd_term = _normal_pdf_scalar_numba(z_d, 0.558, 0.085)
    source_term = _normal_pdf_scalar_numba(z_s, mu_zs, sigma_zs) if z_s >= 0.0 else 0.0
    gamma_integrand = np.empty(gamma_dense.shape[0], dtype=np.float64)

    for gamma_index in prange(gamma_dense.shape[0]):
        gamma_value = gamma_dense[gamma_index]
        theta_ein = _theta_ein_from_m5_gamma_numba(
            z_d,
            z_s,
            m5_dense[gamma_index],
            gamma_value,
            z_table,
            comoving_distance_table_mpc,
            speed_of_light_km_s,
            gravitational_constant,
        )
        cs_factor = _clipped_linear_interp_scalar_numba(
            cross_section_gamma_grid,
            cross_section_values,
            gamma_value,
        )
        cross_section = math.pi * (cs_factor * theta_ein) ** 2
        find_term = _discovery_probability_scalar_numba(theta_ein, theta0, loga)

        mstar_integrand = np.empty(mstar_points, dtype=np.float64)
        for mstar_index in range(mstar_points):
            m_star = m_star_start + mstar_index * m_star_step
            mu_r_value = mu_r0 + beta_r * (m_star - 11.4)
            if has_nu_r == 1:
                mu_r_value = mu_r_value + nu_r * (math.log10(max(n_observed, 1.0e-8)) - math.log10(4.0))
            delta_r = observed_log_effective_radius_kpc - mu_r_value

            mu5_value = mu5_0 + beta5 * (m_star - 11.4) + xi5 * delta_r
            mu_gamma_value = mu_gamma_0 + beta_gamma * (m_star - 11.4) + xi_gamma * delta_r
            sigma_term = 1.0
            if num_sigma > 0 and has_s2_dense == 1:
                sigma_model = math.sqrt(max(s2_dense[gamma_index], 0.0) * 10.0 ** m5_dense[gamma_index])
                for sigma_index in range(sigma_observed.shape[0]):
                    sigma_term *= _normal_pdf_scalar_numba(
                        sigma_observed[sigma_index],
                        sigma_model,
                        sigma_error[sigma_index],
                    )

            mstar_integrand[mstar_index] = (
                zd_term
                * _normal_pdf_scalar_numba(m_star, log_stellar_mass_obs, log_stellar_mass_err)
                * _skew_normal_pdf_scalar_numba(m_star, mass_function_loc, mass_function_scale, mass_function_alpha)
                * _normal_pdf_scalar_numba(observed_log_effective_radius_kpc, mu_r_value, sigma_r)
                * _normal_pdf_scalar_numba(m5_dense[gamma_index], mu5_value, sigma5)
                * _normal_pdf_scalar_numba(gamma_value, mu_gamma_value, sigma_gamma)
                * source_term
                * find_term
                * cross_section
                * jacobian_dense[gamma_index]
                * sigma_term
            )

        gamma_integrand[gamma_index] = _trapz_uniform_numba(mstar_integrand, m_star_start, m_star_step)

    likelihood_value = _trapz_uniform_numba(gamma_integrand, gamma_dense[0], gamma_step)
    return max(likelihood_value, 1.0e-300)


def _effective_radius_log10_kpc(observation: ObservationRecord, cosmology) -> float:
    """Convert the observed effective radius from arcsec to log10(kpc)."""

    radius_kpc = observation.effective_radius_arcsec * cosmology.kpc_per_arcsec(observation.z_d)
    return math.log10(max(radius_kpc, 1.0e-8))


def _mu_r(profile_spec: ProfileSpec, m_star: np.ndarray, n_value: float) -> np.ndarray:
    """Return the mean size relation `mu_R(m*, n)` for the active profile."""

    mean = profile_spec.mu_r0 + profile_spec.beta_r * (m_star - 11.4)
    if profile_spec.nu_r is not None:
        mean = mean + profile_spec.nu_r * (math.log10(max(n_value, 1.0e-8)) - math.log10(4.0))
    return mean


def prepare_observation_for_inference(
    observation: ObservationRecord,
    cosmology,
    gamma_points: int,
) -> PreparedObservation:
    """
    Precompute all per-lens arrays that do not depend on the hyper-parameters.

    Doing this once up front avoids repeating interpolation and radius
    conversion work for every single `log_prob` call.
    """

    gamma_dense = np.linspace(
        float(observation.gamma_grid_17.min()),
        float(observation.gamma_grid_17.max()),
        gamma_points,
    )
    m5_dense = clipped_linear_interp(observation.gamma_grid_17, observation.m5_grid_17, gamma_dense)
    jacobian_dense = np.abs(
        clipped_linear_interp(observation.gamma_grid_17, observation.dm5_dthetaein_grid_17, gamma_dense)
    )
    s2_dense = (
        clipped_linear_interp(observation.gamma_grid_17, observation.s2_grid_17, gamma_dense)
        if observation.s2_grid_17 is not None
        else None
    )
    return PreparedObservation(
        lens_id=observation.lens_id,
        z_d=observation.z_d,
        z_s=observation.z_s,
        log_stellar_mass_obs=observation.log_stellar_mass_obs,
        log_stellar_mass_err=observation.log_stellar_mass_err,
        n_observed=observation.n_observed,
        effective_radius_arcsec=observation.effective_radius_arcsec,
        einstein_radius_arcsec=observation.einstein_radius_arcsec,
        num_sigma=observation.num_sigma,
        sigma_observed=observation.sigma_observed,
        sigma_error=observation.sigma_error,
        gamma_dense=gamma_dense,
        m5_dense=m5_dense,
        jacobian_dense=jacobian_dense,
        s2_dense=s2_dense,
        observed_log_effective_radius_kpc=_effective_radius_log10_kpc(observation, cosmology),
    )


def single_lens_likelihood(
    hyper_params: HyperParams,
    observation: PreparedObservation,
    profile_spec: ProfileSpec,
    cosmology,
    cross_section_grid,
    mstar_points: int,
) -> float:
    """
    Compute the single-lens likelihood via 2D quadrature.

    The implementation favors clarity over aggressive optimization because the
    project is currently at the structure-building stage. The interfaces are
    written so a later numba optimization pass can target this module without
    changing callers.
    """

    s2_dense = observation.s2_dense if observation.s2_dense is not None else np.zeros(1, dtype=float)
    return float(
        _single_lens_likelihood_numba(
            hyper_params.to_array(),
            observation.z_d,
            observation.z_s,
            observation.log_stellar_mass_obs,
            max(observation.log_stellar_mass_err, 1.0e-6),
            observation.n_observed,
            observation.num_sigma,
            observation.sigma_observed,
            observation.sigma_error,
            observation.observed_log_effective_radius_kpc,
            observation.gamma_dense,
            observation.m5_dense,
            observation.jacobian_dense,
            s2_dense,
            1 if observation.s2_dense is not None else 0,
            cross_section_grid.gamma_grid,
            cross_section_grid.cs_over_theta_ein,
            profile_spec.mass_function_loc,
            profile_spec.mass_function_scale,
            profile_spec.mass_function_alpha,
            profile_spec.mu_r0,
            profile_spec.beta_r,
            profile_spec.sigma_r,
            profile_spec.nu_r if profile_spec.nu_r is not None else 0.0,
            1 if profile_spec.nu_r is not None else 0,
            cosmology.z_table,
            cosmology.comoving_distance_table_mpc,
            cosmology.speed_of_light_km_s,
            cosmology.gravitational_constant,
            mstar_points,
        )
    )
