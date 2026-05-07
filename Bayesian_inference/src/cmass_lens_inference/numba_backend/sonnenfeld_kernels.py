"""
Sonnenfeld 2024 SLACS production kernels for the Numba backend.

The implementation mirrors the current Sonnenfeld model semantics in the
runtime components: parent-population density, quadratic size relation,
finite-fibre `theta_E x gamma` cross-section, velocity-dispersion proxy
selection, and ordinary Gaussian effective source-redshift density.
"""

from __future__ import annotations

import math

import numba as nb
import numpy as np

from .primitives import (
    C_KM_S,
    interp_cross_section_theta_gamma,
    interp_sigma_unit_clip,
    normal_pdf,
    p_find,
    theta_ein_arcsec,
    trapezoid_1d,
    truncated_normal_pdf,
    truncnorm_sample,
)


@nb.njit(cache=True, inline="always")
def _unpack_theta(theta: np.ndarray) -> tuple[float, ...]:
    """Unpack the fixed 12D Sonnenfeld parameter vector."""

    return (
        theta[0],
        theta[1],
        theta[2],
        theta[3],
        theta[4],
        theta[5],
        theta[6],
        theta[7],
        theta[8],
        theta[9],
        theta[10],
        theta[11],
    )


@nb.njit(cache=True)
def theta_e_est_from_sigma_proxy(
    sigma_proxy: float,
    zd: float,
    zs: float,
    z_grid: np.ndarray,
    chi_kpc_grid: np.ndarray,
) -> float:
    """Convert the velocity-dispersion proxy into an estimated Einstein radius."""

    if sigma_proxy <= 0.0 or zd <= 0.0 or zs <= zd:
        return 0.0
    chi_l = np.interp(zd, z_grid, chi_kpc_grid)
    chi_s = np.interp(zs, z_grid, chi_kpc_grid)
    ds = chi_s / (1.0 + zs)
    dls = (chi_s - chi_l) / (1.0 + zs)
    if ds <= 0.0 or dls <= 0.0:
        return 0.0
    return 4.0 * math.pi * (sigma_proxy / C_KM_S) ** 2 * dls / ds * 206265.0


@nb.njit(cache=True)
def _active_truncation_mass_threshold(
    zd: float,
    mstar_pivot: float,
    coefficients: np.ndarray,
) -> float:
    """
    Evaluate the paper Table-1 truncation polynomial in the active coordinate.

    Runtime preprocessing shifts `mstar_pivot` for h-units.  The difference
    relative to the paper physical pivot is therefore the location offset that
    must be applied to the polynomial threshold.
    """

    physical_threshold = 0.0
    power = 1.0
    for index in range(coefficients.shape[0]):
        physical_threshold += coefficients[index] * power
        power *= zd
    return physical_threshold + (mstar_pivot - 11.3)


@nb.njit(cache=True)
def _parent_density_for_draw(
    zd: float,
    mstar: float,
    mstar_pivot: float,
    mbar: float,
    parent_alpha: float,
    truncation_mass_scatter: float,
    truncation_coefficients: np.ndarray,
) -> float:
    """Return the unnormalized Sonnenfeld Table-1 parent density."""

    threshold = _active_truncation_mass_threshold(zd, mstar_pivot, truncation_coefficients)
    completeness = math.atan((mstar - threshold) / truncation_mass_scatter) / math.pi + 0.5
    schechter_mass = 10.0 ** (mstar - mbar)
    schechter = 10.0 ** ((mstar - mbar) * (parent_alpha + 1.0))
    return max(zd, 1.0e-6) ** 2 * completeness * schechter * math.exp(-schechter_mass)


@nb.njit(cache=True, inline="always")
def _size_relation_mean(
    mstar: float,
    size_mu0: float,
    size_mu1: float,
    size_mu2: float,
) -> float:
    """Return Sonnenfeld Equation 29 in the active coordinate."""

    return size_mu0 + size_mu1 * mstar + size_mu2 * mstar * mstar


@nb.njit(cache=True, inline="always")
def _draw_population_state(
    theta: np.ndarray,
    nrm: np.ndarray,
    mstar_pivot: float,
    mbar: float,
    parent_alpha: float,
    truncation_mass_scatter: float,
    truncation_coefficients: np.ndarray,
    size_mu0: float,
    size_mu1: float,
    size_sigma: float,
    size_mu2: float,
    n_fixed: float,
    use_sersic_index: int,
    gamma_trunc_low: float,
    gamma_trunc_high: float,
    parent_zd_min: float,
    parent_zd_max: float,
    parent_mstar_min: float,
    parent_mstar_max: float,
) -> tuple[float, float, float, float, float, float, float, float, float, float]:
    """
    Map one fixed normal row into a Sonnenfeld population draw.

    The draw distribution is an integration proposal for `(z_d, mstar)` and an
    exact Gaussian proposal for `z_s`.  The returned parent/proposal correction
    is applied by the normalization kernel.
    """

    (
        mu5_0,
        beta5,
        xi5,
        sigma5,
        mu_gamma_0,
        beta_gamma,
        xi_gamma,
        sigma_gamma,
        mu_zs,
        sigma_zs,
        _theta0,
        _loga,
    ) = _unpack_theta(theta)

    zd_center = 0.5 * (parent_zd_min + parent_zd_max)
    zd_scale = 0.25
    mstar_center = 0.5 * (parent_mstar_min + parent_mstar_max)
    mstar_scale = 0.35
    zd = truncnorm_sample(zd_center, zd_scale, parent_zd_min, parent_zd_max, nrm[0])
    mstar = truncnorm_sample(mstar_center, mstar_scale, parent_mstar_min, parent_mstar_max, nrm[1])
    zs = mu_zs + sigma_zs * nrm[2]
    n_value = n_fixed
    if use_sersic_index == 1:
        n_value = max(4.0 + 0.4 * nrm[3], 0.5)

    mu_r = _size_relation_mean(mstar, size_mu0, size_mu1, size_mu2)
    log_re = mu_r + size_sigma * nrm[4]
    delta_r = log_re - mu_r
    mstar_shift = mstar - mstar_pivot
    log_enclosed_mass = mu5_0 + beta5 * mstar_shift + xi5 * delta_r + sigma5 * nrm[5]
    mu_gamma = mu_gamma_0 + beta_gamma * mstar_shift + xi_gamma * delta_r
    gamma = truncnorm_sample(mu_gamma, sigma_gamma, gamma_trunc_low, gamma_trunc_high, nrm[6])

    parent_density = _parent_density_for_draw(
        zd,
        mstar,
        mstar_pivot,
        mbar,
        parent_alpha,
        truncation_mass_scatter,
        truncation_coefficients,
    )
    proposal_density = (
        truncated_normal_pdf(zd, zd_center, zd_scale, parent_zd_min, parent_zd_max)
        * truncated_normal_pdf(mstar, mstar_center, mstar_scale, parent_mstar_min, parent_mstar_max)
    )
    return (
        zd,
        zs,
        mstar,
        n_value,
        log_re,
        delta_r,
        log_enclosed_mass,
        gamma,
        parent_density,
        max(proposal_density, 1.0e-300),
    )


@nb.njit(cache=True, parallel=True, fastmath=True)
def normalization_mc_numba(
    theta: np.ndarray,
    base_normals: np.ndarray,
    z_grid: np.ndarray,
    chi_kpc_grid: np.ndarray,
    cs_theta_e_axis: np.ndarray,
    cs_gamma_axis: np.ndarray,
    cs_cross_section_grid: np.ndarray,
    population_gamma_axis: np.ndarray,
    population_zd_axis: np.ndarray,
    population_log_re_kpc_axis: np.ndarray,
    population_n_axis: np.ndarray,
    population_sigma_unit_grid: np.ndarray,
    mass_radius_kpc: float,
    mass_log_physical_offset: float,
    mstar_pivot: float,
    mbar: float,
    parent_alpha: float,
    truncation_mass_scatter: float,
    truncation_coefficients: np.ndarray,
    size_mu0: float,
    size_mu1: float,
    size_sigma: float,
    size_mu2: float,
    n_fixed: float,
    use_sersic_index: int,
    gamma_trunc_low: float,
    gamma_trunc_high: float,
    parent_zd_min: float,
    parent_zd_max: float,
    parent_mstar_min: float,
    parent_mstar_max: float,
    sigma_proxy_fractional_scatter: float,
) -> float:
    """Estimate the Sonnenfeld selection normalization."""

    if theta.shape[0] != 12:
        return 0.0
    (
        _mu5_0,
        _beta5,
        _xi5,
        sigma5,
        _mu_gamma_0,
        _beta_gamma,
        _xi_gamma,
        sigma_gamma,
        _mu_zs,
        sigma_zs,
        theta0,
        loga,
    ) = _unpack_theta(theta)
    if sigma5 <= 0.0 or sigma_gamma <= 0.0 or sigma_zs <= 0.0:
        return 0.0

    n_samples = base_normals.shape[0]
    total = 0.0
    for sample_index in nb.prange(n_samples):
        nrm = base_normals[sample_index]
        (
            zd,
            zs,
            mstar,
            n_value,
            log_re,
            _delta_r,
            log_enclosed_mass,
            gamma,
            parent_density,
            proposal_density,
        ) = _draw_population_state(
            theta,
            nrm,
            mstar_pivot,
            mbar,
            parent_alpha,
            truncation_mass_scatter,
            truncation_coefficients,
            size_mu0,
            size_mu1,
            size_sigma,
            size_mu2,
            n_fixed,
            use_sersic_index,
            gamma_trunc_low,
            gamma_trunc_high,
            parent_zd_min,
            parent_zd_max,
            parent_mstar_min,
            parent_mstar_max,
        )
        if zd <= 0.0 or zs <= zd or not math.isfinite(gamma):
            continue
        if mstar < parent_mstar_min or mstar > parent_mstar_max:
            continue

        theta_e = theta_ein_arcsec(
            zd,
            zs,
            log_enclosed_mass,
            gamma,
            z_grid,
            chi_kpc_grid,
            mass_radius_kpc,
            mass_log_physical_offset,
        )
        cross_section = interp_cross_section_theta_gamma(
            theta_e,
            gamma,
            cs_theta_e_axis,
            cs_gamma_axis,
            cs_cross_section_grid,
        )
        sigma_unit = interp_sigma_unit_clip(
            gamma,
            zd,
            log_re,
            n_value,
            population_gamma_axis,
            population_zd_axis,
            population_log_re_kpc_axis,
            population_n_axis,
            population_sigma_unit_grid,
            1,
        )
        if theta_e <= 0.0 or cross_section <= 0.0 or sigma_unit <= 0.0:
            continue
        sigma_model = math.sqrt(max(sigma_unit * 10.0**log_enclosed_mass, 1.0e-30))
        sigma_proxy = sigma_model * max(0.1, 1.0 + sigma_proxy_fractional_scatter * nrm[7])
        theta_est = theta_e_est_from_sigma_proxy(sigma_proxy, zd, zs, z_grid, chi_kpc_grid)
        find_probability = p_find(theta_est, theta0, loga)
        if theta_est <= 0.0 or find_probability <= 0.0:
            continue
        total += cross_section * find_probability * parent_density / proposal_density

    return total / n_samples


@nb.njit(cache=True, parallel=True, fastmath=True)
def log_likelihood_lenses_numba(
    theta: np.ndarray,
    z_grid: np.ndarray,
    chi_kpc_grid: np.ndarray,
    cs_theta_e_axis: np.ndarray,
    cs_gamma_axis: np.ndarray,
    cs_cross_section_grid: np.ndarray,
    gamma_grid_int: np.ndarray,
    mass_grid_int: np.ndarray,
    dmass_dthetaein_grid_int: np.ndarray,
    s2_grid_int: np.ndarray,
    has_s2: np.ndarray,
    num_sigma: np.ndarray,
    sigma_obs: np.ndarray,
    sigma_err: np.ndarray,
    zd: np.ndarray,
    zs: np.ndarray,
    parent_mstar_density_grid: np.ndarray,
    size_density_grid: np.ndarray,
    delta_r_grid: np.ndarray,
    mstar_shift_grid: np.ndarray,
    mstar_grid: np.ndarray,
    mass_radius_kpc: float,
    mass_log_physical_offset: float,
) -> float:
    """Evaluate all Sonnenfeld per-lens likelihood integrals."""

    if theta.shape[0] != 12:
        return -np.inf
    (
        mu5_0,
        beta5,
        xi5,
        sigma5,
        mu_gamma_0,
        beta_gamma,
        xi_gamma,
        sigma_gamma,
        mu_zs,
        sigma_zs,
        theta0,
        loga,
    ) = _unpack_theta(theta)
    if sigma5 <= 0.0 or sigma_gamma <= 0.0 or sigma_zs <= 0.0:
        return -np.inf

    n_lens = zd.shape[0]
    n_gamma = gamma_grid_int.shape[0]
    n_mstar = mstar_grid.shape[1]
    ll_terms = np.zeros(n_lens, dtype=np.float64)
    valid = np.ones(n_lens, dtype=np.int64)

    for lens_index in nb.prange(n_lens):
        p_zs = normal_pdf(zs[lens_index], mu_zs, sigma_zs)
        if p_zs <= 0.0:
            valid[lens_index] = 0
            continue

        gamma_integrand = np.zeros(n_gamma, dtype=np.float64)
        for gamma_index in range(n_gamma):
            gamma = gamma_grid_int[gamma_index]
            log_enclosed_mass = mass_grid_int[lens_index, gamma_index]
            jacobian = abs(dmass_dthetaein_grid_int[lens_index, gamma_index])
            if jacobian <= 0.0:
                continue

            theta_e = theta_ein_arcsec(
                zd[lens_index],
                zs[lens_index],
                log_enclosed_mass,
                gamma,
                z_grid,
                chi_kpc_grid,
                mass_radius_kpc,
                mass_log_physical_offset,
            )
            cross_section = interp_cross_section_theta_gamma(
                theta_e,
                gamma,
                cs_theta_e_axis,
                cs_gamma_axis,
                cs_cross_section_grid,
            )
            sigma_model = math.sqrt(max(s2_grid_int[lens_index, gamma_index] * 10.0**log_enclosed_mass, 1.0e-30))
            sigma_find_proxy = sigma_model
            if num_sigma[lens_index] >= 1:
                sigma_find_proxy = sigma_obs[lens_index, 0]
            theta_est = theta_e_est_from_sigma_proxy(
                sigma_find_proxy,
                zd[lens_index],
                zs[lens_index],
                z_grid,
                chi_kpc_grid,
            )
            find_probability = p_find(theta_est, theta0, loga)

            sigma_probability = 1.0
            if num_sigma[lens_index] > 0:
                if has_s2[lens_index] == 0:
                    sigma_probability = 0.0
                else:
                    for sigma_index in range(num_sigma[lens_index]):
                        sigma_probability *= normal_pdf(
                            sigma_obs[lens_index, sigma_index],
                            sigma_model,
                            sigma_err[lens_index, sigma_index],
                        )
            if (
                theta_e <= 0.0
                or cross_section <= 0.0
                or theta_est <= 0.0
                or find_probability <= 0.0
                or sigma_probability <= 0.0
            ):
                continue

            mstar_integrand = np.zeros(n_mstar, dtype=np.float64)
            for mstar_index in range(n_mstar):
                mu5 = (
                    mu5_0
                    + beta5 * mstar_shift_grid[lens_index, mstar_index]
                    + xi5 * delta_r_grid[lens_index, mstar_index]
                )
                mu_gamma = (
                    mu_gamma_0
                    + beta_gamma * mstar_shift_grid[lens_index, mstar_index]
                    + xi_gamma * delta_r_grid[lens_index, mstar_index]
                )
                mstar_integrand[mstar_index] = (
                    parent_mstar_density_grid[lens_index, mstar_index]
                    * size_density_grid[lens_index, mstar_index]
                    * normal_pdf(log_enclosed_mass, mu5, sigma5)
                    * normal_pdf(gamma, mu_gamma, sigma_gamma)
                )

            integrated_mstar = trapezoid_1d(mstar_integrand, mstar_grid[lens_index])
            gamma_integrand[gamma_index] = (
                integrated_mstar
                * p_zs
                * find_probability
                * cross_section
                * jacobian
                * sigma_probability
            )

        lens_integral = trapezoid_1d(gamma_integrand, gamma_grid_int)
        if lens_integral <= 0.0 or not math.isfinite(lens_integral):
            valid[lens_index] = 0
            continue
        ll_terms[lens_index] = math.log(lens_integral)

    total = 0.0
    for lens_index in range(n_lens):
        if valid[lens_index] == 0:
            return -np.inf
        total += ll_terms[lens_index]
    return total


__all__ = [
    "log_likelihood_lenses_numba",
    "normalization_mc_numba",
    "theta_e_est_from_sigma_proxy",
]
