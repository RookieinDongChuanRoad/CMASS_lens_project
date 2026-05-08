"""Sonnenfeld 2024 SLACS posterior assembly and model-owned Numba kernels.

This module is the complete Sonnenfeld posterior implementation.  The private
fused kernels live beside the posterior structure because their theta order,
selection correction, and likelihood reduction are specific to this model.
Reusable numerical primitives remain in ``numba_backend.kernels``.
"""

from __future__ import annotations

import math
from time import perf_counter

import numba as nb
import numpy as np

from ...numba_backend.diagnostics import build_reject_result, build_timing_blob
from ...numba_backend.kernels.distributions import normal_pdf, truncated_normal_pdf, truncnorm_sample
from ...numba_backend.kernels.integration import trapezoid_1d
from ...numba_backend.kernels.interpolation import interp_sigma_unit_clip
from ...numba_backend.kernels.lensing import theta_ein_arcsec
from ...numba_backend.kernels.selection import theta_e_est_from_sigma_proxy
from ...numba_backend.kernels.selection_likelihood import (
    cross_section_find_weight,
    gaussian_source_redshift_density,
    observed_sigma_likelihood,
    sigma_model_from_s2,
)
from ...types import CompiledModel
from .paper_constants import TRUNCATION_MASS_POLYNOMIAL_COEFFICIENTS


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
        if theta_e <= 0.0 or sigma_unit <= 0.0:
            continue
        sigma_model = sigma_model_from_s2(sigma_unit, log_enclosed_mass)
        sigma_proxy = sigma_model * max(0.1, 1.0 + sigma_proxy_fractional_scatter * nrm[7])
        theta_est = theta_e_est_from_sigma_proxy(sigma_proxy, zd, zs, z_grid, chi_kpc_grid)
        selection_weight = cross_section_find_weight(
            theta_e,
            gamma,
            theta_est,
            theta0,
            loga,
            cs_theta_e_axis,
            cs_gamma_axis,
            cs_cross_section_grid,
        )
        if selection_weight <= 0.0:
            continue
        total += selection_weight * parent_density / proposal_density

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
        p_zs = gaussian_source_redshift_density(zs[lens_index], mu_zs, sigma_zs)
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
            sigma_model = sigma_model_from_s2(s2_grid_int[lens_index, gamma_index], log_enclosed_mass)
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
            selection_weight = cross_section_find_weight(
                theta_e,
                gamma,
                theta_est,
                theta0,
                loga,
                cs_theta_e_axis,
                cs_gamma_axis,
                cs_cross_section_grid,
            )
            sigma_probability = observed_sigma_likelihood(
                lens_index,
                num_sigma,
                has_s2,
                sigma_obs,
                sigma_err,
                sigma_model,
            )
            if selection_weight <= 0.0 or sigma_probability <= 0.0:
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
                * selection_weight
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


def log_prob(theta: np.ndarray, compiled_model: CompiledModel, total_start: float) -> tuple[float, np.void]:
    """Evaluate the Sonnenfeld posterior using Sonnenfeld-specific kernels."""

    context = compiled_model.context
    normalization_start = perf_counter()
    z_norm = normalization_mc_numba(
        theta=theta,
        base_normals=context.base_normals,
        z_grid=context.z_grid,
        chi_kpc_grid=context.chi_kpc_grid,
        cs_theta_e_axis=context.cs_theta_e_axis,
        cs_gamma_axis=context.cs_gamma_axis,
        cs_cross_section_grid=context.cs_cross_section_grid,
        population_gamma_axis=context.population_gamma_axis,
        population_zd_axis=context.population_zd_axis,
        population_log_re_kpc_axis=context.population_log_re_kpc_axis,
        population_n_axis=context.population_n_axis,
        population_sigma_unit_grid=context.population_sigma_unit_grid,
        mass_radius_kpc=context.mass_radius_kpc,
        mass_log_physical_offset=context.mass_log_physical_offset,
        mstar_pivot=context.mstar_pivot,
        mbar=context.mbar,
        parent_alpha=context.parent_alpha,
        truncation_mass_scatter=context.truncation_mass_scatter,
        truncation_coefficients=TRUNCATION_MASS_POLYNOMIAL_COEFFICIENTS,
        size_mu0=context.size_mu0,
        size_mu1=context.size_mu1,
        size_sigma=context.size_sigma,
        size_mu2=context.size_mu2,
        n_fixed=context.n_fixed,
        use_sersic_index=context.use_sersic_index,
        gamma_trunc_low=context.gamma_trunc_low,
        gamma_trunc_high=context.gamma_trunc_high,
        parent_zd_min=context.parent_zd_min,
        parent_zd_max=context.parent_zd_max,
        parent_mstar_min=context.parent_mstar_min,
        parent_mstar_max=context.parent_mstar_max,
        sigma_proxy_fractional_scatter=context.sigma_proxy_fractional_scatter,
    )
    normalization_seconds = perf_counter() - normalization_start
    if (not np.isfinite(z_norm)) or z_norm <= context.normalization_min_value:
        return build_reject_result(total_start, compiled_model, "sonnenfeld2024_slacs")

    likelihood_start = perf_counter()
    likelihood_value = log_likelihood_lenses_numba(
        theta=theta,
        z_grid=context.z_grid,
        chi_kpc_grid=context.chi_kpc_grid,
        cs_theta_e_axis=context.cs_theta_e_axis,
        cs_gamma_axis=context.cs_gamma_axis,
        cs_cross_section_grid=context.cs_cross_section_grid,
        gamma_grid_int=context.gamma_grid_int,
        mass_grid_int=context.mass_grid_int,
        dmass_dthetaein_grid_int=context.dmass_dthetaein_grid_int,
        s2_grid_int=context.s2_grid_int,
        has_s2=context.has_s2,
        num_sigma=context.num_sigma,
        sigma_obs=context.sigma_obs,
        sigma_err=context.sigma_err,
        zd=context.zd,
        zs=context.zs,
        parent_mstar_density_grid=context.parent_mstar_density_grid,
        size_density_grid=context.size_density_grid,
        delta_r_grid=context.delta_r_grid,
        mstar_shift_grid=context.mstar_shift_grid,
        mstar_grid=context.mstar_grid,
        mass_radius_kpc=context.mass_radius_kpc,
        mass_log_physical_offset=context.mass_log_physical_offset,
    )
    likelihood_seconds = perf_counter() - likelihood_start
    total_seconds = perf_counter() - total_start
    blob = build_timing_blob(
        total_log_prob_seconds=total_seconds,
        likelihood_seconds=likelihood_seconds,
        normalization_seconds=normalization_seconds,
        fp_prior_seconds=0.0,
        normalization_value=float(z_norm),
        fp_prior_log_term=0.0,
        fpfit_mu=math.nan,
        fpfit_beta=math.nan,
        fpfit_xi=math.nan,
        fpfit_scatter=math.nan,
        kernel="sonnenfeld2024_slacs",
        parallel_strategy=compiled_model.parallelism.strategy,
    )
    if not np.isfinite(likelihood_value):
        return -np.inf, blob
    return float(likelihood_value - context.zd.shape[0] * math.log(z_norm)), blob


__all__ = ["log_prob"]
