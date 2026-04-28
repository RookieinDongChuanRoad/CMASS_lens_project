"""
Monolithic all-lens likelihood kernel.

The production model evaluates the complete sample likelihood in one compiled
kernel rather than dispatching one lens at a time from Python. This is the key
structural change required to match the reference implementation's throughput.
"""

from __future__ import annotations

import math

import numpy as np
import numba as nb

from .primitives import (
    gamma_population_mean,
    normal_pdf,
    p_find,
    theta_dimension_for_gamma_mode,
    theta_ein_arcsec,
    truncated_normal_pdf_nonneg,
    unpack_model_theta,
)


@nb.njit(cache=True, parallel=True, fastmath=True)
def log_likelihood_lenses_numba(
    theta: np.ndarray,
    z_grid: np.ndarray,
    chi_kpc_grid: np.ndarray,
    cs_over_theta_int: np.ndarray,
    mass_grid_int: np.ndarray,
    dmass_dthetaein_grid_int: np.ndarray,
    s2_grid_int: np.ndarray,
    has_s2: np.ndarray,
    num_sigma: np.ndarray,
    sigma_obs: np.ndarray,
    sigma_err: np.ndarray,
    zd: np.ndarray,
    zs: np.ndarray,
    p_zd_fixed: np.ndarray,
    mstar_grid: np.ndarray,
    mstar_shift11p4: np.ndarray,
    sigma_star_shift9p0_grid: np.ndarray,
    mstar_integrand_base: np.ndarray,
    delta_r_grid: np.ndarray,
    gamma_grid_int: np.ndarray,
    mass_radius_kpc: float,
    gamma_mode_code: int,
    mass_log_physical_offset: float = 0.0,
) -> float:
    """
    Evaluate the full sample log-likelihood in one `numba` kernel.

    The compiled context precomputes the parameter-independent pieces of the
    inner `m*` integrand once. This kernel then rebuilds the two-stage discrete
    quadrature explicitly so the code mirrors the mathematical likelihood:
    first integrate over `m*`, then integrate those results over `gamma`.
    """

    if theta.shape[0] != theta_dimension_for_gamma_mode(gamma_mode_code):
        return -np.inf

    (
        mu5_0,
        beta5,
        xi5,
        sigma5,
        mu_gamma_0,
        beta_gamma,
        xi_gamma,
        beta_sigma_star_gamma,
        sigma_gamma,
        mu_zs,
        sigma_zs,
        theta0,
        loga,
    ) = unpack_model_theta(theta, gamma_mode_code)

    if sigma5 <= 0.0 or sigma_gamma <= 0.0 or sigma_zs <= 0.0:
        return -np.inf

    n_lens = zd.shape[0]
    n_gamma = gamma_grid_int.shape[0]
    n_mstar = mstar_integrand_base.shape[1]
    ll_terms = np.zeros(n_lens, dtype=np.float64)
    valid = np.ones(n_lens, dtype=np.int64)

    for i in nb.prange(n_lens):
        p_zd = p_zd_fixed[i]
        if p_zd <= 0.0:
            valid[i] = 0
            continue

        p_zs = truncated_normal_pdf_nonneg(zs[i], mu_zs, sigma_zs)
        if p_zs <= 0.0:
            valid[i] = 0
            continue

        gamma_integrand = np.zeros(n_gamma, dtype=np.float64)
        for kg in range(n_gamma):
            gamma = gamma_grid_int[kg]
            log_enclosed_mass = mass_grid_int[i, kg]
            jac = abs(dmass_dthetaein_grid_int[i, kg])
            if jac <= 0.0:
                continue

            theta_e = theta_ein_arcsec(
                zd[i],
                zs[i],
                log_enclosed_mass,
                gamma,
                z_grid,
                chi_kpc_grid,
                mass_radius_kpc,
                mass_log_physical_offset,
            )
            if theta_e <= 0.0:
                continue

            area = math.pi * (cs_over_theta_int[kg] * theta_e) ** 2
            pf = p_find(theta_e, theta0, loga)
            if area <= 0.0 or pf <= 0.0:
                continue

            p_sigma = 1.0
            if num_sigma[i] > 0:
                if has_s2[i] == 0:
                    p_sigma = 0.0
                else:
                    sigma_model = math.sqrt(max(s2_grid_int[i, kg] * (10.0**log_enclosed_mass), 1.0e-30))
                    for ks in range(num_sigma[i]):
                        p_sigma *= normal_pdf(sigma_obs[i, ks], sigma_model, sigma_err[i, ks])
            if p_sigma <= 0.0:
                continue

            mstar_integrand = np.zeros(n_mstar, dtype=np.float64)
            for km in range(n_mstar):
                fixed_base = mstar_integrand_base[i, km]
                if fixed_base <= 0.0:
                    continue
                mu5 = mu5_0 + beta5 * mstar_shift11p4[i, km] + xi5 * delta_r_grid[i, km]
                mu_gamma = gamma_population_mean(
                    mu_gamma_0,
                    beta_gamma,
                    xi_gamma,
                    beta_sigma_star_gamma,
                    mstar_shift11p4[i, km],
                    delta_r_grid[i, km],
                    sigma_star_shift9p0_grid[i, km],
                    gamma_mode_code,
                )
                mstar_integrand[km] = (
                    fixed_base
                    * normal_pdf(log_enclosed_mass, mu5, sigma5)
                    * normal_pdf(gamma, mu_gamma, sigma_gamma)
                )

            integrated_mstar = np.trapezoid(mstar_integrand, mstar_grid[i])
            gamma_integrand[kg] = integrated_mstar * p_zd * p_zs * pf * area * jac * p_sigma

        int_gamma = np.trapezoid(gamma_integrand, gamma_grid_int)

        if int_gamma <= 0.0:
            valid[i] = 0
            continue

        ll_terms[i] = math.log(int_gamma)

    total = 0.0
    for i in range(n_lens):
        if valid[i] == 0:
            return -np.inf
        total += ll_terms[i]
    return total
