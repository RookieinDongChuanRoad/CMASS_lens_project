"""
Monolithic Monte Carlo normalization kernel.

This file owns the heavy-weight selection-normalization computation used by
the production `log_prob` path. The implementation stays entirely inside
`numba` so every evaluation can reuse one compiled kernel and one fixed random
basis without falling back to SciPy helpers.
"""

from __future__ import annotations

import math

import numba as nb
import numpy as np

from .primitives import (
    interp1d_clip,
    mu_r,
    p_find,
    phi_standard,
    skewnorm_sample,
    theta_ein_arcsec,
    truncnorm_sample,
)


@nb.njit(cache=True, parallel=True, fastmath=True)
def normalization_mc_numba(
    theta: np.ndarray,
    base_normals: np.ndarray,
    cs_gamma_grid: np.ndarray,
    cs_over_theta: np.ndarray,
    z_grid: np.ndarray,
    chi_kpc_grid: np.ndarray,
    mu_d: float,
    sigma_d: float,
    mass_function_loc: float,
    mass_function_scale: float,
    mass_function_alpha: float,
    mu_r0: float,
    beta_r: float,
    sigma_r: float,
    nu_r: float,
    use_sersic_index: int,
    n_fixed: float,
    mu_n0: float,
    beta_n: float,
    sigma_n: float,
    gamma_trunc_low: float,
    gamma_trunc_high: float,
    mass_radius_kpc: float,
) -> float:
    """
    Estimate the selection normalization for one hyper-parameter vector.

    The kernel mirrors the reference implementation's structure:
    one loop over the fixed random basis, profile-specific draws injected by
    scalar constants, and a final average of discovery-weighted cross-sections.
    """

    mu5_0 = theta[0]
    beta5 = theta[1]
    xi5 = theta[2]
    sigma5 = theta[3]
    mu_gamma_0 = theta[4]
    beta_gamma = theta[5]
    xi_gamma = theta[6]
    sigma_gamma = theta[7]
    mu_zs = theta[8]
    sigma_zs = theta[9]
    theta0 = theta[10]
    loga = theta[11]

    if sigma5 <= 0.0 or sigma_gamma <= 0.0 or sigma_zs <= 0.0:
        return 0.0

    n_samples = base_normals.shape[0]
    z0 = (0.0 - mu_zs) / sigma_zs
    trunc_den = 1.0 - phi_standard(z0)
    if trunc_den <= 0.0:
        return 0.0
    acc = 0.0

    for i in nb.prange(n_samples):
        nrm = base_normals[i]
        zd = mu_d + sigma_d * nrm[0]
        zs = mu_zs + sigma_zs * nrm[1]
        if zd <= 0.0 or zs <= 0.0 or zs <= zd:
            continue

        mstar = skewnorm_sample(
            mass_function_loc,
            mass_function_scale,
            mass_function_alpha,
            nrm[2],
            nrm[3],
        )

        if use_sersic_index == 1:
            logn = mu_n0 + beta_n * (mstar - 11.4) + sigma_n * nrm[4]
            n_value = 10.0**logn
            mu_r_draw = mu_r(mstar, n_value, use_sersic_index, mu_r0, beta_r, nu_r)
            re_draw = mu_r_draw + sigma_r * nrm[5]
            delta_r = re_draw - mu_r_draw
            log_enclosed_mass = mu5_0 + beta5 * (mstar - 11.4) + xi5 * delta_r + sigma5 * nrm[6]
            mu_gamma = mu_gamma_0 + beta_gamma * (mstar - 11.4) + xi_gamma * delta_r
            gamma = truncnorm_sample(
                mu_gamma,
                sigma_gamma,
                gamma_trunc_low,
                gamma_trunc_high,
                nrm[7],
            )
        else:
            n_value = n_fixed
            mu_r_draw = mu_r(mstar, n_value, use_sersic_index, mu_r0, beta_r, nu_r)
            re_draw = mu_r_draw + sigma_r * nrm[4]
            delta_r = re_draw - mu_r_draw
            log_enclosed_mass = mu5_0 + beta5 * (mstar - 11.4) + xi5 * delta_r + sigma5 * nrm[5]
            mu_gamma = mu_gamma_0 + beta_gamma * (mstar - 11.4) + xi_gamma * delta_r
            gamma = truncnorm_sample(
                mu_gamma,
                sigma_gamma,
                gamma_trunc_low,
                gamma_trunc_high,
                nrm[7],
            )

        if not math.isfinite(gamma):
            continue

        theta_e = theta_ein_arcsec(
            zd,
            zs,
            log_enclosed_mass,
            gamma,
            z_grid,
            chi_kpc_grid,
            mass_radius_kpc,
        )
        if theta_e <= 0.0:
            continue

        cs = interp1d_clip(gamma, cs_gamma_grid, cs_over_theta)
        area = math.pi * (cs * theta_e) ** 2
        pf = p_find(theta_e, theta0, loga)
        acc += (1.0 / trunc_den) * pf * area

    return acc / n_samples
