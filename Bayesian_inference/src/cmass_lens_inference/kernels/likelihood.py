"""
Monolithic all-lens likelihood kernel.

The production model evaluates the complete sample likelihood in one compiled
kernel rather than dispatching one lens at a time from Python. This is the key
structural change required to match the reference implementation's throughput.
"""

from __future__ import annotations

import math

import numba as nb
import numpy as np

from .primitives import normal_pdf, p_find, theta_ein_arcsec, truncated_normal_pdf_nonneg


@nb.njit(cache=True, parallel=True, fastmath=True)
def log_likelihood_lenses_numba(
    theta: np.ndarray,
    z_grid: np.ndarray,
    chi_kpc_grid: np.ndarray,
    cs_over_theta_int: np.ndarray,
    m5_grid_int: np.ndarray,
    jac_grid_int: np.ndarray,
    s2_grid_int: np.ndarray,
    has_s2: np.ndarray,
    num_sigma: np.ndarray,
    sigma_obs: np.ndarray,
    sigma_err: np.ndarray,
    zd: np.ndarray,
    zs: np.ndarray,
    p_zd_fixed: np.ndarray,
    mstar_shift11p4: np.ndarray,
    mstar_base: np.ndarray,
    delta_r_grid: np.ndarray,
    gamma_grid_int: np.ndarray,
    gamma_w: np.ndarray,
) -> float:
    """
    Evaluate the full sample log-likelihood in one `numba` kernel.

    All parameter-independent factors are precomputed into the compiled context.
    The kernel therefore only needs to evaluate the hyper-parameter-dependent
    Gaussian terms and multiply them into the fixed bases.
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
        return -np.inf

    n_lens = zd.shape[0]
    n_gamma = gamma_grid_int.shape[0]
    n_mstar = mstar_base.shape[1]
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

        int_gamma = 0.0
        for kg in range(n_gamma):
            gamma = gamma_grid_int[kg]
            m5 = m5_grid_int[i, kg]
            jac = jac_grid_int[i, kg]
            if jac <= 0.0:
                continue

            theta_e = theta_ein_arcsec(zd[i], zs[i], m5, gamma, z_grid, chi_kpc_grid)
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
                    sigma_model = math.sqrt(max(s2_grid_int[i, kg] * (10.0**m5), 1.0e-30))
                    for ks in range(num_sigma[i]):
                        p_sigma *= normal_pdf(sigma_obs[i, ks], sigma_model, sigma_err[i, ks])
            if p_sigma <= 0.0:
                continue

            int_mstar = 0.0
            for km in range(n_mstar):
                base = mstar_base[i, km]
                if base <= 0.0:
                    continue
                mu5 = mu5_0 + beta5 * mstar_shift11p4[i, km] + xi5 * delta_r_grid[i, km]
                mu_gamma = mu_gamma_0 + beta_gamma * mstar_shift11p4[i, km] + xi_gamma * delta_r_grid[i, km]
                int_mstar += base * normal_pdf(m5, mu5, sigma5) * normal_pdf(gamma, mu_gamma, sigma_gamma)

            int_gamma += gamma_w[kg] * int_mstar * p_zd * p_zs * pf * area * jac * p_sigma

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
