"""Selection primitives used by production Numba kernels."""

from __future__ import annotations

import math

import numba as nb
import numpy as np

from .constants import C_KM_S


@nb.njit(cache=True)
def p_find(theta_est: float, theta0: float, loga: float) -> float:
    """Return the numerically stable sigmoid discovery probability."""

    slope = 10.0**loga
    x = -slope * (theta_est - theta0)
    if x > 60.0:
        return 0.0
    if x < -60.0:
        return 1.0
    return 1.0 / (1.0 + math.exp(x))


@nb.njit(cache=True)
def theta_e_est_from_sigma_proxy(
    sigma_proxy: float,
    zd: float,
    zs: float,
    z_grid: np.ndarray,
    chi_kpc_grid: np.ndarray,
) -> float:
    """Convert a velocity-dispersion proxy into an estimated Einstein radius."""

    if sigma_proxy <= 0.0 or zd <= 0.0 or zs <= zd:
        return 0.0
    chi_l = np.interp(zd, z_grid, chi_kpc_grid)
    chi_s = np.interp(zs, z_grid, chi_kpc_grid)
    ds = chi_s / (1.0 + zs)
    dls = (chi_s - chi_l) / (1.0 + zs)
    if ds <= 0.0 or dls <= 0.0:
        return 0.0
    return 4.0 * math.pi * (sigma_proxy / C_KM_S) ** 2 * dls / ds * 206265.0


__all__ = ["p_find", "theta_e_est_from_sigma_proxy"]
