"""Lensing geometry primitives used by production Numba kernels."""

from __future__ import annotations

import math

import numba as nb
import numpy as np

from .constants import C_KM_S, G_KPC_KMS2_MSUN
from .interpolation import interp1d_clip


@nb.njit(cache=True)
def comoving_chi_kpc(z: float, z_grid: np.ndarray, chi_kpc_grid: np.ndarray) -> float:
    """Look up comoving distance in kpc on a precomputed cosmology table."""

    if z <= 0.0:
        return 0.0
    return interp1d_clip(z, z_grid, chi_kpc_grid)


@nb.njit(cache=True)
def theta_ein_arcsec(
    zd: float,
    zs: float,
    log_enclosed_mass: float,
    gamma: float,
    z_grid: np.ndarray,
    chi_kpc_grid: np.ndarray,
    mass_radius_kpc: float,
    mass_log_physical_offset: float,
) -> float:
    """Return Einstein radius in arcsec for a projected power-law lens."""

    if zd <= 0.0 or zs <= zd or gamma <= 1.0:
        return 0.0

    chi_l = comoving_chi_kpc(zd, z_grid, chi_kpc_grid)
    chi_s = comoving_chi_kpc(zs, z_grid, chi_kpc_grid)
    if chi_s <= chi_l:
        return 0.0

    dl = chi_l / (1.0 + zd)
    ds = chi_s / (1.0 + zs)
    dls = (chi_s - chi_l) / (1.0 + zs)
    if dl <= 0.0 or ds <= 0.0 or dls <= 0.0:
        return 0.0

    sigma_crit = (C_KM_S * C_KM_S) / (4.0 * math.pi * G_KPC_KMS2_MSUN) * (ds / (dl * dls))
    base = (10.0 ** (log_enclosed_mass + mass_log_physical_offset)) / (
        math.pi * sigma_crit * (mass_radius_kpc ** (3.0 - gamma))
    )
    if base <= 0.0:
        return 0.0

    r_ein = base ** (1.0 / (gamma - 1.0))
    return r_ein / dl * 206265.0


__all__ = ["comoving_chi_kpc", "theta_ein_arcsec"]
