"""Interpolation primitives used by production Numba kernels."""

from __future__ import annotations

import math

import numba as nb
import numpy as np


@nb.njit(cache=True, inline="always")
def axis_bracket_clip(x: float, axis: np.ndarray) -> tuple[int, int, float]:
    """Return clipped interpolation indices and interpolation weight for one axis."""

    n_axis = axis.shape[0]
    if n_axis <= 1:
        return 0, 0, 0.0
    if x <= axis[0]:
        return 0, 0, 0.0
    if x >= axis[n_axis - 1]:
        return n_axis - 1, n_axis - 1, 0.0

    lo = 0
    hi = n_axis - 1
    while hi - lo > 1:
        mid = (hi + lo) // 2
        if axis[mid] <= x:
            lo = mid
        else:
            hi = mid

    width = axis[hi] - axis[lo]
    if width <= 0.0:
        return lo, hi, 0.0
    return lo, hi, (x - axis[lo]) / width


@nb.njit(cache=True)
def interp_cross_section_theta_gamma(
    theta_e: float,
    gamma: float,
    theta_e_axis: np.ndarray,
    gamma_axis: np.ndarray,
    cross_section_grid: np.ndarray,
) -> float:
    """Bilinearly interpolate canonical finite/source-plane cross-sections."""

    if theta_e < theta_e_axis[0] or theta_e > theta_e_axis[theta_e_axis.shape[0] - 1]:
        return 0.0

    theta_lo, theta_hi, theta_weight = axis_bracket_clip(theta_e, theta_e_axis)
    gamma_lo, gamma_hi, gamma_weight = axis_bracket_clip(gamma, gamma_axis)
    v00 = cross_section_grid[theta_lo, gamma_lo]
    v01 = cross_section_grid[theta_lo, gamma_hi]
    v10 = cross_section_grid[theta_hi, gamma_lo]
    v11 = cross_section_grid[theta_hi, gamma_hi]
    low_theta = v00 * (1.0 - gamma_weight) + v01 * gamma_weight
    high_theta = v10 * (1.0 - gamma_weight) + v11 * gamma_weight
    value = low_theta * (1.0 - theta_weight) + high_theta * theta_weight
    if not math.isfinite(value):
        return 0.0
    return value


@nb.njit(cache=True)
def interp_sigma_unit_clip(
    gamma: float,
    zd: float,
    log_re_kpc: float,
    n_value: float,
    gamma_axis: np.ndarray,
    zd_axis: np.ndarray,
    log_re_kpc_axis: np.ndarray,
    n_axis: np.ndarray,
    sigma_unit_grid: np.ndarray,
    has_n_axis: int,
) -> float:
    """Multilinearly interpolate normalized sigma-unit grids with clipping."""

    gamma_lo, gamma_hi, gamma_weight = axis_bracket_clip(gamma, gamma_axis)
    zd_lo, zd_hi, zd_weight = axis_bracket_clip(zd, zd_axis)
    re_lo, re_hi, re_weight = axis_bracket_clip(log_re_kpc, log_re_kpc_axis)
    if has_n_axis == 1:
        n_lo, n_hi, n_weight = axis_bracket_clip(n_value, n_axis)
    else:
        n_lo = 0
        n_hi = 0
        n_weight = 0.0

    result = 0.0
    for gamma_corner in range(2):
        gamma_index = gamma_lo if gamma_corner == 0 else gamma_hi
        gamma_factor = (1.0 - gamma_weight) if gamma_corner == 0 else gamma_weight
        for zd_corner in range(2):
            zd_index = zd_lo if zd_corner == 0 else zd_hi
            zd_factor = (1.0 - zd_weight) if zd_corner == 0 else zd_weight
            for re_corner in range(2):
                re_index = re_lo if re_corner == 0 else re_hi
                re_factor = (1.0 - re_weight) if re_corner == 0 else re_weight
                for n_corner in range(2):
                    n_index = n_lo if n_corner == 0 else n_hi
                    n_factor = (1.0 - n_weight) if n_corner == 0 else n_weight
                    result += (
                        gamma_factor
                        * zd_factor
                        * re_factor
                        * n_factor
                        * sigma_unit_grid[gamma_index, zd_index, re_index, n_index]
                    )
    return result


@nb.njit(cache=True)
def interp1d_clip(x: float, xp: np.ndarray, fp: np.ndarray) -> float:
    """Linearly interpolate one vector with clipped boundary values."""

    lo, hi, weight = axis_bracket_clip(x, xp)
    return fp[lo] * (1.0 - weight) + fp[hi] * weight


__all__ = [
    "axis_bracket_clip",
    "interp1d_clip",
    "interp_cross_section_theta_gamma",
    "interp_sigma_unit_clip",
]
