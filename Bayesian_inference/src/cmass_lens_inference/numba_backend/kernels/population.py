"""Population relation helper kernels."""

from __future__ import annotations

import math

import numba as nb

from .constants import LOG10_4


@nb.njit(cache=True)
def linear_size_relation_mean(
    mstar: float,
    n_value: float,
    use_sersic_index: int,
    intercept: float,
    mass_slope: float,
    sersic_slope: float,
    stellar_mass_pivot: float,
) -> float:
    """Return a linear size-relation mean with an optional Sersic-index term."""

    value = intercept + mass_slope * (mstar - stellar_mass_pivot)
    if use_sersic_index == 1:
        value += sersic_slope * (math.log10(max(n_value, 1.0e-12)) - LOG10_4)
    return value


@nb.njit(cache=True, inline="always")
def smooth_truncated_schechter_density(
    zd: float,
    mstar: float,
    mbar: float,
    parent_alpha: float,
    truncation_mass_scatter: float,
    truncation_threshold: float,
) -> float:
    """Return an unnormalized smooth-truncated Schechter density."""

    completeness = math.atan((mstar - truncation_threshold) / truncation_mass_scatter) / math.pi + 0.5
    schechter_mass = 10.0 ** (mstar - mbar)
    schechter = 10.0 ** ((mstar - mbar) * (parent_alpha + 1.0))
    return max(zd, 1.0e-6) ** 2 * completeness * schechter * math.exp(-schechter_mass)


@nb.njit(cache=True, inline="always")
def constant_gamma_mean(intercept: float) -> float:
    """Return a constant density-slope mean."""

    return intercept


@nb.njit(cache=True, inline="always")
def mass_size_linear_gamma_mean(
    intercept: float,
    mass_slope: float,
    size_residual_slope: float,
    mstar_shift: float,
    size_residual: float,
) -> float:
    """Return a density-slope mean linear in mass and size residual."""

    return intercept + mass_slope * mstar_shift + size_residual_slope * size_residual


@nb.njit(cache=True, inline="always")
def sigma_star_linear_gamma_mean(
    intercept: float,
    sigma_star_slope: float,
    sigma_star_shift: float,
) -> float:
    """Return a density-slope mean linear in the sigma-star proxy."""

    return intercept + sigma_star_slope * sigma_star_shift


@nb.njit(cache=True, inline="always")
def gaussian_linear_mass_mean(
    intercept: float,
    mass_slope: float,
    size_residual_slope: float,
    mstar_shift: float,
    size_residual: float,
) -> float:
    """Return an aperture-mass relation mean linear in mass and size residual."""

    return intercept + mass_slope * mstar_shift + size_residual_slope * size_residual


@nb.njit(cache=True, inline="always")
def quadratic_size_relation_mean(
    mstar: float,
    intercept: float,
    linear_term: float,
    quadratic_term: float,
) -> float:
    """Return a quadratic size-relation mean."""

    return intercept + linear_term * mstar + quadratic_term * mstar * mstar


__all__ = [
    "constant_gamma_mean",
    "gaussian_linear_mass_mean",
    "linear_size_relation_mean",
    "mass_size_linear_gamma_mean",
    "quadratic_size_relation_mean",
    "sigma_star_linear_gamma_mean",
    "smooth_truncated_schechter_density",
]
