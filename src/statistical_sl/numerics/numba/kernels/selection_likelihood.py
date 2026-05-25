"""
Shared selection and observed-likelihood kernels.

This module sits above the scalar numerical primitives but below any concrete
model.  It owns small scientific fragments that are already used by more than
one production model:

* `theta_E x gamma` cross-section weighting combined with the survey detection
  probability;
* source-redshift density choices that differ by model but share the same
  call-site role;
* observed velocity-dispersion likelihood factors.

The functions intentionally remain scalar and explicit.  Numba can inline them
into each model's larger integration loops, while model-owned kernels keep
control of the surrounding integration variables and scientific assumptions.
"""

from __future__ import annotations

import math

import numba as nb
import numpy as np

from .distributions import normal_pdf, truncated_normal_pdf_nonneg
from .interpolation import interp_cross_section_theta_gamma
from .selection import p_find


@nb.njit(cache=True, inline="always")
def cross_section_find_weight(
    theta_e: float,
    gamma: float,
    theta_for_detection: float,
    theta0: float,
    loga: float,
    theta_e_axis: np.ndarray,
    gamma_axis: np.ndarray,
    cross_section_grid: np.ndarray,
) -> float:
    """
    Return the product of cross-section and survey detection probability.

    `theta_e` is the physical Einstein radius used to look up the precomputed
    source-plane or finite-fibre cross-section.  `theta_for_detection` is the
    angular scale used by the selection function, which may be the true
    Einstein radius or a model-owned proxy.
    """

    if theta_e <= 0.0 or theta_for_detection <= 0.0 or not math.isfinite(gamma):
        return 0.0

    cross_section = interp_cross_section_theta_gamma(
        theta_e,
        gamma,
        theta_e_axis,
        gamma_axis,
        cross_section_grid,
    )
    if cross_section <= 0.0:
        return 0.0

    find_probability = p_find(theta_for_detection, theta0, loga)
    if find_probability <= 0.0:
        return 0.0
    return cross_section * find_probability


@nb.njit(cache=True, inline="always")
def gaussian_source_redshift_density(zs: float, mu_zs: float, sigma_zs: float) -> float:
    """
    Return the ordinary Gaussian effective source-redshift density.

    Keeping the wrapper separate from `normal_pdf` makes the source-redshift
    contract visible at call sites without duplicating the low-level
    normal-density implementation.
    """

    return normal_pdf(zs, mu_zs, sigma_zs)


@nb.njit(cache=True, inline="always")
def truncated_nonnegative_source_redshift_density(
    zs: float,
    mu_zs: float,
    sigma_zs: float,
) -> float:
    """
    Return a non-negative truncated Gaussian source-redshift density.

    The truncation lower bound is fixed at zero so the density is normalized
    over physically allowed redshifts.
    """

    return truncated_normal_pdf_nonneg(zs, mu_zs, sigma_zs)


@nb.njit(cache=True, inline="always")
def sigma_model_from_s2(s2_value: float, log_enclosed_mass: float) -> float:
    """
    Convert a dimensionless S2 grid value and enclosed mass into sigma.

    The lower floor preserves the historical likelihood behavior for tiny or
    numerically negative grid products: the model remains finite rather than
    taking a square-root of an invalid value.
    """

    return math.sqrt(max(s2_value * 10.0**log_enclosed_mass, 1.0e-30))


@nb.njit(cache=True, inline="always")
def observed_sigma_likelihood(
    lens_index: int,
    observed_sigma_count: np.ndarray,
    has_s2: np.ndarray,
    sigma_obs: np.ndarray,
    sigma_err: np.ndarray,
    sigma_model: float,
) -> float:
    """
    Return the product of observed velocity-dispersion likelihood factors.

    A lens with no observed sigma contributes a neutral factor of one.  A lens
    that does have observations but lacks an S2 model grid contributes zero,
    because the likelihood cannot be evaluated consistently for that row.
    """

    n_observed = observed_sigma_count[lens_index]
    if n_observed <= 0:
        return 1.0
    if has_s2[lens_index] == 0:
        return 0.0

    probability = 1.0
    for sigma_index in range(n_observed):
        probability *= normal_pdf(
            sigma_obs[lens_index, sigma_index],
            sigma_model,
            sigma_err[lens_index, sigma_index],
        )
    return probability


__all__ = [
    "cross_section_find_weight",
    "gaussian_source_redshift_density",
    "observed_sigma_likelihood",
    "sigma_model_from_s2",
    "truncated_nonnegative_source_redshift_density",
]
