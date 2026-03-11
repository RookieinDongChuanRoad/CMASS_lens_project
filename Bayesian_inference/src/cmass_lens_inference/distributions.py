"""
Probability distribution helpers used by the inference code.

These helpers are intentionally explicit and side-effect free so they can be
tested independently from the sampler and I/O layers.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.special import erf
from scipy.stats import truncnorm


SQRT_TWO = math.sqrt(2.0)
SQRT_TWO_PI = math.sqrt(2.0 * math.pi)


def normal_pdf(x: np.ndarray | float, mean: np.ndarray | float, sigma: np.ndarray | float) -> np.ndarray:
    """Return the Gaussian probability density."""

    x_array = np.asarray(x, dtype=float)
    sigma_array = np.asarray(sigma, dtype=float)
    sigma_array = np.where(sigma_array <= 0.0, np.nan, sigma_array)
    coefficient = 1.0 / (sigma_array * SQRT_TWO_PI)
    exponent = -0.5 * ((x_array - mean) / sigma_array) ** 2
    return coefficient * np.exp(exponent)


def skew_normal_pdf(x: np.ndarray | float, loc: float, scale: float, alpha: float) -> np.ndarray:
    """
    Return the skew-normal density used for the stellar-mass function.

    The implementation follows the requirement document directly rather than
    delegating to SciPy so the mathematical form remains visible in project
    code.
    """

    z_value = (np.asarray(x, dtype=float) - loc) / scale
    phi = np.exp(-0.5 * z_value**2) / SQRT_TWO_PI
    cdf_term = 0.5 * (1.0 + erf(alpha * z_value / SQRT_TWO))
    return 2.0 / scale * phi * cdf_term


def truncated_normal_pdf(x: np.ndarray | float, mean: float, sigma: float, low: float, high: float) -> np.ndarray:
    """Evaluate a truncated Gaussian density using SciPy's stable implementation."""

    a = (low - mean) / sigma
    b = (high - mean) / sigma
    return truncnorm.pdf(x, a, b, loc=mean, scale=sigma)


def source_redshift_pdf(z_s: np.ndarray | float, mean: float, sigma: float) -> np.ndarray:
    """Return the effective source-redshift density with a hard `z_s >= 0` cut."""

    z_values = np.asarray(z_s, dtype=float)
    density = normal_pdf(z_values, mean, sigma)
    return np.where(z_values >= 0.0, density, 0.0)


def discovery_probability(theta_ein: np.ndarray | float, theta0: float, loga: float) -> np.ndarray:
    """Return the sigmoid detection efficiency `P_find`."""

    slope = 10.0**loga
    return 1.0 / (1.0 + np.exp(-slope * (np.asarray(theta_ein, dtype=float) - theta0)))
