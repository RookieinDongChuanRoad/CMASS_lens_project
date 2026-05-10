"""Distribution primitives used by production Numba kernels."""

from __future__ import annotations

import math

import numba as nb

from .constants import SQRT2, SQRT2PI


@nb.njit(cache=True)
def phi_standard(x: float) -> float:
    """Return the standard normal CDF for one scalar value."""

    return 0.5 * (1.0 + math.erf(x / SQRT2))


@nb.njit(cache=True)
def normal_pdf(x: float, mu: float, sigma: float) -> float:
    """Return a Gaussian density, mapping invalid non-positive scatter to zero."""

    if sigma <= 0.0:
        return 0.0
    z = (x - mu) / sigma
    return math.exp(-0.5 * z * z) / (sigma * SQRT2PI)


@nb.njit(cache=True)
def truncated_normal_pdf_nonneg(x: float, mu: float, sigma: float) -> float:
    """Return the PDF of a Gaussian truncated below zero."""

    if sigma <= 0.0 or x < 0.0:
        return 0.0
    denominator = 1.0 - phi_standard((0.0 - mu) / sigma)
    if denominator <= 0.0:
        return 0.0
    return normal_pdf(x, mu, sigma) / denominator


@nb.njit(cache=True)
def normal_ppf(p: float) -> float:
    """Approximate the inverse standard-normal CDF with Acklam's formula."""

    if p <= 0.0:
        return -1.0e300
    if p >= 1.0:
        return 1.0e300

    a1 = -3.969683028665376e01
    a2 = 2.209460984245205e02
    a3 = -2.759285104469687e02
    a4 = 1.383577518672690e02
    a5 = -3.066479806614716e01
    a6 = 2.506628277459239e00

    b1 = -5.447609879822406e01
    b2 = 1.615858368580409e02
    b3 = -1.556989798598866e02
    b4 = 6.680131188771972e01
    b5 = -1.328068155288572e01

    c1 = -7.784894002430293e-03
    c2 = -3.223964580411365e-01
    c3 = -2.400758277161838e00
    c4 = -2.549732539343734e00
    c5 = 4.374664141464968e00
    c6 = 2.938163982698783e00

    d1 = 7.784695709041462e-03
    d2 = 3.224671290700398e-01
    d3 = 2.445134137142996e00
    d4 = 3.754408661907416e00

    plow = 0.02425
    phigh = 1.0 - plow
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        numerator = ((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6
        denominator = (((d1 * q + d2) * q + d3) * q + d4) * q + 1.0
        return numerator / denominator
    if p > phigh:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        numerator = ((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6
        denominator = (((d1 * q + d2) * q + d3) * q + d4) * q + 1.0
        return -(numerator / denominator)

    q = p - 0.5
    r = q * q
    numerator = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q
    denominator = (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)
    return numerator / denominator


@nb.njit(cache=True)
def skewnorm_sample(loc: float, scale: float, alpha: float, z0: float, z1: float) -> float:
    """Map two standard-normal draws into one skew-normal variate."""

    delta = alpha / math.sqrt(1.0 + alpha * alpha)
    return loc + scale * (delta * abs(z0) + math.sqrt(1.0 - delta * delta) * z1)


@nb.njit(cache=True)
def truncnorm_sample(loc: float, scale: float, low: float, high: float, z_u: float) -> float:
    """Map one standard-normal draw into a finite-support normal variate."""

    if scale <= 0.0 or high <= low:
        return math.nan
    lower_cdf = phi_standard((low - loc) / scale)
    upper_cdf = phi_standard((high - loc) / scale)
    if upper_cdf <= lower_cdf:
        return math.nan
    unit = phi_standard(z_u)
    if unit <= 1.0e-12:
        unit = 1.0e-12
    elif unit >= 1.0 - 1.0e-12:
        unit = 1.0 - 1.0e-12
    sample = loc + scale * normal_ppf(lower_cdf + (upper_cdf - lower_cdf) * unit)
    if sample < low:
        return low
    if sample > high:
        return high
    return sample


@nb.njit(cache=True)
def truncated_normal_pdf(
    x: float,
    loc: float,
    scale: float,
    low: float,
    high: float,
) -> float:
    """Return the finite-support Gaussian proposal density used by MC draws."""

    if scale <= 0.0 or high <= low or x < low or x > high:
        return 0.0
    denominator = phi_standard((high - loc) / scale) - phi_standard((low - loc) / scale)
    if denominator <= 0.0:
        return 0.0
    return normal_pdf(x, loc, scale) / denominator


__all__ = [
    "normal_pdf",
    "normal_ppf",
    "phi_standard",
    "skewnorm_sample",
    "truncated_normal_pdf",
    "truncated_normal_pdf_nonneg",
    "truncnorm_sample",
]
