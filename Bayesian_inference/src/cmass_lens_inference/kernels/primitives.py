"""
Shared `numba` primitives for the performance-critical inference path.

Why this file exists:
- likelihood and normalization must use identical numerical approximations
- low-level math kernels should not depend on dataclasses or I/O code
- future performance work needs one obvious place to tune primitive behavior

The functions here are intentionally generic. They know nothing about HDF5,
profiles, YAML, logging, progress bars, or sampler orchestration.
"""

from __future__ import annotations

import math

import numba as nb
import numpy as np
from astropy.constants import G, c
import astropy.units as u


SQRT2 = math.sqrt(2.0)
SQRT2PI = math.sqrt(2.0 * math.pi)
LOG10_4 = math.log10(4.0)
LOG10_2PI = math.log10(2.0 * math.pi)
C_KM_S = c.to("km/s").value
G_KPC_KMS2_MSUN = G.to(u.kpc * u.km**2 / (u.s**2 * u.Msun)).value
GAMMA_MODE_DEPENDENT_CODE = 0
GAMMA_MODE_INDEPENDENT_CODE = 1
GAMMA_MODE_SIGMA_STAR_DEPENDENT_CODE = 2



@nb.njit(cache=True)
def interp1d_clip(x: float, xp: np.ndarray, fp: np.ndarray) -> float:
    """Linearly interpolate with clipping to the nearest boundary value."""

    n = xp.shape[0]
    if x <= xp[0]:
        return fp[0]
    if x >= xp[n - 1]:
        return fp[n - 1]

    lo = 0
    hi = n - 1
    while hi - lo > 1:
        mid = (hi + lo) // 2
        if xp[mid] <= x:
            lo = mid
        else:
            hi = mid

    x0 = xp[lo]
    x1 = xp[hi]
    y0 = fp[lo]
    y1 = fp[hi]
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)


@nb.njit(cache=True, inline="always")
def _interp_axis_bracket_clip(x: float, axis: np.ndarray) -> tuple[int, int, float]:
    """
    Return the clipped interpolation bracket and weight for one monotonic axis.

    Why this helper exists:
    - the FP prior needs multilinear interpolation on precomputed Jeans tables
    - numba does not provide a reusable high-level interpolator for mixed 3-D
      and 4-D grids
    - keeping the axis search in one primitive avoids duplicating binary-search
      logic across the normalization and future diagnostic kernels
    """

    n = axis.shape[0]
    if n <= 1:
        return 0, 0, 0.0
    if x <= axis[0]:
        return 0, 0, 0.0
    if x >= axis[n - 1]:
        return n - 1, n - 1, 0.0

    lo = 0
    hi = n - 1
    while hi - lo > 1:
        mid = (hi + lo) // 2
        if axis[mid] <= x:
            lo = mid
        else:
            hi = mid

    x0 = axis[lo]
    x1 = axis[hi]
    if x1 <= x0:
        return lo, hi, 0.0
    return lo, hi, (x - x0) / (x1 - x0)


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
    """
    Multilinear interpolation on the FP sigma-unit grid with boundary clipping.

    The legacy FP prior reads a precomputed Jeans table rather than solving the
    dynamics model inside the sampler. This helper preserves that contract in a
    `numba`-friendly way for both supported profile families:
    - `sersic`: full 4-D interpolation over `(gamma, zd, logRe, n)`
    - `devauc`: the same code path with a degenerate length-one `n` axis
    """

    gamma_lo, gamma_hi, gamma_weight = _interp_axis_bracket_clip(gamma, gamma_axis)
    zd_lo, zd_hi, zd_weight = _interp_axis_bracket_clip(zd, zd_axis)
    log_re_lo, log_re_hi, log_re_weight = _interp_axis_bracket_clip(log_re_kpc, log_re_kpc_axis)

    if has_n_axis == 1:
        n_lo, n_hi, n_weight = _interp_axis_bracket_clip(n_value, n_axis)
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
            for log_re_corner in range(2):
                log_re_index = log_re_lo if log_re_corner == 0 else log_re_hi
                log_re_factor = (1.0 - log_re_weight) if log_re_corner == 0 else log_re_weight
                for n_corner in range(2):
                    n_index = n_lo if n_corner == 0 else n_hi
                    n_factor = (1.0 - n_weight) if n_corner == 0 else n_weight
                    result += (
                        gamma_factor
                        * zd_factor
                        * log_re_factor
                        * n_factor
                        * sigma_unit_grid[gamma_index, zd_index, log_re_index, n_index]
                    )
    return result


@nb.njit(cache=True)
def normal_pdf(x: float, mu: float, sigma: float) -> float:
    """Gaussian density used throughout the hierarchy."""

    if sigma <= 0.0:
        return 0.0
    z = (x - mu) / sigma
    return math.exp(-0.5 * z * z) / (sigma * SQRT2PI)


@nb.njit(cache=True)
def phi_standard(x: float) -> float:
    """Standard normal CDF helper."""

    return 0.5 * (1.0 + math.erf(x / SQRT2))


@nb.njit(cache=True)
def normal_ppf(p: float) -> float:
    """
    Inverse standard normal CDF using Acklam's approximation.

    This replaces SciPy `ppf` calls in the hot path so normalization can stay
    entirely inside `numba`.
    """

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
        num = ((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6
        den = (((d1 * q + d2) * q + d3) * q + d4) * q + 1.0
        return num / den
    if p > phigh:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        num = ((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6
        den = (((d1 * q + d2) * q + d3) * q + d4) * q + 1.0
        return -(num / den)

    q = p - 0.5
    r = q * q
    num = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q
    den = (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)
    return num / den


@nb.njit(cache=True)
def truncated_normal_pdf_nonneg(x: float, mu: float, sigma: float) -> float:
    """PDF of a Gaussian truncated below at zero."""

    if sigma <= 0.0 or x < 0.0:
        return 0.0
    z0 = (0.0 - mu) / sigma
    den = 1.0 - phi_standard(z0)
    if den <= 0.0:
        return 0.0
    return normal_pdf(x, mu, sigma) / den


@nb.njit(cache=True)
def skewnorm_pdf(x: float, loc: float, scale: float, alpha: float) -> float:
    """Skew-normal density shared by the prior and normalization code."""

    if scale <= 0.0:
        return 0.0
    t = (x - loc) / scale
    std_pdf = math.exp(-0.5 * t * t) / SQRT2PI
    cdf_term = 0.5 * (1.0 + math.erf(alpha * t / SQRT2))
    return 2.0 * std_pdf * cdf_term / scale


@nb.njit(cache=True)
def skewnorm_sample(loc: float, scale: float, alpha: float, z0: float, z1: float) -> float:
    """Skew-normal sampling via the standard-normal construction."""

    delta = alpha / math.sqrt(1.0 + alpha * alpha)
    return loc + scale * (delta * abs(z0) + math.sqrt(1.0 - delta * delta) * z1)


@nb.njit(cache=True)
def truncnorm_sample(loc: float, scale: float, low: float, high: float, z_u: float) -> float:
    """Sample a truncated Gaussian using inverse-CDF logic inside numba."""

    if scale <= 0.0 or high <= low:
        return math.nan
    a = (low - loc) / scale
    b = (high - loc) / scale
    pa = phi_standard(a)
    pb = phi_standard(b)
    if pb <= pa:
        return math.nan

    u = phi_standard(z_u)
    if u <= 1.0e-12:
        u = 1.0e-12
    elif u >= 1.0 - 1.0e-12:
        u = 1.0 - 1.0e-12

    q = pa + (pb - pa) * u
    x = loc + scale * normal_ppf(q)
    if x < low:
        return low
    if x > high:
        return high
    return x


@nb.njit(cache=True)
def comoving_chi_kpc(z: float, z_grid: np.ndarray, chi_kpc_grid: np.ndarray) -> float:
    """Comoving distance lookup on a precomputed kpc grid."""

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
    mass_radius_kpc: float = 5.0,
    log_mass_physical_offset: float = 0.0,
) -> float:
    """
    Einstein radius in arcseconds from `(m_R, gamma)` and distance tables.

    The same power-law lens can be parameterized by `m5`, `m10`, or any other
    supported enclosed projected mass definition. The only definition-specific
    ingredients are the physical aperture radius `R` and the optional additive
    conversion from the serialized mass unit into physical `Msun`.
    """

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
    base = (10.0 ** (log_enclosed_mass + log_mass_physical_offset)) / (
        math.pi * sigma_crit * (mass_radius_kpc ** (3.0 - gamma))
    )
    if base <= 0.0:
        return 0.0

    r_ein = base ** (1.0 / (gamma - 1.0))
    return r_ein / dl * 206265.0


@nb.njit(cache=True)
def p_find(theta_est: float, theta0: float, loga: float) -> float:
    """Discovery probability from the requirements' sigmoid selection model."""

    a = 10.0**loga
    x = -a * (theta_est - theta0)
    if x > 60.0:
        return 0.0
    if x < -60.0:
        return 1.0
    return 1.0 / (1.0 + math.exp(x))


@nb.njit(cache=True)
def mu_r(
    mstar: float,
    n_value: float,
    use_sersic_index: int,
    mu_r0: float,
    beta_r: float,
    nu_r: float,
    stellar_mass_pivot: float = 11.4,
) -> float:
    """Mean size relation used by both likelihood and normalization kernels."""

    out = mu_r0 + beta_r * (mstar - stellar_mass_pivot)
    if use_sersic_index == 1:
        out += nu_r * (math.log10(max(n_value, 1.0e-12)) - LOG10_4)
    return out


@nb.njit(cache=True, inline="always")
def theta_dimension_for_gamma_mode(gamma_mode_code: int) -> int:
    """Return the sampled theta dimension for the requested gamma mode."""

    if gamma_mode_code == GAMMA_MODE_DEPENDENT_CODE:
        return 12
    if gamma_mode_code == GAMMA_MODE_INDEPENDENT_CODE:
        return 10
    if gamma_mode_code == GAMMA_MODE_SIGMA_STAR_DEPENDENT_CODE:
        return 11
    return -1


@nb.njit(cache=True, inline="always")
def unpack_model_theta(theta: np.ndarray, gamma_mode_code: int) -> tuple[float, ...]:
    """
    Unpack the mode-aware model parameter vector into a fixed scalar bundle.

    The independent gamma mode removes two sampled parameters from the vector,
    but the kernels still consume one normalized tuple so the main numerical
    loops do not need to duplicate indexing logic.
    """

    mu5_0 = theta[0]
    beta5 = theta[1]
    xi5 = theta[2]
    sigma5 = theta[3]
    mu_gamma_0 = theta[4]
    if gamma_mode_code == GAMMA_MODE_DEPENDENT_CODE:
        beta_gamma = theta[5]
        xi_gamma = theta[6]
        beta_sigma_star_gamma = 0.0
        sigma_gamma = theta[7]
        mu_zs = theta[8]
        sigma_zs = theta[9]
        theta0 = theta[10]
        loga = theta[11]
    elif gamma_mode_code == GAMMA_MODE_INDEPENDENT_CODE:
        beta_gamma = 0.0
        xi_gamma = 0.0
        beta_sigma_star_gamma = 0.0
        sigma_gamma = theta[5]
        mu_zs = theta[6]
        sigma_zs = theta[7]
        theta0 = theta[8]
        loga = theta[9]
    else:
        beta_gamma = 0.0
        xi_gamma = 0.0
        beta_sigma_star_gamma = theta[5]
        sigma_gamma = theta[6]
        mu_zs = theta[7]
        sigma_zs = theta[8]
        theta0 = theta[9]
        loga = theta[10]

    return (
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
    )


@nb.njit(cache=True, inline="always")
def gamma_population_mean(
    mu_gamma_0: float,
    beta_gamma: float,
    xi_gamma: float,
    beta_sigma_star_gamma: float,
    mstar_shift11p4: float,
    delta_r: float,
    sigma_star_shift9p0: float,
    gamma_mode_code: int,
) -> float:
    """Return the conditional gamma mean for the active gamma mode."""

    if gamma_mode_code == GAMMA_MODE_DEPENDENT_CODE:
        return mu_gamma_0 + beta_gamma * mstar_shift11p4 + xi_gamma * delta_r
    if gamma_mode_code == GAMMA_MODE_SIGMA_STAR_DEPENDENT_CODE:
        return mu_gamma_0 + beta_sigma_star_gamma * sigma_star_shift9p0
    return mu_gamma_0
