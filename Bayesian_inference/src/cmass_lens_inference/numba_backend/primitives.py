"""
Shared Numba primitives for production inference kernels.

These functions are deliberately low-level and model-agnostic.  They own the
small pieces that every model kernel needs repeatedly: table interpolation,
Gaussian densities, deterministic proposal transforms, power-law Einstein
radius geometry, and selection sigmoid evaluation.  Keeping them here prevents
CMASS and Sonnenfeld kernels from carrying subtly different numerical helpers.
"""

from __future__ import annotations

import math

from astropy.constants import G, c
import astropy.units as u
import numba as nb
import numpy as np


SQRT2 = math.sqrt(2.0)
SQRT2PI = math.sqrt(2.0 * math.pi)
LOG10_2PI = math.log10(2.0 * math.pi)
LOG10_4 = math.log10(4.0)
C_KM_S = float(c.to("km/s").value)
G_KPC_KMS2_MSUN = float(G.to(u.kpc * u.km**2 / (u.s**2 * u.Msun)).value)

GAMMA_MODE_DEPENDENT_CODE = 0
GAMMA_MODE_INDEPENDENT_CODE = 1
GAMMA_MODE_SIGMA_STAR_DEPENDENT_CODE = 2


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
    """
    Approximate the inverse standard-normal CDF with Acklam's formula.

    The MC normalization maps fixed standard-normal rows into truncated
    variables.  Numba cannot call SciPy inside hot kernels, so this standalone
    approximation keeps the transform compiled and deterministic.
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


@nb.njit(cache=True, inline="always")
def _axis_bracket_clip(x: float, axis: np.ndarray) -> tuple[int, int, float]:
    """
    Return clipped interpolation indices and interpolation weight for one axis.

    Clipping is used for gamma/redshift/size/n axes where canonical schema
    semantics allow nearest-edge extrapolation.  The cross-section theta axis
    uses this helper only after the caller has separately enforced the
    zero-outside-theta boundary rule.
    """

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
    """
    Bilinearly interpolate canonical finite/source-plane cross-sections.

    Canonical v1 uses `zero_outside_theta_clip_gamma`: gamma is clipped to the
    nearest prepared plane, while theta_E outside the prepared range means the
    cross-section is physically unavailable and must contribute zero.  Both
    CMASS and Sonnenfeld production kernels route through this function.
    """

    if theta_e < theta_e_axis[0] or theta_e > theta_e_axis[theta_e_axis.shape[0] - 1]:
        return 0.0

    theta_lo, theta_hi, theta_weight = _axis_bracket_clip(theta_e, theta_e_axis)
    gamma_lo, gamma_hi, gamma_weight = _axis_bracket_clip(gamma, gamma_axis)
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

    gamma_lo, gamma_hi, gamma_weight = _axis_bracket_clip(gamma, gamma_axis)
    zd_lo, zd_hi, zd_weight = _axis_bracket_clip(zd, zd_axis)
    re_lo, re_hi, re_weight = _axis_bracket_clip(log_re_kpc, log_re_kpc_axis)
    if has_n_axis == 1:
        n_lo, n_hi, n_weight = _axis_bracket_clip(n_value, n_axis)
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

    lo, hi, weight = _axis_bracket_clip(x, xp)
    return fp[lo] * (1.0 - weight) + fp[hi] * weight


@nb.njit(cache=True)
def trapezoid_1d(y: np.ndarray, x: np.ndarray) -> float:
    """Return an explicit trapezoid integral over one one-dimensional grid."""

    total = 0.0
    for index in range(x.shape[0] - 1):
        total += 0.5 * (y[index + 1] + y[index]) * (x[index + 1] - x[index])
    return total


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
def mu_r(
    mstar: float,
    n_value: float,
    use_sersic_index: int,
    mu_r0: float,
    beta_r: float,
    nu_r: float,
    stellar_mass_pivot: float,
) -> float:
    """Return the CMASS parent size-relation mean."""

    value = mu_r0 + beta_r * (mstar - stellar_mass_pivot)
    if use_sersic_index == 1:
        value += nu_r * (math.log10(max(n_value, 1.0e-12)) - LOG10_4)
    return value


@nb.njit(cache=True, inline="always")
def theta_dimension_for_gamma_mode(gamma_mode_code: int) -> int:
    """Return the sampled dimension for CMASS gamma-mode static codes."""

    if gamma_mode_code == GAMMA_MODE_DEPENDENT_CODE:
        return 12
    if gamma_mode_code == GAMMA_MODE_INDEPENDENT_CODE:
        return 10
    if gamma_mode_code == GAMMA_MODE_SIGMA_STAR_DEPENDENT_CODE:
        return 11
    return -1


@nb.njit(cache=True, inline="always")
def unpack_cmass_theta(theta: np.ndarray, gamma_mode_code: int) -> tuple[float, ...]:
    """
    Return one fixed scalar tuple for the active CMASS parameterization.

    The current production CMASS model uses the sigma-star dependent 11D vector,
    but retaining the mode-aware unpacker keeps legacy fixtures readable and
    makes rejection behavior explicit if a future model variant reuses the
    kernel.
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
def cmass_gamma_population_mean(
    mu_gamma_0: float,
    beta_gamma: float,
    xi_gamma: float,
    beta_sigma_star_gamma: float,
    mstar_shift11p4: float,
    delta_r: float,
    sigma_star_shift9p0: float,
    gamma_mode_code: int,
) -> float:
    """Return the CMASS conditional gamma mean for the active static code."""

    if gamma_mode_code == GAMMA_MODE_DEPENDENT_CODE:
        return mu_gamma_0 + beta_gamma * mstar_shift11p4 + xi_gamma * delta_r
    if gamma_mode_code == GAMMA_MODE_SIGMA_STAR_DEPENDENT_CODE:
        return mu_gamma_0 + beta_sigma_star_gamma * sigma_star_shift9p0
    return mu_gamma_0


__all__ = [
    "C_KM_S",
    "LOG10_2PI",
    "LOG10_4",
    "cmass_gamma_population_mean",
    "interp_cross_section_theta_gamma",
    "interp_sigma_unit_clip",
    "mu_r",
    "normal_pdf",
    "p_find",
    "phi_standard",
    "skewnorm_sample",
    "theta_dimension_for_gamma_mode",
    "theta_ein_arcsec",
    "trapezoid_1d",
    "truncated_normal_pdf",
    "truncated_normal_pdf_nonneg",
    "truncnorm_sample",
    "unpack_cmass_theta",
]
