"""
Reusable JAX numerical primitives for lens-population models.

These functions are intentionally model-agnostic: they know about Gaussian
densities, deterministic reparameterized draws, cosmology-table interpolation,
and generic power-law lensing geometry, but they do not know which population
model produced ``mstar``, ``gamma``, or a selection weight.  Concrete models
compose these primitives inside their own likelihood and normalization logic.
"""

from __future__ import annotations

import math

import jax.numpy as jnp
from jax.scipy.special import erf, ndtri
from astropy.constants import G, c
import astropy.units as u


SQRT2 = math.sqrt(2.0)
SQRT2PI = math.sqrt(2.0 * math.pi)
LOG10_2PI = math.log10(2.0 * math.pi)
LOG10_4 = math.log10(4.0)
C_KM_S = float(c.to("km/s").value)
G_KPC_KMS2_MSUN = float(G.to(u.kpc * u.km**2 / (u.s**2 * u.Msun)).value)


def as_jax_array(value) -> jnp.ndarray:
    """
    Convert context arrays into JAX arrays with explicit backend ownership.

    JAX accepts NumPy arrays directly, but routing conversion through one
    helper makes the backend precision/array policy easy to audit.
    """

    return jnp.asarray(value)


def trapezoid_last_axis(y: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    """
    Integrate ``y`` along its last axis with explicit trapezoid weights.

    The helper keeps broadcasting behavior transparent for lens/gamma/mstar
    arrays whose integration grids may be lens-specific.
    """

    widths = x[..., 1:] - x[..., :-1]
    return jnp.sum(0.5 * (y[..., 1:] + y[..., :-1]) * widths, axis=-1)


def normal_pdf(x: jnp.ndarray, mu: jnp.ndarray, sigma: jnp.ndarray) -> jnp.ndarray:
    """Gaussian density with invalid non-positive scatter mapped to zero."""

    safe_sigma = jnp.where(sigma > 0.0, sigma, 1.0)
    z = (x - mu) / safe_sigma
    density = jnp.exp(-0.5 * z * z) / (safe_sigma * SQRT2PI)
    return jnp.where(sigma > 0.0, density, 0.0)


def phi_standard(x: jnp.ndarray) -> jnp.ndarray:
    """Standard normal CDF written with JAX primitives."""

    return 0.5 * (1.0 + erf(x / SQRT2))


def truncated_normal_pdf_nonneg(
    x: jnp.ndarray,
    mu: jnp.ndarray,
    sigma: jnp.ndarray,
) -> jnp.ndarray:
    """PDF of a Gaussian truncated below at zero."""

    z0 = (0.0 - mu) / sigma
    denominator = 1.0 - phi_standard(z0)
    density = normal_pdf(x, mu, sigma) / denominator
    return jnp.where((sigma > 0.0) & (x >= 0.0) & (denominator > 0.0), density, 0.0)


def skewnorm_sample(
    loc: float,
    scale: float,
    alpha: float,
    z0: jnp.ndarray,
    z1: jnp.ndarray,
) -> jnp.ndarray:
    """Sample a skew-normal variate from two fixed standard-normal draws."""

    delta = alpha / jnp.sqrt(1.0 + alpha * alpha)
    return loc + scale * (delta * jnp.abs(z0) + jnp.sqrt(1.0 - delta * delta) * z1)


def truncnorm_sample(
    loc: jnp.ndarray,
    scale: jnp.ndarray,
    low: float,
    high: float,
    z_u: jnp.ndarray,
) -> jnp.ndarray:
    """
    Sample a truncated Gaussian by inverse CDF from a fixed normal draw.

    The random basis is fixed once per run.  This function only transforms one
    deterministic column of that basis, which keeps MC normalization
    differentiable with respect to model hyper-parameters.
    """

    a = (low - loc) / scale
    b = (high - loc) / scale
    pa = phi_standard(a)
    pb = phi_standard(b)
    u = jnp.clip(phi_standard(z_u), 1.0e-12, 1.0 - 1.0e-12)
    q = pa + (pb - pa) * u
    sample = loc + scale * ndtri(q)
    sample = jnp.clip(sample, low, high)
    return jnp.where((scale > 0.0) & (high > low) & (pb > pa), sample, low)


def theta_ein_arcsec(
    zd: jnp.ndarray,
    zs: jnp.ndarray,
    log_enclosed_mass: jnp.ndarray,
    gamma: jnp.ndarray,
    z_grid: jnp.ndarray,
    chi_kpc_grid: jnp.ndarray,
    mass_radius_kpc: float,
    mass_log_physical_offset: float,
) -> jnp.ndarray:
    """
    Einstein radius in arcsec for a projected power-law mass model.

    All distance lookups are table interpolations so the function remains
    differentiable with respect to model hyper-parameters while treating the
    cosmology grid as fixed run context.
    """

    chi_l = jnp.interp(zd, z_grid, chi_kpc_grid)
    chi_s = jnp.interp(zs, z_grid, chi_kpc_grid)
    dl = chi_l / (1.0 + zd)
    ds = chi_s / (1.0 + zs)
    dls = (chi_s - chi_l) / (1.0 + zs)
    sigma_crit = (C_KM_S * C_KM_S) / (4.0 * math.pi * G_KPC_KMS2_MSUN) * (ds / (dl * dls))
    base = (10.0 ** (log_enclosed_mass + mass_log_physical_offset)) / (
        math.pi * sigma_crit * (mass_radius_kpc ** (3.0 - gamma))
    )
    r_ein = base ** (1.0 / (gamma - 1.0))
    theta = r_ein / dl * 206265.0
    valid = (
        (zd > 0.0)
        & (zs > zd)
        & (gamma > 1.0)
        & (chi_s > chi_l)
        & (dl > 0.0)
        & (ds > 0.0)
        & (dls > 0.0)
        & (base > 0.0)
    )
    return jnp.where(valid, theta, 0.0)


def interp_cross_section_theta_gamma(
    theta_e: jnp.ndarray,
    gamma: jnp.ndarray,
    theta_e_axis: jnp.ndarray,
    gamma_axis: jnp.ndarray,
    cross_section_grid: jnp.ndarray,
) -> jnp.ndarray:
    """
    Bilinear interpolation for the canonical theta_E x gamma cross-section.

    The canonical boundary policy used by the first schema version is
    `zero_outside_theta_clip_gamma`: theta_E outside the prepared range has no
    valid cross-section, while gamma is clipped to the closest tabulated plane.
    This matches the schema document and keeps selection math independent of
    whether the grid came from CMASS's old separable approximation or a future
    Sonnenfeld finite-fibre calculation.
    """

    theta_inside = (theta_e >= theta_e_axis[0]) & (theta_e <= theta_e_axis[-1])
    theta_clipped = jnp.clip(theta_e, theta_e_axis[0], theta_e_axis[-1])
    gamma_clipped = jnp.clip(gamma, gamma_axis[0], gamma_axis[-1])

    def bracket(x: jnp.ndarray, axis: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        size = axis.shape[0]
        hi = jnp.clip(jnp.searchsorted(axis, x, side="right"), 1, size - 1)
        lo = hi - 1
        lo = jnp.where(x <= axis[0], 0, lo)
        hi = jnp.where(x <= axis[0], 0, hi)
        lo = jnp.where(x >= axis[-1], size - 1, lo)
        hi = jnp.where(x >= axis[-1], size - 1, hi)
        denominator = jnp.where(axis[hi] > axis[lo], axis[hi] - axis[lo], 1.0)
        weight = jnp.where(hi == lo, 0.0, (x - axis[lo]) / denominator)
        return lo, hi, weight

    theta_lo, theta_hi, theta_weight = bracket(theta_clipped, theta_e_axis)
    gamma_lo, gamma_hi, gamma_weight = bracket(gamma_clipped, gamma_axis)
    v00 = cross_section_grid[theta_lo, gamma_lo]
    v01 = cross_section_grid[theta_lo, gamma_hi]
    v10 = cross_section_grid[theta_hi, gamma_lo]
    v11 = cross_section_grid[theta_hi, gamma_hi]
    low_theta = v00 * (1.0 - gamma_weight) + v01 * gamma_weight
    high_theta = v10 * (1.0 - gamma_weight) + v11 * gamma_weight
    interpolated = low_theta * (1.0 - theta_weight) + high_theta * theta_weight
    return jnp.where(theta_inside & jnp.isfinite(interpolated), interpolated, 0.0)


def interp_sigma_unit_clip_scalar(
    gamma: jnp.ndarray,
    zd: jnp.ndarray,
    log_re_kpc: jnp.ndarray,
    n_value: jnp.ndarray,
    gamma_axis: jnp.ndarray,
    zd_axis: jnp.ndarray,
    log_re_kpc_axis: jnp.ndarray,
    n_axis: jnp.ndarray,
    sigma_unit_grid: jnp.ndarray,
    has_n_axis: int,
) -> jnp.ndarray:
    """
    Multilinear interpolation on a sigma-unit grid with boundary clipping.

    The implementation uses ``searchsorted`` plus explicit corner accumulation
    so it works for both 3-D de Vaucouleurs tables and 4-D Sersic tables after
    context building normalizes both to ``(gamma, zd, logRe, n)``.
    """

    def bracket(x: jnp.ndarray, axis: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        size = axis.shape[0]
        hi = jnp.clip(jnp.searchsorted(axis, x, side="right"), 1, size - 1)
        lo = hi - 1
        lo = jnp.where(x <= axis[0], 0, lo)
        hi = jnp.where(x <= axis[0], 0, hi)
        lo = jnp.where(x >= axis[-1], size - 1, lo)
        hi = jnp.where(x >= axis[-1], size - 1, hi)
        denominator = jnp.where(axis[hi] > axis[lo], axis[hi] - axis[lo], 1.0)
        weight = jnp.where(hi == lo, 0.0, (x - axis[lo]) / denominator)
        return lo, hi, weight

    gamma_lo, gamma_hi, gamma_weight = bracket(gamma, gamma_axis)
    zd_lo, zd_hi, zd_weight = bracket(zd, zd_axis)
    re_lo, re_hi, re_weight = bracket(log_re_kpc, log_re_kpc_axis)
    n_lo_raw, n_hi_raw, n_weight_raw = bracket(n_value, n_axis)
    n_lo = jnp.where(has_n_axis == 1, n_lo_raw, 0)
    n_hi = jnp.where(has_n_axis == 1, n_hi_raw, 0)
    n_weight = jnp.where(has_n_axis == 1, n_weight_raw, 0.0)

    result = 0.0
    for gamma_corner in range(2):
        gamma_index = jnp.where(gamma_corner == 0, gamma_lo, gamma_hi)
        gamma_factor = jnp.where(gamma_corner == 0, 1.0 - gamma_weight, gamma_weight)
        for zd_corner in range(2):
            zd_index = jnp.where(zd_corner == 0, zd_lo, zd_hi)
            zd_factor = jnp.where(zd_corner == 0, 1.0 - zd_weight, zd_weight)
            for re_corner in range(2):
                re_index = jnp.where(re_corner == 0, re_lo, re_hi)
                re_factor = jnp.where(re_corner == 0, 1.0 - re_weight, re_weight)
                for n_corner in range(2):
                    n_index = jnp.where(n_corner == 0, n_lo, n_hi)
                    n_factor = jnp.where(n_corner == 0, 1.0 - n_weight, n_weight)
                    result = result + (
                        gamma_factor
                        * zd_factor
                        * re_factor
                        * n_factor
                        * sigma_unit_grid[gamma_index, zd_index, re_index, n_index]
                    )
    return result


__all__ = [
    "LOG10_2PI",
    "LOG10_4",
    "SQRT2",
    "SQRT2PI",
    "as_jax_array",
    "interp_cross_section_theta_gamma",
    "interp_sigma_unit_clip_scalar",
    "normal_pdf",
    "phi_standard",
    "skewnorm_sample",
    "theta_ein_arcsec",
    "trapezoid_last_axis",
    "truncated_normal_pdf_nonneg",
    "truncnorm_sample",
]
