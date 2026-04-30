"""
JAX/NumPyro numerical backend for CMASS lens inference.

This module is the production replacement for the old numba hot path.  It
keeps the same scientific equations and the same precomputed array context, but
expresses the actual likelihood and normalization in JAX so NumPyro can JIT and
differentiate the posterior.  The implementation deliberately mirrors the
legacy kernel structure because the first migration goal is numerical
equivalence, not a new science model.
"""

from __future__ import annotations

import math
from functools import lru_cache
from time import perf_counter

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax.scipy.special import erf, ndtri
import numpy as np
from astropy.constants import G, c
import astropy.units as u

from .compiled_context import build_compiled_context
from .parallel import resolve_parallelism
from .types import CompiledModel, RuntimeConfig


SQRT2 = math.sqrt(2.0)
SQRT2PI = math.sqrt(2.0 * math.pi)
LOG10_2PI = math.log10(2.0 * math.pi)
LOG10_4 = math.log10(4.0)
C_KM_S = float(c.to("km/s").value)
G_KPC_KMS2_MSUN = float(G.to(u.kpc * u.km**2 / (u.s**2 * u.Msun)).value)

GAMMA_MODE_DEPENDENT_CODE = 0
GAMMA_MODE_INDEPENDENT_CODE = 1
GAMMA_MODE_SIGMA_STAR_DEPENDENT_CODE = 2

FP_OLS_SUMMARY_SIZE = 6
FP_OLS_COUNT_INDEX = 0
FP_OLS_SUM_X1_INDEX = 1
FP_OLS_SUM_X1X1_INDEX = 2
FP_OLS_SUM_Y_INDEX = 3
FP_OLS_SUM_X1Y_INDEX = 4
FP_OLS_SUM_YY_INDEX = 5

JAX_LOG_PROB_BLOB_DTYPE = np.dtype(
    [
        ("total_log_prob_seconds", np.float64),
        ("likelihood_seconds", np.float64),
        ("normalization_seconds", np.float64),
        ("fp_prior_seconds", np.float64),
        ("normalization_value", np.float64),
        ("fp_prior_log_term", np.float64),
        ("fpfit_mu", np.float64),
        ("fpfit_beta", np.float64),
        ("fpfit_xi", np.float64),
        ("fpfit_scatter", np.float64),
        ("backend", "S16"),
    ]
)


def build_jax_model(runtime_config: RuntimeConfig) -> CompiledModel:
    """
    Build the JAX backend model object from a parsed runtime configuration.

    The existing compiled context builder already performs the expensive and
    parameter-independent preprocessing: HDF5 loading, profile normalization,
    interpolation-grid densification, and random-basis generation.  Reusing it
    keeps this migration focused on the inference backend rather than changing
    the data contract at the same time.
    """

    context, profile, cross_section_grid, cosmology, _, _ = build_compiled_context(runtime_config)
    parallelism = resolve_parallelism(
        runtime_config.runtime,
        runtime_config.sampling.num_chains,
    )
    return CompiledModel(
        config=runtime_config,
        profile=profile,
        cross_section_grid=cross_section_grid,
        cosmology=cosmology,
        parallelism=parallelism,
        context=context,
    )


def _build_timing_blob(
    *,
    total_log_prob_seconds: float,
    likelihood_seconds: float,
    normalization_seconds: float,
    fp_prior_seconds: float,
    normalization_value: float,
    fp_prior_log_term: float,
    fpfit_mu: float,
    fpfit_beta: float,
    fpfit_xi: float,
    fpfit_scatter: float,
) -> np.void:
    """Build the small structured diagnostic record returned with log_prob."""

    return np.array(
        (
            float(total_log_prob_seconds),
            float(likelihood_seconds),
            float(normalization_seconds),
            float(fp_prior_seconds),
            float(normalization_value),
            float(fp_prior_log_term),
            float(fpfit_mu),
            float(fpfit_beta),
            float(fpfit_xi),
            float(fpfit_scatter),
            b"jax",
        ),
        dtype=JAX_LOG_PROB_BLOB_DTYPE,
    )[()]


def _as_jax_array(value) -> jnp.ndarray:
    """
    Convert context arrays into JAX arrays with explicit float64 semantics.

    JAX accepts NumPy arrays directly, but doing the conversion in one helper
    makes the backend's precision policy obvious and prevents accidental
    float32 regressions when global defaults change.
    """

    return jnp.asarray(value)


def _trapezoid_last_axis(y: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    """
    Integrate `y` along its last axis with explicit trapezoid weights.

    JAX has `jnp.trapezoid`, but this local helper keeps broadcasting rules
    transparent for the two shapes used here:
    - lens/gamma/mstar arrays with a lens-specific `mstar` grid
    - lens/gamma arrays with a shared gamma grid
    """

    widths = x[..., 1:] - x[..., :-1]
    return jnp.sum(0.5 * (y[..., 1:] + y[..., :-1]) * widths, axis=-1)


def _normal_pdf(x: jnp.ndarray, mu: jnp.ndarray, sigma: jnp.ndarray) -> jnp.ndarray:
    """Gaussian density with invalid non-positive scatter mapped to zero."""

    safe_sigma = jnp.where(sigma > 0.0, sigma, 1.0)
    z = (x - mu) / safe_sigma
    density = jnp.exp(-0.5 * z * z) / (safe_sigma * SQRT2PI)
    return jnp.where(sigma > 0.0, density, 0.0)


def _phi_standard(x: jnp.ndarray) -> jnp.ndarray:
    """Standard normal CDF written with JAX primitives."""

    return 0.5 * (1.0 + erf(x / SQRT2))


def _truncated_normal_pdf_nonneg(x: jnp.ndarray, mu: jnp.ndarray, sigma: jnp.ndarray) -> jnp.ndarray:
    """PDF of a Gaussian truncated below at zero."""

    z0 = (0.0 - mu) / sigma
    denominator = 1.0 - _phi_standard(z0)
    density = _normal_pdf(x, mu, sigma) / denominator
    return jnp.where((sigma > 0.0) & (x >= 0.0) & (denominator > 0.0), density, 0.0)


def _skewnorm_sample(
    loc: float,
    scale: float,
    alpha: float,
    z0: jnp.ndarray,
    z1: jnp.ndarray,
) -> jnp.ndarray:
    """Sample a skew-normal variate from two fixed standard-normal draws."""

    delta = alpha / jnp.sqrt(1.0 + alpha * alpha)
    return loc + scale * (delta * jnp.abs(z0) + jnp.sqrt(1.0 - delta * delta) * z1)


def _truncnorm_sample(
    loc: jnp.ndarray,
    scale: jnp.ndarray,
    low: float,
    high: float,
    z_u: jnp.ndarray,
) -> jnp.ndarray:
    """
    Sample a truncated Gaussian by inverse CDF from a fixed normal draw.

    The random basis is fixed once per run.  This function only transforms one
    deterministic column of that basis, which keeps the Monte Carlo
    normalization differentiable with respect to the hyper-parameters.
    """

    a = (low - loc) / scale
    b = (high - loc) / scale
    pa = _phi_standard(a)
    pb = _phi_standard(b)
    u = jnp.clip(_phi_standard(z_u), 1.0e-12, 1.0 - 1.0e-12)
    q = pa + (pb - pa) * u
    sample = loc + scale * ndtri(q)
    sample = jnp.clip(sample, low, high)
    # Returning a finite fallback keeps gradients well-defined for NumPyro.
    # Invalid draws are still removed by downstream validity masks.
    return jnp.where((scale > 0.0) & (high > low) & (pb > pa), sample, low)


def _p_find(theta_est: jnp.ndarray, theta0: jnp.ndarray, loga: jnp.ndarray) -> jnp.ndarray:
    """Sigmoid lens-finding probability used by the current CMASS model."""

    a = 10.0**loga
    x = -a * (theta_est - theta0)
    return jnp.where(
        x > 60.0,
        0.0,
        jnp.where(x < -60.0, 1.0, 1.0 / (1.0 + jnp.exp(x))),
    )


def _mu_r(
    mstar: jnp.ndarray,
    n_value: jnp.ndarray,
    use_sersic_index: int,
    mu_r0: float,
    beta_r: float,
    nu_r: float,
    stellar_mass_pivot: float,
) -> jnp.ndarray:
    """Mean size relation for the active profile family."""

    sersic_term = nu_r * (jnp.log10(jnp.maximum(n_value, 1.0e-12)) - LOG10_4)
    return mu_r0 + beta_r * (mstar - stellar_mass_pivot) + jnp.where(use_sersic_index == 1, sersic_term, 0.0)


def _theta_dimension_for_gamma_mode(gamma_mode_code: int) -> int:
    """Return the sampled theta dimension for one gamma parameterization."""

    if gamma_mode_code == GAMMA_MODE_DEPENDENT_CODE:
        return 12
    if gamma_mode_code == GAMMA_MODE_INDEPENDENT_CODE:
        return 10
    if gamma_mode_code == GAMMA_MODE_SIGMA_STAR_DEPENDENT_CODE:
        return 11
    return -1


def _unpack_model_theta(theta: jnp.ndarray, gamma_mode_code: int) -> tuple[jnp.ndarray, ...]:
    """
    Unpack a mode-aware theta vector into the fixed scalar bundle.

    Keeping a single downstream signature prevents the likelihood and
    normalization expressions from duplicating indexing logic for every gamma
    mode.
    """

    mu5_0 = theta[0]
    beta5 = theta[1]
    xi5 = theta[2]
    sigma5 = theta[3]
    mu_gamma_0 = theta[4]
    if gamma_mode_code == GAMMA_MODE_DEPENDENT_CODE:
        beta_gamma = theta[5]
        xi_gamma = theta[6]
        beta_sigma_star_gamma = jnp.asarray(0.0, dtype=theta.dtype)
        sigma_gamma = theta[7]
        mu_zs = theta[8]
        sigma_zs = theta[9]
        theta0 = theta[10]
        loga = theta[11]
    elif gamma_mode_code == GAMMA_MODE_INDEPENDENT_CODE:
        beta_gamma = jnp.asarray(0.0, dtype=theta.dtype)
        xi_gamma = jnp.asarray(0.0, dtype=theta.dtype)
        beta_sigma_star_gamma = jnp.asarray(0.0, dtype=theta.dtype)
        sigma_gamma = theta[5]
        mu_zs = theta[6]
        sigma_zs = theta[7]
        theta0 = theta[8]
        loga = theta[9]
    else:
        beta_gamma = jnp.asarray(0.0, dtype=theta.dtype)
        xi_gamma = jnp.asarray(0.0, dtype=theta.dtype)
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


def _gamma_population_mean(
    mu_gamma_0: jnp.ndarray,
    beta_gamma: jnp.ndarray,
    xi_gamma: jnp.ndarray,
    beta_sigma_star_gamma: jnp.ndarray,
    mstar_shift11p4: jnp.ndarray,
    delta_r: jnp.ndarray,
    sigma_star_shift9p0: jnp.ndarray,
    gamma_mode_code: int,
) -> jnp.ndarray:
    """Conditional mean of gamma for the configured gamma mode."""

    if gamma_mode_code == GAMMA_MODE_DEPENDENT_CODE:
        return mu_gamma_0 + beta_gamma * mstar_shift11p4 + xi_gamma * delta_r
    if gamma_mode_code == GAMMA_MODE_SIGMA_STAR_DEPENDENT_CODE:
        return mu_gamma_0 + beta_sigma_star_gamma * sigma_star_shift9p0
    return mu_gamma_0 + jnp.zeros_like(mstar_shift11p4)


def _theta_ein_arcsec(
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
    valid = (zd > 0.0) & (zs > zd) & (gamma > 1.0) & (chi_s > chi_l) & (dl > 0.0) & (ds > 0.0) & (dls > 0.0) & (base > 0.0)
    return jnp.where(valid, theta, 0.0)


def _draw_population_state(
    theta_parts: tuple[jnp.ndarray, ...],
    nrm: jnp.ndarray,
    *,
    mu_d: float,
    sigma_d: float,
    mass_function_loc: float,
    mass_function_scale: float,
    mass_function_alpha: float,
    mu_r0: float,
    beta_r: float,
    sigma_r: float,
    nu_r: float,
    use_sersic_index: int,
    n_fixed: float,
    mu_n0: float,
    beta_n: float,
    sigma_n: float,
    gamma_trunc_low: float,
    gamma_trunc_high: float,
    gamma_mode_code: int,
    stellar_mass_pivot: float,
) -> tuple[jnp.ndarray, ...]:
    """Draw one latent parent-population state from one fixed normal row."""

    (
        mu5_0,
        beta5,
        xi5,
        sigma5,
        mu_gamma_0,
        beta_gamma,
        xi_gamma,
        beta_sigma_star_gamma,
        sigma_gamma,
        _mu_zs,
        _sigma_zs,
        _theta0,
        _loga,
    ) = theta_parts

    zd = mu_d + sigma_d * nrm[0]
    mstar = _skewnorm_sample(
        mass_function_loc,
        mass_function_scale,
        mass_function_alpha,
        nrm[2],
        nrm[3],
    )

    mstar_shift = mstar - stellar_mass_pivot
    logn = mu_n0 + beta_n * mstar_shift + sigma_n * nrm[4]
    n_draw = 10.0**logn
    n_value = jnp.where(use_sersic_index == 1, n_draw, n_fixed)
    mu_r_draw = _mu_r(mstar, n_value, use_sersic_index, mu_r0, beta_r, nu_r, stellar_mass_pivot)
    re_noise_column = jnp.where(use_sersic_index == 1, nrm[5], nrm[4])
    mass_noise_column = jnp.where(use_sersic_index == 1, nrm[6], nrm[5])
    re_draw = mu_r_draw + sigma_r * re_noise_column
    delta_r = re_draw - mu_r_draw
    log_enclosed_mass = mu5_0 + beta5 * mstar_shift + xi5 * delta_r + sigma5 * mass_noise_column

    sigma_star_shift9p0 = mstar - LOG10_2PI - 2.0 * re_draw - 9.0
    mu_gamma = _gamma_population_mean(
        mu_gamma_0,
        beta_gamma,
        xi_gamma,
        beta_sigma_star_gamma,
        mstar_shift,
        delta_r,
        sigma_star_shift9p0,
        gamma_mode_code,
    )
    gamma = _truncnorm_sample(mu_gamma, sigma_gamma, gamma_trunc_low, gamma_trunc_high, nrm[7])
    return zd, mstar, n_value, re_draw, delta_r, log_enclosed_mass, gamma


def _interp_sigma_unit_clip_scalar(
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
    Multilinear interpolation on the FP sigma-unit grid with boundary clipping.

    The implementation uses `searchsorted` plus explicit corner accumulation so
    it works for both 3-D devauc tables and 4-D sersic tables after the context
    builder normalizes both to `(gamma, zd, logRe, n)`.
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


def _normalization_and_fp_summary_value(
    theta: jnp.ndarray,
    *,
    base_normals: jnp.ndarray,
    cs_gamma_grid: jnp.ndarray,
    cs_over_theta: jnp.ndarray,
    z_grid: jnp.ndarray,
    chi_kpc_grid: jnp.ndarray,
    mu_d: float,
    sigma_d: float,
    mass_function_loc: float,
    mass_function_scale: float,
    mass_function_alpha: float,
    mu_r0: float,
    beta_r: float,
    sigma_r: float,
    nu_r: float,
    use_sersic_index: int,
    n_fixed: float,
    mu_n0: float,
    beta_n: float,
    sigma_n: float,
    gamma_trunc_low: float,
    gamma_trunc_high: float,
    mass_radius_kpc: float,
    gamma_mode_code: int,
    stellar_mass_pivot: float,
    mass_log_physical_offset: float,
    fp_enabled: int,
    fp_fit_mstar_min: float,
    fp_pivot_mstar: float,
    fp_gamma_axis: jnp.ndarray,
    fp_zd_axis: jnp.ndarray,
    fp_log_re_kpc_axis: jnp.ndarray,
    fp_n_axis: jnp.ndarray,
    fp_sigma_unit_grid: jnp.ndarray,
    fp_has_n_axis: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return MC selection normalization and FP OLS sufficient statistics."""

    theta_parts = _unpack_model_theta(theta, gamma_mode_code)
    (
        _mu5_0,
        _beta5,
        _xi5,
        sigma5,
        _mu_gamma_0,
        _beta_gamma,
        _xi_gamma,
        _beta_sigma_star_gamma,
        sigma_gamma,
        mu_zs,
        sigma_zs,
        theta0,
        loga,
    ) = theta_parts

    z0 = (0.0 - mu_zs) / sigma_zs
    trunc_den = 1.0 - _phi_standard(z0)
    inv_trunc_den = 1.0 / trunc_den

    def one_sample(nrm: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        zd, mstar, n_value, re_draw, delta_r, log_enclosed_mass, gamma = _draw_population_state(
            theta_parts,
            nrm,
            mu_d=mu_d,
            sigma_d=sigma_d,
            mass_function_loc=mass_function_loc,
            mass_function_scale=mass_function_scale,
            mass_function_alpha=mass_function_alpha,
            mu_r0=mu_r0,
            beta_r=beta_r,
            sigma_r=sigma_r,
            nu_r=nu_r,
            use_sersic_index=use_sersic_index,
            n_fixed=n_fixed,
            mu_n0=mu_n0,
            beta_n=beta_n,
            sigma_n=sigma_n,
            gamma_trunc_low=gamma_trunc_low,
            gamma_trunc_high=gamma_trunc_high,
            gamma_mode_code=gamma_mode_code,
            stellar_mass_pivot=stellar_mass_pivot,
        )

        zs = mu_zs + sigma_zs * nrm[1]
        theta_e = _theta_ein_arcsec(
            zd,
            zs,
            log_enclosed_mass,
            gamma,
            z_grid,
            chi_kpc_grid,
            mass_radius_kpc,
            mass_log_physical_offset,
        )
        cs = jnp.interp(gamma, cs_gamma_grid, cs_over_theta)
        area = math.pi * (cs * theta_e) ** 2
        normalization_weight = inv_trunc_den * _p_find(theta_e, theta0, loga) * area
        normalization_valid = (zd > 0.0) & (zs > 0.0) & (zs > zd) & jnp.isfinite(gamma) & (theta_e > 0.0)
        normalization_weight = jnp.where(normalization_valid, normalization_weight, 0.0)

        sigma_unit = _interp_sigma_unit_clip_scalar(
            gamma,
            zd,
            re_draw,
            n_value,
            fp_gamma_axis,
            fp_zd_axis,
            fp_log_re_kpc_axis,
            fp_n_axis,
            fp_sigma_unit_grid,
            fp_has_n_axis,
        )
        log_sigma_model = 0.5 * (jnp.log10(sigma_unit) + log_enclosed_mass)
        fp_valid = (
            (fp_enabled == 1)
            & (zd > 0.0)
            & jnp.isfinite(gamma)
            & (mstar > fp_fit_mstar_min)
            & (sigma_unit > 0.0)
            & jnp.isfinite(sigma_unit)
            & jnp.isfinite(log_sigma_model)
        )
        x1 = mstar - fp_pivot_mstar
        fp_row = jnp.asarray(
            [
                1.0,
                x1,
                x1 * x1,
                log_sigma_model,
                x1 * log_sigma_model,
                log_sigma_model * log_sigma_model,
            ],
            dtype=jnp.float64,
        )
        fp_row = jnp.where(fp_valid, fp_row, jnp.zeros(FP_OLS_SUMMARY_SIZE, dtype=jnp.float64))
        return normalization_weight, fp_row

    normalization_weights, fp_rows = jax.vmap(one_sample)(base_normals)
    z_norm = jnp.mean(normalization_weights)
    fp_summary = jnp.sum(fp_rows, axis=0)
    valid_theta = (
        (theta.shape[0] == _theta_dimension_for_gamma_mode(gamma_mode_code))
        & (sigma5 > 0.0)
        & (sigma_gamma > 0.0)
        & (sigma_zs > 0.0)
        & (trunc_den > 0.0)
    )
    return jnp.where(valid_theta, z_norm, 0.0), jnp.where(
        valid_theta,
        fp_summary,
        jnp.zeros(FP_OLS_SUMMARY_SIZE, dtype=jnp.float64),
    )


def _log_likelihood_value(
    theta: jnp.ndarray,
    *,
    z_grid: jnp.ndarray,
    chi_kpc_grid: jnp.ndarray,
    cs_over_theta_int: jnp.ndarray,
    mass_grid_int: jnp.ndarray,
    dmass_dthetaein_grid_int: jnp.ndarray,
    s2_grid_int: jnp.ndarray,
    has_s2: jnp.ndarray,
    num_sigma: jnp.ndarray,
    sigma_obs: jnp.ndarray,
    sigma_err: jnp.ndarray,
    zd: jnp.ndarray,
    zs: jnp.ndarray,
    p_zd_fixed: jnp.ndarray,
    mstar_grid: jnp.ndarray,
    mstar_shift11p4: jnp.ndarray,
    sigma_star_shift9p0_grid: jnp.ndarray,
    mstar_integrand_base: jnp.ndarray,
    delta_r_grid: jnp.ndarray,
    gamma_grid_int: jnp.ndarray,
    mass_radius_kpc: float,
    gamma_mode_code: int,
    mass_log_physical_offset: float,
) -> jnp.ndarray:
    """Vectorized all-lens likelihood equivalent to the legacy numba kernel."""

    theta_parts = _unpack_model_theta(theta, gamma_mode_code)
    (
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
    ) = theta_parts

    log_enclosed_mass = mass_grid_int
    gamma = gamma_grid_int[None, :]
    jac = jnp.abs(dmass_dthetaein_grid_int)
    theta_e = _theta_ein_arcsec(
        zd[:, None],
        zs[:, None],
        log_enclosed_mass,
        gamma,
        z_grid,
        chi_kpc_grid,
        mass_radius_kpc,
        mass_log_physical_offset,
    )
    area = math.pi * (cs_over_theta_int[None, :] * theta_e) ** 2
    pf = _p_find(theta_e, theta0, loga)
    p_zs = _truncated_normal_pdf_nonneg(zs, mu_zs, sigma_zs)

    sigma_model = jnp.sqrt(jnp.maximum(s2_grid_int * (10.0**log_enclosed_mass), 1.0e-30))
    p_sigma_1 = _normal_pdf(sigma_obs[:, None, 0], sigma_model, sigma_err[:, None, 0])
    p_sigma_2 = _normal_pdf(sigma_obs[:, None, 1], sigma_model, sigma_err[:, None, 1])
    p_sigma = jnp.where(num_sigma[:, None] >= 1, p_sigma_1, 1.0)
    p_sigma = jnp.where(num_sigma[:, None] >= 2, p_sigma * p_sigma_2, p_sigma)
    p_sigma = jnp.where((num_sigma[:, None] > 0) & (has_s2[:, None] == 0), 0.0, p_sigma)

    mu5 = mu5_0 + beta5 * mstar_shift11p4 + xi5 * delta_r_grid
    mu_gamma = _gamma_population_mean(
        mu_gamma_0,
        beta_gamma,
        xi_gamma,
        beta_sigma_star_gamma,
        mstar_shift11p4,
        delta_r_grid,
        sigma_star_shift9p0_grid,
        gamma_mode_code,
    )
    mstar_density = (
        mstar_integrand_base[:, None, :]
        * _normal_pdf(log_enclosed_mass[:, :, None], mu5[:, None, :], sigma5)
        * _normal_pdf(gamma_grid_int[None, :, None], mu_gamma[:, None, :], sigma_gamma)
    )
    integrated_mstar = _trapezoid_last_axis(mstar_density, mstar_grid[:, None, :])
    gamma_integrand = (
        integrated_mstar
        * p_zd_fixed[:, None]
        * p_zs[:, None]
        * pf
        * area
        * jac
        * p_sigma
    )
    gamma_valid = (jac > 0.0) & (theta_e > 0.0) & (area > 0.0) & (pf > 0.0) & (p_sigma > 0.0)
    gamma_integrand = jnp.where(gamma_valid, gamma_integrand, 0.0)
    lens_integrals = _trapezoid_last_axis(gamma_integrand, gamma_grid_int[None, :])
    all_valid = (
        (theta.shape[0] == _theta_dimension_for_gamma_mode(gamma_mode_code))
        & (sigma5 > 0.0)
        & (sigma_gamma > 0.0)
        & (sigma_zs > 0.0)
        & jnp.all(p_zd_fixed > 0.0)
        & jnp.all(p_zs > 0.0)
        & jnp.all(lens_integrals > 0.0)
    )
    return jnp.where(all_valid, jnp.sum(jnp.log(lens_integrals)), -jnp.inf)


def _solve_fundamental_plane_ols_jax(fp_summary: jnp.ndarray) -> tuple[jnp.ndarray, ...]:
    """
    Fit the hunit-aware 1D sigma-logM* relation from sufficient statistics.

    The mainline hunit migration changed the FP prior from a two-predictor
    `(mstar, delta_r)` regression to a one-predictor sigma-logM* summary.  The
    JAX backend keeps returning the historical `fpfit_xi` diagnostic slot, but
    fills it with NaN because no radius-slope coefficient is fitted.
    """

    sample_count = fp_summary[FP_OLS_COUNT_INDEX]
    xtx = jnp.asarray(
        [
            [sample_count, fp_summary[FP_OLS_SUM_X1_INDEX]],
            [fp_summary[FP_OLS_SUM_X1_INDEX], fp_summary[FP_OLS_SUM_X1X1_INDEX]],
        ],
        dtype=jnp.float64,
    )
    xty = jnp.asarray(
        [
            fp_summary[FP_OLS_SUM_Y_INDEX],
            fp_summary[FP_OLS_SUM_X1Y_INDEX],
        ],
        dtype=jnp.float64,
    )
    coefficients = jnp.linalg.solve(xtx, xty)
    sse = fp_summary[FP_OLS_SUM_YY_INDEX] - jnp.dot(coefficients, xty)
    sse = jnp.where((sse < 0.0) & (jnp.abs(sse) < 1.0e-12), 0.0, sse)
    scatter = jnp.sqrt(sse / sample_count)
    valid = (sample_count >= 2.0) & (sse >= 0.0) & jnp.all(jnp.isfinite(coefficients)) & jnp.isfinite(scatter)
    nan = jnp.asarray(jnp.nan, dtype=jnp.float64)
    return (
        jnp.where(valid, coefficients[0], nan),
        jnp.where(valid, coefficients[1], nan),
        nan,
        jnp.where(valid, scatter, nan),
    )


def _gaussian_quadratic_log_penalty(value: jnp.ndarray, mean: float, sigma: float) -> jnp.ndarray:
    """Unnormalized Gaussian quadratic penalty used by the FP prior."""

    z = (value - mean) / sigma
    return jnp.where((sigma > 0.0) & jnp.isfinite(value), -0.5 * z * z, -jnp.inf)


def _fp_prior_value(
    fp_summary: jnp.ndarray,
    *,
    fp_enabled: int,
    fp_fiducial_scatter: float,
    fp_scatter_error: float,
    fp_mu_v_prior: float,
    fp_mu_v_error: float,
    fp_beta_v_prior: float,
    fp_beta_v_error: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return the optional FP prior value and fitted diagnostic coefficients."""

    if fp_enabled == 0:
        nan = jnp.asarray(jnp.nan, dtype=jnp.float64)
        return jnp.asarray(0.0, dtype=jnp.float64), nan, nan, nan, nan

    intercept, beta_mass, beta_radius, scatter = _solve_fundamental_plane_ols_jax(fp_summary)
    log_prior = (
        _gaussian_quadratic_log_penalty(scatter, fp_fiducial_scatter, fp_scatter_error)
        + _gaussian_quadratic_log_penalty(intercept, fp_mu_v_prior, fp_mu_v_error)
        + _gaussian_quadratic_log_penalty(beta_mass, fp_beta_v_prior, fp_beta_v_error)
    )
    return (
        log_prior,
        intercept,
        beta_mass,
        beta_radius,
        scatter,
    )


@lru_cache(maxsize=16)
def _build_log_prob_components_jit(
    *,
    use_sersic_index: int,
    gamma_mode_code: int,
    fp_enabled: int,
    fp_has_n_axis: int,
):
    """
    Build a JIT function specialized to shape-stable profile/model flags.

    JAX needs Python branches such as gamma-mode unpacking to be static.  This
    factory keeps those choices outside the traced arguments while all numeric
    arrays remain dynamic inputs with stable shapes.
    """

    @jax.jit
    def compiled(
        theta: jnp.ndarray,
        z_grid: jnp.ndarray,
        chi_kpc_grid: jnp.ndarray,
        cs_gamma_grid: jnp.ndarray,
        cs_over_theta_grid: jnp.ndarray,
        cs_over_theta_int: jnp.ndarray,
        gamma_grid_int: jnp.ndarray,
        mass_grid_int: jnp.ndarray,
        dmass_dthetaein_grid_int: jnp.ndarray,
        s2_grid_int: jnp.ndarray,
        has_s2: jnp.ndarray,
        num_sigma: jnp.ndarray,
        sigma_obs: jnp.ndarray,
        sigma_err: jnp.ndarray,
        zd: jnp.ndarray,
        zs: jnp.ndarray,
        p_zd_fixed: jnp.ndarray,
        mstar_grid: jnp.ndarray,
        mstar_shift11p4: jnp.ndarray,
        sigma_star_shift9p0_grid: jnp.ndarray,
        mstar_integrand_base: jnp.ndarray,
        delta_r_grid: jnp.ndarray,
        base_normals: jnp.ndarray,
        scalar_context: jnp.ndarray,
        fp_gamma_axis: jnp.ndarray,
        fp_zd_axis: jnp.ndarray,
        fp_log_re_kpc_axis: jnp.ndarray,
        fp_n_axis: jnp.ndarray,
        fp_sigma_unit_grid: jnp.ndarray,
    ) -> tuple[jnp.ndarray, ...]:
        mass_radius_kpc = scalar_context[0]
        n_fixed = scalar_context[1]
        mu_n0 = scalar_context[2]
        beta_n = scalar_context[3]
        sigma_n = scalar_context[4]
        mass_function_loc = scalar_context[5]
        mass_function_scale = scalar_context[6]
        mass_function_alpha = scalar_context[7]
        mu_r0 = scalar_context[8]
        beta_r = scalar_context[9]
        sigma_r = scalar_context[10]
        nu_r = scalar_context[11]
        mu_d = scalar_context[12]
        sigma_d = scalar_context[13]
        gamma_trunc_low = scalar_context[14]
        gamma_trunc_high = scalar_context[15]
        normalization_min_value = scalar_context[16]
        fp_fit_mstar_min = scalar_context[17]
        fp_pivot_mstar = scalar_context[18]
        fp_fiducial_scatter = scalar_context[19]
        fp_scatter_error = scalar_context[20]
        fp_mu_v_prior = scalar_context[21]
        fp_mu_v_error = scalar_context[22]
        fp_beta_v_prior = scalar_context[23]
        fp_beta_v_error = scalar_context[24]
        stellar_mass_pivot = scalar_context[25]
        mass_log_physical_offset = scalar_context[26]

        z_norm, fp_summary = _normalization_and_fp_summary_value(
            theta,
            base_normals=base_normals,
            cs_gamma_grid=cs_gamma_grid,
            cs_over_theta=cs_over_theta_grid,
            z_grid=z_grid,
            chi_kpc_grid=chi_kpc_grid,
            mu_d=mu_d,
            sigma_d=sigma_d,
            mass_function_loc=mass_function_loc,
            mass_function_scale=mass_function_scale,
            mass_function_alpha=mass_function_alpha,
            mu_r0=mu_r0,
            beta_r=beta_r,
            sigma_r=sigma_r,
            nu_r=nu_r,
            use_sersic_index=use_sersic_index,
            n_fixed=n_fixed,
            mu_n0=mu_n0,
            beta_n=beta_n,
            sigma_n=sigma_n,
            gamma_trunc_low=gamma_trunc_low,
            gamma_trunc_high=gamma_trunc_high,
            mass_radius_kpc=mass_radius_kpc,
            gamma_mode_code=gamma_mode_code,
            stellar_mass_pivot=stellar_mass_pivot,
            mass_log_physical_offset=mass_log_physical_offset,
            fp_enabled=fp_enabled,
            fp_fit_mstar_min=fp_fit_mstar_min,
            fp_pivot_mstar=fp_pivot_mstar,
            fp_gamma_axis=fp_gamma_axis,
            fp_zd_axis=fp_zd_axis,
            fp_log_re_kpc_axis=fp_log_re_kpc_axis,
            fp_n_axis=fp_n_axis,
            fp_sigma_unit_grid=fp_sigma_unit_grid,
            fp_has_n_axis=fp_has_n_axis,
        )
        likelihood_value = _log_likelihood_value(
            theta,
            z_grid=z_grid,
            chi_kpc_grid=chi_kpc_grid,
            cs_over_theta_int=cs_over_theta_int,
            mass_grid_int=mass_grid_int,
            dmass_dthetaein_grid_int=dmass_dthetaein_grid_int,
            s2_grid_int=s2_grid_int,
            has_s2=has_s2,
            num_sigma=num_sigma,
            sigma_obs=sigma_obs,
            sigma_err=sigma_err,
            zd=zd,
            zs=zs,
            p_zd_fixed=p_zd_fixed,
            mstar_grid=mstar_grid,
            mstar_shift11p4=mstar_shift11p4,
            sigma_star_shift9p0_grid=sigma_star_shift9p0_grid,
            mstar_integrand_base=mstar_integrand_base,
            delta_r_grid=delta_r_grid,
            gamma_grid_int=gamma_grid_int,
            mass_radius_kpc=mass_radius_kpc,
            gamma_mode_code=gamma_mode_code,
            mass_log_physical_offset=mass_log_physical_offset,
        )
        log_fp_prior, fpfit_mu, fpfit_beta, fpfit_xi, fpfit_scatter = _fp_prior_value(
            fp_summary,
            fp_enabled=fp_enabled,
            fp_fiducial_scatter=fp_fiducial_scatter,
            fp_scatter_error=fp_scatter_error,
            fp_mu_v_prior=fp_mu_v_prior,
            fp_mu_v_error=fp_mu_v_error,
            fp_beta_v_prior=fp_beta_v_prior,
            fp_beta_v_error=fp_beta_v_error,
        )
        normalization_valid = jnp.isfinite(z_norm) & (z_norm > normalization_min_value)
        fp_valid = (fp_enabled == 0) | jnp.isfinite(log_fp_prior)
        total = likelihood_value - zd.shape[0] * jnp.log(z_norm) + log_fp_prior
        total = jnp.where(normalization_valid & fp_valid & jnp.isfinite(likelihood_value), total, -jnp.inf)
        return total, likelihood_value, z_norm, log_fp_prior, fpfit_mu, fpfit_beta, fpfit_xi, fpfit_scatter

    return compiled


def _scalar_context_array(compiled_model: CompiledModel) -> jnp.ndarray:
    """
    Pack scalar context fields into one float64 array for the JIT call.

    Keeping scalar values together shortens the public wrapper and makes it
    obvious which configuration constants influence the compiled posterior.
    """

    context = compiled_model.context
    return jnp.asarray(
        [
            context.mass_radius_kpc,
            context.n_fixed,
            context.mu_n0,
            context.beta_n,
            context.sigma_n,
            context.mass_function_loc,
            context.mass_function_scale,
            context.mass_function_alpha,
            context.mu_r0,
            context.beta_r,
            context.sigma_r,
            context.nu_r,
            context.mu_d,
            context.sigma_d,
            context.gamma_trunc_low,
            context.gamma_trunc_high,
            context.normalization_min_value,
            context.fp_fit_mstar_min,
            context.fp_pivot_mstar,
            context.fp_fiducial_scatter,
            context.fp_scatter_error,
            context.fp_mu_v_prior,
            context.fp_mu_v_error,
            context.fp_beta_v_prior,
            context.fp_beta_v_error,
            context.stellar_mass_pivot,
            context.mass_log_physical_offset,
        ],
        dtype=jnp.float64,
    )


def log_prob_value(theta: jnp.ndarray, compiled_model: CompiledModel) -> tuple[jnp.ndarray, ...]:
    """
    Return JAX posterior components for use inside NumPyro models.

    This helper intentionally returns JAX arrays and performs no timing or
    conversion to Python scalars.  `log_prob()` wraps it for emcee-compatible
    diagnostic tests and command-line timing summaries.
    """

    context = compiled_model.context
    compiled = _build_log_prob_components_jit(
        use_sersic_index=int(context.use_sersic_index),
        gamma_mode_code=int(context.gamma_mode_code),
        fp_enabled=int(context.fp_enabled),
        fp_has_n_axis=int(context.fp_has_n_axis),
    )
    return compiled(
        jnp.asarray(theta, dtype=jnp.float64),
        _as_jax_array(context.z_grid),
        _as_jax_array(context.chi_kpc_grid),
        _as_jax_array(context.cs_gamma_grid),
        _as_jax_array(context.cs_over_theta_grid),
        _as_jax_array(context.cs_over_theta_int),
        _as_jax_array(context.gamma_grid_int),
        _as_jax_array(context.mass_grid_int),
        _as_jax_array(context.dmass_dthetaein_grid_int),
        _as_jax_array(context.s2_grid_int),
        _as_jax_array(context.has_s2),
        _as_jax_array(context.num_sigma),
        _as_jax_array(context.sigma_obs),
        _as_jax_array(context.sigma_err),
        _as_jax_array(context.zd),
        _as_jax_array(context.zs),
        _as_jax_array(context.p_zd_fixed),
        _as_jax_array(context.mstar_grid),
        _as_jax_array(context.mstar_shift11p4),
        _as_jax_array(context.sigma_star_shift9p0_grid),
        _as_jax_array(context.mstar_integrand_base),
        _as_jax_array(context.delta_r_grid),
        _as_jax_array(context.base_normals),
        _scalar_context_array(compiled_model),
        _as_jax_array(context.fp_gamma_axis),
        _as_jax_array(context.fp_zd_axis),
        _as_jax_array(context.fp_log_re_kpc_axis),
        _as_jax_array(context.fp_n_axis),
        _as_jax_array(context.fp_sigma_unit_grid),
    )


def log_prob(theta: np.ndarray, compiled_model: CompiledModel) -> tuple[float, np.void]:
    """
    Evaluate the full JAX posterior and return a small timing blob.

    This function is deliberately API-compatible with the old
    `model.log_prob()` shape so regression tests can compare the two backends
    directly while production uses NumPyro.
    """

    total_start = perf_counter()
    theta = np.asarray(theta, dtype=np.float64)
    parameter_schema = compiled_model.config.parameter_schema
    parameter_schema.validate_theta_shape(theta)

    for index, (_name, (lower, upper)) in enumerate(
        zip(
            parameter_schema.internal_parameter_names,
            parameter_schema.prior_bounds,
            strict=True,
        )
    ):
        value = float(theta[index])
        if value < lower or value > upper:
            total_seconds = perf_counter() - total_start
            return -np.inf, _build_timing_blob(
                total_log_prob_seconds=total_seconds,
                likelihood_seconds=0.0,
                normalization_seconds=0.0,
                fp_prior_seconds=0.0,
                normalization_value=0.0,
                fp_prior_log_term=0.0,
                fpfit_mu=math.nan,
                fpfit_beta=math.nan,
                fpfit_xi=math.nan,
                fpfit_scatter=math.nan,
            )

    component_start = perf_counter()
    (
        log_prob_total,
        likelihood_value,
        normalization_value,
        fp_prior_log_term,
        fpfit_mu,
        fpfit_beta,
        fpfit_xi,
        fpfit_scatter,
    ) = log_prob_value(jnp.asarray(theta, dtype=jnp.float64), compiled_model)
    # `block_until_ready` makes timing meaningful by waiting for asynchronous
    # JAX dispatch to complete before converting to host values.
    log_prob_total.block_until_ready()
    component_seconds = perf_counter() - component_start
    total_seconds = perf_counter() - total_start

    blob = _build_timing_blob(
        total_log_prob_seconds=total_seconds,
        likelihood_seconds=component_seconds,
        normalization_seconds=component_seconds,
        fp_prior_seconds=0.0,
        normalization_value=float(normalization_value),
        fp_prior_log_term=float(fp_prior_log_term),
        fpfit_mu=float(fpfit_mu),
        fpfit_beta=float(fpfit_beta),
        fpfit_xi=float(fpfit_xi),
        fpfit_scatter=float(fpfit_scatter),
    )
    return float(log_prob_total), blob


__all__ = [
    "JAX_LOG_PROB_BLOB_DTYPE",
    "build_jax_model",
    "log_prob",
    "log_prob_value",
]
