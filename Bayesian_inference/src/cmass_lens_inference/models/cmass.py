"""Default CMASS lens-population model."""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from ..canonical_dataset import (
    CAPABILITY_LENSING_CROSS_SECTION_THETA_GAMMA_V1,
    CAPABILITY_LENSING_MASS_GRIDS_V1,
    CAPABILITY_LENS_OBSERVATIONS_V1,
    CAPABILITY_VELOCITY_DISPERSION_FP_WITHIN_RE_V1,
    CAPABILITY_VELOCITY_DISPERSION_PER_LENS_S2_V1,
)
from ..jax_backend.primitives import (
    LOG10_2PI,
    LOG10_4,
    interp_cross_section_theta_gamma as _interp_cross_section_theta_gamma,
    interp_sigma_unit_clip_scalar as _interp_sigma_unit_clip_scalar,
    normal_pdf as _normal_pdf,
    phi_standard as _phi_standard,
    skewnorm_sample as _skewnorm_sample,
    theta_ein_arcsec as _theta_ein_arcsec,
    trapezoid_last_axis as _trapezoid_last_axis,
    truncated_normal_pdf_nonneg as _truncated_normal_pdf_nonneg,
    truncnorm_sample as _truncnorm_sample,
)
from ..jax_backend.selection import sigmoid_find_probability as _p_find
from ..mass_definition import H_UNITS_V1
from ..model_interfaces import ModelSpec, ParameterSpec
from .cmass_context import CMASSJaxContext


GAMMA_MODE_SIGMA_STAR_DEPENDENT_CODE = 2
GAMMA_DISTRIBUTION_SIGMA_STAR_DEPENDENT = "sigma_star_dependent"
MODEL_NAME = "cmass"
MODEL_COMPONENT_KEY = "default"
MASS_APERTURE_KPC = 5

INTERNAL_MASS_PARAMETER_NAMES: tuple[str, ...] = (
    "mu5_0",
    "beta5",
    "xi5",
    "sigma5",
)

SIGMA_STAR_DEPENDENT_GAMMA_PARAMETER_NAMES: tuple[str, ...] = (
    "mu_gamma_0",
    "beta_sigma_star_gamma",
    "sigma_gamma",
)

TAIL_PARAMETER_NAMES: tuple[str, ...] = (
    "mu_zs",
    "sigma_zs",
    "theta0",
    "loga",
)
INTERNAL_PARAMETER_NAMES: tuple[str, ...] = (
    INTERNAL_MASS_PARAMETER_NAMES
    + SIGMA_STAR_DEPENDENT_GAMMA_PARAMETER_NAMES
    + TAIL_PARAMETER_NAMES
)
PUBLIC_PARAMETER_NAMES: tuple[str, ...] = (
    ("mu5h_0", "beta5h", "xi5h", "sigma5h")
    + SIGMA_STAR_DEPENDENT_GAMMA_PARAMETER_NAMES
    + TAIL_PARAMETER_NAMES
)

DEFAULT_BOX_PRIOR_BOUNDS_BY_INTERNAL_NAME: dict[str, tuple[float, float]] = {
    "mu5_0": (9.0, 12.0),
    "beta5": (-3.0, 3.0),
    "xi5": (-3.0, 3.0),
    "sigma5": (1.0e-2, 0.2),
    "mu_gamma_0": (1.5, 2.5),
    "beta_sigma_star_gamma": (-3.0, 3.0),
    "sigma_gamma": (0.0, 0.5),
    "mu_zs": (1.0, 3.0),
    "sigma_zs": (0.0, 2.0),
    "theta0": (0.0, 3.0),
    "loga": (-1.0, 3.0),
}
PARAMETER_SPECS: tuple[ParameterSpec, ...] = tuple(
    ParameterSpec(
        internal_name=internal_name,
        public_name=public_name,
        bounds=DEFAULT_BOX_PRIOR_BOUNDS_BY_INTERNAL_NAME[internal_name],
    )
    for internal_name, public_name in zip(
        INTERNAL_PARAMETER_NAMES,
        PUBLIC_PARAMETER_NAMES,
        strict=True,
    )
)

FP_OLS_SUMMARY_SIZE = 6
FP_OLS_COUNT_INDEX = 0
FP_OLS_SUM_X1_INDEX = 1
FP_OLS_SUM_X1X1_INDEX = 2
FP_OLS_SUM_Y_INDEX = 3
FP_OLS_SUM_X1Y_INDEX = 4
FP_OLS_SUM_YY_INDEX = 5


class CMASSTheta(NamedTuple):
    mu5_0: jnp.ndarray
    beta5: jnp.ndarray
    xi5: jnp.ndarray
    sigma5: jnp.ndarray
    mu_gamma_0: jnp.ndarray
    beta_sigma_star_gamma: jnp.ndarray
    sigma_gamma: jnp.ndarray
    mu_zs: jnp.ndarray
    sigma_zs: jnp.ndarray
    theta0: jnp.ndarray
    loga: jnp.ndarray


class CMASSPopulationDraw(NamedTuple):
    zd: jnp.ndarray
    mstar: jnp.ndarray
    n_value: jnp.ndarray
    re_draw: jnp.ndarray
    delta_r: jnp.ndarray
    log_enclosed_mass: jnp.ndarray
    gamma: jnp.ndarray


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


def unpack_theta(theta: jnp.ndarray) -> CMASSTheta:
    """Unpack the fixed 11D CMASS parameter vector."""

    return CMASSTheta(
        mu5_0=theta[0],
        beta5=theta[1],
        xi5=theta[2],
        sigma5=theta[3],
        mu_gamma_0=theta[4],
        beta_sigma_star_gamma=theta[5],
        sigma_gamma=theta[6],
        mu_zs=theta[7],
        sigma_zs=theta[8],
        theta0=theta[9],
        loga=theta[10],
    )


def _gamma_population_mean(
    mu_gamma_0: jnp.ndarray,
    beta_sigma_star_gamma: jnp.ndarray,
    sigma_star_shift9p0: jnp.ndarray,
) -> jnp.ndarray:
    """Conditional mean of gamma for the fixed sigma-star CMASS model."""

    return mu_gamma_0 + beta_sigma_star_gamma * sigma_star_shift9p0


def draw_population(
    theta_parts: CMASSTheta,
    nrm: jnp.ndarray,
    context: CMASSJaxContext,
    static: dict[str, int],
) -> CMASSPopulationDraw:
    """Draw one latent parent-population state from one fixed normal row."""

    scalar_context = context.scalar_context
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
    stellar_mass_pivot = scalar_context[25]
    use_sersic_index = static["use_sersic_index"]
    n_fixed = scalar_context[1]
    mu_n0 = scalar_context[2]
    beta_n = scalar_context[3]
    sigma_n = scalar_context[4]
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
    log_enclosed_mass = (
        theta_parts.mu5_0
        + theta_parts.beta5 * mstar_shift
        + theta_parts.xi5 * delta_r
        + theta_parts.sigma5 * mass_noise_column
    )

    sigma_star_shift9p0 = mstar - LOG10_2PI - 2.0 * re_draw - 9.0
    mu_gamma = _gamma_population_mean(
        theta_parts.mu_gamma_0,
        theta_parts.beta_sigma_star_gamma,
        sigma_star_shift9p0,
    )
    gamma = _truncnorm_sample(
        mu_gamma,
        theta_parts.sigma_gamma,
        gamma_trunc_low,
        gamma_trunc_high,
        nrm[7],
    )
    return CMASSPopulationDraw(
        zd=zd,
        mstar=mstar,
        n_value=n_value,
        re_draw=re_draw,
        delta_r=delta_r,
        log_enclosed_mass=log_enclosed_mass,
        gamma=gamma,
    )


def validate_theta(
    theta: jnp.ndarray,
    theta_parts: CMASSTheta,
    context: CMASSJaxContext,
    static: dict[str, int],
) -> jnp.ndarray:
    """Return the differentiable validity mask for one CMASS theta vector."""

    del context, static
    z0 = (0.0 - theta_parts.mu_zs) / theta_parts.sigma_zs
    trunc_den = 1.0 - _phi_standard(z0)
    return (
        (theta.shape[0] == len(INTERNAL_PARAMETER_NAMES))
        & (theta_parts.sigma5 > 0.0)
        & (theta_parts.sigma_gamma > 0.0)
        & (theta_parts.sigma_zs > 0.0)
        & (trunc_den > 0.0)
    )


def _selection_weight_for_source(
    theta_parts: CMASSTheta,
    draw: CMASSPopulationDraw,
    context: CMASSJaxContext,
    zs: jnp.ndarray,
    mass_radius_kpc: jnp.ndarray,
    mass_log_physical_offset: jnp.ndarray,
) -> jnp.ndarray:
    """Shared selection expression once source redshift is known."""

    z0 = (0.0 - theta_parts.mu_zs) / theta_parts.sigma_zs
    inv_trunc_den = 1.0 / (1.0 - _phi_standard(z0))
    theta_e = _theta_ein_arcsec(
        draw.zd,
        zs,
        draw.log_enclosed_mass,
        draw.gamma,
        context.z_grid,
        context.chi_kpc_grid,
        mass_radius_kpc,
        mass_log_physical_offset,
    )
    cross_section = _interp_cross_section_theta_gamma(
        theta_e,
        draw.gamma,
        context.cs_theta_e_axis,
        context.cs_gamma_grid,
        context.cs_cross_section_grid,
    )
    weight = inv_trunc_den * _p_find(theta_e, theta_parts.theta0, theta_parts.loga) * cross_section
    valid = (
        (draw.zd > 0.0)
        & (zs > 0.0)
        & (zs > draw.zd)
        & jnp.isfinite(draw.gamma)
        & (theta_e > 0.0)
        & (cross_section > 0.0)
    )
    return jnp.where(valid, weight, 0.0)


def selection_weight_from_normal(
    theta_parts: CMASSTheta,
    draw: CMASSPopulationDraw,
    nrm: jnp.ndarray,
    context: CMASSJaxContext,
    static: dict[str, int],
) -> jnp.ndarray:
    """Return selection weight using the source-redshift normal in the MC row."""

    del static
    scalar_context = context.scalar_context
    zs = theta_parts.mu_zs + theta_parts.sigma_zs * nrm[1]
    return _selection_weight_for_source(
        theta_parts,
        draw,
        context,
        zs,
        scalar_context[0],
        scalar_context[26],
    )


def summary_row(
    theta_parts: CMASSTheta,
    draw: CMASSPopulationDraw,
    context: CMASSJaxContext,
    static: dict[str, int],
) -> jnp.ndarray:
    """Return one FP OLS sufficient-statistics row for one MC draw."""

    scalar_context = context.scalar_context
    fp_enabled = static["fp_enabled"]
    fp_has_n_axis = static["fp_has_n_axis"]
    fp_fit_mstar_min = scalar_context[17]
    fp_pivot_mstar = scalar_context[18]
    sigma_unit = _interp_sigma_unit_clip_scalar(
        draw.gamma,
        draw.zd,
        draw.re_draw,
        draw.n_value,
        context.fp_gamma_axis,
        context.fp_zd_axis,
        context.fp_log_re_kpc_axis,
        context.fp_n_axis,
        context.fp_sigma_unit_grid,
        fp_has_n_axis,
    )
    log_sigma_model = 0.5 * (jnp.log10(sigma_unit) + draw.log_enclosed_mass)
    fp_valid = (
        (fp_enabled == 1)
        & (draw.zd > 0.0)
        & jnp.isfinite(draw.gamma)
        & (draw.mstar > fp_fit_mstar_min)
        & (sigma_unit > 0.0)
        & jnp.isfinite(sigma_unit)
        & jnp.isfinite(log_sigma_model)
    )
    x1 = draw.mstar - fp_pivot_mstar
    row = jnp.asarray(
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
    return jnp.where(fp_valid, row, jnp.zeros(FP_OLS_SUMMARY_SIZE, dtype=jnp.float64))


def lens_integrals(
    theta_parts: CMASSTheta,
    context: CMASSJaxContext,
    static: dict[str, int],
) -> jnp.ndarray:
    """Return the per-lens CMASS likelihood integrals before log reduction."""

    del static
    scalar_context = context.scalar_context
    log_enclosed_mass = context.mass_grid_int
    gamma = context.gamma_grid_int[None, :]
    jac = jnp.abs(context.dmass_dthetaein_grid_int)
    theta_e = _theta_ein_arcsec(
        context.zd[:, None],
        context.zs[:, None],
        log_enclosed_mass,
        gamma,
        context.z_grid,
        context.chi_kpc_grid,
        scalar_context[0],
        scalar_context[26],
    )
    cross_section = _interp_cross_section_theta_gamma(
        theta_e,
        gamma,
        context.cs_theta_e_axis,
        context.cs_gamma_grid,
        context.cs_cross_section_grid,
    )
    pf = _p_find(theta_e, theta_parts.theta0, theta_parts.loga)
    p_zs = _truncated_normal_pdf_nonneg(context.zs, theta_parts.mu_zs, theta_parts.sigma_zs)

    sigma_model = jnp.sqrt(jnp.maximum(context.s2_grid_int * (10.0**log_enclosed_mass), 1.0e-30))
    p_sigma_1 = _normal_pdf(context.sigma_obs[:, None, 0], sigma_model, context.sigma_err[:, None, 0])
    p_sigma_2 = _normal_pdf(context.sigma_obs[:, None, 1], sigma_model, context.sigma_err[:, None, 1])
    p_sigma = jnp.where(context.num_sigma[:, None] >= 1, p_sigma_1, 1.0)
    p_sigma = jnp.where(context.num_sigma[:, None] >= 2, p_sigma * p_sigma_2, p_sigma)
    p_sigma = jnp.where((context.num_sigma[:, None] > 0) & (context.has_s2[:, None] == 0), 0.0, p_sigma)

    mu5 = (
        theta_parts.mu5_0
        + theta_parts.beta5 * context.mstar_shift11p4
        + theta_parts.xi5 * context.delta_r_grid
    )
    mu_gamma = _gamma_population_mean(
        theta_parts.mu_gamma_0,
        theta_parts.beta_sigma_star_gamma,
        context.sigma_star_shift9p0_grid,
    )
    mstar_density = (
        context.mstar_integrand_base[:, None, :]
        * _normal_pdf(log_enclosed_mass[:, :, None], mu5[:, None, :], theta_parts.sigma5)
        * _normal_pdf(context.gamma_grid_int[None, :, None], mu_gamma[:, None, :], theta_parts.sigma_gamma)
    )
    integrated_mstar = _trapezoid_last_axis(mstar_density, context.mstar_grid[:, None, :])
    gamma_integrand = (
        integrated_mstar
        * context.p_zd_fixed[:, None]
        * p_zs[:, None]
        * pf
        * cross_section
        * jac
        * p_sigma
    )
    gamma_valid = (jac > 0.0) & (theta_e > 0.0) & (cross_section > 0.0) & (pf > 0.0) & (p_sigma > 0.0)
    gamma_integrand = jnp.where(gamma_valid, gamma_integrand, 0.0)
    return _trapezoid_last_axis(gamma_integrand, context.gamma_grid_int[None, :])


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


def extra_prior(
    fp_summary: jnp.ndarray,
    context: CMASSJaxContext,
    static: dict[str, int],
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return FP-prior contribution and diagnostics."""

    scalar_context = context.scalar_context
    return _fp_prior_value(
        fp_summary,
        fp_enabled=static["fp_enabled"],
        fp_fiducial_scatter=scalar_context[19],
        fp_scatter_error=scalar_context[20],
        fp_mu_v_prior=scalar_context[21],
        fp_mu_v_error=scalar_context[22],
        fp_beta_v_prior=scalar_context[23],
        fp_beta_v_error=scalar_context[24],
    )


def get_model_spec() -> ModelSpec:
    """
    Return the human-authored scientific specification for CMASS.

    This spec is intentionally limited to model-owned facts and formulas.  The
    registry pairs it with `cmass_runtime.get_runtime_adapter()` to build the
    lower-level backend definition used by JAX and NumPyro.
    """

    return ModelSpec(
        name=MODEL_NAME,
        component_key=MODEL_COMPONENT_KEY,
        required_unit_convention=H_UNITS_V1,
        mass_aperture_kpc=MASS_APERTURE_KPC,
        parameters=PARAMETER_SPECS,
        metadata={
            "gamma_distribution": GAMMA_DISTRIBUTION_SIGMA_STAR_DEPENDENT,
            "mass_definition": "m5_hinvkpc",
            "unit_convention": H_UNITS_V1,
        },
        required_capabilities=(
            CAPABILITY_LENS_OBSERVATIONS_V1,
            CAPABILITY_LENSING_MASS_GRIDS_V1,
            CAPABILITY_LENSING_CROSS_SECTION_THETA_GAMMA_V1,
        ),
        optional_capabilities=(
            CAPABILITY_VELOCITY_DISPERSION_PER_LENS_S2_V1,
            CAPABILITY_VELOCITY_DISPERSION_FP_WITHIN_RE_V1,
        ),
        static_codes={"gamma_mode": GAMMA_MODE_SIGMA_STAR_DEPENDENT_CODE},
        unpack_theta=unpack_theta,
        validate_theta=validate_theta,
        draw_population=draw_population,
        selection_weight=selection_weight_from_normal,
        summary_row=summary_row,
        lens_integrals=lens_integrals,
        extra_prior=extra_prior,
    )


__all__ = [
    "GAMMA_DISTRIBUTION_SIGMA_STAR_DEPENDENT",
    "PARAMETER_SPECS",
    "get_model_spec",
]
