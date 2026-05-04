"""
Current CMASS lens-population model.

This module owns the scientific contract that used to be spread across
``config.py``, ``parameter_schema.py``, and ``jax_model.py``:

- how CMASS model components are named in YAML
- which mass-aperture convention those components imply
- how the sampled theta vector is ordered
- how gamma-distribution variants change that theta vector
- how the CMASS likelihood, selection normalization, and FP summary are
  evaluated in JAX

The common sampler and backend layers call this module through
``ModelDefinition`` instead of hard-coding CMASS assumptions.  That is the
boundary future models, including Sonnenfeld 2024, will implement separately.
"""

from __future__ import annotations

import math
from functools import lru_cache
from time import perf_counter

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from ..jax_backend.context_builder import build_compiled_context
from ..jax_backend.primitives import (
    LOG10_2PI,
    LOG10_4,
    as_jax_array as _as_jax_array,
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
from ..mass_definition import H_UNITS_V1, LEGACY_FIXED_KPC, MassDefinition, get_mass_definition
from ..model_interfaces import ModelDefinition
from ..parallel import resolve_parallelism
from ..parameter_schema import ParameterSchema
from ..types import CompiledModel, RuntimeConfig


GAMMA_MODE_DEPENDENT_CODE = 0
GAMMA_MODE_INDEPENDENT_CODE = 1
GAMMA_MODE_SIGMA_STAR_DEPENDENT_CODE = 2

GAMMA_DISTRIBUTION_DEPENDENT = "dependent"
GAMMA_DISTRIBUTION_INDEPENDENT = "independent"
GAMMA_DISTRIBUTION_SIGMA_STAR_DEPENDENT = "sigma_star_dependent"

MASS_COMPONENT_M5 = "m5"
MASS_COMPONENT_M10 = "m10"
MASS_COMPONENT_M5_HINVKPC = "m5_hinvkpc"
MASS_COMPONENT_M10_HINVKPC = "m10_hinvkpc"

DEFAULT_COMPONENTS: dict[str, str] = {
    "mass_definition": MASS_COMPONENT_M5,
    "gamma_distribution": GAMMA_DISTRIBUTION_DEPENDENT,
}

SUPPORTED_MASS_COMPONENTS = {
    MASS_COMPONENT_M5,
    MASS_COMPONENT_M10,
    MASS_COMPONENT_M5_HINVKPC,
    MASS_COMPONENT_M10_HINVKPC,
}
SUPPORTED_GAMMA_DISTRIBUTIONS = {
    GAMMA_DISTRIBUTION_DEPENDENT,
    GAMMA_DISTRIBUTION_INDEPENDENT,
    GAMMA_DISTRIBUTION_SIGMA_STAR_DEPENDENT,
}

INTERNAL_MASS_PARAMETER_NAMES: tuple[str, ...] = (
    "mu5_0",
    "beta5",
    "xi5",
    "sigma5",
)

DEPENDENT_GAMMA_PARAMETER_NAMES: tuple[str, ...] = (
    "mu_gamma_0",
    "beta_gamma",
    "xi_gamma",
    "sigma_gamma",
)

INDEPENDENT_GAMMA_PARAMETER_NAMES: tuple[str, ...] = (
    "mu_gamma_0",
    "sigma_gamma",
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

DEFAULT_BOX_PRIOR_BOUNDS_BY_INTERNAL_NAME: dict[str, tuple[float, float]] = {
    "mu5_0": (9.0, 12.0),
    "beta5": (-3.0, 3.0),
    "xi5": (-3.0, 3.0),
    "sigma5": (1.0e-2, 0.2),
    "mu_gamma_0": (1.5, 2.5),
    "beta_gamma": (-3.0, 3.0),
    "xi_gamma": (-3.0, 3.0),
    "beta_sigma_star_gamma": (-3.0, 3.0),
    "sigma_gamma": (0.0, 0.5),
    "mu_zs": (1.0, 3.0),
    "sigma_zs": (0.0, 2.0),
    "theta0": (0.0, 3.0),
    "loga": (-1.0, 3.0),
}

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


def normalize_components(raw_components: dict[str, str] | None) -> dict[str, str]:
    """
    Validate and normalize the CMASS component selection.

    Only two CMASS components are currently model-defining:
    ``mass_definition`` selects the aperture/naming family, while
    ``gamma_distribution`` selects the conditional gamma parameterization.
    Profile, unit convention, FP prior, and source data remain global runtime
    config because they are shared infrastructure rather than model variants.
    """

    if raw_components is None:
        raise ValueError(
            "Config section 'model.components' is required for cmass_current "
            "and must include mass_definition and gamma_distribution."
        )
    unexpected = sorted(set(raw_components).difference(DEFAULT_COMPONENTS))
    if unexpected:
        raise ValueError(
            "Config section 'model.components' contains unsupported CMASS "
            f"component keys: {', '.join(unexpected)}."
        )
    missing = sorted(set(DEFAULT_COMPONENTS).difference(raw_components))
    if missing:
        raise ValueError(
            "Config section 'model.components' is missing required CMASS "
            f"component keys: {', '.join(missing)}."
        )
    components = {key: str(raw_components[key]) for key in DEFAULT_COMPONENTS}

    if components["mass_definition"] not in SUPPORTED_MASS_COMPONENTS:
        raise ValueError(
            "Unsupported CMASS mass_definition component "
            f"'{components['mass_definition']}'. Expected one of: "
            f"{', '.join(sorted(SUPPORTED_MASS_COMPONENTS))}."
        )
    if components["gamma_distribution"] not in SUPPORTED_GAMMA_DISTRIBUTIONS:
        raise ValueError(
            "Unsupported CMASS gamma_distribution component "
            f"'{components['gamma_distribution']}'. Expected one of: "
            f"{', '.join(sorted(SUPPORTED_GAMMA_DISTRIBUTIONS))}."
        )
    return components


def resolve_mass_definition(components: dict[str, str], unit_convention: str) -> MassDefinition:
    """
    Convert the CMASS mass component into the convention-aware MassDefinition.

    The component name is intentionally explicit about h-dependent apertures.
    This catches accidental combinations such as ``unit_convention:
    h_units_v1`` with ``mass_definition: m5`` before any HDF5 file is opened.
    """

    component = components["mass_definition"]
    expected_unit = H_UNITS_V1 if component.endswith("_hinvkpc") else LEGACY_FIXED_KPC
    if expected_unit != unit_convention:
        raise ValueError(
            "CMASS mass_definition component "
            f"'{component}' requires unit_convention '{expected_unit}', "
            f"but the config declares '{unit_convention}'."
        )
    aperture = 10 if component.startswith("m10") else 5
    return get_mass_definition(aperture, unit_convention=unit_convention)


def _gamma_distribution_code(gamma_distribution: str) -> int:
    """Return the compact static code consumed by compiled CMASS kernels."""

    if gamma_distribution == GAMMA_DISTRIBUTION_DEPENDENT:
        return GAMMA_MODE_DEPENDENT_CODE
    if gamma_distribution == GAMMA_DISTRIBUTION_INDEPENDENT:
        return GAMMA_MODE_INDEPENDENT_CODE
    if gamma_distribution == GAMMA_DISTRIBUTION_SIGMA_STAR_DEPENDENT:
        return GAMMA_MODE_SIGMA_STAR_DEPENDENT_CODE
    raise ValueError(f"Unsupported CMASS gamma_distribution component '{gamma_distribution}'.")


def _parameter_name_contract(
    gamma_distribution: str,
    mass_definition: MassDefinition,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """
    Return internal/public parameter-name families for one CMASS variant.

    The internal mass slots stay named after the historical 5-kpc family so the
    retained legacy oracle and migrated JAX kernel can share one formula.  The
    public names come from ``MassDefinition`` and carry the selected aperture
    and h-unit convention.
    """

    if gamma_distribution == GAMMA_DISTRIBUTION_DEPENDENT:
        gamma_names = DEPENDENT_GAMMA_PARAMETER_NAMES
    elif gamma_distribution == GAMMA_DISTRIBUTION_INDEPENDENT:
        gamma_names = INDEPENDENT_GAMMA_PARAMETER_NAMES
    elif gamma_distribution == GAMMA_DISTRIBUTION_SIGMA_STAR_DEPENDENT:
        gamma_names = SIGMA_STAR_DEPENDENT_GAMMA_PARAMETER_NAMES
    else:
        raise ValueError(f"Unsupported CMASS gamma_distribution component '{gamma_distribution}'.")

    return (
        INTERNAL_MASS_PARAMETER_NAMES + gamma_names + TAIL_PARAMETER_NAMES,
        mass_definition.public_parameter_names + gamma_names + TAIL_PARAMETER_NAMES,
    )


def default_public_box_prior(
    *,
    components: dict[str, str],
    mass_definition: MassDefinition,
) -> dict[str, list[float]]:
    """Expose the CMASS default box prior under the active public names."""

    internal_names, public_names = _parameter_name_contract(
        gamma_distribution=components["gamma_distribution"],
        mass_definition=mass_definition,
    )
    return {
        public_name: [
            float(DEFAULT_BOX_PRIOR_BOUNDS_BY_INTERNAL_NAME[internal_name][0]),
            float(DEFAULT_BOX_PRIOR_BOUNDS_BY_INTERNAL_NAME[internal_name][1]),
        ]
        for internal_name, public_name in zip(internal_names, public_names, strict=True)
    }


def build_parameter_schema(
    *,
    components: dict[str, str],
    mass_definition: MassDefinition,
    public_box_prior: dict[str, list[float]] | None,
) -> ParameterSchema:
    """
    Build the CMASS-specific sampled-parameter schema.

    The generic ``ParameterSchema`` validates mappings and bounds.  This model
    module decides which parameters exist for each gamma-distribution variant.
    """

    gamma_distribution = components["gamma_distribution"]
    internal_names, public_names = _parameter_name_contract(
        gamma_distribution=gamma_distribution,
        mass_definition=mass_definition,
    )
    template_schema = ParameterSchema(
        model_name="cmass_current",
        model_component_key=gamma_distribution,
        internal_parameter_names=internal_names,
        public_parameter_names=public_names,
        prior_bounds=tuple((0.0, 0.0) for _ in internal_names),
        static_codes={"gamma_mode": _gamma_distribution_code(gamma_distribution)},
        model_metadata={
            "gamma_distribution": gamma_distribution,
            "mass_definition": mass_definition.label,
        },
    )
    normalized_prior_bounds = template_schema.normalize_public_box_prior(
        public_box_prior
        if public_box_prior is not None
        else default_public_box_prior(
            components=components,
            mass_definition=mass_definition,
        )
    )
    return ParameterSchema(
        model_name="cmass_current",
        model_component_key=gamma_distribution,
        internal_parameter_names=internal_names,
        public_parameter_names=public_names,
        prior_bounds=normalized_prior_bounds,
        static_codes={"gamma_mode": _gamma_distribution_code(gamma_distribution)},
        model_metadata={
            "gamma_distribution": gamma_distribution,
            "mass_definition": mass_definition.label,
        },
    )


def build_compiled_jax_model(runtime_config: RuntimeConfig) -> CompiledModel:
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


def get_model_definition() -> ModelDefinition:
    """
    Return the registry definition for the current CMASS model.

    The callables are kept in one object so config parsing, context building,
    NumPyro sampling, and benchmark scripts all resolve the same model boundary
    instead of importing CMASS implementation details directly.
    """

    return ModelDefinition(
        name="cmass_current",
        default_components=dict(DEFAULT_COMPONENTS),
        normalize_components=normalize_components,
        resolve_mass_definition=resolve_mass_definition,
        build_parameter_schema=build_parameter_schema,
        build_compiled_model=build_compiled_jax_model,
        log_prob_value=log_prob_value,
        log_prob=log_prob,
    )


__all__ = [
    "DEFAULT_COMPONENTS",
    "GAMMA_DISTRIBUTION_DEPENDENT",
    "GAMMA_DISTRIBUTION_INDEPENDENT",
    "GAMMA_DISTRIBUTION_SIGMA_STAR_DEPENDENT",
    "JAX_LOG_PROB_BLOB_DTYPE",
    "build_compiled_jax_model",
    "build_parameter_schema",
    "default_public_box_prior",
    "get_model_definition",
    "log_prob",
    "log_prob_value",
    "normalize_components",
    "resolve_mass_definition",
]
