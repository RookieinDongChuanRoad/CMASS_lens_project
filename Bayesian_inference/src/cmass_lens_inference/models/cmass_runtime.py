"""
Runtime adapter for the default CMASS model.

This module is the thin engineering boundary that still remains before a fully
generic data-loading layer exists.  It performs two jobs:

1. build the validated CMASS NumPy source context from the canonical inference
   dataset;
2. declare how that context is packed into the JAX context consumed by
   `models.cmass`.

The scientific equations stay in `cmass.py`.  The generic backend consumes this
declaration and handles JAX array conversion, scalar packing, static flags, and
`CompiledModel` construction.
"""

from __future__ import annotations

import math

import numpy as np

from ..canonical_dataset import (
    CAPABILITY_LENSING_CROSS_SECTION_THETA_GAMMA_V1,
    CAPABILITY_LENSING_MASS_GRIDS_V1,
    CAPABILITY_LENS_OBSERVATIONS_V1,
    CAPABILITY_VELOCITY_DISPERSION_FP_WITHIN_RE_V1,
    CanonicalInferenceDataset,
    CanonicalSigmaGrid,
    load_canonical_inference_dataset,
)
from ..compiled_context import build_random_basis
from ..cosmology import FlatLambdaCDM
from ..model_interfaces import (
    CompiledContextBundle,
    ContextArraySpec,
    ContextScalarSpec,
    DataSpec,
    ModelRuntimeAdapter,
    StaticContextSpec,
)
from ..profiles import build_profile_spec
from ..types import RuntimeConfig
from .cmass_context import CMASSJaxContext, CMASSModelContext

LOG10_2PI = math.log10(2.0 * math.pi)


def build_context_bundle(runtime_config: RuntimeConfig) -> CompiledContextBundle:
    """
    Build the CMASS source-context bundle for the generic backend.

    Production inference is canonical-only: raw observation/cross-section
    products must be normalized by the data-preparation step before this
    adapter is called.  Keeping the guard here gives programmatic callers the
    same clear failure as the YAML parser if they construct `RuntimeConfig`
    manually.
    """

    if runtime_config.data.inference_dataset_path is None:
        raise ValueError(
            "The CMASS runtime requires data.inference_dataset_path. "
            "Raw observation/cross-section inputs are only supported by legacy "
            "oracle utilities outside the production inference path."
        )
    return _build_context_bundle_from_canonical_dataset(runtime_config)


def _canonical_required_capabilities(runtime_config: RuntimeConfig) -> tuple[str, ...]:
    """
    Return the canonical capabilities required by the CMASS runtime.

    FP-within-Re is only required when the optional FP prior is enabled.  The
    lens-level S2 capability is validated from the `num_sigma`/`has_s2` masks,
    so sigma-free datasets do not need to carry a physically meaningful S2
    block.
    """

    capabilities = [
        CAPABILITY_LENS_OBSERVATIONS_V1,
        CAPABILITY_LENSING_MASS_GRIDS_V1,
        CAPABILITY_LENSING_CROSS_SECTION_THETA_GAMMA_V1,
    ]
    if runtime_config.fp_prior.enabled:
        capabilities.append(CAPABILITY_VELOCITY_DISPERSION_FP_WITHIN_RE_V1)
    return tuple(capabilities)


def _shared_mass_gamma_axis(dataset: CanonicalInferenceDataset) -> np.ndarray:
    """Return one representative mass-grid gamma axis from canonical input."""

    gamma_grid = np.asarray(dataset.mass_grids.gamma_grid, dtype=np.float64)
    if gamma_grid.ndim == 1:
        return gamma_grid
    return gamma_grid[0]


def _lens_gamma_axis(dataset: CanonicalInferenceDataset, lens_index: int) -> np.ndarray:
    """Return the gamma axis for one lens, supporting 1D or per-lens storage."""

    gamma_grid = np.asarray(dataset.mass_grids.gamma_grid, dtype=np.float64)
    if gamma_grid.ndim == 1:
        return gamma_grid
    return gamma_grid[lens_index]


def _normalize_sigma_grid(
    sigma_grid: CanonicalSigmaGrid | None,
    *,
    profile_fixed_n: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Normalize optional canonical sigma grids to the JAX interpolation shape.

    CMASS hooks expect `(gamma, zd, logRe, n)` grids.  Canonical sigma blocks may
    omit redshift or Sersic-index axes when the physical table is independent of
    that coordinate, so this helper injects singleton compatibility axes.
    """

    if sigma_grid is None:
        return (
            np.zeros(1, dtype=np.float64),
            np.zeros(1, dtype=np.float64),
            np.zeros(1, dtype=np.float64),
            np.zeros(1, dtype=np.float64),
            np.zeros((1, 1, 1, 1), dtype=np.float64),
            0,
        )

    gamma_axis = np.asarray(sigma_grid.gamma_axis, dtype=np.float64)
    zd_axis = np.asarray(sigma_grid.zd_axis, dtype=np.float64)
    log_re_axis = np.asarray(sigma_grid.log_re_axis, dtype=np.float64)
    n_axis = np.asarray(sigma_grid.n_axis, dtype=np.float64)
    values = np.asarray(sigma_grid.sigma_unit_grid, dtype=np.float64)

    if values.ndim == 2:
        values = values[:, None, :, None]
        has_n_axis = 0
    elif values.ndim == 3:
        if values.shape[1] == log_re_axis.size:
            values = values[:, None, :, :]
            has_n_axis = 1
        else:
            values = values[..., None]
            has_n_axis = 0
    elif values.ndim == 4:
        has_n_axis = 1
    else:
        raise ValueError(f"Unsupported canonical sigma grid ndim={values.ndim}.")

    if not has_n_axis:
        n_axis = np.asarray([profile_fixed_n if profile_fixed_n is not None else 4.0], dtype=np.float64)
    return gamma_axis, zd_axis, log_re_axis, n_axis, values, has_n_axis


def _build_context_bundle_from_canonical_dataset(runtime_config: RuntimeConfig) -> CompiledContextBundle:
    """
    Build the CMASS source context from a canonical inference dataset.

    This path assumes all raw-file alias handling and unit conversion has
    already happened during data preparation.  It only performs model-specific
    deterministic preprocessing: quadrature grids, stellar-mass integrand
    factors, hunit-aware population constants, and random-basis generation.
    """

    profile = build_profile_spec(runtime_config.profile.name)
    dataset = load_canonical_inference_dataset(
        runtime_config.data.inference_dataset_path,
        expected_unit_convention=runtime_config.unit_convention,
        expected_h_ref=runtime_config.h_ref,
        expected_profile_name=profile.name,
        expected_mass_definition_label=runtime_config.mass_definition.label,
        required_capabilities=_canonical_required_capabilities(runtime_config),
    )
    cosmology = FlatLambdaCDM(
        h0=runtime_config.cosmology.h0,
        omega_m=runtime_config.cosmology.omega_m,
    )
    random_basis = build_random_basis(
        runtime_config.integration.normalization_samples,
        runtime_config.sampling.random_seed,
    )

    n_lens = len(dataset.lenses.lens_id)
    n_gamma = runtime_config.integration.gamma_points
    n_mstar = runtime_config.integration.mstar_points
    source_gamma_axis = _shared_mass_gamma_axis(dataset)
    gamma_grid_int = np.linspace(
        float(source_gamma_axis[0]),
        float(source_gamma_axis[-1]),
        n_gamma,
        dtype=np.float64,
    )
    mass_grid_int = np.zeros((n_lens, n_gamma), dtype=np.float64)
    dmass_dthetaein_grid_int = np.zeros((n_lens, n_gamma), dtype=np.float64)
    s2_grid_int = np.zeros((n_lens, n_gamma), dtype=np.float64)
    for lens_index in range(n_lens):
        lens_gamma_axis = _lens_gamma_axis(dataset, lens_index)
        mass_grid_int[lens_index] = np.interp(
            gamma_grid_int,
            lens_gamma_axis,
            dataset.mass_grids.log_enclosed_mass_grid[lens_index],
        )
        dmass_dthetaein_grid_int[lens_index] = np.interp(
            gamma_grid_int,
            lens_gamma_axis,
            dataset.mass_grids.dmass_dthetaein_grid[lens_index],
        )
        s2_grid_int[lens_index] = np.interp(
            gamma_grid_int,
            lens_gamma_axis,
            dataset.mass_grids.s2_grid[lens_index],
        )

    log10_h_ref = math.log10(runtime_config.h_ref)
    if runtime_config.unit_convention == "h_units_v1":
        stellar_mass_pivot = 11.4 + 2.0 * log10_h_ref
        mass_function_loc = profile.mass_function_loc + 2.0 * log10_h_ref
        mu_r0 = profile.mu_r0 + log10_h_ref
    else:
        stellar_mass_pivot = 11.4
        mass_function_loc = profile.mass_function_loc
        mu_r0 = profile.mu_r0

    sqrt2 = math.sqrt(2.0)
    sqrt2pi = math.sqrt(2.0 * math.pi)
    zd = np.asarray(dataset.lenses.z_d, dtype=np.float64)
    zs = np.asarray(dataset.lenses.z_s, dtype=np.float64)
    p_zd_fixed = np.exp(-0.5 * ((zd - 0.558) / 0.085) ** 2) / (0.085 * sqrt2pi)

    mstar_grid = np.zeros((n_lens, n_mstar), dtype=np.float64)
    mstar_shift11p4 = np.zeros((n_lens, n_mstar), dtype=np.float64)
    sigma_star_shift9p0_grid = np.zeros((n_lens, n_mstar), dtype=np.float64)
    mstar_integrand_base = np.zeros((n_lens, n_mstar), dtype=np.float64)
    delta_r_grid = np.zeros((n_lens, n_mstar), dtype=np.float64)
    for lens_index in range(n_lens):
        observed_mstar = float(dataset.lenses.log_mstar_obs[lens_index])
        observed_mstar_err = float(dataset.lenses.log_mstar_err[lens_index])
        observed_log_re = float(dataset.lenses.log_re_obs[lens_index])
        n_value = profile.fixed_n if profile.fixed_n is not None else max(float(dataset.lenses.n_obs[lens_index]), 1.0e-8)
        mstar_grid[lens_index] = np.linspace(
            observed_mstar - 5.0 * observed_mstar_err,
            observed_mstar + 5.0 * observed_mstar_err,
            n_mstar,
            dtype=np.float64,
        )
        for mstar_index, mstar in enumerate(mstar_grid[lens_index]):
            shift = mstar - stellar_mass_pivot
            mu_r_value = mu_r0 + profile.beta_r * shift
            if profile.nu_r is not None:
                mu_r_value += profile.nu_r * (math.log10(max(n_value, 1.0e-12)) - math.log10(4.0))
            delta_r = observed_log_re - mu_r_value

            p_mobs = np.exp(-0.5 * ((observed_mstar - mstar) / observed_mstar_err) ** 2)
            p_mobs /= observed_mstar_err * sqrt2pi
            t = (mstar - mass_function_loc) / profile.mass_function_scale
            p_s = 2.0 * np.exp(-0.5 * t * t) / sqrt2pi
            p_s *= 0.5 * (1.0 + math.erf(profile.mass_function_alpha * t / sqrt2))
            p_s /= profile.mass_function_scale
            p_r = np.exp(-0.5 * ((observed_log_re - mu_r_value) / profile.sigma_r) ** 2)
            p_r /= profile.sigma_r * sqrt2pi

            mstar_shift11p4[lens_index, mstar_index] = shift
            sigma_star_shift9p0_grid[lens_index, mstar_index] = mstar - LOG10_2PI - 2.0 * observed_log_re - 9.0
            delta_r_grid[lens_index, mstar_index] = delta_r
            mstar_integrand_base[lens_index, mstar_index] = p_mobs * p_s * p_r

    fp_gamma_axis, fp_zd_axis, fp_log_re_axis, fp_n_axis, fp_sigma_grid, fp_has_n_axis = _normalize_sigma_grid(
        dataset.velocity_dispersion.fp_within_re if runtime_config.fp_prior.enabled else None,
        profile_fixed_n=profile.fixed_n,
    )
    context = CMASSModelContext(
        z_grid=np.ascontiguousarray(cosmology.z_table, dtype=np.float64),
        chi_kpc_grid=np.ascontiguousarray(cosmology.comoving_distance_table_mpc * 1000.0, dtype=np.float64),
        cs_gamma_grid=np.ascontiguousarray(dataset.cross_section.gamma_axis, dtype=np.float64),
        cs_over_theta_grid=np.zeros_like(np.asarray(dataset.cross_section.gamma_axis, dtype=np.float64)),
        cs_theta_e_axis=np.ascontiguousarray(dataset.cross_section.theta_e_axis, dtype=np.float64),
        cs_cross_section_grid=np.ascontiguousarray(dataset.cross_section.cross_section_grid, dtype=np.float64),
        cs_over_theta_int=np.zeros(n_gamma, dtype=np.float64),
        gamma_grid_int=np.ascontiguousarray(gamma_grid_int, dtype=np.float64),
        mass_grid_int=np.ascontiguousarray(mass_grid_int, dtype=np.float64),
        dmass_dthetaein_grid_int=np.ascontiguousarray(dmass_dthetaein_grid_int, dtype=np.float64),
        s2_grid_int=np.ascontiguousarray(s2_grid_int, dtype=np.float64),
        has_s2=np.ascontiguousarray(dataset.mass_grids.has_s2, dtype=np.int64),
        num_sigma=np.ascontiguousarray(dataset.lenses.num_sigma, dtype=np.int64),
        sigma_obs=np.ascontiguousarray(dataset.lenses.sigma_obs, dtype=np.float64),
        sigma_err=np.ascontiguousarray(dataset.lenses.sigma_err, dtype=np.float64),
        zd=np.ascontiguousarray(zd, dtype=np.float64),
        zs=np.ascontiguousarray(zs, dtype=np.float64),
        p_zd_fixed=np.ascontiguousarray(p_zd_fixed, dtype=np.float64),
        mstar_grid=np.ascontiguousarray(mstar_grid, dtype=np.float64),
        mstar_shift11p4=np.ascontiguousarray(mstar_shift11p4, dtype=np.float64),
        stellar_mass_pivot=float(stellar_mass_pivot),
        sigma_star_shift9p0_grid=np.ascontiguousarray(sigma_star_shift9p0_grid, dtype=np.float64),
        mstar_integrand_base=np.ascontiguousarray(mstar_integrand_base, dtype=np.float64),
        delta_r_grid=np.ascontiguousarray(delta_r_grid, dtype=np.float64),
        base_normals=random_basis.base_normals,
        mass_radius_kpc=float(runtime_config.mass_definition.physical_radius_kpc(runtime_config.h_ref)),
        mass_log_physical_offset=float(runtime_config.mass_definition.log_mass_physical_offset(runtime_config.h_ref)),
        use_sersic_index=1 if profile.uses_observed_n_in_likelihood else 0,
        n_fixed=profile.fixed_n if profile.fixed_n is not None else 4.0,
        mu_n0=profile.mu_n0 if profile.mu_n0 is not None else 0.0,
        beta_n=profile.beta_n if profile.beta_n is not None else 0.0,
        sigma_n=profile.sigma_n if profile.sigma_n is not None else 0.0,
        mass_function_loc=mass_function_loc,
        mass_function_scale=profile.mass_function_scale,
        mass_function_alpha=profile.mass_function_alpha,
        mu_r0=mu_r0,
        beta_r=profile.beta_r,
        sigma_r=profile.sigma_r,
        nu_r=profile.nu_r if profile.nu_r is not None else 0.0,
        mu_d=0.558,
        sigma_d=0.085,
        gamma_trunc_low=1.2,
        gamma_trunc_high=2.8,
        normalization_min_value=1.0e-10,
        gamma_mode_code=runtime_config.parameter_schema.gamma_mode_code,
        fp_enabled=1 if runtime_config.fp_prior.enabled else 0,
        fp_fit_mstar_min=runtime_config.fp_prior.fit_mstar_min,
        fp_pivot_mstar=runtime_config.fp_prior.pivot_mstar,
        fp_fiducial_scatter=runtime_config.fp_prior.fiducial_scatter,
        fp_scatter_error=runtime_config.fp_prior.scatter_error,
        fp_mu_v_prior=runtime_config.fp_prior.mu_v_prior,
        fp_mu_v_error=runtime_config.fp_prior.mu_v_error,
        fp_beta_v_prior=runtime_config.fp_prior.beta_v_prior,
        fp_beta_v_error=runtime_config.fp_prior.beta_v_error,
        fp_gamma_axis=np.ascontiguousarray(fp_gamma_axis, dtype=np.float64),
        fp_zd_axis=np.ascontiguousarray(fp_zd_axis, dtype=np.float64),
        fp_log_re_kpc_axis=np.ascontiguousarray(fp_log_re_axis, dtype=np.float64),
        fp_n_axis=np.ascontiguousarray(fp_n_axis, dtype=np.float64),
        fp_sigma_unit_grid=np.ascontiguousarray(fp_sigma_grid, dtype=np.float64),
        fp_has_n_axis=int(fp_has_n_axis),
    )
    return CompiledContextBundle(
        context=context,
        profile=profile,
        cross_section_grid=dataset.cross_section,
        cosmology=cosmology,
        random_basis=random_basis,
        observations=(),
        metadata={
            "canonical_dataset_path": str(dataset.path),
            "canonical_schema_version": dataset.metadata.schema_version,
            "canonical_capabilities": tuple(sorted(dataset.metadata.capabilities)),
            "canonical_profile_name": dataset.metadata.profile_name,
            "canonical_mass_definition_label": dataset.metadata.mass_definition_label,
        },
    )


def get_data_spec() -> DataSpec:
    """
    Return the CMASS context-packing declaration.

    The scalar field order intentionally matches the previous hand-written
    `_scalar_context_array` implementation.  CMASS hooks index this array by
    position inside JIT-compiled code, so changing the order would be a
    scientific behavior change and must be covered by explicit tests.
    """

    return DataSpec(
        jax_context_type=CMASSJaxContext,
        array_fields=(
            ContextArraySpec("z_grid"),
            ContextArraySpec("chi_kpc_grid"),
            ContextArraySpec("cs_gamma_grid"),
            ContextArraySpec("cs_over_theta_grid"),
            ContextArraySpec("cs_theta_e_axis"),
            ContextArraySpec("cs_cross_section_grid"),
            ContextArraySpec("cs_over_theta_int"),
            ContextArraySpec("gamma_grid_int"),
            ContextArraySpec("mass_grid_int"),
            ContextArraySpec("dmass_dthetaein_grid_int"),
            ContextArraySpec("s2_grid_int"),
            ContextArraySpec("has_s2"),
            ContextArraySpec("num_sigma"),
            ContextArraySpec("sigma_obs"),
            ContextArraySpec("sigma_err"),
            ContextArraySpec("zd"),
            ContextArraySpec("zs"),
            ContextArraySpec("p_zd_fixed"),
            ContextArraySpec("mstar_grid"),
            ContextArraySpec("mstar_shift11p4"),
            ContextArraySpec("sigma_star_shift9p0_grid"),
            ContextArraySpec("mstar_integrand_base"),
            ContextArraySpec("delta_r_grid"),
            ContextArraySpec("base_normals"),
            ContextArraySpec("fp_gamma_axis"),
            ContextArraySpec("fp_zd_axis"),
            ContextArraySpec("fp_log_re_kpc_axis"),
            ContextArraySpec("fp_n_axis"),
            ContextArraySpec("fp_sigma_unit_grid"),
        ),
        scalar_fields=(
            ContextScalarSpec("mass_radius_kpc"),
            ContextScalarSpec("n_fixed"),
            ContextScalarSpec("mu_n0"),
            ContextScalarSpec("beta_n"),
            ContextScalarSpec("sigma_n"),
            ContextScalarSpec("mass_function_loc"),
            ContextScalarSpec("mass_function_scale"),
            ContextScalarSpec("mass_function_alpha"),
            ContextScalarSpec("mu_r0"),
            ContextScalarSpec("beta_r"),
            ContextScalarSpec("sigma_r"),
            ContextScalarSpec("nu_r"),
            ContextScalarSpec("mu_d"),
            ContextScalarSpec("sigma_d"),
            ContextScalarSpec("gamma_trunc_low"),
            ContextScalarSpec("gamma_trunc_high"),
            ContextScalarSpec("normalization_min_value"),
            ContextScalarSpec("fp_fit_mstar_min"),
            ContextScalarSpec("fp_pivot_mstar"),
            ContextScalarSpec("fp_fiducial_scatter"),
            ContextScalarSpec("fp_scatter_error"),
            ContextScalarSpec("fp_mu_v_prior"),
            ContextScalarSpec("fp_mu_v_error"),
            ContextScalarSpec("fp_beta_v_prior"),
            ContextScalarSpec("fp_beta_v_error"),
            ContextScalarSpec("stellar_mass_pivot"),
            ContextScalarSpec("mass_log_physical_offset"),
        ),
        static_fields=(
            StaticContextSpec("use_sersic_index"),
            StaticContextSpec("fp_enabled"),
            StaticContextSpec("fp_has_n_axis"),
        ),
        normalization_samples_field="base_normals",
        normalization_min_value_field="normalization_min_value",
    )


def get_runtime_adapter() -> ModelRuntimeAdapter:
    """
    Return the runtime adapter paired with `models.cmass.get_model_spec()`.

    No JAX packing functions are exposed here.  The generic backend derives
    them from `DataSpec`, which is the point of this refactor.
    """

    return ModelRuntimeAdapter(
        build_context_bundle=build_context_bundle,
        data_spec=get_data_spec(),
    )


__all__ = ["build_context_bundle", "get_data_spec", "get_runtime_adapter"]
