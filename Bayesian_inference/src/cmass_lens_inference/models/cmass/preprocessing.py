"""
Deterministic preprocessing for the default CMASS model.

This module is model-specific, but sampler-agnostic.  It converts a validated
canonical inference dataset into `CMASSModelContext`, the NumPy source context
consumed by production backend kernels.

Why this boundary exists:
- `cmass.py` should stay readable as the human-authored scientific model.
- `cmass_runtime.py` should stay thin glue between registry/runtime and data.
- CMASS-only deterministic quantities such as stellar-mass quadrature weights,
  h-unit population pivots, and FP-grid adaptation still need a clear home.
"""

from __future__ import annotations

import math

import numpy as np

from ...canonical_dataset import (
    CAPABILITY_LENSING_CROSS_SECTION_THETA_GAMMA_V1,
    CAPABILITY_LENSING_MASS_GRIDS_V1,
    CAPABILITY_LENS_OBSERVATIONS_V1,
    CAPABILITY_VELOCITY_DISPERSION_FP_WITHIN_RE_V1,
    CanonicalInferenceDataset,
    load_canonical_inference_dataset,
)
from ...canonical_context import (
    canonical_dataset_metadata,
    interpolate_lensing_mass_grids,
    normalize_sigma_grid,
    shared_gamma_axis,
)
from ...compiled_context import build_random_basis
from ...cosmology import FlatLambdaCDM
from ...mass_definition import H_UNITS_V1, LEGACY_FIXED_KPC
from ...model_interfaces import CompiledContextBundle
from ...profiles import build_profile_spec
from ...types import ProfileSpec, RuntimeConfig
from .constants import (
    CMASS_GAMMA_TRUNC_HIGH,
    CMASS_GAMMA_TRUNC_LOW,
    CMASS_LENS_REDSHIFT_MEAN,
    CMASS_LENS_REDSHIFT_SCATTER,
    CMASS_NORMALIZATION_MIN_VALUE,
    CMASS_STELLAR_MASS_PIVOT,
)
from .context import CMASSModelContext

LOG10_2PI = math.log10(2.0 * math.pi)


def required_canonical_capabilities(runtime_config: RuntimeConfig) -> tuple[str, ...]:
    """
    Return the canonical capabilities required by the CMASS model.

    FP-within-Re data is required only when the optional FP prior is enabled.
    Per-lens S2 consistency is validated by the canonical reader through the
    `num_sigma` and `has_s2` arrays, so sigma-free datasets do not need an
    additional required capability here.
    """

    capabilities = [
        CAPABILITY_LENS_OBSERVATIONS_V1,
        CAPABILITY_LENSING_MASS_GRIDS_V1,
        CAPABILITY_LENSING_CROSS_SECTION_THETA_GAMMA_V1,
    ]
    if runtime_config.fp_prior.enabled:
        capabilities.append(CAPABILITY_VELOCITY_DISPERSION_FP_WITHIN_RE_V1)
    return tuple(capabilities)


def load_cmass_canonical_dataset(
    runtime_config: RuntimeConfig,
    *,
    profile: ProfileSpec | None = None,
) -> CanonicalInferenceDataset:
    """
    Load the canonical dataset with CMASS-specific expectations.

    The generic canonical reader owns schema validation.  This wrapper provides
    the model-specific capability list and expected profile/mass labels so
    `cmass_runtime.py` does not need to know which capability names CMASS uses.
    """

    if runtime_config.data.inference_dataset_path is None:
        raise ValueError("CMASS preprocessing requires data.inference_dataset_path.")

    active_profile = profile or build_profile_spec(runtime_config.profile.name)
    return load_canonical_inference_dataset(
        runtime_config.data.inference_dataset_path,
        expected_unit_convention=runtime_config.unit_convention,
        expected_h_ref=runtime_config.h_ref,
        expected_profile_name=active_profile.name,
        expected_mass_definition_label=runtime_config.mass_definition.label,
        required_capabilities=required_canonical_capabilities(runtime_config),
    )


def _hunit_population_constants(runtime_config: RuntimeConfig, profile: ProfileSpec) -> tuple[float, float, float]:
    """
    Return hunit-aware CMASS population pivots.

    The default CMASS model currently requires `h_units_v1`, but this helper
    keeps the legacy-fixed-kpc branch visible because the formulas are inherited
    from the hunit merge and tests still compare canonical behavior against
    legacy oracle contexts constructed outside production config parsing.
    """

    log10_h_ref = math.log10(runtime_config.h_ref)
    if runtime_config.unit_convention == "h_units_v1":
        return (
            CMASS_STELLAR_MASS_PIVOT + 2.0 * log10_h_ref,
            profile.mass_function_loc + 2.0 * log10_h_ref,
            profile.mu_r0 + log10_h_ref,
        )
    return CMASS_STELLAR_MASS_PIVOT, profile.mass_function_loc, profile.mu_r0


def _active_fp_prior_mass_locations(runtime_config: RuntimeConfig) -> tuple[float, float]:
    """
    Return the FP fit threshold and pivot in the active stellar-mass coordinate.

    The FP prior constants are historical physical-coordinate values: the
    published/legacy convention says "fit above log M_* = 11.0" and evaluate
    the intercept at "log M_* = 11.3".  CMASS h-unit inference, however, feeds
    the posterior kernels the active latent coordinate
    ``log10[M_*/(h^-2 Msun)]``.  A location on that translated mass axis must
    therefore move by ``2 log10(h_ref)`` before the kernel applies the cut or
    subtracts the pivot.  The prior target values themselves are not shifted;
    only the coordinate locations used to select and center the fitted relation
    move into the same system as the latent draws.
    """

    if runtime_config.unit_convention == H_UNITS_V1:
        log10_h_ref = math.log10(runtime_config.h_ref)
        mass_axis_shift = 2.0 * log10_h_ref
        return (
            runtime_config.fp_prior.fit_mstar_min + mass_axis_shift,
            runtime_config.fp_prior.pivot_mstar + mass_axis_shift,
        )
    if runtime_config.unit_convention == LEGACY_FIXED_KPC:
        return runtime_config.fp_prior.fit_mstar_min, runtime_config.fp_prior.pivot_mstar
    raise ValueError(
        "CMASS FP prior supports unit_convention "
        f"'{H_UNITS_V1}' or '{LEGACY_FIXED_KPC}', got "
        f"'{runtime_config.unit_convention}'."
    )


def _stellar_mass_quadrature_arrays(
    *,
    runtime_config: RuntimeConfig,
    dataset: CanonicalInferenceDataset,
    profile: ProfileSpec,
    stellar_mass_pivot: float,
    mass_function_loc: float,
    mu_r0: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build CMASS stellar-mass quadrature arrays.

    These arrays are parameter-independent but model-specific: they encode the
    observed stellar-mass likelihood, foreground stellar-mass function, size
    relation, and sigma-star proxy used by the current CMASS likelihood hooks.
    """

    n_lens = len(dataset.lenses.lens_id)
    n_mstar = runtime_config.integration.mstar_points
    sqrt2 = math.sqrt(2.0)
    sqrt2pi = math.sqrt(2.0 * math.pi)

    mstar_grid = np.zeros((n_lens, n_mstar), dtype=np.float64)
    mstar_shift11p4 = np.zeros((n_lens, n_mstar), dtype=np.float64)
    sigma_star_shift9p0_grid = np.zeros((n_lens, n_mstar), dtype=np.float64)
    mstar_integrand_base = np.zeros((n_lens, n_mstar), dtype=np.float64)
    delta_r_grid = np.zeros((n_lens, n_mstar), dtype=np.float64)

    for lens_index in range(n_lens):
        observed_mstar = float(dataset.lenses.log_mstar_obs[lens_index])
        observed_mstar_err = float(dataset.lenses.log_mstar_err[lens_index])
        observed_log_re = float(dataset.lenses.log_re_obs[lens_index])
        n_value = (
            profile.fixed_n
            if profile.fixed_n is not None
            else max(float(dataset.lenses.n_obs[lens_index]), 1.0e-8)
        )
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
            sigma_star_shift9p0_grid[lens_index, mstar_index] = (
                mstar - LOG10_2PI - 2.0 * observed_log_re - 9.0
            )
            delta_r_grid[lens_index, mstar_index] = delta_r
            mstar_integrand_base[lens_index, mstar_index] = p_mobs * p_s * p_r

    return (
        mstar_grid,
        mstar_shift11p4,
        sigma_star_shift9p0_grid,
        mstar_integrand_base,
        delta_r_grid,
    )


def build_cmass_context_from_canonical_dataset(
    runtime_config: RuntimeConfig,
    *,
    dataset: CanonicalInferenceDataset | None = None,
    profile: ProfileSpec | None = None,
) -> CompiledContextBundle:
    """
    Build the CMASS source context from a canonical inference dataset.

    Callers may pass an already loaded dataset/profile to keep runtime glue
    explicit.  Tests and small utilities may omit them and let this function
    perform the canonical load itself.
    """

    active_profile = profile or build_profile_spec(runtime_config.profile.name)
    active_dataset = dataset or load_cmass_canonical_dataset(runtime_config, profile=active_profile)
    cosmology = FlatLambdaCDM(
        h0=runtime_config.cosmology.h0,
        omega_m=runtime_config.cosmology.omega_m,
    )
    random_basis = build_random_basis(
        runtime_config.integration.normalization_samples,
        runtime_config.sampling.random_seed,
    )

    n_lens = len(active_dataset.lenses.lens_id)
    n_gamma = runtime_config.integration.gamma_points
    gamma_grid_int = shared_gamma_axis(active_dataset.mass_grids.gamma_grid, n_points=n_gamma)
    mass_grid_int, dmass_dthetaein_grid_int, s2_grid_int = interpolate_lensing_mass_grids(
        active_dataset.mass_grids,
        gamma_grid_int,
    )
    stellar_mass_pivot, mass_function_loc, mu_r0 = _hunit_population_constants(
        runtime_config,
        active_profile,
    )

    sqrt2pi = math.sqrt(2.0 * math.pi)
    zd = np.asarray(active_dataset.lenses.z_d, dtype=np.float64)
    zs = np.asarray(active_dataset.lenses.z_s, dtype=np.float64)
    p_zd_fixed = np.exp(
        -0.5 * ((zd - CMASS_LENS_REDSHIFT_MEAN) / CMASS_LENS_REDSHIFT_SCATTER) ** 2
    ) / (CMASS_LENS_REDSHIFT_SCATTER * sqrt2pi)

    (
        mstar_grid,
        mstar_shift11p4,
        sigma_star_shift9p0_grid,
        mstar_integrand_base,
        delta_r_grid,
    ) = _stellar_mass_quadrature_arrays(
        runtime_config=runtime_config,
        dataset=active_dataset,
        profile=active_profile,
        stellar_mass_pivot=stellar_mass_pivot,
        mass_function_loc=mass_function_loc,
        mu_r0=mu_r0,
    )
    fp_fit_mstar_min, fp_pivot_mstar = _active_fp_prior_mass_locations(runtime_config)

    fp_gamma_axis, fp_zd_axis, fp_log_re_axis, fp_n_axis, fp_sigma_grid, fp_has_n_axis = (
        normalize_sigma_grid(
            active_dataset.velocity_dispersion.fp_within_re if runtime_config.fp_prior.enabled else None,
            profile_fixed_n=active_profile.fixed_n,
        )
    )
    context = CMASSModelContext(
        z_grid=np.ascontiguousarray(cosmology.z_table, dtype=np.float64),
        chi_kpc_grid=np.ascontiguousarray(cosmology.comoving_distance_table_mpc * 1000.0, dtype=np.float64),
        cs_gamma_grid=np.ascontiguousarray(active_dataset.cross_section.gamma_axis, dtype=np.float64),
        cs_over_theta_grid=np.zeros_like(np.asarray(active_dataset.cross_section.gamma_axis, dtype=np.float64)),
        cs_theta_e_axis=np.ascontiguousarray(active_dataset.cross_section.theta_e_axis, dtype=np.float64),
        cs_cross_section_grid=np.ascontiguousarray(active_dataset.cross_section.cross_section_grid, dtype=np.float64),
        cs_over_theta_int=np.zeros(n_gamma, dtype=np.float64),
        gamma_grid_int=np.ascontiguousarray(gamma_grid_int, dtype=np.float64),
        mass_grid_int=np.ascontiguousarray(mass_grid_int, dtype=np.float64),
        dmass_dthetaein_grid_int=np.ascontiguousarray(dmass_dthetaein_grid_int, dtype=np.float64),
        s2_grid_int=np.ascontiguousarray(s2_grid_int, dtype=np.float64),
        has_s2=np.ascontiguousarray(active_dataset.mass_grids.has_s2, dtype=np.int64),
        num_sigma=np.ascontiguousarray(active_dataset.lenses.num_sigma, dtype=np.int64),
        sigma_obs=np.ascontiguousarray(active_dataset.lenses.sigma_obs, dtype=np.float64),
        sigma_err=np.ascontiguousarray(active_dataset.lenses.sigma_err, dtype=np.float64),
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
        use_sersic_index=1 if active_profile.uses_observed_n_in_likelihood else 0,
        n_fixed=active_profile.fixed_n if active_profile.fixed_n is not None else 4.0,
        mu_n0=active_profile.mu_n0 if active_profile.mu_n0 is not None else 0.0,
        beta_n=active_profile.beta_n if active_profile.beta_n is not None else 0.0,
        sigma_n=active_profile.sigma_n if active_profile.sigma_n is not None else 0.0,
        mass_function_loc=mass_function_loc,
        mass_function_scale=active_profile.mass_function_scale,
        mass_function_alpha=active_profile.mass_function_alpha,
        mu_r0=mu_r0,
        beta_r=active_profile.beta_r,
        sigma_r=active_profile.sigma_r,
        nu_r=active_profile.nu_r if active_profile.nu_r is not None else 0.0,
        mu_d=CMASS_LENS_REDSHIFT_MEAN,
        sigma_d=CMASS_LENS_REDSHIFT_SCATTER,
        gamma_trunc_low=CMASS_GAMMA_TRUNC_LOW,
        gamma_trunc_high=CMASS_GAMMA_TRUNC_HIGH,
        normalization_min_value=CMASS_NORMALIZATION_MIN_VALUE,
        gamma_mode_code=runtime_config.parameter_schema.gamma_mode_code,
        fp_enabled=1 if runtime_config.fp_prior.enabled else 0,
        fp_fit_mstar_min=float(fp_fit_mstar_min),
        fp_pivot_mstar=float(fp_pivot_mstar),
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
        profile=active_profile,
        cross_section_grid=active_dataset.cross_section,
        cosmology=cosmology,
        random_basis=random_basis,
        observations=(),
        metadata=canonical_dataset_metadata(active_dataset),
    )


__all__ = [
    "build_cmass_context_from_canonical_dataset",
    "load_cmass_canonical_dataset",
    "required_canonical_capabilities",
]
