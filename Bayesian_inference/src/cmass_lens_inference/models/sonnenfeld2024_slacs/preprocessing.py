"""Deterministic preprocessing for the Sonnenfeld 2024 SLACS model.

The canonical dataset reader validates raw HDF5 shape and capability
contracts.  This module performs the model-specific deterministic work that is
still required before the production backend can evaluate Sonnenfeld kernels:

- load a complete canonical dataset with the velocity-dispersion proxy grid;
- interpolate per-lens mass tracks onto the configured gamma integration axis;
- build per-lens stellar-mass quadrature arrays;
- place paper Table-1 mass-location constants in the active model coordinate;
- normalize the population sigma-unit grid to the rank expected by the kernel.

No sampled parameter enters this module.  That separation is important because
preprocessing should run once per inference run, while likelihood kernels run
for every proposed hyper-parameter vector.
"""

from __future__ import annotations

import math

import numpy as np

from ...canonical_dataset import CanonicalInferenceDataset, load_canonical_inference_dataset
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
from . import paper_constants as parameters
from .assembly import REQUIRED_CAPABILITIES
from .context import SonnenfeldModelContext


def load_sonnenfeld_canonical_dataset(
    runtime_config: RuntimeConfig,
    *,
    profile: ProfileSpec | None = None,
) -> CanonicalInferenceDataset:
    """
    Load canonical data with Sonnenfeld-specific capability requirements.

    Sonnenfeld v1 consumes the same canonical dataset entry as every production
    model, but it requires the additional ``population_sigma_unit`` block used
    by the velocity-dispersion proxy selection function.
    """

    if runtime_config.data.inference_dataset_path is None:
        raise ValueError("Sonnenfeld preprocessing requires data.inference_dataset_path.")
    active_profile = profile or build_profile_spec(runtime_config.profile.name)
    return load_canonical_inference_dataset(
        runtime_config.data.inference_dataset_path,
        expected_unit_convention=runtime_config.unit_convention,
        expected_h_ref=runtime_config.h_ref,
        expected_profile_name=active_profile.name,
        expected_mass_definition_label=runtime_config.mass_definition.label,
        required_capabilities=REQUIRED_CAPABILITIES,
    )


def _active_mass_locations(runtime_config: RuntimeConfig) -> tuple[float, float]:
    """
    Return paper mass-location constants in the active canonical coordinate.

    Sonnenfeld's Table-1 constants are paper-native physical stellar-mass
    locations.  The registry now exposes two concrete models:

    - ``sonnenfeld2024_slacs`` consumes legacy fixed-kpc / physical-mass
      canonical input, so the constants must remain unshifted.
    - ``sonnenfeld2024_slacs_hunit`` consumes h-unit canonical input, so every
      location in the stellar-mass coordinate is shifted by ``2 log10(h_ref)``.

    Keeping this conversion in preprocessing, instead of inside hot kernels,
    makes the hot path unit-agnostic and keeps model names tied to one explicit
    coordinate contract.
    """

    if runtime_config.unit_convention == LEGACY_FIXED_KPC:
        return parameters.MSTAR_PIVOT_PHYSICAL, parameters.MBAR_PHYSICAL
    if runtime_config.unit_convention == H_UNITS_V1:
        return (
            parameters.shift_physical_mass_location_to_hunits(
                parameters.MSTAR_PIVOT_PHYSICAL,
                runtime_config.h_ref,
            ),
            parameters.shift_physical_mass_location_to_hunits(
                parameters.MBAR_PHYSICAL,
                runtime_config.h_ref,
            ),
        )
    raise ValueError(
        "Sonnenfeld preprocessing supports unit_convention "
        f"'{LEGACY_FIXED_KPC}' or '{H_UNITS_V1}', got "
        f"'{runtime_config.unit_convention}'."
    )


def truncation_mass_threshold(
    zd: np.ndarray,
    *,
    h_ref: float,
    unit_convention: str = H_UNITS_V1,
) -> np.ndarray:
    """
    Evaluate the Sonnenfeld Table-1 low-mass truncation threshold.

    The polynomial coefficients are stored in physical stellar-mass
    coordinates.  The paper-native model compares this threshold directly to
    physical ``log_mstar`` values.  The explicit hunit variant compares against
    ``log10(M*/(h^-2 Msun))``, so the threshold must be shifted into that same
    coordinate before constructing the completeness term.
    """

    powers = np.vstack([np.asarray(zd, dtype=np.float64) ** index for index in range(6)])
    physical_threshold = parameters.TRUNCATION_MASS_POLYNOMIAL_COEFFICIENTS @ powers
    if unit_convention == LEGACY_FIXED_KPC:
        return physical_threshold
    if unit_convention == H_UNITS_V1:
        return physical_threshold + 2.0 * math.log10(float(h_ref))
    raise ValueError(
        "Sonnenfeld truncation threshold supports unit_convention "
        f"'{LEGACY_FIXED_KPC}' or '{H_UNITS_V1}', got '{unit_convention}'."
    )


def _parent_density_grid(
    *,
    zd: np.ndarray,
    mstar_grid: np.ndarray,
    mbar: float,
    parent_alpha: float,
    h_ref: float,
    unit_convention: str,
) -> np.ndarray:
    """
    Evaluate the unnormalized Sonnenfeld parent ``P_g(z_d, m*)`` on a grid.

    The normalization cancels between likelihood and selection normalization,
    so the model only needs a positive density up to an overall constant.  The
    redshift factor is represented by a simple comoving-volume proxy ``z_d^2``
    in this first production implementation; the expensive survey parent
    fitting stays outside runtime, consistent with the canonical-dataset
    boundary.
    """

    threshold = truncation_mass_threshold(
        zd,
        h_ref=h_ref,
        unit_convention=unit_convention,
    )[:, None]
    completeness_argument = (mstar_grid - threshold) / parameters.TRUNCATION_MASS_SCATTER
    completeness = np.arctan(completeness_argument) / math.pi + 0.5
    schechter_mass = np.power(10.0, mstar_grid - mbar)
    schechter = np.power(10.0, (mstar_grid - mbar) * (parent_alpha + 1.0))
    schechter *= np.exp(-schechter_mass)
    volume_proxy = np.square(np.maximum(zd[:, None], 1.0e-6))
    return np.ascontiguousarray(volume_proxy * completeness * schechter, dtype=np.float64)


def _stellar_mass_quadrature_arrays(
    *,
    runtime_config: RuntimeConfig,
    dataset: CanonicalInferenceDataset,
    profile: ProfileSpec,
    mstar_pivot: float,
    mbar: float,
    size_mu0: float,
    size_mu1: float,
    size_mu2: float,
    size_sigma: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build per-lens stellar-mass quadrature arrays for likelihood evaluation.

    The per-lens likelihood integrates over latent true stellar mass while
    conditioning on observed stellar mass and size.  This helper precomputes
    every factor that does not depend on sampled mass/gamma hyper-parameters.
    """

    n_lens = len(dataset.lenses.lens_id)
    n_mstar = runtime_config.integration.mstar_points
    sqrt2pi = math.sqrt(2.0 * math.pi)
    mstar_grid = np.zeros((n_lens, n_mstar), dtype=np.float64)
    parent_density = np.zeros((n_lens, n_mstar), dtype=np.float64)
    size_density = np.zeros((n_lens, n_mstar), dtype=np.float64)
    delta_r_grid = np.zeros((n_lens, n_mstar), dtype=np.float64)
    mstar_shift_grid = np.zeros((n_lens, n_mstar), dtype=np.float64)

    for lens_index in range(n_lens):
        observed_mstar = float(dataset.lenses.log_mstar_obs[lens_index])
        observed_mstar_err = float(dataset.lenses.log_mstar_err[lens_index])
        observed_log_re = float(dataset.lenses.log_re_obs[lens_index])
        mstar_grid[lens_index] = np.linspace(
            observed_mstar - 5.0 * observed_mstar_err,
            observed_mstar + 5.0 * observed_mstar_err,
            n_mstar,
            dtype=np.float64,
        )
        mstar_shift = mstar_grid[lens_index] - mstar_pivot
        mu_r_value = size_mu0 + size_mu1 * mstar_grid[lens_index] + size_mu2 * np.square(
            mstar_grid[lens_index]
        )
        delta_r = observed_log_re - mu_r_value
        p_size = np.exp(-0.5 * np.square(delta_r / size_sigma))
        p_size /= size_sigma * sqrt2pi
        p_mobs = np.exp(-0.5 * np.square((observed_mstar - mstar_grid[lens_index]) / observed_mstar_err))
        p_mobs /= observed_mstar_err * sqrt2pi

        mstar_shift_grid[lens_index] = mstar_shift
        delta_r_grid[lens_index] = delta_r
        size_density[lens_index] = p_size * p_mobs

    parent_density[:] = _parent_density_grid(
        zd=np.asarray(dataset.lenses.z_d, dtype=np.float64),
        mstar_grid=mstar_grid,
        mbar=mbar,
        parent_alpha=parameters.PARENT_ALPHA,
        h_ref=runtime_config.h_ref,
        unit_convention=runtime_config.unit_convention,
    )
    return (
        np.ascontiguousarray(mstar_grid, dtype=np.float64),
        np.ascontiguousarray(parent_density, dtype=np.float64),
        np.ascontiguousarray(size_density, dtype=np.float64),
        np.ascontiguousarray(delta_r_grid, dtype=np.float64),
        np.ascontiguousarray(mstar_shift_grid, dtype=np.float64),
    )


def build_sonnenfeld_context_from_canonical_dataset(
    runtime_config: RuntimeConfig,
    *,
    dataset: CanonicalInferenceDataset | None = None,
    profile: ProfileSpec | None = None,
) -> CompiledContextBundle:
    """
    Build the Sonnenfeld source context from canonical inference data.

    The returned bundle is consumed by the generic ``ModelRuntimeAdapter``.  It
    deliberately reuses backend-level canonical helpers where possible, while
    keeping Sonnenfeld-specific deterministic quantities in this component.
    """

    active_profile = profile or build_profile_spec(runtime_config.profile.name)
    active_dataset = dataset or load_sonnenfeld_canonical_dataset(runtime_config, profile=active_profile)
    if active_dataset.velocity_dispersion.population_sigma_unit is None:
        raise ValueError("Sonnenfeld canonical input must include population_sigma_unit.")

    cosmology = FlatLambdaCDM(
        h0=runtime_config.cosmology.h0,
        omega_m=runtime_config.cosmology.omega_m,
    )
    random_basis = build_random_basis(
        runtime_config.integration.normalization_samples,
        runtime_config.sampling.random_seed,
    )
    gamma_grid_int = shared_gamma_axis(
        active_dataset.mass_grids.gamma_grid,
        n_points=runtime_config.integration.gamma_points,
    )
    mass_grid_int, dmass_dthetaein_grid_int, s2_grid_int = interpolate_lensing_mass_grids(
        active_dataset.mass_grids,
        gamma_grid_int,
    )
    mstar_pivot, mbar = _active_mass_locations(runtime_config)
    size_mu0, size_mu1, size_mu2 = parameters.active_size_relation_coefficients(
        h_ref=runtime_config.h_ref,
        unit_convention=runtime_config.unit_convention,
    )
    (
        mstar_grid,
        parent_mstar_density_grid,
        size_density_grid,
        delta_r_grid,
        mstar_shift_grid,
    ) = _stellar_mass_quadrature_arrays(
        runtime_config=runtime_config,
        dataset=active_dataset,
        profile=active_profile,
        mstar_pivot=mstar_pivot,
        mbar=mbar,
        size_mu0=size_mu0,
        size_mu1=size_mu1,
        size_mu2=size_mu2,
        size_sigma=parameters.SIZE_SCATTER,
    )
    (
        population_gamma_axis,
        population_zd_axis,
        population_log_re_axis,
        population_n_axis,
        population_sigma_unit_grid,
        population_has_n_axis,
    ) = normalize_sigma_grid(
        active_dataset.velocity_dispersion.population_sigma_unit,
        profile_fixed_n=active_profile.fixed_n,
    )
    if population_has_n_axis == 0 and active_profile.fixed_n is None:
        raise ValueError("Sonnenfeld sersic population sigma grid must include n_axis.")

    context = SonnenfeldModelContext(
        z_grid=np.ascontiguousarray(cosmology.z_table, dtype=np.float64),
        chi_kpc_grid=np.ascontiguousarray(cosmology.comoving_distance_table_mpc * 1000.0, dtype=np.float64),
        cs_theta_e_axis=np.ascontiguousarray(active_dataset.cross_section.theta_e_axis, dtype=np.float64),
        cs_gamma_axis=np.ascontiguousarray(active_dataset.cross_section.gamma_axis, dtype=np.float64),
        cs_cross_section_grid=np.ascontiguousarray(active_dataset.cross_section.cross_section_grid, dtype=np.float64),
        gamma_grid_int=np.ascontiguousarray(gamma_grid_int, dtype=np.float64),
        mass_grid_int=np.ascontiguousarray(mass_grid_int, dtype=np.float64),
        dmass_dthetaein_grid_int=np.ascontiguousarray(dmass_dthetaein_grid_int, dtype=np.float64),
        s2_grid_int=np.ascontiguousarray(s2_grid_int, dtype=np.float64),
        has_s2=np.ascontiguousarray(active_dataset.mass_grids.has_s2, dtype=np.int64),
        num_sigma=np.ascontiguousarray(active_dataset.lenses.num_sigma, dtype=np.int64),
        sigma_obs=np.ascontiguousarray(active_dataset.lenses.sigma_obs, dtype=np.float64),
        sigma_err=np.ascontiguousarray(active_dataset.lenses.sigma_err, dtype=np.float64),
        zd=np.ascontiguousarray(active_dataset.lenses.z_d, dtype=np.float64),
        zs=np.ascontiguousarray(active_dataset.lenses.z_s, dtype=np.float64),
        log_mstar_obs=np.ascontiguousarray(active_dataset.lenses.log_mstar_obs, dtype=np.float64),
        log_mstar_err=np.ascontiguousarray(active_dataset.lenses.log_mstar_err, dtype=np.float64),
        log_re_obs=np.ascontiguousarray(active_dataset.lenses.log_re_obs, dtype=np.float64),
        n_obs=np.ascontiguousarray(active_dataset.lenses.n_obs, dtype=np.float64),
        theta_e_obs=np.ascontiguousarray(active_dataset.lenses.theta_e_obs, dtype=np.float64),
        mstar_grid=mstar_grid,
        parent_mstar_density_grid=parent_mstar_density_grid,
        size_density_grid=size_density_grid,
        delta_r_grid=delta_r_grid,
        mstar_shift_grid=mstar_shift_grid,
        base_normals=np.ascontiguousarray(random_basis.base_normals, dtype=np.float64),
        population_gamma_axis=np.ascontiguousarray(population_gamma_axis, dtype=np.float64),
        population_zd_axis=np.ascontiguousarray(population_zd_axis, dtype=np.float64),
        population_log_re_kpc_axis=np.ascontiguousarray(population_log_re_axis, dtype=np.float64),
        population_n_axis=np.ascontiguousarray(population_n_axis, dtype=np.float64),
        population_sigma_unit_grid=np.ascontiguousarray(population_sigma_unit_grid, dtype=np.float64),
        mass_radius_kpc=float(runtime_config.mass_definition.physical_radius_kpc(runtime_config.h_ref)),
        mass_log_physical_offset=float(runtime_config.mass_definition.log_mass_physical_offset(runtime_config.h_ref)),
        mstar_pivot=float(mstar_pivot),
        mbar=float(mbar),
        parent_alpha=float(parameters.PARENT_ALPHA),
        truncation_mass_scatter=float(parameters.TRUNCATION_MASS_SCATTER),
        size_mu0=float(size_mu0),
        size_mu1=float(size_mu1),
        size_sigma=float(parameters.SIZE_SCATTER),
        size_mu2=float(size_mu2),
        n_fixed=float(active_profile.fixed_n if active_profile.fixed_n is not None else 4.0),
        use_sersic_index=1 if active_profile.uses_observed_n_in_likelihood else 0,
        gamma_trunc_low=float(parameters.GAMMA_TRUNC_LOW),
        gamma_trunc_high=float(parameters.GAMMA_TRUNC_HIGH),
        parent_zd_min=float(parameters.PARENT_ZD_MIN),
        parent_zd_max=float(parameters.PARENT_ZD_MAX),
        parent_mstar_min=float(mbar + parameters.PARENT_MSTAR_MIN_OFFSET),
        parent_mstar_max=float(mbar + parameters.PARENT_MSTAR_MAX_OFFSET),
        sigma_proxy_fractional_scatter=float(parameters.SIGMA_PROXY_FRACTIONAL_SCATTER),
        normalization_min_value=1.0e-12,
    )
    return CompiledContextBundle(
        context=context,
        profile=active_profile,
        cross_section_grid=active_dataset.cross_section,
        cosmology=cosmology,
        random_basis=random_basis,
        observations=(),
        metadata=canonical_dataset_metadata(active_dataset)
        | {
            "sonnenfeld_mstar_pivot_active": float(mstar_pivot),
            "sonnenfeld_mbar_active": float(mbar),
            "sonnenfeld_sigma_proxy_fractional_scatter": float(
                parameters.SIGMA_PROXY_FRACTIONAL_SCATTER
            ),
        },
    )


__all__ = [
    "build_sonnenfeld_context_from_canonical_dataset",
    "load_sonnenfeld_canonical_dataset",
    "truncation_mass_threshold",
]
