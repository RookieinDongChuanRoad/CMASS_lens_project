"""Deterministic preprocessing for the CMASS lens-only model."""

from __future__ import annotations

import math

import numpy as np

from statistical_sl.inference.canonical_context import canonical_dataset_metadata
from statistical_sl.inference.canonical_dataset import (
    CAPABILITY_LENSING_MASS_GRIDS_V1,
    CAPABILITY_LENS_OBSERVATIONS_V1,
    CAPABILITY_VELOCITY_DISPERSION_PER_LENS_S2_V1,
    CanonicalInferenceDataset,
    load_canonical_inference_dataset,
)
from statistical_sl.models.interfaces import CompiledContextBundle
from statistical_sl.inference.profiles import build_profile_spec
from statistical_sl.inference.types import ProfileSpec, RuntimeConfig
from ..cmass.preprocessing import build_cmass_context_from_canonical_dataset
from .context import CMASSLensOnlyContext


def required_canonical_capabilities(_runtime_config: RuntimeConfig) -> tuple[str, ...]:
    """
    Return canonical capabilities required by the lens-only model.

    The current canonical reader still expects a cross-section block to exist,
    but the lens-only scientific contract does not require the cross-section
    capability and the posterior ignores the loaded cross-section values.  The
    per-lens S2 velocity-dispersion capability is required because this model
    still evaluates the observed velocity-dispersion likelihood.
    """

    return (
        CAPABILITY_LENS_OBSERVATIONS_V1,
        CAPABILITY_LENSING_MASS_GRIDS_V1,
        CAPABILITY_VELOCITY_DISPERSION_PER_LENS_S2_V1,
    )


def load_cmass_lens_only_canonical_dataset(
    runtime_config: RuntimeConfig,
    *,
    profile: ProfileSpec | None = None,
) -> CanonicalInferenceDataset:
    """Load the canonical dataset with lens-only scientific requirements."""

    if runtime_config.data.inference_dataset_path is None:
        raise ValueError("CMASS lens-only preprocessing requires data.inference_dataset_path.")

    active_profile = profile or build_profile_spec(runtime_config.profile.name)
    return load_canonical_inference_dataset(
        runtime_config.data.inference_dataset_path,
        expected_unit_convention=runtime_config.unit_convention,
        expected_h_ref=runtime_config.h_ref,
        expected_profile_name=active_profile.name,
        expected_mass_definition_label=runtime_config.mass_definition.label,
        required_capabilities=required_canonical_capabilities(runtime_config),
    )


def _mstar_observation_density(
    *,
    dataset: CanonicalInferenceDataset,
    mstar_grid: np.ndarray,
) -> np.ndarray:
    """
    Build `P(logMstar_obs | logMstar)` on each per-lens quadrature grid.

    This array is parameter-independent and can be precomputed.  It is kept
    separate from the sampled lens stellar-mass distribution because
    `mu_mstar_lens` and `sigma_mstar_lens` change at every posterior call.
    """

    sqrt2pi = math.sqrt(2.0 * math.pi)
    density = np.zeros_like(mstar_grid, dtype=np.float64)
    for lens_index in range(mstar_grid.shape[0]):
        observed = float(dataset.lenses.log_mstar_obs[lens_index])
        error = float(dataset.lenses.log_mstar_err[lens_index])
        if error <= 0.0:
            raise ValueError(
                "CMASS lens-only requires positive log_mstar_err for every lens; "
                f"lens index {lens_index} has {error}."
            )
        values = mstar_grid[lens_index]
        density[lens_index] = np.exp(-0.5 * ((observed - values) / error) ** 2)
        density[lens_index] /= error * sqrt2pi
    return np.ascontiguousarray(density, dtype=np.float64)


def build_cmass_lens_only_context_from_canonical_dataset(
    runtime_config: RuntimeConfig,
    *,
    dataset: CanonicalInferenceDataset | None = None,
    profile: ProfileSpec | None = None,
) -> CompiledContextBundle:
    """
    Build the CMASS lens-only context from a canonical inference dataset.

    The base CMASS context is reused for mass-grid interpolation, kinematic
    grids, h-unit pivots, and per-lens deterministic covariates.  The lens-only
    posterior consumes only the subset relevant to the observed-sample
    likelihood and deliberately ignores parent-population and selection arrays
    such as `mstar_integrand_base` and cross-section weights.
    """

    if runtime_config.fp_prior.enabled:
        raise ValueError("cmass_lens_only does not support fp_prior.enabled=true.")

    active_profile = profile or build_profile_spec(runtime_config.profile.name)
    active_dataset = dataset or load_cmass_lens_only_canonical_dataset(
        runtime_config,
        profile=active_profile,
    )
    base_bundle = build_cmass_context_from_canonical_dataset(
        runtime_config,
        dataset=active_dataset,
        profile=active_profile,
    )
    base_context = base_bundle.context
    lens_only_context = CMASSLensOnlyContext(
        base=base_context,
        mstar_observation_density=_mstar_observation_density(
            dataset=active_dataset,
            mstar_grid=base_context.mstar_grid,
        ),
    )
    metadata = {
        **canonical_dataset_metadata(active_dataset),
        "selection_correction": False,
        "target_population": "observed_cmass_lenses",
    }
    return CompiledContextBundle(
        context=lens_only_context,
        profile=base_bundle.profile,
        cross_section_grid=base_bundle.cross_section_grid,
        cosmology=base_bundle.cosmology,
        random_basis=base_bundle.random_basis,
        observations=(),
        metadata=metadata,
    )


__all__ = [
    "build_cmass_lens_only_context_from_canonical_dataset",
    "load_cmass_lens_only_canonical_dataset",
    "required_canonical_capabilities",
]
