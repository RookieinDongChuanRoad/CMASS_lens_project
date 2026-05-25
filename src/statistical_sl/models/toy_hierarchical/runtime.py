"""Runtime adapter for the toy hierarchical model.

The adapter intentionally ignores the canonical HDF5 path.  That is acceptable
for this architecture acceptance model because the goal is to exercise the
production framework boundary with a deterministic synthetic context, not to
introduce a new scientific dataset format.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from statistical_sl.models.interfaces import (
    CompiledContextBundle,
    ContextArraySpec,
    ContextScalarSpec,
    DataSpec,
    ModelRuntimeAdapter,
)
from statistical_sl.inference.types import CrossSectionGrid, ProfileSpec, RuntimeConfig


@dataclass(frozen=True)
class ToyHierarchicalContext:
    """
    Parameter-independent arrays consumed by the toy production posterior.

    `observed_values` are synthetic population measurements.  `observed_errors`
    are fixed measurement uncertainties.  The normalization fields are present
    so this context can satisfy the same DataSpec shape expected of production
    runtime adapters.
    """

    observed_values: np.ndarray
    observed_errors: np.ndarray
    normalization_samples: int
    normalization_min_value: float


def _build_profile_spec(runtime_config: RuntimeConfig) -> ProfileSpec:
    """
    Build a minimal profile object for generic run metadata.

    The toy model does not use CMASS profile constants, but the shared
    `CompiledModel` container still carries a profile field.  Supplying an
    explicit placeholder keeps the runtime object inspectable.
    """

    return ProfileSpec(
        name=runtime_config.profile.name,
        fixed_n=None,
        uses_observed_n_in_likelihood=False,
        observation_field_aliases={},
        mass_function_loc=0.0,
        mass_function_scale=1.0,
        mass_function_alpha=0.0,
        mu_r0=0.0,
        beta_r=0.0,
        sigma_r=1.0,
        nu_r=None,
        mu_n0=None,
        beta_n=None,
        sigma_n=None,
    )


def build_context_bundle(runtime_config: RuntimeConfig) -> CompiledContextBundle:
    """
    Build a deterministic synthetic context for architecture tests.

    The values are intentionally small and well-conditioned so a short emcee
    smoke run can finish quickly while still exercising finite likelihoods,
    out-of-bounds rejection, diagnostic blobs, and HDFBackend output.
    """

    observed_values = np.asarray([-0.15, 0.05, 0.20, 0.35], dtype=np.float64)
    observed_errors = np.asarray([0.08, 0.10, 0.09, 0.11], dtype=np.float64)
    context = ToyHierarchicalContext(
        observed_values=observed_values,
        observed_errors=observed_errors,
        normalization_samples=runtime_config.integration.normalization_samples,
        normalization_min_value=1.0e-12,
    )
    return CompiledContextBundle(
        context=context,
        profile=_build_profile_spec(runtime_config),
        cross_section_grid=CrossSectionGrid(
            gamma_grid=np.asarray([], dtype=np.float64),
            cs_over_theta_ein=np.asarray([], dtype=np.float64),
        ),
        cosmology=None,
        random_basis=None,
        observations=(),
        metadata={
            "canonical_capabilities": (),
            "canonical_schema_version": "toy_synthetic_context_v1",
            "canonical_profile_name": runtime_config.profile.name,
            "canonical_mass_definition_label": runtime_config.mass_definition.label,
        },
    )


def get_data_spec() -> DataSpec:
    """Return the generic context declaration for the toy runtime."""

    return DataSpec(
        backend_context_type=ToyHierarchicalContext,
        array_fields=(
            ContextArraySpec("observed_values"),
            ContextArraySpec("observed_errors"),
        ),
        scalar_fields=(
            ContextScalarSpec("normalization_samples"),
            ContextScalarSpec("normalization_min_value"),
        ),
        static_fields=(),
        normalization_samples_field="normalization_samples",
        normalization_min_value_field="normalization_min_value",
    )


def get_runtime_adapter() -> ModelRuntimeAdapter:
    """Return the runtime adapter consumed by the model registry."""

    return ModelRuntimeAdapter(
        build_context_bundle=build_context_bundle,
        data_spec=get_data_spec(),
    )


__all__ = [
    "ToyHierarchicalContext",
    "build_context_bundle",
    "get_data_spec",
    "get_runtime_adapter",
]
