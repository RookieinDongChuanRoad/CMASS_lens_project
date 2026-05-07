"""
Adapt high-level model specs to the production Numba backend contract.

The adapter is intentionally small.  Model files provide readable scientific
declarations, runtime adapters build validated NumPy source contexts, and this
module turns those two pieces into the registry-facing `ModelDefinition` used
by config parsing and the Numba likelihood engine.
"""

from __future__ import annotations

from ..mass_definition import get_mass_definition
from ..model_interfaces import ModelDefinition, ModelRuntimeAdapter, ModelSpec
from ..parallel import resolve_parallelism
from ..parameter_schema import ParameterSchema
from ..types import CompiledModel, RuntimeConfig


def _resolve_mass_definition_for_spec(model_spec: ModelSpec, unit_convention: str):
    """
    Resolve the model-owned mass definition under the declared unit contract.

    A concrete model name owns one unit convention and one mass aperture.  If a
    config asks for a different unit convention, failing here points directly at
    the model/config mismatch before any HDF5 or kernel work starts.
    """

    if unit_convention != model_spec.required_unit_convention:
        raise ValueError(
            f"The '{model_spec.name}' model requires unit_convention "
            f"'{model_spec.required_unit_convention}', but the config declares "
            f"'{unit_convention}'."
        )
    return get_mass_definition(
        model_spec.mass_aperture_kpc,
        unit_convention=unit_convention,
    )


def _build_parameter_schema_for_spec(
    model_spec: ModelSpec,
    *,
    mass_definition,
    public_box_prior,
) -> ParameterSchema:
    """
    Build the backend parameter schema from a human-authored model spec.

    `mass_definition` is accepted for the registry callable shape.  The current
    `ModelSpec` already declares public names and bounds explicitly, so the
    argument is used only as a compatibility hook and does not rewrite the
    schema.
    """

    del mass_definition
    template_schema = ParameterSchema(
        model_name=model_spec.name,
        model_component_key=model_spec.component_key,
        internal_parameter_names=tuple(parameter.internal_name for parameter in model_spec.parameters),
        public_parameter_names=tuple(parameter.public_name for parameter in model_spec.parameters),
        prior_bounds=tuple(parameter.bounds for parameter in model_spec.parameters),
        static_codes=dict(model_spec.static_codes),
        model_metadata=dict(model_spec.metadata),
    )
    prior_bounds = (
        tuple(parameter.bounds for parameter in model_spec.parameters)
        if public_box_prior is None
        else template_schema.normalize_public_box_prior(public_box_prior)
    )
    return ParameterSchema(
        model_name=template_schema.model_name,
        model_component_key=template_schema.model_component_key,
        internal_parameter_names=template_schema.internal_parameter_names,
        public_parameter_names=template_schema.public_parameter_names,
        prior_bounds=prior_bounds,
        static_codes=template_schema.static_codes,
        model_metadata=template_schema.model_metadata,
    )


def build_compiled_model_from_runtime_adapter(
    runtime_config: RuntimeConfig,
    runtime_adapter: ModelRuntimeAdapter,
) -> CompiledModel:
    """
    Build the shared `CompiledModel` container for Numba production kernels.

    Runtime adapters are model-specific because canonical datasets still need
    model-owned preprocessing.  This helper owns the common framework pieces:
    resolving process/thread policy, preserving metadata, and packaging the
    opaque source context for the likelihood engine.
    """

    bundle = runtime_adapter.build_context_bundle(runtime_config)
    parallelism = resolve_parallelism(
        runtime_config.runtime,
        runtime_config.sampling.n_walkers,
    )
    return CompiledModel(
        config=runtime_config,
        profile=bundle.profile,
        cross_section_grid=bundle.cross_section_grid,
        cosmology=bundle.cosmology,
        parallelism=parallelism,
        context=bundle.context,
        data_metadata=dict(bundle.metadata),
    )


def build_model_definition(
    model_spec: ModelSpec,
    runtime_adapter: ModelRuntimeAdapter,
) -> ModelDefinition:
    """
    Combine a scientific spec and runtime adapter into a registry entry.

    The returned definition deliberately contains only production Numba/emcee
    entrypoints.  The Numba backend dispatches through `backend_kernel`, while
    the framework still gets model-owned schema, mass-definition, and
    context-build callables.
    """

    return ModelDefinition(
        name=model_spec.name,
        required_capabilities=tuple(model_spec.required_capabilities),
        optional_capabilities=tuple(model_spec.optional_capabilities),
        resolve_mass_definition=lambda unit_convention: _resolve_mass_definition_for_spec(
            model_spec,
            unit_convention,
        ),
        build_parameter_schema=lambda **kwargs: _build_parameter_schema_for_spec(
            model_spec,
            **kwargs,
        ),
        build_compiled_model=lambda runtime_config: build_compiled_model_from_runtime_adapter(
            runtime_config,
            runtime_adapter,
        ),
        backend_kernel=model_spec.backend_kernel,
    )


__all__ = ["build_compiled_model_from_runtime_adapter", "build_model_definition"]
