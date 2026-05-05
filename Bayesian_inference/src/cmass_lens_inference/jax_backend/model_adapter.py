"""
Adapt high-level model specs to the low-level JAX backend contract.

Model authors should not need to know how the production backend builds
`ParameterSchema`, resolves mass definitions, or wires compiled contexts into
`ModelDefinition`.  This module centralizes that translation so concrete model
files can stay focused on the scientific formulas.
"""

from __future__ import annotations

from ..mass_definition import get_mass_definition
from ..model_interfaces import ModelDefinition, ModelRuntimeAdapter, ModelSpec
from ..parameter_schema import ParameterSchema
from .context_builder import (
    build_compiled_model_from_runtime_adapter,
    build_jax_context_from_data_spec,
    normalization_min_value_from_data_spec,
    normalization_samples_from_data_spec,
    static_jit_kwargs_from_data_spec,
)


def _resolve_mass_definition_for_spec(model_spec: ModelSpec, unit_convention: str):
    """
    Resolve the model's fixed mass definition under the declared unit contract.

    The first adapter milestone keeps mass-aperture choice as a model constant.
    If the config declares a different unit convention, failing here gives the
    user a direct model/config mismatch instead of letting I/O fail later.
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
    Build the backend schema from one human-authored parameter declaration.

    `mass_definition` is accepted because the existing config loader passes it
    through the model-definition callable.  The current `ModelSpec` already
    stores public parameter names explicitly, so the argument is a validation
    hook rather than an input that changes the schema.
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


def build_model_definition(
    model_spec: ModelSpec,
    runtime_adapter: ModelRuntimeAdapter,
) -> ModelDefinition:
    """
    Combine a scientific spec and runtime adapter into backend callables.

    `ModelDefinition` remains the narrow interface consumed by the JAX engine.
    This factory is the only place where high-level model declarations are
    converted into that lower-level shape.
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
        static_jit_kwargs=lambda compiled_model: static_jit_kwargs_from_data_spec(
            compiled_model.context,
            runtime_adapter.data_spec,
        ),
        to_jax_context=lambda compiled_model: build_jax_context_from_data_spec(
            compiled_model.context,
            runtime_adapter.data_spec,
        ),
        normalization_samples=lambda model_context: normalization_samples_from_data_spec(
            model_context,
            runtime_adapter.data_spec,
        ),
        normalization_min_value=lambda model_context: normalization_min_value_from_data_spec(
            model_context,
            runtime_adapter.data_spec,
        ),
        unpack_theta=model_spec.unpack_theta,
        validate_theta=model_spec.validate_theta,
        draw_population=model_spec.draw_population,
        selection_weight=model_spec.selection_weight,
        summary_row=model_spec.summary_row,
        lens_integrals=model_spec.lens_integrals,
        extra_prior=model_spec.extra_prior,
    )


__all__ = ["build_model_definition"]
