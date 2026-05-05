"""
Common compiled-context and JAX-context builders.

Model runtime adapters return validated NumPy context bundles.  The generic
helpers below turn those bundles into `CompiledModel`, JAX pytrees, and JIT
static flags without model files hand-writing backend boilerplate.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from ..model_interfaces import DataSpec, ModelRuntimeAdapter
from ..parallel import resolve_parallelism
from ..types import CompiledModel, RuntimeConfig
from .primitives import as_jax_array


def _read_context_attribute(context: Any, field_name: str) -> Any:
    """
    Read one declared context field with a model-facing error message.

    A typo in a `DataSpec` should fail at context-build time with the missing
    model field, not later inside a traced JAX function where the stack is much
    harder to interpret.
    """

    try:
        return getattr(context, field_name)
    except AttributeError as exc:
        raise AttributeError(
            f"DataSpec requested context field '{field_name}', but "
            f"{type(context).__name__} does not define it."
        ) from exc


def _target_name_for_field(data_spec: DataSpec, declared_field_name: str) -> str:
    """
    Resolve a source or target field name to the generated JAX-context name.

    Normalization fields are configured by name.  Accepting either source or
    target names keeps the config readable when a model chooses to rename a
    field at the JAX boundary.
    """

    for array_spec in data_spec.array_fields:
        if declared_field_name in {array_spec.source_name, array_spec.output_name}:
            return array_spec.output_name
    return declared_field_name


def _scalar_index_by_source_name(data_spec: DataSpec) -> dict[str, int]:
    """Return scalar-context indices keyed by NumPy source-field name."""

    return {
        scalar_spec.source_name: index
        for index, scalar_spec in enumerate(data_spec.scalar_fields)
    }


def build_jax_context_from_data_spec(context: Any, data_spec: DataSpec) -> Any:
    """
    Convert a model's NumPy context into its declared JAX context type.

    Arrays are converted with the existing backend helper so dtype and contiguity
    rules remain centralized.  Scalars are packed into one float64 array because
    the hot kernels index them repeatedly and that representation is stable
    under JAX tracing.
    """

    values: dict[str, Any] = {}
    for array_spec in data_spec.array_fields:
        values[array_spec.output_name] = as_jax_array(
            _read_context_attribute(context, array_spec.source_name)
        )

    if data_spec.scalar_context_name is not None:
        scalar_values = [
            _read_context_attribute(context, scalar_spec.source_name)
            for scalar_spec in data_spec.scalar_fields
        ]
        values[data_spec.scalar_context_name] = jnp.asarray(
            scalar_values,
            dtype=jnp.float64,
        )

    return data_spec.jax_context_type(**values)


def static_jit_kwargs_from_data_spec(context: Any, data_spec: DataSpec) -> dict[str, int]:
    """
    Build static JIT kwargs from a raw model context and its `DataSpec`.

    Values are cast to plain Python integers because these flags form part of
    the JIT cache key and are intentionally not traced as JAX arrays.
    """

    return {
        static_spec.output_name: int(_read_context_attribute(context, static_spec.source_name))
        for static_spec in data_spec.static_fields
    }


def normalization_samples_from_data_spec(jax_context: Any, data_spec: DataSpec) -> Any:
    """Return fixed MC normalization samples from a generated JAX context."""

    field_name = _target_name_for_field(data_spec, data_spec.normalization_samples_field)
    return _read_context_attribute(jax_context, field_name)


def normalization_min_value_from_data_spec(jax_context: Any, data_spec: DataSpec) -> Any:
    """
    Return the minimum accepted normalization from the generated context.

    The value may be packed in `scalar_context` or exposed as a direct field.
    CMASS uses the scalar route to preserve its existing hot-path layout.
    """

    scalar_index = _scalar_index_by_source_name(data_spec).get(
        data_spec.normalization_min_value_field
    )
    if scalar_index is not None:
        if data_spec.scalar_context_name is None:
            raise ValueError(
                "DataSpec declares scalar normalization_min_value_field but "
                "has no scalar_context_name."
            )
        return _read_context_attribute(jax_context, data_spec.scalar_context_name)[scalar_index]

    field_name = _target_name_for_field(data_spec, data_spec.normalization_min_value_field)
    return _read_context_attribute(jax_context, field_name)


def build_compiled_model_from_runtime_adapter(
    runtime_config: RuntimeConfig,
    runtime_adapter: ModelRuntimeAdapter,
) -> CompiledModel:
    """
    Build the standard `CompiledModel` container from a runtime adapter.

    The adapter owns source-context construction.  This helper owns the common
    backend container shape and parallelism resolution so model runtime files
    do not need to repeat it.
    """

    bundle = runtime_adapter.build_context_bundle(runtime_config)
    parallelism = resolve_parallelism(
        runtime_config.runtime,
        runtime_config.sampling.num_chains,
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


__all__ = [
    "build_compiled_model_from_runtime_adapter",
    "build_jax_context_from_data_spec",
    "normalization_min_value_from_data_spec",
    "normalization_samples_from_data_spec",
    "static_jit_kwargs_from_data_spec",
]
