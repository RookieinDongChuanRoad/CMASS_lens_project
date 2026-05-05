"""Interfaces for registry-backed scientific lens-population models."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .mass_definition import MassDefinition
from .parameter_schema import ParameterSchema
from .types import CompiledModel, RuntimeConfig


@dataclass(frozen=True)
class ParameterSpec:
    """
    Human-authored declaration for one sampled model parameter.

    The model file should describe the scientific parameter surface in this
    compact form.  The runtime adapter later turns the list into the lower-level
    `ParameterSchema` used by config parsing, NumPyro, and output metadata.
    Keeping this declaration small is what lets future model authors define a
    new inference target without knowing how the backend serializes schemas.
    """

    internal_name: str
    public_name: str
    bounds: tuple[float, float]


@dataclass(frozen=True)
class ModelSpec:
    """
    High-level scientific model contract intended for model authors.

    This object owns the parts that are genuinely model-specific: model name,
    required unit convention, mass aperture, sampled parameters, metadata, and
    the JAX-compatible scientific formula hooks.  It deliberately does not
    include compiled-context construction, JAX pytree packing, or registry
    boilerplate; those are framework concerns handled by `ModelRuntimeAdapter`.
    """

    name: str
    component_key: str
    required_unit_convention: str
    mass_aperture_kpc: int
    parameters: tuple[ParameterSpec, ...]
    metadata: Mapping[str, str | float | int | bool]
    required_capabilities: tuple[str, ...]
    optional_capabilities: tuple[str, ...]
    static_codes: Mapping[str, int]
    unpack_theta: Callable[[Any], Any]
    validate_theta: Callable[[Any, Any, Any, Mapping[str, int]], Any]
    draw_population: Callable[[Any, Any, Any, Mapping[str, int]], Any]
    selection_weight: Callable[[Any, Any, Any, Any, Mapping[str, int]], Any]
    summary_row: Callable[[Any, Any, Any, Mapping[str, int]], Any]
    lens_integrals: Callable[[Any, Any, Mapping[str, int]], Any]
    extra_prior: Callable[[Any, Any, Mapping[str, int]], tuple[Any, ...]]


@dataclass(frozen=True)
class ContextArraySpec:
    """
    Declare one array field that should move from NumPy context to JAX context.

    `source_name` is the attribute on the model's compiled NumPy context.
    `target_name` is the attribute exposed on the JAX context.  Most fields keep
    the same name; the optional target lets future models use cleaner runtime
    names without changing their HDF5-normalized source objects.
    """

    source_name: str
    target_name: str | None = None

    @property
    def output_name(self) -> str:
        """Return the JAX-context field name used by generated builders."""

        return self.target_name or self.source_name


@dataclass(frozen=True)
class ContextScalarSpec:
    """
    Declare one scalar packed into the JAX `scalar_context` array.

    The order of this tuple is part of the model's numerical contract because
    jitted scientific hooks index the compact scalar array.  Tests should cover
    this order for every production model.
    """

    source_name: str


@dataclass(frozen=True)
class StaticContextSpec:
    """
    Declare one source-context value that becomes a JIT static flag.

    Static flags are kept out of the JAX pytree because they control traced
    branches and therefore belong in the JIT cache key.
    """

    source_name: str
    target_name: str | None = None

    @property
    def output_name(self) -> str:
        """Return the static-key name consumed by model hooks."""

        return self.target_name or self.source_name


@dataclass(frozen=True)
class DataSpec:
    """
    Declarative description of the data a model exposes to the JAX backend.

    This v1 does not try to load arbitrary HDF5 schemas.  A model runtime still
    builds its validated NumPy context, then this spec tells the generic backend
    how to pack that context into the shape expected by scientific hooks.
    """

    jax_context_type: type
    array_fields: tuple[ContextArraySpec, ...]
    scalar_fields: tuple[ContextScalarSpec, ...]
    static_fields: tuple[StaticContextSpec, ...]
    normalization_samples_field: str
    normalization_min_value_field: str
    scalar_context_name: str | None = "scalar_context"


@dataclass(frozen=True)
class CompiledContextBundle:
    """
    Standard result returned by model-specific source-context builders.

    The generic backend uses `context` to build `CompiledModel.context`, while
    the remaining objects preserve the existing metadata and output contracts.
    """

    context: Any
    profile: Any
    cross_section_grid: Any
    cosmology: Any
    random_basis: Any
    observations: Sequence[Any]
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelRuntimeAdapter:
    """
    Engineering adapter that connects a model spec to the production backend.

    A fully generic HDF5 `DataSpec` does not exist yet, so each implemented
    model still needs one runtime adapter that knows how to build a validated
    NumPy source context.  JAX packing and static-flag extraction are now
    declared through `data_spec` and handled by the generic backend.
    """

    build_context_bundle: Callable[[RuntimeConfig], CompiledContextBundle]
    data_spec: DataSpec


@dataclass(frozen=True)
class ModelDefinition:
    """
    Complete registry entry for one concrete scientific model.

    Model modules own scientific formulas and parameter names.  The JAX backend
    owns JIT compilation, vectorized Monte Carlo normalization, posterior
    reduction, and host diagnostics.  The callables below are intentionally
    small hooks so adding another model does not require copying that execution
    framework.
    """

    name: str
    required_capabilities: tuple[str, ...]
    optional_capabilities: tuple[str, ...]
    resolve_mass_definition: Callable[[str], MassDefinition]
    build_parameter_schema: Callable[..., ParameterSchema]
    build_compiled_model: Callable[[RuntimeConfig], CompiledModel]
    static_jit_kwargs: Callable[[CompiledModel], Mapping[str, int]]
    to_jax_context: Callable[[CompiledModel], Any]
    normalization_samples: Callable[[Any], Any]
    normalization_min_value: Callable[[Any], Any]
    unpack_theta: Callable[[Any], Any]
    validate_theta: Callable[[Any, Any, Any, Mapping[str, int]], Any]
    draw_population: Callable[[Any, Any, Any, Mapping[str, int]], Any]
    selection_weight: Callable[[Any, Any, Any, Any, Mapping[str, int]], Any]
    summary_row: Callable[[Any, Any, Any, Mapping[str, int]], Any]
    lens_integrals: Callable[[Any, Any, Mapping[str, int]], Any]
    extra_prior: Callable[[Any, Any, Mapping[str, int]], tuple[Any, ...]]
