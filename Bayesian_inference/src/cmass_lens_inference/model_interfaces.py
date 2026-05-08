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
    `ParameterSchema` used by config parsing, sampler initialization, and
    output metadata.
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
    the backend implementation key.  It deliberately does not include
    compiled-context construction, sampler wiring, output writing, or registry
    boilerplate; those are framework concerns handled by `ModelRuntimeAdapter`
    and the active production backend.
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
    backend_kernel: str


@dataclass(frozen=True)
class ContextArraySpec:
    """
    Declare one array field that a backend may copy from the source context.

    `source_name` is the attribute on the model's compiled NumPy context.
    `target_name` is the attribute exposed on a backend-owned context when that
    backend needs a separate packed object.  Most fields keep the same name; the
    optional target lets future models use cleaner runtime names without
    changing their HDF5-normalized source objects.
    """

    source_name: str
    target_name: str | None = None

    @property
    def output_name(self) -> str:
        """Return the backend-context field name used by generated builders."""

        return self.target_name or self.source_name


@dataclass(frozen=True)
class ContextScalarSpec:
    """
    Declare one scalar packed into a backend scalar-context array.

    The order of this tuple is part of the model's numerical contract because
    compiled kernels may index the compact scalar array.  Tests should cover
    this order for every production model.
    """

    source_name: str


@dataclass(frozen=True)
class StaticContextSpec:
    """
    Declare one source-context value that becomes a static backend flag.

    Static flags are kept out of compact numeric arrays because they control
    branches and therefore belong in backend setup rather than sampled theta.
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
    Declarative description of the data a model exposes to backend kernels.

    This v1 does not try to load arbitrary HDF5 schemas.  A model runtime still
    builds its validated NumPy context, then this spec tells backend adapters
    how to pack that context into the shape expected by scientific kernels.
    """

    backend_context_type: type
    array_fields: tuple[ContextArraySpec, ...]
    scalar_fields: tuple[ContextScalarSpec, ...]
    static_fields: tuple[StaticContextSpec, ...]
    normalization_samples_field: str
    normalization_min_value_field: str


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
    NumPy source context.  Context packing and static-flag extraction are now
    declared through `data_spec` and handled by the active backend.
    """

    build_context_bundle: Callable[[RuntimeConfig], CompiledContextBundle]
    data_spec: DataSpec


@dataclass(frozen=True)
class ModelDefinition:
    """
    Complete registry entry for one concrete scientific model.

    Model modules own scientific declarations and parameter names.  The active
    production backend owns compilation, posterior reduction, and diagnostics.
    `evaluate_log_prob` is the model-owned posterior adapter.  Keeping the
    callable on the registry definition means the backend likelihood engine no
    longer grows a new dispatch branch whenever a model package is added.
    """

    name: str
    required_capabilities: tuple[str, ...]
    optional_capabilities: tuple[str, ...]
    resolve_mass_definition: Callable[[str], MassDefinition]
    build_parameter_schema: Callable[..., ParameterSchema]
    build_compiled_model: Callable[[RuntimeConfig], CompiledModel]
    evaluate_log_prob: Callable[[Any, CompiledModel, float], tuple[float, Any]]
    backend_kernel: str
