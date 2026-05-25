"""Interfaces for reusable scientific model components.

The declarations in this module are intentionally lightweight.  They are not a
runtime DSL and they do not generate Numba kernels.  Their job is to make the
science-facing component contract auditable: parameters, context needs,
capabilities, and metadata live with the component that requires them.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

from ..interfaces import ParameterSpec


@dataclass(frozen=True)
class KernelRef:
    """Reference to a reusable Numba kernel required by a component."""

    module: str
    name: str


@dataclass(frozen=True)
class ComponentSpec:
    """
    Scientific component declaration consumed by model assembly.

    `parameters` is ordered within the component.  A model assembly may choose
    several components and decide the global component-block order, but it
    should not rewrite the names or bounds that belong to those components.
    """

    name: str
    kind: str
    parameters: tuple[ParameterSpec, ...] = ()
    required_context_fields: tuple[str, ...] = ()
    required_capabilities: tuple[str, ...] = ()
    optional_capabilities: tuple[str, ...] = ()
    required_kernels: tuple[KernelRef, ...] = ()
    metadata: Mapping[str, str | float | int | bool] = field(default_factory=dict)


def aggregate_parameters(components: Iterable[ComponentSpec]) -> tuple[ParameterSpec, ...]:
    """
    Concatenate component parameter blocks while rejecting duplicate names.

    Duplicate internal names would make the flat `theta` vector ambiguous.
    Duplicate public names would make config box priors ambiguous.  Failing in
    assembly keeps those mistakes out of sampler and backend code.
    """

    component_tuple = tuple(components)
    if len(component_tuple) == 1:
        return component_tuple[0].parameters

    parameters: list[ParameterSpec] = []
    seen_internal: set[str] = set()
    seen_public: set[str] = set()
    for component in component_tuple:
        for parameter in component.parameters:
            if parameter.internal_name in seen_internal:
                raise ValueError(f"Duplicate internal parameter '{parameter.internal_name}'.")
            if parameter.public_name in seen_public:
                raise ValueError(f"Duplicate public parameter '{parameter.public_name}'.")
            seen_internal.add(parameter.internal_name)
            seen_public.add(parameter.public_name)
            parameters.append(parameter)
    return tuple(parameters)


def _aggregate_unique_ordered(values: Iterable[str]) -> tuple[str, ...]:
    """Return values in first-seen order, removing duplicates."""

    ordered: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return tuple(ordered)


def aggregate_required_capabilities(
    components: Iterable[ComponentSpec],
    *,
    extra: Iterable[str] = (),
) -> tuple[str, ...]:
    """Aggregate required capabilities from selected components."""

    component_tuple = tuple(components)
    return _aggregate_unique_ordered(
        [
            *(
                capability
                for component in component_tuple
                for capability in component.required_capabilities
            ),
            *_aggregate_unique_ordered(extra),
        ]
    )


def aggregate_optional_capabilities(
    components: Iterable[ComponentSpec],
    *,
    extra: Iterable[str] = (),
) -> tuple[str, ...]:
    """Aggregate optional capabilities from selected components."""

    component_tuple = tuple(components)
    return _aggregate_unique_ordered(
        [
            *(
                capability
                for component in component_tuple
                for capability in component.optional_capabilities
            ),
            *_aggregate_unique_ordered(extra),
        ]
    )


__all__ = [
    "ComponentSpec",
    "KernelRef",
    "aggregate_optional_capabilities",
    "aggregate_parameters",
    "aggregate_required_capabilities",
]
