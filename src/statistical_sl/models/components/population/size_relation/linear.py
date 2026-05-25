"""Linear size-relation component."""

from __future__ import annotations

from collections.abc import Mapping

from ....interfaces import ParameterSpec
from ...interfaces import ComponentSpec, KernelRef


def linear_size_relation_component(
    *,
    parameters: tuple[ParameterSpec, ...] = (),
    required_context_fields: tuple[str, ...] = (),
    required_capabilities: tuple[str, ...] = (),
    metadata: Mapping[str, str | float | int | bool] | None = None,
) -> ComponentSpec:
    """Return a reusable linear size-relation declaration."""

    return ComponentSpec(
        name="population.size_relation.linear",
        kind="size_relation",
        parameters=parameters,
        required_context_fields=required_context_fields,
        required_capabilities=required_capabilities,
        required_kernels=(KernelRef("population", "linear_size_relation_mean"),),
        metadata=dict(metadata or {}),
    )


__all__ = ["linear_size_relation_component"]
