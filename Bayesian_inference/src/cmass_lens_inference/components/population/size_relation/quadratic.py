"""Quadratic size-relation component."""

from __future__ import annotations

from collections.abc import Mapping

from ....model_interfaces import ParameterSpec
from ...interfaces import ComponentSpec, KernelRef


def quadratic_size_relation_component(
    *,
    parameters: tuple[ParameterSpec, ...] = (),
    required_context_fields: tuple[str, ...] = (),
    required_capabilities: tuple[str, ...] = (),
    metadata: Mapping[str, str | float | int | bool] | None = None,
) -> ComponentSpec:
    """Return a reusable quadratic size-relation declaration."""

    return ComponentSpec(
        name="population.size_relation.quadratic",
        kind="size_relation",
        parameters=parameters,
        required_context_fields=required_context_fields,
        required_capabilities=required_capabilities,
        required_kernels=(KernelRef("population", "quadratic_size_relation_mean"),),
        metadata=dict(metadata or {}),
    )


__all__ = ["quadratic_size_relation_component"]
