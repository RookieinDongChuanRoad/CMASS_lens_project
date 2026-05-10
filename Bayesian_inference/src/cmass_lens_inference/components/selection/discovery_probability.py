"""Survey discovery probability component."""

from __future__ import annotations

from collections.abc import Mapping

from ...model_interfaces import ParameterSpec
from ..interfaces import ComponentSpec, KernelRef


def discovery_probability_component(
    *,
    parameters: tuple[ParameterSpec, ...],
    required_context_fields: tuple[str, ...] = (),
    required_capabilities: tuple[str, ...] = (),
    metadata: Mapping[str, str | float | int | bool] | None = None,
) -> ComponentSpec:
    """Return a reusable sigmoid discovery-probability declaration."""

    return ComponentSpec(
        name="selection.discovery_probability.sigmoid",
        kind="selection_function",
        parameters=parameters,
        required_context_fields=required_context_fields,
        required_capabilities=required_capabilities,
        required_kernels=(KernelRef("selection", "p_find"),),
        metadata=dict(metadata or {}),
    )


__all__ = ["discovery_probability_component"]
