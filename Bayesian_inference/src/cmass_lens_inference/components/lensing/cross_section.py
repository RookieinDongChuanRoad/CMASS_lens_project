"""Theta-gamma cross-section component."""

from __future__ import annotations

from collections.abc import Mapping

from ..interfaces import ComponentSpec, KernelRef


def theta_gamma_cross_section_component(
    *,
    required_context_fields: tuple[str, ...] = (),
    required_capabilities: tuple[str, ...] = (),
    metadata: Mapping[str, str | float | int | bool] | None = None,
) -> ComponentSpec:
    """Return a reusable theta-gamma cross-section declaration."""

    return ComponentSpec(
        name="lensing.cross_section.theta_gamma",
        kind="lensing_cross_section",
        required_context_fields=required_context_fields,
        required_capabilities=required_capabilities,
        required_kernels=(KernelRef("selection_likelihood", "cross_section_find_weight"),),
        metadata=dict(metadata or {}),
    )


__all__ = ["theta_gamma_cross_section_component"]
