"""Smoothly truncated Schechter stellar-mass-function component."""

from __future__ import annotations

from collections.abc import Mapping

from ....interfaces import ParameterSpec
from ...interfaces import ComponentSpec, KernelRef


def smooth_truncated_schechter_component(
    *,
    parameters: tuple[ParameterSpec, ...] = (),
    required_context_fields: tuple[str, ...] = (),
    required_capabilities: tuple[str, ...] = (),
    metadata: Mapping[str, str | float | int | bool] | None = None,
) -> ComponentSpec:
    """Return a reusable smooth-truncated Schechter density declaration."""

    return ComponentSpec(
        name="population.stellar_mass_function.smooth_truncated_schechter",
        kind="stellar_mass_function",
        parameters=parameters,
        required_context_fields=required_context_fields,
        required_capabilities=required_capabilities,
        required_kernels=(
            KernelRef("population", "smooth_truncated_schechter_density"),
        ),
        metadata=dict(metadata or {}),
    )


__all__ = ["smooth_truncated_schechter_component"]
