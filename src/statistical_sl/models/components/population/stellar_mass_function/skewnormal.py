"""Skew-normal stellar-mass-function component."""

from __future__ import annotations

from collections.abc import Mapping

from ....interfaces import ParameterSpec
from ...interfaces import ComponentSpec, KernelRef


def skewnormal_stellar_mass_function_component(
    *,
    parameters: tuple[ParameterSpec, ...] = (),
    required_context_fields: tuple[str, ...] = (),
    required_capabilities: tuple[str, ...] = (),
    metadata: Mapping[str, str | float | int | bool] | None = None,
) -> ComponentSpec:
    """Return a reusable skew-normal stellar-mass-function declaration."""

    return ComponentSpec(
        name="population.stellar_mass_function.skewnormal",
        kind="stellar_mass_function",
        parameters=parameters,
        required_context_fields=required_context_fields,
        required_capabilities=required_capabilities,
        required_kernels=(
            KernelRef("distributions", "skewnorm_sample"),
            KernelRef("distributions", "normal_pdf"),
        ),
        metadata=dict(metadata or {}),
    )


__all__ = ["skewnormal_stellar_mass_function_component"]
