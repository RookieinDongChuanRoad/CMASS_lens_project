"""Gaussian linear aperture-mass-relation component."""

from __future__ import annotations

from collections.abc import Mapping

from ....interfaces import ParameterSpec
from ...interfaces import ComponentSpec, KernelRef


def gaussian_linear_aperture_mass_component(
    *,
    parameters: tuple[ParameterSpec, ...],
    required_context_fields: tuple[str, ...] = (),
    required_capabilities: tuple[str, ...] = (),
    metadata: Mapping[str, str | float | int | bool] | None = None,
) -> ComponentSpec:
    """Return a reusable Gaussian linear aperture-mass relation declaration."""

    return ComponentSpec(
        name="population.aperture_mass_relation.gaussian_linear",
        kind="aperture_mass_relation",
        parameters=parameters,
        required_context_fields=required_context_fields,
        required_capabilities=required_capabilities,
        required_kernels=(KernelRef("population", "gaussian_linear_mass_mean"),),
        metadata=dict(metadata or {}),
    )


__all__ = ["gaussian_linear_aperture_mass_component"]
