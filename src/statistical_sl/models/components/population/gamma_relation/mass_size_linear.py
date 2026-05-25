"""Mass-and-size-residual density-slope relation component."""

from __future__ import annotations

from collections.abc import Mapping

from ....interfaces import ParameterSpec
from ...interfaces import ComponentSpec, KernelRef


def mass_size_linear_gamma_component(
    *,
    parameters: tuple[ParameterSpec, ...],
    required_context_fields: tuple[str, ...] = (),
    required_capabilities: tuple[str, ...] = (),
    metadata: Mapping[str, str | float | int | bool] | None = None,
) -> ComponentSpec:
    """Return a reusable mass/size-residual density-slope relation declaration."""

    return ComponentSpec(
        name="population.gamma_relation.mass_size_linear",
        kind="gamma_relation",
        parameters=parameters,
        required_context_fields=required_context_fields,
        required_capabilities=required_capabilities,
        required_kernels=(KernelRef("population", "mass_size_linear_gamma_mean"),),
        metadata=dict(metadata or {}),
    )


__all__ = ["mass_size_linear_gamma_component"]
