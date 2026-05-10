"""Sigma-star density-slope relation component."""

from __future__ import annotations

from collections.abc import Mapping

from ....model_interfaces import ParameterSpec
from ...interfaces import ComponentSpec, KernelRef


def sigma_star_linear_gamma_component(
    *,
    parameters: tuple[ParameterSpec, ...],
    required_context_fields: tuple[str, ...] = (),
    required_capabilities: tuple[str, ...] = (),
    metadata: Mapping[str, str | float | int | bool] | None = None,
) -> ComponentSpec:
    """Return a reusable sigma-star density-slope relation declaration."""

    return ComponentSpec(
        name="population.gamma_relation.sigma_star_linear",
        kind="gamma_relation",
        parameters=parameters,
        required_context_fields=required_context_fields,
        required_capabilities=required_capabilities,
        required_kernels=(KernelRef("population", "sigma_star_linear_gamma_mean"),),
        metadata=dict(metadata or {}),
    )


__all__ = ["sigma_star_linear_gamma_component"]
