"""Nonnegative truncated Gaussian source-redshift component."""

from __future__ import annotations

from collections.abc import Mapping

from ....model_interfaces import ParameterSpec
from ...interfaces import ComponentSpec, KernelRef


def truncated_nonnegative_gaussian_source_redshift_component(
    *,
    parameters: tuple[ParameterSpec, ...],
    required_context_fields: tuple[str, ...] = (),
    required_capabilities: tuple[str, ...] = (),
    metadata: Mapping[str, str | float | int | bool] | None = None,
) -> ComponentSpec:
    """Return a reusable nonnegative truncated Gaussian source-redshift declaration."""

    return ComponentSpec(
        name="population.source_redshift.truncated_nonnegative_gaussian",
        kind="source_redshift_density",
        parameters=parameters,
        required_context_fields=required_context_fields,
        required_capabilities=required_capabilities,
        required_kernels=(
            KernelRef("selection_likelihood", "truncated_nonnegative_source_redshift_density"),
        ),
        metadata=dict(metadata or {}),
    )


__all__ = ["truncated_nonnegative_gaussian_source_redshift_component"]
