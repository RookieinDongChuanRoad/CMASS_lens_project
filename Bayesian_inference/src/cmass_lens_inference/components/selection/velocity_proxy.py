"""Velocity-proxy selection component."""

from __future__ import annotations

from collections.abc import Mapping

from ..interfaces import ComponentSpec, KernelRef


def velocity_proxy_selection_component(
    *,
    required_context_fields: tuple[str, ...] = (),
    required_capabilities: tuple[str, ...] = (),
    metadata: Mapping[str, str | float | int | bool] | None = None,
) -> ComponentSpec:
    """Return a reusable velocity-proxy selection declaration."""

    return ComponentSpec(
        name="selection.velocity_proxy",
        kind="selection_correction",
        required_context_fields=required_context_fields,
        required_capabilities=required_capabilities,
        required_kernels=(KernelRef("selection", "theta_e_est_from_sigma_proxy"),),
        metadata=dict(metadata or {}),
    )


__all__ = ["velocity_proxy_selection_component"]
