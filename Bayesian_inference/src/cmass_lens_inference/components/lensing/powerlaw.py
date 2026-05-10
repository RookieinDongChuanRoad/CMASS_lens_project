"""Power-law lensing geometry component."""

from __future__ import annotations

from collections.abc import Mapping

from ..interfaces import ComponentSpec, KernelRef


def powerlaw_lensing_component(
    *,
    required_context_fields: tuple[str, ...] = (),
    required_capabilities: tuple[str, ...] = (),
    metadata: Mapping[str, str | float | int | bool] | None = None,
) -> ComponentSpec:
    """Return a reusable power-law lensing geometry declaration."""

    return ComponentSpec(
        name="lensing.powerlaw",
        kind="lensing_geometry",
        required_context_fields=required_context_fields,
        required_capabilities=required_capabilities,
        required_kernels=(KernelRef("lensing", "theta_ein_arcsec"),),
        metadata=dict(metadata or {}),
    )


__all__ = ["powerlaw_lensing_component"]
