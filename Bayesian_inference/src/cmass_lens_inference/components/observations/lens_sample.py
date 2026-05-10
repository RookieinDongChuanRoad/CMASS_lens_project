"""Observed lens-sample component."""

from __future__ import annotations

from collections.abc import Mapping

from ..interfaces import ComponentSpec


def lens_sample_component(
    *,
    required_context_fields: tuple[str, ...] = (),
    required_capabilities: tuple[str, ...] = (),
    metadata: Mapping[str, str | float | int | bool] | None = None,
) -> ComponentSpec:
    """Return a generic observed lens-sample component declaration."""

    return ComponentSpec(
        name="observations.lens_sample",
        kind="observed_lens_sample",
        required_context_fields=required_context_fields,
        required_capabilities=required_capabilities,
        metadata=dict(metadata or {}),
    )


__all__ = ["lens_sample_component"]
