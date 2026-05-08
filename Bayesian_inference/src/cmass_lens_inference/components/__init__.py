"""Reusable scientific component declarations."""

from __future__ import annotations

from .interfaces import (
    ComponentSpec,
    KernelRef,
    aggregate_optional_capabilities,
    aggregate_parameters,
    aggregate_required_capabilities,
)

__all__ = [
    "ComponentSpec",
    "KernelRef",
    "aggregate_optional_capabilities",
    "aggregate_parameters",
    "aggregate_required_capabilities",
]
