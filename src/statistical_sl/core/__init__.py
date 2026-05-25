"""Shared contracts for Statistical_SL workflow stages.

The ``core`` package is intentionally narrow.  It exposes schema names, unit
contracts, mass-definition contracts, manifest names, and artifact names that
multiple workflow stages need to agree on.  It should not grow workflow
orchestration, model-specific runtime code, or backend glue.
"""

from __future__ import annotations

__all__ = [
    "artifacts",
    "canonical_schema",
    "manifests",
    "mass_definition",
    "unit_conventions",
]
