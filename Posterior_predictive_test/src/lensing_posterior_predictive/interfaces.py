"""Typed contracts for model-aware posterior-predictive dispatch.

The PPT package owns workflow concerns: reading a completed run directory,
selecting posterior draws, and writing diagnostic artifacts.  A scientific
model owns how its posterior draw maps to a replicated lens population.  This
module defines the thin boundary between those two responsibilities so the
workflow can dispatch by ``model.name`` without importing concrete model
packages in generic orchestration code.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from cmass_lens_inference.mass_definition import MassDefinition
from cmass_lens_inference.types import RuntimeConfig


PPCContextBundle = tuple[Any, Any, Any, Any, Any, list[Any]]


@dataclass(frozen=True)
class PredictiveDefinition:
    """
    Registry entry for one model's posterior-predictive capability.

    The definition is intentionally small at this stage.  It records the public
    diagnostics contract and exposes the model-owned context builder used by
    existing diagnostics.  Later phases can extend the same boundary with a
    full predictive engine that returns model-specific payload schemas.
    """

    model_name: str
    backend: str
    supported_diagnostics: tuple[str, ...]
    required_external_inputs: tuple[str, ...]
    artifact_schema_version: str
    build_context: Callable[[RuntimeConfig], PPCContextBundle]
    run_diagnostics: Callable[..., dict[str, Any]]
    trend_category_names: tuple[str, ...]
    build_trend_panel_order: Callable[[MassDefinition], tuple[str, ...]]


__all__ = ["PPCContextBundle", "PredictiveDefinition"]
