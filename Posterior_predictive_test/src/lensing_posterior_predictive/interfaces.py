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
class DiagnosticsExecution:
    """
    Resolved execution policy for one PPC diagnostics run.

    The generic PPC layer resolves CPU budget and records artifact metadata.
    Model adapters own how that budget is consumed. The default policy is
    Numba kernel parallelism inside the adapter, not generic Python process
    parallelism around model-specific code.

    Fields:
    - ``strategy`` names the resolved execution mode recorded for artifacts.
    - ``cpu_count`` is the detected CPU count before PPC-specific reservation.
    - ``reserve_cores`` is the core reservation that participates in the
      compute-budget calculation.
    - ``compute_budget`` is the usable CPU budget after subtracting reserved
      cores and applying any configured upper limit.
    - ``requested_worker_processes`` is the user-facing process request;
      ``None`` means the PPC layer should choose automatically.
    - ``worker_processes`` is the resolved Python process-pool size; ``0``
      means no Python process pool under a kernel-only strategy.
    - ``kernel_threads_per_process`` is the Numba or math-library thread count
      that model adapters should use inside each diagnostics process.
    """

    strategy: str
    cpu_count: int
    reserve_cores: int
    compute_budget: int
    requested_worker_processes: int | None
    worker_processes: int
    kernel_threads_per_process: int

    def to_dict(self) -> dict[str, int | str | None]:
        """Serialize diagnostics execution metadata into PPC artifacts."""
        return {
            "strategy": self.strategy,
            "cpu_count": self.cpu_count,
            "reserve_cores": self.reserve_cores,
            "compute_budget": self.compute_budget,
            "requested_worker_processes": self.requested_worker_processes,
            "worker_processes": self.worker_processes,
            "kernel_threads_per_process": self.kernel_threads_per_process,
        }


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


__all__ = ["DiagnosticsExecution", "PPCContextBundle", "PredictiveDefinition"]
