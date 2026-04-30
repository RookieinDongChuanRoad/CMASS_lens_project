"""
Typed result contracts for the standalone posterior-predictive package.

These dataclasses deliberately live outside `cmass_lens_inference` because the
user wants `Bayesian_inference` to remain focused on the inference engine. The
standalone PPT package still depends on the inference core, but its workflow
results are now owned and versioned here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class PosteriorPredictiveResult:
    """Structured summary of one posterior predictive execution."""

    run_id: str
    profile_name: str
    input_run_dir: Path
    result_dir: Path
    status: str
    burn_in_applied: int
    n_replicates: int
    sample_sizes: dict[str, int]
    sigma_table_path: Path
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the result into JSON-friendly data."""

        payload = asdict(self)
        for key, value in list(payload.items()):
            if isinstance(value, Path):
                payload[key] = str(value)
        return payload

@dataclass
class PosteriorPredictiveMonitorResult:
    """Structured summary of the external-table monitor workflow."""

    status: str
    external_dir: Path
    not_before: str
    devauc_table_path: Path
    sersic_table_path: Path
    devauc_table_mtime: str
    sersic_table_mtime: str
    devauc_result: PosteriorPredictiveResult | None = None
    sersic_result: PosteriorPredictiveResult | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the nested monitor result into JSON-friendly data."""

        return {
            "status": self.status,
            "external_dir": str(self.external_dir),
            "not_before": self.not_before,
            "devauc_table_path": str(self.devauc_table_path),
            "sersic_table_path": str(self.sersic_table_path),
            "devauc_table_mtime": self.devauc_table_mtime,
            "sersic_table_mtime": self.sersic_table_mtime,
            "devauc_result": None if self.devauc_result is None else self.devauc_result.to_dict(),
            "sersic_result": None if self.sersic_result is None else self.sersic_result.to_dict(),
            "metadata": self.metadata,
        }


@dataclass
class PosteriorTrendResult:
    """Structured summary of one posterior trend evaluation."""

    run_id: str
    profile_name: str
    input_run_dir: Path
    result_dir: Path
    status: str
    burn_in_applied: int
    n_posterior_draws: int
    n_mass_bins: int
    sigma_table_path: Path
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the result into JSON-friendly data."""

        payload = asdict(self)
        for key, value in list(payload.items()):
            if isinstance(value, Path):
                payload[key] = str(value)
        return payload


@dataclass
class PosteriorDiagnosticsResult:
    """Structured summary of one joint PPC + trend diagnostics execution."""

    run_id: str
    profile_name: str
    input_run_dir: Path
    result_dir: Path
    status: str
    burn_in_applied: int
    n_posterior_draws: int
    parent_sample_size: int
    n_mass_bins: int
    sigma_table_path: Path
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the result into JSON-friendly data."""

        payload = asdict(self)
        for key, value in list(payload.items()):
            if isinstance(value, Path):
                payload[key] = str(value)
        return payload


@dataclass
class Fig8ObservationAnnotationResult:
    """Structured summary of the in-place Fig. 8 observation-overlay workflow."""

    status: str
    outputs_root: Path
    processed_run_count: int
    processed_runs: list[dict[str, Any]] = field(default_factory=list)
    skipped_runs: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the result into JSON-friendly data."""

        payload = asdict(self)
        for key, value in list(payload.items()):
            if isinstance(value, Path):
                payload[key] = str(value)
        return payload
