"""
Configuration loading for posterior diagnostics workflows.

This module gives posterior diagnostics the same durable config surface that
inference already has. The config intentionally stores reusable execution
settings separately from the concrete inference run directory: a run directory
is produced by inference and is often known only after a pipeline run starts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


SUPPORTED_SCHEMA_VERSION = "statistical_sl_posterior_predictive_config_v1"
SUPPORTED_WORKFLOW = "posterior_diagnostics"
DEFAULT_DIAGNOSTICS_MASS_BIN_COUNT = 19
DEFAULT_DIAGNOSTICS_MASS_BIN_MIN = 10.15
DEFAULT_DIAGNOSTICS_MASS_BIN_MAX = 12.05
DEFAULT_DIAGNOSTICS_PARENT_SAMPLE_SIZE = 10000


def _require_mapping(payload: dict[str, Any], section_name: str) -> dict[str, Any]:
    """Return a required YAML mapping section with a direct error message."""

    section = payload.get(section_name)
    if not isinstance(section, dict):
        raise TypeError(f"Posterior diagnostics config section '{section_name}' must be a mapping.")
    return section


def _optional_path(raw_value: object) -> Path | None:
    """Normalize optional path values while preserving omitted inputs as None."""

    if raw_value is None:
        return None
    text = str(raw_value)
    if "${" in text:
        return None
    return Path(text).expanduser().resolve()


@dataclass(frozen=True)
class PosteriorDiagnosticsConfig:
    """Typed settings needed to run one posterior diagnostics workflow."""

    path: Path
    model_name: str
    profile_name: str
    inference_run_dir: Path | None
    sigma_table_path: Path | None
    output_root_dir: Path
    result_dir_template: str | None
    n_posterior_draws: int | None
    burn_in: str | int
    random_seed: int
    parent_sample_size: int
    worker_processes: int | None
    n_mass_bins: int
    mass_bin_min: float
    mass_bin_max: float

    def to_run_kwargs(
        self,
        *,
        run_dir_override: str | Path | None = None,
        diagnostic_run_id: str | None = None,
    ) -> dict[str, object]:
        """Return kwargs accepted by `run_posterior_diagnostics`."""

        run_dir = Path(run_dir_override).expanduser().resolve() if run_dir_override is not None else self.inference_run_dir
        if run_dir is None:
            raise ValueError(
                "Posterior diagnostics require an inference run directory. "
                "Set inputs.inference_run_dir in the config or pass --run-dir."
            )

        return {
            "run_dir": str(run_dir),
            "sigma_table_path": self.sigma_table_path,
            "output_root_dir": self.output_root_dir,
            "diagnostic_run_id": diagnostic_run_id,
            "n_posterior_draws": self.n_posterior_draws,
            "burn_in": self.burn_in,
            "random_seed": self.random_seed,
            "parent_sample_size": self.parent_sample_size,
            "worker_processes": self.worker_processes,
            "n_mass_bins": self.n_mass_bins,
            "mass_bin_min": self.mass_bin_min,
            "mass_bin_max": self.mass_bin_max,
        }


def load_posterior_diagnostics_config(config_path: str | Path) -> PosteriorDiagnosticsConfig:
    """Load and validate one posterior diagnostics YAML config."""

    path = Path(config_path).expanduser().resolve()
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("Posterior diagnostics config must be a YAML mapping.")
    if payload.get("schema_version") != SUPPORTED_SCHEMA_VERSION:
        raise ValueError(f"Unsupported posterior diagnostics config schema: {payload.get('schema_version')!r}")
    if payload.get("workflow") != SUPPORTED_WORKFLOW:
        raise ValueError(f"Unsupported posterior diagnostics workflow: {payload.get('workflow')!r}")

    model = _require_mapping(payload, "model")
    profile = _require_mapping(payload, "profile")
    inputs = _require_mapping(payload, "inputs")
    execution = _require_mapping(payload, "execution")
    outputs = _require_mapping(payload, "outputs")

    return PosteriorDiagnosticsConfig(
        path=path,
        model_name=str(model["name"]),
        profile_name=str(profile["name"]),
        inference_run_dir=_optional_path(inputs.get("inference_run_dir")),
        sigma_table_path=_optional_path(inputs.get("sigma_table_path")),
        output_root_dir=Path(outputs.get("output_root_dir", "workspace/outputs")).expanduser().resolve(),
        result_dir_template=None if outputs.get("result_dir") is None else str(outputs["result_dir"]),
        n_posterior_draws=(
            None
            if execution.get("n_posterior_draws") is None
            else int(execution["n_posterior_draws"])
        ),
        burn_in=execution.get("burn_in", "auto"),
        random_seed=int(execution.get("random_seed", 20260309)),
        parent_sample_size=int(execution.get("parent_sample_size", DEFAULT_DIAGNOSTICS_PARENT_SAMPLE_SIZE)),
        worker_processes=(
            None
            if execution.get("worker_processes") is None
            else int(execution["worker_processes"])
        ),
        n_mass_bins=int(execution.get("n_mass_bins", DEFAULT_DIAGNOSTICS_MASS_BIN_COUNT)),
        mass_bin_min=float(execution.get("mass_bin_min", DEFAULT_DIAGNOSTICS_MASS_BIN_MIN)),
        mass_bin_max=float(execution.get("mass_bin_max", DEFAULT_DIAGNOSTICS_MASS_BIN_MAX)),
    )
