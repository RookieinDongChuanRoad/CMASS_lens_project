"""
Pipeline recipe parsing for Statistical_SL workflows.

Recipes are the user-facing orchestration contract. This parser supports the
post-canonical recipes repaired in this plan, while deliberately not banning
full recipes that include data-preparation steps for future end-to-end runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


SUPPORTED_SCHEMA_VERSION = "statistical_sl_pipeline_v1"


@dataclass(frozen=True)
class PipelineRecipe:
    """Resolved pipeline recipe."""

    path: Path
    name: str
    mode: str
    workspace_root: Path
    steps: dict[str, dict[str, Any]]
    inference_config_path: Path
    posterior_diagnostics_config_path: Path
    dataset_path: Path
    inference_output_run_dir_template: str
    diagnostics_result_dir_template: str


def _resolve_workspace_root(recipe_path: Path, raw_workspace_root: object) -> Path:
    """Resolve recipe-local workspace roots such as '../..'."""

    return (recipe_path.parent / str(raw_workspace_root)).resolve()


def _resolve_recipe_relative_path(recipe_path: Path, raw_path: object) -> Path:
    """Resolve config paths relative to the recipe file location."""

    return (recipe_path.parent / str(raw_path)).resolve()


def _require_step(steps: dict[str, Any], step_name: str) -> dict[str, Any]:
    """Return a required step mapping with a clear error message."""

    step = steps.get(step_name)
    if not isinstance(step, dict):
        raise ValueError(f"Pipeline recipe requires a mapping step named '{step_name}'.")
    return step


def load_pipeline_recipe(recipe_path: str | Path) -> PipelineRecipe:
    """Load and validate one pipeline recipe."""

    path = Path(recipe_path).expanduser().resolve()
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("Pipeline recipe must be a YAML mapping.")
    if payload.get("schema_version") != SUPPORTED_SCHEMA_VERSION:
        raise ValueError(f"Unsupported pipeline recipe schema: {payload.get('schema_version')!r}")

    steps = payload.get("steps")
    if not isinstance(steps, dict):
        raise ValueError("Pipeline recipe requires a mapping 'steps' section.")

    inference_step = _require_step(steps, "inference")
    posterior_step = _require_step(steps, "posterior_predictive")
    workspace_root = _resolve_workspace_root(path, payload["workspace_root"])
    inference_config_path = _resolve_recipe_relative_path(path, inference_step["config"])
    posterior_config_path = _resolve_recipe_relative_path(path, posterior_step["config"])
    dataset_path = workspace_root / str(inference_step["dataset"])

    missing_paths = [
        candidate
        for candidate in (workspace_root, inference_config_path, posterior_config_path, dataset_path)
        if not candidate.exists()
    ]
    if missing_paths:
        formatted = "\n".join(str(candidate) for candidate in missing_paths)
        raise FileNotFoundError(f"Pipeline recipe references missing paths:\n{formatted}")

    return PipelineRecipe(
        path=path,
        name=str(payload["name"]),
        mode=str(payload.get("mode", "full")),
        workspace_root=workspace_root,
        steps={key: dict(value) for key, value in steps.items()},
        inference_config_path=inference_config_path,
        posterior_diagnostics_config_path=posterior_config_path,
        dataset_path=dataset_path,
        inference_output_run_dir_template=str(inference_step["output_run_dir"]),
        diagnostics_result_dir_template=str(posterior_step["result_dir"]),
    )
