from __future__ import annotations

from pathlib import Path

import yaml


def test_workspace_layout_exists_with_canonical_runtime_boundaries() -> None:
    """
    Phase 4 requires a concrete workspace boundary, not only a design note.

    The tracked files inside ``workspace/data`` and ``workspace/outputs`` are
    documentation sentinels.  Real datasets, caches, and run outputs remain
    ignored by git, but the directory contract must still be visible to users
    and automation.
    """

    repository_root = Path(__file__).resolve().parents[1]
    workspace_root = repository_root / "workspace"
    required_directories = (
        workspace_root / "configs" / "data_preparation",
        workspace_root / "configs" / "inference",
        workspace_root / "configs" / "posterior_predictive",
        workspace_root / "recipes",
        workspace_root / "scripts",
        workspace_root / "notebooks",
        workspace_root / "reports",
        workspace_root / "data",
        workspace_root / "outputs",
    )

    for directory_path in required_directories:
        assert directory_path.is_dir(), directory_path

    assert (workspace_root / "data" / "README.md").is_file()
    assert (workspace_root / "outputs" / "README.md").is_file()


def test_cmass_pipeline_recipe_references_existing_step_configs() -> None:
    """
    The pipeline recipe is the user-facing orchestration contract.

    This test deliberately validates only structural fields.  It should not run
    the expensive scientific workflow; it proves that the recipe declares a
    workspace root, references real single-step configs, and keeps inference
    artifacts grouped under one run directory before PPC diagnostics write
    their subdirectory.
    """

    repository_root = Path(__file__).resolve().parents[1]
    recipe_path = repository_root / "workspace" / "recipes" / "cmass" / "sersic_diagnostics.yaml"
    recipe = yaml.safe_load(recipe_path.read_text(encoding="utf-8"))

    assert recipe["schema_version"] == "statistical_sl_pipeline_v1"
    assert recipe["workspace_root"] == "../.."
    assert set(recipe["steps"]) == {"data_preparation", "inference", "posterior_predictive"}

    recipe_dir = recipe_path.parent
    for step_payload in recipe["steps"].values():
        config_path = (recipe_dir / step_payload["config"]).resolve()
        assert config_path.is_file(), config_path

    inference_output_run_dir = recipe["steps"]["inference"]["output_run_dir"]
    assert inference_output_run_dir == "outputs/cmass/${run_id}"
    assert recipe["steps"]["posterior_predictive"]["run_dir"] == "${steps.inference.output_run_dir}"
    assert (
        recipe["steps"]["posterior_predictive"]["output_dir"]
        == "${steps.inference.output_run_dir}/posterior_predictive/diagnostics/${diagnostic_run_id}"
    )
