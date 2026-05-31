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


def test_workspace_post_canonical_recipes_reference_existing_step_configs() -> None:
    """
    Post-canonical recipes are the current audited inference + diagnostics slice.

    Full recipes that include data preparation may exist separately. This test
    only validates recipes explicitly marked `mode: post_canonical`.
    """

    repository_root = Path(__file__).resolve().parents[1]
    recipe_root = repository_root / "workspace" / "recipes"
    recipe_paths = []
    for candidate in sorted(recipe_root.glob("**/*.yaml")):
        payload = yaml.safe_load(candidate.read_text(encoding="utf-8"))
        if payload.get("mode") == "post_canonical":
            recipe_paths.append(candidate)

    assert recipe_paths, "No post-canonical workspace pipeline recipes found."

    for recipe_path in recipe_paths:
        recipe = yaml.safe_load(recipe_path.read_text(encoding="utf-8"))
        assert recipe["schema_version"] == "statistical_sl_pipeline_v1"
        assert recipe["workspace_root"] == "../.."
        assert set(recipe["steps"]) == {"inference", "posterior_predictive"}

        recipe_dir = recipe_path.parent
        inference_step = recipe["steps"]["inference"]
        posterior_step = recipe["steps"]["posterior_predictive"]

        inference_config_path = (recipe_dir / inference_step["config"]).resolve()
        posterior_config_path = (recipe_dir / posterior_step["config"]).resolve()
        dataset_path = (repository_root / "workspace" / inference_step["dataset"]).resolve()

        assert inference_config_path.is_file(), inference_config_path
        assert posterior_config_path.is_file(), posterior_config_path
        assert dataset_path.is_file(), dataset_path
        assert posterior_step["run_dir"] == "${steps.inference.output_run_dir}"
        assert posterior_step["output_root_dir"] == "outputs"
        assert posterior_step["result_dir"] == (
            "${steps.inference.output_run_dir}/posterior_predictive/diagnostics/${diagnostic_run_id}"
        )


def test_workspace_inference_configs_cover_current_production_models() -> None:
    """
    The workspace is now the public run surface for production inference.

    The historical standalone inference-config tree is gone, so every
    production model that the pipeline skill can mention must have a concrete
    workspace config.  The configs also need to point at workspace-owned data
    and output roots so new runs do not silently keep writing to the old
    top-level ``data`` / ``outputs`` layout.
    """

    repository_root = Path(__file__).resolve().parents[1]
    config_root = repository_root / "workspace" / "configs" / "inference"
    expected_configs = (
        config_root / "cmass" / "devauc.yaml",
        config_root / "cmass" / "sersic.yaml",
        config_root / "sonnenfeld2024_slacs" / "sonnenfeld2024_slacs.yaml",
        config_root / "sonnenfeld2024_slacs" / "sonnenfeld2024_slacs_hunit.yaml",
        config_root / "sonnenfeld2024_slacs" / "sonnenfeld2024_slacs_sigma_star_gamma.yaml",
        config_root / "sonnenfeld2024_slacs" / "sonnenfeld2024_slacs_sigma_star_gamma_hunit.yaml",
    )

    for config_path in expected_configs:
        assert config_path.is_file(), config_path
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        dataset_path = Path(payload["data"]["inference_dataset_path"])
        output_root = Path(payload["output"]["root_dir"])

        assert dataset_path.parts[:3] == ("workspace", "data", "canonical")
        assert output_root == Path("workspace/outputs")
        old_config_root = "Bayesian" + "_inference"
        assert old_config_root not in config_path.read_text(encoding="utf-8")
