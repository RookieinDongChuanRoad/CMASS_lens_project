from __future__ import annotations

from pathlib import Path


def test_load_pipeline_recipe_resolves_post_canonical_steps() -> None:
    """A post-canonical recipe should resolve configs without running data prep."""

    from statistical_sl.pipeline.recipe import load_pipeline_recipe

    recipe = load_pipeline_recipe("workspace/recipes/cmass/devauc_diagnostics_from_canonical.yaml")

    assert recipe.name == "cmass_devauc_diagnostics_from_canonical"
    assert recipe.mode == "post_canonical"
    assert recipe.workspace_root == Path("workspace").resolve()
    assert set(recipe.steps) == {"inference", "posterior_predictive"}
    assert recipe.inference_config_path == Path("workspace/configs/inference/cmass/devauc.yaml").resolve()
    assert recipe.posterior_diagnostics_config_path == Path(
        "workspace/configs/posterior_predictive/cmass/devauc_diagnostics.yaml"
    ).resolve()
    assert recipe.dataset_path == Path(
        "workspace/data/canonical/inference_dataset_devauc_slit_m5_hunits_v1.hdf5"
    ).resolve()


def test_pipeline_dry_run_describes_existing_post_canonical_workflow() -> None:
    """Dry-run must show concrete actions while avoiding scientific execution."""

    from statistical_sl.pipeline.recipe import load_pipeline_recipe
    from statistical_sl.pipeline.runner import plan_pipeline_run

    recipe = load_pipeline_recipe("workspace/recipes/cmass/devauc_diagnostics_from_canonical.yaml")
    plan = plan_pipeline_run(recipe, diagnostic_run_id="diagnostic-smoke")

    assert plan.recipe_name == "cmass_devauc_diagnostics_from_canonical"
    assert plan.inference_config_path == Path("workspace/configs/inference/cmass/devauc.yaml").resolve()
    assert plan.posterior_diagnostics_config_path == Path(
        "workspace/configs/posterior_predictive/cmass/devauc_diagnostics.yaml"
    ).resolve()
    assert plan.diagnostic_run_id == "diagnostic-smoke"
    assert "run_inference" in plan.actions[0]
    assert "run_posterior_corner" in plan.actions[1]
    assert "run_posterior_diagnostics" in plan.actions[2]
