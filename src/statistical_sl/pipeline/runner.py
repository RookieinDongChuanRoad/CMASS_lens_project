"""
Pipeline execution for inference plus posterior diagnostics.

The runner delegates scientific work to existing workflow APIs. Its narrow
responsibility is orchestration: validate the recipe, call inference, then pass
the completed run directory into posterior diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from statistical_sl.inference.posterior_corner import run_posterior_corner
from statistical_sl.inference.runner import run_inference
from statistical_sl.posterior_predictive.config import load_posterior_diagnostics_config
from statistical_sl.posterior_predictive.predictive import run_posterior_diagnostics
from statistical_sl.pipeline.recipe import PipelineRecipe


@dataclass(frozen=True)
class PipelineDryRunPlan:
    """Human-readable description of a pipeline run that has not executed."""

    recipe_name: str
    inference_config_path: Path
    posterior_diagnostics_config_path: Path
    diagnostic_run_id: str | None
    actions: tuple[str, ...]


@dataclass(frozen=True)
class PipelineRunResult:
    """Structured result of one executed pipeline."""

    recipe_name: str
    inference_run_dir: Path
    posterior_corner_figure_path: Path
    posterior_corner_result_path: Path
    diagnostics_result_dir: Path
    status: str


def plan_pipeline_run(
    recipe: PipelineRecipe,
    *,
    diagnostic_run_id: str | None = None,
) -> PipelineDryRunPlan:
    """Return the actions a pipeline run would execute."""

    diagnostics_config = load_posterior_diagnostics_config(recipe.posterior_diagnostics_config_path)
    actions = (
        f"run_inference(config_path={recipe.inference_config_path})",
        (
            "run_posterior_corner("
            "run_dir=<inference_result.run_dir>, "
            f"burn_in={diagnostics_config.burn_in!r})"
        ),
        (
            "run_posterior_diagnostics("
            "run_dir=<inference_result.run_dir>, "
            f"config_path={diagnostics_config.path}, "
            f"diagnostic_run_id={diagnostic_run_id!r})"
        ),
    )
    return PipelineDryRunPlan(
        recipe_name=recipe.name,
        inference_config_path=recipe.inference_config_path,
        posterior_diagnostics_config_path=recipe.posterior_diagnostics_config_path,
        diagnostic_run_id=diagnostic_run_id,
        actions=actions,
    )


def run_pipeline(
    recipe: PipelineRecipe,
    *,
    diagnostic_run_id: str | None = None,
) -> PipelineRunResult:
    """Execute inference followed by posterior diagnostics for one recipe."""

    inference_result = run_inference(str(recipe.inference_config_path))
    diagnostics_config = load_posterior_diagnostics_config(recipe.posterior_diagnostics_config_path)
    posterior_corner_result = run_posterior_corner(
        run_dir=inference_result.run_dir,
        burn_in=diagnostics_config.burn_in,
    )
    diagnostics_result = run_posterior_diagnostics(
        **diagnostics_config.to_run_kwargs(
            run_dir_override=inference_result.run_dir,
            diagnostic_run_id=diagnostic_run_id,
        )
    )
    return PipelineRunResult(
        recipe_name=recipe.name,
        inference_run_dir=Path(inference_result.run_dir),
        posterior_corner_figure_path=Path(posterior_corner_result.figure_path),
        posterior_corner_result_path=Path(posterior_corner_result.result_path),
        diagnostics_result_dir=Path(diagnostics_result.result_dir),
        status="completed",
    )
