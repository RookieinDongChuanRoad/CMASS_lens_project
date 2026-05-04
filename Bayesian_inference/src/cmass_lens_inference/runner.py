"""
Public run/resume entrypoints.

This module is the orchestration layer: it wires together configuration,
profile selection, I/O normalization, cosmology setup, sampling, checkpointing,
and output serialization.
"""

from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path

from .config import load_runtime_config
from .io import WITHIN_RE_SIGMA_DEFINITION
from .jax_backend.likelihood_engine import build_compiled_model as build_jax_compiled_model
from .mass_definition import mass_definition_metadata
from .numpyro_sampler import run_numpyro_sampler
from .outputs import (
    append_run_log,
    create_run_layout,
    load_numpyro_checkpoint,
    refresh_latest_pointer,
    save_numpyro_checkpoint,
    save_numpyro_posterior_artifacts,
    write_config_snapshot,
    write_metadata,
    write_run_result,
)
from .types import RunResult, RuntimeContext


def _build_runtime_context(runtime_config) -> RuntimeContext:
    """Assemble the shared runtime objects needed by the sampler."""

    compiled_model = build_jax_compiled_model(runtime_config)
    return RuntimeContext(
        config=runtime_config,
        profile=compiled_model.profile,
        observations=[],
        prepared_observations=[],
        cross_section_grid=compiled_model.cross_section_grid,
        random_basis=None,
        cosmology=compiled_model.cosmology,
        parallelism=compiled_model.parallelism,
        compiled_model=compiled_model,
    )


def _run_with_layout(
    runtime_context: RuntimeContext,
    run_layout,
    raw_config_text: str,
    config_path: Path,
    post_warmup_state=None,
    start_step: int = 0,
) -> RunResult:
    """Execute a run once the output layout has already been created."""

    if runtime_context.config.runtime.disable_hdf5_file_locking:
        os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

    config_summary = {
        "profile": runtime_context.config.profile.name,
        "model": {
            "name": runtime_context.config.model.name,
            "components": runtime_context.config.model.components,
        },
        "unit_convention": runtime_context.config.unit_convention,
        "h_ref": runtime_context.config.h_ref,
        "mass_definition": mass_definition_metadata(runtime_context.config.mass_definition),
        "sampling": {
            "num_chains": runtime_context.config.sampling.num_chains,
            "num_samples": runtime_context.config.sampling.num_samples,
            "num_warmup": runtime_context.config.sampling.num_warmup,
            "thinning": runtime_context.config.sampling.thinning,
            "chain_method": runtime_context.config.sampling.chain_method,
            "parameter_order": list(runtime_context.config.parameter_schema.public_parameter_names),
            "initial_center": runtime_context.config.sampling.initial_center.to_public_dict(),
        },
        "box_prior": runtime_context.config.parameter_schema.serialize_public_box_prior(),
        "integration": {
            "gamma_points": runtime_context.config.integration.gamma_points,
            "mstar_points": runtime_context.config.integration.mstar_points,
            "normalization_samples": runtime_context.config.integration.normalization_samples,
        },
        "fp_prior": {
            "enabled": runtime_context.config.fp_prior.enabled,
            "fit_mstar_min": runtime_context.config.fp_prior.fit_mstar_min,
            "pivot_mstar": runtime_context.config.fp_prior.pivot_mstar,
            "fiducial_scatter": runtime_context.config.fp_prior.fiducial_scatter,
            "scatter_error": runtime_context.config.fp_prior.scatter_error,
            "mu_v_prior": runtime_context.config.fp_prior.mu_v_prior,
            "mu_v_error": runtime_context.config.fp_prior.mu_v_error,
            "beta_v_prior": runtime_context.config.fp_prior.beta_v_prior,
            "beta_v_error": runtime_context.config.fp_prior.beta_v_error,
        },
        "sigma_table_path": (
            str(runtime_context.config.data.sigma_table_path)
            if runtime_context.config.data.sigma_table_path is not None
            else None
        ),
        "sigma_table_mass_definition": (
            runtime_context.config.mass_definition.label
            if runtime_context.config.data.sigma_table_path is not None
            else None
        ),
        "fp_sigma_definition": (
            WITHIN_RE_SIGMA_DEFINITION
            if runtime_context.config.fp_prior.enabled and runtime_context.config.data.sigma_table_path is not None
            else None
        ),
        "fp_sigma_table_leaf_path": (
            f"/{WITHIN_RE_SIGMA_DEFINITION}/{runtime_context.config.mass_definition.label}"
            if runtime_context.config.fp_prior.enabled and runtime_context.config.data.sigma_table_path is not None
            else None
        ),
        "parallelism": runtime_context.parallelism.to_dict(),
        "chain_storage": "numpyro_arviz_netcdf",
    }
    write_config_snapshot(run_layout.run_dir, raw_config_text)
    write_metadata(
        run_layout.run_dir,
        profile_name=runtime_context.config.profile.name,
        config_path=config_path,
        observation_path=runtime_context.config.data.observation_path,
        output_root_dir=runtime_context.config.output.root_dir,
        random_seed=runtime_context.config.sampling.random_seed,
        config_summary=config_summary,
    )
    append_run_log(
        run_layout.logs_dir,
        f"start profile {runtime_context.config.profile.name} | backend numpyro_jax | "
        f"model {runtime_context.config.model.name} | strategy {runtime_context.parallelism.strategy} | "
        f"chains {runtime_context.config.sampling.num_chains}",
    )

    sampler_result = run_numpyro_sampler(
        runtime_context,
        logs_dir=run_layout.logs_dir,
        post_warmup_state=post_warmup_state,
    )
    completed_steps = start_step + runtime_context.config.sampling.num_samples
    save_numpyro_checkpoint(
        run_layout.checkpoints_dir,
        sampler_result.samples_by_chain,
        sampler_result.log_prob_by_chain,
        completed_steps,
        sampler_result.last_state,
    )
    samples_path, posterior_path = save_numpyro_posterior_artifacts(
        run_layout.run_dir,
        samples_by_chain=sampler_result.samples_by_chain,
        log_prob_by_chain=sampler_result.log_prob_by_chain,
        diagnostics=sampler_result.diagnostics,
        parameter_names=list(runtime_context.config.parameter_schema.public_parameter_names),
    )
    if runtime_context.config.output.overwrite_latest:
        refresh_latest_pointer(run_layout.profile_dir, run_layout.run_id)

    run_result = RunResult(
        run_id=run_layout.run_id,
        profile_name=runtime_context.config.profile.name,
        run_dir=run_layout.run_dir,
        status="completed",
        start_step=start_step,
        completed_steps=completed_steps,
        acceptance_fraction_mean=sampler_result.acceptance_fraction_mean,
        config_path=config_path,
        input_observation_path=runtime_context.config.data.observation_path,
        output_root_dir=runtime_context.config.output.root_dir,
        checkpoint_step=completed_steps,
        metadata={
            **config_summary,
            "posterior_path": str(posterior_path),
            "samples_path": str(samples_path),
            "num_divergences": sampler_result.diagnostics["num_divergences"],
        },
    )
    write_run_result(run_layout.run_dir, run_result)
    return run_result


def run_inference(config_path: str, label: str | None = None) -> RunResult:
    """Public API for launching a new run from a YAML configuration file."""

    resolved_config_path = Path(config_path).expanduser().resolve()
    runtime_config = load_runtime_config(resolved_config_path)
    if label is not None:
        runtime_config = replace(
            runtime_config,
            output=replace(
                runtime_config.output,
                run_label=label,
            ),
        )

    runtime_context = _build_runtime_context(runtime_config)
    run_layout = create_run_layout(
        root_dir=runtime_config.output.root_dir,
        profile_name=runtime_config.profile.name,
        run_label=runtime_config.output.run_label,
    )
    raw_config_text = resolved_config_path.read_text(encoding="utf-8")
    return _run_with_layout(
        runtime_context,
        run_layout,
        raw_config_text,
        resolved_config_path,
    )


def resume_inference(run_dir: str) -> RunResult:
    """
    Public API for resuming an existing run from its on-disk artifacts.

    The restart contract is:
    - read the configuration snapshot stored in the run directory
    - restore the latest checkpoint if it exists
    - continue sampling in the same run directory
    """

    resolved_run_dir = Path(run_dir).expanduser().resolve()
    config_snapshot_path = resolved_run_dir / "config_snapshot.yaml"
    runtime_config = load_runtime_config(config_snapshot_path)
    runtime_context = _build_runtime_context(runtime_config)
    run_layout = create_run_layout(
        root_dir=runtime_config.output.root_dir,
        profile_name=runtime_config.profile.name,
        run_label=runtime_config.output.run_label,
        timestamp_text="_".join(resolved_run_dir.name.split("_")[:2]),
    )
    # Reuse the existing run directory rather than creating a fresh one.
    run_layout = run_layout.__class__(
        root_dir=run_layout.root_dir,
        profile_dir=run_layout.profile_dir,
        run_id=resolved_run_dir.name,
        run_dir=resolved_run_dir,
        checkpoints_dir=resolved_run_dir / "checkpoints",
        logs_dir=resolved_run_dir / "logs",
    )
    try:
        post_warmup_state, checkpoint_step = load_numpyro_checkpoint(resolved_run_dir / "checkpoints")
    except FileNotFoundError:
        legacy_step_path = resolved_run_dir / "checkpoints" / "latest_step.txt"
        if not legacy_step_path.exists():
            raise FileNotFoundError(
                f"Run directory {resolved_run_dir} does not contain a usable NumPyro checkpoint."
            )
        post_warmup_state = None
        checkpoint_step = int(legacy_step_path.read_text(encoding="utf-8").strip())
        append_run_log(
            run_layout.logs_dir,
            f"resume legacy checkpoint step only | checkpoint_step {checkpoint_step}",
        )
    start_step = checkpoint_step
    append_run_log(
        run_layout.logs_dir,
        f"resume numpyro checkpoint | checkpoint_step {checkpoint_step}",
    )
    raw_config_text = config_snapshot_path.read_text(encoding="utf-8")
    return _run_with_layout(
        runtime_context,
        run_layout,
        raw_config_text,
        config_snapshot_path,
        post_warmup_state=post_warmup_state,
        start_step=start_step,
    )
