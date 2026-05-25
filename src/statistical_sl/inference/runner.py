"""
Public run/resume entrypoints for production inference.

This module is the orchestration layer.  It wires together configuration,
canonical-data runtime contexts, the Numba posterior backend, emcee sampling,
checkpointing, and output metadata.  Scientific formulas stay in model-specific
Numba kernels; sampler mechanics stay in `emcee_sampler.py`.
"""

from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path

from statistical_sl.inference.canonical_dataset import CAPABILITY_VELOCITY_DISPERSION_FP_WITHIN_RE_V1
from statistical_sl.inference.config import load_runtime_config
from statistical_sl.inference.emcee_sampler import (
    create_chain_backend,
    load_backend_state,
    run_emcee_sampler,
)
from statistical_sl.inference.outputs import (
    append_run_log,
    create_run_layout,
    load_emcee_checkpoint,
    refresh_latest_pointer,
    save_emcee_checkpoint,
    write_config_snapshot,
    write_metadata,
    write_run_manifest,
    write_run_result,
)
from statistical_sl.inference.types import RunLayout, RunResult, RuntimeContext
from statistical_sl.core.mass_definition import mass_definition_metadata
from statistical_sl.inference.backends.numba_emcee.likelihood_engine import (
    build_compiled_model as build_numba_compiled_model,
)


def _build_runtime_context(runtime_config) -> RuntimeContext:
    """
    Assemble the shared runtime objects needed by emcee log-prob evaluation.

    The compiled model is a framework container around a model-specific NumPy
    source context.  It is intentionally opaque to the runner: only the Numba
    likelihood engine knows which kernel consumes that context.
    """

    compiled_model = build_numba_compiled_model(runtime_config)
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


def _config_summary(runtime_context: RuntimeContext) -> dict:
    """
    Build the output metadata summary for one production run.

    This payload is intentionally redundant with config files because long
    inference runs need self-describing artifacts.  The backend/sampler fields
    are explicit so later audits can prove a run used `numba/emcee`.
    """

    if runtime_context.compiled_model is None:
        data_metadata = {}
    else:
        data_metadata = dict(runtime_context.compiled_model.data_metadata)
    canonical_capabilities = list(data_metadata.get("canonical_capabilities", ()))
    has_fp_within_re = CAPABILITY_VELOCITY_DISPERSION_FP_WITHIN_RE_V1 in set(canonical_capabilities)

    return {
        "profile": runtime_context.config.profile.name,
        "model": {
            "name": runtime_context.config.model.name,
            "metadata": dict(runtime_context.config.parameter_schema.model_metadata),
        },
        "unit_convention": runtime_context.config.unit_convention,
        "h_ref": runtime_context.config.h_ref,
        "mass_definition": mass_definition_metadata(runtime_context.config.mass_definition),
        "sampling": {
            "n_walkers": runtime_context.config.sampling.n_walkers,
            "n_steps": runtime_context.config.sampling.n_steps,
            "burn_in": runtime_context.config.sampling.burn_in,
            "parameter_order": list(runtime_context.config.parameter_schema.public_parameter_names),
            "initial_center": runtime_context.config.sampling.initial_center.to_public_dict(),
        },
        "box_prior": runtime_context.config.parameter_schema.serialize_public_box_prior(),
        "integration": {
            "gamma_points": runtime_context.config.integration.gamma_points,
            "mstar_points": runtime_context.config.integration.mstar_points,
            "normalization_samples": runtime_context.config.integration.normalization_samples,
        },
        "data": {
            "inference_dataset_path": str(runtime_context.config.data.inference_dataset_path),
            "canonical_capabilities": canonical_capabilities,
            "canonical_schema_version": data_metadata.get("canonical_schema_version"),
            "canonical_profile_name": data_metadata.get("canonical_profile_name"),
            "canonical_mass_definition_label": data_metadata.get("canonical_mass_definition_label"),
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
        "fp_sigma_definition": (
            "within_re"
            if runtime_context.config.fp_prior.enabled and has_fp_within_re
            else None
        ),
        "fp_sigma_table_leaf_path": (
            "/velocity_dispersion_grids/fp_within_re"
            if runtime_context.config.fp_prior.enabled and has_fp_within_re
            else None
        ),
        "parallelism": runtime_context.parallelism.to_dict(),
        "backend": "numba_emcee",
        "kernel_backend": "numba",
        "sampler": "emcee",
        "chain_storage": "emcee_hdf_backend",
    }


def _run_with_layout(
    runtime_context: RuntimeContext,
    run_layout: RunLayout,
    raw_config_text: str,
    config_path: Path,
    *,
    start_state=None,
    start_step: int = 0,
    reset_chain_backend: bool = True,
) -> RunResult:
    """Execute one run after the output layout has been created."""

    if runtime_context.config.runtime.disable_hdf5_file_locking:
        os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

    if runtime_context.config.data.inference_dataset_path is None:
        raise ValueError("Production inference requires data.inference_dataset_path.")

    config_summary = _config_summary(runtime_context)
    write_config_snapshot(run_layout.run_dir, raw_config_text)
    write_metadata(
        run_layout.run_dir,
        profile_name=runtime_context.config.profile.name,
        config_path=config_path,
        inference_dataset_path=runtime_context.config.data.inference_dataset_path,
        output_root_dir=runtime_context.config.output.root_dir,
        random_seed=runtime_context.config.sampling.random_seed,
        config_summary=config_summary,
    )
    write_run_manifest(
        run_layout.run_dir,
        profile_name=runtime_context.config.profile.name,
        config_path=config_path,
        inference_dataset_path=runtime_context.config.data.inference_dataset_path,
        output_root_dir=runtime_context.config.output.root_dir,
        random_seed=runtime_context.config.sampling.random_seed,
        config_summary=config_summary,
    )
    append_run_log(
        run_layout.logs_dir,
        f"start profile {runtime_context.config.profile.name} | backend numba_emcee | "
        f"model {runtime_context.config.model.name} | strategy {runtime_context.parallelism.strategy} | "
        f"walkers {runtime_context.config.sampling.n_walkers}",
    )

    chain_path = run_layout.run_dir / "chain.h5"
    chain_backend = create_chain_backend(
        chain_path,
        runtime_context.config.sampling.n_walkers,
        runtime_context.config.parameter_schema.n_dim,
        reset=reset_chain_backend,
    )
    sampler_result = run_emcee_sampler(
        runtime_context,
        chain_backend,
        start_state=start_state,
        start_step=start_step,
        logs_dir=run_layout.logs_dir,
        checkpoint_callback=lambda coords, log_prob, step: save_emcee_checkpoint(
            run_layout.checkpoints_dir,
            coords,
            log_prob,
            step,
        ),
    )
    completed_steps = start_step + runtime_context.config.sampling.n_steps
    save_emcee_checkpoint(
        run_layout.checkpoints_dir,
        sampler_result.final_coords,
        sampler_result.final_log_prob,
        completed_steps,
    )
    append_run_log(
        run_layout.logs_dir,
        f"emcee complete | completed_steps {completed_steps} | "
        f"acceptance_fraction_mean {sampler_result.acceptance_fraction_mean:.6f}",
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
        input_inference_dataset_path=runtime_context.config.data.inference_dataset_path,
        input_observation_path=None,
        output_root_dir=runtime_context.config.output.root_dir,
        checkpoint_step=completed_steps,
        metadata={
            **config_summary,
            "chain_path": str(chain_path),
        },
    )
    write_run_result(run_layout.run_dir, run_result)
    return run_result


def run_inference(config_path: str, label: str | None = None) -> RunResult:
    """Launch a new production emcee run from a YAML configuration file."""

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
        reset_chain_backend=True,
    )


def resume_inference(run_dir: str) -> RunResult:
    """
    Resume an existing emcee run from its on-disk artifacts.

    The preferred restart point is `chain.h5` because it stores coordinates,
    log-probabilities, blobs, and random state.  Lightweight checkpoint arrays
    are a fallback for interrupted runs where the backend state cannot be read.
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
    run_layout = RunLayout(
        root_dir=run_layout.root_dir,
        profile_dir=run_layout.profile_dir,
        run_id=resolved_run_dir.name,
        run_dir=resolved_run_dir,
        checkpoints_dir=resolved_run_dir / "checkpoints",
        logs_dir=resolved_run_dir / "logs",
    )

    chain_path = resolved_run_dir / "chain.h5"
    start_state, backend_step = load_backend_state(chain_path)
    reset_chain_backend = False
    if start_state is not None:
        checkpoint_step = backend_step
        append_run_log(
            run_layout.logs_dir,
            f"resume emcee hdf backend | checkpoint_step {checkpoint_step}",
        )
    else:
        coords, _log_prob, checkpoint_step = load_emcee_checkpoint(resolved_run_dir / "checkpoints")
        start_state = coords
        reset_chain_backend = True
        append_run_log(
            run_layout.logs_dir,
            f"resume lightweight emcee checkpoint | checkpoint_step {checkpoint_step}",
        )

    raw_config_text = config_snapshot_path.read_text(encoding="utf-8")
    return _run_with_layout(
        runtime_context,
        run_layout,
        raw_config_text,
        config_snapshot_path,
        start_state=start_state,
        start_step=checkpoint_step,
        reset_chain_backend=reset_chain_backend,
    )
