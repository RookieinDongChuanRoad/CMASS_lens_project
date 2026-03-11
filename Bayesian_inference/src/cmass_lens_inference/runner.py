"""
Public run/resume entrypoints.

This module is the orchestration layer: it wires together configuration,
profile selection, I/O normalization, cosmology setup, sampling, checkpointing,
and output serialization.
"""

from __future__ import annotations

import os
from pathlib import Path

from .compiled_context import build_compiled_context
from .config import load_runtime_config
from .outputs import (
    append_run_log,
    create_run_layout,
    load_checkpoint,
    refresh_latest_pointer,
    save_checkpoint,
    write_config_snapshot,
    write_metadata,
    write_run_result,
)
from .parallel import resolve_parallelism
from .sampler import create_chain_backend, load_backend_state, run_ensemble_sampler
from .types import CompiledModel, RunResult, RuntimeContext


def _build_runtime_context(runtime_config) -> RuntimeContext:
    """Assemble the shared runtime objects needed by the sampler."""

    compiled_context, profile_spec, cross_section_grid, cosmology, random_basis, observations = build_compiled_context(
        runtime_config
    )
    resolved_parallelism = resolve_parallelism(
        runtime_config.runtime,
        runtime_config.sampling.n_walkers,
    )
    compiled_model = CompiledModel(
        config=runtime_config,
        profile=profile_spec,
        cross_section_grid=cross_section_grid,
        cosmology=cosmology,
        parallelism=resolved_parallelism,
        context=compiled_context,
    )
    return RuntimeContext(
        config=runtime_config,
        profile=profile_spec,
        observations=observations,
        prepared_observations=[],
        cross_section_grid=cross_section_grid,
        random_basis=random_basis,
        cosmology=cosmology,
        parallelism=resolved_parallelism,
        compiled_model=compiled_model,
    )


def _run_with_layout(
    runtime_context: RuntimeContext,
    run_layout,
    raw_config_text: str,
    config_path: Path,
    chain_backend,
    start_state=None,
    start_step: int = 0,
) -> RunResult:
    """Execute a run once the output layout has already been created."""

    if runtime_context.config.runtime.disable_hdf5_file_locking:
        os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

    config_summary = {
        "profile": runtime_context.config.profile.name,
        "sampling": {
            "n_walkers": runtime_context.config.sampling.n_walkers,
            "n_steps": runtime_context.config.sampling.n_steps,
            "warmup": runtime_context.config.sampling.warmup,
        },
        "integration": {
            "gamma_points": runtime_context.config.integration.gamma_points,
            "mstar_points": runtime_context.config.integration.mstar_points,
            "normalization_samples": runtime_context.config.integration.normalization_samples,
        },
        "parallelism": runtime_context.parallelism.to_dict(),
        "chain_storage": "emcee_hdf_backend",
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
        f"start profile {runtime_context.config.profile.name} | strategy {runtime_context.parallelism.strategy} | "
        f"workers {runtime_context.parallelism.worker_processes} | kernel_threads {runtime_context.parallelism.kernel_threads_per_process}",
    )

    final_coords, final_log_prob, acceptance_fraction_mean = run_ensemble_sampler(
        runtime_context,
        chain_backend=chain_backend,
        start_state=start_state,
        start_step=start_step,
        logs_dir=run_layout.logs_dir,
        checkpoint_callback=lambda coords, log_prob, step: save_checkpoint(run_layout.checkpoints_dir, coords, log_prob, step),
    )
    completed_steps = start_step + runtime_context.config.sampling.n_steps
    save_checkpoint(run_layout.checkpoints_dir, final_coords, final_log_prob, completed_steps)
    if runtime_context.config.output.overwrite_latest:
        refresh_latest_pointer(run_layout.profile_dir, run_layout.run_id)

    run_result = RunResult(
        run_id=run_layout.run_id,
        profile_name=runtime_context.config.profile.name,
        run_dir=run_layout.run_dir,
        status="completed",
        start_step=start_step,
        completed_steps=completed_steps,
        acceptance_fraction_mean=acceptance_fraction_mean,
        config_path=config_path,
        input_observation_path=runtime_context.config.data.observation_path,
        output_root_dir=runtime_context.config.output.root_dir,
        checkpoint_step=completed_steps,
        metadata=config_summary,
    )
    write_run_result(run_layout.run_dir, run_result)
    return run_result


def run_inference(config_path: str, label: str | None = None) -> RunResult:
    """Public API for launching a new run from a YAML configuration file."""

    resolved_config_path = Path(config_path).expanduser().resolve()
    runtime_config = load_runtime_config(resolved_config_path)
    if label is not None:
        runtime_config = runtime_config.__class__(
            profile=runtime_config.profile,
            data=runtime_config.data,
            sampling=runtime_config.sampling,
            integration=runtime_config.integration,
            runtime=runtime_config.runtime,
            output=runtime_config.output.__class__(
                root_dir=runtime_config.output.root_dir,
                run_label=label,
                overwrite_latest=runtime_config.output.overwrite_latest,
            ),
        )

    runtime_context = _build_runtime_context(runtime_config)
    run_layout = create_run_layout(
        root_dir=runtime_config.output.root_dir,
        profile_name=runtime_config.profile.name,
        run_label=runtime_config.output.run_label,
    )
    raw_config_text = resolved_config_path.read_text(encoding="utf-8")
    chain_backend = create_chain_backend(
        run_layout.run_dir / "chain.h5",
        runtime_config.sampling.n_walkers,
        12,
        reset=True,
    )
    return _run_with_layout(
        runtime_context,
        run_layout,
        raw_config_text,
        resolved_config_path,
        chain_backend=chain_backend,
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
    chain_path = resolved_run_dir / "chain.h5"
    backend_state, backend_step = load_backend_state(chain_path)
    checkpoint_state = None
    checkpoint_step = 0
    try:
        checkpoint_coords, checkpoint_log_prob, checkpoint_step = load_checkpoint(resolved_run_dir / "checkpoints")
        checkpoint_state = (checkpoint_coords, checkpoint_log_prob)
    except FileNotFoundError:
        checkpoint_state = None

    if backend_state is not None:
        if getattr(backend_state, "blobs", None) is None:
            # Older runs or synthetic tests can have a valid backend chain but
            # no stored blobs. Since the new production path always emits
            # timing blobs, `emcee` requires us to restart from coordinates
            # only in this case so it can recompute the missing blob values on
            # the first resumed step.
            start_state = backend_state.coords
            append_run_log(
                run_layout.logs_dir,
                f"resume backend missing blobs | backend_step {backend_step} | source backend_coords_only",
            )
        else:
            start_state = backend_state
        start_step = backend_step
        if checkpoint_state is not None and checkpoint_step != backend_step:
            append_run_log(
                run_layout.logs_dir,
                f"resume checkpoint/backend mismatch | checkpoint_step {checkpoint_step} | backend_step {backend_step} | source backend",
            )
    elif checkpoint_state is not None:
        start_state = checkpoint_state[0]
        start_step = checkpoint_step
        append_run_log(
            run_layout.logs_dir,
            f"resume backend unavailable | checkpoint_step {checkpoint_step} | source checkpoint",
        )
    else:
        raise FileNotFoundError(
            f"Run directory {resolved_run_dir} does not contain a usable chain backend or checkpoint."
        )

    chain_backend = create_chain_backend(
        chain_path,
        runtime_config.sampling.n_walkers,
        12,
        reset=False,
    )
    raw_config_text = config_snapshot_path.read_text(encoding="utf-8")
    return _run_with_layout(
        runtime_context,
        run_layout,
        raw_config_text,
        config_snapshot_path,
        chain_backend=chain_backend,
        start_state=start_state,
        start_step=start_step,
    )
