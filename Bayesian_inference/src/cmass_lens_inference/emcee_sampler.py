"""
emcee sampler orchestration for the production Numba backend.

The scientific hot path lives in `numba_backend.likelihood_engine`.  This module
owns only sampler mechanics: walker initialization, optional process-pool
evaluation, checkpoint callbacks, run-log timing summaries, and the persistent
`chain.h5` HDFBackend contract.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import multiprocessing
from pathlib import Path

import emcee
import numpy as np
from tqdm.auto import tqdm

from .numba_backend.likelihood_engine import (
    NUMBA_DIAGNOSTIC_BLOB_DTYPE,
    log_prob as compiled_model_log_prob,
)
from .outputs import append_run_log
from .parallel import apply_thread_limits
from .types import HyperParams, RuntimeContext


_PROCESS_LOCAL_EVALUATOR = None


@dataclass(frozen=True)
class EmceeSamplerResult:
    """Compact summary returned by one emcee run or resume call."""

    final_coords: np.ndarray
    final_log_prob: np.ndarray
    acceptance_fraction_mean: float


class LogProbEvaluator:
    """
    Pickle-friendly callable that adapts backend blobs to emcee's tuple format.

    emcee treats `return logp, blob_field_1, blob_field_2, ...` as one
    structured blob when `blobs_dtype` is provided.  Returning fields in dtype
    order here keeps single-process and process-pool evaluation identical.
    """

    def __init__(self, runtime_context: RuntimeContext) -> None:
        self.runtime_context = runtime_context

    def close(self) -> None:
        """Stable lifecycle hook for symmetry with process-pool cleanup."""

    def __call__(self, theta_vector: np.ndarray) -> tuple:
        if self.runtime_context.compiled_model is None:
            raise RuntimeError("RuntimeContext is missing the compiled model required by emcee.")
        log_prob_value, blob = compiled_model_log_prob(
            np.asarray(theta_vector, dtype=np.float64),
            self.runtime_context.compiled_model,
        )
        return (
            float(log_prob_value),
            float(blob["total_log_prob_seconds"]),
            float(blob["likelihood_seconds"]),
            float(blob["normalization_seconds"]),
            float(blob["fp_prior_seconds"]),
            float(blob["normalization_value"]),
            float(blob["fp_prior_log_term"]),
            float(blob["fpfit_mu"]),
            float(blob["fpfit_beta"]),
            float(blob["fpfit_xi"]),
            float(blob["fpfit_scatter"]),
            bytes(blob["backend"]),
            bytes(blob["kernel"]),
            bytes(blob["parallel_strategy"]),
        )


def _worker_initializer(runtime_context: RuntimeContext) -> None:
    """Initialize process-local evaluator state under macOS spawn semantics."""

    apply_thread_limits(runtime_context.parallelism.kernel_threads_per_process)
    global _PROCESS_LOCAL_EVALUATOR
    _PROCESS_LOCAL_EVALUATOR = LogProbEvaluator(runtime_context)


def _process_local_log_prob(theta_vector: np.ndarray) -> tuple:
    """Delegate process-pool log-probability calls to worker-local state."""

    if _PROCESS_LOCAL_EVALUATOR is None:
        raise RuntimeError("Process-local log-prob evaluator was not initialized.")
    return _PROCESS_LOCAL_EVALUATOR(theta_vector)


def initialize_walkers(
    initial_center: HyperParams,
    n_walkers: int,
    jitter_scale: float,
    seed: int,
) -> np.ndarray:
    """
    Initialize an in-bounds walker cloud around the configured center.

    The explicit box prior is enforced before sampling starts.  If the jitter
    scale is too large for the configured bounds, failing here is clearer than
    seeding invalid walkers that only produce `-inf` log-probabilities.
    """

    rng = np.random.default_rng(seed)
    center = initial_center.to_array()
    parameter_schema = initial_center.parameter_schema
    parameter_schema.validate_theta_in_bounds(center, label="Initial center")

    lower_bounds = np.asarray([lower for lower, _ in parameter_schema.prior_bounds], dtype=float)
    upper_bounds = np.asarray([upper for _, upper in parameter_schema.prior_bounds], dtype=float)
    if jitter_scale <= 0.0:
        return np.repeat(center[None, :], n_walkers, axis=0)

    accepted: list[np.ndarray] = []
    accepted_count = 0
    attempted_count = 0
    max_attempted = max(256, 128 * n_walkers)
    while accepted_count < n_walkers and attempted_count < max_attempted:
        batch_size = max(n_walkers - accepted_count, n_walkers)
        candidates = center[None, :] + rng.normal(
            loc=0.0,
            scale=jitter_scale,
            size=(batch_size, center.size),
        )
        valid_mask = np.all(
            (candidates >= lower_bounds[None, :]) & (candidates <= upper_bounds[None, :]),
            axis=1,
        )
        if np.any(valid_mask):
            accepted.append(candidates[valid_mask])
            accepted_count += int(np.count_nonzero(valid_mask))
        attempted_count += batch_size

    if accepted_count < n_walkers:
        raise ValueError(
            "Unable to initialize walker coordinates within the configured "
            "box_prior bounds. Reduce sampling.initial_jitter_scale or relax "
            "the explicit bounds."
        )
    return np.vstack(accepted)[:n_walkers]


def create_chain_backend(
    chain_path: Path,
    n_walkers: int,
    n_dim: int,
    *,
    reset: bool,
) -> emcee.backends.HDFBackend:
    """
    Create or reopen the HDFBackend that is the chain source of truth.

    Fresh runs pass `reset=True` so stale data cannot contaminate a new run.
    Resume runs pass `reset=False` so emcee appends to the existing backend.
    """

    backend = emcee.backends.HDFBackend(str(chain_path))
    if reset:
        backend.reset(n_walkers, n_dim)
    return backend


def load_backend_state(chain_path: Path) -> tuple[emcee.State | None, int]:
    """
    Return the last persisted emcee state when `chain.h5` is usable.

    The function fails soft because interrupted early runs may contain no HDF
    backend yet.  The runner can then fall back to the lightweight checkpoint
    arrays, or start from a fresh walker cloud for new runs.
    """

    if not chain_path.exists():
        return None, 0
    try:
        backend = emcee.backends.HDFBackend(str(chain_path), read_only=True)
        if (not backend.initialized) or int(backend.iteration) <= 0:
            return None, 0
        return backend.get_last_sample(), int(backend.iteration)
    except (AttributeError, KeyError, OSError, ValueError):
        return None, 0


def _summarize_recent_blobs(recent_blobs, current_step: int, total_steps: int, parallelism) -> str:
    """Aggregate recent timing blobs into a one-line run-log summary."""

    total_times = np.asarray([float(blob["total_log_prob_seconds"]) for blob in recent_blobs], dtype=float)
    likelihood_times = np.asarray([float(blob["likelihood_seconds"]) for blob in recent_blobs], dtype=float)
    normalization_times = np.asarray([float(blob["normalization_seconds"]) for blob in recent_blobs], dtype=float)
    fp_prior_times = np.asarray([float(blob["fp_prior_seconds"]) for blob in recent_blobs], dtype=float)
    return (
        f"step {current_step}/{total_steps} | "
        f"lp {total_times.mean():.2f}s | "
        f"lens {likelihood_times.mean():.2f}s | "
        f"norm {normalization_times.mean():.2f}s | "
        f"fp {fp_prior_times.mean():.2f}s | "
        f"workers {parallelism.worker_processes or parallelism.kernel_threads_per_process} | "
        f"strategy {parallelism.strategy}"
    )


def run_emcee_sampler(
    runtime_context: RuntimeContext,
    chain_backend: emcee.backends.HDFBackend,
    *,
    start_state: emcee.State | np.ndarray | None = None,
    start_step: int = 0,
    logs_dir: Path | None = None,
    checkpoint_callback=None,
) -> EmceeSamplerResult:
    """
    Run emcee for the configured number of steps and return final walker state.

    Checkpoint writing is delegated to the runner through `checkpoint_callback`
    so this module stays focused on sampler control flow and does not own the
    filesystem layout.
    """

    apply_thread_limits(runtime_context.parallelism.kernel_threads_per_process)
    sampling = runtime_context.config.sampling
    log_prob_fn = LogProbEvaluator(runtime_context)
    skip_initial_state_check = False

    if start_state is None:
        sampler_initial_state: emcee.State | np.ndarray = initialize_walkers(
            sampling.initial_center,
            sampling.n_walkers,
            sampling.initial_jitter_scale,
            sampling.random_seed,
        )
    else:
        sampler_initial_state = start_state
        skip_initial_state_check = True
        if isinstance(start_state, np.ndarray) and np.allclose(start_state, start_state[0]):
            # Raw checkpoint coordinates can be valid on disk but degenerate for
            # emcee's independence check.  Regenerating a tiny bounded cloud is
            # safer than resuming into an immediate linear-dependence failure.
            sampler_initial_state = initialize_walkers(
                sampling.initial_center,
                sampling.n_walkers,
                sampling.initial_jitter_scale,
                sampling.random_seed,
            )

    process_pool = None
    if runtime_context.parallelism.strategy == "process_pool":
        context = multiprocessing.get_context("spawn")
        process_pool = context.Pool(
            processes=runtime_context.parallelism.worker_processes,
            initializer=_worker_initializer,
            initargs=(runtime_context,),
        )
        sampler_log_prob_fn = _process_local_log_prob
    else:
        sampler_log_prob_fn = log_prob_fn

    sampler = emcee.EnsembleSampler(
        sampling.n_walkers,
        runtime_context.config.parameter_schema.n_dim,
        sampler_log_prob_fn,
        pool=process_pool,
        backend=chain_backend,
        blobs_dtype=NUMBA_DIAGNOSTIC_BLOB_DTYPE,
    )
    recent_blobs = deque()
    try:
        for relative_step, state in enumerate(
            sampler.sample(
                sampler_initial_state,
                iterations=sampling.n_steps,
                progress=runtime_context.config.runtime.progress,
                skip_initial_state_check=skip_initial_state_check,
            ),
            start=1,
        ):
            current_step = start_step + relative_step
            if checkpoint_callback is not None and current_step % runtime_context.config.runtime.checkpoint_every == 0:
                checkpoint_callback(
                    np.asarray(state.coords, dtype=float),
                    np.asarray(state.log_prob, dtype=float),
                    current_step,
                )

            if state.blobs is not None:
                recent_blobs.extend(blob for blob in np.ravel(state.blobs) if blob is not None)

            if (
                logs_dir is not None
                and runtime_context.config.runtime.show_stage_timing
                and recent_blobs
                and current_step % runtime_context.config.runtime.progress_summary_every == 0
            ):
                summary_line = _summarize_recent_blobs(
                    recent_blobs,
                    current_step,
                    start_step + sampling.n_steps,
                    runtime_context.parallelism,
                )
                append_run_log(logs_dir, summary_line)
                if runtime_context.config.runtime.progress:
                    tqdm.write(summary_line)
                recent_blobs.clear()

        if logs_dir is not None and runtime_context.config.runtime.show_stage_timing and recent_blobs:
            summary_line = _summarize_recent_blobs(
                recent_blobs,
                start_step + sampling.n_steps,
                start_step + sampling.n_steps,
                runtime_context.parallelism,
            )
            append_run_log(logs_dir, summary_line)
            if runtime_context.config.runtime.progress:
                tqdm.write(summary_line)
            recent_blobs.clear()
    finally:
        if process_pool is not None:
            process_pool.close()
            process_pool.join()
        log_prob_fn.close()

    last_sample = sampler.get_last_sample()
    return EmceeSamplerResult(
        final_coords=np.asarray(last_sample.coords, dtype=float),
        final_log_prob=np.asarray(last_sample.log_prob, dtype=float),
        acceptance_fraction_mean=float(np.mean(sampler.acceptance_fraction)),
    )


__all__ = [
    "EmceeSamplerResult",
    "create_chain_backend",
    "initialize_walkers",
    "load_backend_state",
    "run_emcee_sampler",
]
