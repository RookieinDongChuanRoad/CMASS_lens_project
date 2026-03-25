"""
Sampler and log-posterior orchestration.

After the performance refactor, this module is intentionally thin. It owns the
`emcee` control flow, checkpoint cadence, progress reporting, and process-pool
setup, while the scientific hot path is delegated to `model.log_prob()`.
"""

from __future__ import annotations

import multiprocessing
from collections import deque
from pathlib import Path

import emcee
import numpy as np
from tqdm.auto import tqdm

from .model import LOG_PROB_BLOB_DTYPE, log_prob as compiled_model_log_prob
from .outputs import append_run_log
from .parallel import apply_thread_limits
from .types import HyperParams, RuntimeContext


_PROCESS_LOCAL_EVALUATOR = None


class LogProbEvaluator:
    """
    Pickle-friendly log-probability evaluator with timing instrumentation.

    A top-level callable object is required because macOS `spawn` multiprocessing
    cannot safely execute nested-closure implementations.
    """

    def __init__(self, runtime_context: RuntimeContext) -> None:
        self.runtime_context = runtime_context

    def close(self) -> None:
        """Provide a stable close hook for caller symmetry."""

    def __call__(
        self,
        theta_vector: np.ndarray,
    ) -> tuple[float, float, float, float, float, float, float, float, float, float, float, bytes]:
        if self.runtime_context.compiled_model is None:
            raise RuntimeError("RuntimeContext is missing the compiled model required by the sampler.")
        log_prob_value, blob = compiled_model_log_prob(
            np.asarray(theta_vector, dtype=float),
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
            bytes(blob["parallel_strategy"]),
        )


def _worker_initializer(runtime_context: RuntimeContext) -> None:
    """Initialize worker-local state for process-pool log-prob evaluation."""

    apply_thread_limits(runtime_context.parallelism.kernel_threads_per_process)
    global _PROCESS_LOCAL_EVALUATOR
    _PROCESS_LOCAL_EVALUATOR = LogProbEvaluator(runtime_context)


def _process_local_log_prob(
    theta_vector: np.ndarray,
) -> tuple[float, float, float, float, float, float, float, float, float, float, float, bytes]:
    """Delegate process-pool log-prob evaluation to worker-local state."""

    if _PROCESS_LOCAL_EVALUATOR is None:
        raise RuntimeError("Process-local log-prob evaluator was not initialized.")
    return _PROCESS_LOCAL_EVALUATOR(theta_vector)


def build_log_prob_fn(runtime_context: RuntimeContext):
    """
    Build the log-posterior function expected by `emcee`.

    The closure captures the fixed runtime context so each log-probability
    evaluation only receives the mode-aware sampled parameter vector.
    """

    return LogProbEvaluator(runtime_context)


def create_chain_backend(
    chain_path: Path,
    n_walkers: int,
    n_dim: int,
    reset: bool,
) -> emcee.backends.HDFBackend:
    """
    Create the single backend file used as the chain source of truth.

    Why the reset flag is explicit:
    - new runs must clear any stale backend contents before sampling starts
    - resume runs must preserve the existing stored chain and append to it
    """

    backend = emcee.backends.HDFBackend(str(chain_path))
    if reset:
        backend.reset(n_walkers, n_dim)
    return backend


def load_backend_state(chain_path: Path) -> tuple[emcee.State | None, int]:
    """
    Read the most recent persisted sampler state from an existing backend.

    The function fails soft because older runs or interrupted runs can be
    missing `chain.h5` entirely. In that case callers may still decide to fall
    back to checkpoint files.
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


def initialize_walkers(initial_center: HyperParams, n_walkers: int, jitter_scale: float, seed: int) -> np.ndarray:
    """
    Initialize walkers around the configured center with Gaussian jitter.

    The explicit box-prior contract now applies at initialization time as well:
    if the requested jitter cannot produce enough in-bounds walkers, fail fast
    instead of silently seeding invalid coordinates that would later be
    rejected by the log-probability function.
    """

    rng = np.random.default_rng(seed)
    center = initial_center.to_array()
    parameter_schema = initial_center.parameter_schema
    parameter_schema.validate_theta_in_bounds(center, label="Initial center")

    lower_bounds = np.asarray(
        [lower for lower, _ in parameter_schema.prior_bounds],
        dtype=float,
    )
    upper_bounds = np.asarray(
        [upper for _, upper in parameter_schema.prior_bounds],
        dtype=float,
    )

    if jitter_scale <= 0.0:
        return np.repeat(center[None, :], n_walkers, axis=0)

    accepted_walkers: list[np.ndarray] = []
    accepted_count = 0
    attempted_draws = 0
    max_attempted_draws = max(256, 128 * n_walkers)
    while accepted_count < n_walkers and attempted_draws < max_attempted_draws:
        batch_size = max(n_walkers - accepted_count, n_walkers)
        jitter = rng.normal(loc=0.0, scale=jitter_scale, size=(batch_size, center.size))
        candidates = center[None, :] + jitter
        valid_mask = np.all(
            (candidates >= lower_bounds[None, :]) & (candidates <= upper_bounds[None, :]),
            axis=1,
        )
        if np.any(valid_mask):
            accepted_walkers.append(candidates[valid_mask])
            accepted_count += int(np.count_nonzero(valid_mask))
        attempted_draws += batch_size

    if accepted_count < n_walkers:
        raise ValueError(
            "Unable to initialize walker coordinates within the configured "
            "box_prior bounds. Reduce `sampling.initial_jitter_scale` or relax "
            "the explicit bounds."
        )

    return np.vstack(accepted_walkers)[:n_walkers]


def run_ensemble_sampler(
    runtime_context: RuntimeContext,
    chain_backend: emcee.backends.HDFBackend,
    start_state: emcee.State | np.ndarray | None = None,
    start_step: int = 0,
    logs_dir: Path | None = None,
    checkpoint_callback=None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Run the `emcee` ensemble sampler for the configured number of steps.

    The function returns the final walker coordinates, their final
    log-probabilities, and the mean acceptance fraction. Writing checkpoints and
    output files is intentionally left to the runner/output layer.
    """

    apply_thread_limits(runtime_context.parallelism.kernel_threads_per_process)
    log_prob_fn = build_log_prob_fn(runtime_context)
    skip_initial_state_check = False
    if start_state is None:
        sampler_initial_state: emcee.State | np.ndarray = initialize_walkers(
            runtime_context.config.sampling.initial_center,
            runtime_context.config.sampling.n_walkers,
            runtime_context.config.sampling.initial_jitter_scale,
            runtime_context.config.sampling.random_seed,
        )
    else:
        sampler_initial_state = start_state
        if isinstance(start_state, np.ndarray):
            if np.allclose(start_state, start_state[0]):
                # A persisted checkpoint can be structurally valid on disk
                # while still being unusable for `emcee` if every walker
                # collapsed to the same point. For raw-coordinate resumes we
                # therefore regenerate a small jitter cloud instead of failing
                # before the first step.
                sampler_initial_state = initialize_walkers(
                    runtime_context.config.sampling.initial_center,
                    runtime_context.config.sampling.n_walkers,
                    runtime_context.config.sampling.initial_jitter_scale,
                    runtime_context.config.sampling.random_seed,
                )
            # Resume states can legitimately be tightly clustered, so the
            # independence check is skipped whenever we continue from persisted
            # coordinates rather than a fresh initialization.
            skip_initial_state_check = True
        else:
            # A true `emcee.State` restored from the backend already contains
            # coordinates, log-probabilities, blobs, and random state, so it is
            # the most faithful restart point and should bypass the initial
            # independence heuristic.
            skip_initial_state_check = True

    use_process_pool = runtime_context.parallelism.strategy == "process_pool"
    process_pool = None
    if use_process_pool:
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
        runtime_context.config.sampling.n_walkers,
        runtime_context.config.parameter_schema.n_dim,
        sampler_log_prob_fn,
        pool=process_pool,
        backend=chain_backend,
        blobs_dtype=LOG_PROB_BLOB_DTYPE,
    )
    recent_blobs = deque()
    try:
        for relative_step, state in enumerate(
            sampler.sample(
                sampler_initial_state,
                iterations=runtime_context.config.sampling.n_steps,
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
                    start_step + runtime_context.config.sampling.n_steps,
                    runtime_context.parallelism,
                )
                append_run_log(logs_dir, summary_line)
                if runtime_context.config.runtime.progress:
                    tqdm.write(summary_line)
                recent_blobs.clear()

        if logs_dir is not None and runtime_context.config.runtime.show_stage_timing and recent_blobs:
            summary_line = _summarize_recent_blobs(
                recent_blobs,
                start_step + runtime_context.config.sampling.n_steps,
                start_step + runtime_context.config.sampling.n_steps,
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
        if hasattr(log_prob_fn, "close"):
            log_prob_fn.close()

    return (
        np.asarray(sampler.get_last_sample().coords, dtype=float),
        np.asarray(sampler.get_last_sample().log_prob, dtype=float),
        float(np.mean(sampler.acceptance_fraction)),
    )


def _summarize_recent_blobs(
    recent_blobs,
    current_step: int,
    total_steps: int,
    parallelism,
) -> str:
    """Aggregate recent timing blobs into a one-line summary."""

    total_times = np.array([float(blob["total_log_prob_seconds"]) for blob in recent_blobs], dtype=float)
    likelihood_times = np.array([float(blob["likelihood_seconds"]) for blob in recent_blobs], dtype=float)
    normalization_times = np.array([float(blob["normalization_seconds"]) for blob in recent_blobs], dtype=float)
    fp_prior_times = np.array([float(blob["fp_prior_seconds"]) for blob in recent_blobs], dtype=float)
    return (
        f"step {current_step}/{total_steps} | "
        f"lp {total_times.mean():.2f}s | "
        f"lens {likelihood_times.mean():.2f}s | "
        f"norm {normalization_times.mean():.2f}s | "
        f"fp {fp_prior_times.mean():.2f}s | "
        f"workers {parallelism.worker_processes or parallelism.kernel_threads_per_process} | "
        f"strategy {parallelism.strategy}"
    )
