"""
Parallel-execution policy and process/thread limiting helpers.

This module translates user-facing runtime config into a concrete execution
plan. Keeping the policy here avoids scattering CPU-budget logic throughout
the sampler and runner.
"""

from __future__ import annotations

import os

from statistical_sl.inference.types import ResolvedParallelism, RuntimeOptions
from statistical_sl.numerics.numba.runtime import THREAD_LIMIT_ENV_VARS, apply_thread_limits


def resolve_parallelism(
    runtime_options: RuntimeOptions,
    n_walkers: int,
    cpu_count: int | None = None,
) -> ResolvedParallelism:
    """
    Resolve the concrete parallel-execution settings for a run.

    Resolution rules are intentionally centralized because they become part of
    the reproducibility metadata for every inference result.
    """

    detected_cpu_count = max(1, int(cpu_count or os.cpu_count() or 1))
    reserve_cores = max(0, int(runtime_options.reserve_cores))
    auto_budget = max(1, detected_cpu_count - reserve_cores)
    requested_threads = int(runtime_options.num_threads)
    compute_budget = auto_budget if requested_threads <= 0 else max(1, min(requested_threads, auto_budget))

    requested_strategy = runtime_options.parallel_strategy.strip().lower()
    if requested_strategy == "auto":
        strategy = "kernel_only"
    else:
        strategy = requested_strategy

    if strategy == "off":
        worker_processes = 0
        kernel_threads_per_process = 1
    elif strategy == "kernel_only":
        worker_processes = 0
        kernel_threads_per_process = compute_budget
    elif strategy == "process_pool":
        worker_processes = min(n_walkers, compute_budget)
        kernel_threads_per_process = 1
    else:
        raise ValueError(f"Unsupported parallel strategy: {runtime_options.parallel_strategy}")

    return ResolvedParallelism(
        strategy=strategy,
        cpu_count=detected_cpu_count,
        reserve_cores=reserve_cores,
        compute_budget=compute_budget,
        worker_processes=worker_processes,
        kernel_threads_per_process=kernel_threads_per_process,
    )


__all__ = ["THREAD_LIMIT_ENV_VARS", "apply_thread_limits", "resolve_parallelism"]
