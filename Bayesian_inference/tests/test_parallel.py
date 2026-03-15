"""
Tests for runtime parallel-strategy resolution and worker thread limiting.

These tests lock the policy layer before implementation because the resolved
parallel settings become part of the run's public metadata contract.
"""

from __future__ import annotations

import os

from cmass_lens_inference.parallel import apply_thread_limits, resolve_parallelism
from cmass_lens_inference.types import RuntimeOptions


def test_resolve_parallelism_auto_uses_kernel_only_on_14_core_machine() -> None:
    """
    After the monolithic-kernel refactor, auto mode should default to a
    single-process numba path on this workstation profile.
    """

    runtime_options = RuntimeOptions(
        checkpoint_every=100,
        parallel_strategy="auto",
        progress=False,
        progress_summary_every=25,
        show_stage_timing=True,
        disable_hdf5_file_locking=False,
        num_threads=0,
        reserve_cores=2,
    )

    resolved = resolve_parallelism(runtime_options, n_walkers=24, cpu_count=14)

    assert resolved.strategy == "kernel_only"
    assert resolved.cpu_count == 14
    assert resolved.compute_budget == 12
    assert resolved.worker_processes == 0
    assert resolved.kernel_threads_per_process == 12


def test_resolve_parallelism_kernel_only_keeps_work_in_one_process() -> None:
    """
    Kernel-only mode should not allocate worker processes.
    """

    runtime_options = RuntimeOptions(
        checkpoint_every=100,
        parallel_strategy="kernel_only",
        progress=False,
        progress_summary_every=25,
        show_stage_timing=True,
        disable_hdf5_file_locking=False,
        num_threads=8,
        reserve_cores=2,
    )

    resolved = resolve_parallelism(runtime_options, n_walkers=24, cpu_count=14)

    assert resolved.strategy == "kernel_only"
    assert resolved.compute_budget == 8
    assert resolved.worker_processes == 0
    assert resolved.kernel_threads_per_process == 8


def test_apply_thread_limits_sets_single_thread_environment(monkeypatch) -> None:
    """
    Worker initialization must clamp common math-library thread counts to one.
    """

    for variable_name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "OMP_MAX_ACTIVE_LEVELS",
        "KMP_WARNINGS",
    ):
        monkeypatch.delenv(variable_name, raising=False)

    apply_thread_limits(1)

    assert os.environ["OMP_NUM_THREADS"] == "1"
    assert os.environ["OPENBLAS_NUM_THREADS"] == "1"
    assert os.environ["MKL_NUM_THREADS"] == "1"
    assert os.environ["VECLIB_MAXIMUM_THREADS"] == "1"
    assert os.environ["NUMEXPR_NUM_THREADS"] == "1"
    assert os.environ["OMP_MAX_ACTIVE_LEVELS"] == "1"
    assert os.environ["KMP_WARNINGS"] == "0"
