"""
Run the copied legacy reference pipeline inside the isolated workspace.

The legacy scripts are preserved as much as possible. This driver only handles
parameterization, output paths, multiprocessing setup, and summary generation.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import sys
from pathlib import Path

import emcee
import numpy as np

from workspace_support import (
    CURRENT_GAMMA_MODE,
    MODE_SETTINGS,
    NOTEBOOK_PARAMETER_LABELS,
    REFERENCE_PARAMETER_ORDER,
    TOOLS_ROOT,
    WORKSPACE_ROOT,
    build_reference_run_spec,
    current_center_as_reference_vector,
)


THREAD_LIMIT_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def _apply_single_thread_limits() -> None:
    """Clamp worker-local math libraries to one thread each."""

    for variable_name in THREAD_LIMIT_ENV_VARS:
        os.environ[variable_name] = "1"
    os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"
    os.environ["KMP_WARNINGS"] = "0"
    try:
        import numba

        numba.set_num_threads(1)
    except Exception:
        pass


def _prepare_reference_imports(reference_pipeline_root: Path) -> None:
    """Ensure the copied legacy scripts and their helper tools are importable."""

    for candidate in (reference_pipeline_root, TOOLS_ROOT):
        candidate_text = str(candidate)
        if candidate_text not in sys.path:
            sys.path.insert(0, candidate_text)


def _import_log_prob_callable(reference_pipeline_root: Path, module_name: str, function_name: str):
    """Import the selected legacy log-probability callable."""

    _prepare_reference_imports(reference_pipeline_root)
    module = __import__(module_name, fromlist=[function_name])
    return getattr(module, function_name)


def run_reference_pipeline(profile_name: str, mode_name: str) -> dict[str, object]:
    """
    Execute one legacy reference-pipeline run and persist a summary JSON.

    The copied reference scripts keep their original relative `./data/...`
    assumptions, so the driver changes into `reference_pipeline/` before
    importing them. Output paths remain absolute and continue to land under
    `key_tests/output/reference/...`.
    """

    mode = MODE_SETTINGS[mode_name]
    output_root = WORKSPACE_ROOT / "output" / "reference" / profile_name / mode_name
    output_root.mkdir(parents=True, exist_ok=True)
    spec = build_reference_run_spec(
        profile_name=profile_name,
        mode_name=mode_name,
        output_root=output_root,
    )

    backend_path = spec.backend_path
    if backend_path.exists():
        backend_path.unlink()

    backend = emcee.backends.HDFBackend(str(backend_path))
    backend.reset(spec.n_walkers, len(REFERENCE_PARAMETER_ORDER))

    initial_center = np.asarray(current_center_as_reference_vector(), dtype=float)
    rng = np.random.default_rng(7)
    starting_value = initial_center[None, :] + 0.01 * rng.normal(
        size=(spec.n_walkers, initial_center.size)
    ) * initial_center[None, :]

    reference_pipeline_root = WORKSPACE_ROOT / "reference_pipeline"
    original_cwd = Path.cwd()
    try:
        os.chdir(reference_pipeline_root)
        log_prob_callable = _import_log_prob_callable(
            reference_pipeline_root=reference_pipeline_root,
            module_name=spec.module_name,
            function_name=spec.log_prob_function_name,
        )
        _apply_single_thread_limits()
        context = multiprocessing.get_context("spawn")
        with context.Pool(processes=spec.pool_processes, initializer=_apply_single_thread_limits) as pool:
            sampler = emcee.EnsembleSampler(
                spec.n_walkers,
                len(REFERENCE_PARAMETER_ORDER),
                log_prob_callable,
                backend=backend,
                pool=pool,
            )
            sampler.run_mcmc(starting_value, nsteps=spec.n_steps, progress=False)
            acceptance_fraction_mean = float(np.mean(sampler.acceptance_fraction))
    finally:
        os.chdir(original_cwd)

    summary = {
        "implementation": "reference",
        "profile_name": spec.profile_name,
        "mode_name": spec.mode_name,
        "requested_steps": spec.n_steps,
        "warmup": mode.warmup,
        "discard": mode.discard,
        "gamma_mode": CURRENT_GAMMA_MODE,
        "parameter_order": list(REFERENCE_PARAMETER_ORDER),
        "parameter_labels": NOTEBOOK_PARAMETER_LABELS,
        "module_name": spec.module_name,
        "log_prob_function_name": spec.log_prob_function_name,
        "output_root": str(spec.output_dir),
        "chain_path": str(spec.backend_path),
        "summary_path": str(spec.run_summary_path),
        "pool_processes": spec.pool_processes,
        "completed_steps": spec.n_steps,
        "acceptance_fraction_mean": acceptance_fraction_mean,
    }
    spec.run_summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def _parse_args() -> argparse.Namespace:
    """Parse the CLI arguments for isolated reference-pipeline runs."""

    parser = argparse.ArgumentParser(description="Run the copied reference pipeline inside key_tests.")
    parser.add_argument("--profile", choices=("sersic", "devauc"), required=True)
    parser.add_argument("--mode", choices=("smoke", "compare"), required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    print(json.dumps(run_reference_pipeline(args.profile, args.mode), indent=2, sort_keys=True))
