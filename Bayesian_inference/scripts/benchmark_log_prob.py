#!/usr/bin/env python3
"""
Benchmark the current CMASS inference implementation against the reference tree.

The script serves two purposes:
- quantify whether the current monolithic-kernel refactor has caught up with
  `/Users/liurongfu/Desktop/CMASS_lens`
- provide a repeatable smoke benchmark for different runtime strategies

Results are written under the long-lived outputs root instead of the repo so
benchmark artifacts do not pollute source control.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import tempfile
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path

os.environ.setdefault("OMP_MAX_ACTIVE_LEVELS", "1")
os.environ.setdefault("KMP_WARNINGS", "0")

import numba
import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
REFERENCE_ROOT = Path("/Users/liurongfu/Desktop/CMASS_lens")
REFERENCE_SOURCE_ROOT = REFERENCE_ROOT / "src"
OUTPUT_ROOT = Path("/Users/liurongfu/Work/CMASS_lens_project/outputs/benchmarks")

for candidate in (SOURCE_ROOT, REFERENCE_SOURCE_ROOT):
    text = str(candidate)
    if text not in sys.path:
        sys.path.insert(0, text)

from cmass_lens_inference.config import load_runtime_config
from cmass_lens_inference.model import build_compiled_model, log_prob
from cmass_lens_inference.parallel import apply_thread_limits, resolve_parallelism
from cmass_lens_inference.runner import run_inference

from cmass_lens.cosmology.distances import build_distance_table
from cmass_lens.inference.params import build_parameter_spec
from cmass_lens.io.datasets import load_cs_grid, load_lens_dataset
from cmass_lens.likelihood.model import ModelContext as ReferenceModelContext
from cmass_lens.profiles.fixed import build_fixed_profile
from cmass_lens.utils.config import load_yaml, merge_dict


def parse_args() -> argparse.Namespace:
    """Parse the small benchmark CLI surface."""

    parser = argparse.ArgumentParser(description="Benchmark current and reference CMASS inference implementations.")
    parser.add_argument("--profile", choices=["devauc", "sersic"], default="devauc")
    parser.add_argument("--strategy", choices=["auto", "off", "kernel_only", "process_pool"], default="auto")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--run-smoke", action="store_true")
    parser.add_argument("--smoke-steps", type=int, default=20)
    parser.add_argument("--smoke-warmup", type=int, default=0)
    parser.add_argument("--num-threads", type=int, default=0)
    return parser.parse_args()


def _runtime_config_for_benchmark(profile_name: str, strategy: str, num_threads: int):
    """Load the current config and apply in-memory benchmark overrides."""

    runtime_config = load_runtime_config(PROJECT_ROOT / "configs" / f"{profile_name}.yaml")
    runtime_options = replace(
        runtime_config.runtime,
        parallel_strategy=strategy,
        num_threads=num_threads,
        progress=False,
        show_stage_timing=False,
    )
    sampling = replace(runtime_config.sampling, n_steps=1, warmup=0)
    output = replace(runtime_config.output, run_label=f"benchmark_{profile_name}_{strategy}")
    return replace(runtime_config, runtime=runtime_options, sampling=sampling, output=output)


def benchmark_current_log_prob(profile_name: str, strategy: str, repeats: int, num_threads: int) -> dict[str, float | int | str]:
    """Benchmark the current implementation's compiled `log_prob` path."""

    runtime_config = _runtime_config_for_benchmark(profile_name, strategy, num_threads)
    compiled_model = build_compiled_model(runtime_config)
    resolved = resolve_parallelism(runtime_config.runtime, runtime_config.sampling.n_walkers)
    apply_thread_limits(resolved.kernel_threads_per_process)
    theta = runtime_config.sampling.initial_center.to_array()

    # One warm-up call is required so the timings measure execution, not JIT compilation.
    log_prob(theta, compiled_model)

    total_times: list[float] = []
    likelihood_times: list[float] = []
    normalization_times: list[float] = []
    values: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter()
        value, blob = log_prob(theta, compiled_model)
        total_times.append(time.perf_counter() - started)
        likelihood_times.append(float(blob["likelihood_seconds"]))
        normalization_times.append(float(blob["normalization_seconds"]))
        values.append(float(value))

    return {
        "strategy": resolved.strategy,
        "compute_budget": resolved.compute_budget,
        "worker_processes": resolved.worker_processes,
        "kernel_threads_per_process": resolved.kernel_threads_per_process,
        "likelihood_median_seconds": statistics.median(likelihood_times),
        "normalization_median_seconds": statistics.median(normalization_times),
        "log_prob_median_seconds": statistics.median(total_times),
        "log_prob_value": values[-1],
    }


def build_reference_context(profile_name: str, compute_budget: int) -> tuple[ReferenceModelContext, np.ndarray]:
    """Build the reference tree's compiled context for the requested profile."""

    cosmology_cfg = load_yaml(REFERENCE_ROOT / "configs/common/cosmology.yaml")
    prior_cfg = load_yaml(REFERENCE_ROOT / "configs/model/common_priors.yaml")
    infer_cfg = load_yaml(REFERENCE_ROOT / "configs/inference/default.yaml")
    model_cfg = load_yaml(REFERENCE_ROOT / "configs/model" / f"{profile_name}.yaml")
    merged_cfg = merge_dict(
        {},
        {
            "cosmology": cosmology_cfg,
            "model": model_cfg,
            "inference": infer_cfg,
            "priors": prior_cfg,
        },
    )
    merged_cfg["inference"]["performance"]["n_processes"] = 1
    merged_cfg["inference"]["performance"]["numba_threads_per_process"] = compute_budget

    lens_data = load_lens_dataset(model_cfg["observation_hdf5"], model_name=profile_name)
    cs_grid = load_cs_grid(model_cfg["cs_grid_hdf5"])
    distance_table = build_distance_table(
        h0=float(cosmology_cfg["H0"]),
        omega_m=float(cosmology_cfg["Omega_m"]),
        z_max=float(cosmology_cfg["z_table_max"]),
        n_grid=int(cosmology_cfg["z_table_size"]),
    )
    fixed_profile = build_fixed_profile(model_cfg)
    param_spec = build_parameter_spec(model_cfg=model_cfg, prior_cfg=prior_cfg)
    zd_cfg = prior_cfg["zd_fixed_dist"]

    context = ReferenceModelContext.build(
        lens_data=lens_data,
        cs_grid=cs_grid,
        distance_table=distance_table,
        param_spec=param_spec,
        fixed_profile=fixed_profile,
        mu_d=float(zd_cfg["mu_d"]),
        sigma_d=float(zd_cfg["sigma_d"]),
        n_gamma=int(merged_cfg["inference"]["integration"]["n_gamma"]),
        n_mstar=int(merged_cfg["inference"]["integration"]["n_mstar"]),
        mstar_range_sigma=float(merged_cfg["inference"]["integration"]["mstar_range_sigma"]),
        n_norm=int(merged_cfg["inference"]["normalization"]["n_norm"]),
        normalization_min_value=float(merged_cfg["inference"]["normalization"]["min_value"]),
        gamma_trunc_low=float(merged_cfg["inference"]["normalization"]["gamma_trunc_low"]),
        gamma_trunc_high=float(merged_cfg["inference"]["normalization"]["gamma_trunc_high"]),
        seed=int(merged_cfg["inference"]["seed"]),
    )
    # The current project and the reference share the same parameter names/order.
    current_config = load_runtime_config(PROJECT_ROOT / "configs" / f"{profile_name}.yaml")
    reference_theta = np.array(
        [getattr(current_config.sampling.initial_center, name) for name in param_spec.names],
        dtype=np.float64,
    )
    return context, reference_theta


def benchmark_reference_log_prob(profile_name: str, repeats: int, compute_budget: int) -> dict[str, float | str]:
    """Benchmark the reference implementation's `log_prob` path."""

    numba.set_num_threads(max(1, int(compute_budget)))
    context, theta = build_reference_context(profile_name, compute_budget)
    context.log_prob(theta)

    total_times: list[float] = []
    values: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter()
        value = context.log_prob(theta)
        total_times.append(time.perf_counter() - started)
        values.append(float(value))

    return {
        "log_prob_median_seconds": statistics.median(total_times),
        "log_prob_value": values[-1],
    }


def run_current_smoke(profile_name: str, strategy: str, num_threads: int, smoke_steps: int, smoke_warmup: int) -> dict[str, float | str]:
    """Run a short end-to-end current-implementation smoke benchmark."""

    runtime_config = _runtime_config_for_benchmark(profile_name, strategy, num_threads)
    runtime_config = replace(
        runtime_config,
        sampling=replace(runtime_config.sampling, n_steps=smoke_steps, warmup=smoke_warmup),
        output=replace(runtime_config.output, run_label=f"smoke_{profile_name}_{strategy}"),
    )

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8") as handle:
        config_path = Path(handle.name)
        yaml.safe_dump(
            {
                "profile": {"name": runtime_config.profile.name},
                "data": {
                    "observation_path": str(runtime_config.data.observation_path),
                    "cross_section_path": str(runtime_config.data.cross_section_path),
                },
                "sampling": {
                    "n_walkers": runtime_config.sampling.n_walkers,
                    "n_steps": runtime_config.sampling.n_steps,
                    "warmup": runtime_config.sampling.warmup,
                    "random_seed": runtime_config.sampling.random_seed,
                    "initial_center": runtime_config.sampling.initial_center.to_dict(),
                    "initial_jitter_scale": runtime_config.sampling.initial_jitter_scale,
                },
                "integration": {
                    "gamma_points": runtime_config.integration.gamma_points,
                    "mstar_points": runtime_config.integration.mstar_points,
                    "normalization_samples": runtime_config.integration.normalization_samples,
                },
                "runtime": {
                    "distance_table_max_z": runtime_config.runtime.distance_table_max_z,
                    "distance_table_size": runtime_config.runtime.distance_table_size,
                    "checkpoint_every": runtime_config.runtime.checkpoint_every,
                    "parallel_strategy": runtime_config.runtime.parallel_strategy,
                    "progress": runtime_config.runtime.progress,
                    "progress_summary_every": runtime_config.runtime.progress_summary_every,
                    "show_stage_timing": runtime_config.runtime.show_stage_timing,
                    "disable_hdf5_file_locking": runtime_config.runtime.disable_hdf5_file_locking,
                    "num_threads": runtime_config.runtime.num_threads,
                    "reserve_cores": runtime_config.runtime.reserve_cores,
                },
                "output": {
                    "root_dir": str(runtime_config.output.root_dir),
                    "run_label": runtime_config.output.run_label,
                    "overwrite_latest": runtime_config.output.overwrite_latest,
                },
            },
            handle,
            sort_keys=False,
        )

    started = time.perf_counter()
    result = run_inference(str(config_path), label=f"smoke_{profile_name}_{strategy}")
    elapsed = time.perf_counter() - started
    return {
        "smoke_total_seconds": elapsed,
        "smoke_steps": smoke_steps,
        "smoke_throughput_steps_per_second": smoke_steps / elapsed if elapsed > 0.0 else 0.0,
        "run_dir": str(result.run_dir),
    }


def main() -> None:
    """Execute the selected benchmark and persist a JSON summary."""

    args = parse_args()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    current_metrics = benchmark_current_log_prob(args.profile, args.strategy, args.repeats, args.num_threads)
    reference_metrics = benchmark_reference_log_prob(
        args.profile,
        args.repeats,
        int(current_metrics["compute_budget"]),
    )

    payload: dict[str, object] = {
        "timestamp": datetime.now().isoformat(),
        "profile": args.profile,
        "strategy": args.strategy,
        "repeats": args.repeats,
        "current": current_metrics,
        "reference": reference_metrics,
        "speed_ratio_vs_reference": (
            float(current_metrics["log_prob_median_seconds"]) / float(reference_metrics["log_prob_median_seconds"])
        ),
    }

    if args.run_smoke:
        payload["smoke"] = run_current_smoke(
            args.profile,
            args.strategy,
            args.num_threads,
            args.smoke_steps,
            args.smoke_warmup,
        )

    output_path = OUTPUT_ROOT / (
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.profile}_{args.strategy}_benchmark.json"
    )
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"benchmark_path": str(output_path), **payload}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
