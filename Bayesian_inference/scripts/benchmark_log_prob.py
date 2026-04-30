#!/usr/bin/env python3
"""
Benchmark the JAX/NumPyro CMASS inference backend.

The first JAX call includes compilation and should not be compared with steady
state execution.  This script records both numbers explicitly so performance
work can distinguish one-time compile cost from repeated posterior evaluation
speed inside NumPyro.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
OUTPUT_ROOT = Path("/Users/liurongfu/Work/CMASS_lens_project/outputs/benchmarks")

if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from cmass_lens_inference.config import load_runtime_config
from cmass_lens_inference.jax_model import build_jax_model, log_prob
from cmass_lens_inference.runner import run_inference


def parse_args() -> argparse.Namespace:
    """Parse the benchmark CLI."""

    parser = argparse.ArgumentParser(description="Benchmark the JAX CMASS inference backend.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "current_jax.yaml"))
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--run-smoke", action="store_true")
    parser.add_argument("--smoke-samples", type=int, default=2)
    parser.add_argument("--smoke-warmup", type=int, default=1)
    return parser.parse_args()


def benchmark_jax_log_prob(config_path: Path, repeats: int) -> dict[str, float | str | int]:
    """
    Measure first-call compile time and repeated JAX log-probability latency.

    The returned `first_call_seconds` includes JAX tracing/compilation.  The
    median steady-state value is measured only after that warmup call.
    """

    runtime_config = load_runtime_config(config_path)
    compiled_model = build_jax_model(runtime_config)
    theta = runtime_config.sampling.initial_center.to_array()

    first_started = time.perf_counter()
    first_value, first_blob = log_prob(theta, compiled_model)
    first_call_seconds = time.perf_counter() - first_started

    repeated_times: list[float] = []
    repeated_values: list[float] = []
    for _ in range(max(1, int(repeats))):
        started = time.perf_counter()
        value, _blob = log_prob(theta, compiled_model)
        repeated_times.append(time.perf_counter() - started)
        repeated_values.append(float(value))

    return {
        "backend": "jax",
        "model": runtime_config.model.name,
        "profile": runtime_config.profile.name,
        "first_call_seconds": first_call_seconds,
        "first_log_prob_value": float(first_value),
        "first_normalization_value": float(first_blob["normalization_value"]),
        "steady_log_prob_median_seconds": statistics.median(repeated_times),
        "steady_log_prob_min_seconds": min(repeated_times),
        "steady_log_prob_value": repeated_values[-1],
        "repeats": int(repeats),
    }


def run_numpyro_smoke(config_path: Path, smoke_samples: int, smoke_warmup: int) -> dict[str, float | str | int]:
    """Run a tiny NumPyro smoke benchmark using a temporary YAML snapshot."""

    runtime_config = load_runtime_config(config_path)
    runtime_config = replace(
        runtime_config,
        sampling=replace(
            runtime_config.sampling,
            num_chains=1,
            num_samples=int(smoke_samples),
            num_warmup=int(smoke_warmup),
            chain_method="sequential",
            n_steps=int(smoke_samples),
            warmup=int(smoke_warmup),
        ),
        output=replace(runtime_config.output, run_label=f"numpyro_smoke_{runtime_config.profile.name}"),
    )

    config_payload = {
        "profile": {"name": runtime_config.profile.name},
        "model": {
            "name": runtime_config.model.name,
            "components": runtime_config.model.components,
        },
        "mass_definition": {"enclosed_radius_kpc": runtime_config.mass_definition.radius_kpc},
        "gamma_model": {"mode": runtime_config.gamma_model.mode},
        "data": {
            "observation_path": str(runtime_config.data.observation_path),
            "cross_section_path": str(runtime_config.data.cross_section_path),
        },
        "box_prior": runtime_config.parameter_schema.serialize_public_box_prior(),
        "sampling": {
            "n_walkers": runtime_config.sampling.n_walkers,
            "n_steps": runtime_config.sampling.n_steps,
            "warmup": runtime_config.sampling.warmup,
            "num_chains": runtime_config.sampling.num_chains,
            "num_samples": runtime_config.sampling.num_samples,
            "num_warmup": runtime_config.sampling.num_warmup,
            "thinning": runtime_config.sampling.thinning,
            "chain_method": runtime_config.sampling.chain_method,
            "random_seed": runtime_config.sampling.random_seed,
            "initial_center": runtime_config.sampling.initial_center.to_public_dict(
                runtime_config.mass_definition
            ),
            "initial_jitter_scale": runtime_config.sampling.initial_jitter_scale,
        },
        "integration": {
            "gamma_points": runtime_config.integration.gamma_points,
            "mstar_points": runtime_config.integration.mstar_points,
            "normalization_samples": runtime_config.integration.normalization_samples,
        },
        "cosmology": {
            "h0": runtime_config.cosmology.h0,
            "omega_m": runtime_config.cosmology.omega_m,
        },
        "runtime": {
            "checkpoint_every": runtime_config.runtime.checkpoint_every,
            "parallel_strategy": runtime_config.runtime.parallel_strategy,
            "progress": False,
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
    }

    temp_config_path = OUTPUT_ROOT / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_numpyro_smoke.yaml"
    temp_config_path.write_text(yaml.safe_dump(config_payload, sort_keys=False), encoding="utf-8")
    started = time.perf_counter()
    result = run_inference(str(temp_config_path))
    elapsed = time.perf_counter() - started
    return {
        "smoke_total_seconds": elapsed,
        "num_samples": int(smoke_samples),
        "num_warmup": int(smoke_warmup),
        "acceptance_fraction_mean": result.acceptance_fraction_mean,
        "run_dir": str(result.run_dir),
    }


def main() -> None:
    """Execute the benchmark and persist a JSON summary."""

    args = parse_args()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    config_path = Path(args.config).expanduser().resolve()
    payload: dict[str, object] = {
        "timestamp": datetime.now().isoformat(),
        "config_path": str(config_path),
        "log_prob": benchmark_jax_log_prob(config_path, args.repeats),
    }
    if args.run_smoke:
        payload["numpyro_smoke"] = run_numpyro_smoke(config_path, args.smoke_samples, args.smoke_warmup)

    output_path = OUTPUT_ROOT / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_jax_benchmark.json"
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"benchmark_path": str(output_path), **payload}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
