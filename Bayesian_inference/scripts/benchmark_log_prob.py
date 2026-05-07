#!/usr/bin/env python3
"""
Benchmark the production Numba/emcee inference backend.

The benchmark measures the backend-owned `log_prob` entrypoint, not sampler
throughput.  The first call is reported separately because Numba may compile
specialized kernels on first use; steady-state timings are measured only after
that warmup call.  Optional emcee smoke execution is intentionally disabled by
default because production configs can contain long chains.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
OUTPUT_ROOT = Path("/Users/liurongfu/Work/CMASS_lens_project/outputs/benchmarks")

if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from cmass_lens_inference.config import load_runtime_config
from cmass_lens_inference.numba_backend.likelihood_engine import build_compiled_model, log_prob
from cmass_lens_inference.runner import run_inference


def parse_args() -> argparse.Namespace:
    """Parse command-line options for backend benchmarking."""

    parser = argparse.ArgumentParser(description="Benchmark the Numba/emcee production backend.")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "current_numba_emcee.yaml"),
        help="YAML runtime config to benchmark.",
    )
    parser.add_argument("--repeats", type=int, default=5, help="Steady-state log_prob repeats.")
    parser.add_argument(
        "--output-dir",
        default=str(OUTPUT_ROOT),
        help="Directory where the JSON benchmark summary is written.",
    )
    parser.add_argument(
        "--run-smoke",
        action="store_true",
        help="Run the supplied config through runner.py after log_prob timing.",
    )
    return parser.parse_args()


def _decode_blob_string(value: object) -> str:
    """Decode one fixed-width bytes field from the Numba diagnostic blob."""

    if isinstance(value, bytes):
        return value.decode("utf-8").rstrip("\x00")
    if hasattr(value, "decode"):
        return value.decode("utf-8").rstrip("\x00")
    return str(value)


def benchmark_numba_log_prob(config_path: Path, repeats: int) -> dict[str, float | str | int]:
    """
    Measure first-call and steady-state Numba log-probability latency.

    The returned `first_call_seconds` includes any Numba compilation triggered
    by this process.  The steady-state fields use the same compiled model and
    theta vector, so they represent repeated posterior evaluations.
    """

    runtime_config = load_runtime_config(config_path)
    compiled_model = build_compiled_model(runtime_config)
    theta = runtime_config.sampling.initial_center.to_array()

    first_started = time.perf_counter()
    first_value, first_blob = log_prob(theta, compiled_model)
    first_call_seconds = time.perf_counter() - first_started

    repeated_times: list[float] = []
    repeated_values: list[float] = []
    repeated_normalizations: list[float] = []
    for _ in range(max(1, int(repeats))):
        started = time.perf_counter()
        value, blob = log_prob(theta, compiled_model)
        repeated_times.append(time.perf_counter() - started)
        repeated_values.append(float(value))
        repeated_normalizations.append(float(blob["normalization_value"]))

    return {
        "backend": "numba",
        "sampler": "emcee",
        "model": runtime_config.model.name,
        "profile": runtime_config.profile.name,
        "kernel": _decode_blob_string(first_blob["kernel"]),
        "parallel_strategy": _decode_blob_string(first_blob["parallel_strategy"]),
        "first_call_seconds": float(first_call_seconds),
        "first_log_prob_value": float(first_value),
        "first_normalization_value": float(first_blob["normalization_value"]),
        "steady_log_prob_median_seconds": float(statistics.median(repeated_times)),
        "steady_log_prob_min_seconds": float(min(repeated_times)),
        "steady_log_prob_mean_seconds": float(statistics.mean(repeated_times)),
        "steady_log_prob_value": float(repeated_values[-1]),
        "steady_normalization_value": float(repeated_normalizations[-1]),
        "repeats": int(repeats),
        "n_walkers": int(runtime_config.sampling.n_walkers),
        "n_steps": int(runtime_config.sampling.n_steps),
        "burn_in": int(runtime_config.sampling.burn_in),
        "gamma_points": int(runtime_config.integration.gamma_points),
        "mstar_points": int(runtime_config.integration.mstar_points),
        "normalization_samples": int(runtime_config.integration.normalization_samples),
    }


def run_emcee_smoke(config_path: Path) -> dict[str, float | str | int]:
    """
    Run the supplied config through the production runner.

    The caller must supply a short config when using this option.  The function
    deliberately does not rewrite chain length or output paths, because hidden
    mutation in a benchmark tool would make acceptance evidence hard to audit.
    """

    started = time.perf_counter()
    result = run_inference(str(config_path))
    elapsed = time.perf_counter() - started
    return {
        "smoke_total_seconds": float(elapsed),
        "completed_steps": int(result.completed_steps),
        "acceptance_fraction_mean": float(result.acceptance_fraction_mean),
        "chain_path": str(result.chain_path),
        "run_dir": str(result.run_dir),
    }


def main() -> None:
    """Run the benchmark and write a JSON summary for later comparison."""

    args = parse_args()
    output_root = Path(args.output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    config_path = Path(args.config).expanduser().resolve()
    payload: dict[str, object] = {
        "timestamp": datetime.now().isoformat(),
        "config_path": str(config_path),
        "log_prob": benchmark_numba_log_prob(config_path, args.repeats),
    }
    if args.run_smoke:
        payload["emcee_smoke"] = run_emcee_smoke(config_path)

    output_path = output_root / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_numba_benchmark.json"
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"benchmark_path": str(output_path), **payload}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
