#!/usr/bin/env python3
"""Compute BIC summaries for the fixed set of 2026-04-20+ production runs.

This script exists for one concrete comparison task:
- compare the four completed `devauc` runs since 2026-04-20
- compare the four completed `sersic` runs since 2026-04-20

The scientific convention is intentionally explicit and local to this script:
- use the post-burn-in posterior samples already stored in `chain.h5`
- treat the maximum of `log_prob - fp_prior_log_term` as the likelihood proxy
- use `n = 23` lens systems as the BIC sample size

Why this is a standalone script instead of being folded into the main runner:
- BIC is a cross-run comparison artifact, not a native per-run inference output
- the current request is scoped to a fixed historical set of completed runs
- keeping it standalone avoids entangling production inference code with a
  one-off model-selection reporting workflow
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np


REPO_ROOT = Path("/Users/liurongfu/Work/CMASS_lens_project")
SUMMARY_OUTPUT_DIR = REPO_ROOT / "outputs" / "_staging" / "20260422_bic_after_20260420"
DEFAULT_BURN_IN = 2000
DEFAULT_SAMPLE_SIZE = 23


RUN_DIRS = (
    REPO_ROOT / "outputs" / "devauc" / "20260420_125501_devauc_m10_sigma_star_fp_prior_slit_rebuilt_obs_within_re_20260420",
    REPO_ROOT / "outputs" / "devauc" / "20260421_144356_devauc_m10_independent_fp_prior_slit_rebuilt_obs_within_re_test_20260421",
    REPO_ROOT / "outputs" / "devauc" / "20260421_162512_devauc_m10_independent_fp_prior_slit_good_drop2sigma_within_re_20260421",
    REPO_ROOT / "outputs" / "devauc" / "20260421_170915_devauc_m10_sigma_star_fp_prior_slit_good_drop2sigma_within_re_20260421",
    REPO_ROOT / "outputs" / "sersic" / "20260420_130706_sersic_m10_sigma_star_fp_prior_slit_rebuilt_obs_within_re_20260420",
    REPO_ROOT / "outputs" / "sersic" / "20260421_145549_sersic_m10_independent_fp_prior_slit_rebuilt_obs_within_re_test_20260421",
    REPO_ROOT / "outputs" / "sersic" / "20260421_163640_sersic_m10_independent_fp_prior_slit_good_drop2sigma_within_re_20260421",
    REPO_ROOT / "outputs" / "sersic" / "20260421_172028_sersic_m10_sigma_star_fp_prior_slit_good_drop2sigma_within_re_20260421",
)


@dataclass(frozen=True)
class BicResult:
    """One fully materialized BIC result for one completed run."""

    run_id: str
    run_dir: str
    profile: str
    gamma_mode: str
    observation_path: str
    burn_in: int
    k: int
    n: int
    max_log_like: float
    bic: float
    bic_definition: str
    log_like_definition: str
    chain_path: str
    fp_prior_field_name: str


def _load_completed_run_payload(run_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load the persisted run metadata and fail fast on incomplete runs."""

    run_result_path = run_dir / "run_result.json"
    metadata_path = run_dir / "metadata.json"
    if not run_result_path.exists():
        raise FileNotFoundError(f"Missing run result: {run_result_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata: {metadata_path}")

    run_result = json.loads(run_result_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if int(run_result.get("completed_steps", -1)) != 10000:
        raise ValueError(f"Run did not complete 10000 steps: {run_dir}")
    return run_result, metadata


def _load_log_likelihood_proxy(chain_path: Path, burn_in: int) -> np.ndarray:
    """Return the post-burn-in likelihood proxy array for one run.

    The chain stores posterior `log_prob`, which already includes the FP prior
    penalty. BIC requires a likelihood term, so this helper subtracts the
    stored `fp_prior_log_term` field from every retained sample.
    """

    if not chain_path.exists():
        raise FileNotFoundError(f"Missing chain backend: {chain_path}")

    with h5py.File(chain_path, "r") as handle:
        mcmc_group = handle["mcmc"]
        if "blobs" not in mcmc_group:
            raise KeyError(f"Run backend is missing `mcmc/blobs`: {chain_path}")
        blobs = mcmc_group["blobs"]
        if "fp_prior_log_term" not in blobs.dtype.names:
            raise KeyError(f"Run backend blobs are missing `fp_prior_log_term`: {chain_path}")
        log_prob = np.asarray(mcmc_group["log_prob"][burn_in:], dtype=float)
        fp_prior = np.asarray(blobs["fp_prior_log_term"][burn_in:], dtype=float)

    return log_prob - fp_prior


def _compute_bic_for_run(run_dir: Path, burn_in: int, n_value: int) -> BicResult:
    """Compute one BIC result and return the fully annotated payload."""

    run_result, metadata = _load_completed_run_payload(run_dir)
    config_summary = metadata["config_summary"]
    parameter_order = list(config_summary["sampling"]["parameter_order"])
    chain_path = run_dir / "chain.h5"

    log_like_proxy = _load_log_likelihood_proxy(chain_path=chain_path, burn_in=burn_in)
    max_log_like = float(np.nanmax(log_like_proxy))
    bic_value = float(len(parameter_order) * math.log(n_value) - 2.0 * max_log_like)

    return BicResult(
        run_id=run_result["run_id"],
        run_dir=str(run_dir),
        profile=metadata["profile_name"],
        gamma_mode=config_summary["gamma_mode"],
        observation_path=run_result["input_observation_path"],
        burn_in=burn_in,
        k=len(parameter_order),
        n=n_value,
        max_log_like=max_log_like,
        bic=bic_value,
        bic_definition="BIC = k * ln(n) - 2 * max_log_like_proxy",
        log_like_definition=(
            "max_log_like_proxy = max(log_prob - fp_prior_log_term) over post-burn-in "
            "samples stored in chain.h5"
        ),
        chain_path=str(chain_path),
        fp_prior_field_name="fp_prior_log_term",
    )


def _write_per_run_result(result: BicResult) -> None:
    """Persist one machine-readable BIC payload into the run directory itself."""

    output_path = Path(result.run_dir) / "bic_result.json"
    output_path.write_text(json.dumps(asdict(result), indent=2, sort_keys=True), encoding="utf-8")


def _group_results_by_profile(results: list[BicResult]) -> dict[str, list[BicResult]]:
    """Group run results by profile and sort by BIC ascending within each group."""

    grouped: dict[str, list[BicResult]] = {"devauc": [], "sersic": []}
    for result in results:
        grouped.setdefault(result.profile, []).append(result)
    for profile_name in grouped:
        grouped[profile_name].sort(key=lambda item: item.bic)
    return grouped


def _build_summary_payload(results: list[BicResult]) -> dict[str, Any]:
    """Create the cross-run JSON summary with per-profile delta-BIC values."""

    grouped = _group_results_by_profile(results)
    profile_payloads: dict[str, list[dict[str, Any]]] = {}
    for profile_name, items in grouped.items():
        if not items:
            profile_payloads[profile_name] = []
            continue
        best_bic = min(item.bic for item in items)
        profile_payloads[profile_name] = [
            {
                **asdict(item),
                "delta_bic": float(item.bic - best_bic),
            }
            for item in items
        ]

    overall_sorted = sorted(results, key=lambda item: item.bic)
    return {
        "burn_in": DEFAULT_BURN_IN,
        "sample_size_n": DEFAULT_SAMPLE_SIZE,
        "run_count": len(results),
        "bic_definition": "BIC = k * ln(n) - 2 * max_log_like_proxy",
        "log_like_definition": (
            "max_log_like_proxy = max(log_prob - fp_prior_log_term) over post-burn-in "
            "samples stored in chain.h5"
        ),
        "profiles": profile_payloads,
        "overall_ranking": [asdict(item) for item in overall_sorted],
    }


def _write_summary_json(summary_payload: dict[str, Any]) -> None:
    """Write the canonical machine-readable summary JSON."""

    SUMMARY_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = SUMMARY_OUTPUT_DIR / "bic_summary.json"
    output_path.write_text(json.dumps(summary_payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_summary_csv(summary_payload: dict[str, Any]) -> None:
    """Write one flat CSV table for spreadsheet-friendly inspection."""

    SUMMARY_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = SUMMARY_OUTPUT_DIR / "bic_summary.csv"
    fieldnames = [
        "profile",
        "run_id",
        "gamma_mode",
        "observation_path",
        "k",
        "n",
        "burn_in",
        "max_log_like",
        "bic",
        "delta_bic",
        "run_dir",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for profile_name in ("devauc", "sersic"):
            for row in summary_payload["profiles"][profile_name]:
                writer.writerow({key: row.get(key) for key in fieldnames})


def _format_profile_table(rows: list[dict[str, Any]]) -> str:
    """Render one compact markdown table for the human-readable report."""

    header = "| run_id | gamma_mode | obs | k | max_log_like | BIC | delta_BIC |\n|---|---|---|---:|---:|---:|---:|"
    body = "\n".join(
        f"| {row['run_id']} | {row['gamma_mode']} | {Path(row['observation_path']).name} | "
        f"{row['k']} | {row['max_log_like']:.6f} | {row['bic']:.6f} | {row['delta_bic']:.6f} |"
        for row in rows
    )
    return f"{header}\n{body}"


def _write_summary_report(summary_payload: dict[str, Any]) -> None:
    """Write a short markdown report for quick human inspection."""

    SUMMARY_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = SUMMARY_OUTPUT_DIR / "bic_report.md"
    lines = [
        "# BIC Summary After 2026-04-20",
        "",
        f"- Sample size `n`: {summary_payload['sample_size_n']}",
        f"- Burn-in removed: {summary_payload['burn_in']}",
        f"- Run count: {summary_payload['run_count']}",
        "",
        "## Devauc",
        "",
        _format_profile_table(summary_payload["profiles"]["devauc"]),
        "",
        "## Sersic",
        "",
        _format_profile_table(summary_payload["profiles"]["sersic"]),
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Compute BIC for the fixed run set and persist per-run plus summary outputs."""

    results = [_compute_bic_for_run(run_dir=run_dir, burn_in=DEFAULT_BURN_IN, n_value=DEFAULT_SAMPLE_SIZE) for run_dir in RUN_DIRS]
    for result in results:
        _write_per_run_result(result)

    summary_payload = _build_summary_payload(results)
    _write_summary_json(summary_payload)
    _write_summary_csv(summary_payload)
    _write_summary_report(summary_payload)

    for result in results:
        print(result.run_id, f"BIC={result.bic:.6f}")


if __name__ == "__main__":
    main()
