"""
Run the current CMASS inference implementation inside the isolated workspace.

The wrapper keeps all mutable artifacts inside `key_tests/` while reusing the
real source tree from `Bayesian_inference`. It also writes a compact summary
JSON that later report and notebook steps can consume without rediscovering the
run layout.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from workspace_support import (
    CURRENT_GAMMA_MODE,
    CURRENT_PARAMETER_ORDER,
    CURRENT_PROJECT_ROOT,
    MODE_SETTINGS,
    NOTEBOOK_PARAMETER_LABELS,
    WORKSPACE_ROOT,
    build_current_config_payload,
)


def _ensure_current_project_on_path() -> None:
    """Expose the real current implementation package to this wrapper script."""

    for candidate in (CURRENT_PROJECT_ROOT, CURRENT_PROJECT_ROOT / "src"):
        candidate_text = str(candidate)
        if candidate_text not in sys.path:
            sys.path.insert(0, candidate_text)


def run_current_pipeline(profile_name: str, mode_name: str) -> dict[str, object]:
    """
    Execute one current-pipeline run and persist a machine-readable summary.

    The summary becomes the stable handoff contract for report generation and
    notebook visualization. This avoids any fragile directory scraping later.
    """

    _ensure_current_project_on_path()

    from cmass_lens_inference.runner import run_inference

    mode = MODE_SETTINGS[mode_name]
    output_root = WORKSPACE_ROOT / "output" / "current" / profile_name / mode_name
    output_root.mkdir(parents=True, exist_ok=True)
    config_path = WORKSPACE_ROOT / "configs" / f"current_{profile_name}_{mode_name}.yaml"

    payload = build_current_config_payload(
        profile_name=profile_name,
        mode_name=mode_name,
        output_root=output_root,
    )
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    run_result = run_inference(str(config_path), label=mode_name)
    summary = {
        "implementation": "current",
        "profile_name": profile_name,
        "mode_name": mode_name,
        "requested_steps": mode.n_steps,
        "warmup": mode.warmup,
        "discard": mode.discard,
        "gamma_mode": CURRENT_GAMMA_MODE,
        "parameter_order": list(CURRENT_PARAMETER_ORDER),
        "parameter_labels": NOTEBOOK_PARAMETER_LABELS,
        "config_path": str(config_path),
        "output_root": str(output_root),
        "run_dir": str(run_result.run_dir),
        "chain_path": str(Path(run_result.run_dir) / "chain.h5"),
        "run_result_path": str(Path(run_result.run_dir) / "run_result.json"),
        "metadata_path": str(Path(run_result.run_dir) / "metadata.json"),
        "run_log_path": str(Path(run_result.run_dir) / "logs" / "run.log"),
        "completed_steps": int(run_result.completed_steps),
        "acceptance_fraction_mean": float(run_result.acceptance_fraction_mean),
    }
    summary_path = output_root / "current_run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def _parse_args() -> argparse.Namespace:
    """Parse the small CLI surface for isolated current-pipeline runs."""

    parser = argparse.ArgumentParser(description="Run the current CMASS pipeline inside key_tests.")
    parser.add_argument("--profile", choices=("sersic", "devauc"), required=True)
    parser.add_argument("--mode", choices=("smoke", "compare"), required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    print(json.dumps(run_current_pipeline(args.profile, args.mode), indent=2, sort_keys=True))
