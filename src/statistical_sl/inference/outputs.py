"""
Filesystem output management.

Long-running MCMC jobs need predictable, restartable output directories. This
module owns that contract so the rest of the code can focus on scientific
logic rather than path conventions.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from statistical_sl.core.manifests import RUN_MANIFEST_FILENAME
from statistical_sl.inference.types import RunLayout, RunResult


def create_run_layout(
    root_dir: str | Path,
    profile_name: str,
    run_label: str,
    timestamp_text: str | None = None,
) -> RunLayout:
    """Create the on-disk directory tree for a single run."""

    root = Path(root_dir).expanduser().resolve()
    profile_dir = root / profile_name
    profile_dir.mkdir(parents=True, exist_ok=True)
    stamp = timestamp_text or datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_label = run_label.strip() or "default"
    run_id = f"{stamp}_{profile_name}_{safe_label}"
    run_dir = profile_dir / run_id
    checkpoints_dir = run_dir / "checkpoints"
    logs_dir = run_dir / "logs"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    (logs_dir / "run.log").touch()
    return RunLayout(
        root_dir=root,
        profile_dir=profile_dir,
        run_id=run_id,
        run_dir=run_dir,
        checkpoints_dir=checkpoints_dir,
        logs_dir=logs_dir,
    )


def refresh_latest_pointer(profile_dir: Path, run_id: str) -> None:
    """
    Refresh the stable pointer to the most recent run.

    Preferred behavior is a symlink named `latest`. If the filesystem does not
    allow symlinks, the function falls back to a plain-text `LATEST_RUN` file.
    """

    latest_path = profile_dir / "latest"
    try:
        if latest_path.exists() or latest_path.is_symlink():
            latest_path.unlink()
        latest_path.symlink_to(run_id)
    except OSError:
        (profile_dir / "LATEST_RUN").write_text(f"{run_id}\n", encoding="utf-8")


def save_emcee_checkpoint(
    checkpoints_dir: Path,
    walker_coords: np.ndarray,
    log_prob: np.ndarray,
    step: int,
) -> None:
    """
    Persist the lightweight emcee resume checkpoint.

    `chain.h5` remains the source of truth for production samples.  These small
    arrays are an operational safety net for interrupted runs where HDFBackend
    state may not yet be readable but the runner has reached a checkpoint
    boundary.
    """

    np.save(checkpoints_dir / "latest_walker_coords.npy", np.asarray(walker_coords, dtype=float))
    np.save(checkpoints_dir / "latest_log_prob.npy", np.asarray(log_prob, dtype=float))
    (checkpoints_dir / "latest_step.txt").write_text(str(step), encoding="utf-8")


def load_emcee_checkpoint(checkpoints_dir: Path) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Load the lightweight emcee checkpoint arrays.

    The caller decides whether the raw coordinates are good enough to resume or
    whether a persisted `chain.h5` state should take precedence.
    """

    coords_path = checkpoints_dir / "latest_walker_coords.npy"
    log_prob_path = checkpoints_dir / "latest_log_prob.npy"
    step_path = checkpoints_dir / "latest_step.txt"
    if not coords_path.exists() or not log_prob_path.exists() or not step_path.exists():
        raise FileNotFoundError(f"No complete emcee checkpoint exists in {checkpoints_dir}.")
    return (
        np.load(coords_path),
        np.load(log_prob_path),
        int(step_path.read_text(encoding="utf-8").strip()),
    )


def write_config_snapshot(run_dir: Path, raw_config_text: str) -> Path:
    """Persist the exact configuration used to launch the run."""

    path = run_dir / "config_snapshot.yaml"
    path.write_text(raw_config_text, encoding="utf-8")
    return path


def _collect_git_metadata() -> dict[str, Any]:
    """
    Collect Git metadata when available.

    The current project directory is not a Git repository, so this function
    must fail soft rather than breaking the run.
    """

    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        commit = completed.stdout.strip()
        return {"commit": commit}
    except Exception:
        return {"commit": None}


def write_metadata(
    run_dir: Path,
    profile_name: str,
    config_path: Path,
    inference_dataset_path: Path,
    output_root_dir: Path,
    random_seed: int,
    config_summary: dict[str, Any],
) -> Path:
    """Write a metadata manifest describing the execution environment."""

    payload = {
        "profile_name": profile_name,
        "config_path": str(config_path),
        "input_inference_dataset_path": str(inference_dataset_path),
        "output_root_dir": str(output_root_dir),
        "random_seed": random_seed,
        "started_at": datetime.now().isoformat(),
        "git": _collect_git_metadata(),
        "chain_storage": config_summary.get("chain_storage"),
        "canonical_capabilities": config_summary.get("data", {}).get("canonical_capabilities", []),
        "unit_convention": config_summary.get("unit_convention"),
        "h_ref": config_summary.get("h_ref"),
        "mass_definition": config_summary.get("mass_definition"),
        "parallelism": config_summary.get("parallelism", {}),
        "fp_prior": config_summary.get("fp_prior", {}),
        "fp_sigma_definition": config_summary.get("fp_sigma_definition"),
        "fp_sigma_table_leaf_path": config_summary.get("fp_sigma_table_leaf_path"),
        "config_summary": config_summary,
    }
    path = run_dir / "metadata.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def write_run_manifest(
    run_dir: Path,
    *,
    profile_name: str,
    config_path: Path,
    inference_dataset_path: Path,
    output_root_dir: Path,
    random_seed: int,
    config_summary: dict[str, Any],
) -> Path:
    """
    Write the shared run manifest used by cross-workflow tooling.

    ``metadata.json`` is the inference-specific execution record.  The manifest
    carries the narrower package-level contract that downstream workflow code
    can look for without knowing which stage produced the run directory.
    Keeping both files explicit avoids silently changing old metadata readers
    while still aligning new runs with the repository-wide artifact names.
    """

    payload = {
        "schema_version": "statistical_sl_run_manifest_v1",
        "stage": "inference",
        "profile_name": profile_name,
        "model_name": config_summary.get("model", {}).get("name"),
        "config_path": str(config_path),
        "input_inference_dataset_path": str(inference_dataset_path),
        "output_root_dir": str(output_root_dir),
        "random_seed": int(random_seed),
        "started_at": datetime.now().isoformat(),
        "chain_path": "chain.h5",
        "run_result_path": "run_result.json",
        "metadata_path": "metadata.json",
        "chain_storage": config_summary.get("chain_storage"),
        "backend": config_summary.get("backend"),
        "sampler": config_summary.get("sampler"),
        "unit_convention": config_summary.get("unit_convention"),
        "h_ref": config_summary.get("h_ref"),
        "mass_definition": config_summary.get("mass_definition"),
        "parameter_order": config_summary.get("sampling", {}).get("parameter_order", []),
        "parallelism": config_summary.get("parallelism", {}),
    }
    path = run_dir / RUN_MANIFEST_FILENAME
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def write_run_result(run_dir: Path, run_result: RunResult) -> Path:
    """Serialize the run summary to JSON for later automation and inspection."""

    path = run_dir / "run_result.json"
    path.write_text(json.dumps(run_result.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return path


def append_run_log(logs_dir: Path, line: str) -> Path:
    """Append a single structured line to the run log."""

    path = logs_dir / "run.log"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"{line}\n")
    return path
