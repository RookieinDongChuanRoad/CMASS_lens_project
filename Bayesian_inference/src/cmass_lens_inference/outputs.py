"""
Filesystem output management.

Long-running MCMC jobs need predictable, restartable output directories. This
module owns that contract so the rest of the code can focus on scientific
logic rather than path conventions.
"""

from __future__ import annotations

import json
import pickle
import subprocess
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml
import arviz as az

from .types import RunLayout, RunResult


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


def save_checkpoint(checkpoints_dir: Path, coords: np.ndarray, log_prob: np.ndarray, step: int) -> None:
    """Persist the latest walker state so a run can be resumed later."""

    np.save(checkpoints_dir / "latest_coords.npy", coords)
    np.save(checkpoints_dir / "latest_log_prob.npy", log_prob)
    (checkpoints_dir / "latest_step.txt").write_text(str(step), encoding="utf-8")


def load_checkpoint(checkpoints_dir: Path) -> tuple[np.ndarray, np.ndarray, int]:
    """Load the latest persisted walker state."""

    coords = np.load(checkpoints_dir / "latest_coords.npy")
    log_prob = np.load(checkpoints_dir / "latest_log_prob.npy")
    step = int((checkpoints_dir / "latest_step.txt").read_text(encoding="utf-8").strip())
    return coords, log_prob, step


def save_numpyro_checkpoint(
    checkpoints_dir: Path,
    samples_by_chain: np.ndarray,
    log_prob_by_chain: np.ndarray,
    step: int,
    last_state: Any,
) -> None:
    """
    Persist the latest NumPyro state needed for resume.

    The `.npy` files keep a lightweight, human-inspectable summary equivalent
    to the old walker checkpoint.  The pickled NumPyro state stores the full
    post-warmup sampler state so a resumed run can continue through NumPyro's
    supported `post_warmup_state` API.
    """

    np.save(checkpoints_dir / "latest_samples_by_chain.npy", samples_by_chain)
    np.save(checkpoints_dir / "latest_log_prob_by_chain.npy", log_prob_by_chain)
    (checkpoints_dir / "latest_step.txt").write_text(str(step), encoding="utf-8")
    with (checkpoints_dir / "numpyro_last_state.pkl").open("wb") as handle:
        pickle.dump(last_state, handle)


def load_numpyro_checkpoint(checkpoints_dir: Path) -> tuple[Any, int]:
    """Load a serialized NumPyro post-warmup sampler state."""

    step = int((checkpoints_dir / "latest_step.txt").read_text(encoding="utf-8").strip())
    with (checkpoints_dir / "numpyro_last_state.pkl").open("rb") as handle:
        last_state = pickle.load(handle)
    return last_state, step


def save_numpyro_posterior_artifacts(
    run_dir: Path,
    *,
    samples_by_chain: np.ndarray,
    log_prob_by_chain: np.ndarray,
    diagnostics: dict[str, Any],
    parameter_names: list[str],
) -> tuple[Path, Path]:
    """
    Write NumPyro posterior artifacts in both compact and analysis-friendly forms.

    `samples.npz` is the fastest local format for pipeline code that only needs
    arrays.  `posterior.nc` is written through ArviZ so external analysis tools
    can inspect chains, sample stats, and coordinates without knowing this
    project's internal file conventions.
    """

    samples_path = run_dir / "samples.npz"
    np.savez_compressed(
        samples_path,
        samples_by_chain=np.asarray(samples_by_chain, dtype=float),
        flat_samples=np.asarray(samples_by_chain, dtype=float).reshape(-1, samples_by_chain.shape[-1]),
        log_prob_by_chain=np.asarray(log_prob_by_chain, dtype=float),
        parameter_names=np.asarray(parameter_names, dtype="U"),
    )

    posterior_payload = {
        parameter_name: np.asarray(samples_by_chain[:, :, parameter_index], dtype=float)
        for parameter_index, parameter_name in enumerate(parameter_names)
    }
    sample_stats = {
        "lp": np.asarray(log_prob_by_chain, dtype=float),
    }
    for key, value in diagnostics.get("extra_fields", {}).items():
        array_value = np.asarray(value)
        if array_value.shape[:2] == samples_by_chain.shape[:2]:
            sample_stats[key] = array_value

    inference_data = az.from_dict(
        posterior=posterior_payload,
        sample_stats=sample_stats,
        coords={"parameter": parameter_names},
    )
    posterior_path = run_dir / "posterior.nc"
    inference_data.to_netcdf(posterior_path)
    return samples_path, posterior_path


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
    observation_path: Path,
    output_root_dir: Path,
    random_seed: int,
    config_summary: dict[str, Any],
) -> Path:
    """Write a metadata manifest describing the execution environment."""

    payload = {
        "profile_name": profile_name,
        "config_path": str(config_path),
        "input_observation_path": str(observation_path),
        "output_root_dir": str(output_root_dir),
        "random_seed": random_seed,
        "started_at": datetime.now().isoformat(),
        "git": _collect_git_metadata(),
        "chain_storage": config_summary.get("chain_storage"),
        "parallelism": config_summary.get("parallelism", {}),
        "fp_prior": config_summary.get("fp_prior", {}),
        "sigma_table_path": config_summary.get("sigma_table_path"),
        "sigma_table_mass_definition": config_summary.get("sigma_table_mass_definition"),
        "config_summary": config_summary,
    }
    path = run_dir / "metadata.json"
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
