"""
Integration tests for the minimal run/resume workflow.

The goal of these tests is not to validate the full scientific model. Instead,
they lock in the first end-to-end contract:
configuration -> data loading -> run directory creation -> checkpoint writing ->
CLI entrypoints.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import emcee
import h5py
import numpy as np
import pytest
import yaml

from cmass_lens_inference.cli import build_argument_parser
from cmass_lens_inference.config import load_runtime_config
from cmass_lens_inference.model import LOG_PROB_BLOB_DTYPE
from cmass_lens_inference.outputs import create_run_layout, save_checkpoint
from cmass_lens_inference.runner import resume_inference, run_inference


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _seed_backend_with_steps(chain_path: Path, n_walkers: int, n_dim: int, n_steps: int) -> None:
    """
    Create a minimal but real `emcee` backend file for resume tests.

    Why this helper exists:
    - The production code is expected to treat `chain.h5` as the source of
      truth during resume.
    - A manually created top-level HDF5 dataset is no longer sufficient once
      the project standardizes on pure `emcee.backends.HDFBackend` output.
    - Using the backend's own `reset/grow/save_step` APIs keeps the fixture
      faithful to the on-disk format that real runs now produce.
    """

    backend = emcee.backends.HDFBackend(str(chain_path))
    backend.reset(n_walkers, n_dim)
    blobs = np.zeros(n_walkers, dtype=LOG_PROB_BLOB_DTYPE)
    blobs["parallel_strategy"] = b"kernel_only"
    backend.grow(n_steps, blobs)
    random_state = np.random.RandomState(123).get_state()

    for step_index in range(n_steps):
        walker_offsets = np.linspace(0.0, 1.0e-3, n_walkers, dtype=float)[:, None]
        coords = np.full((n_walkers, n_dim), 1.0 + step_index, dtype=float) + walker_offsets
        log_prob = np.full(n_walkers, -float(step_index), dtype=float)
        blobs["total_log_prob_seconds"] = 1.0e-6 * (step_index + 1)
        state = emcee.State(coords, log_prob=log_prob, blobs=blobs.copy(), random_state=random_state)
        backend.save_step(state, np.ones(n_walkers, dtype=bool))


def test_run_inference_creates_required_output_files(synthetic_config_path: Path) -> None:
    """
    A minimal inference run should create the required output artifacts under
    the configured output root and return a typed summary object.
    """

    run_result = run_inference(str(synthetic_config_path))

    assert run_result.profile_name == "sersic"
    assert run_result.status == "completed"
    assert run_result.run_dir.exists()
    assert (run_result.run_dir / "config_snapshot.yaml").exists()
    assert (run_result.run_dir / "metadata.json").exists()
    assert (run_result.run_dir / "chain.h5").exists()
    assert (run_result.run_dir / "run_result.json").exists()
    assert (run_result.run_dir / "checkpoints" / "latest_step.txt").exists()
    assert (run_result.run_dir / "logs" / "run.log").exists()

    serialized = json.loads((run_result.run_dir / "run_result.json").read_text(encoding="utf-8"))
    assert serialized["profile_name"] == "sersic"
    assert serialized["status"] == "completed"
    assert serialized["metadata"]["chain_storage"] == "emcee_hdf_backend"
    assert serialized["metadata"]["parallelism"]["strategy"] in {"off", "kernel_only", "process_pool"}
    metadata = json.loads((run_result.run_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["chain_storage"] == "emcee_hdf_backend"
    assert metadata["config_summary"]["gamma_mode"] == "dependent"
    assert metadata["parallelism"]["compute_budget"] >= 1
    assert metadata["parallelism"]["cpu_count"] >= metadata["parallelism"]["compute_budget"]
    run_log_text = (run_result.run_dir / "logs" / "run.log").read_text(encoding="utf-8")
    assert "strategy" in run_log_text
    assert "lp" in run_log_text
    backend = emcee.backends.HDFBackend(str(run_result.run_dir / "chain.h5"))
    assert backend.iteration == 3
    assert backend.get_chain().shape == (3, 24, 12)
    assert backend.get_log_prob().shape == (3, 24)
    with h5py.File(run_result.run_dir / "chain.h5", "r") as handle:
        assert "chain" not in handle
        assert "log_prob" not in handle
        assert "mcmc" in handle


def test_chain_h5_is_readable_by_emcee_hdf_backend(synthetic_config_path: Path) -> None:
    """
    Downstream analysis notebooks should be able to open `chain.h5` directly
    with `emcee.backends.HDFBackend` and use `get_chain` / `get_log_prob`.
    """

    run_result = run_inference(str(synthetic_config_path))

    backend = emcee.backends.HDFBackend(str(run_result.run_dir / "chain.h5"))
    samples = backend.get_chain(flat=True)
    log_prob = backend.get_log_prob(flat=True)

    assert backend.iteration == 3
    assert samples.shape == (3 * 24, 12)
    assert log_prob.shape == (3 * 24,)


def test_run_inference_uses_independent_gamma_parameter_dimension(
    synthetic_independent_config_path: Path,
) -> None:
    """
    Independent gamma mode should shrink the persisted chain dimension to 10.

    This test locks the public sampler/backend contract so downstream tooling
    reads the exact parameter vector implied by the chosen gamma mode.
    """

    run_result = run_inference(str(synthetic_independent_config_path))

    backend = emcee.backends.HDFBackend(str(run_result.run_dir / "chain.h5"))
    assert backend.get_chain().shape == (3, 24, 10)
    assert run_result.metadata["gamma_mode"] == "independent"
    assert run_result.metadata["sampling"]["parameter_order"] == [
        "mu5_0",
        "beta5",
        "xi5",
        "sigma5",
        "mu_gamma_0",
        "sigma_gamma",
        "mu_zs",
        "sigma_zs",
        "theta0",
        "loga",
    ]


def test_resume_inference_reads_existing_checkpoint(synthetic_config_path: Path) -> None:
    """
    Resume should use the previous run directory and advance from the stored
    checkpoint rather than creating a new run tree.
    """

    runtime_config = load_runtime_config(synthetic_config_path)
    run_layout = create_run_layout(
        root_dir=runtime_config.output.root_dir,
        profile_name=runtime_config.profile.name,
        run_label=runtime_config.output.run_label,
        timestamp_text="20260308_180000",
    )
    _seed_backend_with_steps(
        run_layout.run_dir / "chain.h5",
        runtime_config.sampling.n_walkers,
        runtime_config.parameter_schema.n_dim,
        5,
    )
    save_checkpoint(
        run_layout.checkpoints_dir,
        coords=np.ones((runtime_config.sampling.n_walkers, runtime_config.parameter_schema.n_dim)),
        log_prob=np.zeros(runtime_config.sampling.n_walkers),
        step=5,
    )
    (run_layout.run_dir / "config_snapshot.yaml").write_text(
        synthetic_config_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    run_result = resume_inference(str(run_layout.run_dir))

    assert run_result.run_dir == run_layout.run_dir
    assert run_result.start_step == 5
    assert run_result.completed_steps == 8
    assert run_result.status == "completed"
    backend = emcee.backends.HDFBackend(str(run_layout.run_dir / "chain.h5"))
    assert backend.iteration == 8


def test_resume_inference_migrates_legacy_run_snapshot_missing_gamma_mode(
    synthetic_config_path: Path,
) -> None:
    """
    Resume should auto-migrate historical run snapshots that predate gamma mode.

    The migration is intentionally limited to the run-local snapshot so users
    can continue to resume older runs without mutating their source configs.
    """

    runtime_config = load_runtime_config(synthetic_config_path)
    run_layout = create_run_layout(
        root_dir=runtime_config.output.root_dir,
        profile_name=runtime_config.profile.name,
        run_label=runtime_config.output.run_label,
        timestamp_text="20260308_181500",
    )
    _seed_backend_with_steps(
        run_layout.run_dir / "chain.h5",
        runtime_config.sampling.n_walkers,
        runtime_config.parameter_schema.n_dim,
        5,
    )
    save_checkpoint(
        run_layout.checkpoints_dir,
        coords=np.ones((runtime_config.sampling.n_walkers, runtime_config.parameter_schema.n_dim)),
        log_prob=np.zeros(runtime_config.sampling.n_walkers),
        step=5,
    )
    legacy_config_payload = yaml.safe_load(synthetic_config_path.read_text(encoding="utf-8"))
    legacy_config_payload.pop("gamma_model")
    (run_layout.run_dir / "config_snapshot.yaml").write_text(
        yaml.safe_dump(legacy_config_payload, sort_keys=False),
        encoding="utf-8",
    )

    run_result = resume_inference(str(run_layout.run_dir))

    assert run_result.status == "completed"
    migrated_snapshot = yaml.safe_load((run_layout.run_dir / "config_snapshot.yaml").read_text(encoding="utf-8"))
    assert migrated_snapshot["gamma_model"]["mode"] == "dependent"


def test_cli_run_command_executes_minimal_pipeline(synthetic_config_path: Path) -> None:
    """
    The CLI should expose the documented `run` command and report the created
    run directory as machine-readable JSON.
    """

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "cmass_lens_inference.cli",
            "run",
            "--config",
            str(synthetic_config_path),
            "--label",
            "cli",
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
    )

    payload = json.loads(completed.stdout)
    assert payload["profile_name"] == "sersic"
    assert payload["status"] == "completed"
    assert Path(payload["run_dir"]).exists()
    assert payload["metadata"]["parallelism"]["compute_budget"] >= 1


def test_run_inference_supports_process_pool_strategy(synthetic_config_path: Path) -> None:
    """
    The process-pool strategy should run end-to-end under macOS spawn mode and
    record its resolved parallel settings in the artifacts.
    """

    config_text = synthetic_config_path.read_text(encoding="utf-8")
    config_text = config_text.replace("parallel_strategy: auto", "parallel_strategy: process_pool")
    config_text = config_text.replace("num_threads: 0", "num_threads: 2")
    synthetic_config_path.write_text(config_text, encoding="utf-8")

    run_result = run_inference(str(synthetic_config_path))

    assert run_result.metadata["parallelism"]["strategy"] == "process_pool"
    assert run_result.metadata["parallelism"]["worker_processes"] == 2
    run_log_text = (run_result.run_dir / "logs" / "run.log").read_text(encoding="utf-8")
    assert "strategy process_pool" in run_log_text


def test_run_inference_serializes_m10_mass_definition_metadata(
    synthetic_m10_config_path: Path,
) -> None:
    """
    Metadata and run-result payloads should expose the public `m10` naming surface.

    Downstream PPC and trend code use these serialized payloads to decide which
    sigma tables, labels, and result keys to load. This contract must therefore
    be explicit and definition-aware.
    """

    run_result = run_inference(str(synthetic_m10_config_path))

    assert run_result.metadata["mass_definition"]["label"] == "m10"
    assert run_result.metadata["mass_definition"]["enclosed_radius_kpc"] == 10.0
    assert "mu10_0" in run_result.metadata["sampling"]["initial_center"]
    assert "mu5_0" not in run_result.metadata["sampling"]["initial_center"]

    metadata = json.loads((run_result.run_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["config_summary"]["mass_definition"]["label"] == "m10"
    assert metadata["config_summary"]["mass_definition"]["enclosed_radius_kpc"] == 10.0
    assert "mu10_0" in metadata["config_summary"]["sampling"]["initial_center"]
    assert "mu5_0" not in metadata["config_summary"]["sampling"]["initial_center"]


def test_cli_no_longer_exposes_ppt_family_commands() -> None:
    """The inference CLI should stay focused on run/resume after PPT migration."""

    parser = build_argument_parser()
    subparser_actions = [action for action in parser._actions if getattr(action, "choices", None)]
    command_choices = subparser_actions[0].choices

    assert "posterior-predictive" not in command_choices
    assert "posterior-predictive-monitor" not in command_choices
    assert "posterior-trends" not in command_choices
