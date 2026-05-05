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

import numpy as np
import pytest
import yaml

from cmass_lens_inference.cli import build_argument_parser
from cmass_lens_inference.config import load_runtime_config
from cmass_lens_inference.runner import resume_inference, run_inference


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _force_single_chain_for_orchestration_test(config_path: Path) -> None:
    """
    Keep runner/CLI orchestration tests intentionally small.

    The production default is now `2 * ndim` chains so real runs keep the same
    broad initialization convention.  These integration tests are different:
    they only need to prove that run/resume,
    metadata, checkpoints, and output files are wired correctly.  Pinning one
    sequential chain here prevents those contract tests from becoming expensive
    implicit sampler-quality tests.
    """

    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload.setdefault("sampling", {}).update(
        {
            "num_chains": 1,
            "chain_method": "sequential",
        }
    )
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def test_run_inference_creates_required_output_files(synthetic_config_path: Path) -> None:
    """
    A minimal inference run should create the required output artifacts under
    the configured output root and return a typed summary object.
    """

    _force_single_chain_for_orchestration_test(synthetic_config_path)
    run_result = run_inference(str(synthetic_config_path))

    assert run_result.profile_name == "sersic"
    assert run_result.status == "completed"
    assert run_result.run_dir.exists()
    assert (run_result.run_dir / "config_snapshot.yaml").exists()
    assert (run_result.run_dir / "metadata.json").exists()
    assert (run_result.run_dir / "samples.npz").exists()
    assert (run_result.run_dir / "posterior.nc").exists()
    assert (run_result.run_dir / "run_result.json").exists()
    assert (run_result.run_dir / "checkpoints" / "latest_step.txt").exists()
    assert (run_result.run_dir / "checkpoints" / "numpyro_last_state.pkl").exists()
    assert (run_result.run_dir / "logs" / "run.log").exists()

    serialized = json.loads((run_result.run_dir / "run_result.json").read_text(encoding="utf-8"))
    assert serialized["profile_name"] == "sersic"
    assert serialized["status"] == "completed"
    assert serialized["metadata"]["chain_storage"] == "numpyro_arviz_netcdf"
    assert serialized["metadata"]["parallelism"]["strategy"] in {"off", "kernel_only", "process_pool"}
    metadata = json.loads((run_result.run_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["chain_storage"] == "numpyro_arviz_netcdf"
    assert metadata["config_summary"]["model"]["name"] == "cmass"
    assert metadata["config_summary"]["model"]["metadata"]["gamma_distribution"] == "sigma_star_dependent"
    assert metadata["config_summary"]["box_prior"]["mu5h_0"] == [9.0, 12.0]
    assert metadata["parallelism"]["compute_budget"] >= 1
    assert metadata["parallelism"]["cpu_count"] >= metadata["parallelism"]["compute_budget"]
    run_log_text = (run_result.run_dir / "logs" / "run.log").read_text(encoding="utf-8")
    assert "strategy" in run_log_text
    assert "numpyro complete" in run_log_text
    with np.load(run_result.run_dir / "samples.npz") as payload:
        assert payload["samples_by_chain"].shape == (1, 3, 11)
        assert payload["log_prob_by_chain"].shape == (1, 3)


def test_run_inference_serializes_fp_prior_metadata(
    synthetic_fp_prior_config_path: Path,
) -> None:
    """
    FP-enabled runs should persist canonical within-Re sigma provenance.

    Sigma-table paths are no longer production inputs.  The run metadata should
    instead point at the canonical dataset and record that the dataset exposes
    the FP-within-Re capability used by the prior.
    """

    _force_single_chain_for_orchestration_test(synthetic_fp_prior_config_path)
    runtime_config = load_runtime_config(synthetic_fp_prior_config_path)
    run_result = run_inference(str(synthetic_fp_prior_config_path))

    metadata_payload = json.loads((run_result.run_dir / "metadata.json").read_text(encoding="utf-8"))
    run_result_payload = json.loads((run_result.run_dir / "run_result.json").read_text(encoding="utf-8"))

    assert run_result.metadata["fp_prior"]["enabled"] is True
    assert run_result.metadata["data"]["inference_dataset_path"] == str(runtime_config.data.inference_dataset_path)
    assert "velocity_dispersion.fp_within_re.v1" in run_result.metadata["data"]["canonical_capabilities"]
    assert run_result.metadata["fp_sigma_definition"] == "within_re"
    assert run_result.metadata["fp_sigma_table_leaf_path"] == "/velocity_dispersion_grids/fp_within_re"
    assert metadata_payload["fp_prior"]["enabled"] is True
    assert metadata_payload["input_inference_dataset_path"] == str(runtime_config.data.inference_dataset_path)
    assert "velocity_dispersion.fp_within_re.v1" in metadata_payload["canonical_capabilities"]
    assert metadata_payload["fp_sigma_definition"] == "within_re"
    assert metadata_payload["fp_sigma_table_leaf_path"] == "/velocity_dispersion_grids/fp_within_re"
    assert run_result_payload["metadata"]["fp_prior"]["enabled"] is True
    assert run_result_payload["metadata"]["data"]["inference_dataset_path"] == str(
        runtime_config.data.inference_dataset_path
    )
    assert "velocity_dispersion.fp_within_re.v1" in run_result_payload["metadata"]["data"]["canonical_capabilities"]
    assert run_result_payload["metadata"]["fp_sigma_definition"] == "within_re"
    assert run_result_payload["metadata"]["fp_sigma_table_leaf_path"] == "/velocity_dispersion_grids/fp_within_re"


def test_samples_npz_is_readable_as_primary_numpyro_backend(synthetic_config_path: Path) -> None:
    """
    Downstream analysis notebooks should be able to open `samples.npz` directly
    and recover posterior samples plus log-probability diagnostics.
    """

    _force_single_chain_for_orchestration_test(synthetic_config_path)
    run_result = run_inference(str(synthetic_config_path))

    with np.load(run_result.run_dir / "samples.npz") as payload:
        samples = payload["flat_samples"]
        log_prob = payload["log_prob_by_chain"].reshape(-1)

    assert samples.shape == (3, 11)
    assert log_prob.shape == (3,)


def test_run_inference_uses_default_cmass_parameter_dimension(
    synthetic_config_path: Path,
) -> None:
    """The default CMASS model should persist an 11D h-unit parameter chain."""

    _force_single_chain_for_orchestration_test(synthetic_config_path)
    run_result = run_inference(str(synthetic_config_path))

    with np.load(run_result.run_dir / "samples.npz") as payload:
        assert payload["samples_by_chain"].shape == (1, 3, 11)
    assert run_result.metadata["model"]["metadata"]["gamma_distribution"] == "sigma_star_dependent"
    assert run_result.metadata["sampling"]["parameter_order"] == [
        "mu5h_0",
        "beta5h",
        "xi5h",
        "sigma5h",
        "mu_gamma_0",
        "beta_sigma_star_gamma",
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

    _force_single_chain_for_orchestration_test(synthetic_config_path)
    first_result = run_inference(str(synthetic_config_path))
    runtime_config = load_runtime_config(first_result.run_dir / "config_snapshot.yaml")

    run_result = resume_inference(str(first_result.run_dir))

    assert run_result.run_dir == first_result.run_dir
    assert run_result.start_step == 3
    assert run_result.completed_steps == 6
    assert run_result.status == "completed"
    with np.load(first_result.run_dir / "samples.npz") as payload:
        assert payload["samples_by_chain"].shape == (1, 3, runtime_config.parameter_schema.n_dim)


def test_resume_inference_rejects_run_snapshot_missing_box_prior(
    synthetic_config_path: Path,
) -> None:
    """Resume should use the same explicit config contract as fresh runs."""

    _force_single_chain_for_orchestration_test(synthetic_config_path)
    first_result = run_inference(str(synthetic_config_path))
    legacy_config_payload = yaml.safe_load(synthetic_config_path.read_text(encoding="utf-8"))
    legacy_config_payload.pop("box_prior")
    (first_result.run_dir / "config_snapshot.yaml").write_text(
        yaml.safe_dump(legacy_config_payload, sort_keys=False),
        encoding="utf-8",
    )

    with pytest.raises(KeyError, match="Missing required config section: box_prior"):
        resume_inference(str(first_result.run_dir))


def test_cli_run_command_executes_minimal_pipeline(synthetic_config_path: Path) -> None:
    """
    The CLI should expose the documented `run` command and report the created
    run directory as machine-readable JSON.
    """

    _force_single_chain_for_orchestration_test(synthetic_config_path)
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

    _force_single_chain_for_orchestration_test(synthetic_config_path)
    config_text = synthetic_config_path.read_text(encoding="utf-8")
    config_text = config_text.replace("parallel_strategy: auto", "parallel_strategy: process_pool")
    config_text = config_text.replace("num_threads: 0", "num_threads: 2")
    synthetic_config_path.write_text(config_text, encoding="utf-8")

    run_result = run_inference(str(synthetic_config_path))

    assert run_result.metadata["parallelism"]["strategy"] == "process_pool"
    assert run_result.metadata["parallelism"]["worker_processes"] == 1
    run_log_text = (run_result.run_dir / "logs" / "run.log").read_text(encoding="utf-8")
    assert "strategy process_pool" in run_log_text


def test_run_inference_supports_process_pool_strategy_with_fp_prior(
    synthetic_fp_prior_config_path: Path,
) -> None:
    """
    FP-enabled runs should stay compatible with the process-pool strategy.

    The main performance target is `kernel_only`, but keeping `fp_prior=true`
    functional under spawn-based walker parallelism protects an existing public
    runtime option and guards against nested-thread regressions.
    """

    _force_single_chain_for_orchestration_test(synthetic_fp_prior_config_path)
    config_text = synthetic_fp_prior_config_path.read_text(encoding="utf-8")
    config_text = config_text.replace("parallel_strategy: auto", "parallel_strategy: process_pool")
    config_text = config_text.replace("num_threads: 0", "num_threads: 2")
    synthetic_fp_prior_config_path.write_text(config_text, encoding="utf-8")

    run_result = run_inference(str(synthetic_fp_prior_config_path))

    assert run_result.metadata["parallelism"]["strategy"] == "process_pool"
    assert run_result.metadata["parallelism"]["worker_processes"] == 1
    assert run_result.metadata["fp_prior"]["enabled"] is True
    run_log_text = (run_result.run_dir / "logs" / "run.log").read_text(encoding="utf-8")
    assert "strategy process_pool" in run_log_text
    assert "numpyro complete" in run_log_text


def test_cli_no_longer_exposes_ppt_family_commands() -> None:
    """The inference CLI should stay focused on run/resume after PPT migration."""

    parser = build_argument_parser()
    subparser_actions = [action for action in parser._actions if getattr(action, "choices", None)]
    command_choices = subparser_actions[0].choices

    assert "posterior-predictive" not in command_choices
    assert "posterior-predictive-monitor" not in command_choices
    assert "posterior-trends" not in command_choices
