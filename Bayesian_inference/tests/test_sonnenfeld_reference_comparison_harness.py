"""Tests for the Sonnenfeld external-reference comparison harness."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "compare_sonnenfeld_with_reference.py"
)


def _load_harness_module():
    """Import the comparison script as a module without executing its CLI."""

    spec = importlib.util.spec_from_file_location(
        "compare_sonnenfeld_with_reference",
        SCRIPT_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_audit_manifest_marks_missing_reference_artifacts_as_data_gated(
    tmp_path: Path,
) -> None:
    """Missing grids should gate later phases instead of producing a false pass."""

    harness = _load_harness_module()
    reference_root = tmp_path / "reference"
    reference_root.mkdir()
    (reference_root / "SLACS_table.cat").write_text("lens data\n", encoding="utf-8")
    (reference_root / "full_inference.hdf5").write_bytes(b"placeholder")

    payload = harness.build_audit_payload(
        reference_root=reference_root,
        candidate_root=tmp_path / "candidate",
        model="sonnenfeld2024_slacs",
        candidate_dataset_path=None,
    )

    assert payload["status"] == "skipped_data_gated"
    assert payload["model"] == "sonnenfeld2024_slacs"
    assert payload["reference"]["root"] == str(reference_root)
    assert payload["reference"]["artifacts"]["SLACS_table.cat"]["status"] == "present"
    assert payload["reference"]["artifacts"]["fibre_crosssect_grid.hdf5"]["status"] == "missing"
    assert payload["candidate"]["canonical_dataset"]["status"] == "missing"
    assert "skipped_data_gated" in payload["status_options"]
    assert "not_comparable" in payload["status_options"]


def test_audit_manifest_records_hash_size_and_hdf5_schema(tmp_path: Path) -> None:
    """Present artifacts should carry enough manifest data to be auditable."""

    h5py = __import__("h5py")
    harness = _load_harness_module()
    reference_root = tmp_path / "reference"
    reference_root.mkdir()
    table_path = reference_root / "SLACS_table.cat"
    table_path.write_text("lens data\n", encoding="utf-8")
    chain_path = reference_root / "full_inference.hdf5"
    with h5py.File(chain_path, "w") as handle:
        handle.create_dataset("chain", data=[1.0, 2.0])

    artifact = harness.describe_artifact(
        chain_path,
        required_for_full_comparison=True,
    )

    assert artifact["status"] == "present"
    assert artifact["sha256"]
    assert artifact["size_bytes"] > 0
    assert artifact["hdf5"]["datasets"]["chain"]["shape"] == [2]


def test_cli_audit_only_writes_manifest_and_summary(tmp_path: Path) -> None:
    """The audit-only CLI should write JSON and Markdown without importing MCMC scripts."""

    reference_root = tmp_path / "reference"
    output_dir = tmp_path / "out"
    reference_root.mkdir()
    (reference_root / "SLACS_table.cat").write_text("lens data\n", encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--reference-root",
            str(reference_root),
            "--candidate-root",
            str(tmp_path / "candidate"),
            "--model",
            "sonnenfeld2024_slacs",
            "--audit-only",
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    stdout_payload = json.loads(completed.stdout)
    manifest_path = Path(stdout_payload["manifest_path"])
    summary_path = Path(stdout_payload["summary_path"])

    assert manifest_path.exists()
    assert summary_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "skipped_data_gated"
    assert manifest["reference"]["artifacts"]["SLACS_table.cat"]["status"] == "present"


def test_reference_theta_mapper_names_each_permutation_explicitly() -> None:
    """Reference and local 12D theta vectors have different orders."""

    harness = _load_harness_module()
    reference_theta = {
        "mu_m5": 1.0,
        "sigma_m5": 2.0,
        "beta_m5": 3.0,
        "xi_m5": 4.0,
        "mu_gamma": 5.0,
        "sigma_gamma": 6.0,
        "beta_gamma": 7.0,
        "xi_gamma": 8.0,
        "mu_zs": 9.0,
        "sigma_zs": 10.0,
        "t_find": 11.0,
        "la_find": 12.0,
    }

    mapped = harness.map_reference_theta_to_local(reference_theta)

    assert mapped["reference_order"] == list(harness.REFERENCE_PARAMETER_NAMES)
    assert mapped["local_order"] == list(harness.LOCAL_PARAMETER_NAMES)
    assert mapped["local_values"] == {
        "mu5_0": 1.0,
        "beta5": 3.0,
        "xi5": 4.0,
        "sigma5": 2.0,
        "mu_gamma_0": 5.0,
        "beta_gamma": 7.0,
        "xi_gamma": 8.0,
        "sigma_gamma": 6.0,
        "mu_zs": 9.0,
        "sigma_zs": 10.0,
        "theta0": 11.0,
        "loga": 12.0,
    }
    assert mapped["local_theta"] == [1.0, 3.0, 4.0, 2.0, 5.0, 7.0, 8.0, 6.0, 9.0, 10.0, 11.0, 12.0]


def test_reference_theta_mapper_rejects_missing_or_wrong_length_values() -> None:
    """Partial or wrong-dimensional theta input should fail before comparison."""

    harness = _load_harness_module()

    try:
        harness.map_reference_theta_to_local({"mu_m5": 1.0})
    except ValueError as exc:
        assert "missing reference theta parameters" in str(exc)
    else:  # pragma: no cover - failure branch
        raise AssertionError("missing theta parameters were accepted")

    try:
        harness.map_reference_theta_to_local([1.0, 2.0])
    except ValueError as exc:
        assert "expected 12 reference theta values" in str(exc)
    else:  # pragma: no cover - failure branch
        raise AssertionError("wrong-length theta vector was accepted")


def test_primitive_payload_compares_reference_formulas_against_local_primitives() -> None:
    """Primitive comparisons should run without any large HDF5 artifacts."""

    harness = _load_harness_module()

    payload = harness.build_primitive_payload()

    assert payload["phase"] == "primitive"
    assert payload["status"] == "passed"
    assert payload["comparisons"]["pfind"]["status"] == "passed"
    assert payload["comparisons"]["source_redshift_mask"]["status"] == "passed"
    assert payload["comparisons"]["size_relation"]["status"] == "passed"
    assert payload["comparisons"]["fp_prior_defaults"]["status"] == "passed"
    assert payload["comparisons"]["fp_ols2"]["status"] == "passed"
    assert payload["comparisons"]["fp_prior_defaults"]["reference"]["fiducial_scatter"] == 0.047
    assert payload["comparisons"]["fp_prior_defaults"]["local"]["scatter_error"] == 0.008


def test_cli_primitive_stage_writes_comparison_without_hdf5(tmp_path: Path) -> None:
    """The primitive stage should be runnable even when grid artifacts are absent."""

    reference_root = tmp_path / "reference"
    output_dir = tmp_path / "out"
    reference_root.mkdir()

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--reference-root",
            str(reference_root),
            "--candidate-root",
            str(tmp_path / "candidate"),
            "--model",
            "sonnenfeld2024_slacs",
            "--stages",
            "primitive",
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    stdout_payload = json.loads(completed.stdout)
    comparison_path = Path(stdout_payload["comparison_path"])
    summary_path = Path(stdout_payload["summary_path"])

    assert comparison_path.exists()
    assert summary_path.exists()
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    assert comparison["status"] == "passed"
    assert comparison["comparisons"]["fp_ols2"]["status"] == "passed"


def test_grid_stage_is_data_gated_when_required_artifacts_are_missing(
    tmp_path: Path,
) -> None:
    """Grid comparison should report missing inputs instead of inventing a pass."""

    harness = _load_harness_module()
    reference_root = tmp_path / "reference"
    reference_root.mkdir()
    (reference_root / "SLACS_table.cat").write_text("lens data\n", encoding="utf-8")

    payload = harness.build_data_grid_payload(
        reference_root=reference_root,
        candidate_root=tmp_path / "candidate",
        candidate_dataset_path=None,
    )

    assert payload["phase"] == "data_grid"
    assert payload["status"] == "skipped_data_gated"
    assert "reference:fibre_crosssect_grid.hdf5" in payload["missing_required_artifacts"]
    assert "candidate:canonical_dataset" in payload["missing_required_artifacts"]
