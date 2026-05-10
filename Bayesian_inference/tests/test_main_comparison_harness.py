"""Tests for the branch-aware log-probability comparison harness."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "compare_log_prob_with_main.py"
)


def _load_harness_module():
    """Import the script as a module without executing its CLI entrypoint."""

    spec = importlib.util.spec_from_file_location("compare_log_prob_with_main", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_comparison_payload_schema_contains_numeric_and_timing_fields() -> None:
    """The report payload should expose values needed for acceptance review."""

    harness = _load_harness_module()
    main_result = {
        "log_prob_value": -10.0,
        "normalization_value": 0.5,
        "steady_log_prob_median_seconds": 2.0,
        "steady_log_prob_min_seconds": 1.8,
        "steady_log_prob_mean_seconds": 2.1,
        "first_call_seconds": 12.0,
    }
    candidate_result = {
        "log_prob_value": -9.75,
        "normalization_value": 0.45,
        "steady_log_prob_median_seconds": 1.0,
        "steady_log_prob_min_seconds": 0.9,
        "steady_log_prob_mean_seconds": 1.1,
        "first_call_seconds": 4.0,
    }

    payload = harness.build_comparison_payload(
        case_name="cmass_synthetic_sersic",
        main_git_hash="main-hash",
        candidate_git_hash="candidate-hash",
        main_result=main_result,
        candidate_result=candidate_result,
        environment={"python": "3.x", "numba_threads": 1},
    )

    assert payload["case_name"] == "cmass_synthetic_sersic"
    assert payload["main_git_hash"] == "main-hash"
    assert payload["candidate_git_hash"] == "candidate-hash"
    assert payload["main"]["log_prob_value"] == -10.0
    assert payload["candidate"]["normalization_value"] == 0.45
    assert payload["comparison"]["log_prob_abs_diff"] == 0.25
    assert payload["comparison"]["normalization_abs_diff"] == pytest.approx(0.05)
    assert payload["comparison"]["steady_speed_ratio_candidate_over_main"] == 0.5


def test_normalize_branch_result_accepts_numpy_blob_scalar_names() -> None:
    """Branch subprocess results should be reduced to a stable JSON shape."""

    harness = _load_harness_module()
    normalized = harness.normalize_branch_result(
        {
            "first_call_seconds": 3.0,
            "first_log_prob_value": -5.0,
            "first_normalization_value": 0.25,
            "steady_log_prob_value": -5.0,
            "steady_normalization_value": 0.25,
            "steady_log_prob_median_seconds": 0.1,
            "steady_log_prob_min_seconds": 0.09,
            "steady_log_prob_mean_seconds": 0.11,
        }
    )

    assert normalized["log_prob_value"] == -5.0
    assert normalized["normalization_value"] == 0.25
    assert normalized["steady_log_prob_median_seconds"] == 0.1
