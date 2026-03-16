from __future__ import annotations

import json
from pathlib import Path

import pytest

from workspace_support import (
    CURRENT_GAMMA_MODE,
    CURRENT_INITIAL_CENTER,
    CURRENT_PARAMETER_ORDER,
    MODE_SETTINGS,
    NOTEBOOK_PARAMETER_LABELS,
    PROFILE_SETTINGS,
    REFERENCE_PARAMETER_ORDER,
    archive_existing_compare_artifacts,
    build_current_config_payload,
    build_reference_run_spec,
    current_center_as_reference_vector,
)


def test_build_current_config_payload_smoke_has_expected_runtime_overrides(tmp_path: Path) -> None:
    output_root = tmp_path / "output" / "current" / "sersic" / "smoke"

    payload = build_current_config_payload(
        profile_name="sersic",
        mode_name="smoke",
        output_root=output_root,
    )

    assert payload["profile"]["name"] == "sersic"
    assert payload["sampling"]["n_walkers"] == 24
    assert payload["sampling"]["n_steps"] == MODE_SETTINGS["smoke"].n_steps
    assert payload["sampling"]["warmup"] == MODE_SETTINGS["smoke"].warmup
    assert payload["runtime"]["checkpoint_every"] == MODE_SETTINGS["smoke"].checkpoint_every
    assert payload["runtime"]["parallel_strategy"] == "kernel_only"
    assert payload["runtime"]["num_threads"] == 12
    assert payload["runtime"]["reserve_cores"] == 2
    assert payload["runtime"]["progress"] is False
    assert Path(payload["output"]["root_dir"]) == output_root
    assert payload["gamma_model"]["mode"] == CURRENT_GAMMA_MODE
    assert payload["sampling"]["initial_center"] == CURRENT_INITIAL_CENTER


def test_build_current_config_payload_compare_uses_profile_specific_data_paths(tmp_path: Path) -> None:
    output_root = tmp_path / "output" / "current" / "devauc" / "compare"

    payload = build_current_config_payload(
        profile_name="devauc",
        mode_name="compare",
        output_root=output_root,
    )

    assert Path(payload["data"]["observation_path"]) == PROFILE_SETTINGS["devauc"].observation_path
    assert Path(payload["data"]["cross_section_path"]) == PROFILE_SETTINGS["devauc"].cross_section_path
    assert payload["gamma_model"]["mode"] == CURRENT_GAMMA_MODE
    assert payload["sampling"]["n_steps"] == MODE_SETTINGS["compare"].n_steps
    assert payload["sampling"]["warmup"] == MODE_SETTINGS["compare"].warmup
    assert payload["runtime"]["checkpoint_every"] == MODE_SETTINGS["compare"].checkpoint_every


def test_build_reference_run_spec_maps_profile_and_mode_to_expected_function_and_paths(tmp_path: Path) -> None:
    output_root = tmp_path / "output" / "reference" / "devauc" / "compare"

    spec = build_reference_run_spec(
        profile_name="devauc",
        mode_name="compare",
        output_root=output_root,
    )

    assert spec.profile_name == "devauc"
    assert spec.module_name == "Population_deV_model"
    assert spec.log_prob_function_name == "P_d_eta"
    assert spec.n_walkers == 24
    assert spec.n_steps == MODE_SETTINGS["compare"].n_steps
    assert spec.pool_processes == 12
    assert spec.output_dir == output_root
    assert spec.backend_path == output_root / "reference_chain.h5"
    assert spec.run_summary_path == output_root / "reference_run_summary.json"
    assert spec.data_links["observations_deV_with_m5_grids.hdf5"].name == "observations_deV_with_m5_grids.hdf5"
    assert spec.data_links["cs_grid_power.h5"].name == "cs_grid_power.h5"


def test_compare_mode_uses_long_run_runtime_settings() -> None:
    compare = MODE_SETTINGS["compare"]

    assert compare.n_steps == 10_000
    assert compare.warmup == 2_000
    assert compare.checkpoint_every == 500
    assert compare.discard == 2_000


def test_notebook_parameter_labels_keep_loga_and_theta0_order() -> None:
    assert NOTEBOOK_PARAMETER_LABELS[-2:] == [r"$\theta_0$", r"$\log a$"]
    assert len(NOTEBOOK_PARAMETER_LABELS) == 12


def test_unknown_profile_raises_clear_error(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unsupported profile"):
        build_current_config_payload("unknown", "smoke", tmp_path)


def test_reference_vector_reorders_theta0_and_loga_for_legacy_modules() -> None:
    reference_vector = current_center_as_reference_vector()

    assert CURRENT_PARAMETER_ORDER[-2:] == ("theta0", "loga")
    assert REFERENCE_PARAMETER_ORDER[-2:] == ("loga", "theta0")
    assert reference_vector[10] == CURRENT_INITIAL_CENTER["loga"]
    assert reference_vector[11] == CURRENT_INITIAL_CENTER["theta0"]


def test_archive_existing_compare_artifacts_moves_stale_results_and_report_assets(tmp_path: Path) -> None:
    stale_compare_dir = tmp_path / "output" / "current" / "sersic" / "compare"
    stale_compare_dir.mkdir(parents=True)
    (stale_compare_dir / "current_run_summary.json").write_text(
        json.dumps(
            {
                "implementation": "current",
                "profile_name": "sersic",
                "mode_name": "compare",
                "requested_steps": 80,
            }
        ),
        encoding="utf-8",
    )
    (stale_compare_dir / "stale.txt").write_text("old compare output", encoding="utf-8")

    report_path = tmp_path / "reports" / "pipeline_comparison.md"
    report_path.parent.mkdir(parents=True)
    report_path.write_text("old report", encoding="utf-8")

    manifest_path = tmp_path / "reports" / "run_manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")

    figure_path = tmp_path / "output" / "figures" / "sersic_compare_corner.png"
    figure_path.parent.mkdir(parents=True)
    figure_path.write_text("png-bytes", encoding="utf-8")

    archive_summary = archive_existing_compare_artifacts(tmp_path)

    archive_root = tmp_path / "output" / "archive" / "compare_80step"
    archived_compare_dir = archive_root / "output" / "current" / "sersic" / "compare"

    assert archive_summary["archived"] is True
    assert stale_compare_dir.exists() is False
    assert (archived_compare_dir / "current_run_summary.json").exists()
    assert (archive_root / "reports" / "pipeline_comparison.md").read_text(encoding="utf-8") == "old report"
    assert (archive_root / "reports" / "run_manifest.json").read_text(encoding="utf-8") == "{}"
    assert (archive_root / "output" / "figures" / "sersic_compare_corner.png").read_text(encoding="utf-8") == "png-bytes"
    assert report_path.exists() is False
    assert manifest_path.exists() is False
    assert figure_path.exists() is False
