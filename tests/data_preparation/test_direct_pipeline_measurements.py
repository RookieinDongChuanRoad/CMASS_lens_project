"""Tests for trusted velocity-measurement ingestion.

The direct pipeline accepts one upstream measurement contract and one adapter
for the current pPXF CSV export.  Both paths must resolve into the same
accepted-observation model while keeping rejected rows in audit provenance.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from statistical_sl.data_preparation.direct_pipeline.records import SigmaObservation
from statistical_sl.data_preparation.direct_pipeline.measurements import (
    PpxfAdapterConfig,
    VelocityMeasurementReadResult,
    load_ppxf_velocity_measurements,
    read_velocity_measurements_v1_csv,
)


VELOCITY_MEASUREMENTS_V1_STANDARD_COLUMNS = [
    "schema_version",
    "lens_id",
    "obs_tag",
    "sigma_kms",
    "sigma_err_kms",
    "sigma_error_kind",
    "measurement_status",
    "use_for_likelihood",
    "source_system",
    "source_file",
    "aperture_shape",
    "aperture_width_arcsec",
    "aperture_height_arcsec",
    "aperture_radius_arcsec",
    "seeing_fwhm_arcsec",
]


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> Path:
    """Write a small CSV fixture with explicit columns and rows."""

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_velocity_measurements_v1_reader_parses_accepted_and_rejected_rows(tmp_path: Path) -> None:
    """The standard upstream contract should parse likelihood aperture metadata.

    A velocity-dispersion measurement is not only a scalar sigma value.  The
    upstream measurement row also defines the aperture and seeing under which
    the sigma was extracted; without that contract, the direct canonical
    pipeline cannot build the correct observed-aperture `s2_grid`.
    """

    csv_path = _write_csv(
        tmp_path / "velocity_measurements_v1.csv",
        VELOCITY_MEASUREMENTS_V1_STANDARD_COLUMNS,
        [
            {
                "schema_version": "velocity_measurements_v1",
                "lens_id": "023817-054555",
                "obs_tag": "A",
                "sigma_kms": 226.4,
                "sigma_err_kms": 9.6,
                "sigma_error_kind": "statistical",
                "measurement_status": "success",
                "use_for_likelihood": "true",
                "source_system": "HSCJ023817-054555A",
                "source_file": "/tmp/ppxf.csv",
                "aperture_shape": "rectangular",
                "aperture_width_arcsec": 1.6,
                "aperture_height_arcsec": 0.9,
                "aperture_radius_arcsec": "",
                "seeing_fwhm_arcsec": 0.7,
            },
            {
                "schema_version": "velocity_measurements_v1",
                "lens_id": "023817-054556",
                "obs_tag": "",
                "sigma_kms": 0.0,
                "sigma_err_kms": 8.0,
                "sigma_error_kind": "statistical",
                "measurement_status": "failed",
                "use_for_likelihood": "false",
                "source_system": "HSCJ023817-054556",
                "source_file": "/tmp/ppxf.csv",
                "aperture_shape": "",
                "aperture_width_arcsec": "",
                "aperture_height_arcsec": "",
                "aperture_radius_arcsec": "",
                "seeing_fwhm_arcsec": "",
            },
        ],
    )

    result = read_velocity_measurements_v1_csv(csv_path)

    assert isinstance(result, VelocityMeasurementReadResult)
    assert len(result.accepted) == 1
    assert len(result.rejected) == 1
    observation = result.accepted[0]
    assert isinstance(observation, SigmaObservation)
    assert observation.lens_id == "023817-054555"
    assert observation.obs_tag == "A"
    assert observation.sigma_kms == pytest.approx(226.4)
    assert observation.sigma_err_kms == pytest.approx(9.6)
    assert observation.sigma_error_kind == "statistical"
    assert observation.aperture_shape == "rectangular"
    assert observation.aperture_width_arcsec == pytest.approx(1.6)
    assert observation.aperture_height_arcsec == pytest.approx(0.9)
    assert observation.aperture_radius_arcsec is None
    assert observation.seeing_fwhm_arcsec == pytest.approx(0.7)
    assert result.rejected[0].lens_id == "023817-054556"
    assert result.rejected[0].reason == "use_for_likelihood=false"


def test_velocity_measurements_v1_reader_parses_circular_aperture_rows(tmp_path: Path) -> None:
    """Circular measurement rows should carry radius and seeing into SigmaObservation."""

    csv_path = _write_csv(
        tmp_path / "velocity_measurements_v1.csv",
        VELOCITY_MEASUREMENTS_V1_STANDARD_COLUMNS,
        [
            {
                "schema_version": "velocity_measurements_v1",
                "lens_id": "slacs-a",
                "obs_tag": "",
                "sigma_kms": 250.0,
                "sigma_err_kms": 12.0,
                "sigma_error_kind": "total",
                "measurement_status": "success",
                "use_for_likelihood": "true",
                "source_system": "slacs-a",
                "source_file": "/tmp/ppxf.csv",
                "aperture_shape": "circular",
                "aperture_width_arcsec": "",
                "aperture_height_arcsec": "",
                "aperture_radius_arcsec": 1.5,
                "seeing_fwhm_arcsec": 1.2,
            }
        ],
    )

    observation = read_velocity_measurements_v1_csv(csv_path).accepted[0]

    assert observation.aperture_shape == "circular"
    assert observation.aperture_radius_arcsec == pytest.approx(1.5)
    assert observation.aperture_width_arcsec is None
    assert observation.aperture_height_arcsec is None
    assert observation.seeing_fwhm_arcsec == pytest.approx(1.2)


def test_velocity_measurements_v1_reader_requires_aperture_for_accepted_rows(tmp_path: Path) -> None:
    """Accepted likelihood rows must not rely on hidden dataset-level aperture defaults."""

    csv_path = _write_csv(
        tmp_path / "velocity_measurements_v1.csv",
        VELOCITY_MEASUREMENTS_V1_STANDARD_COLUMNS,
        [
            {
                "schema_version": "velocity_measurements_v1",
                "lens_id": "023817-054555",
                "obs_tag": "A",
                "sigma_kms": 226.4,
                "sigma_err_kms": 9.6,
                "sigma_error_kind": "total",
                "measurement_status": "success",
                "use_for_likelihood": "true",
                "source_system": "HSCJ023817-054555A",
                "source_file": "/tmp/ppxf.csv",
                "aperture_shape": "rectangular",
                "aperture_width_arcsec": 1.6,
                "aperture_height_arcsec": 0.9,
                "aperture_radius_arcsec": "",
                "seeing_fwhm_arcsec": "",
            }
        ],
    )

    with pytest.raises(ValueError, match="023817-054555.*seeing_fwhm_arcsec"):
        read_velocity_measurements_v1_csv(csv_path)


def test_velocity_measurements_v1_reader_requires_rectangular_dimensions(tmp_path: Path) -> None:
    """Rectangular accepted rows must define both slit width and slit height."""

    csv_path = _write_csv(
        tmp_path / "velocity_measurements_v1.csv",
        VELOCITY_MEASUREMENTS_V1_STANDARD_COLUMNS,
        [
            {
                "schema_version": "velocity_measurements_v1",
                "lens_id": "023817-054555",
                "obs_tag": "A",
                "sigma_kms": 226.4,
                "sigma_err_kms": 9.6,
                "sigma_error_kind": "total",
                "measurement_status": "success",
                "use_for_likelihood": "true",
                "source_system": "HSCJ023817-054555A",
                "source_file": "/tmp/ppxf.csv",
                "aperture_shape": "rectangular",
                "aperture_width_arcsec": 1.6,
                "aperture_height_arcsec": "",
                "aperture_radius_arcsec": "",
                "seeing_fwhm_arcsec": 0.9,
            }
        ],
    )

    with pytest.raises(ValueError, match="023817-054555.*aperture_height_arcsec"):
        read_velocity_measurements_v1_csv(csv_path)


def test_velocity_measurements_v1_reader_rejects_bad_schema_version(tmp_path: Path) -> None:
    """The upstream contract should fail fast when a different schema arrives."""

    csv_path = _write_csv(
        tmp_path / "velocity_measurements_v1.csv",
        VELOCITY_MEASUREMENTS_V1_STANDARD_COLUMNS,
        [
            {
                "schema_version": "velocity_measurements_v2",
                "lens_id": "023817-054555",
                "obs_tag": "A",
                "sigma_kms": 226.4,
                "sigma_err_kms": 9.6,
                "sigma_error_kind": "total",
                "measurement_status": "success",
                "use_for_likelihood": "true",
                "source_system": "HSCJ023817-054555A",
                "source_file": "/tmp/ppxf.csv",
                "aperture_shape": "rectangular",
                "aperture_width_arcsec": 1.6,
                "aperture_height_arcsec": 0.9,
                "aperture_radius_arcsec": "",
                "seeing_fwhm_arcsec": 0.7,
            }
        ],
    )

    with pytest.raises(ValueError, match="velocity_measurements_v1"):
        read_velocity_measurements_v1_csv(csv_path)


def test_velocity_measurements_v1_reader_requires_all_standard_columns(tmp_path: Path) -> None:
    """The v1 contract requires every documented standard column to be present."""

    csv_path = _write_csv(
        tmp_path / "velocity_measurements_v1.csv",
        [column for column in VELOCITY_MEASUREMENTS_V1_STANDARD_COLUMNS if column != "obs_tag"],
        [
            {
                "schema_version": "velocity_measurements_v1",
                "lens_id": "023817-054555",
                "sigma_kms": 226.4,
                "sigma_err_kms": 9.6,
                "sigma_error_kind": "total",
                "measurement_status": "success",
                "use_for_likelihood": "true",
                "source_system": "HSCJ023817-054555A",
                "source_file": "/tmp/ppxf.csv",
                "aperture_shape": "rectangular",
                "aperture_width_arcsec": 1.6,
                "aperture_height_arcsec": 0.9,
                "aperture_radius_arcsec": "",
                "seeing_fwhm_arcsec": 0.7,
            }
        ],
    )

    with pytest.raises(ValueError, match="missing required columns: obs_tag"):
        read_velocity_measurements_v1_csv(csv_path)


def test_ppxf_adapter_uses_statistical_error_by_default_and_rejects_failed_rows(tmp_path: Path) -> None:
    """The legacy pPXF adapter should still parse complete measurement rows."""

    csv_path = _write_csv(
        tmp_path / "ppxf_results_optimal.csv",
        [
            "system",
            "base_name",
            "obs_tag",
            "extraction_method",
            "z_lens",
            "z_source",
            "primary_status",
            "sigma_primary_kms",
            "sigma_stat_kms",
            "sigma_sys_window_kms",
            "sigma_total_kms",
            "warnings",
            "aperture_shape",
            "aperture_width_arcsec",
            "aperture_height_arcsec",
            "aperture_radius_arcsec",
            "seeing_fwhm_arcsec",
        ],
        [
            {
                "system": "HSCJ023817-054555A",
                "base_name": "023817-054555",
                "obs_tag": "A",
                "extraction_method": "abba_vis_positive",
                "z_lens": 0.599,
                "z_source": 1.763,
                "primary_status": "SUCCESS",
                "sigma_primary_kms": 226.4,
                "sigma_stat_kms": 9.6,
                "sigma_sys_window_kms": 2.0,
                "sigma_total_kms": 9.8,
                "warnings": "",
                "aperture_shape": "rectangular",
                "aperture_width_arcsec": 1.6,
                "aperture_height_arcsec": 0.9,
                "aperture_radius_arcsec": "",
                "seeing_fwhm_arcsec": 0.7,
            },
            {
                "system": "HSCJ023817-054555B",
                "base_name": "023817-054555",
                "obs_tag": "B",
                "extraction_method": "abba_vis_positive",
                "z_lens": 0.599,
                "z_source": 1.763,
                "primary_status": "FAILED",
                "sigma_primary_kms": 228.1,
                "sigma_stat_kms": 9.4,
                "sigma_sys_window_kms": 2.0,
                "sigma_total_kms": 9.7,
                "warnings": "low snr",
                "aperture_shape": "",
                "aperture_width_arcsec": "",
                "aperture_height_arcsec": "",
                "aperture_radius_arcsec": "",
                "seeing_fwhm_arcsec": "",
            },
        ],
    )

    result = load_ppxf_velocity_measurements(csv_path)

    assert len(result.accepted) == 1
    assert len(result.rejected) == 1
    observation = result.accepted[0]
    assert observation.lens_id == "023817-054555"
    assert observation.obs_tag == "A"
    assert observation.source_system == "HSCJ023817-054555A"
    assert observation.source_file == str(csv_path.resolve())
    assert observation.sigma_err_kms == pytest.approx(9.6)
    assert observation.aperture_shape == "rectangular"
    assert observation.seeing_fwhm_arcsec == pytest.approx(0.7)
    assert result.rejected[0].reason == "primary_status!=SUCCESS"


def test_ppxf_adapter_requires_aperture_for_success_rows(tmp_path: Path) -> None:
    """Legacy pPXF success rows must now carry the same aperture contract."""

    csv_path = _write_csv(
        tmp_path / "ppxf_results_optimal.csv",
        [
            "system",
            "base_name",
            "obs_tag",
            "primary_status",
            "sigma_primary_kms",
            "sigma_stat_kms",
            "sigma_total_kms",
        ],
        [
            {
                "system": "HSCJ023817-054555A",
                "base_name": "023817-054555",
                "obs_tag": "A",
                "primary_status": "SUCCESS",
                "sigma_primary_kms": 226.4,
                "sigma_stat_kms": 9.6,
                "sigma_total_kms": 11.1,
            }
        ],
    )

    with pytest.raises(ValueError, match="aperture_shape"):
        load_ppxf_velocity_measurements(csv_path)


def test_ppxf_adapter_can_choose_total_sigma_error(tmp_path: Path) -> None:
    """The adapter should be able to map total uncertainties when configured."""

    csv_path = _write_csv(
        tmp_path / "ppxf_results_optimal.csv",
        [
            "system",
            "base_name",
            "obs_tag",
            "primary_status",
            "sigma_primary_kms",
            "sigma_stat_kms",
            "sigma_total_kms",
            "aperture_shape",
            "aperture_width_arcsec",
            "aperture_height_arcsec",
            "aperture_radius_arcsec",
            "seeing_fwhm_arcsec",
        ],
        [
            {
                "system": "HSCJ023817-054555A",
                "base_name": "023817-054555",
                "obs_tag": "A",
                "primary_status": "SUCCESS",
                "sigma_primary_kms": 226.4,
                "sigma_stat_kms": 9.6,
                "sigma_total_kms": 11.1,
                "aperture_shape": "rectangular",
                "aperture_width_arcsec": 1.6,
                "aperture_height_arcsec": 0.9,
                "aperture_radius_arcsec": "",
                "seeing_fwhm_arcsec": 0.9,
            }
        ],
    )

    result = load_ppxf_velocity_measurements(
        csv_path,
        adapter_config=PpxfAdapterConfig(error_column="sigma_total_kms"),
    )

    assert len(result.accepted) == 1
    assert result.accepted[0].sigma_err_kms == pytest.approx(11.1)
