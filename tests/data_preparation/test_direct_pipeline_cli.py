"""Tests for the direct-pipeline YAML parser and CLI wiring."""

from __future__ import annotations

from pathlib import Path

import h5py
import pytest

from statistical_sl.data_preparation.cli import build_parser
from statistical_sl.data_preparation.direct_pipeline.config import load_direct_pipeline_config


def _write_common_inputs(tmp_path: Path) -> dict[str, Path]:
    """Create tiny source files that satisfy the config parser's path checks."""

    catalog_path = tmp_path / "summary_table_deV.txt"
    catalog_path.write_text(
        "# name zd zs rein_arcsec re_arcsec nser logmchab logmchab_err sigma sigma_err imag_Ser imag_deV reff_deV logmchab_deV\n"
        "lens-a 0.5 1.5 1.0 0.8 4.0 11.2 0.08 240 15 18.0 18.1 0.8 11.1\n",
        encoding="utf-8",
    )

    measurement_path = tmp_path / "ppxf_results_optimal.csv"
    measurement_path.write_text(
        "system,base_name,obs_tag,primary_status,sigma_primary_kms,sigma_stat_kms,sigma_total_kms\n"
        "lens-a,lens-a,A,SUCCESS,240.0,15.0,16.0\n",
        encoding="utf-8",
    )

    cross_section_path = tmp_path / "cs_grid_power.h5"
    with h5py.File(cross_section_path, "w") as handle:
        compressed = handle.create_group("compressed_grids")
        compressed.create_dataset("gamma_grids", data=[1.2, 2.0])
        compressed.create_dataset("cs_over_theta_ein_grid", data=[0.2, 0.3])

    return {
        "catalog": catalog_path,
        "measurement": measurement_path,
        "cross_section": cross_section_path,
    }


def _write_valid_config(tmp_path: Path, *, measurement_type: str = "ppxf_results_adapter") -> Path:
    """Write one direct-pipeline YAML config used by parser tests."""

    paths = _write_common_inputs(tmp_path)
    config_path = tmp_path / "direct_pipeline.yaml"
    config_path.write_text(
        f"""
schema_version: statistical_sl_direct_data_preparation_v1
output:
  canonical_hdf5: {tmp_path / 'inference_dataset.hdf5'}
  audit_json: {tmp_path / 'inference_dataset.audit.json'}
catalog:
  type: cmass_summary_table
  path: {paths['catalog']}
  profile_name: devauc
velocity_measurements:
  type: {measurement_type}
  path: {paths['measurement']}
  error_column: sigma_stat_kms
  missing_policy: num_sigma_zero
  max_observations_per_lens: 2
units:
  unit_convention: h_units_v1
  h_ref: 0.7
  mass_definition_label: m5_hinvkpc
  mass_radius_kpc: 5.0
aperture:
  observation_flavor: slit
  sigma_definition: observed_aperture
  shape: rectangular
  width_arcsec: 1.6
  height_arcsec: 0.9
  seeing_fwhm_arcsec: 0.9
grids:
  gamma_axis: [1.2, 2.0]
  theta_e_axis: [0.5, 1.0]
cross_section:
  type: cmass_power_law
  source_hdf5: {paths['cross_section']}
""",
        encoding="utf-8",
    )
    return config_path


def test_load_direct_pipeline_config_parses_expected_sections(tmp_path: Path) -> None:
    """The YAML parser should return a structured direct-pipeline config."""

    config = load_direct_pipeline_config(_write_valid_config(tmp_path))

    assert config.catalog.type == "cmass_summary_table"
    assert config.velocity_measurements.error_column == "sigma_stat_kms"
    assert config.unit_policy().unit_convention == "h_units_v1"
    assert config.unit_policy().h_ref == pytest.approx(0.7)
    assert config.aperture_policy_ref().aperture_policy.shape == "rectangular"
    assert config.grids.gamma_axis.tolist() == [1.2, 2.0]


def test_load_direct_pipeline_config_accepts_axis_range_specs(tmp_path: Path) -> None:
    """Production configs should be able to describe axes as min/max/points."""

    config_path = _write_valid_config(tmp_path)
    content = config_path.read_text(encoding="utf-8").replace(
        "  gamma_axis: [1.2, 2.0]\n  theta_e_axis: [0.5, 1.0]\n",
        "  gamma_axis:\n    min: 1.2\n    max: 2.8\n    points: 3\n  theta_e_axis:\n    min: 0.0\n    max: 1.0\n    points: 2\n",
    )
    config_path.write_text(content, encoding="utf-8")

    config = load_direct_pipeline_config(config_path)

    assert config.grids.gamma_axis.tolist() == [1.2, 2.0, 2.8]
    assert config.grids.theta_e_axis.tolist() == [0.0, 1.0]


def test_load_direct_pipeline_config_rejects_cmass_catalog_columns_without_explicit_trust(tmp_path: Path) -> None:
    """CMASS summary tables should not trust catalog sigma by default."""

    config_path = _write_valid_config(tmp_path, measurement_type="catalog_columns")

    with pytest.raises(ValueError, match="trust_catalog_sigma"):
        load_direct_pipeline_config(config_path)


def test_load_direct_pipeline_config_rejects_missing_external_sigma_source(tmp_path: Path) -> None:
    """External measurement mode should fail when the source CSV is absent."""

    config_path = _write_valid_config(tmp_path)
    missing_measurement = tmp_path / "missing_measurements.csv"
    content = config_path.read_text(encoding="utf-8").replace(
        str(tmp_path / "ppxf_results_optimal.csv"),
        str(missing_measurement),
    )
    config_path.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError, match="does not exist"):
        load_direct_pipeline_config(config_path)


def test_load_direct_pipeline_config_rejects_missing_aperture_section(tmp_path: Path) -> None:
    """The direct pipeline should require explicit aperture metadata by default."""

    config_path = _write_valid_config(tmp_path)
    content = config_path.read_text(encoding="utf-8").replace(
        "\naperture:\n  observation_flavor: slit\n  sigma_definition: observed_aperture\n  shape: rectangular\n  width_arcsec: 1.6\n  height_arcsec: 0.9\n  seeing_fwhm_arcsec: 0.9\n",
        "\n",
    )
    config_path.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError, match="aperture"):
        load_direct_pipeline_config(config_path)


def test_cli_exposes_new_direct_pipeline_flags() -> None:
    """The existing top-level parser should surface the new direct-build mode."""

    parser = build_parser()
    args = parser.parse_args(["--build-canonical-direct", "--config", "/tmp/direct.yaml"])

    assert args.build_canonical_direct is True
    assert args.config == Path("/tmp/direct.yaml")
    assert args.build_power_law_cross_section_hdf5 is False
