"""Integration tests for direct source-to-canonical dataset builds."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

from statistical_sl.data_preparation import cli as data_preparation_cli
from statistical_sl.data_preparation.direct_pipeline import grid_builders, lens_preparer
from statistical_sl.data_preparation.direct_pipeline.runner import run_direct_canonical_build


def _patch_lightweight_physics(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace expensive physical kernels with deterministic tiny arrays.

    These tests exercise orchestration, source-policy decisions, provenance,
    and HDF5 serialization.  The detailed mass and Jeans kernels are already
    covered by lower-level tests, so running the real integrations here would
    slow the suite without improving the contract coverage.
    """

    monkeypatch.setattr(lens_preparer, "sigma_critical_surface_density", lambda _zd, _zs: 2.0e9)

    def fake_mass_grid(**kwargs: object) -> np.ndarray:
        gamma_axis = np.asarray(kwargs["gamma_grid"], dtype=float)
        return 10.0 + 0.1 * np.arange(gamma_axis.size, dtype=float)

    def fake_derivative_grid(**kwargs: object) -> np.ndarray:
        gamma_axis = np.asarray(kwargs["gamma_grid"], dtype=float)
        return 0.01 + 0.01 * np.arange(gamma_axis.size, dtype=float)

    def fake_sigma_unit_grid(**kwargs: object) -> np.ndarray:
        gamma_axis = np.asarray(kwargs["gamma_grid"], dtype=float)
        return 1.0e-5 + 1.0e-6 * np.arange(gamma_axis.size, dtype=float)

    monkeypatch.setattr(grid_builders, "compute_mass_grid", fake_mass_grid)
    monkeypatch.setattr(grid_builders, "compute_dmass_dthetaein_grid", fake_derivative_grid)
    monkeypatch.setattr(grid_builders, "compute_sigma_unit_grid", fake_sigma_unit_grid)


def _write_cmass_summary(path: Path) -> Path:
    """Write a CMASS table covering single, double, and missing sigma cases."""

    path.write_text(
        "# name zd zs rein_arcsec re_arcsec nser logmchab logmchab_err sigma sigma_err imag_Ser imag_deV reff_deV logmchab_deV\n"
        "lens-a 0.50 1.50 1.0 0.8 4.0 11.20 0.08 999.0 99.0 18.0 18.1 0.80 11.10\n"
        "lens-b 0.55 1.60 1.1 0.9 4.0 11.30 0.09 888.0 88.0 18.2 18.3 0.90 11.20\n"
        "lens-c 0.60 1.70 1.2 1.0 4.0 11.40 0.10 777.0 77.0 18.4 18.5 1.00 11.30\n",
        encoding="utf-8",
    )
    return path


def _write_velocity_measurements_csv(path: Path) -> Path:
    """Write external sigma rows with per-measurement aperture contracts."""

    path.write_text(
        "schema_version,lens_id,obs_tag,sigma_kms,sigma_err_kms,sigma_error_kind,"
        "measurement_status,use_for_likelihood,source_system,source_file,"
        "aperture_shape,aperture_width_arcsec,aperture_height_arcsec,aperture_radius_arcsec,seeing_fwhm_arcsec\n"
        f"velocity_measurements_v1,lens-a,,240.0,15.0,statistical,success,true,lens-a,{path},"
        "rectangular,1.6,0.9,,0.7\n"
        f"velocity_measurements_v1,lens-b,A,250.0,12.0,statistical,success,true,lens-bA,{path},"
        "circular,,,1.5,1.2\n"
        f"velocity_measurements_v1,lens-b,B,255.0,14.0,statistical,success,true,lens-bB,{path},"
        "circular,,,1.5,1.2\n",
        encoding="utf-8",
    )
    return path


def _write_cmass_cross_section(path: Path) -> Path:
    """Write the compressed CMASS ratio fixture consumed by the provider."""

    with h5py.File(path, "w") as handle:
        compressed = handle.create_group("compressed_grids")
        compressed.create_dataset("gamma_grids", data=np.asarray([1.2, 2.0], dtype=float))
        compressed.create_dataset("cs_over_theta_ein_grid", data=np.asarray([0.2, 0.3], dtype=float))
    return path


def _write_sonnenfeld_cross_section(path: Path) -> Path:
    """Write the finite-fibre Sonnenfeld cross-section fixture."""

    with h5py.File(path, "w") as handle:
        handle.create_dataset("tein_grid", data=np.asarray([0.0, 1.0], dtype=float))
        handle.create_dataset("gamma_grid", data=np.asarray([1.2, 2.0], dtype=float))
        handle.create_dataset("mufibre3_cs_grid", data=np.asarray([[0.0, 0.0], [0.5, 1.5]], dtype=float))
    return path


def _write_slacs_table(path: Path) -> Path:
    """Write a compact SLACS table whose catalog sigma columns are explicitly trusted."""

    path.write_text(
        "# name RA dec zd zs Reff_arcsec Reff_kpc theta_Ein Rein_kpc lMstar_Chab lMstar_err veldisp(km/s) veldisp_err(km/s)\n"
        "slacs-a 10.0 -1.0 0.20 0.80 1.5 4.5 1.0 3.0 11.1 0.05 250.0 12.0\n"
        "slacs-b 11.0 -1.5 0.25 0.90 1.6 4.8 1.1 3.3 11.2 0.06 260.0 13.0\n",
        encoding="utf-8",
    )
    return path


def test_direct_pipeline_builds_cmass_slit_dataset_with_external_sigma(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """CMASS catalog sigma stays ignored while external pPXF rows drive num_sigma."""

    _patch_lightweight_physics(monkeypatch)
    catalog_path = _write_cmass_summary(tmp_path / "summary_table_deV.txt")
    measurement_path = _write_velocity_measurements_csv(tmp_path / "velocity_measurements_v1.csv")
    cross_section_path = _write_cmass_cross_section(tmp_path / "cs_grid_power.h5")
    config_path = tmp_path / "cmass_direct.yaml"
    output_path = tmp_path / "cmass_inference_dataset.hdf5"
    audit_path = tmp_path / "cmass_inference_dataset.audit.json"
    config_path.write_text(
        f"""
schema_version: statistical_sl_direct_data_preparation_v1
output:
  canonical_hdf5: {output_path}
  audit_json: {audit_path}
catalog:
  type: cmass_summary_table
  path: {catalog_path}
  profile_name: devauc
velocity_measurements:
  type: velocity_measurements_v1
  path: {measurement_path}
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
  source_hdf5: {cross_section_path}
""",
        encoding="utf-8",
    )

    result = run_direct_canonical_build(config_path)

    assert result.canonical_hdf5 == output_path.resolve()
    assert result.audit_json == audit_path.resolve()
    with h5py.File(result.canonical_hdf5, "r") as handle:
        assert handle["lenses"]["num_sigma"][:].tolist() == [1, 2, 0]
        np.testing.assert_allclose(handle["lenses"]["sigma_obs"][:, 0], np.asarray([240.0, 250.0, 0.0]))
        np.testing.assert_allclose(handle["lenses"]["sigma_obs"][:, 1], np.asarray([0.0, 255.0, 0.0]))
        assert handle["velocity_dispersion_grids"]["per_lens_s2"]["has_s2"][:].tolist() == [True, True, False]
        assert handle["lensing_cross_section"].attrs["source_mode"] == "cmass_power_law"
        aperture_shapes = [
            value.decode("utf-8") if isinstance(value, bytes) else value
            for value in handle["lenses"]["aperture_shape"][:].tolist()
        ]
        assert aperture_shapes == ["rectangular", "circular", "rectangular"]
        np.testing.assert_allclose(handle["lenses"]["seeing_fwhm_arcsec"][:], np.asarray([0.7, 1.2, 0.9]))

    audit = json.loads(result.audit_json.read_text(encoding="utf-8"))
    assert audit["catalog"]["ignored_columns"]["sigma"] == "untrusted_catalog_value"
    assert audit["measurement"]["source_path"] == str(measurement_path.resolve())
    assert audit["resolver"]["missing_lens_ids"] == ["lens-c"]


def test_direct_pipeline_builds_slacs_dataset_from_trusted_catalog_columns(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SLACS compatibility mode may use catalog sigma only under explicit trust."""

    _patch_lightweight_physics(monkeypatch)
    catalog_path = _write_slacs_table(tmp_path / "SLACS_table.cat")
    cross_section_path = _write_sonnenfeld_cross_section(tmp_path / "fibre_crosssect_grid.hdf5")
    config_path = tmp_path / "slacs_direct.yaml"
    output_path = tmp_path / "slacs_inference_dataset.hdf5"
    audit_path = tmp_path / "slacs_inference_dataset.audit.json"
    config_path.write_text(
        f"""
schema_version: statistical_sl_direct_data_preparation_v1
output:
  canonical_hdf5: {output_path}
  audit_json: {audit_path}
catalog:
  type: slacs_table
  path: {catalog_path}
  profile_name: devauc
velocity_measurements:
  type: catalog_columns
  missing_policy: fail
  trust_catalog_sigma: true
units:
  unit_convention: legacy_fixed_kpc
  h_ref: 0.7
  mass_definition_label: m5
  mass_radius_kpc: 5.0
aperture:
  observation_flavor: slacs_fibre
  sigma_definition: observed_aperture
  shape: circular
  radius_arcsec: 1.5
  seeing_fwhm_arcsec: 1.5
grids:
  gamma_axis: [1.2, 2.0]
  theta_e_axis: [0.0, 1.0]
cross_section:
  type: sonnenfeld_fibre
  source_hdf5: {cross_section_path}
""",
        encoding="utf-8",
    )

    result = run_direct_canonical_build(config_path)

    with h5py.File(result.canonical_hdf5, "r") as handle:
        assert handle["lenses"]["num_sigma"][:].tolist() == [1, 1]
        np.testing.assert_allclose(handle["lenses"]["sigma_obs"][:, 0], np.asarray([250.0, 260.0]))
        assert handle["velocity_dispersion_grids"]["per_lens_s2"]["has_s2"][:].tolist() == [True, True]
        assert handle["lensing_cross_section"].attrs["source_mode"] == "sonnenfeld_fibre"
        observation_flavor = handle["lenses"]["observation_flavor"][0]
        if isinstance(observation_flavor, bytes):
            observation_flavor = observation_flavor.decode("utf-8")
        assert observation_flavor == "slacs_fibre"

    audit = json.loads(result.audit_json.read_text(encoding="utf-8"))
    assert audit["measurement"]["source_path"] is None
    assert audit["measurement"]["measurement_mode"] == "catalog_columns"
    assert audit["measurement"]["extra"]["catalog_source_path"] == str(catalog_path.resolve())


def test_data_preparation_cli_consumes_workspace_direct_config_shape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The public CLI should consume the checked-in workspace direct config shape."""

    _patch_lightweight_physics(monkeypatch)
    catalog_path = _write_cmass_summary(tmp_path / "summary_table_deV.txt")
    measurement_path = _write_velocity_measurements_csv(tmp_path / "velocity_measurements_v1.csv")
    cross_section_path = _write_cmass_cross_section(tmp_path / "cs_grid_power.h5")
    output_path = tmp_path / "workspace_shape_inference_dataset.hdf5"
    audit_path = tmp_path / "workspace_shape_inference_dataset.audit.json"

    repository_root = Path(__file__).resolve().parents[2]
    workspace_config_path = (
        repository_root / "workspace" / "configs" / "data_preparation" / "cmass" / "devauc_direct_hunits.yaml"
    )
    config_text = workspace_config_path.read_text(encoding="utf-8")
    config_text = config_text.replace(
        "../../../data/canonical/inference_dataset_cmass_devauc_direct_hunits.hdf5",
        str(output_path),
    )
    config_text = config_text.replace(
        "../../../outputs/data_preparation/cmass/devauc_direct_hunits.audit.json",
        str(audit_path),
    )
    config_text = config_text.replace("../../../data/raw/summary_table_deV.txt", str(catalog_path))
    config_text = config_text.replace("../../../data/external/cmass_velocity_measurements_v1.csv", str(measurement_path))
    config_text = config_text.replace("../../../data/external/cs_grid_power.h5", str(cross_section_path))

    config_path = tmp_path / "workspace_shape_cmass_direct.yaml"
    config_path.write_text(config_text, encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        ["statistical-sl", "--build-canonical-direct", "--config", str(config_path)],
    )

    assert data_preparation_cli.main() == 0
    assert output_path.exists()
    assert audit_path.exists()
