"""Tests for writing and validating direct canonical payloads."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from statistical_sl.data_preparation.direct_pipeline.records import CanonicalDatasetPayload
from statistical_sl.data_preparation.direct_pipeline.validator import validate_canonical_hdf5
from statistical_sl.data_preparation.direct_pipeline.writer import write_canonical_dataset_payload
from statistical_sl.core.canonical_schema import (
    CAPABILITY_LENS_OBSERVATIONS_V1,
    CAPABILITY_LENSING_CROSS_SECTION_THETA_GAMMA_V1,
    CAPABILITY_LENSING_MASS_GRIDS_V1,
    CAPABILITY_VELOCITY_DISPERSION_PER_LENS_S2_V1,
    TOP_LEVEL_BLOCKS,
)


def _payload(*, include_capabilities: bool = True, non_finite: bool = False) -> CanonicalDatasetPayload:
    """Create one compact valid payload, with optional validation defects."""

    mass_grid = np.asarray([[10.0, 10.2], [10.1, 10.3]], dtype=float)
    if non_finite:
        mass_grid[0, 0] = np.nan

    metadata = {
        "schema_version": "canonical_dataset_payload_v1",
        "unit_convention": "h_units_v1",
        "h_ref": 0.7,
        "profile_name": "devauc",
        "mass_definition_label": "m5_hinvkpc",
        "mass_radius_kpc": 5.0,
        "aperture_contract": "per_lens",
    }
    if include_capabilities:
        metadata["capabilities"] = (
            CAPABILITY_LENS_OBSERVATIONS_V1,
            CAPABILITY_LENSING_MASS_GRIDS_V1,
            CAPABILITY_LENSING_CROSS_SECTION_THETA_GAMMA_V1,
            CAPABILITY_VELOCITY_DISPERSION_PER_LENS_S2_V1,
        )

    return CanonicalDatasetPayload(
        metadata=metadata,
        lenses={
            "lens_id": np.asarray(["lens-a", "lens-b"], dtype=object),
            "z_d": np.asarray([0.5, 0.6], dtype=float),
            "z_s": np.asarray([1.5, 1.6], dtype=float),
            "num_sigma": np.asarray([1, 0], dtype=np.int64),
            "sigma_obs": np.asarray([[240.0, 0.0], [0.0, 0.0]], dtype=float),
            "sigma_err": np.asarray([[15.0, 1.0], [1.0, 1.0]], dtype=float),
            "observation_flavor": np.asarray(["slit", ""], dtype=object),
            "sigma_definition": np.asarray(["observed_aperture", ""], dtype=object),
            "aperture_shape": np.asarray(["rectangular", ""], dtype=object),
            "aperture_width_arcsec": np.asarray([1.6, 0.0], dtype=float),
            "aperture_height_arcsec": np.asarray([0.9, 0.0], dtype=float),
            "aperture_radius_arcsec": np.asarray([0.0, 0.0], dtype=float),
            "seeing_fwhm_arcsec": np.asarray([0.9, 0.0], dtype=float),
        },
        lensing_mass_grids={
            "lens_id": np.asarray(["lens-a", "lens-b"], dtype=object),
            "gamma_axis": np.asarray([1.2, 2.0], dtype=float),
            "log_enclosed_mass_grid": mass_grid,
            "dmass_dthetaein_grid": np.asarray([[0.9, 1.0], [1.1, 1.2]], dtype=float),
            "mass_definition_label": "m5_hinvkpc",
        },
        lensing_cross_section={
            "theta_e_axis": np.asarray([0.5, 1.0], dtype=float),
            "gamma_axis": np.asarray([1.2, 2.0], dtype=float),
            "cross_section_grid": np.asarray([[0.1, 0.2], [0.3, 0.4]], dtype=float),
            "source_mode": "cmass_power_law",
        },
        velocity_dispersion_grids={
            "lens_id": np.asarray(["lens-a", "lens-b"], dtype=object),
            "gamma_axis": np.asarray([1.2, 2.0], dtype=float),
            "s2_grid": np.asarray([[1.0e-5, 1.1e-5], [0.0, 0.0]], dtype=float),
            "has_s2": np.asarray([True, False], dtype=bool),
        },
        provenance={
            "catalog": {"source_path": "/tmp/summary_table_deV.txt"},
            "measurement": {"source_path": "/tmp/ppxf_results_optimal.csv"},
        },
    )


def test_write_canonical_dataset_payload_creates_complete_hdf5(tmp_path: Path) -> None:
    """A valid payload should write the agreed top-level canonical blocks."""

    output_path = write_canonical_dataset_payload(_payload(), tmp_path / "inference_dataset.hdf5")

    validate_canonical_hdf5(output_path)
    with h5py.File(output_path, "r") as handle:
        assert set(handle.keys()) == set(TOP_LEVEL_BLOCKS)
        assert "capabilities" in handle["metadata"]
        assert handle["metadata"].attrs["aperture_contract"] == "per_lens"
        assert handle["lenses"]["num_sigma"][:].tolist() == [1, 0]
        decoded_shapes = [
            value.decode("utf-8") if isinstance(value, bytes) else value
            for value in handle["lenses"]["aperture_shape"][:].tolist()
        ]
        assert decoded_shapes == ["rectangular", ""]
        assert "per_lens_s2" in handle["velocity_dispersion_grids"]
        assert handle["velocity_dispersion_grids"]["per_lens_s2"]["has_s2"][:].tolist() == [True, False]


def test_writer_failed_validation_leaves_no_partial_output(tmp_path: Path) -> None:
    """Invalid payloads should not leave a partially written target file."""

    output_path = tmp_path / "broken_inference_dataset.hdf5"

    with pytest.raises(ValueError, match="capabilities"):
        write_canonical_dataset_payload(_payload(include_capabilities=False), output_path)

    assert not output_path.exists()


def test_hdf5_validator_catches_missing_capability_block(tmp_path: Path) -> None:
    """Read-back validation should reject files without declared capabilities."""

    output_path = write_canonical_dataset_payload(_payload(), tmp_path / "inference_dataset.hdf5")
    with h5py.File(output_path, "r+") as handle:
        del handle["metadata"]["capabilities"]

    with pytest.raises(ValueError, match="capabilities"):
        validate_canonical_hdf5(output_path)


def test_hdf5_validator_catches_non_finite_arrays(tmp_path: Path) -> None:
    """Read-back validation should reject NaN or infinite numeric datasets."""

    output_path = write_canonical_dataset_payload(_payload(), tmp_path / "inference_dataset.hdf5")
    with h5py.File(output_path, "r+") as handle:
        handle["lensing_mass_grids"]["log_enclosed_mass_grid"][0, 0] = np.nan

    with pytest.raises(ValueError, match="non-finite"):
        validate_canonical_hdf5(output_path)


def test_hdf5_validator_catches_sigma_without_s2_grid(tmp_path: Path) -> None:
    """A lens with num_sigma > 0 must have an available per-lens s2 row."""

    output_path = write_canonical_dataset_payload(_payload(), tmp_path / "inference_dataset.hdf5")
    with h5py.File(output_path, "r+") as handle:
        del handle["velocity_dispersion_grids"]["per_lens_s2"]["s2_grid"]

    with pytest.raises(ValueError, match="s2_grid"):
        validate_canonical_hdf5(output_path)


def test_hdf5_validator_catches_sigma_without_aperture_metadata(tmp_path: Path) -> None:
    """A sigma-bearing lens must carry a complete per-lens aperture contract."""

    output_path = write_canonical_dataset_payload(_payload(), tmp_path / "inference_dataset.hdf5")
    with h5py.File(output_path, "r+") as handle:
        handle["lenses"]["seeing_fwhm_arcsec"][0] = 0.0

    with pytest.raises(ValueError, match="aperture"):
        validate_canonical_hdf5(output_path)


def test_hdf5_validator_catches_mismatched_lens_and_grid_dimensions(tmp_path: Path) -> None:
    """Mass-grid rows must stay aligned with the lens block."""

    output_path = write_canonical_dataset_payload(_payload(), tmp_path / "inference_dataset.hdf5")
    with h5py.File(output_path, "r+") as handle:
        del handle["lensing_mass_grids"]["log_enclosed_mass_grid"]
        handle["lensing_mass_grids"].create_dataset(
            "log_enclosed_mass_grid",
            data=np.asarray([[10.0, 10.2]], dtype=float),
        )

    with pytest.raises(ValueError, match="dimension"):
        validate_canonical_hdf5(output_path)
