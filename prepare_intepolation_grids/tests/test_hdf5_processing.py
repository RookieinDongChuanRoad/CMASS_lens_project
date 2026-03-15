"""Tests for HDF5 batch processing and safe-write behavior."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from interpolation_grids.config import DERIVATIVE_DATASET_NAME, GAMMA_DATASET_NAME
from interpolation_grids.io.hdf5 import process_hdf5_file


def _create_sample_input(path: Path) -> None:
    """Create a compact HDF5 fixture that exercises update/skip behavior."""

    with h5py.File(path, "w") as handle:
        with_s2 = handle.create_group("with_s2")
        with_s2.attrs["zd"] = 0.599
        with_s2.attrs["zs"] = 1.763
        with_s2.attrs["sigma_crit"] = 2_223_801_018.8799353
        with_s2.attrs["rein_arcsec"] = 0.929
        with_s2.attrs["r_ein_kpc"] = 6.4
        with_s2.attrs["re_arcsec"] = 0.706
        with_s2.attrs["nser"] = 4.589
        with_s2.attrs["aperture_width"] = 0.8
        with_s2.create_dataset(GAMMA_DATASET_NAME, data=np.linspace(1.2, 2.8, 17))
        with_s2.create_dataset("m5_grid", data=np.zeros(17))
        with_s2.create_dataset(DERIVATIVE_DATASET_NAME, data=np.zeros(17))
        with_s2.create_dataset("s2_grid", data=np.zeros(17))

        no_s2 = handle.create_group("no_s2")
        no_s2.attrs["zd"] = 0.599
        no_s2.attrs["zs"] = 1.763
        no_s2.attrs["sigma_crit"] = 2_223_801_018.8799353
        no_s2.attrs["rein_arcsec"] = 0.929
        no_s2.attrs["r_ein_kpc"] = 6.4
        no_s2.attrs["re_arcsec"] = 0.706
        no_s2.attrs["nser"] = 4.589
        no_s2.create_dataset(GAMMA_DATASET_NAME, data=np.linspace(1.2, 2.8, 17))
        no_s2.create_dataset("m5_grid", data=np.zeros(17))
        no_s2.create_dataset(DERIVATIVE_DATASET_NAME, data=np.zeros(17))


def test_process_hdf5_file_writes_updated_grids_without_mutating_input(tmp_path: Path) -> None:
    """Default processing should write a new file and leave the source untouched."""

    input_path = tmp_path / "input.hdf5"
    output_path = tmp_path / "output.hdf5"
    _create_sample_input(input_path)

    original_bytes = input_path.read_bytes()
    summary = process_hdf5_file(
        input_path=input_path,
        output_path=output_path,
        overwrite_in_place=False,
    )

    assert input_path.read_bytes() == original_bytes
    assert output_path.exists()
    assert summary.total_groups == 2
    assert summary.updated_m5 == 2
    assert summary.updated_dm5 == 2
    assert summary.updated_s2 == 1
    assert not summary.failures

    with h5py.File(output_path, "r") as handle:
        assert "s2_grid" in handle["with_s2"]
        assert "s2_grid" not in handle["no_s2"]
        assert np.any(handle["with_s2"]["m5_grid"][:] != 0.0)
        assert np.any(handle["with_s2"][DERIVATIVE_DATASET_NAME][:] != 0.0)
        assert "mass_definitions" in handle["with_s2"]
        assert set(handle["with_s2"]["mass_definitions"].keys()) == {"m5", "m10"}
        assert np.any(handle["with_s2"]["mass_definitions"]["m5"]["mass_grid"][:] != 0.0)
        assert np.any(handle["with_s2"]["mass_definitions"]["m10"]["mass_grid"][:] != 0.0)
        assert np.any(handle["with_s2"]["mass_definitions"]["m5"]["dmass_dthetaein_grid"][:] != 0.0)
        assert np.any(handle["with_s2"]["mass_definitions"]["m10"]["dmass_dthetaein_grid"][:] != 0.0)
        assert "s2_grid" in handle["with_s2"]["mass_definitions"]["m5"]
        assert "s2_grid" in handle["with_s2"]["mass_definitions"]["m10"]


def test_process_hdf5_file_can_limit_work_to_selected_groups(tmp_path: Path) -> None:
    """Group filtering should support cheap single-galaxy debugging runs."""

    input_path = tmp_path / "input.hdf5"
    output_path = tmp_path / "output.hdf5"
    _create_sample_input(input_path)

    summary = process_hdf5_file(
        input_path=input_path,
        output_path=output_path,
        overwrite_in_place=False,
        group_names=("no_s2",),
    )

    assert summary.total_groups == 1
    assert summary.updated_m5 == 1
    assert summary.updated_dm5 == 1
    assert summary.updated_s2 == 0

    with h5py.File(output_path, "r") as handle:
        assert np.all(handle["with_s2"]["m5_grid"][:] == 0.0)
        assert np.any(handle["no_s2"]["m5_grid"][:] != 0.0)
        assert "mass_definitions" not in handle["with_s2"]
        assert "mass_definitions" in handle["no_s2"]
