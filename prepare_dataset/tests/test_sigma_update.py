"""Regression tests for CSV-driven sigma updates in project HDF5 files.

These tests focus on the contract required by the user request:

- only groups with ``num_sigma != 0`` are eligible for updates
- ``base_name`` matches the HDF5 group name
- two-observation systems must be written in ``A`` then ``B`` order
- preview mode must not mutate the input file
- in-place mode must preserve datasets while replacing only sigma attributes
"""

from __future__ import annotations

import csv
from pathlib import Path

import h5py
import numpy as np
import pytest

from prepare_dataset.io.sigma_updates import (
    SigmaUpdateValidationError,
    plan_sigma_updates_for_files,
    update_sigma_attributes_in_hdf5,
)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Create a compact CSV fixture with only the columns used by the updater."""

    fieldnames = [
        "base_name",
        "obs_tag",
        "sigma_primary_kms",
        "sigma_stat_kms",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _create_hdf5_fixture(path: Path) -> None:
    """Create a minimal HDF5 file that mirrors the production schema surface."""

    with h5py.File(path, "w") as handle:
        skipped = handle.create_group("skip-me")
        skipped.attrs["num_sigma"] = np.int64(0)
        skipped.create_dataset("gamma_grid", data=np.linspace(1.2, 2.8, 17))
        skipped.create_dataset("s2_grid", data=np.arange(17, dtype=float))

        single = handle.create_group("single-galaxy")
        single.attrs["num_sigma"] = np.int64(1)
        single.attrs["sigma"] = np.asarray([219], dtype=np.int64)
        single.attrs["sigma_err"] = np.asarray([10], dtype=np.int64)
        single.create_dataset("gamma_grid", data=np.linspace(1.2, 2.8, 17))
        single.create_dataset("s2_grid", data=np.arange(17, dtype=float))

        double = handle.create_group("double-galaxy")
        double.attrs["num_sigma"] = np.int64(2)
        double.attrs["sigma"] = np.asarray([264, 267], dtype=np.int64)
        double.attrs["sigma_err"] = np.asarray([7, 9], dtype=np.int64)
        double.create_dataset("gamma_grid", data=np.linspace(1.2, 2.8, 17))
        double.create_dataset("s2_grid", data=np.arange(17, dtype=float))


def test_plan_sigma_updates_orders_double_observations_and_rounds_values(tmp_path: Path) -> None:
    """Planning should derive the exact integer arrays that will be written."""

    csv_path = tmp_path / "sigma.csv"
    hdf5_path = tmp_path / "input.hdf5"
    _write_csv(
        csv_path,
        [
            {
                "base_name": "single-galaxy",
                "obs_tag": "",
                "sigma_primary_kms": 225.869123,
                "sigma_stat_kms": 9.587872,
            },
            {
                "base_name": "double-galaxy",
                "obs_tag": "B",
                "sigma_primary_kms": 269.682745,
                "sigma_stat_kms": 7.398339,
            },
            {
                "base_name": "double-galaxy",
                "obs_tag": "A",
                "sigma_primary_kms": 271.480496,
                "sigma_stat_kms": 7.103578,
            },
        ],
    )
    _create_hdf5_fixture(hdf5_path)

    plans = plan_sigma_updates_for_files(csv_path=csv_path, hdf5_paths=[hdf5_path])

    assert len(plans) == 1
    preview_rows = {item.group_name: item for item in plans[0].group_updates}

    assert preview_rows["single-galaxy"].new_sigma.tolist() == [226]
    assert preview_rows["single-galaxy"].new_sigma_err.tolist() == [10]
    assert preview_rows["single-galaxy"].matched_obs_tags == []
    assert preview_rows["double-galaxy"].new_sigma.tolist() == [271, 270]
    assert preview_rows["double-galaxy"].new_sigma_err.tolist() == [7, 7]
    assert preview_rows["double-galaxy"].matched_obs_tags == ["A", "B"]


def test_update_sigma_attributes_preview_mode_leaves_source_file_unchanged(tmp_path: Path) -> None:
    """Without in-place overwrite, the updater should only produce a preview plan."""

    csv_path = tmp_path / "sigma.csv"
    hdf5_path = tmp_path / "input.hdf5"
    _write_csv(
        csv_path,
        [
            {
                "base_name": "single-galaxy",
                "obs_tag": "",
                "sigma_primary_kms": 225.869123,
                "sigma_stat_kms": 9.587872,
            },
            {
                "base_name": "double-galaxy",
                "obs_tag": "A",
                "sigma_primary_kms": 271.480496,
                "sigma_stat_kms": 7.103578,
            },
            {
                "base_name": "double-galaxy",
                "obs_tag": "B",
                "sigma_primary_kms": 269.682745,
                "sigma_stat_kms": 7.398339,
            },
        ],
    )
    _create_hdf5_fixture(hdf5_path)

    original_bytes = hdf5_path.read_bytes()
    result = update_sigma_attributes_in_hdf5(
        csv_path=csv_path,
        hdf5_paths=[hdf5_path],
        overwrite_in_place=False,
        create_backup=False,
    )

    assert hdf5_path.read_bytes() == original_bytes
    assert result[0].backup_path is None
    assert result[0].was_written is False


def test_update_sigma_attributes_in_place_creates_backup_and_preserves_datasets(tmp_path: Path) -> None:
    """In-place writes should update only sigma attributes and leave datasets intact."""

    csv_path = tmp_path / "sigma.csv"
    hdf5_path = tmp_path / "input.hdf5"
    _write_csv(
        csv_path,
        [
            {
                "base_name": "single-galaxy",
                "obs_tag": "",
                "sigma_primary_kms": 225.869123,
                "sigma_stat_kms": 9.587872,
            },
            {
                "base_name": "double-galaxy",
                "obs_tag": "A",
                "sigma_primary_kms": 271.480496,
                "sigma_stat_kms": 7.103578,
            },
            {
                "base_name": "double-galaxy",
                "obs_tag": "B",
                "sigma_primary_kms": 269.682745,
                "sigma_stat_kms": 7.398339,
            },
        ],
    )
    _create_hdf5_fixture(hdf5_path)

    with h5py.File(hdf5_path, "r") as handle:
        original_dataset = handle["double-galaxy"]["s2_grid"][:]

    result = update_sigma_attributes_in_hdf5(
        csv_path=csv_path,
        hdf5_paths=[hdf5_path],
        overwrite_in_place=True,
        create_backup=True,
    )

    assert result[0].backup_path is not None
    assert result[0].backup_path.exists()
    assert result[0].was_written is True

    with h5py.File(hdf5_path, "r") as handle:
        assert handle["single-galaxy"].attrs["sigma"].tolist() == [226]
        assert handle["single-galaxy"].attrs["sigma_err"].tolist() == [10]
        assert handle["double-galaxy"].attrs["sigma"].tolist() == [271, 270]
        assert handle["double-galaxy"].attrs["sigma_err"].tolist() == [7, 7]
        assert np.array_equal(handle["double-galaxy"]["s2_grid"][:], original_dataset)
        assert handle["skip-me"].attrs["num_sigma"] == 0


def test_plan_sigma_updates_rejects_missing_double_observation_tags(tmp_path: Path) -> None:
    """Two-observation systems must provide exactly one ``A`` row and one ``B`` row."""

    csv_path = tmp_path / "sigma.csv"
    hdf5_path = tmp_path / "input.hdf5"
    _write_csv(
        csv_path,
        [
            {
                "base_name": "single-galaxy",
                "obs_tag": "",
                "sigma_primary_kms": 225.869123,
                "sigma_stat_kms": 9.587872,
            },
            {
                "base_name": "double-galaxy",
                "obs_tag": "A",
                "sigma_primary_kms": 271.480496,
                "sigma_stat_kms": 7.103578,
            },
            {
                "base_name": "double-galaxy",
                "obs_tag": "A",
                "sigma_primary_kms": 269.682745,
                "sigma_stat_kms": 7.398339,
            },
        ],
    )
    _create_hdf5_fixture(hdf5_path)

    with pytest.raises(SigmaUpdateValidationError, match="double-galaxy"):
        plan_sigma_updates_for_files(csv_path=csv_path, hdf5_paths=[hdf5_path])


def test_plan_sigma_updates_accepts_real_csv_shape_and_uses_stat_error_column(tmp_path: Path) -> None:
    """Extra PPXF columns should be ignored while `sigma_stat_kms` remains authoritative.

    The real CSV exported by the spectroscopy workflow carries many more columns
    than the updater needs, including `sigma_total_kms`. This regression test
    locks the current contract: planning must still use `sigma_primary_kms` for
    the value and `sigma_stat_kms` for the uncertainty.
    """

    csv_path = tmp_path / "ppxf_results_optimal.csv"
    hdf5_path = tmp_path / "input.hdf5"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "system",
                "base_name",
                "obs_tag",
                "sigma_primary_kms",
                "sigma_stat_kms",
                "sigma_total_kms",
                "warnings",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "system": "single-galaxy",
                "base_name": "single-galaxy",
                "obs_tag": "",
                "sigma_primary_kms": 225.869123,
                "sigma_stat_kms": 9.587872,
                "sigma_total_kms": 31.25,
                "warnings": "",
            }
        )
        writer.writerow(
            {
                "system": "double-galaxy-A",
                "base_name": "double-galaxy",
                "obs_tag": "A",
                "sigma_primary_kms": 271.480496,
                "sigma_stat_kms": 7.103578,
                "sigma_total_kms": 71.0,
                "warnings": "",
            }
        )
        writer.writerow(
            {
                "system": "double-galaxy-B",
                "base_name": "double-galaxy",
                "obs_tag": "B",
                "sigma_primary_kms": 269.682745,
                "sigma_stat_kms": 7.398339,
                "sigma_total_kms": 55.0,
                "warnings": "",
            }
        )
    _create_hdf5_fixture(hdf5_path)

    plans = plan_sigma_updates_for_files(csv_path=csv_path, hdf5_paths=[hdf5_path])
    preview_rows = {item.group_name: item for item in plans[0].group_updates}

    assert preview_rows["single-galaxy"].new_sigma.tolist() == [226]
    assert preview_rows["single-galaxy"].new_sigma_err.tolist() == [10]
    assert preview_rows["double-galaxy"].new_sigma.tolist() == [271, 270]
    assert preview_rows["double-galaxy"].new_sigma_err.tolist() == [7, 7]
