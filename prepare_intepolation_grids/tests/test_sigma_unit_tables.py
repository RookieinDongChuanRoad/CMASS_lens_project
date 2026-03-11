"""Tests for the new sigma-unit HDF5 interpolation-table workflow.

These tests lock in the contract for the PPT-facing Jeans tables:
- the numerical grid stores `S_unit`, not the legacy `s2_grid`
- the builder writes the new explicit HDF5 schema
- table values on grid nodes agree with direct Jeans evaluation
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from interpolation_grids.config import (
    SIGMA_UNIT_DEVAUC_LOG_RE_KPC_AXIS,
    SIGMA_UNIT_GAMMA_AXIS,
    SIGMA_UNIT_SERSIC_LOG_RE_KPC_AXIS,
    SIGMA_UNIT_SERSIC_N_AXIS,
    SIGMA_UNIT_ZD_AXIS,
)
from interpolation_grids.io.sigma_tables import build_default_sigma_unit_hdf5_tables, build_sigma_unit_table
from interpolation_grids.physics.jeans import compute_sigma_unit


def test_parallel_sigma_unit_builder_matches_serial_on_small_grids() -> None:
    """Parallel workers must produce exactly the same values as serial code."""

    for profile_name in ("devauc", "sersic"):
        serial_table = build_sigma_unit_table(
            profile_name=profile_name,
            gamma_axis=np.array([1.2, 2.0, 2.8]),
            zd_axis=np.array([0.43, 0.82]),
            log_re_kpc_axis=np.array([0.50, 1.10]),
            n_axis=None if profile_name == "devauc" else np.array([2.5, 7.0, 10.5]),
            workers=1,
        )
        parallel_table = build_sigma_unit_table(
            profile_name=profile_name,
            gamma_axis=np.array([1.2, 2.0, 2.8]),
            zd_axis=np.array([0.43, 0.82]),
            log_re_kpc_axis=np.array([0.50, 1.10]),
            n_axis=None if profile_name == "devauc" else np.array([2.5, 7.0, 10.5]),
            workers=2,
        )

        np.testing.assert_allclose(parallel_table.values, serial_table.values, rtol=1e-12, atol=1e-12)


def test_build_sigma_unit_table_matches_direct_jeans_values_on_selected_grid_nodes() -> None:
    """Grid nodes should agree exactly with direct Jeans evaluation."""

    for profile_name in ("devauc", "sersic"):
        table = build_sigma_unit_table(
            profile_name=profile_name,
            gamma_axis=np.array([1.2, 2.0, 2.8]),
            zd_axis=np.array([0.43, 0.82]),
            log_re_kpc_axis=np.array([0.50, 1.10]),
            n_axis=None if profile_name == "devauc" else np.array([2.5, 7.0, 10.5]),
        )
        gamma_indices = tuple(range(len(table.gamma_axis)))
        zd_indices = tuple(range(len(table.zd_axis)))
        log_re_indices = tuple(range(len(table.log_re_kpc_axis)))

        for gamma_index in gamma_indices:
            for zd_index in zd_indices:
                for log_re_index in log_re_indices:
                    re_kpc = 10.0 ** float(table.log_re_kpc_axis[log_re_index])
                    if profile_name == "devauc":
                        expected = compute_sigma_unit(
                            profile_name=profile_name,
                            gamma=float(table.gamma_axis[gamma_index]),
                            zd=float(table.zd_axis[zd_index]),
                            re_kpc=re_kpc,
                        )
                        actual = float(table.values[gamma_index, zd_index, log_re_index])
                    else:
                        n_indices = tuple(range(len(table.n_axis)))
                        for n_index in n_indices:
                            expected = compute_sigma_unit(
                                profile_name=profile_name,
                                gamma=float(table.gamma_axis[gamma_index]),
                                zd=float(table.zd_axis[zd_index]),
                                re_kpc=re_kpc,
                                n_value=float(table.n_axis[n_index]),
                            )
                            actual = float(table.values[gamma_index, zd_index, log_re_index, n_index])
                            np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)
                        continue

                    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)


def test_build_default_sigma_unit_hdf5_tables_writes_new_schema_and_expected_filenames(tmp_path: Path) -> None:
    """Default writer should emit the new explicit HDF5 schema for both profiles."""

    output_paths = build_default_sigma_unit_hdf5_tables(
        output_directory=tmp_path,
        gamma_axis=np.array([1.2, 2.8]),
        zd_axis=np.array([0.43, 0.82]),
        devauc_log_re_kpc_axis=np.array([0.45, 1.20]),
        sersic_log_re_kpc_axis=np.array([0.50, 1.40]),
        sersic_n_axis=np.array([2.5, 10.5]),
    )

    assert output_paths["devauc"] == tmp_path / "jeans_deV_grid.h5"
    assert output_paths["sersic"] == tmp_path / "jeans_sers_grid.h5"

    assert np.array_equal(SIGMA_UNIT_GAMMA_AXIS, np.linspace(1.2, 2.8, 17))
    assert np.array_equal(SIGMA_UNIT_ZD_AXIS, np.linspace(0.43, 0.82, 21))
    assert np.array_equal(SIGMA_UNIT_DEVAUC_LOG_RE_KPC_AXIS, np.linspace(0.45, 1.20, 21))
    assert np.array_equal(SIGMA_UNIT_SERSIC_LOG_RE_KPC_AXIS, np.linspace(0.50, 1.40, 21))
    assert np.array_equal(SIGMA_UNIT_SERSIC_N_AXIS, np.linspace(2.5, 10.5, 21))

    with h5py.File(output_paths["devauc"], "r") as handle:
        assert set(handle.keys()) == {"profile_name", "gamma_axis", "zd_axis", "log_re_kpc_axis", "s_unit_grid"}
        assert handle["profile_name"][()].decode("utf-8") == "devauc"
        assert handle["s_unit_grid"].shape == (2, 2, 2)
        assert handle.attrs["schema_version"] == "sigma_unit_hdf5_v1"
        assert handle.attrs["quantity_name"] == "S_unit"
        assert handle.attrs["units"] == "km2 s-2 per 10**m5"

    with h5py.File(output_paths["sersic"], "r") as handle:
        assert set(handle.keys()) == {"profile_name", "gamma_axis", "zd_axis", "log_re_kpc_axis", "n_axis", "s_unit_grid"}
        assert handle["profile_name"][()].decode("utf-8") == "sersic"
        assert handle["s_unit_grid"].shape == (2, 2, 2, 2)
        assert np.all(np.diff(handle["n_axis"][:]) > 0.0)


def test_build_default_sigma_unit_hdf5_tables_can_limit_work_to_one_profile(tmp_path: Path) -> None:
    """Production builder should support targeted single-profile reruns."""

    output_paths = build_default_sigma_unit_hdf5_tables(
        output_directory=tmp_path,
        gamma_axis=np.array([1.2, 2.8]),
        zd_axis=np.array([0.43, 0.82]),
        devauc_log_re_kpc_axis=np.array([0.45, 1.20]),
        profiles=("devauc",),
        workers=1,
    )

    assert set(output_paths) == {"devauc"}
    assert output_paths["devauc"].exists()
    assert not (tmp_path / "jeans_sers_grid.h5").exists()
