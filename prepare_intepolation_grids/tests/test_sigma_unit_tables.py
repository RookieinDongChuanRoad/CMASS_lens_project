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
    BOSS_CIRCULAR_APERTURE_POLICY,
    SIGMA_UNIT_DEVAUC_LOG_RE_KPC_AXIS,
    SIGMA_UNIT_GAMMA_AXIS,
    SIGMA_UNIT_SERSIC_LOG_RE_KPC_AXIS,
    SIGMA_UNIT_SERSIC_N_AXIS,
    SIGMA_UNIT_ZD_AXIS,
)
from interpolation_grids.cli import build_parser
from interpolation_grids.io.sigma_tables import (
    build_default_sigma_unit_hdf5_tables,
    build_sigma_unit_table,
    repack_legacy_sigma_unit_hdf5_tables_into_bundles,
    write_sigma_unit_table_hdf5,
)
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


def test_build_sigma_unit_table_records_boss_aperture_metadata_and_changes_values() -> None:
    """BOSS tables must carry circular-aperture metadata and differ from slit tables."""

    slit_table = build_sigma_unit_table(
        profile_name="devauc",
        gamma_axis=np.array([1.2, 2.0, 2.8]),
        zd_axis=np.array([0.43, 0.82]),
        log_re_kpc_axis=np.array([0.50, 1.10]),
        workers=1,
    )
    boss_table = build_sigma_unit_table(
        profile_name="devauc",
        gamma_axis=np.array([1.2, 2.0, 2.8]),
        zd_axis=np.array([0.43, 0.82]),
        log_re_kpc_axis=np.array([0.50, 1.10]),
        workers=1,
        observation_flavor="boss",
        aperture_policy=BOSS_CIRCULAR_APERTURE_POLICY,
    )

    assert boss_table.observation_flavor == "boss"
    assert boss_table.aperture_shape == "circular"
    assert boss_table.aperture_radius_arcsec == 1.0
    assert boss_table.seeing_fwhm_arcsec == 0.9
    assert not np.allclose(boss_table.values, slit_table.values)


def test_build_default_sigma_unit_hdf5_tables_writes_bundle_schema_and_expected_filenames(tmp_path: Path) -> None:
    """Default writer should emit one per-profile bundle with slit/boss groups."""

    output_paths = build_default_sigma_unit_hdf5_tables(
        output_directory=tmp_path,
        gamma_axis=np.array([1.2, 2.8]),
        zd_axis=np.array([0.43, 0.82]),
        devauc_log_re_kpc_axis=np.array([0.45, 1.20]),
        sersic_log_re_kpc_axis=np.array([0.50, 1.40]),
        sersic_n_axis=np.array([2.5, 10.5]),
    )

    assert output_paths["devauc"] == tmp_path / "jeans_deV_sigma_bundle.h5"
    assert output_paths["sersic"] == tmp_path / "jeans_sers_sigma_bundle.h5"

    assert np.array_equal(SIGMA_UNIT_GAMMA_AXIS, np.linspace(1.2, 2.8, 17))
    assert np.array_equal(SIGMA_UNIT_ZD_AXIS, np.linspace(0.43, 0.82, 21))
    assert np.array_equal(SIGMA_UNIT_DEVAUC_LOG_RE_KPC_AXIS, np.linspace(0.45, 1.20, 21))
    assert np.array_equal(SIGMA_UNIT_SERSIC_LOG_RE_KPC_AXIS, np.linspace(0.50, 1.40, 21))
    assert np.array_equal(SIGMA_UNIT_SERSIC_N_AXIS, np.linspace(2.5, 10.5, 21))

    with h5py.File(output_paths["devauc"], "r") as handle:
        assert set(handle.keys()) == {"profile_name", "slit", "boss"}
        assert handle["profile_name"][()].decode("utf-8") == "devauc"
        assert handle.attrs["schema_version"] == "sigma_unit_bundle_hdf5_v2"
        assert handle.attrs["quantity_name"] == "S_unit"
        boss_m10 = handle["boss"]["m10"]
        assert set(boss_m10.keys()) == {"gamma_axis", "zd_axis", "log_re_kpc_axis", "s_unit_grid"}
        assert boss_m10["s_unit_grid"].shape == (2, 2, 2)
        assert boss_m10.attrs["mass_definition_label"] == "m10"
        assert boss_m10.attrs["mass_radius_kpc"] == 10.0
        assert boss_m10.attrs["units"] == "km2 s-2 per 10**m10"
        assert boss_m10.attrs["observation_flavor"] == "boss"
        assert boss_m10.attrs["aperture_shape"] == "circular"
        assert boss_m10.attrs["aperture_radius_arcsec"] == 1.0

    with h5py.File(output_paths["sersic"], "r") as handle:
        assert set(handle.keys()) == {"profile_name", "slit", "boss"}
        assert handle["profile_name"][()].decode("utf-8") == "sersic"
        slit_m5 = handle["slit"]["m5"]
        assert set(slit_m5.keys()) == {"gamma_axis", "zd_axis", "log_re_kpc_axis", "n_axis", "s_unit_grid"}
        assert slit_m5["s_unit_grid"].shape == (2, 2, 2, 2)
        assert slit_m5.attrs["mass_definition_label"] == "m5"
        assert slit_m5.attrs["mass_radius_kpc"] == 5.0
        assert slit_m5.attrs["observation_flavor"] == "slit"
        assert slit_m5.attrs["aperture_shape"] == "rectangular"
        assert slit_m5.attrs["aperture_width_arcsec"] == 1.6
        assert slit_m5.attrs["aperture_height_arcsec"] == 0.9
        assert np.all(np.diff(slit_m5["n_axis"][:]) > 0.0)


def test_build_default_sigma_unit_hdf5_tables_can_limit_work_to_one_profile(tmp_path: Path) -> None:
    """Targeted reruns should refresh only requested bundle leaves."""

    output_paths = build_default_sigma_unit_hdf5_tables(
        output_directory=tmp_path,
        gamma_axis=np.array([1.2, 2.8]),
        zd_axis=np.array([0.43, 0.82]),
        devauc_log_re_kpc_axis=np.array([0.45, 1.20]),
        profiles=("devauc",),
        observation_flavors=("slit",),
        mass_radii_kpc=(10,),
        workers=1,
    )

    assert set(output_paths) == {"devauc"}
    assert output_paths["devauc"].exists()

    build_default_sigma_unit_hdf5_tables(
        output_directory=tmp_path,
        gamma_axis=np.array([1.2, 2.8]),
        zd_axis=np.array([0.43, 0.82]),
        devauc_log_re_kpc_axis=np.array([0.45, 1.20]),
        profiles=("devauc",),
        observation_flavors=("boss",),
        mass_radii_kpc=(5,),
        workers=1,
    )

    with h5py.File(output_paths["devauc"], "r") as handle:
        assert set(handle["slit"].keys()) == {"m10"}
        assert set(handle["boss"].keys()) == {"m5"}
        assert handle["slit"]["m10"].attrs["observation_flavor"] == "slit"
        assert handle["boss"]["m5"].attrs["observation_flavor"] == "boss"


def test_repack_legacy_sigma_unit_hdf5_tables_into_bundles_migrates_only_slit_leaves(tmp_path: Path) -> None:
    """Legacy repack should copy existing slit tables into new bundles without touching the source files."""

    legacy_m5_path = tmp_path / "jeans_deV_m5_grid.h5"
    legacy_m10_path = tmp_path / "jeans_deV_m10_grid.h5"

    write_sigma_unit_table_hdf5(
        build_sigma_unit_table(
            profile_name="devauc",
            mass_radius_kpc=5.0,
            gamma_axis=np.array([1.2, 2.8]),
            zd_axis=np.array([0.43, 0.82]),
            log_re_kpc_axis=np.array([0.45, 1.20]),
            workers=1,
        ),
        legacy_m5_path,
    )
    write_sigma_unit_table_hdf5(
        build_sigma_unit_table(
            profile_name="devauc",
            mass_radius_kpc=10.0,
            gamma_axis=np.array([1.2, 2.8]),
            zd_axis=np.array([0.43, 0.82]),
            log_re_kpc_axis=np.array([0.45, 1.20]),
            workers=1,
        ),
        legacy_m10_path,
    )
    legacy_snapshots = {
        legacy_m5_path.name: legacy_m5_path.read_bytes(),
        legacy_m10_path.name: legacy_m10_path.read_bytes(),
    }

    output_paths = repack_legacy_sigma_unit_hdf5_tables_into_bundles(
        input_directory=tmp_path,
        output_directory=tmp_path,
        profiles=("devauc",),
        mass_radii_kpc=(5.0, 10.0),
    )

    assert output_paths["devauc"] == tmp_path / "jeans_deV_sigma_bundle.h5"
    assert legacy_m5_path.read_bytes() == legacy_snapshots[legacy_m5_path.name]
    assert legacy_m10_path.read_bytes() == legacy_snapshots[legacy_m10_path.name]

    with h5py.File(output_paths["devauc"], "r") as handle:
        assert set(handle.keys()) == {"profile_name", "slit", "boss"}
        assert set(handle["slit"].keys()) == {"m5", "m10"}
        assert set(handle["boss"].keys()) == set()
        slit_m10 = handle["slit"]["m10"]
        assert slit_m10.attrs["observation_flavor"] == "slit"
        assert slit_m10.attrs["aperture_shape"] == "rectangular"
        assert slit_m10.attrs["aperture_width_arcsec"] == 1.6
        assert slit_m10.attrs["aperture_height_arcsec"] == 0.9

        with h5py.File(legacy_m10_path, "r") as legacy_handle:
            np.testing.assert_allclose(slit_m10["s_unit_grid"][:], legacy_handle["s_unit_grid"][:])


def test_sigma_unit_cli_exposes_observation_flavor_selector() -> None:
    """The public CLI must let callers build only slit, boss, or both flavors."""

    parser = build_parser()
    args = parser.parse_args(
        [
            "--build-sigma-unit-hdf5",
            "--profile",
            "devauc",
            "--observation-flavor",
            "boss",
        ]
    )

    assert args.build_sigma_unit_hdf5 is True
    assert args.profile == "devauc"
    assert args.observation_flavor == "boss"


def test_sigma_unit_cli_exposes_legacy_repack_mode() -> None:
    """The public CLI must expose the migration path from flat legacy files to bundles."""

    parser = build_parser()
    args = parser.parse_args(
        [
            "--repack-legacy-sigma-unit-hdf5",
            "--profile",
            "devauc",
        ]
    )

    assert args.repack_legacy_sigma_unit_hdf5 is True
    assert args.profile == "devauc"
