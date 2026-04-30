"""Tests for the new sigma-unit HDF5 interpolation-table workflow.

These tests lock in the contract for the PPT-facing Jeans tables:
- the numerical grid stores `S_unit`, not the legacy `s2_grid`
- the builder writes the new explicit HDF5 schema
- table values on grid nodes agree with direct Jeans evaluation
"""

from __future__ import annotations

from pathlib import Path
import sys

import h5py
import numpy as np
import pytest

from interpolation_grids.config import (
    BOSS_CIRCULAR_APERTURE_POLICY,
    DEFAULT_PRODUCTION_APERTURE_POLICY,
    H_UNITS_V1,
    SIGMA_UNIT_DEVAUC_LOG_RE_KPC_AXIS,
    SIGMA_UNIT_GAMMA_AXIS,
    SIGMA_UNIT_SERSIC_LOG_RE_KPC_AXIS,
    SIGMA_UNIT_SERSIC_N_AXIS,
    SIGMA_UNIT_ZD_AXIS,
)
from interpolation_grids.cli import build_parser, main
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


def test_build_sigma_unit_table_can_emit_h_unit_metadata_and_scaling(tmp_path: Path) -> None:
    """h-units sigma tables should shift size axes and scale S_unit analytically."""

    h_ref = 0.7
    gamma_axis = np.array([1.7, 2.0, 2.3])
    physical_log_re_axis = np.array([0.5])
    h_unit_log_re_axis = physical_log_re_axis + np.log10(h_ref)
    legacy_table = build_sigma_unit_table(
        profile_name="devauc",
        gamma_axis=gamma_axis,
        zd_axis=np.array([0.55]),
        log_re_kpc_axis=physical_log_re_axis,
        workers=1,
    )
    h_unit_table = build_sigma_unit_table(
        profile_name="devauc",
        gamma_axis=gamma_axis,
        zd_axis=np.array([0.55]),
        log_re_kpc_axis=h_unit_log_re_axis,
        workers=1,
        unit_convention=H_UNITS_V1,
        h_ref=h_ref,
    )

    assert h_unit_table.unit_convention == H_UNITS_V1
    assert h_unit_table.mass_definition_label == "m5_hinvkpc"
    np.testing.assert_allclose(h_unit_table.log_re_kpc_axis, h_unit_log_re_axis)
    np.testing.assert_allclose(
        h_unit_table.values,
        legacy_table.values * np.power(h_ref, 2.0 - gamma_axis)[:, None, None],
        rtol=1e-10,
        atol=1e-12,
    )

    output_path = write_sigma_unit_table_hdf5(h_unit_table, tmp_path / "h_unit_sigma.h5")
    with h5py.File(output_path, "r") as handle:
        assert handle.attrs["unit_convention"] == H_UNITS_V1
        assert handle.attrs["h_ref"] == h_ref
        assert handle.attrs["mass_definition_label"] == "m5_hinvkpc"
        assert handle.attrs["units"] == "km2 s-2 per 10**m5_hinvkpc"


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
    assert boss_table.seeing_fwhm_arcsec == 1.5
    assert not np.allclose(boss_table.values, slit_table.values)


def test_build_sigma_unit_table_supports_within_re_sigma_definition() -> None:
    """The builder must support the low-dimensional within-Re sigma definition."""

    devauc_table = build_sigma_unit_table(
        profile_name="devauc",
        gamma_axis=np.array([1.2, 2.0, 2.8]),
        log_re_kpc_axis=np.array([0.50, 1.10]),
        workers=1,
        sigma_definition="within_re",
    )
    sersic_table = build_sigma_unit_table(
        profile_name="sersic",
        gamma_axis=np.array([1.2, 2.0, 2.8]),
        log_re_kpc_axis=np.array([0.50, 1.10]),
        n_axis=np.array([2.5, 7.0, 10.5]),
        workers=1,
        sigma_definition="within_re",
    )

    assert devauc_table.sigma_definition == "within_re"
    assert devauc_table.bundle_group_name == "within_re"
    assert devauc_table.observation_flavor is None
    assert devauc_table.zd_axis is None
    assert devauc_table.values.shape == (3, 2)
    assert devauc_table.aperture_shape == "circular"
    assert devauc_table.aperture_radius_arcsec is None
    assert devauc_table.seeing_fwhm_arcsec is None

    assert sersic_table.sigma_definition == "within_re"
    assert sersic_table.bundle_group_name == "within_re"
    assert sersic_table.observation_flavor is None
    assert sersic_table.zd_axis is None
    assert sersic_table.values.shape == (3, 2, 3)
    assert sersic_table.n_axis is not None


def test_boss_aperture_policy_uses_flavor_specific_seeing_constant() -> None:
    """BOSS seeing must come from the BOSS flavor policy, not the slit default."""

    assert DEFAULT_PRODUCTION_APERTURE_POLICY.seeing_fwhm_arcsec == 0.9
    assert BOSS_CIRCULAR_APERTURE_POLICY.seeing_fwhm_arcsec == 1.5
    assert (
        BOSS_CIRCULAR_APERTURE_POLICY.seeing_fwhm_arcsec
        != DEFAULT_PRODUCTION_APERTURE_POLICY.seeing_fwhm_arcsec
    )


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
        assert set(handle.keys()) == {"profile_name", "slit", "boss", "within_re"}
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
        within_re_m5 = handle["within_re"]["m5"]
        assert set(within_re_m5.keys()) == {"gamma_axis", "log_re_kpc_axis", "s_unit_grid"}
        assert within_re_m5["s_unit_grid"].shape == (2, 2)
        assert within_re_m5.attrs["sigma_definition"] == "within_re"
        assert within_re_m5.attrs["aperture_shape"] == "circular"
        assert within_re_m5.attrs["aperture_radius_mode"] == "effective_radius"
        assert within_re_m5.attrs["seeing_mode"] == "none"
        assert "zd_axis" not in within_re_m5

    with h5py.File(output_paths["sersic"], "r") as handle:
        assert set(handle.keys()) == {"profile_name", "slit", "boss", "within_re"}
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
        within_re_m10 = handle["within_re"]["m10"]
        assert set(within_re_m10.keys()) == {"gamma_axis", "log_re_kpc_axis", "n_axis", "s_unit_grid"}
        assert within_re_m10["s_unit_grid"].shape == (2, 2, 2)
        assert within_re_m10.attrs["sigma_definition"] == "within_re"
        assert within_re_m10.attrs["aperture_shape"] == "circular"
        assert within_re_m10.attrs["aperture_radius_mode"] == "effective_radius"
        assert within_re_m10.attrs["seeing_mode"] == "none"
        assert "zd_axis" not in within_re_m10


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
        assert set(handle["within_re"].keys()) == {"m5", "m10"}
        assert handle["slit"]["m10"].attrs["observation_flavor"] == "slit"
        assert handle["boss"]["m5"].attrs["observation_flavor"] == "boss"
        assert handle["within_re"]["m5"].attrs["sigma_definition"] == "within_re"
        assert handle["within_re"]["m10"].attrs["sigma_definition"] == "within_re"


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
        assert set(handle.keys()) == {"profile_name", "slit", "boss", "within_re"}
        assert set(handle["slit"].keys()) == {"m5", "m10"}
        assert set(handle["boss"].keys()) == set()
        assert set(handle["within_re"].keys()) == set()
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


def test_sigma_unit_cli_exposes_sigma_definition_selector() -> None:
    """The public CLI must expose the observed-aperture versus within-Re selector."""

    parser = build_parser()
    args = parser.parse_args(
        [
            "--build-sigma-unit-hdf5",
            "--sigma-definition",
            "within_re",
        ]
    )

    assert args.build_sigma_unit_hdf5 is True
    assert args.sigma_definition == "within_re"


def test_sigma_unit_cli_rejects_within_re_when_observation_flavor_is_explicit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The runtime CLI guard must reject treating within-Re as an observation flavor."""

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "python",
            "-m",
            "interpolation_grids",
            "--build-sigma-unit-hdf5",
            "--sigma-definition",
            "within_re",
            "--observation-flavor",
            "boss",
        ],
    )
    with pytest.raises(SystemExit, match="2"):
        main()


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
