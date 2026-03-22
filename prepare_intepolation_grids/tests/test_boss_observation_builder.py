"""Regression tests for the BOSS observation-file builder.

These tests lock down the new behavior requested for the BOSS data product:

- parse the commented summary-table header correctly
- build new HDF5 files from scratch instead of copying old files
- generate full mass/s2 grids plus flat-prior summary attrs
- store explicit circular-aperture metadata for the BOSS products
"""

from __future__ import annotations

import math
from pathlib import Path

import h5py
import numpy as np
import pytest

from interpolation_grids.io.boss_observations import (
    build_boss_observation_hdf5_files,
    read_boss_summary_table,
)
from interpolation_grids.physics.jeans import kpc_per_arcsec


SUMMARY_HEADER = (
    "# name zd zs rein_arcsec re_arcsec nser logmchab logmchab_err "
    "sigma sigma_err imag_Ser imag_deV reff_deV logmchab_deV"
)


def _write_summary_fixture(path: Path) -> None:
    """Write a compact BOSS-style summary table fixture."""

    path.write_text(
        "\n".join(
            [
                SUMMARY_HEADER,
                "023817-054555 0.599 1.763 0.929 0.706 4.589 11.591 0.101 291 71 19.18 19.20 0.62 11.58",
                "121052-011905 0.700 2.295 1.279 2.205 7.779 11.697 0.110 347 78 19.39 19.84 0.87 11.52",
            ]
        ),
        encoding="utf-8",
    )


def _expected_log10_sigma_star(log_stellar_mass: float, effective_radius_arcsec: float, zd: float) -> float:
    """Recompute the surface-density-like stellar-mass summary used by the files."""

    effective_radius_kpc = effective_radius_arcsec * kpc_per_arcsec(zd)
    return log_stellar_mass - math.log10(2.0 * math.pi * effective_radius_kpc**2)


def test_read_boss_summary_table_parses_commented_header(tmp_path: Path) -> None:
    """The summary-table parser must understand the project's commented header format."""

    summary_path = tmp_path / "summary_table_deV.txt"
    _write_summary_fixture(summary_path)

    rows = read_boss_summary_table(summary_path)

    assert [row.name for row in rows] == ["023817-054555", "121052-011905"]
    assert rows[0].sigma == 291
    assert rows[0].sigma_err == 71
    assert rows[1].reff_deV == 0.87
    assert rows[1].logmchab_deV == 11.52


def test_build_boss_observation_hdf5_files_writes_full_grid_products(tmp_path: Path) -> None:
    """The builder should produce complete BOSS observation files for both profiles."""

    summary_path = tmp_path / "summary_table_deV.txt"
    _write_summary_fixture(summary_path)

    output_paths = build_boss_observation_hdf5_files(
        summary_path=summary_path,
        output_directory=tmp_path,
    )

    assert set(output_paths) == {"devauc", "sersic"}
    assert output_paths["devauc"].name == "observations_deV_with_BOSS_mass_grids.hdf5"
    assert output_paths["sersic"].name == "observations_with_BOSS_mass_grids_all.hdf5"

    for output_path in output_paths.values():
        assert output_path.exists()
        with h5py.File(output_path, "r") as handle:
            assert sorted(handle.keys()) == ["023817-054555", "121052-011905"]
            for group_name in handle:
                group = handle[group_name]
                sigma_values = np.asarray(group.attrs["sigma"]).tolist()
                sigma_errors = np.asarray(group.attrs["sigma_err"]).tolist()

                assert int(group.attrs["num_sigma"]) == 1
                assert sigma_values in ([291], [347])
                assert sigma_errors in ([71], [78])
                assert group.attrs["aperture_shape"] == "circular"
                assert float(group.attrs["aperture_radius_arcsec"]) == 1.0
                assert float(group.attrs["seeing_fwhm_arcsec"]) == 0.9

                for attr_name in (
                    "gamma_lower",
                    "gamma_mid",
                    "gamma_upper",
                    "m5_lower",
                    "m5_mid",
                    "m5_upper",
                    "m10_lower",
                    "m10_mid",
                    "m10_upper",
                ):
                    assert attr_name in group.attrs

                assert "gamma_grid" in group
                assert "m5_grid" in group
                assert "dm5_dthetaein_grid" in group
                assert "s2_grid" in group
                assert "mass_definitions" in group
                for label in ("m5", "m10"):
                    subgroup = group["mass_definitions"][label]
                    assert "mass_grid" in subgroup
                    assert "dmass_dthetaein_grid" in subgroup
                    assert "s2_grid" in subgroup

    with h5py.File(output_paths["sersic"], "r") as handle:
        group = handle["023817-054555"]
        expected = _expected_log10_sigma_star(
            log_stellar_mass=float(group.attrs["logmchab"]),
            effective_radius_arcsec=float(group.attrs["re_arcsec"]),
            zd=float(group.attrs["zd"]),
        )
        assert float(group.attrs["log10_Sigma_star"]) == pytest.approx(expected)

    with h5py.File(output_paths["devauc"], "r") as handle:
        group = handle["023817-054555"]
        assert "reff_deV" in group.attrs
        assert "logmchab_deV" in group.attrs
        expected = _expected_log10_sigma_star(
            log_stellar_mass=float(group.attrs["logmchab_deV"]),
            effective_radius_arcsec=float(group.attrs["reff_deV"]),
            zd=float(group.attrs["zd"]),
        )
        assert float(group.attrs["log10_Sigma_star"]) == pytest.approx(expected)
