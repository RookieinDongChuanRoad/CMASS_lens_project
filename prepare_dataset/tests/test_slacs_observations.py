"""Tests for SLACS table ingestion and fixed-kpc observation HDF5 writing."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from prepare_dataset.config import H_UNITS_V1, LEGACY_FIXED_KPC
from prepare_dataset.io import slacs_observations


SLACS_HEADER = (
    "# name RA dec zd zs Reff_arcsec Reff_kpc theta_Ein Rein_kpc "
    "lMstar_Chab lMstar_err veldisp(km/s) veldisp_err(km/s)\n"
)


def _write_slacs_cat(path: Path, rows: list[str]) -> Path:
    """Write a tiny SLACS-like catalog fixture with the production column order."""

    path.write_text(SLACS_HEADER + "".join(rows), encoding="utf-8")
    return path


def test_read_slacs_table_validates_column_contract_and_physical_values(tmp_path: Path) -> None:
    """The raw table reader should fail early on malformed lens-level inputs."""

    catalog_path = _write_slacs_cat(
        tmp_path / "SLACS_table.cat",
        [
            "SDSSJ0001 1.0 2.0 0.20 0.60 1.50 5.00 1.10 3.70 11.10 0.08 240.0 15.0\n",
            "SDSSJ0002 3.0 4.0 0.30 0.80 1.80 6.00 1.30 4.50 11.30 0.09 260.0 18.0\n",
        ],
    )

    rows = slacs_observations.read_slacs_table(catalog_path, expected_rows=2)

    assert [row.name for row in rows] == ["SDSSJ0001", "SDSSJ0002"]
    assert rows[0].zs > rows[0].zd
    assert rows[1].velocity_dispersion_err == pytest.approx(18.0)

    bad_catalog_path = _write_slacs_cat(
        tmp_path / "bad_SLACS_table.cat",
        ["SDSSJ0003 1.0 2.0 0.50 0.40 1.50 5.00 1.10 3.70 11.10 0.08 240.0 15.0\n"],
    )
    with pytest.raises(ValueError, match="zs.*zd"):
        slacs_observations.read_slacs_table(bad_catalog_path, expected_rows=1)


def test_write_slacs_observation_hdf5_creates_m5_mass_and_s2_grids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The SLACS HDF5 builder should emit the canonical-writer raw contract."""

    catalog_path = _write_slacs_cat(
        tmp_path / "SLACS_table.cat",
        ["SDSSJ0001 1.0 2.0 0.20 0.60 1.50 5.00 1.10 3.70 11.10 0.08 240.0 15.0\n"],
    )
    gamma_axis = np.asarray([1.2, 2.0], dtype=float)

    def fake_sigma_unit_grid(**kwargs: object) -> np.ndarray:
        """Return a deterministic vector so this test does not run Jeans solves."""

        np.testing.assert_allclose(kwargs["gamma_grid"], gamma_axis)
        assert kwargs["profile_name"] == "devauc"
        assert kwargs["mass_radius_kpc"] == 5.0
        assert kwargs["unit_convention"] == LEGACY_FIXED_KPC
        return np.asarray([1.0e-5, 2.0e-5], dtype=float)

    monkeypatch.setattr(slacs_observations, "compute_sigma_unit_grid", fake_sigma_unit_grid)
    output_path = slacs_observations.write_slacs_observation_hdf5(
        catalog_path=catalog_path,
        output_path=tmp_path / "observations_SLACS_deV_with_mass_grids_fixed_m5.hdf5",
        gamma_axis=gamma_axis,
        derivative_theta_samples=9,
        expected_rows=1,
    )

    with h5py.File(output_path, "r") as handle:
        assert handle.attrs["unit_convention"] == LEGACY_FIXED_KPC
        assert handle.attrs["profile_name"] == "devauc"
        assert handle.attrs["mass_definition_label"] == "m5"
        assert sorted(handle.keys()) == ["SDSSJ0001"]

        lens_group = handle["SDSSJ0001"]
        assert lens_group.attrs["num_sigma"] == 1
        assert lens_group.attrs["sigma"][0] == pytest.approx(240.0)
        assert lens_group.attrs["sigma_err"][0] == pytest.approx(15.0)
        assert lens_group.attrs["nser"] == pytest.approx(4.0)
        assert lens_group.attrs["log10_reff_deV_kpc"] == pytest.approx(np.log10(5.0))
        np.testing.assert_allclose(lens_group["gamma_grid"][()], gamma_axis)

        mass_group = lens_group["mass_definitions/m5"]
        assert mass_group.attrs["unit_convention"] == LEGACY_FIXED_KPC
        assert mass_group.attrs["mass_definition_label"] == "m5"
        assert mass_group.attrs["mass_radius_kpc"] == pytest.approx(5.0)
        assert mass_group["mass_grid"].shape == (2,)
        assert mass_group["dmass_dthetaein_grid"].shape == (2,)
        np.testing.assert_allclose(mass_group["s2_grid"][()], np.asarray([1.0e-5, 2.0e-5]))


def test_write_slacs_population_sigma_unit_hdf5_uses_slacs_z_axis_and_unit_convention(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The SLACS population table should use the low-redshift SLACS grid for h-units too."""

    captured_kwargs: dict[str, object] = {}
    sentinel_table = object()

    def fake_build_sigma_unit_table(**kwargs: object) -> object:
        captured_kwargs.update(kwargs)
        return sentinel_table

    def fake_write_sigma_unit_table_hdf5(table: object, output_path: Path | str) -> Path:
        assert table is sentinel_table
        return Path(output_path)

    monkeypatch.setattr(slacs_observations, "build_sigma_unit_table", fake_build_sigma_unit_table)
    monkeypatch.setattr(slacs_observations, "write_sigma_unit_table_hdf5", fake_write_sigma_unit_table_hdf5)

    output_path = slacs_observations.write_slacs_population_sigma_unit_hdf5(
        output_path=tmp_path / "slacs_population_sigma_unit_m5_hunits_v1.h5",
        unit_convention=H_UNITS_V1,
        h_ref=0.7,
        workers=3,
    )

    assert output_path == tmp_path / "slacs_population_sigma_unit_m5_hunits_v1.h5"
    assert captured_kwargs["profile_name"] == "devauc"
    assert captured_kwargs["mass_radius_kpc"] == 5.0
    assert captured_kwargs["unit_convention"] == H_UNITS_V1
    assert captured_kwargs["h_ref"] == pytest.approx(0.7)
    assert captured_kwargs["workers"] == 3
    np.testing.assert_allclose(
        captured_kwargs["zd_axis"],
        slacs_observations.SLACS_POPULATION_SIGMA_ZD_AXIS,
    )
    np.testing.assert_allclose(
        captured_kwargs["zd_axis"],
        np.linspace(0.05, 0.40, 21, dtype=float),
    )
