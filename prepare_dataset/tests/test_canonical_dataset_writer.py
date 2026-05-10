"""Tests for canonical inference dataset construction.

The writer is deliberately data-preparation-only: it converts the current
CMASS-style observation, mass-grid, and cross-section files into one canonical
HDF5 input product.  Bayesian inference readers and validators are out of
scope for this package-level migration.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from prepare_dataset.config import H_UNITS_V1, LEGACY_FIXED_KPC
from prepare_dataset.dataset_schema.canonical import (
    CAPABILITY_LENS_OBSERVATIONS_V1,
    CAPABILITY_LENSING_CROSS_SECTION_THETA_GAMMA_V1,
    CAPABILITY_LENSING_MASS_GRIDS_V1,
    CAPABILITY_VELOCITY_DISPERSION_POPULATION_SIGMA_UNIT_V1,
    CAPABILITY_VELOCITY_DISPERSION_PER_LENS_S2_V1,
    TOP_LEVEL_BLOCKS,
)
from prepare_dataset.dataset_schema.writer import write_canonical_inference_dataset


def _write_observation_file(path: Path, *, omit_required_s2: bool = False) -> Path:
    """Create a two-lens hunit observation file with one sigma-free lens."""

    gamma_grid = np.asarray([1.2, 2.0, 2.8], dtype=float)
    with h5py.File(path, "w") as handle:
        handle.attrs["unit_convention"] = H_UNITS_V1
        handle.attrs["h_ref"] = 0.7
        handle.attrs["mass_definition_label"] = "m5_hinvkpc"
        for index, lens_id in enumerate(("lens_with_sigma", "lens_without_sigma")):
            group = handle.create_group(lens_id)
            group.attrs["unit_convention"] = H_UNITS_V1
            group.attrs["h_ref"] = 0.7
            group.attrs["zd"] = 0.5 + 0.1 * index
            group.attrs["zs"] = 1.5 + 0.1 * index
            group.attrs["logmchab_h2"] = 11.1 + 0.1 * index
            group.attrs["logmchab_err"] = 0.08
            group.attrs["log10_re_hinv_kpc"] = 0.7 + 0.1 * index
            group.attrs["nser"] = 4.0 + index
            group.attrs["rein_arcsec"] = 1.0 + 0.1 * index
            group.attrs["num_sigma"] = 1 if index == 0 else 0
            group.attrs["sigma"] = np.asarray([250.0 + index], dtype=float)
            group.attrs["sigma_err"] = np.asarray([20.0], dtype=float)
            group.create_dataset("gamma_grid", data=gamma_grid)
            mass_root = group.create_group("mass_definitions")
            mass_group = mass_root.create_group("m5_hinvkpc")
            mass_group.attrs["unit_convention"] = H_UNITS_V1
            mass_group.attrs["h_ref"] = 0.7
            mass_group.attrs["mass_definition_label"] = "m5_hinvkpc"
            mass_group.attrs["mass_radius_kpc"] = 5.0
            mass_group.create_dataset("mass_grid", data=np.asarray([10.0, 10.2, 10.4]) + index)
            mass_group.create_dataset("dmass_dthetaein_grid", data=np.asarray([0.9, 1.0, 1.1]))
            if index == 0 and not omit_required_s2:
                mass_group.create_dataset("s2_grid", data=np.asarray([1.0, 1.2, 1.4]) * 1.0e-5)
    return path


def _write_cross_section_file(path: Path) -> Path:
    """Create the legacy one-dimensional CMASS cross-section input."""

    with h5py.File(path, "w") as handle:
        compressed = handle.create_group("compressed_grids")
        compressed.create_dataset("gamma_grid", data=np.asarray([1.2, 2.0, 2.8], dtype=float))
        compressed.create_dataset("cs_over_theta_ein", data=np.asarray([0.2, 0.3, 0.4], dtype=float))
    return path


def _write_fibre_cross_section_file(path: Path) -> Path:
    """Create a Sonnenfeld-style two-dimensional finite-fibre cross-section input."""

    with h5py.File(path, "w") as handle:
        handle.create_dataset("tein_grid", data=np.asarray([0.0, 1.0], dtype=float))
        handle.create_dataset("gamma_grid", data=np.asarray([1.2, 2.0, 2.8], dtype=float))
        handle.create_dataset(
            "mufibre2_cs_grid",
            data=np.asarray(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 2.0, 3.0],
                ],
                dtype=float,
            ),
        )
        handle.create_dataset(
            "mufibre3_cs_grid",
            data=np.asarray(
                [
                    [0.0, 0.0, 0.0],
                    [0.5, 1.0, 1.5],
                ],
                dtype=float,
            ),
        )
        handle.create_dataset("ycaust_grid", data=np.asarray([[0.0, 0.0, 0.0], [0.1, 0.2, 0.3]], dtype=float))
        handle.attrs["generator_name"] = "prepare_dataset.fibre_cross_section"
    return path


def _write_fixed_m5_observation_file(path: Path) -> Path:
    """Create a minimal legacy fixed-kpc observation file for SLACS-style tests."""

    gamma_grid = np.asarray([1.2, 2.0, 2.8], dtype=float)
    with h5py.File(path, "w") as handle:
        handle.attrs["unit_convention"] = LEGACY_FIXED_KPC
        handle.attrs["mass_definition_label"] = "m5"
        handle.attrs["profile_name"] = "devauc"
        for index, lens_id in enumerate(("slacs_a", "slacs_b")):
            group = handle.create_group(lens_id)
            group.attrs["unit_convention"] = LEGACY_FIXED_KPC
            group.attrs["zd"] = 0.2 + 0.1 * index
            group.attrs["zs"] = 0.6 + 0.1 * index
            group.attrs["logmchab_deV"] = 11.0 + 0.1 * index
            group.attrs["logmchab_err"] = 0.08
            group.attrs["log10_reff_deV_kpc"] = 0.5 + 0.1 * index
            group.attrs["nser"] = 4.0
            group.attrs["rein_arcsec"] = 1.1 + 0.1 * index
            group.attrs["num_sigma"] = 1
            group.attrs["sigma"] = np.asarray([240.0 + index], dtype=float)
            group.attrs["sigma_err"] = np.asarray([15.0], dtype=float)
            group.create_dataset("gamma_grid", data=gamma_grid)
            mass_group = group.create_group("mass_definitions").create_group("m5")
            mass_group.attrs["unit_convention"] = LEGACY_FIXED_KPC
            mass_group.attrs["mass_definition_label"] = "m5"
            mass_group.attrs["mass_radius_kpc"] = 5.0
            mass_group.create_dataset("mass_grid", data=np.asarray([10.0, 10.2, 10.4]) + index)
            mass_group.create_dataset("dmass_dthetaein_grid", data=np.asarray([0.8, 0.9, 1.0]))
            mass_group.create_dataset("s2_grid", data=np.asarray([2.0, 2.2, 2.4]) * 1.0e-5)
    return path


def _write_population_sigma_table(path: Path) -> Path:
    """Create the flat sigma-unit table shape copied into Sonnenfeld canonical files."""

    with h5py.File(path, "w") as handle:
        handle.attrs["schema_version"] = "sigma_unit_table_v1"
        handle.attrs["profile_name"] = "devauc"
        handle.attrs["unit_convention"] = LEGACY_FIXED_KPC
        handle.attrs["mass_definition_label"] = "m5"
        handle.attrs["mass_radius_kpc"] = 5.0
        handle.attrs["sigma_definition"] = "observed_aperture"
        handle.attrs["aperture_shape"] = "circular"
        handle.attrs["aperture_radius_arcsec"] = 1.5
        handle.attrs["seeing_fwhm_arcsec"] = 1.5
        handle.create_dataset("profile_name", data=np.bytes_("devauc"))
        handle.create_dataset("gamma_axis", data=np.asarray([1.2, 2.0, 2.8], dtype=float))
        handle.create_dataset("zd_axis", data=np.asarray([0.1, 0.3], dtype=float))
        handle.create_dataset("log_re_kpc_axis", data=np.asarray([-0.2, 0.0, 0.2], dtype=float))
        handle.create_dataset("s_unit_grid", data=np.ones((3, 2, 3), dtype=float))
    return path


def test_write_canonical_inference_dataset_creates_expected_schema_blocks(tmp_path: Path) -> None:
    """The writer should emit the agreed top-level canonical HDF5 blocks."""

    observation_path = _write_observation_file(tmp_path / "observations.hdf5")
    cross_section_path = _write_cross_section_file(tmp_path / "cross_section.h5")
    output_path = write_canonical_inference_dataset(
        observation_path=observation_path,
        cross_section_path=cross_section_path,
        output_path=tmp_path / "inference_dataset.hdf5",
        profile_name="sersic",
        mass_definition_label="m5_hinvkpc",
        unit_convention=H_UNITS_V1,
        h_ref=0.7,
        theta_e_axis=np.asarray([0.5, 1.0, 2.0], dtype=float),
    )

    with h5py.File(output_path, "r") as handle:
        assert set(handle.keys()) == set(TOP_LEVEL_BLOCKS)
        metadata = handle["metadata"]
        assert metadata.attrs["schema_version"] == "canonical_inference_dataset_v1"
        assert metadata.attrs["unit_convention"] == H_UNITS_V1
        assert metadata.attrs["h_ref"] == pytest.approx(0.7)
        assert metadata.attrs["profile_name"] == "sersic"
        assert metadata.attrs["mass_definition_label"] == "m5_hinvkpc"
        capabilities = {
            item.decode("utf-8") if isinstance(item, bytes) else str(item)
            for item in metadata["capabilities"][()]
        }
        assert {
            CAPABILITY_LENS_OBSERVATIONS_V1,
            CAPABILITY_LENSING_MASS_GRIDS_V1,
            CAPABILITY_LENSING_CROSS_SECTION_THETA_GAMMA_V1,
            CAPABILITY_VELOCITY_DISPERSION_PER_LENS_S2_V1,
        }.issubset(capabilities)

        lenses = handle["lenses"]
        assert lenses["z_d"].shape == (2,)
        assert lenses["sigma_obs"].shape == (2, 2)
        np.testing.assert_array_equal(lenses["num_sigma"][()], np.asarray([1, 0]))

        mass_grids = handle["lensing_mass_grids"]
        assert mass_grids["log_enclosed_mass_grid"].shape == (2, 3)
        assert mass_grids["s2_grid"].shape == (2, 3)
        np.testing.assert_array_equal(mass_grids["has_s2"][()], np.asarray([1, 0]))

        cross_section = handle["lensing_cross_section"]
        assert cross_section["cross_section_grid"].shape == (3, 3)


def test_write_canonical_inference_dataset_records_observation_contract_metadata(tmp_path: Path) -> None:
    """Canonical metadata should preserve explicit observed-aperture geometry."""

    observation_path = _write_observation_file(tmp_path / "observations.hdf5")
    with h5py.File(observation_path, "a") as handle:
        for group in handle.values():
            group.attrs["observation_flavor"] = "boss"
            group.attrs["sigma_definition"] = "observed_aperture"
            group.attrs["aperture_shape"] = "circular"
            group.attrs["aperture_radius_arcsec"] = 1.0
            group.attrs["seeing_fwhm_arcsec"] = 1.5

    output_path = write_canonical_inference_dataset(
        observation_path=observation_path,
        cross_section_path=_write_cross_section_file(tmp_path / "cross_section.h5"),
        output_path=tmp_path / "neutral_filename.hdf5",
        profile_name="sersic",
        mass_definition_label="m5_hinvkpc",
        unit_convention=H_UNITS_V1,
        h_ref=0.7,
        theta_e_axis=np.asarray([0.5, 1.0, 2.0], dtype=float),
    )

    with h5py.File(output_path, "r") as handle:
        metadata = handle["metadata"].attrs
        assert metadata["observation_flavor"] == "boss"
        assert metadata["sigma_definition"] == "observed_aperture"
        assert metadata["aperture_shape"] == "circular"
        assert metadata["aperture_radius_arcsec"] == pytest.approx(1.0)
        assert metadata["seeing_fwhm_arcsec"] == pytest.approx(1.5)
        cross_section = handle["lensing_cross_section"]
        expected = np.pi * (np.asarray([0.5, 1.0, 2.0])[:, None] * np.asarray([0.2, 0.3, 0.4])[None, :]) ** 2
        np.testing.assert_allclose(cross_section["cross_section_grid"][()], expected)

        per_lens_s2 = handle["velocity_dispersion_grids"]["per_lens_s2"]
        assert per_lens_s2["s2_grid"].shape == (2, 3)
        np.testing.assert_array_equal(per_lens_s2["has_s2"][()], np.asarray([1, 0]))


def test_write_canonical_inference_dataset_accepts_fibre_cross_section_grid(tmp_path: Path) -> None:
    """The canonical writer should preserve Sonnenfeld finite-fibre area grids directly."""

    observation_path = _write_observation_file(tmp_path / "observations.hdf5")
    cross_section_path = _write_fibre_cross_section_file(tmp_path / "fibre_crosssect_grid.hdf5")
    output_path = write_canonical_inference_dataset(
        observation_path=observation_path,
        cross_section_path=cross_section_path,
        output_path=tmp_path / "inference_dataset_fibre.hdf5",
        profile_name="sersic",
        mass_definition_label="m5_hinvkpc",
        unit_convention=H_UNITS_V1,
        h_ref=0.7,
        theta_e_axis=np.asarray([0.2, 0.4], dtype=float),
    )

    with h5py.File(output_path, "r") as handle:
        cross_section = handle["lensing_cross_section"]
        np.testing.assert_allclose(cross_section["theta_e_axis"][()], np.asarray([0.0, 1.0]))
        np.testing.assert_allclose(cross_section["gamma_axis"][()], np.asarray([1.2, 2.0, 2.8]))
        np.testing.assert_allclose(
            cross_section["cross_section_grid"][()],
            np.asarray([[0.0, 0.0, 0.0], [0.5, 1.0, 1.5]], dtype=float),
        )
        assert cross_section.attrs["source"] == "mufibre3_cs_grid"


def test_write_canonical_inference_dataset_copies_population_sigma_unit(tmp_path: Path) -> None:
    """Sonnenfeld datasets require a population-level sigma-unit interpolation grid."""

    observation_path = _write_fixed_m5_observation_file(tmp_path / "slacs_observations.hdf5")
    cross_section_path = _write_fibre_cross_section_file(tmp_path / "fibre_crosssect_grid.hdf5")
    population_sigma_path = _write_population_sigma_table(tmp_path / "population_sigma_unit.h5")
    output_path = write_canonical_inference_dataset(
        observation_path=observation_path,
        cross_section_path=cross_section_path,
        output_path=tmp_path / "sonnenfeld_canonical.hdf5",
        profile_name="devauc",
        mass_definition_label="m5",
        unit_convention=LEGACY_FIXED_KPC,
        h_ref=0.7,
        population_sigma_path=population_sigma_path,
    )

    with h5py.File(output_path, "r") as handle:
        capabilities = {
            item.decode("utf-8") if isinstance(item, bytes) else str(item)
            for item in handle["metadata/capabilities"][()]
        }
        assert CAPABILITY_VELOCITY_DISPERSION_POPULATION_SIGMA_UNIT_V1 in capabilities
        population_group = handle["velocity_dispersion_grids/population_sigma_unit"]
        np.testing.assert_allclose(population_group["gamma_axis"][()], np.asarray([1.2, 2.0, 2.8]))
        np.testing.assert_allclose(population_group["zd_axis"][()], np.asarray([0.1, 0.3]))
        np.testing.assert_allclose(population_group["log_re_kpc_axis"][()], np.asarray([-0.2, 0.0, 0.2]))
        assert population_group["s_unit_grid"].shape == (3, 2, 3)
        assert population_group.attrs["mass_definition_label"] == "m5"


def test_write_canonical_inference_dataset_rejects_sigma_lens_without_s2_grid(tmp_path: Path) -> None:
    """A sigma-bearing lens without `s2_grid` should fail before writing output."""

    observation_path = _write_observation_file(tmp_path / "observations_missing_s2.hdf5", omit_required_s2=True)
    cross_section_path = _write_cross_section_file(tmp_path / "cross_section.h5")
    output_path = tmp_path / "bad_inference_dataset.hdf5"

    with pytest.raises(ValueError, match="num_sigma.*s2_grid"):
        write_canonical_inference_dataset(
            observation_path=observation_path,
            cross_section_path=cross_section_path,
            output_path=output_path,
            profile_name="sersic",
            mass_definition_label="m5_hinvkpc",
            unit_convention=H_UNITS_V1,
            h_ref=0.7,
        )

    assert not output_path.exists()
