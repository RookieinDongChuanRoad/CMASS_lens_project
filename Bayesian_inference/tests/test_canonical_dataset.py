"""Tests for canonical inference dataset reading and validation."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from cmass_lens_inference.canonical_dataset import (
    CAPABILITY_LENSING_CROSS_SECTION_THETA_GAMMA_V1,
    CAPABILITY_LENSING_MASS_GRIDS_V1,
    CAPABILITY_LENS_OBSERVATIONS_V1,
    CAPABILITY_VELOCITY_DISPERSION_PER_LENS_S2_V1,
    CAPABILITY_VELOCITY_DISPERSION_POPULATION_SIGMA_UNIT_V1,
    load_canonical_inference_dataset,
)


def _write_canonical_dataset(
    path: Path,
    *,
    has_s2: bool = True,
    num_sigma: int = 1,
    declare_population_sigma_unit: bool = False,
    write_population_sigma_unit: bool = False,
    malformed_population_sigma_unit: bool = False,
) -> Path:
    """
    Write the smallest canonical HDF5 dataset needed by inference reader tests.

    The fixture follows the schema design in `docs/reports` rather than the
    legacy CMASS observation layout.  Tests therefore exercise the new reader
    boundary directly instead of reusing old raw-data helpers.
    """

    gamma_grid = np.asarray([1.3, 2.0, 2.7], dtype=np.float64)
    theta_axis = np.asarray([0.0, 1.0, 2.0], dtype=np.float64)
    with h5py.File(path, "w") as handle:
        metadata = handle.create_group("metadata")
        metadata.attrs["schema_version"] = "canonical_inference_dataset_v1"
        metadata.attrs["unit_convention"] = "h_units_v1"
        metadata.attrs["h_ref"] = 0.7
        metadata.attrs["profile_name"] = "sersic"
        metadata.attrs["mass_definition_label"] = "m5_hinvkpc"
        metadata.attrs["mass_radius_kpc"] = 5.0
        metadata.attrs["cosmology_h0"] = 70.0
        metadata.attrs["cosmology_omega_m"] = 0.3
        string_dtype = h5py.string_dtype(encoding="utf-8")
        capabilities = [
            CAPABILITY_LENS_OBSERVATIONS_V1,
            CAPABILITY_LENSING_MASS_GRIDS_V1,
            CAPABILITY_LENSING_CROSS_SECTION_THETA_GAMMA_V1,
            CAPABILITY_VELOCITY_DISPERSION_PER_LENS_S2_V1,
        ]
        if declare_population_sigma_unit:
            capabilities.append(CAPABILITY_VELOCITY_DISPERSION_POPULATION_SIGMA_UNIT_V1)
        metadata.create_dataset(
            "capabilities",
            data=np.asarray(capabilities, dtype=object),
            dtype=string_dtype,
        )

        lenses = handle.create_group("lenses")
        lenses.create_dataset("lens_id", data=np.asarray(["lens-0"], dtype=object), dtype=string_dtype)
        lenses.create_dataset("z_d", data=np.asarray([0.55], dtype=np.float64))
        lenses.create_dataset("z_s", data=np.asarray([1.75], dtype=np.float64))
        lenses.create_dataset("log_mstar_obs", data=np.asarray([10.99], dtype=np.float64))
        lenses.create_dataset("log_mstar_err", data=np.asarray([0.05], dtype=np.float64))
        lenses.create_dataset("log_re_obs", data=np.asarray([0.64], dtype=np.float64))
        lenses.create_dataset("n_obs", data=np.asarray([4.2], dtype=np.float64))
        lenses.create_dataset("theta_e_obs", data=np.asarray([1.3], dtype=np.float64))
        lenses.create_dataset("num_sigma", data=np.asarray([num_sigma], dtype=np.int64))
        lenses.create_dataset("sigma_obs", data=np.asarray([[320000.0, 0.0]], dtype=np.float64))
        lenses.create_dataset("sigma_err", data=np.asarray([[20000.0, 1.0]], dtype=np.float64))

        mass_grids = handle.create_group("lensing_mass_grids")
        mass_grids.create_dataset("gamma_grid", data=gamma_grid[None, :])
        mass_grids.create_dataset(
            "log_enclosed_mass_grid",
            data=np.asarray([[11.7, 11.3, 10.9]], dtype=np.float64),
        )
        mass_grids.create_dataset(
            "dmass_dthetaein_grid",
            data=np.asarray([[-2.0, -1.5, -1.0]], dtype=np.float64),
        )
        mass_grids.create_dataset("s2_grid", data=np.asarray([[0.8, 1.0, 1.2]], dtype=np.float64))
        mass_grids.create_dataset("has_s2", data=np.asarray([1 if has_s2 else 0], dtype=np.int64))

        cross_section = handle.create_group("lensing_cross_section")
        cross_section.create_dataset("theta_e_axis", data=theta_axis)
        cross_section.create_dataset("gamma_axis", data=gamma_grid)
        cross_section.create_dataset(
            "cross_section_grid",
            data=np.pi * (theta_axis[:, None] * np.asarray([0.6, 1.0, 1.4])[None, :]) ** 2,
        )
        cross_section.attrs["boundary_policy"] = "zero_outside_theta_clip_gamma"

        velocity = handle.create_group("velocity_dispersion_grids")
        per_lens_s2 = velocity.create_group("per_lens_s2")
        per_lens_s2.create_dataset("s2_grid", data=np.asarray([[0.8, 1.0, 1.2]], dtype=np.float64))
        per_lens_s2.create_dataset("has_s2", data=np.asarray([1 if has_s2 else 0], dtype=np.int64))
        if write_population_sigma_unit:
            population_sigma = velocity.create_group("population_sigma_unit")
            population_sigma.create_dataset("gamma_axis", data=gamma_grid)
            population_sigma.create_dataset("zd_axis", data=np.asarray([0.2, 0.7], dtype=np.float64))
            population_sigma.create_dataset("log_re_kpc_axis", data=np.asarray([0.1, 0.6], dtype=np.float64))
            population_sigma.create_dataset("n_axis", data=np.asarray([3.0, 5.0], dtype=np.float64))
            shape = (3, 2, 2) if malformed_population_sigma_unit else (3, 2, 2, 2)
            population_sigma.create_dataset("s_unit_grid", data=np.ones(shape, dtype=np.float64))

    return path


def test_load_canonical_inference_dataset_reads_schema_blocks(tmp_path: Path) -> None:
    """The reader should expose typed arrays from the canonical HDF5 blocks."""

    dataset_path = _write_canonical_dataset(tmp_path / "canonical.hdf5")

    dataset = load_canonical_inference_dataset(
        dataset_path,
        expected_unit_convention="h_units_v1",
        expected_h_ref=0.7,
        expected_profile_name="sersic",
        expected_mass_definition_label="m5_hinvkpc",
        required_capabilities=(
            CAPABILITY_LENS_OBSERVATIONS_V1,
            CAPABILITY_LENSING_MASS_GRIDS_V1,
        ),
    )

    assert dataset.metadata.schema_version == "canonical_inference_dataset_v1"
    assert dataset.metadata.capabilities >= {
        CAPABILITY_LENS_OBSERVATIONS_V1,
        CAPABILITY_LENSING_MASS_GRIDS_V1,
    }
    np.testing.assert_allclose(dataset.lenses.log_mstar_obs, np.asarray([10.99]))
    assert dataset.cross_section.cross_section_grid.shape == (3, 3)


def test_load_canonical_inference_dataset_exposes_observation_contract_metadata(tmp_path: Path) -> None:
    """Reader metadata should expose canonical observed-aperture contract attrs."""

    dataset_path = _write_canonical_dataset(tmp_path / "neutral_name.hdf5")
    with h5py.File(dataset_path, "a") as handle:
        metadata = handle["metadata"].attrs
        metadata["observation_flavor"] = "boss"
        metadata["sigma_definition"] = "observed_aperture"
        metadata["aperture_shape"] = "circular"
        metadata["aperture_radius_arcsec"] = 1.0
        metadata["seeing_fwhm_arcsec"] = 1.5

    dataset = load_canonical_inference_dataset(
        dataset_path,
        expected_unit_convention="h_units_v1",
        expected_h_ref=0.7,
        expected_profile_name="sersic",
        expected_mass_definition_label="m5_hinvkpc",
        required_capabilities=(
            CAPABILITY_LENS_OBSERVATIONS_V1,
            CAPABILITY_LENSING_MASS_GRIDS_V1,
            CAPABILITY_LENSING_CROSS_SECTION_THETA_GAMMA_V1,
        ),
    )

    assert dataset.metadata.observation_flavor == "boss"
    assert dataset.metadata.sigma_definition == "observed_aperture"
    assert dataset.metadata.aperture_shape == "circular"
    assert dataset.metadata.aperture_radius_arcsec == pytest.approx(1.0)
    assert dataset.metadata.seeing_fwhm_arcsec == pytest.approx(1.5)


def test_load_canonical_inference_dataset_rejects_missing_required_capability(tmp_path: Path) -> None:
    """Model capability checks should fail before runtime context construction."""

    dataset_path = _write_canonical_dataset(tmp_path / "canonical.hdf5")

    with pytest.raises(ValueError, match="missing required capabilities"):
        load_canonical_inference_dataset(
            dataset_path,
            expected_unit_convention="h_units_v1",
            expected_h_ref=0.7,
            expected_profile_name="sersic",
            expected_mass_definition_label="m5_hinvkpc",
            required_capabilities=("not.present.v1",),
        )


def test_load_canonical_inference_dataset_rejects_sigma_lens_without_s2(tmp_path: Path) -> None:
    """A sigma-bearing lens without per-lens S2 is invalid canonical input."""

    dataset_path = _write_canonical_dataset(tmp_path / "bad_canonical.hdf5", has_s2=False, num_sigma=1)

    with pytest.raises(ValueError, match="num_sigma.*has_s2"):
        load_canonical_inference_dataset(
            dataset_path,
            expected_unit_convention="h_units_v1",
            expected_h_ref=0.7,
            expected_profile_name="sersic",
            expected_mass_definition_label="m5_hinvkpc",
            required_capabilities=(CAPABILITY_VELOCITY_DISPERSION_PER_LENS_S2_V1,),
        )


def test_load_canonical_inference_dataset_reads_population_sigma_unit_block(tmp_path: Path) -> None:
    """The reader should expose Sonnenfeld's normalization sigma proxy grid."""

    dataset_path = _write_canonical_dataset(
        tmp_path / "canonical_with_population_sigma.hdf5",
        declare_population_sigma_unit=True,
        write_population_sigma_unit=True,
    )

    dataset = load_canonical_inference_dataset(
        dataset_path,
        expected_unit_convention="h_units_v1",
        expected_h_ref=0.7,
        expected_profile_name="sersic",
        expected_mass_definition_label="m5_hinvkpc",
        required_capabilities=(CAPABILITY_VELOCITY_DISPERSION_POPULATION_SIGMA_UNIT_V1,),
    )

    sigma_grid = dataset.velocity_dispersion.population_sigma_unit
    assert sigma_grid is not None
    np.testing.assert_allclose(sigma_grid.gamma_axis, np.asarray([1.3, 2.0, 2.7]))
    np.testing.assert_allclose(sigma_grid.zd_axis, np.asarray([0.2, 0.7]))
    np.testing.assert_allclose(sigma_grid.log_re_axis, np.asarray([0.1, 0.6]))
    np.testing.assert_allclose(sigma_grid.n_axis, np.asarray([3.0, 5.0]))
    assert sigma_grid.sigma_unit_grid.shape == (3, 2, 2, 2)


def test_load_canonical_inference_dataset_rejects_declared_population_sigma_without_block(
    tmp_path: Path,
) -> None:
    """A declared population-sigma capability must have a matching HDF5 block."""

    dataset_path = _write_canonical_dataset(
        tmp_path / "missing_population_sigma_block.hdf5",
        declare_population_sigma_unit=True,
        write_population_sigma_unit=False,
    )

    with pytest.raises(ValueError, match="population_sigma_unit"):
        load_canonical_inference_dataset(
            dataset_path,
            expected_unit_convention="h_units_v1",
            expected_h_ref=0.7,
            expected_profile_name="sersic",
            expected_mass_definition_label="m5_hinvkpc",
            required_capabilities=(CAPABILITY_VELOCITY_DISPERSION_POPULATION_SIGMA_UNIT_V1,),
        )


def test_load_canonical_inference_dataset_rejects_population_sigma_shape_mismatch(
    tmp_path: Path,
) -> None:
    """Population sigma grids must match gamma, zd, log-Re, and n axes."""

    dataset_path = _write_canonical_dataset(
        tmp_path / "bad_population_sigma_shape.hdf5",
        declare_population_sigma_unit=True,
        write_population_sigma_unit=True,
        malformed_population_sigma_unit=True,
    )

    with pytest.raises(ValueError, match="population_sigma_unit.*s_unit_grid"):
        load_canonical_inference_dataset(
            dataset_path,
            expected_unit_convention="h_units_v1",
            expected_h_ref=0.7,
            expected_profile_name="sersic",
            expected_mass_definition_label="m5_hinvkpc",
            required_capabilities=(CAPABILITY_VELOCITY_DISPERSION_POPULATION_SIGMA_UNIT_V1,),
        )
