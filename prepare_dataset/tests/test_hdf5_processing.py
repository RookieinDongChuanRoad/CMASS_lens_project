"""Tests for HDF5 batch processing and safe-write behavior.

These tests lock down the raw-observation rebuilding contract:

- `gamma_grid` remains the only root-level grid contract
- mass- and sigma-dependent grids live only under `mass_definitions`
- `s2_grid` eligibility follows `num_sigma`, not the legacy root dataset
- malformed sigma-related metadata must fail fast instead of being skipped
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from prepare_dataset.config import (
    DEFAULT_PRODUCTION_APERTURE_POLICY,
    DERIVATIVE_DATASET_NAME,
    GAMMA_DATASET_NAME,
    H_UNITS_V1,
    S2_DATASET_NAME,
)
from prepare_dataset.io.hdf5 import process_hdf5_file, resolve_group_aperture_policy
from prepare_dataset.models import AperturePolicy


def _create_sample_input(path: Path) -> None:
    """Create a compact HDF5 fixture that exercises update/skip behavior.

    The fixture intentionally starts from a legacy-shaped file:

    - every group carries root-level `m5_grid` and `dm5_dthetaein_grid`
    - sigma-eligible groups may also carry a legacy root-level `s2_grid`
    - the rebuilt output must delete all three legacy root datasets while
      preserving `gamma_grid` and populating `mass_definitions/...`
    """

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
        with_s2.attrs["num_sigma"] = 1
        with_s2.create_dataset(GAMMA_DATASET_NAME, data=np.linspace(1.2, 2.8, 17))
        with_s2.create_dataset("m5_grid", data=np.zeros(17))
        with_s2.create_dataset(DERIVATIVE_DATASET_NAME, data=np.zeros(17))
        with_s2.create_dataset(S2_DATASET_NAME, data=np.zeros(17))

        needs_s2 = handle.create_group("needs_s2")
        needs_s2.attrs["zd"] = 0.599
        needs_s2.attrs["zs"] = 1.763
        needs_s2.attrs["sigma_crit"] = 2_223_801_018.8799353
        needs_s2.attrs["rein_arcsec"] = 0.929
        needs_s2.attrs["r_ein_kpc"] = 6.4
        needs_s2.attrs["re_arcsec"] = 0.706
        needs_s2.attrs["nser"] = 4.589
        needs_s2.attrs["num_sigma"] = 1
        needs_s2.create_dataset(GAMMA_DATASET_NAME, data=np.linspace(1.2, 2.8, 17))
        needs_s2.create_dataset("m5_grid", data=np.zeros(17))
        needs_s2.create_dataset(DERIVATIVE_DATASET_NAME, data=np.zeros(17))

        no_sigma = handle.create_group("no_sigma")
        no_sigma.attrs["zd"] = 0.599
        no_sigma.attrs["zs"] = 1.763
        no_sigma.attrs["sigma_crit"] = 2_223_801_018.8799353
        no_sigma.attrs["rein_arcsec"] = 0.929
        no_sigma.attrs["r_ein_kpc"] = 6.4
        no_sigma.attrs["re_arcsec"] = 0.706
        no_sigma.attrs["nser"] = 4.589
        no_sigma.attrs["num_sigma"] = 0
        no_sigma.create_dataset(GAMMA_DATASET_NAME, data=np.linspace(1.2, 2.8, 17))
        no_sigma.create_dataset("m5_grid", data=np.zeros(17))
        no_sigma.create_dataset(DERIVATIVE_DATASET_NAME, data=np.zeros(17))
        no_sigma.create_dataset(S2_DATASET_NAME, data=np.zeros(17))


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
    assert summary.total_groups == 3
    assert summary.updated_m5 == 3
    assert summary.updated_dm5 == 3
    assert summary.updated_s2 == 2
    assert not summary.failures

    with h5py.File(output_path, "r") as handle:
        for group_name in ("with_s2", "needs_s2", "no_sigma"):
            assert GAMMA_DATASET_NAME in handle[group_name]
            assert "m5_grid" not in handle[group_name]
            assert DERIVATIVE_DATASET_NAME not in handle[group_name]
            assert S2_DATASET_NAME not in handle[group_name]

        assert "mass_definitions" in handle["with_s2"]
        assert set(handle["with_s2"]["mass_definitions"].keys()) == {"m5", "m10"}
        assert np.any(handle["with_s2"]["mass_definitions"]["m5"]["mass_grid"][:] != 0.0)
        assert np.any(handle["with_s2"]["mass_definitions"]["m10"]["mass_grid"][:] != 0.0)
        assert np.any(handle["with_s2"]["mass_definitions"]["m5"]["dmass_dthetaein_grid"][:] != 0.0)
        assert np.any(handle["with_s2"]["mass_definitions"]["m10"]["dmass_dthetaein_grid"][:] != 0.0)
        assert "s2_grid" in handle["with_s2"]["mass_definitions"]["m5"]
        assert "s2_grid" in handle["with_s2"]["mass_definitions"]["m10"]
        assert "s2_grid" in handle["needs_s2"]["mass_definitions"]["m5"]
        assert "s2_grid" in handle["needs_s2"]["mass_definitions"]["m10"]
        assert "s2_grid" not in handle["no_sigma"]["mass_definitions"]["m5"]
        assert "s2_grid" not in handle["no_sigma"]["mass_definitions"]["m10"]


def test_process_hdf5_file_can_write_h_unit_mass_definitions_and_metadata(tmp_path: Path) -> None:
    """h-units rebuilds should write only h-unit mass groups and explicit metadata."""

    input_path = tmp_path / "input_h_units.hdf5"
    output_path = tmp_path / "output_h_units.hdf5"
    _create_sample_input(input_path)
    with h5py.File(input_path, "r+") as handle:
        group = handle["no_sigma"]
        group.attrs["logmchab"] = 11.4
        group.attrs["logmchab_deV"] = 11.5
        group.attrs["reff_deV"] = 0.8
        stale_mass_root = group.require_group("mass_definitions")
        stale_mass_root.create_group("m5")
        stale_mass_root.create_group("m10")

    summary = process_hdf5_file(
        input_path=input_path,
        output_path=output_path,
        overwrite_in_place=False,
        group_names=("no_sigma",),
        unit_convention=H_UNITS_V1,
        h_ref=0.7,
    )

    assert summary.total_groups == 1
    assert summary.updated_m5 == 1
    assert summary.updated_dm5 == 1
    assert summary.updated_s2 == 0
    assert not summary.failures

    with h5py.File(output_path, "r") as handle:
        assert handle.attrs["unit_convention"] == H_UNITS_V1
        group = handle["no_sigma"]
        assert group.attrs["unit_convention"] == H_UNITS_V1
        np.testing.assert_allclose(group.attrs["logmchab_h2"], 11.4 + 2.0 * np.log10(0.7))
        np.testing.assert_allclose(group.attrs["r_ein_hinv_kpc"], group.attrs["r_ein_kpc"] * 0.7)
        assert "log10_re_hinv_kpc" in group.attrs
        assert "log10_reff_deV_hinv_kpc" in group.attrs
        assert set(group["mass_definitions"].keys()) == {"m5_hinvkpc", "m10_hinvkpc"}
        m5_group = group["mass_definitions"]["m5_hinvkpc"]
        assert m5_group.attrs["unit_convention"] == H_UNITS_V1
        assert m5_group.attrs["mass_h_power"] == -1
        assert m5_group.attrs["aperture_h_power"] == -1
        assert m5_group.attrs["units"] == "km2 s-2 per 10**m5_hinvkpc"
        assert "mass_grid" in m5_group
        assert "dmass_dthetaein_grid" in m5_group


def test_process_hdf5_file_can_limit_work_to_selected_groups(tmp_path: Path) -> None:
    """Group filtering should support cheap single-galaxy debugging runs."""

    input_path = tmp_path / "input.hdf5"
    output_path = tmp_path / "output.hdf5"
    _create_sample_input(input_path)

    summary = process_hdf5_file(
        input_path=input_path,
        output_path=output_path,
        overwrite_in_place=False,
        group_names=("needs_s2",),
    )

    assert summary.total_groups == 1
    assert summary.updated_m5 == 1
    assert summary.updated_dm5 == 1
    assert summary.updated_s2 == 1

    with h5py.File(output_path, "r") as handle:
        assert np.all(handle["with_s2"]["m5_grid"][:] == 0.0)
        assert "mass_definitions" not in handle["with_s2"]
        assert "mass_definitions" in handle["needs_s2"]
        assert "m5_grid" not in handle["needs_s2"]
        assert DERIVATIVE_DATASET_NAME not in handle["needs_s2"]
        assert S2_DATASET_NAME not in handle["needs_s2"]
        assert S2_DATASET_NAME in handle["needs_s2"]["mass_definitions"]["m5"]
        assert S2_DATASET_NAME in handle["needs_s2"]["mass_definitions"]["m10"]


def test_process_hdf5_file_records_failure_for_invalid_num_sigma(tmp_path: Path) -> None:
    """Unsupported `num_sigma` values must fail instead of silently skipping sigma."""

    input_path = tmp_path / "input_invalid_num_sigma.hdf5"
    output_path = tmp_path / "output_invalid_num_sigma.hdf5"
    _create_sample_input(input_path)

    with h5py.File(input_path, "r+") as handle:
        handle["needs_s2"].attrs["num_sigma"] = 3

    summary = process_hdf5_file(
        input_path=input_path,
        output_path=output_path,
        overwrite_in_place=False,
    )

    assert summary.updated_m5 == 2
    assert summary.updated_dm5 == 2
    assert summary.updated_s2 == 1
    assert summary.failures == ["needs_s2: needs_s2 has unsupported num_sigma=3"]


def test_process_hdf5_file_records_failure_when_sigma_group_lacks_required_geometry(tmp_path: Path) -> None:
    """Sigma-eligible groups must fail fast when `compute_s2_grid()` lacks inputs."""

    input_path = tmp_path / "input_missing_geometry.hdf5"
    output_path = tmp_path / "output_missing_geometry.hdf5"
    _create_sample_input(input_path)

    with h5py.File(input_path, "r+") as handle:
        del handle["needs_s2"].attrs["re_arcsec"]

    summary = process_hdf5_file(
        input_path=input_path,
        output_path=output_path,
        overwrite_in_place=False,
    )

    assert summary.updated_m5 == 3
    assert summary.updated_dm5 == 3
    assert summary.updated_s2 == 1
    assert summary.failures == ["needs_s2: needs_s2 is missing re_arcsec or nser for the Sersic branch"]


def test_resolve_group_aperture_policy_returns_explicit_rectangular_policy(tmp_path: Path) -> None:
    """Complete modern rectangular attrs should resolve to an explicit policy."""

    input_path = tmp_path / "rectangular_policy.hdf5"
    _create_sample_input(input_path)

    with h5py.File(input_path, "r+") as handle:
        group = handle["needs_s2"]
        group.attrs["aperture_shape"] = "rectangular"
        group.attrs["aperture_width_arcsec"] = 1.62
        group.attrs["aperture_height_arcsec"] = 1.0
        group.attrs["aperture_radius_arcsec"] = np.nan
        group.attrs["seeing_fwhm_arcsec"] = 0.6

        policy = resolve_group_aperture_policy(group)

    assert policy == AperturePolicy.rectangular(
        width_arcsec=1.62,
        height_arcsec=1.0,
        seeing_fwhm_arcsec=0.6,
    )


def test_resolve_group_aperture_policy_ignores_legacy_aperture_width_only(tmp_path: Path) -> None:
    """A lone legacy `aperture_width` attr must not become an explicit policy."""

    input_path = tmp_path / "legacy_width_only.hdf5"
    _create_sample_input(input_path)

    with h5py.File(input_path, "r") as handle:
        assert resolve_group_aperture_policy(handle["with_s2"]) is None


def test_resolve_group_aperture_policy_rejects_incomplete_explicit_metadata(tmp_path: Path) -> None:
    """Modern aperture metadata should fail fast when the schema is incomplete."""

    input_path = tmp_path / "incomplete_policy.hdf5"
    _create_sample_input(input_path)

    with h5py.File(input_path, "r+") as handle:
        group = handle["needs_s2"]
        group.attrs["aperture_shape"] = "rectangular"
        group.attrs["aperture_width_arcsec"] = 1.62
        group.attrs["seeing_fwhm_arcsec"] = 0.6

        try:
            resolve_group_aperture_policy(group)
        except ValueError as exc:
            assert "aperture_height_arcsec" in str(exc)
        else:
            raise AssertionError("Expected incomplete rectangular aperture metadata to fail")


def test_process_hdf5_file_prefers_group_explicit_aperture_over_file_default(tmp_path: Path) -> None:
    """Mixed-aperture files should use the explicit group policy when present."""

    input_path = tmp_path / "mixed_aperture_input.hdf5"
    output_path = tmp_path / "mixed_aperture_output.hdf5"
    _create_sample_input(input_path)

    with h5py.File(input_path, "r+") as handle:
        default_group = handle["with_s2"]
        explicit_group = handle["needs_s2"]
        for attr_name in ("zd", "zs", "sigma_crit", "rein_arcsec", "r_ein_kpc", "re_arcsec", "nser", "num_sigma"):
            explicit_group.attrs[attr_name] = default_group.attrs[attr_name]
        explicit_group.attrs["aperture_shape"] = "rectangular"
        explicit_group.attrs["aperture_width_arcsec"] = 1.62
        explicit_group.attrs["aperture_height_arcsec"] = 1.0
        explicit_group.attrs["seeing_fwhm_arcsec"] = 0.6
        explicit_group.attrs["aperture_radius_arcsec"] = np.nan

    summary = process_hdf5_file(
        input_path=input_path,
        output_path=output_path,
        overwrite_in_place=False,
        group_names=("with_s2", "needs_s2"),
        aperture_policy=DEFAULT_PRODUCTION_APERTURE_POLICY,
    )

    assert summary.total_groups == 2
    assert summary.updated_s2 == 2
    assert not summary.failures

    with h5py.File(output_path, "r") as handle:
        default_s2 = handle["with_s2"]["mass_definitions"]["m5"]["s2_grid"][:]
        explicit_s2 = handle["needs_s2"]["mass_definitions"]["m5"]["s2_grid"][:]

    assert not np.allclose(default_s2, explicit_s2)
