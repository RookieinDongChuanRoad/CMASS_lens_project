"""Integration tests for the slit-canonical sigma/aperture sync workflow.

The new workflow combines two previously separate operations:

- update existing slit sigma attrs from the PPXF CSV export
- merge SL2S sigma plus explicit aperture metadata into the two canonical slit files

The tests below lock the high-level contract before implementation code lands.
"""

from __future__ import annotations

import csv
from pathlib import Path

import h5py
import numpy as np

from prepare_dataset.io.slit_observation_updates import (
    plan_slit_canonical_updates,
    sync_slit_canonical_updates,
)


def _write_ppxf_csv(path: Path) -> None:
    """Create one compact CSV fixture using the real PPXF export columns."""

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "system",
                "base_name",
                "obs_tag",
                "sigma_primary_kms",
                "sigma_stat_kms",
                "sigma_total_kms",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "system": "csv-group",
                "base_name": "csv-group",
                "obs_tag": "",
                "sigma_primary_kms": 225.869123,
                "sigma_stat_kms": 9.587872,
                "sigma_total_kms": 31.25,
            }
        )


def _write_target_file(path: Path) -> None:
    """Create one canonical slit-like file with one CSV group and one SL2S target."""

    with h5py.File(path, "w") as handle:
        csv_group = handle.create_group("csv-group")
        csv_group.attrs["zd"] = 0.599
        csv_group.attrs["zs"] = 1.763
        csv_group.attrs["sigma_crit"] = 2_223_801_018.8799353
        csv_group.attrs["rein_arcsec"] = 0.929
        csv_group.attrs["r_ein_kpc"] = 6.4
        csv_group.attrs["re_arcsec"] = 0.706
        csv_group.attrs["reff_deV"] = 0.62
        csv_group.attrs["nser"] = 4.589
        csv_group.attrs["num_sigma"] = 1
        csv_group.attrs["sigma"] = np.asarray([219], dtype=np.int64)
        csv_group.attrs["sigma_err"] = np.asarray([10], dtype=np.int64)
        csv_group.create_dataset("gamma_grid", data=np.linspace(1.2, 2.8, 5))
        csv_mass = csv_group.create_group("mass_definitions")
        for label in ("m5", "m10"):
            subgroup = csv_mass.create_group(label)
            subgroup.create_dataset("mass_grid", data=np.linspace(10.0, 11.0, 5))
            subgroup.create_dataset("dmass_dthetaein_grid", data=np.linspace(1.0, 2.0, 5))
            subgroup.create_dataset("s2_grid", data=np.linspace(3.0, 4.0, 5))

        sl2s_group = handle.create_group("sl2s-group")
        sl2s_group.attrs["zd"] = 0.58
        sl2s_group.attrs["zs"] = 1.9
        sl2s_group.attrs["sigma_crit"] = 2_223_801_018.8799353
        sl2s_group.attrs["rein_arcsec"] = 1.0
        sl2s_group.attrs["r_ein_kpc"] = 6.8
        sl2s_group.attrs["re_arcsec"] = 0.9
        sl2s_group.attrs["reff_deV"] = 1.4
        sl2s_group.attrs["nser"] = 3.1
        sl2s_group.attrs["num_sigma"] = 0
        sl2s_group.create_dataset("gamma_grid", data=np.linspace(1.2, 2.8, 5))
        sl2s_mass = sl2s_group.create_group("mass_definitions")
        for label in ("m5", "m10"):
            subgroup = sl2s_mass.create_group(label)
            subgroup.create_dataset("mass_grid", data=np.linspace(10.2, 11.2, 5))
            subgroup.create_dataset("dmass_dthetaein_grid", data=np.linspace(1.1, 2.1, 5))


def _write_sl2s_source(path: Path) -> None:
    """Create one SL2S source file carrying explicit per-group aperture metadata."""

    with h5py.File(path, "w") as handle:
        group = handle.create_group("sl2s-group")
        group.attrs["num_sigma"] = 1
        group.attrs["sigma"] = np.asarray([209], dtype=np.int64)
        group.attrs["sigma_err"] = np.asarray([20], dtype=np.int64)
        group.attrs["aperture_shape"] = "rectangular"
        group.attrs["aperture_width_arcsec"] = 1.6
        group.attrs["aperture_height_arcsec"] = 0.9
        group.attrs["aperture_radius_arcsec"] = np.nan
        group.attrs["seeing_fwhm_arcsec"] = 0.7


def test_plan_slit_canonical_updates_reports_csv_and_sl2s_group_changes(tmp_path: Path) -> None:
    """Preview mode should expose both CSV updates and SL2S merge targets."""

    csv_path = tmp_path / "ppxf_results_optimal.csv"
    devauc_path = tmp_path / "observations_deV_with_mass_grids.hdf5"
    sersic_path = tmp_path / "observations_with_mass_grids_all.hdf5"
    sl2s_path = tmp_path / "observations_deV_with_SL2S_mass_grids.hdf5"
    _write_ppxf_csv(csv_path)
    _write_target_file(devauc_path)
    _write_target_file(sersic_path)
    _write_sl2s_source(sl2s_path)

    plans = plan_slit_canonical_updates(
        csv_path=csv_path,
        slit_hdf5_paths=[devauc_path, sersic_path],
        sl2s_source_path=sl2s_path,
    )

    assert [plan.input_path.name for plan in plans] == [
        "observations_deV_with_mass_grids.hdf5",
        "observations_with_mass_grids_all.hdf5",
    ]
    for plan in plans:
        assert [update.group_name for update in plan.csv_group_updates] == ["csv-group"]
        assert [update.group_name for update in plan.sl2s_group_updates] == ["sl2s-group"]


def test_sync_slit_canonical_updates_writes_sigma_attrs_and_rebuilds_sl2s_s2_grids(tmp_path: Path) -> None:
    """The workflow should update CSV sigma attrs and merge SL2S aperture-aware grids."""

    csv_path = tmp_path / "ppxf_results_optimal.csv"
    devauc_path = tmp_path / "observations_deV_with_mass_grids.hdf5"
    sersic_path = tmp_path / "observations_with_mass_grids_all.hdf5"
    sl2s_path = tmp_path / "observations_deV_with_SL2S_mass_grids.hdf5"
    _write_ppxf_csv(csv_path)
    _write_target_file(devauc_path)
    _write_target_file(sersic_path)
    _write_sl2s_source(sl2s_path)

    with h5py.File(devauc_path, "r") as handle:
        original_csv_s2 = handle["csv-group"]["mass_definitions"]["m5"]["s2_grid"][:]

    results = sync_slit_canonical_updates(
        csv_path=csv_path,
        slit_hdf5_paths=[devauc_path, sersic_path],
        sl2s_source_path=sl2s_path,
        overwrite_in_place=True,
    )

    assert [result.input_path.name for result in results] == [
        "observations_deV_with_mass_grids.hdf5",
        "observations_with_mass_grids_all.hdf5",
    ]
    for result in results:
        assert result.was_written is True
        assert result.rebuilt_group_names == ["sl2s-group"]

    for output_path in (devauc_path, sersic_path):
        with h5py.File(output_path, "r") as handle:
            assert handle["csv-group"].attrs["sigma"].tolist() == [226]
            assert handle["csv-group"].attrs["sigma_err"].tolist() == [10]
            assert np.array_equal(
                handle["csv-group"]["mass_definitions"]["m5"]["s2_grid"][:],
                original_csv_s2,
            )

            sl2s_group = handle["sl2s-group"]
            assert int(sl2s_group.attrs["num_sigma"]) == 1
            assert sl2s_group.attrs["sigma"].tolist() == [209]
            assert sl2s_group.attrs["sigma_err"].tolist() == [20]
            assert sl2s_group.attrs["aperture_shape"] == "rectangular"
            assert float(sl2s_group.attrs["aperture_width_arcsec"]) == 1.6
            assert float(sl2s_group.attrs["aperture_height_arcsec"]) == 0.9
            assert float(sl2s_group.attrs["seeing_fwhm_arcsec"]) == 0.7
            assert "s2_grid" in sl2s_group["mass_definitions"]["m5"]
            assert "s2_grid" in sl2s_group["mass_definitions"]["m10"]
