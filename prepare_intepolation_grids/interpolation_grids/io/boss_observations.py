"""Builder for the BOSS-specific raw observation HDF5 files.

This module creates the new observation products from the summary table rather
than cloning older HDF5 files. That matters because the BOSS files must use a
different aperture geometry, and reusing legacy files would silently preserve
physics choices that no longer apply.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import astropy.units as u
import h5py
import numpy as np
from astropy.constants import G, c

from interpolation_grids.config import (
    BOSS_CIRCULAR_APERTURE_POLICY,
    BOSS_OUTPUT_FILENAMES,
    BOSS_SUMMARY_FILENAME,
    GAMMA_GRID,
    RAW_DATA_DIRECTORY,
    S2_DATASET_NAME,
)
from interpolation_grids.io.flatprior_measurement_updates import update_flatprior_measurement_attrs_in_hdf5
from interpolation_grids.io.hdf5 import process_hdf5_file
from interpolation_grids.physics.jeans import COSMOLOGY, kpc_per_arcsec, uses_devaucouleurs_branch


REQUIRED_SUMMARY_COLUMNS = (
    "name",
    "zd",
    "zs",
    "rein_arcsec",
    "re_arcsec",
    "nser",
    "logmchab",
    "logmchab_err",
    "sigma",
    "sigma_err",
    "imag_Ser",
    "imag_deV",
    "reff_deV",
    "logmchab_deV",
)


class BossObservationBuildValidationError(ValueError):
    """Raised when the BOSS summary table cannot support a safe rebuild."""


@dataclass(frozen=True)
class BossSummaryRow:
    """Normalized representation of one row from `summary_table_deV.txt`."""

    name: str
    zd: float
    zs: float
    rein_arcsec: float
    re_arcsec: float
    nser: float
    logmchab: float
    logmchab_err: float
    sigma: int
    sigma_err: int
    imag_ser: float
    imag_dev: float
    reff_deV: float
    logmchab_deV: float


def _parse_summary_header(summary_path: Path, lines: list[str]) -> list[str]:
    """Return the commented header columns after validating the required names."""

    for line in lines:
        stripped = line.strip()
        if not stripped.startswith("#"):
            continue
        header = stripped[1:].strip().split()
        missing_columns = sorted(set(REQUIRED_SUMMARY_COLUMNS).difference(header))
        if missing_columns:
            missing_text = ", ".join(missing_columns)
            raise BossObservationBuildValidationError(
                f"{summary_path} is missing required columns: {missing_text}"
            )
        return header

    raise BossObservationBuildValidationError(f"{summary_path} has no commented header row.")


def read_boss_summary_table(summary_path: Path | str) -> list[BossSummaryRow]:
    """Read the project BOSS summary table with strict validation."""

    summary_path = Path(summary_path)
    raw_lines = summary_path.read_text(encoding="utf-8").splitlines()
    header = _parse_summary_header(summary_path, raw_lines)

    rows: list[BossSummaryRow] = []
    seen_names: set[str] = set()
    for line_number, raw_line in enumerate(raw_lines, start=1):
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        values = stripped.split()
        if len(values) != len(header):
            raise BossObservationBuildValidationError(
                f"{summary_path}:{line_number} has {len(values)} columns but the header defines {len(header)}."
            )

        row_map = dict(zip(header, values, strict=True))
        lens_name = row_map["name"].strip()
        if not lens_name:
            raise BossObservationBuildValidationError(f"{summary_path}:{line_number} has an empty lens name.")
        if lens_name in seen_names:
            raise BossObservationBuildValidationError(f"{summary_path} contains a duplicate lens name: {lens_name}")
        seen_names.add(lens_name)

        try:
            rows.append(
                BossSummaryRow(
                    name=lens_name,
                    zd=float(row_map["zd"]),
                    zs=float(row_map["zs"]),
                    rein_arcsec=float(row_map["rein_arcsec"]),
                    re_arcsec=float(row_map["re_arcsec"]),
                    nser=float(row_map["nser"]),
                    logmchab=float(row_map["logmchab"]),
                    logmchab_err=float(row_map["logmchab_err"]),
                    sigma=int(np.rint(float(row_map["sigma"]))),
                    sigma_err=int(np.rint(float(row_map["sigma_err"]))),
                    imag_ser=float(row_map["imag_Ser"]),
                    imag_dev=float(row_map["imag_deV"]),
                    reff_deV=float(row_map["reff_deV"]),
                    logmchab_deV=float(row_map["logmchab_deV"]),
                )
            )
        except (TypeError, ValueError) as exc:
            raise BossObservationBuildValidationError(
                f"{summary_path}:{line_number} contains a non-numeric value in a required numeric column."
            ) from exc

    if not rows:
        raise BossObservationBuildValidationError(f"{summary_path} contains no data rows.")

    return rows


def _sigma_critical_surface_density(zd: float, zs: float) -> float:
    """Compute `Sigma_crit` in `Msun / kpc^2` for one lens-source geometry."""

    angular_diameter_lens = COSMOLOGY.angular_diameter_distance(zd)
    angular_diameter_source = COSMOLOGY.angular_diameter_distance(zs)
    angular_diameter_lens_source = COSMOLOGY.angular_diameter_distance_z1z2(zd, zs)
    sigma_crit = (
        (c**2 / (4.0 * math.pi * G))
        * angular_diameter_source
        / (angular_diameter_lens * angular_diameter_lens_source)
    )
    return float(sigma_crit.to(u.Msun / u.kpc**2).value)


def _log10_sigma_star(log_stellar_mass: float, effective_radius_arcsec: float, zd: float) -> float:
    """Compute the stellar surface-density summary stored by the raw files."""

    effective_radius_kpc = effective_radius_arcsec * kpc_per_arcsec(zd)
    return float(log_stellar_mass - math.log10(2.0 * math.pi * effective_radius_kpc**2))


def _write_common_group_attrs(group_handle: h5py.Group, row: BossSummaryRow) -> None:
    """Write the attrs shared by the Sersic and deV BOSS observation files."""

    group_handle.attrs["zd"] = float(row.zd)
    group_handle.attrs["zs"] = float(row.zs)
    group_handle.attrs["rein_arcsec"] = float(row.rein_arcsec)
    group_handle.attrs["re_arcsec"] = float(row.re_arcsec)
    group_handle.attrs["nser"] = float(row.nser)
    group_handle.attrs["logmchab"] = float(row.logmchab)
    group_handle.attrs["logmchab_err"] = float(row.logmchab_err)
    group_handle.attrs["sigma_crit"] = _sigma_critical_surface_density(row.zd, row.zs)
    group_handle.attrs["r_ein_kpc"] = float(row.rein_arcsec * kpc_per_arcsec(row.zd))
    group_handle.attrs["num_sigma"] = np.int64(1)
    group_handle.attrs["sigma"] = np.asarray([row.sigma], dtype=np.int64)
    group_handle.attrs["sigma_err"] = np.asarray([row.sigma_err], dtype=np.int64)
    group_handle.attrs["aperture_shape"] = BOSS_CIRCULAR_APERTURE_POLICY.shape
    group_handle.attrs["aperture_radius_arcsec"] = float(BOSS_CIRCULAR_APERTURE_POLICY.radius_arcsec)
    group_handle.attrs["seeing_fwhm_arcsec"] = float(BOSS_CIRCULAR_APERTURE_POLICY.seeing_fwhm_arcsec)


def _write_profile_specific_attrs(group_handle: h5py.Group, row: BossSummaryRow, profile_name: str) -> None:
    """Write the attrs that differ between the Sersic and deV branches."""

    normalized_profile = profile_name.strip().lower()
    if normalized_profile == "devauc":
        group_handle.attrs["reff_deV"] = float(row.reff_deV)
        group_handle.attrs["logmchab_deV"] = float(row.logmchab_deV)
        group_handle.attrs["log10_Sigma_star"] = _log10_sigma_star(
            log_stellar_mass=row.logmchab_deV,
            effective_radius_arcsec=row.reff_deV,
            zd=row.zd,
        )
        return

    if normalized_profile == "sersic":
        group_handle.attrs["log10_Sigma_star"] = _log10_sigma_star(
            log_stellar_mass=row.logmchab,
            effective_radius_arcsec=row.re_arcsec,
            zd=row.zd,
        )
        return

    raise BossObservationBuildValidationError(f"Unsupported BOSS profile name: {profile_name}")


def _write_skeleton_observation_file(output_path: Path, rows: list[BossSummaryRow], profile_name: str) -> None:
    """Create the minimal HDF5 structure that the existing processors can enrich."""

    if profile_name == "devauc" and not uses_devaucouleurs_branch(output_path.name):
        raise BossObservationBuildValidationError(
            f"deV output path must follow the established naming convention: {output_path.name}"
        )

    with h5py.File(output_path, "w") as handle:
        for row in rows:
            group = handle.create_group(row.name)
            _write_common_group_attrs(group, row)
            _write_profile_specific_attrs(group, row, profile_name=profile_name)
            group.create_dataset("gamma_grid", data=np.asarray(GAMMA_GRID, dtype=float))
            # The existing HDF5 processor refreshes `s2_grid` only when the
            # dataset already exists. A zero placeholder lets us reuse that
            # tested code path while still building the file from scratch.
            group.create_dataset(S2_DATASET_NAME, data=np.zeros_like(GAMMA_GRID, dtype=float))


def build_boss_observation_hdf5_files(
    summary_path: Path | str | None = None,
    output_directory: Path | str | None = None,
) -> dict[str, Path]:
    """Build the two BOSS raw observation files and return their output paths."""

    resolved_summary_path = Path(summary_path) if summary_path is not None else RAW_DATA_DIRECTORY / BOSS_SUMMARY_FILENAME
    resolved_output_directory = (
        Path(output_directory) if output_directory is not None else RAW_DATA_DIRECTORY
    ).expanduser().resolve()
    resolved_output_directory.mkdir(parents=True, exist_ok=True)

    rows = read_boss_summary_table(resolved_summary_path)
    output_paths = {
        profile_name: resolved_output_directory / filename
        for profile_name, filename in BOSS_OUTPUT_FILENAMES.items()
    }

    _write_skeleton_observation_file(output_paths["sersic"], rows=rows, profile_name="sersic")
    _write_skeleton_observation_file(output_paths["devauc"], rows=rows, profile_name="devauc")

    for output_path in output_paths.values():
        process_hdf5_file(
            input_path=output_path,
            output_path=output_path,
            overwrite_in_place=True,
            aperture_policy=BOSS_CIRCULAR_APERTURE_POLICY,
        )

    update_flatprior_measurement_attrs_in_hdf5(
        hdf5_paths=output_paths.values(),
        overwrite_in_place=True,
        create_backup=False,
    )

    return output_paths


__all__ = [
    "BossObservationBuildValidationError",
    "BossSummaryRow",
    "build_boss_observation_hdf5_files",
    "read_boss_summary_table",
]
