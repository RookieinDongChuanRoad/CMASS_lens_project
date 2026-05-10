"""Build Sonnenfeld/SLACS fixed-kpc observation products from the raw table.

This module owns the paper-faithful SLACS ingestion boundary.  The raw catalog
is a lens-level ASCII table, while the canonical writer expects the same
normalized HDF5 contract used by the rest of this repository: per-lens attrs,
one shared gamma axis, mass-definition subgroups, and per-lens sigma-unit
Jeans responses.  Keeping this conversion in `prepare_dataset` preserves the
separation between data preparation and inference/runtime code.
"""

from __future__ import annotations

import math
import tempfile
from dataclasses import dataclass
from pathlib import Path

import astropy.units as u
import h5py
import numpy as np
from astropy.constants import G, c

from prepare_dataset.config import (
    DEFAULT_DERIVATIVE_THETA_SAMPLES,
    GAMMA_GRID,
    LEGACY_FIXED_KPC,
    OBSERVED_APERTURE_SIGMA_DEFINITION,
    RAW_DATA_DIRECTORY,
    sigma_unit_units_for_radius,
)
from prepare_dataset.io.sigma_tables import build_sigma_unit_table, write_sigma_unit_table_hdf5
from prepare_dataset.models import AperturePolicy
from prepare_dataset.physics.jeans import COSMOLOGY, compute_sigma_unit_grid
from prepare_dataset.physics.m5 import compute_dmass_dthetaein_grid, compute_mass_grid


SLACS_EXPECTED_ROWS = 59
SLACS_EXPECTED_COLUMNS = 13
SLACS_PROFILE_NAME = "devauc"
SLACS_N_OBS = 4.0
SLACS_MASS_DEFINITION_LABEL = "m5"
SLACS_MASS_RADIUS_KPC = 5.0
SLACS_OBSERVATION_APERTURE_POLICY = AperturePolicy.circular(
    radius_arcsec=1.5,
    seeing_fwhm_arcsec=1.5,
)
SLACS_RAW_FILENAME = "SLACS_table.cat"
SLACS_OBSERVATION_FILENAME = "observations_SLACS_deV_with_mass_grids_fixed_m5.hdf5"
SLACS_POPULATION_SIGMA_FILENAME = "slacs_population_sigma_unit_m5.h5"


@dataclass(frozen=True)
class SlacsLensRow:
    """One validated lens row from `SLACS_table.cat`.

    The field names mirror the raw table header so audits can trace every
    stored HDF5 value back to the paper/reference source without hidden column
    remapping.
    """

    name: str
    ra_deg: float
    dec_deg: float
    zd: float
    zs: float
    reff_arcsec: float
    reff_kpc: float
    theta_ein_arcsec: float
    rein_kpc: float
    log_mstar_chab: float
    log_mstar_err: float
    velocity_dispersion: float
    velocity_dispersion_err: float


def _parse_slacs_data_row(catalog_path: Path, line_number: int, raw_line: str) -> SlacsLensRow:
    """Parse and validate one non-comment SLACS table row."""

    values = raw_line.split()
    if len(values) != SLACS_EXPECTED_COLUMNS:
        raise ValueError(
            f"{catalog_path}:{line_number} has {len(values)} columns; "
            f"expected {SLACS_EXPECTED_COLUMNS}."
        )

    name = values[0].strip()
    if not name:
        raise ValueError(f"{catalog_path}:{line_number} has an empty lens name.")

    try:
        row = SlacsLensRow(
            name=name,
            ra_deg=float(values[1]),
            dec_deg=float(values[2]),
            zd=float(values[3]),
            zs=float(values[4]),
            reff_arcsec=float(values[5]),
            reff_kpc=float(values[6]),
            theta_ein_arcsec=float(values[7]),
            rein_kpc=float(values[8]),
            log_mstar_chab=float(values[9]),
            log_mstar_err=float(values[10]),
            velocity_dispersion=float(values[11]),
            velocity_dispersion_err=float(values[12]),
        )
    except ValueError as exc:
        raise ValueError(f"{catalog_path}:{line_number} contains a non-numeric value.") from exc

    _validate_slacs_row(catalog_path, line_number, row)
    return row


def _validate_slacs_row(catalog_path: Path, line_number: int, row: SlacsLensRow) -> None:
    """Validate physical assumptions needed by the fixed-m5 SLACS builder."""

    if row.zs <= row.zd:
        raise ValueError(f"{catalog_path}:{line_number} violates zs > zd for lens {row.name}.")
    positive_fields = {
        "Reff_arcsec": row.reff_arcsec,
        "Reff_kpc": row.reff_kpc,
        "theta_Ein": row.theta_ein_arcsec,
        "Rein_kpc": row.rein_kpc,
        "lMstar_err": row.log_mstar_err,
        "veldisp": row.velocity_dispersion,
        "veldisp_err": row.velocity_dispersion_err,
    }
    for field_name, value in positive_fields.items():
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(
                f"{catalog_path}:{line_number} has non-positive or non-finite {field_name} for lens {row.name}."
            )


def read_slacs_table(
    catalog_path: Path | str,
    *,
    expected_rows: int | None = SLACS_EXPECTED_ROWS,
) -> list[SlacsLensRow]:
    """Read `SLACS_table.cat` with strict row, column, and physics validation."""

    resolved_catalog_path = Path(catalog_path).expanduser().resolve()
    rows: list[SlacsLensRow] = []
    seen_names: set[str] = set()
    for line_number, raw_line in enumerate(resolved_catalog_path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        row = _parse_slacs_data_row(resolved_catalog_path, line_number, stripped)
        if row.name in seen_names:
            raise ValueError(f"{resolved_catalog_path} contains duplicate lens name {row.name}.")
        seen_names.add(row.name)
        rows.append(row)

    if expected_rows is not None and len(rows) != int(expected_rows):
        raise ValueError(f"{resolved_catalog_path} has {len(rows)} rows; expected {expected_rows}.")
    if not rows:
        raise ValueError(f"{resolved_catalog_path} contains no SLACS lens rows.")
    return rows


def _sigma_critical_surface_density(zd: float, zs: float) -> float:
    """Compute critical surface density in `Msun / kpc^2` for one lens-source pair."""

    angular_diameter_lens = COSMOLOGY.angular_diameter_distance(zd)
    angular_diameter_source = COSMOLOGY.angular_diameter_distance(zs)
    angular_diameter_lens_source = COSMOLOGY.angular_diameter_distance_z1z2(zd, zs)
    sigma_crit = (
        (c**2 / (4.0 * math.pi * G))
        * angular_diameter_source
        / (angular_diameter_lens * angular_diameter_lens_source)
    )
    return float(sigma_crit.to(u.Msun / u.kpc**2).value)


def _write_slacs_root_attrs(handle: h5py.File, catalog_path: Path) -> None:
    """Write file-level metadata that makes the raw HDF5 product self-describing."""

    handle.attrs["source_catalog"] = str(catalog_path)
    handle.attrs["profile_name"] = SLACS_PROFILE_NAME
    handle.attrs["n_obs"] = SLACS_N_OBS
    handle.attrs["unit_convention"] = LEGACY_FIXED_KPC
    handle.attrs["mass_definition_label"] = SLACS_MASS_DEFINITION_LABEL
    handle.attrs["mass_radius_kpc"] = SLACS_MASS_RADIUS_KPC
    handle.attrs["aperture_shape"] = SLACS_OBSERVATION_APERTURE_POLICY.shape
    handle.attrs["aperture_radius_arcsec"] = float(SLACS_OBSERVATION_APERTURE_POLICY.radius_arcsec)
    handle.attrs["seeing_fwhm_arcsec"] = float(SLACS_OBSERVATION_APERTURE_POLICY.seeing_fwhm_arcsec)


def _write_slacs_lens_group(
    handle: h5py.File,
    *,
    row: SlacsLensRow,
    gamma_axis: np.ndarray,
    derivative_theta_samples: int,
) -> None:
    """Write one lens group with fixed-m5 mass and Jeans response grids."""

    group = handle.create_group(row.name)
    sigma_crit = _sigma_critical_surface_density(row.zd, row.zs)
    kpc_per_arcsec = row.rein_kpc / row.theta_ein_arcsec

    group.attrs["unit_convention"] = LEGACY_FIXED_KPC
    group.attrs["profile_name"] = SLACS_PROFILE_NAME
    group.attrs["ra_deg"] = float(row.ra_deg)
    group.attrs["dec_deg"] = float(row.dec_deg)
    group.attrs["zd"] = float(row.zd)
    group.attrs["zs"] = float(row.zs)
    group.attrs["rein_arcsec"] = float(row.theta_ein_arcsec)
    group.attrs["r_ein_kpc"] = float(row.rein_kpc)
    group.attrs["re_arcsec"] = float(row.reff_arcsec)
    group.attrs["reff_deV"] = float(row.reff_arcsec)
    group.attrs["reff_deV_kpc"] = float(row.reff_kpc)
    group.attrs["log10_re_kpc"] = float(math.log10(row.reff_kpc))
    group.attrs["log10_reff_deV_kpc"] = float(math.log10(row.reff_kpc))
    group.attrs["nser"] = SLACS_N_OBS
    group.attrs["logmchab"] = float(row.log_mstar_chab)
    group.attrs["logmchab_deV"] = float(row.log_mstar_chab)
    group.attrs["logmchab_err"] = float(row.log_mstar_err)
    group.attrs["sigma_crit"] = sigma_crit
    group.attrs["num_sigma"] = np.int64(1)
    group.attrs["sigma"] = np.asarray([row.velocity_dispersion], dtype=float)
    group.attrs["sigma_err"] = np.asarray([row.velocity_dispersion_err], dtype=float)
    group.attrs["aperture_shape"] = SLACS_OBSERVATION_APERTURE_POLICY.shape
    group.attrs["aperture_radius_arcsec"] = float(SLACS_OBSERVATION_APERTURE_POLICY.radius_arcsec)
    group.attrs["seeing_fwhm_arcsec"] = float(SLACS_OBSERVATION_APERTURE_POLICY.seeing_fwhm_arcsec)
    group.create_dataset("gamma_grid", data=np.asarray(gamma_axis, dtype=float))

    mass_group = group.create_group("mass_definitions").create_group(SLACS_MASS_DEFINITION_LABEL)
    mass_group.attrs["unit_convention"] = LEGACY_FIXED_KPC
    mass_group.attrs["mass_definition_label"] = SLACS_MASS_DEFINITION_LABEL
    mass_group.attrs["mass_radius_kpc"] = SLACS_MASS_RADIUS_KPC
    mass_group.attrs["units"] = sigma_unit_units_for_radius(SLACS_MASS_RADIUS_KPC, LEGACY_FIXED_KPC)
    mass_group.create_dataset(
        "mass_grid",
        data=compute_mass_grid(
            gamma_grid=gamma_axis,
            sigma_crit=sigma_crit,
            rein_kpc=row.rein_kpc,
            mass_radius_kpc=SLACS_MASS_RADIUS_KPC,
            unit_convention=LEGACY_FIXED_KPC,
        ),
    )
    mass_group.create_dataset(
        "dmass_dthetaein_grid",
        data=compute_dmass_dthetaein_grid(
            gamma_grid=gamma_axis,
            sigma_crit=sigma_crit,
            theta_ein_arcsec=row.theta_ein_arcsec,
            kpc_per_arcsec=kpc_per_arcsec,
            theta_samples=derivative_theta_samples,
            mass_radius_kpc=SLACS_MASS_RADIUS_KPC,
        ),
    )
    mass_group.create_dataset(
        "s2_grid",
        data=compute_sigma_unit_grid(
            profile_name=SLACS_PROFILE_NAME,
            gamma_grid=gamma_axis,
            zd=row.zd,
            re_kpc=row.reff_kpc,
            mass_radius_kpc=SLACS_MASS_RADIUS_KPC,
            aperture_policy=SLACS_OBSERVATION_APERTURE_POLICY,
            sigma_definition=OBSERVED_APERTURE_SIGMA_DEFINITION,
            unit_convention=LEGACY_FIXED_KPC,
        ),
    )


def write_slacs_observation_hdf5(
    *,
    catalog_path: Path | str = RAW_DATA_DIRECTORY / SLACS_RAW_FILENAME,
    output_path: Path | str = RAW_DATA_DIRECTORY / SLACS_OBSERVATION_FILENAME,
    gamma_axis: np.ndarray | None = None,
    derivative_theta_samples: int = DEFAULT_DERIVATIVE_THETA_SAMPLES,
    expected_rows: int | None = SLACS_EXPECTED_ROWS,
    overwrite: bool = False,
) -> Path:
    """Convert the raw SLACS table into the HDF5 contract consumed by the writer."""

    resolved_catalog_path = Path(catalog_path).expanduser().resolve()
    resolved_output_path = Path(output_path).expanduser().resolve()
    if resolved_output_path.exists() and not overwrite:
        raise FileExistsError(f"{resolved_output_path} already exists. Pass overwrite=True to replace it.")
    if int(derivative_theta_samples) <= 1:
        raise ValueError("derivative_theta_samples must be greater than 1.")

    rows = read_slacs_table(resolved_catalog_path, expected_rows=expected_rows)
    resolved_gamma_axis = np.asarray(GAMMA_GRID if gamma_axis is None else gamma_axis, dtype=float)
    if resolved_gamma_axis.ndim != 1 or resolved_gamma_axis.size == 0:
        raise ValueError("gamma_axis must be a non-empty one-dimensional array.")

    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{resolved_output_path.name}.",
            suffix=".tmp",
            dir=resolved_output_path.parent,
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)

        with h5py.File(temporary_path, "w") as handle:
            _write_slacs_root_attrs(handle, resolved_catalog_path)
            for row in rows:
                _write_slacs_lens_group(
                    handle,
                    row=row,
                    gamma_axis=resolved_gamma_axis,
                    derivative_theta_samples=int(derivative_theta_samples),
                )

        temporary_path.replace(resolved_output_path)
        return resolved_output_path
    except Exception:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
        raise


def write_slacs_population_sigma_unit_hdf5(
    *,
    output_path: Path | str,
    workers: int | None = None,
    overwrite: bool = False,
) -> Path:
    """Build the SLACS population sigma-unit table used by Sonnenfeld selection."""

    resolved_output_path = Path(output_path).expanduser().resolve()
    if resolved_output_path.exists() and not overwrite:
        raise FileExistsError(f"{resolved_output_path} already exists. Pass overwrite=True to replace it.")
    table = build_sigma_unit_table(
        profile_name=SLACS_PROFILE_NAME,
        mass_radius_kpc=SLACS_MASS_RADIUS_KPC,
        workers=workers,
        observation_flavor="slit",
        aperture_policy=SLACS_OBSERVATION_APERTURE_POLICY,
        unit_convention=LEGACY_FIXED_KPC,
    )
    return write_sigma_unit_table_hdf5(table, resolved_output_path)


__all__ = [
    "SLACS_EXPECTED_ROWS",
    "SLACS_MASS_DEFINITION_LABEL",
    "SLACS_MASS_RADIUS_KPC",
    "SLACS_N_OBS",
    "SLACS_OBSERVATION_APERTURE_POLICY",
    "SLACS_OBSERVATION_FILENAME",
    "SLACS_POPULATION_SIGMA_FILENAME",
    "SLACS_PROFILE_NAME",
    "SLACS_RAW_FILENAME",
    "SlacsLensRow",
    "read_slacs_table",
    "write_slacs_observation_hdf5",
    "write_slacs_population_sigma_unit_hdf5",
]
