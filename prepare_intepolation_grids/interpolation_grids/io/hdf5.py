"""HDF5 reading and writing for interpolation-grid generation.

This module exists to keep file mutation concerns separate from numerical
calculations. The code here decides which groups to update, preserves the
existing schema, and applies the project's safe-write behavior.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import h5py
import numpy as np

from interpolation_grids.config import (
    DEFAULT_DERIVATIVE_THETA_SAMPLES,
    DERIVATIVE_DATASET_NAME,
    GAMMA_DATASET_NAME,
    GAMMA_GRID,
    LEGACY_DERIVATIVE_DATASET_NAME,
    MASS_DEFINITIONS_GROUP_NAME,
    MASS_DEFINITION_LABELS,
    MASS_DERIVATIVE_DATASET_NAME,
    MASS_GRID_DATASET_NAME,
    M5_DATASET_NAME,
    S2_DATASET_NAME,
    SUPPORTED_MASS_RADII_KPC,
)
from interpolation_grids.models import GalaxyInputs, ProcessingSummary
from interpolation_grids.physics.jeans import kpc_per_arcsec, compute_s2_grid
from interpolation_grids.physics.m5 import compute_dmass_dthetaein_grid, compute_mass_grid


def _read_attr_or_dataset(group_handle: h5py.Group, key: str) -> float | None:
    """Read a scalar either from attributes or datasets.

    Some legacy files store metadata as attributes while other workflows may
    write them as datasets. Supporting both reduces fragility.
    """

    if key in group_handle.attrs:
        return float(group_handle.attrs[key])
    if key in group_handle:
        return float(group_handle[key][()])
    return None


def build_galaxy_inputs(group_name: str, group_handle: h5py.Group, source_filename: str) -> GalaxyInputs:
    """Normalize one HDF5 group into a structured input object."""

    rein_arcsec = _read_attr_or_dataset(group_handle, "rein_arcsec")
    if rein_arcsec is None:
        raise ValueError(f"{group_name} is missing rein_arcsec")

    rein_kpc = _read_attr_or_dataset(group_handle, "r_ein_kpc")
    if rein_kpc is None:
        zd = _read_attr_or_dataset(group_handle, "zd")
        if zd is None:
            raise ValueError(f"{group_name} is missing both r_ein_kpc and zd")
        rein_kpc = rein_arcsec * kpc_per_arcsec(zd)

    return GalaxyInputs(
        group_name=group_name,
        source_filename=source_filename,
        zd=float(_read_attr_or_dataset(group_handle, "zd")),
        zs=float(_read_attr_or_dataset(group_handle, "zs")),
        sigma_crit=float(_read_attr_or_dataset(group_handle, "sigma_crit")),
        rein_arcsec=float(rein_arcsec),
        r_ein_kpc=float(rein_kpc),
        re_arcsec=_read_attr_or_dataset(group_handle, "re_arcsec"),
        reff_dev_arcsec=_read_attr_or_dataset(group_handle, "reff_deV"),
        nser=_read_attr_or_dataset(group_handle, "nser"),
        aperture_width_arcsec=_read_attr_or_dataset(group_handle, "aperture_width"),
        has_s2_grid=S2_DATASET_NAME in group_handle,
    )


def _write_or_replace_dataset(group_handle: h5py.Group, dataset_name: str, values: np.ndarray) -> None:
    """Replace or create a dataset while preserving a stable schema."""

    if dataset_name in group_handle:
        del group_handle[dataset_name]
    group_handle.create_dataset(dataset_name, data=values)


def _process_group(group_name: str, group_handle: h5py.Group, source_filename: str, summary: ProcessingSummary) -> None:
    """Update all relevant grids for a single galaxy group."""

    galaxy = build_galaxy_inputs(
        group_name=group_name,
        group_handle=group_handle,
        source_filename=source_filename,
    )
    gamma_grid = group_handle[GAMMA_DATASET_NAME][:] if GAMMA_DATASET_NAME in group_handle else GAMMA_GRID

    mass_definitions_handle = group_handle.require_group(MASS_DEFINITIONS_GROUP_NAME)
    mass_grids_by_radius: dict[float, np.ndarray] = {}
    derivative_grids_by_radius: dict[float, np.ndarray] = {}
    theta_scale_kpc_per_arcsec = galaxy.r_ein_kpc / galaxy.rein_arcsec

    for mass_radius_kpc in SUPPORTED_MASS_RADII_KPC:
        mass_grid = compute_mass_grid(
            gamma_grid=gamma_grid,
            sigma_crit=galaxy.sigma_crit,
            rein_kpc=galaxy.r_ein_kpc,
            mass_radius_kpc=mass_radius_kpc,
        )
        derivative_grid = compute_dmass_dthetaein_grid(
            gamma_grid=gamma_grid,
            sigma_crit=galaxy.sigma_crit,
            theta_ein_arcsec=galaxy.rein_arcsec,
            kpc_per_arcsec=theta_scale_kpc_per_arcsec,
            theta_samples=DEFAULT_DERIVATIVE_THETA_SAMPLES,
            mass_radius_kpc=mass_radius_kpc,
        )
        subgroup = mass_definitions_handle.require_group(MASS_DEFINITION_LABELS[float(mass_radius_kpc)])
        _write_or_replace_dataset(subgroup, MASS_GRID_DATASET_NAME, mass_grid)
        _write_or_replace_dataset(subgroup, MASS_DERIVATIVE_DATASET_NAME, derivative_grid)
        mass_grids_by_radius[float(mass_radius_kpc)] = mass_grid
        derivative_grids_by_radius[float(mass_radius_kpc)] = derivative_grid

    _write_or_replace_dataset(group_handle, M5_DATASET_NAME, mass_grids_by_radius[5.0])
    _write_or_replace_dataset(group_handle, DERIVATIVE_DATASET_NAME, derivative_grids_by_radius[5.0])

    # If the old alternate spelling exists, keep the schema clean by removing it
    # after writing the canonical dataset name used by the real files.
    if LEGACY_DERIVATIVE_DATASET_NAME in group_handle:
        del group_handle[LEGACY_DERIVATIVE_DATASET_NAME]

    summary.updated_m5 += 1
    summary.updated_dm5 += 1

    if galaxy.has_s2_grid:
        for mass_radius_kpc in SUPPORTED_MASS_RADII_KPC:
            s2_grid = compute_s2_grid(
                galaxy=galaxy,
                gamma_grid=np.asarray(gamma_grid, dtype=float),
                mass_radius_kpc=mass_radius_kpc,
            )
            subgroup = mass_definitions_handle[MASS_DEFINITION_LABELS[float(mass_radius_kpc)]]
            _write_or_replace_dataset(subgroup, S2_DATASET_NAME, s2_grid)

        _write_or_replace_dataset(
            group_handle,
            S2_DATASET_NAME,
            compute_s2_grid(galaxy=galaxy, gamma_grid=np.asarray(gamma_grid, dtype=float), mass_radius_kpc=5.0),
        )
        summary.updated_s2 += 1


def process_hdf5_file(
    input_path: Path | str,
    output_path: Path | str,
    overwrite_in_place: bool = False,
    group_names: tuple[str, ...] | None = None,
) -> ProcessingSummary:
    """Process one HDF5 file and write the updated result.

    Parameters
    ----------
    input_path, output_path:
        Input and output file paths. When `overwrite_in_place` is false, the
        caller gets a new output file. When true, we still use a temporary file
        first and only replace the input once processing succeeds.
    overwrite_in_place:
        Enable atomic replacement of the input file.
    group_names:
        Optional whitelist of group names to process. This is primarily for
        debugging and targeted regression checks on one or a few galaxies.
    """

    input_path = Path(input_path)
    output_path = Path(output_path)
    final_output_path = input_path if overwrite_in_place else output_path
    summary = ProcessingSummary(input_path=input_path, output_path=final_output_path)

    if overwrite_in_place:
        temp_directory = output_path.parent if output_path.parent.exists() else input_path.parent
        with tempfile.NamedTemporaryFile(prefix=f"{input_path.stem}.", suffix=input_path.suffix, dir=temp_directory, delete=False) as temp_file:
            working_path = Path(temp_file.name)
    else:
        working_path = output_path

    shutil.copy2(input_path, working_path)

    try:
        with h5py.File(working_path, "r+") as handle:
            available_group_names = list(handle.keys())
            names_to_process = group_names or tuple(available_group_names)
            for group_name in names_to_process:
                if group_name not in handle:
                    summary.failures.append(f"{group_name}: group not found")
                    continue
                summary.total_groups += 1
                try:
                    _process_group(
                        group_name=group_name,
                        group_handle=handle[group_name],
                        source_filename=input_path.name,
                        summary=summary,
                    )
                except Exception as exc:  # noqa: BLE001 - we want per-group resilience.
                    summary.failures.append(f"{group_name}: {exc}")
        if overwrite_in_place:
            working_path.replace(input_path)
    finally:
        if overwrite_in_place and working_path.exists():
            working_path.unlink(missing_ok=True)

    return summary
