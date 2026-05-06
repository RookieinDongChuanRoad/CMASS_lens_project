"""HDF5 writers for prepared lensing cross-section grids.

The numerical routines live under :mod:`prepare_dataset.physics` so they can be
unit-tested without touching disk.  This module owns the file format boundary:
it writes either the historical CMASS legacy schema or the Sonnenfeld
finite-fibre schema through a temporary file, then atomically moves it into
place only after the file is complete.
"""

from __future__ import annotations

from pathlib import Path
import tempfile

import h5py
import numpy as np

from prepare_dataset.physics.lensing_cross_section import (
    FibreCrossSectionGrid,
    PowerLawCrossSectionGrid,
    SONNENFELD_CROSS_SECTION_REFERENCE_SHA,
    SONNENFELD_CROSS_SECTION_REFERENCE_URL,
    compute_fibre_cross_section_grid,
    compute_power_law_cross_section_grid,
)


def _prepare_output_path(output_path: str | Path, *, overwrite: bool) -> Path:
    """Resolve the output path and enforce the overwrite contract."""

    resolved = Path(output_path).expanduser().resolve()
    if resolved.exists() and not overwrite:
        raise FileExistsError(f"{resolved} already exists. Pass overwrite=True to replace it.")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def _write_atomically(output_path: Path, writer) -> Path:
    """Write one HDF5 product through a sibling temporary file."""

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            dir=output_path.parent,
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)

        with h5py.File(temporary_path, "w") as handle:
            writer(handle)

        temporary_path.replace(output_path)
        return output_path
    except Exception:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
        raise


def _write_power_law_grid(handle: h5py.File, grid: PowerLawCrossSectionGrid) -> None:
    """Write the legacy CMASS cross-section HDF5 schema."""

    full = handle.create_group("full_grids")
    full.create_dataset("gamma_grids", data=np.asarray(grid.gamma_axis, dtype=float))
    full.create_dataset("theta_ein_grids", data=np.asarray(grid.theta_e_axis, dtype=float))
    full.create_dataset("cs_grid", data=np.asarray(grid.cs_grid, dtype=float))

    compressed = handle.create_group("compressed_grids")
    compressed.create_dataset("gamma_grids", data=np.asarray(grid.gamma_axis, dtype=float))
    compressed.create_dataset("cs_over_theta_ein_grid", data=np.asarray(grid.cs_over_theta_ein_grid, dtype=float))

    handle.attrs["generator_name"] = "prepare_dataset.power_law_cross_section"
    handle.attrs["quantity_full_grids_cs_grid"] = "source_plane_beta_max_arcsec"
    handle.attrs["quantity_compressed_grids_cs_over_theta_ein_grid"] = "beta_max_over_theta_E"
    handle.attrs["area_conversion"] = "pi * (cs_over_theta_ein * theta_E)**2"


def _write_fibre_grid(handle: h5py.File, grid: FibreCrossSectionGrid) -> None:
    """Write the Sonnenfeld finite-fibre cross-section HDF5 schema."""

    handle.create_dataset("tein_grid", data=np.asarray(grid.theta_e_axis, dtype=float))
    handle.create_dataset("gamma_grid", data=np.asarray(grid.gamma_axis, dtype=float))
    handle.create_dataset("mufibre2_cs_grid", data=np.asarray(grid.mufibre2_cs_grid, dtype=float))
    handle.create_dataset("mufibre3_cs_grid", data=np.asarray(grid.mufibre3_cs_grid, dtype=float))
    handle.create_dataset("ycaust_grid", data=np.asarray(grid.ycaust_grid, dtype=float))
    handle.attrs["muB_min"] = float(grid.muB_min)
    handle.attrs["fibre_arcsec"] = float(grid.fibre_arcsec)
    handle.attrs["seeing_arcsec"] = float(grid.seeing_arcsec)
    handle.attrs["nbeta"] = int(grid.beta_points)
    handle.attrs["nr"] = int(grid.radial_points)
    handle.attrs["axis_order"] = "theta_E,gamma"
    handle.attrs["generator_name"] = "prepare_dataset.fibre_cross_section"
    handle.attrs["source_reference_url"] = SONNENFELD_CROSS_SECTION_REFERENCE_URL
    handle.attrs["source_reference_sha"] = SONNENFELD_CROSS_SECTION_REFERENCE_SHA


def write_power_law_cross_section_hdf5(
    output_path: str | Path,
    *,
    gamma_axis: np.ndarray | None = None,
    theta_e_axis: np.ndarray | None = None,
    binary_iterations: int = 30,
    overwrite: bool = False,
) -> Path:
    """Compute and write the legacy CMASS power-law cross-section table."""

    resolved_output = _prepare_output_path(output_path, overwrite=overwrite)
    grid = compute_power_law_cross_section_grid(
        gamma_axis=gamma_axis,
        theta_e_axis=theta_e_axis,
        binary_iterations=binary_iterations,
    )
    return _write_atomically(resolved_output, lambda handle: _write_power_law_grid(handle, grid))


def write_fibre_cross_section_hdf5(
    output_path: str | Path,
    *,
    gamma_axis: np.ndarray | None = None,
    theta_e_axis: np.ndarray | None = None,
    fibre_arcsec: float = 1.5,
    seeing_arcsec: float = 1.5,
    muB_min: float = 1.0,
    beta_points: int = 1001,
    radial_points: int = 16,
    overwrite: bool = False,
    progress: bool = False,
) -> Path:
    """Compute and write the Sonnenfeld finite-fibre cross-section table."""

    resolved_output = _prepare_output_path(output_path, overwrite=overwrite)
    grid = compute_fibre_cross_section_grid(
        gamma_axis=gamma_axis,
        theta_e_axis=theta_e_axis,
        fibre_arcsec=fibre_arcsec,
        seeing_arcsec=seeing_arcsec,
        muB_min=muB_min,
        beta_points=beta_points,
        radial_points=radial_points,
        progress=progress,
    )
    return _write_atomically(resolved_output, lambda handle: _write_fibre_grid(handle, grid))


__all__ = [
    "write_fibre_cross_section_hdf5",
    "write_power_law_cross_section_hdf5",
]
