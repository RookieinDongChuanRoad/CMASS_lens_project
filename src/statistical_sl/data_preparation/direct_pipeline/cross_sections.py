"""Cross-section providers for the direct canonical pipeline.

The direct pipeline needs two scientifically distinct source products:

- the CMASS power-law file, which stores a one-dimensional ratio that must be
  converted into a theta_E x gamma area grid; and
- the Sonnenfeld finite-fibre file, which already stores a two-dimensional
  area grid and should be preserved as-is.

This module keeps those two semantics separate so the payload builder does not
need to infer them from filename conventions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import h5py
import numpy as np


def _freeze_mapping(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    """Return a shallow read-only mapping for provenance payloads."""

    return MappingProxyType(dict(value or {}))


@dataclass(frozen=True)
class CrossSectionProvenance:
    """Minimal provenance attached to a loaded cross-section block."""

    source_path: Path
    source_mode: str
    source_dataset: str
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalize the provenance path and freeze nested metadata."""

        object.__setattr__(self, "source_path", Path(self.source_path).expanduser().resolve())
        object.__setattr__(self, "extra", _freeze_mapping(self.extra))


@dataclass(frozen=True)
class CrossSectionBlock:
    """One canonical theta_E x gamma cross-section grid."""

    theta_e_axis: np.ndarray
    gamma_axis: np.ndarray
    cross_section_grid: np.ndarray
    provenance: CrossSectionProvenance


def _read_required_dataset(handle: h5py.File, dataset_path: str) -> np.ndarray:
    """Read one required HDF5 dataset using an explicit path string."""

    if dataset_path not in handle:
        raise KeyError(f"Missing required dataset: {dataset_path}")
    return np.asarray(handle[dataset_path][()], dtype=float)


def _read_legacy_cmass_ratio(handle: h5py.File, source_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read the one-dimensional CMASS ratio representation."""

    if "compressed_grids" not in handle:
        raise KeyError(f"{source_path} is missing compressed_grids.")
    compressed = handle["compressed_grids"]
    gamma_name = "gamma_grid" if "gamma_grid" in compressed else "gamma_grids"
    ratio_name = "cs_over_theta_ein_grid" if "cs_over_theta_ein_grid" in compressed else "cs_over_theta_ein"
    if gamma_name not in compressed or ratio_name not in compressed:
        raise KeyError(f"{source_path} is missing the CMASS compressed cross-section datasets.")
    return (
        np.asarray(compressed[gamma_name][()], dtype=float),
        np.asarray(compressed[ratio_name][()], dtype=float),
    )


@dataclass(frozen=True)
class CmassPowerLawCrossSectionProvider:
    """Load the CMASS power-law cross-section ratio file."""

    source_path: Path

    def __post_init__(self) -> None:
        """Normalize the source path for stable provenance."""

        object.__setattr__(self, "source_path", Path(self.source_path).expanduser().resolve())

    def load(self, *, theta_e_axis: np.ndarray) -> CrossSectionBlock:
        """Convert the stored ratio into a canonical area grid."""

        theta_axis = np.asarray(theta_e_axis, dtype=float)
        if theta_axis.ndim != 1 or theta_axis.size == 0:
            raise ValueError("theta_e_axis must be a non-empty one-dimensional array.")

        with h5py.File(self.source_path, "r") as handle:
            gamma_axis, cs_over_theta_ein = _read_legacy_cmass_ratio(handle, self.source_path)
            cross_section_grid = np.pi * (theta_axis[:, None] * cs_over_theta_ein[None, :]) ** 2

        return CrossSectionBlock(
            theta_e_axis=theta_axis,
            gamma_axis=gamma_axis,
            cross_section_grid=cross_section_grid,
            provenance=CrossSectionProvenance(
                source_path=self.source_path,
                source_mode="cmass_power_law",
                source_dataset="compressed_grids/cs_over_theta_ein_grid",
            ),
        )


@dataclass(frozen=True)
class SonnenfeldFibreCrossSectionProvider:
    """Load the Sonnenfeld finite-fibre cross-section area grid."""

    source_path: Path

    def __post_init__(self) -> None:
        """Normalize the source path for stable provenance."""

        object.__setattr__(self, "source_path", Path(self.source_path).expanduser().resolve())

    def load(self) -> CrossSectionBlock:
        """Return the stored finite-fibre area grid without any conversion."""

        with h5py.File(self.source_path, "r") as handle:
            theta_axis = _read_required_dataset(handle, "tein_grid")
            gamma_axis = _read_required_dataset(handle, "gamma_grid")
            cross_section_grid = _read_required_dataset(handle, "mufibre3_cs_grid")
            expected_shape = (theta_axis.size, gamma_axis.size)
            if cross_section_grid.shape != expected_shape:
                raise ValueError(
                    f"mufibre3_cs_grid must have shape {expected_shape}, got {cross_section_grid.shape}."
                )

        return CrossSectionBlock(
            theta_e_axis=theta_axis,
            gamma_axis=gamma_axis,
            cross_section_grid=cross_section_grid,
            provenance=CrossSectionProvenance(
                source_path=self.source_path,
                source_mode="sonnenfeld_fibre",
                source_dataset="mufibre3_cs_grid",
            ),
        )


__all__ = [
    "CmassPowerLawCrossSectionProvider",
    "CrossSectionBlock",
    "CrossSectionProvenance",
    "SonnenfeldFibreCrossSectionProvider",
]
