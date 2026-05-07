"""Compatibility re-export for the backend-neutral canonical helpers."""

from __future__ import annotations

from ..canonical_context import (
    canonical_dataset_metadata,
    interpolate_lensing_mass_grids,
    lens_gamma_axis,
    normalize_sigma_grid,
    representative_gamma_axis,
    shared_gamma_axis,
)


normalize_sigma_grid_for_jax = normalize_sigma_grid


__all__ = [
    "canonical_dataset_metadata",
    "interpolate_lensing_mass_grids",
    "lens_gamma_axis",
    "normalize_sigma_grid_for_jax",
    "representative_gamma_axis",
    "shared_gamma_axis",
]
