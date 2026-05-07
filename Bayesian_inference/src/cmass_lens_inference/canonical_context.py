"""
Backend-neutral helpers for canonical inference datasets.

This module owns small, deterministic transformations that are shared by
runtime adapters regardless of the numerical backend.  The helpers understand
canonical dataset array conventions, metadata, and interpolation axes, but they
do not import JAX, NumPyro, emcee, or model-specific likelihood code.  Keeping
them here prevents source-context construction from depending on a retired or
optional backend namespace.
"""

from __future__ import annotations

import numpy as np

from .canonical_dataset import (
    CanonicalInferenceDataset,
    CanonicalLensingMassGrids,
    CanonicalSigmaGrid,
)


def canonical_dataset_metadata(dataset: CanonicalInferenceDataset) -> dict[str, object]:
    """
    Return deterministic, JSON-friendly metadata for output manifests.

    Canonical capabilities are stored as a set-like object by the reader because
    validation needs fast membership checks.  Output manifests need stable order
    instead, so this helper sorts the names once at the framework boundary.
    """

    return {
        "canonical_dataset_path": str(dataset.path),
        "canonical_schema_version": dataset.metadata.schema_version,
        "canonical_capabilities": tuple(sorted(dataset.metadata.capabilities)),
        "canonical_profile_name": dataset.metadata.profile_name,
        "canonical_mass_definition_label": dataset.metadata.mass_definition_label,
    }


def lens_gamma_axis(gamma_grid: np.ndarray, lens_index: int) -> np.ndarray:
    """
    Return the gamma axis for one lens from canonical mass-grid storage.

    Canonical datasets may store either a shared one-dimensional gamma axis or a
    two-dimensional per-lens gamma axis.  Model preprocessors should not need to
    know which physical storage form the writer chose.
    """

    gamma_values = np.asarray(gamma_grid, dtype=np.float64)
    if gamma_values.ndim == 1:
        return gamma_values
    if gamma_values.ndim == 2:
        return gamma_values[lens_index]
    raise ValueError(
        "Canonical gamma_grid must have shape [N_gamma] or [N_lens, N_gamma], "
        f"got {gamma_values.shape}."
    )


def representative_gamma_axis(gamma_grid: np.ndarray) -> np.ndarray:
    """Return one representative gamma axis from canonical storage."""

    return lens_gamma_axis(gamma_grid, 0)


def shared_gamma_axis(gamma_grid: np.ndarray, *, n_points: int) -> np.ndarray:
    """
    Build one evenly spaced integration axis from canonical gamma storage.

    The rule intentionally mirrors the existing CMASS behavior: span the first
    available source axis from its first to last value.  Per-lens interpolation
    still happens later through `lens_gamma_axis()` when source axes differ.
    """

    if n_points <= 0:
        raise ValueError("n_points must be positive.")
    source_axis = representative_gamma_axis(gamma_grid)
    return np.linspace(
        float(source_axis[0]),
        float(source_axis[-1]),
        int(n_points),
        dtype=np.float64,
    )


def interpolate_lensing_mass_grids(
    mass_grids: CanonicalLensingMassGrids,
    target_gamma_axis: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Interpolate canonical per-lens mass tracks onto one target gamma axis.

    The returned arrays are log enclosed mass, derivative, and S2 grids.  This
    helper performs interpolation only; model-specific likelihood decisions
    remain in each model's backend kernel.
    """

    target_axis = np.asarray(target_gamma_axis, dtype=np.float64)
    n_lens = int(np.asarray(mass_grids.log_enclosed_mass_grid).shape[0])
    mass_grid = np.zeros((n_lens, target_axis.size), dtype=np.float64)
    derivative_grid = np.zeros((n_lens, target_axis.size), dtype=np.float64)
    s2_grid = np.zeros((n_lens, target_axis.size), dtype=np.float64)

    for lens_index in range(n_lens):
        source_axis = lens_gamma_axis(mass_grids.gamma_grid, lens_index)
        mass_grid[lens_index] = np.interp(
            target_axis,
            source_axis,
            mass_grids.log_enclosed_mass_grid[lens_index],
        )
        derivative_grid[lens_index] = np.interp(
            target_axis,
            source_axis,
            mass_grids.dmass_dthetaein_grid[lens_index],
        )
        s2_grid[lens_index] = np.interp(
            target_axis,
            source_axis,
            mass_grids.s2_grid[lens_index],
        )

    return mass_grid, derivative_grid, s2_grid


def normalize_sigma_grid(
    sigma_grid: CanonicalSigmaGrid | None,
    *,
    profile_fixed_n: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Normalize optional canonical sigma grids to `(gamma, zd, logRe, n)` order.

    Some canonical sigma products omit redshift or Sersic-index axes when the
    underlying physical table is independent of that coordinate.  Backend
    kernels are simpler and shape-stable when all axes are present, so this
    helper injects singleton compatibility axes and returns whether the source
    table truly carried an `n` axis.
    """

    if sigma_grid is None:
        return (
            np.zeros(1, dtype=np.float64),
            np.zeros(1, dtype=np.float64),
            np.zeros(1, dtype=np.float64),
            np.zeros(1, dtype=np.float64),
            np.zeros((1, 1, 1, 1), dtype=np.float64),
            0,
        )

    gamma_axis = np.asarray(sigma_grid.gamma_axis, dtype=np.float64)
    zd_axis = np.asarray(sigma_grid.zd_axis, dtype=np.float64)
    log_re_axis = np.asarray(sigma_grid.log_re_axis, dtype=np.float64)
    n_axis = np.asarray(sigma_grid.n_axis, dtype=np.float64)
    values = np.asarray(sigma_grid.sigma_unit_grid, dtype=np.float64)

    if values.ndim == 2:
        values = values[:, None, :, None]
        has_n_axis = 0
    elif values.ndim == 3:
        if values.shape[1] == log_re_axis.size:
            values = values[:, None, :, :]
            has_n_axis = 1
        else:
            values = values[..., None]
            has_n_axis = 0
    elif values.ndim == 4:
        has_n_axis = 1
    else:
        raise ValueError(f"Unsupported canonical sigma grid ndim={values.ndim}.")

    if not has_n_axis:
        n_axis = np.asarray(
            [profile_fixed_n if profile_fixed_n is not None else 4.0],
            dtype=np.float64,
        )

    return gamma_axis, zd_axis, log_re_axis, n_axis, values, has_n_axis


__all__ = [
    "canonical_dataset_metadata",
    "interpolate_lensing_mass_grids",
    "lens_gamma_axis",
    "normalize_sigma_grid",
    "representative_gamma_axis",
    "shared_gamma_axis",
]
