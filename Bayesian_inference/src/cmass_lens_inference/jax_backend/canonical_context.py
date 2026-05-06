"""
Reusable helpers for canonical inference datasets.

This module is deliberately model-neutral.  It understands canonical dataset
array conventions, metadata, and interpolation axes, but it does not import any
CMASS context classes or scientific formulas.  Concrete model preprocessors can
use these helpers to avoid reimplementing the same canonical shape handling.
"""

from __future__ import annotations

import numpy as np

from ..canonical_dataset import (
    CanonicalInferenceDataset,
    CanonicalLensingMassGrids,
    CanonicalSigmaGrid,
)


def canonical_dataset_metadata(dataset: CanonicalInferenceDataset) -> dict[str, object]:
    """
    Return deterministic, JSON-friendly metadata for output manifests.

    The canonical reader stores capabilities as a `frozenset`, which is ideal
    for validation but unstable for serialized output order.  Sorting here
    keeps run metadata deterministic and keeps runtime adapters from repeating
    that convention.
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

    Canonical datasets may store either one shared `[N_gamma]` axis or a
    per-lens `[N_lens, N_gamma]` axis.  Model preprocessors should not care
    which storage form was used by data preparation.
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
    """
    Return one representative gamma axis from canonical storage.

    This is useful when a model wants to build a common integration axis from
    the first lens.  Per-lens interpolation still needs `lens_gamma_axis()` for
    correctness when other lenses use slightly different source axes.
    """

    return lens_gamma_axis(gamma_grid, 0)


def shared_gamma_axis(gamma_grid: np.ndarray, *, n_points: int) -> np.ndarray:
    """
    Build one evenly spaced integration axis from canonical gamma storage.

    The returned axis follows the previous CMASS behavior exactly: use the
    first available source axis and span its first-to-last values.  Keeping this
    rule centralized prevents future models from accidentally choosing slightly
    different target axes during migration tests.
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

    The helper returns log enclosed mass, derivative, and per-lens S2 grids in
    the same order used by the canonical schema.  It intentionally performs
    interpolation only; model-specific decisions about how those arrays enter a
    likelihood remain outside this module.
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


def normalize_sigma_grid_for_jax(
    sigma_grid: CanonicalSigmaGrid | None,
    *,
    profile_fixed_n: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Normalize optional canonical sigma grids to `(gamma, zd, logRe, n)` order.

    Some canonical sigma products omit redshift or Sersic-index axes when the
    underlying physical table is independent of that coordinate.  JAX kernels
    are simpler and shape-stable when all axes are present, so this helper
    injects singleton compatibility axes and returns a flag indicating whether
    the source table truly carried an `n` axis.
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
    "normalize_sigma_grid_for_jax",
    "representative_gamma_axis",
    "shared_gamma_axis",
]
