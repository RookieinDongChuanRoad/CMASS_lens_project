"""Read-only helpers for comparing against legacy observation HDF5 files.

The direct pipeline must not depend on the legacy observation HDF5 format at
runtime.  This module exists only for migration tests: it extracts the small
set of lens-level facts needed to prove that the new catalog reader and sigma
resolver can reproduce legacy reference behavior from direct sources.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np


@dataclass(frozen=True)
class LegacyReferenceLens:
    """Lens-level facts read from one legacy observation HDF5 group."""

    lens_id: str
    z_lens: float
    z_source: float
    theta_ein_arcsec: float
    effective_radius_arcsec: float
    log_stellar_mass: float
    log_stellar_mass_err: float
    num_sigma: int


def _read_finite_float_attr(group: h5py.Group, attr_name: str, lens_id: str) -> float:
    """Read one required finite floating-point attribute from a legacy group."""

    if attr_name not in group.attrs:
        raise KeyError(f"{lens_id} is missing required legacy attribute {attr_name!r}.")
    value = float(group.attrs[attr_name])
    if not np.isfinite(value):
        raise ValueError(f"{lens_id} legacy attribute {attr_name!r} is not finite.")
    return value


def _read_num_sigma(group: h5py.Group, lens_id: str) -> int:
    """Read and validate the legacy ``num_sigma`` count."""

    if "num_sigma" not in group.attrs:
        raise KeyError(f"{lens_id} is missing required legacy attribute 'num_sigma'.")
    value = int(group.attrs["num_sigma"])
    if value < 0 or value > 2:
        raise ValueError(f"{lens_id} has unsupported legacy num_sigma={value}.")
    return value


def _ordered_lens_ids(handle: h5py.File, lens_ids: Sequence[str] | None, limit: int | None) -> tuple[str, ...]:
    """Resolve the deterministic group order used by reference comparisons."""

    if lens_ids is None:
        ordered_ids = tuple(sorted(str(lens_id) for lens_id in handle.keys()))
    else:
        ordered_ids = tuple(str(lens_id) for lens_id in lens_ids)

    if limit is not None:
        if int(limit) <= 0:
            raise ValueError("limit must be positive when provided.")
        ordered_ids = ordered_ids[: int(limit)]

    missing_ids = [lens_id for lens_id in ordered_ids if lens_id not in handle]
    if missing_ids:
        raise KeyError(f"legacy reference is missing requested lens ids: {', '.join(missing_ids)}")
    return ordered_ids


def _read_reference_lens(handle: h5py.File, lens_id: str) -> LegacyReferenceLens:
    """Read one legacy group into the narrow reference dataclass."""

    group = handle[lens_id]
    if not isinstance(group, h5py.Group):
        raise TypeError(f"{lens_id} is not an HDF5 group in the legacy reference file.")

    return LegacyReferenceLens(
        lens_id=lens_id,
        z_lens=_read_finite_float_attr(group, "zd", lens_id),
        z_source=_read_finite_float_attr(group, "zs", lens_id),
        theta_ein_arcsec=_read_finite_float_attr(group, "rein_arcsec", lens_id),
        effective_radius_arcsec=_read_finite_float_attr(group, "reff_deV", lens_id),
        log_stellar_mass=_read_finite_float_attr(group, "logmchab_deV", lens_id),
        log_stellar_mass_err=_read_finite_float_attr(group, "logmchab_err", lens_id),
        num_sigma=_read_num_sigma(group, lens_id),
    )


def read_legacy_reference_lenses(
    reference_path: Path | str,
    *,
    lens_ids: Sequence[str] | None = None,
    limit: int | None = None,
) -> tuple[LegacyReferenceLens, ...]:
    """Read a deterministic subset of the old observation HDF5 reference file."""

    resolved_path = Path(reference_path).expanduser().resolve()
    with h5py.File(resolved_path, "r") as handle:
        ordered_ids = _ordered_lens_ids(handle, lens_ids, limit)
        return tuple(_read_reference_lens(handle, lens_id) for lens_id in ordered_ids)


def num_sigma_distribution(reference_lenses: Sequence[LegacyReferenceLens]) -> Mapping[int, int]:
    """Return the ``num_sigma`` distribution as a deterministic plain mapping."""

    counts = Counter(reference_lens.num_sigma for reference_lens in reference_lenses)
    return {key: counts[key] for key in sorted(counts)}


__all__ = [
    "LegacyReferenceLens",
    "num_sigma_distribution",
    "read_legacy_reference_lenses",
]
