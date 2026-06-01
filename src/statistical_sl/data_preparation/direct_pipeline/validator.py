"""Validators for direct canonical payloads and written HDF5 files."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import h5py
import numpy as np

from statistical_sl.data_preparation.direct_pipeline.records import CanonicalDatasetPayload
from statistical_sl.core.canonical_schema import (
    BLOCK_LENSES,
    BLOCK_LENSING_CROSS_SECTION,
    BLOCK_LENSING_MASS_GRIDS,
    BLOCK_METADATA,
    BLOCK_VELOCITY_DISPERSION_GRIDS,
    TOP_LEVEL_BLOCKS,
)


class CanonicalDatasetValidationError(ValueError):
    """Raised when a direct canonical payload or HDF5 file is internally invalid."""


def _require_mapping_key(mapping: Mapping[str, object], key: str, context: str) -> object:
    """Return one required mapping value or raise with a useful context."""

    if key not in mapping:
        raise CanonicalDatasetValidationError(f"{context} is missing required key {key!r}.")
    return mapping[key]


def _validate_numeric_array(value: object, context: str) -> None:
    """Reject non-finite values in numeric numpy arrays."""

    array = np.asarray(value)
    if array.dtype.kind in {"i", "u", "f", "c"} and not np.all(np.isfinite(array)):
        raise CanonicalDatasetValidationError(f"{context} contains non-finite values.")


def _string_values(value: object) -> np.ndarray:
    """Normalize HDF5/NumPy string payloads into a one-dimensional text array."""

    array = np.asarray(value)
    values = [
        item.decode("utf-8") if isinstance(item, (bytes, np.bytes_)) else str(item)
        for item in array.reshape(-1)
    ]
    return np.asarray(values, dtype=object).reshape(array.shape)


def _validate_sigma_aperture_contract(lenses: Mapping[str, object], num_sigma: np.ndarray, context: str) -> None:
    """Require complete per-lens aperture metadata for sigma-bearing lenses.

    ``num_sigma = 0`` lenses remain valid catalog members even when no trusted
    velocity-dispersion measurement exists.  Once a lens contributes to the
    observed-sigma likelihood, however, the aperture and seeing are part of the
    measurement contract and must be explicit.
    """

    aperture_shape = _string_values(_require_mapping_key(lenses, "aperture_shape", context))
    width = np.asarray(_require_mapping_key(lenses, "aperture_width_arcsec", context), dtype=float)
    height = np.asarray(_require_mapping_key(lenses, "aperture_height_arcsec", context), dtype=float)
    radius = np.asarray(_require_mapping_key(lenses, "aperture_radius_arcsec", context), dtype=float)
    seeing = np.asarray(_require_mapping_key(lenses, "seeing_fwhm_arcsec", context), dtype=float)

    expected_shape = num_sigma.shape
    for field_name, values in (
        ("aperture_shape", aperture_shape),
        ("aperture_width_arcsec", width),
        ("aperture_height_arcsec", height),
        ("aperture_radius_arcsec", radius),
        ("seeing_fwhm_arcsec", seeing),
    ):
        if values.shape != expected_shape:
            raise CanonicalDatasetValidationError(f"{context}/{field_name} dimension does not match num_sigma.")

    for index, count in enumerate(num_sigma):
        if count <= 0:
            continue
        shape = str(aperture_shape[index]).strip().lower()
        if not shape:
            raise CanonicalDatasetValidationError(f"{context} aperture metadata is missing for sigma-bearing lens row {index}.")
        if seeing[index] <= 0.0:
            raise CanonicalDatasetValidationError(f"{context} aperture seeing is missing for sigma-bearing lens row {index}.")
        if shape == "rectangular":
            if width[index] <= 0.0 or height[index] <= 0.0 or radius[index] != 0.0:
                raise CanonicalDatasetValidationError(
                    f"{context} rectangular aperture metadata is incomplete for sigma-bearing lens row {index}."
                )
            continue
        if shape == "circular":
            if radius[index] <= 0.0 or width[index] != 0.0 or height[index] != 0.0:
                raise CanonicalDatasetValidationError(
                    f"{context} circular aperture metadata is incomplete for sigma-bearing lens row {index}."
                )
            continue
        raise CanonicalDatasetValidationError(f"{context} has unsupported aperture_shape={shape!r}.")


def _validate_payload_numeric_content(payload: CanonicalDatasetPayload) -> None:
    """Check all numeric arrays in the in-memory payload."""

    for block_name, block in (
        ("lenses", payload.lenses),
        ("lensing_mass_grids", payload.lensing_mass_grids),
        ("lensing_cross_section", payload.lensing_cross_section),
        ("velocity_dispersion_grids", payload.velocity_dispersion_grids),
    ):
        for key, value in block.items():
            _validate_numeric_array(value, f"{block_name}/{key}")


def validate_canonical_dataset_payload(payload: CanonicalDatasetPayload) -> None:
    """Validate the in-memory canonical payload before serialization."""

    if "capabilities" not in payload.metadata:
        raise CanonicalDatasetValidationError("metadata must include a capabilities block.")

    lens_ids = np.asarray(_require_mapping_key(payload.lenses, "lens_id", "lenses"))
    n_lens = lens_ids.shape[0]
    num_sigma = np.asarray(_require_mapping_key(payload.lenses, "num_sigma", "lenses"), dtype=np.int64)
    if num_sigma.shape != (n_lens,):
        raise CanonicalDatasetValidationError("lenses/num_sigma dimension does not match lens_id.")
    _validate_sigma_aperture_contract(payload.lenses, num_sigma, "lenses")

    sigma_obs = np.asarray(_require_mapping_key(payload.lenses, "sigma_obs", "lenses"), dtype=float)
    sigma_err = np.asarray(_require_mapping_key(payload.lenses, "sigma_err", "lenses"), dtype=float)
    if sigma_obs.shape[0] != n_lens or sigma_err.shape != sigma_obs.shape:
        raise CanonicalDatasetValidationError("sigma_obs and sigma_err dimensions must match lens_id.")

    mass_grid = np.asarray(
        _require_mapping_key(payload.lensing_mass_grids, "log_enclosed_mass_grid", "lensing_mass_grids"),
        dtype=float,
    )
    derivative_grid = np.asarray(
        _require_mapping_key(payload.lensing_mass_grids, "dmass_dthetaein_grid", "lensing_mass_grids"),
        dtype=float,
    )
    if mass_grid.shape[0] != n_lens or derivative_grid.shape != mass_grid.shape:
        raise CanonicalDatasetValidationError("lensing_mass_grids dimension does not match lens_id.")

    s2_grid = np.asarray(_require_mapping_key(payload.velocity_dispersion_grids, "s2_grid", "velocity_dispersion_grids"), dtype=float)
    has_s2 = np.asarray(_require_mapping_key(payload.velocity_dispersion_grids, "has_s2", "velocity_dispersion_grids"), dtype=bool)
    if s2_grid.shape != mass_grid.shape or has_s2.shape != (n_lens,):
        raise CanonicalDatasetValidationError("velocity_dispersion_grids dimension does not match lens_id.")
    if np.any((num_sigma > 0) & ~has_s2):
        raise CanonicalDatasetValidationError("num_sigma > 0 requires an available s2_grid row.")

    theta_axis = np.asarray(
        _require_mapping_key(payload.lensing_cross_section, "theta_e_axis", "lensing_cross_section"),
        dtype=float,
    )
    gamma_axis = np.asarray(
        _require_mapping_key(payload.lensing_cross_section, "gamma_axis", "lensing_cross_section"),
        dtype=float,
    )
    cross_section_grid = np.asarray(
        _require_mapping_key(payload.lensing_cross_section, "cross_section_grid", "lensing_cross_section"),
        dtype=float,
    )
    if cross_section_grid.shape != (theta_axis.size, gamma_axis.size):
        raise CanonicalDatasetValidationError("lensing_cross_section dimension does not match axes.")

    _validate_payload_numeric_content(payload)


def _require_group(handle: h5py.File | h5py.Group, name: str) -> h5py.Group:
    """Return one required HDF5 group."""

    if name not in handle or not isinstance(handle[name], h5py.Group):
        raise CanonicalDatasetValidationError(f"HDF5 file is missing required group {name!r}.")
    return handle[name]


def _require_dataset(handle: h5py.File | h5py.Group, name: str) -> h5py.Dataset:
    """Return one required HDF5 dataset."""

    if name not in handle or not isinstance(handle[name], h5py.Dataset):
        raise CanonicalDatasetValidationError(f"HDF5 file is missing required dataset {name!r}.")
    return handle[name]


def _hdf5_lenses_mapping(lenses: h5py.Group) -> dict[str, object]:
    """Read the lens datasets needed by shared payload/HDF5 aperture validation."""

    return {
        "aperture_shape": _require_dataset(lenses, "aperture_shape")[()],
        "aperture_width_arcsec": _require_dataset(lenses, "aperture_width_arcsec")[()],
        "aperture_height_arcsec": _require_dataset(lenses, "aperture_height_arcsec")[()],
        "aperture_radius_arcsec": _require_dataset(lenses, "aperture_radius_arcsec")[()],
        "seeing_fwhm_arcsec": _require_dataset(lenses, "seeing_fwhm_arcsec")[()],
    }


def _validate_hdf5_numeric_content(handle: h5py.File) -> None:
    """Traverse numeric datasets and reject non-finite values."""

    def visitor(name: str, obj: h5py.Dataset | h5py.Group) -> None:
        if not isinstance(obj, h5py.Dataset):
            return
        if obj.dtype.kind in {"i", "u", "f", "c"}:
            values = np.asarray(obj[()])
            if not np.all(np.isfinite(values)):
                raise CanonicalDatasetValidationError(f"{name} contains non-finite values.")

    handle.visititems(visitor)


def validate_canonical_hdf5(path: Path | str) -> None:
    """Validate a written canonical HDF5 file."""

    resolved_path = Path(path).expanduser().resolve()
    with h5py.File(resolved_path, "r") as handle:
        missing_blocks = sorted(set(TOP_LEVEL_BLOCKS).difference(handle.keys()))
        if missing_blocks:
            raise CanonicalDatasetValidationError(f"HDF5 file is missing top-level blocks: {missing_blocks}")

        metadata = _require_group(handle, BLOCK_METADATA)
        _require_dataset(metadata, "capabilities")

        lenses = _require_group(handle, BLOCK_LENSES)
        mass = _require_group(handle, BLOCK_LENSING_MASS_GRIDS)
        cross_section = _require_group(handle, BLOCK_LENSING_CROSS_SECTION)
        velocity = _require_group(handle, BLOCK_VELOCITY_DISPERSION_GRIDS)
        per_lens_s2 = _require_group(velocity, "per_lens_s2")

        lens_ids = np.asarray(_require_dataset(lenses, "lens_id")[()])
        n_lens = lens_ids.shape[0]
        num_sigma = np.asarray(_require_dataset(lenses, "num_sigma")[()], dtype=np.int64)
        if num_sigma.shape != (n_lens,):
            raise CanonicalDatasetValidationError("lenses/num_sigma dimension does not match lens_id.")
        _validate_sigma_aperture_contract(_hdf5_lenses_mapping(lenses), num_sigma, "lenses")

        mass_grid = np.asarray(_require_dataset(mass, "log_enclosed_mass_grid")[()], dtype=float)
        derivative_grid = np.asarray(_require_dataset(mass, "dmass_dthetaein_grid")[()], dtype=float)
        if mass_grid.shape[0] != n_lens or derivative_grid.shape != mass_grid.shape:
            raise CanonicalDatasetValidationError("lensing_mass_grids dimension does not match lens_id.")

        s2_grid = np.asarray(_require_dataset(per_lens_s2, "s2_grid")[()], dtype=float)
        has_s2 = np.asarray(_require_dataset(per_lens_s2, "has_s2")[()], dtype=bool)
        if s2_grid.shape != mass_grid.shape or has_s2.shape != (n_lens,):
            raise CanonicalDatasetValidationError("velocity_dispersion_grids dimension does not match lens_id.")
        if np.any((num_sigma > 0) & ~has_s2):
            raise CanonicalDatasetValidationError("num_sigma > 0 requires an available s2_grid row.")

        theta_axis = np.asarray(_require_dataset(cross_section, "theta_e_axis")[()], dtype=float)
        gamma_axis = np.asarray(_require_dataset(cross_section, "gamma_axis")[()], dtype=float)
        cross_section_grid = np.asarray(_require_dataset(cross_section, "cross_section_grid")[()], dtype=float)
        if cross_section_grid.shape != (theta_axis.size, gamma_axis.size):
            raise CanonicalDatasetValidationError("lensing_cross_section dimension does not match axes.")

        _validate_hdf5_numeric_content(handle)


__all__ = [
    "CanonicalDatasetValidationError",
    "validate_canonical_dataset_payload",
    "validate_canonical_hdf5",
]
