"""
Canonical inference dataset reader and validator.

The data-preparation step owns conversion from raw survey products into the
canonical HDF5 schema.  This module owns the inference-side contract: read that
already-prepared file, validate the metadata/capabilities/shapes needed before
runtime context construction, and expose typed NumPy records to model runtimes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np


CANONICAL_SCHEMA_VERSION = "canonical_inference_dataset_v1"

BLOCK_METADATA = "metadata"
BLOCK_LENSES = "lenses"
BLOCK_LENSING_MASS_GRIDS = "lensing_mass_grids"
BLOCK_LENSING_CROSS_SECTION = "lensing_cross_section"
BLOCK_VELOCITY_DISPERSION_GRIDS = "velocity_dispersion_grids"

CAPABILITY_LENS_OBSERVATIONS_V1 = "lens_observations.v1"
CAPABILITY_LENSING_MASS_GRIDS_V1 = "lensing_mass_grids.v1"
CAPABILITY_LENSING_CROSS_SECTION_THETA_GAMMA_V1 = "lensing_cross_section.theta_gamma_grid.v1"
CAPABILITY_VELOCITY_DISPERSION_PER_LENS_S2_V1 = "velocity_dispersion.per_lens_s2.v1"
CAPABILITY_VELOCITY_DISPERSION_FP_WITHIN_RE_V1 = "velocity_dispersion.fp_within_re.v1"
CAPABILITY_VELOCITY_DISPERSION_POPULATION_SIGMA_UNIT_V1 = "velocity_dispersion.population_sigma_unit.v1"


@dataclass(frozen=True)
class CanonicalMetadata:
    """
    File-level metadata that defines the numerical coordinate system.

    The reader keeps these values explicit so runtime errors can report the
    dataset contract that was actually loaded instead of relying on raw HDF5
    attributes scattered through the code.
    """

    schema_version: str
    unit_convention: str
    h_ref: float
    profile_name: str
    mass_definition_label: str
    mass_radius_kpc: float | None
    cosmology_h0: float | None
    cosmology_omega_m: float | None
    capabilities: frozenset[str]


@dataclass(frozen=True)
class CanonicalLenses:
    """Canonical per-lens observation arrays."""

    lens_id: tuple[str, ...]
    z_d: np.ndarray
    z_s: np.ndarray
    log_mstar_obs: np.ndarray
    log_mstar_err: np.ndarray
    log_re_obs: np.ndarray
    n_obs: np.ndarray
    theta_e_obs: np.ndarray
    num_sigma: np.ndarray
    sigma_obs: np.ndarray
    sigma_err: np.ndarray


@dataclass(frozen=True)
class CanonicalLensingMassGrids:
    """Per-lens gamma-axis mass tracks and optional velocity-dispersion grids."""

    gamma_grid: np.ndarray
    log_enclosed_mass_grid: np.ndarray
    dmass_dthetaein_grid: np.ndarray
    s2_grid: np.ndarray
    has_s2: np.ndarray


@dataclass(frozen=True)
class CanonicalCrossSectionGrid:
    """Unified two-dimensional theta_E x gamma lensing cross-section grid."""

    theta_e_axis: np.ndarray
    gamma_axis: np.ndarray
    cross_section_grid: np.ndarray
    boundary_policy: str


@dataclass(frozen=True)
class CanonicalSigmaGrid:
    """
    Optional sigma-unit interpolation grid.

    Both population sigma proxy grids and within-Re FP-prior grids use the same
    axis convention after preparation.  `zd_axis` and `n_axis` may be singleton
    compatibility axes when the physical table does not depend on that axis.
    """

    gamma_axis: np.ndarray
    zd_axis: np.ndarray
    log_re_axis: np.ndarray
    sigma_unit_grid: np.ndarray
    n_axis: np.ndarray


@dataclass(frozen=True)
class CanonicalVelocityDispersionGrids:
    """Velocity-dispersion capability blocks loaded from canonical input."""

    per_lens_s2_grid: np.ndarray | None
    per_lens_has_s2: np.ndarray | None
    fp_within_re: CanonicalSigmaGrid | None
    population_sigma_unit: CanonicalSigmaGrid | None


@dataclass(frozen=True)
class CanonicalInferenceDataset:
    """All canonical data blocks consumed by inference runtime adapters."""

    path: Path
    metadata: CanonicalMetadata
    lenses: CanonicalLenses
    mass_grids: CanonicalLensingMassGrids
    cross_section: CanonicalCrossSectionGrid
    velocity_dispersion: CanonicalVelocityDispersionGrids


def _decode_hdf5_string(value: Any) -> str:
    """Normalize HDF5 byte/string scalar encodings into a Python string."""

    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray) and value.shape == ():
        return _decode_hdf5_string(value.item())
    return str(value)


def _decode_string_dataset(dataset: h5py.Dataset) -> tuple[str, ...]:
    """Read one UTF-8 string dataset regardless of h5py's scalar type choice."""

    return tuple(_decode_hdf5_string(item) for item in np.atleast_1d(dataset[()]))


def _read_required_dataset(group: h5py.Group, dataset_name: str, *, dtype=None) -> np.ndarray:
    """Read one required dataset with a direct schema-oriented error message."""

    if dataset_name not in group:
        raise KeyError(f"Canonical block '{group.name}' is missing dataset '{dataset_name}'.")
    return np.asarray(group[dataset_name][()], dtype=dtype)


def _read_required_group(handle: h5py.File | h5py.Group, group_name: str) -> h5py.Group:
    """Return one required HDF5 group and fail before partial parsing."""

    if group_name not in handle:
        raise KeyError(f"Canonical dataset is missing required block '{group_name}'.")
    group = handle[group_name]
    if not isinstance(group, h5py.Group):
        raise TypeError(f"Canonical block '{group_name}' must be an HDF5 group.")
    return group


def _optional_float_attr(attrs: h5py.AttributeManager, name: str) -> float | None:
    """Read an optional numeric metadata attribute."""

    if name not in attrs:
        return None
    return float(attrs[name])


def _load_metadata(group: h5py.Group) -> CanonicalMetadata:
    """Load and minimally validate the `/metadata` block."""

    if "capabilities" not in group:
        raise KeyError("Canonical metadata is missing dataset 'capabilities'.")
    schema_version = _decode_hdf5_string(group.attrs.get("schema_version", ""))
    if schema_version != CANONICAL_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported canonical schema_version '{schema_version}'. "
            f"Expected '{CANONICAL_SCHEMA_VERSION}'."
        )
    return CanonicalMetadata(
        schema_version=schema_version,
        unit_convention=_decode_hdf5_string(group.attrs.get("unit_convention", "")),
        h_ref=float(group.attrs.get("h_ref", np.nan)),
        profile_name=_decode_hdf5_string(group.attrs.get("profile_name", "")),
        mass_definition_label=_decode_hdf5_string(group.attrs.get("mass_definition_label", "")),
        mass_radius_kpc=_optional_float_attr(group.attrs, "mass_radius_kpc"),
        cosmology_h0=_optional_float_attr(group.attrs, "cosmology_h0"),
        cosmology_omega_m=_optional_float_attr(group.attrs, "cosmology_omega_m"),
        capabilities=frozenset(_decode_string_dataset(group["capabilities"])),
    )


def _load_lenses(group: h5py.Group) -> CanonicalLenses:
    """Load the canonical lens observation block."""

    lens_ids = _decode_string_dataset(group["lens_id"])
    return CanonicalLenses(
        lens_id=lens_ids,
        z_d=_read_required_dataset(group, "z_d", dtype=np.float64),
        z_s=_read_required_dataset(group, "z_s", dtype=np.float64),
        log_mstar_obs=_read_required_dataset(group, "log_mstar_obs", dtype=np.float64),
        log_mstar_err=_read_required_dataset(group, "log_mstar_err", dtype=np.float64),
        log_re_obs=_read_required_dataset(group, "log_re_obs", dtype=np.float64),
        n_obs=_read_required_dataset(group, "n_obs", dtype=np.float64),
        theta_e_obs=_read_required_dataset(group, "theta_e_obs", dtype=np.float64),
        num_sigma=_read_required_dataset(group, "num_sigma", dtype=np.int64),
        sigma_obs=_read_required_dataset(group, "sigma_obs", dtype=np.float64),
        sigma_err=_read_required_dataset(group, "sigma_err", dtype=np.float64),
    )


def _load_mass_grids(group: h5py.Group) -> CanonicalLensingMassGrids:
    """Load per-lens lensing mass grids."""

    return CanonicalLensingMassGrids(
        gamma_grid=_read_required_dataset(group, "gamma_grid", dtype=np.float64),
        log_enclosed_mass_grid=_read_required_dataset(group, "log_enclosed_mass_grid", dtype=np.float64),
        dmass_dthetaein_grid=_read_required_dataset(group, "dmass_dthetaein_grid", dtype=np.float64),
        s2_grid=_read_required_dataset(group, "s2_grid", dtype=np.float64),
        has_s2=_read_required_dataset(group, "has_s2", dtype=np.int64),
    )


def _load_cross_section(group: h5py.Group) -> CanonicalCrossSectionGrid:
    """Load the unified theta_E x gamma cross-section grid."""

    return CanonicalCrossSectionGrid(
        theta_e_axis=_read_required_dataset(group, "theta_e_axis", dtype=np.float64),
        gamma_axis=_read_required_dataset(group, "gamma_axis", dtype=np.float64),
        cross_section_grid=_read_required_dataset(group, "cross_section_grid", dtype=np.float64),
        boundary_policy=_decode_hdf5_string(group.attrs.get("boundary_policy", "")),
    )


def _validate_one_dimensional_axis(axis: np.ndarray, *, axis_label: str) -> None:
    """
    Validate one interpolation axis used by canonical sigma-unit grids.

    Sigma-unit tables are later interpolated inside backend kernels, where
    unclear rank or empty-axis errors are difficult to diagnose.  The reader
    therefore validates these simple invariants immediately at dataset load
    time.
    """

    if axis.ndim != 1 or axis.size == 0:
        raise ValueError(f"{axis_label} must be a non-empty one-dimensional axis, got shape {axis.shape}.")


def _load_sigma_grid(
    group: h5py.Group,
    *,
    block_label: str,
    require_zd_axis: bool = False,
) -> CanonicalSigmaGrid:
    """
    Load and validate one optional sigma-unit sub-block.

    The canonical schema allows some sigma products, such as within-Re FP
    tables, to omit redshift or Sersic-index axes when the underlying table is
    independent of those coordinates.  Sonnenfeld's population sigma proxy,
    however, needs a real redshift axis to build `theta_E_est` during
    normalization, so callers can require `zd_axis` for that block.
    """

    gamma_axis = _read_required_dataset(group, "gamma_axis", dtype=np.float64)
    log_re_axis = _read_required_dataset(group, "log_re_kpc_axis", dtype=np.float64)
    sigma_unit_grid = _read_required_dataset(group, "s_unit_grid", dtype=np.float64)
    _validate_one_dimensional_axis(gamma_axis, axis_label=f"{block_label}.gamma_axis")
    _validate_one_dimensional_axis(log_re_axis, axis_label=f"{block_label}.log_re_kpc_axis")
    if require_zd_axis and "zd_axis" not in group:
        raise ValueError(f"{block_label} must define zd_axis for population-level sigma interpolation.")
    zd_axis = (
        _read_required_dataset(group, "zd_axis", dtype=np.float64)
        if "zd_axis" in group
        else np.asarray([0.0], dtype=np.float64)
    )
    n_axis = (
        _read_required_dataset(group, "n_axis", dtype=np.float64)
        if "n_axis" in group
        else np.asarray([4.0], dtype=np.float64)
    )
    _validate_one_dimensional_axis(zd_axis, axis_label=f"{block_label}.zd_axis")
    _validate_one_dimensional_axis(n_axis, axis_label=f"{block_label}.n_axis")
    expected_shape = (gamma_axis.size,)
    if "zd_axis" in group:
        expected_shape += (zd_axis.size,)
    expected_shape += (log_re_axis.size,)
    if "n_axis" in group:
        expected_shape += (n_axis.size,)
    if sigma_unit_grid.shape != expected_shape:
        raise ValueError(
            f"{block_label}.s_unit_grid must have shape {expected_shape}, "
            f"got {sigma_unit_grid.shape}."
        )
    return CanonicalSigmaGrid(
        gamma_axis=gamma_axis,
        zd_axis=zd_axis,
        log_re_axis=log_re_axis,
        sigma_unit_grid=sigma_unit_grid,
        n_axis=n_axis,
    )


def _load_velocity_dispersion(group: h5py.Group) -> CanonicalVelocityDispersionGrids:
    """Load optional velocity-dispersion sub-blocks."""

    per_lens_s2_grid = None
    per_lens_has_s2 = None
    if "per_lens_s2" in group:
        per_lens_group = group["per_lens_s2"]
        per_lens_s2_grid = _read_required_dataset(per_lens_group, "s2_grid", dtype=np.float64)
        per_lens_has_s2 = _read_required_dataset(per_lens_group, "has_s2", dtype=np.int64)

    return CanonicalVelocityDispersionGrids(
        per_lens_s2_grid=per_lens_s2_grid,
        per_lens_has_s2=per_lens_has_s2,
        fp_within_re=(
            _load_sigma_grid(group["fp_within_re"], block_label="velocity_dispersion_grids.fp_within_re")
            if "fp_within_re" in group
            else None
        ),
        population_sigma_unit=(
            _load_sigma_grid(
                group["population_sigma_unit"],
                block_label="velocity_dispersion_grids.population_sigma_unit",
                require_zd_axis=True,
            )
            if "population_sigma_unit" in group
            else None
        ),
    )


def _validate_metadata_expectations(
    metadata: CanonicalMetadata,
    *,
    expected_unit_convention: str,
    expected_h_ref: float,
    expected_profile_name: str,
    expected_mass_definition_label: str,
) -> None:
    """Validate config-vs-dataset metadata before any numerical setup."""

    if metadata.unit_convention != expected_unit_convention:
        raise ValueError(
            f"Canonical dataset unit_convention='{metadata.unit_convention}' does not match "
            f"configured unit_convention='{expected_unit_convention}'."
        )
    if not np.isclose(metadata.h_ref, expected_h_ref):
        raise ValueError(
            f"Canonical dataset h_ref={metadata.h_ref:g} does not match configured h_ref={expected_h_ref:g}."
        )
    if metadata.profile_name != expected_profile_name:
        raise ValueError(
            f"Canonical dataset profile_name='{metadata.profile_name}' does not match "
            f"configured profile='{expected_profile_name}'."
        )
    if metadata.mass_definition_label != expected_mass_definition_label:
        raise ValueError(
            f"Canonical dataset mass_definition_label='{metadata.mass_definition_label}' does not match "
            f"configured mass definition '{expected_mass_definition_label}'."
        )


def _validate_required_capabilities(
    metadata: CanonicalMetadata,
    required_capabilities: tuple[str, ...],
) -> None:
    """Ensure the active model's declared capabilities are present."""

    missing = sorted(set(required_capabilities).difference(metadata.capabilities))
    if missing:
        raise ValueError(f"Canonical dataset is missing required capabilities: {missing}.")


def _validate_velocity_capability_blocks(
    metadata: CanonicalMetadata,
    velocity_dispersion: CanonicalVelocityDispersionGrids,
) -> None:
    """
    Ensure declared velocity capabilities have matching loaded HDF5 blocks.

    Capability strings are the contract consumed by model runtimes.  If a file
    advertises a capability but omits the corresponding data block, a model can
    otherwise pass startup capability checks and fail later inside preprocessing
    or compiled-kernel setup.  Failing here keeps the error tied to the
    canonical schema.
    """

    capability_to_block = {
        CAPABILITY_VELOCITY_DISPERSION_PER_LENS_S2_V1: (
            velocity_dispersion.per_lens_s2_grid is not None
            and velocity_dispersion.per_lens_has_s2 is not None,
            "velocity_dispersion_grids/per_lens_s2",
        ),
        CAPABILITY_VELOCITY_DISPERSION_FP_WITHIN_RE_V1: (
            velocity_dispersion.fp_within_re is not None,
            "velocity_dispersion_grids/fp_within_re",
        ),
        CAPABILITY_VELOCITY_DISPERSION_POPULATION_SIGMA_UNIT_V1: (
            velocity_dispersion.population_sigma_unit is not None,
            "velocity_dispersion_grids/population_sigma_unit",
        ),
    }
    missing_blocks = [
        block_path
        for capability, (is_present, block_path) in capability_to_block.items()
        if capability in metadata.capabilities and not is_present
    ]
    if missing_blocks:
        raise ValueError(
            "Canonical dataset declares velocity-dispersion capabilities but "
            f"is missing matching HDF5 blocks: {missing_blocks}."
        )


def _validate_shapes(
    *,
    lenses: CanonicalLenses,
    mass_grids: CanonicalLensingMassGrids,
    cross_section: CanonicalCrossSectionGrid,
    velocity_dispersion: CanonicalVelocityDispersionGrids,
) -> None:
    """Validate cross-block array shapes that the runtime assumes are stable."""

    n_lens = len(lenses.lens_id)
    for field_name in (
        "z_d",
        "z_s",
        "log_mstar_obs",
        "log_mstar_err",
        "log_re_obs",
        "n_obs",
        "theta_e_obs",
        "num_sigma",
    ):
        value = getattr(lenses, field_name)
        if value.shape != (n_lens,):
            raise ValueError(f"lenses.{field_name} must have shape ({n_lens},), got {value.shape}.")
    if lenses.sigma_obs.ndim != 2 or lenses.sigma_obs.shape[0] != n_lens:
        raise ValueError("lenses.sigma_obs must have shape [N_lens, N_sigma_max].")
    if lenses.sigma_err.shape != lenses.sigma_obs.shape:
        raise ValueError("lenses.sigma_err must match lenses.sigma_obs shape.")

    if mass_grids.gamma_grid.ndim == 1:
        expected_grid_shape = (n_lens, mass_grids.gamma_grid.shape[0])
    elif mass_grids.gamma_grid.ndim == 2 and mass_grids.gamma_grid.shape[0] == n_lens:
        expected_grid_shape = mass_grids.gamma_grid.shape
    else:
        raise ValueError("lensing_mass_grids.gamma_grid must have shape [N_gamma] or [N_lens, N_gamma].")
    for field_name in ("log_enclosed_mass_grid", "dmass_dthetaein_grid", "s2_grid"):
        value = getattr(mass_grids, field_name)
        if value.shape != expected_grid_shape:
            raise ValueError(f"lensing_mass_grids.{field_name} must have shape {expected_grid_shape}.")
    if mass_grids.has_s2.shape != (n_lens,):
        raise ValueError("lensing_mass_grids.has_s2 must have shape [N_lens].")

    if cross_section.theta_e_axis.ndim != 1 or cross_section.gamma_axis.ndim != 1:
        raise ValueError("lensing_cross_section theta/gamma axes must be one-dimensional.")
    expected_cross_section_shape = (cross_section.theta_e_axis.size, cross_section.gamma_axis.size)
    if cross_section.cross_section_grid.shape != expected_cross_section_shape:
        raise ValueError(
            "lensing_cross_section.cross_section_grid must have shape "
            f"{expected_cross_section_shape}, got {cross_section.cross_section_grid.shape}."
        )

    missing_s2 = (lenses.num_sigma > 0) & (mass_grids.has_s2 == 0)
    if np.any(missing_s2):
        bad_indices = np.flatnonzero(missing_s2).tolist()
        raise ValueError(
            "Canonical dataset has lenses with num_sigma > 0 but has_s2 is false "
            f"at indices {bad_indices}."
        )

    if velocity_dispersion.per_lens_s2_grid is not None:
        if velocity_dispersion.per_lens_s2_grid.shape != mass_grids.s2_grid.shape:
            raise ValueError("velocity_dispersion_grids.per_lens_s2.s2_grid must match mass-grid s2 shape.")
        if velocity_dispersion.per_lens_has_s2 is None or velocity_dispersion.per_lens_has_s2.shape != (n_lens,):
            raise ValueError("velocity_dispersion_grids.per_lens_s2.has_s2 must have shape [N_lens].")


def load_canonical_inference_dataset(
    file_path: str | Path,
    *,
    expected_unit_convention: str,
    expected_h_ref: float,
    expected_profile_name: str,
    expected_mass_definition_label: str,
    required_capabilities: tuple[str, ...] = (),
) -> CanonicalInferenceDataset:
    """
    Read and validate one canonical inference dataset.

    The reader assumes data preparation has already normalized raw aliases and
    units.  It therefore rejects config/dataset mismatches and shape errors
    aggressively so model runtimes can build contexts without defensive HDF5
    handling.
    """

    path = Path(file_path).expanduser().resolve()
    with h5py.File(path, "r") as handle:
        metadata = _load_metadata(_read_required_group(handle, BLOCK_METADATA))
        lenses = _load_lenses(_read_required_group(handle, BLOCK_LENSES))
        mass_grids = _load_mass_grids(_read_required_group(handle, BLOCK_LENSING_MASS_GRIDS))
        cross_section = _load_cross_section(_read_required_group(handle, BLOCK_LENSING_CROSS_SECTION))
        velocity_dispersion = _load_velocity_dispersion(
            _read_required_group(handle, BLOCK_VELOCITY_DISPERSION_GRIDS)
        )

    _validate_metadata_expectations(
        metadata,
        expected_unit_convention=expected_unit_convention,
        expected_h_ref=expected_h_ref,
        expected_profile_name=expected_profile_name,
        expected_mass_definition_label=expected_mass_definition_label,
    )
    _validate_velocity_capability_blocks(metadata, velocity_dispersion)
    _validate_required_capabilities(metadata, required_capabilities)
    _validate_shapes(
        lenses=lenses,
        mass_grids=mass_grids,
        cross_section=cross_section,
        velocity_dispersion=velocity_dispersion,
    )
    return CanonicalInferenceDataset(
        path=path,
        metadata=metadata,
        lenses=lenses,
        mass_grids=mass_grids,
        cross_section=cross_section,
        velocity_dispersion=velocity_dispersion,
    )


__all__ = [
    "CANONICAL_SCHEMA_VERSION",
    "CAPABILITY_LENSING_CROSS_SECTION_THETA_GAMMA_V1",
    "CAPABILITY_LENSING_MASS_GRIDS_V1",
    "CAPABILITY_LENS_OBSERVATIONS_V1",
    "CAPABILITY_VELOCITY_DISPERSION_FP_WITHIN_RE_V1",
    "CAPABILITY_VELOCITY_DISPERSION_PER_LENS_S2_V1",
    "CAPABILITY_VELOCITY_DISPERSION_POPULATION_SIGMA_UNIT_V1",
    "CanonicalCrossSectionGrid",
    "CanonicalInferenceDataset",
    "CanonicalLenses",
    "CanonicalLensingMassGrids",
    "CanonicalMetadata",
    "CanonicalSigmaGrid",
    "CanonicalVelocityDispersionGrids",
    "load_canonical_inference_dataset",
]
