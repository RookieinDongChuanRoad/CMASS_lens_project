"""In-memory derived-grid builders for the direct canonical pipeline.

This module is the replacement for the current compatibility orchestration pattern where mass
and Jeans grids were first materialized into intermediate observation HDF5
files and then repacked.  The numerical formulas are still owned by the tested
physics modules; this layer only coordinates inputs, validates preparation
state, and assembles rectangular arrays for the canonical payload.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import numpy as np

from statistical_sl.data_preparation.direct_pipeline.policies import MassDefinitionPolicy, UnitPolicy
from statistical_sl.data_preparation.direct_pipeline.records import PreparedLensRecord
from statistical_sl.data_preparation.config import DEFAULT_DERIVATIVE_THETA_SAMPLES, OBSERVED_APERTURE_SIGMA_DEFINITION
from statistical_sl.data_preparation.physics.jeans import compute_sigma_unit_grid
from statistical_sl.data_preparation.physics.m5 import compute_dmass_dthetaein_grid, compute_mass_grid


def _validate_gamma_axis(gamma_axis: np.ndarray) -> np.ndarray:
    """Return a finite, one-dimensional gamma axis used by all grid blocks."""

    axis = np.asarray(gamma_axis, dtype=float)
    if axis.ndim != 1 or axis.size == 0:
        raise ValueError("gamma_axis must be a non-empty one-dimensional array.")
    if not np.all(np.isfinite(axis)):
        raise ValueError("gamma_axis must contain only finite values.")
    return axis


def _require_prepared_float(record: PreparedLensRecord, attr_name: str) -> float:
    """Read one required scalar from a prepared record with a clear error."""

    value = getattr(record, attr_name)
    if value is None:
        raise ValueError(f"{record.lens_id} is missing prepared field {attr_name!r}.")
    return float(value)


def _validate_record_unit_policy(record: PreparedLensRecord, unit_policy: UnitPolicy) -> None:
    """Ensure one prepared record matches the builder's unit policy."""

    if record.unit_convention is not None and record.unit_convention != unit_policy.unit_convention:
        raise ValueError(
            f"{record.lens_id} unit_convention={record.unit_convention!r} does not match "
            f"builder unit_convention={unit_policy.unit_convention!r}."
        )
    if record.h_ref is not None and not np.isclose(float(record.h_ref), float(unit_policy.h_ref)):
        raise ValueError(
            f"{record.lens_id} h_ref={record.h_ref!r} does not match builder h_ref={unit_policy.h_ref!r}."
        )


@dataclass(frozen=True)
class LensingMassGridBlock:
    """Mass-grid arrays aligned to the prepared lens order."""

    lens_ids: tuple[str, ...]
    gamma_axis: np.ndarray
    log_enclosed_mass_grid: np.ndarray
    dmass_dthetaein_grid: np.ndarray
    mass_definition_label: str
    mass_radius_kpc: float
    unit_convention: str
    h_ref: float


@dataclass(frozen=True)
class VelocityDispersionGridBlock:
    """Per-lens velocity-dispersion response grids aligned to the lens order."""

    lens_ids: tuple[str, ...]
    gamma_axis: np.ndarray
    s2_grid: np.ndarray
    has_s2: np.ndarray
    mass_definition_label: str
    mass_radius_kpc: float
    unit_convention: str
    h_ref: float


@dataclass(frozen=True)
class DerivedGridBlocks:
    """All derived grid blocks generated from prepared lens records."""

    mass: LensingMassGridBlock
    velocity: VelocityDispersionGridBlock


def build_lensing_mass_grid_block(
    records: Sequence[PreparedLensRecord],
    *,
    gamma_axis: np.ndarray,
    mass_policy: MassDefinitionPolicy,
    unit_policy: UnitPolicy,
    derivative_theta_samples: int = DEFAULT_DERIVATIVE_THETA_SAMPLES,
) -> LensingMassGridBlock:
    """Build in-memory projected-mass grids for every prepared lens."""

    axis = _validate_gamma_axis(gamma_axis)
    lens_ids: list[str] = []
    mass_rows: list[np.ndarray] = []
    derivative_rows: list[np.ndarray] = []

    for record in records:
        _validate_record_unit_policy(record, unit_policy)
        sigma_crit = _require_prepared_float(record, "sigma_crit")
        theta_ein_kpc = _require_prepared_float(record, "theta_ein_kpc")
        theta_scale_kpc_per_arcsec = theta_ein_kpc / float(record.base_lens.theta_ein_arcsec)

        lens_ids.append(record.lens_id)
        mass_rows.append(
            np.asarray(
                compute_mass_grid(
                    gamma_grid=axis,
                    sigma_crit=sigma_crit,
                    rein_kpc=theta_ein_kpc,
                    mass_radius_kpc=mass_policy.mass_radius_kpc,
                    unit_convention=unit_policy.unit_convention,
                    h_ref=unit_policy.h_ref,
                ),
                dtype=float,
            )
        )
        derivative_rows.append(
            np.asarray(
                compute_dmass_dthetaein_grid(
                    gamma_grid=axis,
                    sigma_crit=sigma_crit,
                    theta_ein_arcsec=record.base_lens.theta_ein_arcsec,
                    kpc_per_arcsec=theta_scale_kpc_per_arcsec,
                    theta_samples=int(derivative_theta_samples),
                    mass_radius_kpc=mass_policy.mass_radius_kpc,
                ),
                dtype=float,
            )
        )

    return LensingMassGridBlock(
        lens_ids=tuple(lens_ids),
        gamma_axis=axis,
        log_enclosed_mass_grid=np.vstack(mass_rows) if mass_rows else np.empty((0, axis.size), dtype=float),
        dmass_dthetaein_grid=np.vstack(derivative_rows) if derivative_rows else np.empty((0, axis.size), dtype=float),
        mass_definition_label=mass_policy.mass_definition_label,
        mass_radius_kpc=mass_policy.mass_radius_kpc,
        unit_convention=unit_policy.unit_convention,
        h_ref=unit_policy.h_ref,
    )


def _n_value_for_record(record: PreparedLensRecord) -> float | None:
    """Return the Sersic index only for the free-Sersic branch."""

    if record.base_lens.profile_name == "devauc":
        return None
    return record.base_lens.sersic_index


def build_velocity_dispersion_grid_block(
    records: Sequence[PreparedLensRecord],
    *,
    gamma_axis: np.ndarray,
    mass_policy: MassDefinitionPolicy,
    unit_policy: UnitPolicy,
) -> VelocityDispersionGridBlock:
    """Build per-lens ``s2_grid`` rows only where sigma likelihood data exists."""

    axis = _validate_gamma_axis(gamma_axis)
    lens_ids: list[str] = []
    s2_rows: list[np.ndarray] = []
    has_s2: list[bool] = []

    for record in records:
        _validate_record_unit_policy(record, unit_policy)
        lens_ids.append(record.lens_id)

        if record.num_sigma == 0:
            s2_rows.append(np.zeros_like(axis, dtype=float))
            has_s2.append(False)
            continue

        if record.aperture_policy is None:
            raise ValueError(f"{record.lens_id} requires an aperture_policy before building s2_grid.")
        effective_radius_kpc = _require_prepared_float(record, "effective_radius_kpc")
        s2_rows.append(
            np.asarray(
                compute_sigma_unit_grid(
                    profile_name=record.base_lens.profile_name,
                    gamma_grid=axis,
                    zd=record.base_lens.z_lens,
                    re_kpc=effective_radius_kpc,
                    n_value=_n_value_for_record(record),
                    mass_radius_kpc=mass_policy.mass_radius_kpc,
                    aperture_policy=record.aperture_policy,
                    sigma_definition=record.sigma_definition or OBSERVED_APERTURE_SIGMA_DEFINITION,
                    unit_convention=unit_policy.unit_convention,
                    h_ref=unit_policy.h_ref,
                ),
                dtype=float,
            )
        )
        has_s2.append(True)

    return VelocityDispersionGridBlock(
        lens_ids=tuple(lens_ids),
        gamma_axis=axis,
        s2_grid=np.vstack(s2_rows) if s2_rows else np.empty((0, axis.size), dtype=float),
        has_s2=np.asarray(has_s2, dtype=bool),
        mass_definition_label=mass_policy.mass_definition_label,
        mass_radius_kpc=mass_policy.mass_radius_kpc,
        unit_convention=unit_policy.unit_convention,
        h_ref=unit_policy.h_ref,
    )


def build_derived_grid_blocks(
    records: Sequence[PreparedLensRecord],
    *,
    gamma_axis: np.ndarray,
    mass_policy: MassDefinitionPolicy,
    unit_policy: UnitPolicy,
    derivative_theta_samples: int = DEFAULT_DERIVATIVE_THETA_SAMPLES,
) -> DerivedGridBlocks:
    """Build all per-lens derived grids directly from prepared records."""

    axis = _validate_gamma_axis(gamma_axis)
    mass_block = build_lensing_mass_grid_block(
        records,
        gamma_axis=axis,
        mass_policy=mass_policy,
        unit_policy=unit_policy,
        derivative_theta_samples=derivative_theta_samples,
    )
    velocity_block = build_velocity_dispersion_grid_block(
        records,
        gamma_axis=axis,
        mass_policy=mass_policy,
        unit_policy=unit_policy,
    )
    return DerivedGridBlocks(mass=mass_block, velocity=velocity_block)


__all__ = [
    "DerivedGridBlocks",
    "LensingMassGridBlock",
    "VelocityDispersionGridBlock",
    "build_derived_grid_blocks",
    "build_lensing_mass_grid_block",
    "build_velocity_dispersion_grid_block",
]
