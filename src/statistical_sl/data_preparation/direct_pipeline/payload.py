"""Assemble the direct canonical dataset payload from in-memory blocks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from statistical_sl.data_preparation.direct_pipeline.cross_sections import CrossSectionBlock
from statistical_sl.data_preparation.direct_pipeline.grid_builders import LensingMassGridBlock, VelocityDispersionGridBlock
from statistical_sl.data_preparation.direct_pipeline.policies import MassDefinitionPolicy, UnitPolicy
from statistical_sl.data_preparation.direct_pipeline.provenance import (
    DirectPipelineProvenance,
    build_direct_pipeline_provenance,
)
from statistical_sl.data_preparation.direct_pipeline.records import CanonicalDatasetPayload, PreparedLensRecord
from statistical_sl.data_preparation.direct_pipeline.sigma_resolver import SigmaResolutionAudit
from statistical_sl.data_preparation.direct_pipeline.measurements import VelocityMeasurementReadResult
from statistical_sl.core.canonical_schema import (
    CAPABILITY_LENS_OBSERVATIONS_V1,
    CAPABILITY_LENSING_CROSS_SECTION_THETA_GAMMA_V1,
    CAPABILITY_LENSING_MASS_GRIDS_V1,
    CAPABILITY_VELOCITY_DISPERSION_PER_LENS_S2_V1,
)


PAYLOAD_SCHEMA_VERSION = "canonical_dataset_payload_v1"


def _validate_consistent_scalar(values: Sequence[Any], field_name: str) -> Any:
    """Return the shared scalar value for a field that must not vary by lens."""

    unique_values = {value for value in values}
    if len(unique_values) != 1:
        raise ValueError(f"All prepared lenses must share one {field_name}.")
    return next(iter(unique_values))


def _validate_consistent_present_scalar(values: Sequence[Any], field_name: str) -> Any:
    """Return one shared non-empty scalar while allowing sigma-free lenses.

    Direct builds may keep lenses with ``num_sigma = 0`` in the sample.  Those
    lenses do not define an observed-aperture likelihood datum, so their
    preparation fields can be empty without changing the file-level semantic
    label used by sigma-bearing lenses.
    """

    present_values = [value for value in values if value not in (None, "")]
    if not present_values:
        return None
    return _validate_consistent_scalar(present_values, field_name)


def _string_array(values: Sequence[str]) -> np.ndarray:
    """Return a UTF-8 string array suitable for later HDF5 writing."""

    return np.asarray(values, dtype=object)


def _optional_path_text(value: Path | None) -> str | None:
    """Convert an optional path into the JSON-friendly text form used by metadata."""

    if value is None:
        return None
    return str(Path(value).expanduser().resolve())


def _aperture_numeric_value(record: PreparedLensRecord, attr_name: str) -> float:
    """Return one aperture scalar, using zero for geometry fields that do not apply.

    HDF5 validators intentionally reject non-finite numeric datasets.  Aperture
    geometry still has optional fields because rectangular and circular
    apertures expose different dimensions.  We therefore encode non-applicable
    dimensions as ``0.0`` and let ``aperture_shape`` carry the semantics of
    which fields are meaningful.
    """

    if record.aperture_policy is None:
        return 0.0
    value = getattr(record.aperture_policy, attr_name)
    return 0.0 if value is None else float(value)


def _build_lenses_block(records: Sequence[PreparedLensRecord]) -> dict[str, np.ndarray]:
    """Convert prepared lens records into the canonical lenses payload block."""

    sigma_obs = np.zeros((len(records), 2), dtype=float)
    sigma_err = np.ones((len(records), 2), dtype=float)

    for index, record in enumerate(records):
        for obs_index, observation in enumerate(record.sigma_observations):
            sigma_obs[index, obs_index] = float(observation.sigma_kms)
            sigma_err[index, obs_index] = float(observation.sigma_err_kms)

    return {
        "lens_id": _string_array([record.lens_id for record in records]),
        "z_d": np.asarray([record.base_lens.z_lens for record in records], dtype=float),
        "z_s": np.asarray([record.base_lens.z_source for record in records], dtype=float),
        "theta_ein_arcsec": np.asarray([record.base_lens.theta_ein_arcsec for record in records], dtype=float),
        "theta_ein_kpc": np.asarray([record.theta_ein_kpc for record in records], dtype=float),
        "effective_radius_arcsec": np.asarray([record.base_lens.effective_radius_arcsec for record in records], dtype=float),
        "effective_radius_kpc": np.asarray([record.effective_radius_kpc for record in records], dtype=float),
        "log_stellar_mass": np.asarray([record.base_lens.log_stellar_mass for record in records], dtype=float),
        "active_log_stellar_mass": np.asarray([record.active_log_stellar_mass for record in records], dtype=float),
        "active_log_effective_radius": np.asarray([record.active_log_effective_radius for record in records], dtype=float),
        "sigma_crit": np.asarray([record.sigma_crit for record in records], dtype=float),
        "num_sigma": np.asarray([record.num_sigma for record in records], dtype=np.int64),
        "sigma_obs": sigma_obs,
        "sigma_err": sigma_err,
        "observation_flavor": _string_array([record.observation_flavor or "" for record in records]),
        "sigma_definition": _string_array([record.sigma_definition or "" for record in records]),
        "aperture_shape": _string_array([record.aperture_policy.shape if record.aperture_policy else "" for record in records]),
        "aperture_width_arcsec": np.asarray(
            [_aperture_numeric_value(record, "width_arcsec") for record in records],
            dtype=float,
        ),
        "aperture_height_arcsec": np.asarray(
            [_aperture_numeric_value(record, "height_arcsec") for record in records],
            dtype=float,
        ),
        "aperture_radius_arcsec": np.asarray(
            [_aperture_numeric_value(record, "radius_arcsec") for record in records],
            dtype=float,
        ),
        "seeing_fwhm_arcsec": np.asarray(
            [_aperture_numeric_value(record, "seeing_fwhm_arcsec") for record in records],
            dtype=float,
        ),
    }


def build_canonical_dataset_payload(
    *,
    prepared_records: Sequence[PreparedLensRecord],
    mass_block: LensingMassGridBlock,
    velocity_block: VelocityDispersionGridBlock,
    cross_section_block: CrossSectionBlock,
    catalog_provenance: DirectPipelineProvenance,
    measurement_result: VelocityMeasurementReadResult,
    sigma_resolution_audit: SigmaResolutionAudit,
    unit_policy: UnitPolicy,
    mass_policy: MassDefinitionPolicy,
) -> CanonicalDatasetPayload:
    """Assemble the final in-memory payload for the canonical writer."""

    records = tuple(prepared_records)
    if not records:
        raise ValueError("prepared_records must not be empty.")

    record_lens_ids = tuple(record.lens_id for record in records)
    if mass_block.lens_ids != record_lens_ids or velocity_block.lens_ids != record_lens_ids:
        raise ValueError("Derived grid blocks must follow the same lens order as prepared_records.")

    profile_name = _validate_consistent_scalar([record.base_lens.profile_name for record in records], "profile_name")
    observation_flavor = _validate_consistent_present_scalar(
        [record.observation_flavor for record in records],
        "observation_flavor",
    )
    sigma_definition = _validate_consistent_present_scalar(
        [record.sigma_definition for record in records],
        "sigma_definition",
    )

    metadata = {
        "schema_version": PAYLOAD_SCHEMA_VERSION,
        "profile_name": profile_name,
        "unit_convention": unit_policy.unit_convention,
        "h_ref": float(unit_policy.h_ref),
        "mass_definition_label": mass_policy.mass_definition_label,
        "mass_radius_kpc": float(mass_policy.mass_radius_kpc),
        "observation_flavor": observation_flavor,
        "sigma_definition": sigma_definition,
        "aperture_contract": "per_lens",
        "lens_count": len(records),
        "num_sigma_distribution": dict(sigma_resolution_audit.num_sigma_distribution),
        "capabilities": (
            CAPABILITY_LENS_OBSERVATIONS_V1,
            CAPABILITY_LENSING_MASS_GRIDS_V1,
            CAPABILITY_LENSING_CROSS_SECTION_THETA_GAMMA_V1,
            CAPABILITY_VELOCITY_DISPERSION_PER_LENS_S2_V1,
        ),
        "catalog_path": _optional_path_text(catalog_provenance.catalog_path),
        "measurement_path": _optional_path_text(catalog_provenance.measurement_path),
        "cross_section_source_mode": cross_section_block.provenance.source_mode,
    }

    provenance = build_direct_pipeline_provenance(
        provenance=catalog_provenance,
        measurement_result=measurement_result,
        sigma_resolution_audit=sigma_resolution_audit,
        cross_section_provenance=cross_section_block.provenance,
    )

    return CanonicalDatasetPayload(
        metadata=metadata,
        lenses=_build_lenses_block(records),
        lensing_mass_grids={
            "lens_id": _string_array(list(mass_block.lens_ids)),
            "gamma_axis": np.asarray(mass_block.gamma_axis, dtype=float),
            "log_enclosed_mass_grid": np.asarray(mass_block.log_enclosed_mass_grid, dtype=float),
            "dmass_dthetaein_grid": np.asarray(mass_block.dmass_dthetaein_grid, dtype=float),
            "mass_definition_label": mass_block.mass_definition_label,
            "mass_radius_kpc": float(mass_block.mass_radius_kpc),
            "unit_convention": mass_block.unit_convention,
            "h_ref": float(mass_block.h_ref),
        },
        lensing_cross_section={
            "theta_e_axis": np.asarray(cross_section_block.theta_e_axis, dtype=float),
            "gamma_axis": np.asarray(cross_section_block.gamma_axis, dtype=float),
            "cross_section_grid": np.asarray(cross_section_block.cross_section_grid, dtype=float),
            "source_mode": cross_section_block.provenance.source_mode,
            "source_dataset": cross_section_block.provenance.source_dataset,
        },
        velocity_dispersion_grids={
            "lens_id": _string_array(list(velocity_block.lens_ids)),
            "gamma_axis": np.asarray(velocity_block.gamma_axis, dtype=float),
            "s2_grid": np.asarray(velocity_block.s2_grid, dtype=float),
            "has_s2": np.asarray(velocity_block.has_s2, dtype=bool),
            "mass_definition_label": velocity_block.mass_definition_label,
            "mass_radius_kpc": float(velocity_block.mass_radius_kpc),
            "unit_convention": velocity_block.unit_convention,
            "h_ref": float(velocity_block.h_ref),
        },
        provenance=provenance,
    )


__all__ = [
    "PAYLOAD_SCHEMA_VERSION",
    "build_canonical_dataset_payload",
]
