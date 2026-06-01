"""Provenance helpers for the direct canonical dataset builder."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from statistical_sl.data_preparation.direct_pipeline.cross_sections import CrossSectionProvenance
from statistical_sl.data_preparation.direct_pipeline.sigma_resolver import SigmaResolutionAudit
from statistical_sl.data_preparation.direct_pipeline.measurements import VelocityMeasurementReadResult, VelocityMeasurementRejectedRow


def _freeze_mapping(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    """Return a shallow read-only mapping for audit payloads."""

    return MappingProxyType(dict(value or {}))


def serialize_rejected_measurement_row(row: VelocityMeasurementRejectedRow) -> dict[str, Any]:
    """Serialize one rejected measurement row into JSON/HDF5-friendly data."""

    return {
        "lens_id": row.lens_id,
        "reason": row.reason,
        "source_path": str(row.source_path),
        "raw_row": dict(row.raw_row),
    }


@dataclass(frozen=True)
class DirectPipelineProvenance:
    """High-level audit inputs assembled by the direct canonical pipeline."""

    catalog_path: Path
    catalog_type: str
    measurement_path: Path | None
    measurement_mode: str
    ignored_catalog_columns: Mapping[str, str] = field(default_factory=dict)
    rejected_measurements: tuple[VelocityMeasurementRejectedRow, ...] = ()
    num_sigma_distribution: Mapping[int, int] = field(default_factory=dict)
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalize file paths and freeze nested audit structures."""

        object.__setattr__(self, "catalog_path", Path(self.catalog_path).expanduser().resolve())
        if self.measurement_path is not None:
            object.__setattr__(self, "measurement_path", Path(self.measurement_path).expanduser().resolve())
        object.__setattr__(self, "ignored_catalog_columns", _freeze_mapping(self.ignored_catalog_columns))
        object.__setattr__(self, "num_sigma_distribution", _freeze_mapping(self.num_sigma_distribution))
        object.__setattr__(self, "extra", _freeze_mapping(self.extra))


def build_direct_pipeline_provenance(
    *,
    provenance: DirectPipelineProvenance,
    measurement_result: VelocityMeasurementReadResult,
    sigma_resolution_audit: SigmaResolutionAudit,
    cross_section_provenance: CrossSectionProvenance,
) -> dict[str, Any]:
    """Convert provenance objects into a payload-ready nested mapping."""

    catalog_entry = {
        "source_path": str(provenance.catalog_path),
        "catalog_type": provenance.catalog_type,
        "ignored_columns": dict(provenance.ignored_catalog_columns),
        "num_sigma_distribution": dict(provenance.num_sigma_distribution),
        "extra": dict(provenance.extra),
    }
    measurement_entry = {
        "source_path": None if provenance.measurement_path is None else str(provenance.measurement_path),
        "measurement_mode": provenance.measurement_mode,
        "row_count": int(
            measurement_result.provenance.get(
                "row_count",
                len(measurement_result.accepted) + len(measurement_result.rejected),
            )
        ),
        "rejected_rows": [serialize_rejected_measurement_row(row) for row in measurement_result.rejected],
        "extra": dict(measurement_result.provenance),
    }
    resolver_entry = {
        "source_type": sigma_resolution_audit.source_type,
        "num_sigma_distribution": dict(sigma_resolution_audit.num_sigma_distribution),
        "missing_lens_ids": list(sigma_resolution_audit.missing_lens_ids),
        "rejected_measurements": [
            serialize_rejected_measurement_row(row) for row in sigma_resolution_audit.rejected_measurements
        ],
        "extra": dict(sigma_resolution_audit.extra),
    }
    cross_section_entry = {
        "source_path": str(cross_section_provenance.source_path),
        "source_mode": cross_section_provenance.source_mode,
        "source_dataset": cross_section_provenance.source_dataset,
        "extra": dict(cross_section_provenance.extra),
    }
    return {
        "catalog": catalog_entry,
        "measurement": measurement_entry,
        "resolver": resolver_entry,
        "cross_section": cross_section_entry,
    }


__all__ = [
    "DirectPipelineProvenance",
    "build_direct_pipeline_provenance",
    "serialize_rejected_measurement_row",
]
