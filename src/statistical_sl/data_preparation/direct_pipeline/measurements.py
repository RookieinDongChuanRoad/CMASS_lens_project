"""Trusted velocity-dispersion measurement readers and adapters.

This module owns the measurement-truth boundary for the direct pipeline.

There are two supported entrypoints:

1. ``velocity_measurements_v1`` CSV files emitted by an upstream measurement
   pipeline.
2. A compatibility adapter for the current pPXF CSV export used by the local
   spectroscopy workflow.

Both paths resolve into the same accepted-observation model, while rejected
rows remain visible in audit provenance instead of being silently dropped.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from statistical_sl.data_preparation.direct_pipeline.records import SigmaObservation


VELOCITY_MEASUREMENT_SCHEMA_VERSION = "velocity_measurements_v1"
VELOCITY_MEASUREMENT_STANDARD_COLUMNS = (
    "schema_version",
    "lens_id",
    "obs_tag",
    "sigma_kms",
    "sigma_err_kms",
    "sigma_error_kind",
    "measurement_status",
    "use_for_likelihood",
    "source_system",
    "source_file",
    "aperture_shape",
    "aperture_width_arcsec",
    "aperture_height_arcsec",
    "aperture_radius_arcsec",
    "seeing_fwhm_arcsec",
)
PPXF_DEFAULT_ERROR_COLUMN = "sigma_stat_kms"
SUPPORTED_PPXF_ERROR_COLUMNS = frozenset({"sigma_stat_kms", "sigma_total_kms"})


def _freeze_mapping(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    """Return a shallow read-only mapping for audit payloads."""

    return MappingProxyType(dict(value or {}))


def _normalize_required_text(value: str, field_name: str) -> str:
    """Return a stripped string and reject empty required fields."""

    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{field_name} must be a non-empty string.")
    return normalized


def _normalize_optional_text(value: str | None) -> str | None:
    """Normalize optional text fields while preserving explicit absence."""

    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _parse_bool(value: str | None, *, field_name: str, default: bool = False) -> bool:
    """Parse a permissive boolean token used by CSV inputs."""

    if value is None:
        return default
    normalized = str(value).strip().lower()
    if not normalized:
        return default
    if normalized in {"1", "true", "t", "yes", "y"}:
        return True
    if normalized in {"0", "false", "f", "no", "n"}:
        return False
    raise ValueError(f"{field_name} must be a boolean-like token, got {value!r}.")


def _require_positive_finite(value: Any, field_name: str) -> float:
    """Validate one measurement scalar that must be positive and finite."""

    numeric_value = float(value)
    if not np.isfinite(numeric_value) or numeric_value <= 0.0:
        raise ValueError(f"{field_name} must be a positive finite number.")
    return numeric_value


def _require_finite(value: Any, field_name: str) -> float:
    """Validate one scalar that must be finite but may be signed."""

    numeric_value = float(value)
    if not np.isfinite(numeric_value):
        raise ValueError(f"{field_name} must be finite.")
    return numeric_value


@dataclass(frozen=True)
class VelocityMeasurementRejectedRow:
    """One measurement row excluded from the canonical likelihood.

    Rejected rows remain part of the audit trail so the pipeline can explain
    why a lens ended up with ``num_sigma = 0`` or why a specific upstream fit
    was not used.
    """

    lens_id: str
    reason: str
    source_path: Path
    raw_row: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalize identity and freeze the audit payload."""

        object.__setattr__(self, "lens_id", _normalize_required_text(self.lens_id, "lens_id"))
        object.__setattr__(self, "reason", _normalize_required_text(self.reason, "reason"))
        object.__setattr__(self, "source_path", Path(self.source_path).expanduser().resolve())
        object.__setattr__(self, "raw_row", _freeze_mapping(self.raw_row))


@dataclass(frozen=True)
class VelocityMeasurementReadResult:
    """Accepted and rejected measurement rows loaded from one source file."""

    accepted: tuple[SigmaObservation, ...]
    rejected: tuple[VelocityMeasurementRejectedRow, ...]
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Freeze provenance for stable downstream handoff."""

        object.__setattr__(self, "provenance", _freeze_mapping(self.provenance))


@dataclass(frozen=True)
class PpxfAdapterConfig:
    """Configuration for mapping the current pPXF CSV into the measurement API."""

    error_column: str = PPXF_DEFAULT_ERROR_COLUMN
    reject_failed_rows: bool = True

    def __post_init__(self) -> None:
        """Validate the selected uncertainty column."""

        error_column = _normalize_required_text(self.error_column, "error_column")
        if error_column not in SUPPORTED_PPXF_ERROR_COLUMNS:
            raise ValueError(
                f"error_column must be one of {sorted(SUPPORTED_PPXF_ERROR_COLUMNS)}, got {error_column!r}."
            )
        object.__setattr__(self, "error_column", error_column)
        object.__setattr__(self, "reject_failed_rows", bool(self.reject_failed_rows))


def _read_csv_dicts(csv_path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Read a CSV file into dictionaries using only the standard library."""

    resolved_path = Path(csv_path).expanduser().resolve()
    with resolved_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{resolved_path} has no header row.")
        rows = [dict(row) for row in reader]
        return list(reader.fieldnames), rows


def _row_float(row: Mapping[str, str], column: str, path: Path, lens_id: str) -> float:
    """Read one numeric CSV field with a lens-aware error message."""

    try:
        return _require_finite(row[column], column)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{path} contains an invalid {column!r} value for lens {lens_id}.") from exc


def _row_positive_float(row: Mapping[str, str], column: str, path: Path, lens_id: str) -> float:
    """Read one strictly positive numeric CSV field."""

    try:
        return _require_positive_finite(row[column], column)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{path} contains an invalid positive {column!r} value for lens {lens_id}.") from exc


def _row_optional_positive_float(row: Mapping[str, str], column: str, path: Path, lens_id: str) -> float | None:
    """Read one optional positive field used by aperture geometry columns.

    CSV fixtures and upstream exports commonly represent missing optional
    fields as blank strings.  Treating blanks as ``None`` keeps circular and
    rectangular aperture contracts explicit without forcing meaningless zeroes
    into the accepted measurement model.
    """

    raw_value = row.get(column)
    if raw_value is None or str(raw_value).strip() == "":
        return None
    try:
        return _require_positive_finite(raw_value, column)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path} contains an invalid positive {column!r} value for lens {lens_id}.") from exc


def _aperture_kwargs_from_row(row: Mapping[str, str], *, source_path: Path, lens_id: str) -> dict[str, object]:
    """Extract the required accepted-measurement aperture contract.

    Accepted velocity-dispersion rows define a likelihood datum, not merely a
    sigma scalar.  The aperture and seeing are therefore mandatory here so the
    direct pipeline can build per-lens observed-aperture `s2_grid` rows without
    silently borrowing a dataset-level default.
    """

    aperture_shape = _normalize_optional_text(row.get("aperture_shape"))
    if aperture_shape is None:
        raise ValueError(f"{source_path} lens {lens_id} is missing required aperture_shape.")
    normalized_shape = aperture_shape.lower()

    width_arcsec = _row_optional_positive_float(row, "aperture_width_arcsec", source_path, lens_id)
    height_arcsec = _row_optional_positive_float(row, "aperture_height_arcsec", source_path, lens_id)
    radius_arcsec = _row_optional_positive_float(row, "aperture_radius_arcsec", source_path, lens_id)
    seeing_fwhm_arcsec = _row_optional_positive_float(row, "seeing_fwhm_arcsec", source_path, lens_id)
    if seeing_fwhm_arcsec is None:
        raise ValueError(f"{source_path} lens {lens_id} is missing required seeing_fwhm_arcsec.")

    if normalized_shape == "rectangular":
        if width_arcsec is None:
            raise ValueError(f"{source_path} lens {lens_id} is missing required aperture_width_arcsec.")
        if height_arcsec is None:
            raise ValueError(f"{source_path} lens {lens_id} is missing required aperture_height_arcsec.")
        if radius_arcsec is not None:
            raise ValueError(
                f"{source_path} lens {lens_id} must leave aperture_radius_arcsec blank for rectangular apertures."
            )
    elif normalized_shape == "circular":
        if radius_arcsec is None:
            raise ValueError(f"{source_path} lens {lens_id} is missing required aperture_radius_arcsec.")
        if width_arcsec is not None:
            raise ValueError(
                f"{source_path} lens {lens_id} must leave aperture_width_arcsec blank for circular apertures."
            )
        if height_arcsec is not None:
            raise ValueError(
                f"{source_path} lens {lens_id} must leave aperture_height_arcsec blank for circular apertures."
            )
    else:
        raise ValueError(f"{source_path} lens {lens_id} has unsupported aperture_shape={aperture_shape!r}.")

    return {
        "aperture_shape": normalized_shape,
        "aperture_width_arcsec": width_arcsec,
        "aperture_height_arcsec": height_arcsec,
        "aperture_radius_arcsec": radius_arcsec,
        "seeing_fwhm_arcsec": seeing_fwhm_arcsec,
    }


def _accepted_observation_from_row(
    row: Mapping[str, str],
    *,
    source_path: Path,
    sigma_err_kms: float,
) -> SigmaObservation:
    """Convert one vetted CSV row into the canonical measurement record."""

    lens_id = _normalize_required_text(row["lens_id"], "lens_id")
    obs_tag = _normalize_optional_text(row.get("obs_tag"))
    measurement_status = _normalize_required_text(row.get("measurement_status", "success"), "measurement_status")
    source_system = _normalize_optional_text(row.get("source_system"))
    source_file = _normalize_optional_text(row.get("source_file"))
    try:
        return SigmaObservation(
            lens_id=lens_id,
            obs_tag=obs_tag,
            sigma_kms=_row_positive_float(row, "sigma_kms", source_path, lens_id),
            sigma_err_kms=sigma_err_kms,
            sigma_error_kind=_normalize_required_text(row.get("sigma_error_kind", "statistical"), "sigma_error_kind"),
            measurement_status=measurement_status,
            use_for_likelihood=True,
            source_system=source_system,
            source_file=source_file,
            **_aperture_kwargs_from_row(row, source_path=source_path, lens_id=lens_id),
        )
    except ValueError as exc:
        raise ValueError(f"{source_path} contains invalid accepted measurement for lens {lens_id}: {exc}") from exc


def read_velocity_measurements_v1_csv(csv_path: Path | str) -> VelocityMeasurementReadResult:
    """Read the upstream measurement contract and separate accepted/rejected rows."""

    resolved_path = Path(csv_path).expanduser().resolve()
    fieldnames, rows = _read_csv_dicts(resolved_path)

    required_columns = set(VELOCITY_MEASUREMENT_STANDARD_COLUMNS)
    missing_columns = sorted(required_columns.difference(fieldnames))
    if missing_columns:
        raise ValueError(f"{resolved_path} is missing required columns: {', '.join(missing_columns)}")

    accepted: list[SigmaObservation] = []
    rejected: list[VelocityMeasurementRejectedRow] = []
    for row in rows:
        schema_version = _normalize_required_text(row["schema_version"], "schema_version")
        if schema_version != VELOCITY_MEASUREMENT_SCHEMA_VERSION:
            raise ValueError(
                f"{resolved_path} expects schema_version={VELOCITY_MEASUREMENT_SCHEMA_VERSION!r}, "
                f"got {schema_version!r}."
            )

        lens_id = _normalize_required_text(row["lens_id"], "lens_id")
        use_for_likelihood = _parse_bool(row.get("use_for_likelihood"), field_name="use_for_likelihood")
        if not use_for_likelihood:
            rejected.append(
                VelocityMeasurementRejectedRow(
                    lens_id=lens_id,
                    reason="use_for_likelihood=false",
                    source_path=resolved_path,
                    raw_row=row,
                )
            )
            continue

        sigma_err_kms = _row_positive_float(row, "sigma_err_kms", resolved_path, lens_id)
        accepted.append(
            _accepted_observation_from_row(
                row,
                source_path=resolved_path,
                sigma_err_kms=sigma_err_kms,
            )
        )

    return VelocityMeasurementReadResult(
        accepted=tuple(accepted),
        rejected=tuple(rejected),
        provenance={
            "source_path": resolved_path,
            "schema_version": VELOCITY_MEASUREMENT_SCHEMA_VERSION,
            "row_count": len(rows),
        },
    )


def load_ppxf_velocity_measurements(
    csv_path: Path | str,
    *,
    adapter_config: PpxfAdapterConfig | None = None,
) -> VelocityMeasurementReadResult:
    """Adapt the current pPXF CSV export into the trusted measurement model."""

    resolved_path = Path(csv_path).expanduser().resolve()
    config = adapter_config or PpxfAdapterConfig()
    fieldnames, rows = _read_csv_dicts(resolved_path)

    required_columns = {
        "system",
        "base_name",
        "obs_tag",
        "primary_status",
        "sigma_primary_kms",
        config.error_column,
        "aperture_shape",
        "seeing_fwhm_arcsec",
    }
    if config.error_column == "sigma_total_kms":
        required_columns.add("sigma_total_kms")
    missing_columns = sorted(required_columns.difference(fieldnames))
    if missing_columns:
        raise ValueError(f"{resolved_path} is missing required columns: {', '.join(missing_columns)}")

    accepted: list[SigmaObservation] = []
    rejected: list[VelocityMeasurementRejectedRow] = []
    for row in rows:
        lens_id = _normalize_required_text(row["base_name"], "base_name")
        status = _normalize_required_text(row.get("primary_status", ""), "primary_status").upper()
        if config.reject_failed_rows and status != "SUCCESS":
            rejected.append(
                VelocityMeasurementRejectedRow(
                    lens_id=lens_id,
                    reason="primary_status!=SUCCESS",
                    source_path=resolved_path,
                    raw_row=row,
                )
            )
            continue

        sigma_err_kms = _row_positive_float(row, config.error_column, resolved_path, lens_id)
        sigma_primary_kms = _row_positive_float(row, "sigma_primary_kms", resolved_path, lens_id)
        obs_tag = _normalize_optional_text(row.get("obs_tag"))
        observation = _accepted_observation_from_row(
            {
                **row,
                "lens_id": lens_id,
                "sigma_kms": str(sigma_primary_kms),
                "sigma_err_kms": str(sigma_err_kms),
                "sigma_error_kind": "statistical" if config.error_column == "sigma_stat_kms" else "total",
                "measurement_status": status,
                "use_for_likelihood": "true",
                "source_system": _normalize_optional_text(row.get("system")) or "",
                "source_file": str(resolved_path),
            },
            source_path=resolved_path,
            sigma_err_kms=sigma_err_kms,
        )
        accepted.append(
            replace(
                observation,
                source_metadata={
                    "extraction_method": _normalize_optional_text(row.get("extraction_method")),
                    "warnings": _normalize_optional_text(row.get("warnings")),
                    "z_lens": _normalize_optional_text(row.get("z_lens")),
                    "z_source": _normalize_optional_text(row.get("z_source")),
                },
            )
        )

    return VelocityMeasurementReadResult(
        accepted=tuple(accepted),
        rejected=tuple(rejected),
        provenance={
            "source_path": resolved_path,
            "adapter": "ppxf_results_adapter",
            "error_column": config.error_column,
            "row_count": len(rows),
        },
    )


__all__ = [
    "PpxfAdapterConfig",
    "VelocityMeasurementReadResult",
    "VelocityMeasurementRejectedRow",
    "VELOCITY_MEASUREMENT_SCHEMA_VERSION",
    "VELOCITY_MEASUREMENT_STANDARD_COLUMNS",
    "load_ppxf_velocity_measurements",
    "read_velocity_measurements_v1_csv",
]
