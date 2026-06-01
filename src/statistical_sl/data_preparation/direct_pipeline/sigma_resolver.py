"""Join catalog lenses with trusted velocity-dispersion measurements.

The resolver owns the business rules around sigma availability:

- external measurement mode may leave a lens with ``num_sigma = 0``
- catalog-column mode requires exactly one sigma observation per lens
- accepted measurements are ordered deterministically
- the canonical model stores at most two observations per lens

It deliberately does not compute physical unit conversions or mass grids.  Its
job is to decide which trusted sigma rows belong to which lens record and to
surface that decision as a prepared, in-memory object for the next stage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from statistical_sl.data_preparation.direct_pipeline.policies import SigmaPolicy
from statistical_sl.data_preparation.direct_pipeline.records import (
    BaseLensRecord,
    PreparedLensRecord,
    SigmaObservation,
)
from statistical_sl.data_preparation.direct_pipeline.measurements import (
    VelocityMeasurementReadResult,
    VelocityMeasurementRejectedRow,
)


def _freeze_mapping(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    """Return a shallow read-only mapping for audit payloads."""

    return MappingProxyType(dict(value or {}))


def _unique_lens_ids(base_lenses: Sequence[BaseLensRecord]) -> tuple[BaseLensRecord, ...]:
    """Validate that the catalog lens identity is unique before joining."""

    seen: set[str] = set()
    ordered_lenses: list[BaseLensRecord] = []
    for lens in base_lenses:
        if lens.lens_id in seen:
            raise ValueError(f"duplicate lens_id {lens.lens_id!r} in base catalog records.")
        seen.add(lens.lens_id)
        ordered_lenses.append(lens)
    return tuple(ordered_lenses)


def _coerce_measurement_input(
    accepted_measurements: Sequence[SigmaObservation] | VelocityMeasurementReadResult | None,
) -> tuple[tuple[SigmaObservation, ...], tuple[VelocityMeasurementRejectedRow, ...]]:
    """Normalize the accepted-measurement input into tuples for grouping."""

    if accepted_measurements is None:
        return (), ()
    if isinstance(accepted_measurements, VelocityMeasurementReadResult):
        return accepted_measurements.accepted, accepted_measurements.rejected
    return tuple(accepted_measurements), ()


def _normalize_catalog_measurements(
    catalog_measurements: Mapping[str, Sequence[SigmaObservation] | SigmaObservation] | None,
) -> dict[str, tuple[SigmaObservation, ...]]:
    """Normalize catalog-column measurements into a lens-id keyed mapping."""

    normalized: dict[str, tuple[SigmaObservation, ...]] = {}
    if catalog_measurements is None:
        return normalized
    for lens_id, value in catalog_measurements.items():
        if isinstance(value, SigmaObservation):
            normalized[str(lens_id)] = (value,)
        else:
            normalized[str(lens_id)] = tuple(value)
    return normalized


def _group_measurements_by_lens(measurements: Sequence[SigmaObservation]) -> dict[str, list[SigmaObservation]]:
    """Collect accepted measurements by lens identity while preserving input order."""

    grouped: dict[str, list[SigmaObservation]] = {}
    for measurement in measurements:
        grouped.setdefault(measurement.lens_id, []).append(measurement)
    return grouped


def _validate_single_observation(observation: SigmaObservation, lens_id: str) -> SigmaObservation:
    """Validate a single measurement row before handing it to the prepared record."""

    if observation.lens_id != lens_id:
        raise ValueError(f"measurement lens_id {observation.lens_id!r} does not match catalog lens_id {lens_id!r}.")
    return observation


def _validate_and_order_observations(
    lens_id: str,
    observations: Sequence[SigmaObservation],
) -> tuple[SigmaObservation, ...]:
    """Validate the accepted sigma rows for one lens and return canonical order."""

    observations = tuple(observations)
    if len(observations) > 2:
        raise ValueError(f"{lens_id} supports at most two accepted sigma measurements.")

    if len(observations) == 1:
        return (_validate_single_observation(observations[0], lens_id),)

    if len(observations) == 0:
        return ()

    tags = [observation.obs_tag for observation in observations]
    if any(tag is None for tag in tags):
        raise ValueError(f"{lens_id} cannot mix untagged and tagged sigma measurements.")

    normalized_tags = [str(tag).upper() for tag in tags]
    if set(normalized_tags) != {"A", "B"}:
        if len(set(normalized_tags)) != len(normalized_tags):
            raise ValueError(f"{lens_id} has duplicate sigma observation tags.")
        raise ValueError(f"{lens_id} requires exactly one A row and one B row when it has two measurements.")

    ordered_by_tag = {str(observation.obs_tag).upper(): _validate_single_observation(observation, lens_id) for observation in observations}
    return (ordered_by_tag["A"], ordered_by_tag["B"])


def _build_distribution(records: Sequence[PreparedLensRecord]) -> Mapping[int, int]:
    """Summarize the resolved ``num_sigma`` counts as a small audit mapping."""

    counts: dict[int, int] = {}
    for record in records:
        counts[record.num_sigma] = counts.get(record.num_sigma, 0) + 1
    return MappingProxyType(counts)


@dataclass(frozen=True)
class SigmaResolutionAudit:
    """Audit trail for one sigma-resolution pass."""

    source_type: str
    num_sigma_distribution: Mapping[int, int]
    missing_lens_ids: tuple[str, ...]
    rejected_measurements: tuple[VelocityMeasurementRejectedRow, ...] = ()
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Freeze audit payloads for deterministic downstream inspection."""

        object.__setattr__(self, "num_sigma_distribution", _freeze_mapping(self.num_sigma_distribution))
        object.__setattr__(self, "extra", _freeze_mapping(self.extra))


@dataclass(frozen=True)
class SigmaResolutionResult:
    """Prepared records and audit details returned by sigma resolution."""

    records: tuple[PreparedLensRecord, ...]
    audit: SigmaResolutionAudit


def resolve_sigma_observations(
    base_lenses: Sequence[BaseLensRecord],
    *,
    sigma_policy: SigmaPolicy,
    accepted_measurements: Sequence[SigmaObservation] | VelocityMeasurementReadResult | None = None,
    catalog_measurements: Mapping[str, Sequence[SigmaObservation] | SigmaObservation] | None = None,
) -> SigmaResolutionResult:
    """Join lens records with trusted sigma measurements under one policy."""

    normalized_lenses = _unique_lens_ids(tuple(base_lenses))
    accepted_rows, rejected_rows = _coerce_measurement_input(accepted_measurements)
    measurement_rows_by_lens = _group_measurements_by_lens(accepted_rows)
    catalog_rows_by_lens = _normalize_catalog_measurements(catalog_measurements)

    resolved_records: list[PreparedLensRecord] = []
    missing_lens_ids: list[str] = []

    if sigma_policy.source_type == "catalog_columns":
        if not sigma_policy.trust_catalog_sigma:
            raise ValueError("catalog_columns mode requires trust_catalog_sigma=True.")
        for base_lens in normalized_lenses:
            observations = catalog_rows_by_lens.get(base_lens.lens_id)
            if observations is None:
                raise ValueError(f"{base_lens.lens_id} is missing a catalog sigma observation.")
            if len(observations) != 1:
                raise ValueError(f"{base_lens.lens_id} must have exactly one catalog sigma observation.")
            resolved_records.append(
                PreparedLensRecord(
                    base_lens=base_lens,
                    sigma_observations=(_validate_single_observation(observations[0], base_lens.lens_id),),
                    preparation_metadata={
                        "sigma_source": "catalog_columns",
                        "resolver_missing_policy": sigma_policy.missing_policy,
                    },
                )
            )
    else:
        known_lens_ids = {lens.lens_id for lens in normalized_lenses}
        extra_measurement_ids = sorted(set(measurement_rows_by_lens).difference(known_lens_ids))
        if extra_measurement_ids:
            raise ValueError(
                f"accepted measurements reference unknown lens ids: {', '.join(extra_measurement_ids)}"
            )

        for base_lens in normalized_lenses:
            observations = measurement_rows_by_lens.get(base_lens.lens_id, [])
            if len(observations) == 0:
                if sigma_policy.missing_policy == "num_sigma_zero":
                    missing_lens_ids.append(base_lens.lens_id)
                    resolved_records.append(
                        PreparedLensRecord(
                            base_lens=base_lens,
                            sigma_observations=(),
                            preparation_metadata={
                                "sigma_source": sigma_policy.source_type,
                                "resolver_missing_policy": sigma_policy.missing_policy,
                            },
                        )
                    )
                    continue
                raise ValueError(f"{base_lens.lens_id} is missing a trusted sigma measurement.")

            ordered_observations = _validate_and_order_observations(base_lens.lens_id, observations)
            resolved_records.append(
                PreparedLensRecord(
                    base_lens=base_lens,
                    sigma_observations=ordered_observations,
                    preparation_metadata={
                        "sigma_source": sigma_policy.source_type,
                        "resolver_missing_policy": sigma_policy.missing_policy,
                    },
                )
            )

    audit = SigmaResolutionAudit(
        source_type=sigma_policy.source_type,
        num_sigma_distribution=_build_distribution(resolved_records),
        missing_lens_ids=tuple(missing_lens_ids),
        rejected_measurements=rejected_rows,
        extra={
            "accepted_measurement_count": len(accepted_rows),
            "catalog_measurement_count": sum(len(value) for value in catalog_rows_by_lens.values()),
        },
    )
    return SigmaResolutionResult(records=tuple(resolved_records), audit=audit)


__all__ = [
    "SigmaResolutionAudit",
    "SigmaResolutionResult",
    "resolve_sigma_observations",
]
