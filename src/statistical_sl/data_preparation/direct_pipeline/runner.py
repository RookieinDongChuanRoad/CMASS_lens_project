"""Orchestrate the direct source-to-canonical build from a YAML config."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from statistical_sl.data_preparation.direct_pipeline.catalogs import CmassSummaryCatalogReader, SlacsTableCatalogReader
from statistical_sl.data_preparation.direct_pipeline.config import DirectPipelineConfig, load_direct_pipeline_config
from statistical_sl.data_preparation.direct_pipeline.cross_sections import CmassPowerLawCrossSectionProvider, SonnenfeldFibreCrossSectionProvider
from statistical_sl.data_preparation.direct_pipeline.grid_builders import build_derived_grid_blocks
from statistical_sl.data_preparation.direct_pipeline.lens_preparer import prepare_lens_records
from statistical_sl.data_preparation.direct_pipeline.payload import build_canonical_dataset_payload
from statistical_sl.data_preparation.direct_pipeline.provenance import DirectPipelineProvenance
from statistical_sl.data_preparation.direct_pipeline.records import CanonicalDatasetPayload, SigmaObservation
from statistical_sl.data_preparation.direct_pipeline.sigma_resolver import resolve_sigma_observations
from statistical_sl.data_preparation.direct_pipeline.measurements import (
    PpxfAdapterConfig,
    VelocityMeasurementReadResult,
    load_ppxf_velocity_measurements,
    read_velocity_measurements_v1_csv,
)
from statistical_sl.data_preparation.direct_pipeline.writer import write_canonical_dataset_payload


def _json_ready(value: Any) -> Any:
    """Convert provenance payloads into JSON-serializable objects."""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_ready(inner_value) for key, inner_value in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


@dataclass(frozen=True)
class DirectPipelineRunResult:
    """Materialized outputs from one direct canonical build."""

    config: DirectPipelineConfig
    canonical_hdf5: Path
    audit_json: Path | None = None
    payload: CanonicalDatasetPayload | None = None


def _build_catalog_measurements(
    catalog_result,
    *,
    trust_catalog_sigma: bool,
) -> tuple[dict[str, tuple[SigmaObservation, ...]], VelocityMeasurementReadResult]:
    """Promote catalog sigma values to trusted observations in compatibility mode."""

    if not trust_catalog_sigma:
        raise ValueError("catalog_columns mode requires trust_catalog_sigma=True.")

    catalog_measurements: dict[str, tuple[SigmaObservation, ...]] = {}
    accepted: list[SigmaObservation] = []
    source_path = catalog_result.provenance.source_path
    sigma_values = catalog_result.provenance.extra.get("catalog_sigma_values")
    if sigma_values is None:
        sigma_values = catalog_result.provenance.extra.get("untrusted_sigma_values")
    if sigma_values is None:
        raise ValueError("catalog_columns mode requires catalog sigma values in catalog provenance.")

    for lens_id, sigma_data in sigma_values.items():
        if not isinstance(sigma_data, Mapping):
            raise ValueError(f"catalog sigma payload for {lens_id!r} must be a mapping.")
        if "sigma_kms" in sigma_data:
            sigma_kms = float(sigma_data["sigma_kms"])
            sigma_err_kms = float(sigma_data["sigma_err_kms"])
        else:
            sigma_kms = float(sigma_data["sigma"])
            sigma_err_kms = float(sigma_data["sigma_err"])
        observation = SigmaObservation(
            lens_id=str(lens_id),
            sigma_kms=sigma_kms,
            sigma_err_kms=sigma_err_kms,
            measurement_status="catalog_columns",
            use_for_likelihood=True,
            source_system=catalog_result.provenance.catalog_type,
            source_file=str(source_path),
            source_metadata={"source_kind": "catalog_columns"},
        )
        catalog_measurements[str(lens_id)] = (observation,)
        accepted.append(observation)

    measurement_result = VelocityMeasurementReadResult(
        accepted=tuple(accepted),
        rejected=(),
        provenance={
            "source_path": None,
            "schema_version": "catalog_columns",
            "row_count": len(accepted),
            "catalog_source_path": source_path,
            "catalog_type": catalog_result.provenance.catalog_type,
        },
    )
    return catalog_measurements, measurement_result


def _load_measurement_result(config: DirectPipelineConfig, catalog_result) -> tuple[VelocityMeasurementReadResult, dict[str, tuple[SigmaObservation, ...]] | None]:
    """Load trusted sigma observations from the configured measurement source."""

    measurement_config = config.velocity_measurements
    if measurement_config.type == "catalog_columns":
        catalog_measurements, measurement_result = _build_catalog_measurements(
            catalog_result,
            trust_catalog_sigma=measurement_config.trust_catalog_sigma,
        )
        return measurement_result, catalog_measurements

    if measurement_config.type == "ppxf_results_adapter":
        assert measurement_config.path is not None
        return (
            load_ppxf_velocity_measurements(
                measurement_config.path,
                adapter_config=PpxfAdapterConfig(error_column=measurement_config.error_column),
            ),
            None,
        )

    assert measurement_config.path is not None
    return read_velocity_measurements_v1_csv(measurement_config.path), None


def _load_catalog_result(config: DirectPipelineConfig):
    """Read the catalog records using the correct source-specific reader."""

    if config.catalog.type == "cmass_summary_table":
        return CmassSummaryCatalogReader(config.catalog.path, profile_name=config.catalog.profile_name).read()
    if config.catalog.type == "slacs_table":
        return SlacsTableCatalogReader(config.catalog.path, profile_name=config.catalog.profile_name).read()
    raise ValueError(f"Unsupported catalog type: {config.catalog.type}")


def _load_cross_section_block(config: DirectPipelineConfig):
    """Load the cross-section block selected by the YAML config."""

    if config.cross_section.type == "cmass_power_law":
        return CmassPowerLawCrossSectionProvider(config.cross_section.source_hdf5).load(
            theta_e_axis=config.grids.theta_e_axis,
        )
    if config.cross_section.type == "sonnenfeld_fibre":
        return SonnenfeldFibreCrossSectionProvider(config.cross_section.source_hdf5).load()
    raise ValueError(f"Unsupported cross_section.type: {config.cross_section.type}")


def _write_audit_json(path: Path, payload: Mapping[str, Any]) -> Path:
    """Write one JSON audit file using the same normalization rules as the writer."""

    resolved_path = Path(path).expanduser().resolve()
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_path.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True), encoding="utf-8")
    return resolved_path


def run_direct_canonical_build(config_path: Path | str) -> DirectPipelineRunResult:
    """Run the direct canonical build end to end from a YAML config file."""

    config = load_direct_pipeline_config(config_path)
    catalog_result = _load_catalog_result(config)
    measurement_result, catalog_measurements = _load_measurement_result(config, catalog_result)

    sigma_resolution_result = resolve_sigma_observations(
        catalog_result.records,
        sigma_policy=config.sigma_policy(),
        accepted_measurements=measurement_result if catalog_measurements is None else None,
        catalog_measurements=catalog_measurements,
    )
    prepared_records = prepare_lens_records(
        sigma_resolution_result.records,
        unit_policy=config.unit_policy(),
        aperture_policy_ref=config.aperture_policy_ref(),
    )
    derived_blocks = build_derived_grid_blocks(
        prepared_records,
        gamma_axis=config.grids.gamma_axis,
        mass_policy=config.mass_policy(),
        unit_policy=config.unit_policy(),
    )
    cross_section_block = _load_cross_section_block(config)

    provenance = DirectPipelineProvenance(
        catalog_path=catalog_result.provenance.source_path,
        catalog_type=catalog_result.provenance.catalog_type,
        measurement_path=measurement_result.provenance.get("source_path"),
        measurement_mode=config.velocity_measurements.type,
        ignored_catalog_columns=dict(catalog_result.provenance.ignored_columns),
        rejected_measurements=measurement_result.rejected,
        num_sigma_distribution=sigma_resolution_result.audit.num_sigma_distribution,
        extra={
            "catalog_provenance": dict(catalog_result.provenance.extra),
            "catalog_available_measurement_columns": dict(catalog_result.provenance.available_measurement_columns),
            "measurement_schema_version": measurement_result.provenance.get("schema_version"),
        },
    )

    payload = build_canonical_dataset_payload(
        prepared_records=prepared_records,
        mass_block=derived_blocks.mass,
        velocity_block=derived_blocks.velocity,
        cross_section_block=cross_section_block,
        catalog_provenance=provenance,
        measurement_result=measurement_result,
        sigma_resolution_audit=sigma_resolution_result.audit,
        unit_policy=config.unit_policy(),
        mass_policy=config.mass_policy(),
    )

    canonical_hdf5 = write_canonical_dataset_payload(payload, config.output.canonical_hdf5)
    audit_json = None
    if config.output.audit_json is not None:
        audit_json = _write_audit_json(config.output.audit_json, payload.provenance)

    return DirectPipelineRunResult(
        config=config,
        canonical_hdf5=canonical_hdf5,
        audit_json=audit_json,
        payload=payload,
    )


__all__ = [
    "DirectPipelineRunResult",
    "run_direct_canonical_build",
]
