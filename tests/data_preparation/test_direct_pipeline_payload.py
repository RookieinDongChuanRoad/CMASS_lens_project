"""Tests for assembling the direct canonical dataset payload."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from statistical_sl.data_preparation.direct_pipeline.cross_sections import CrossSectionBlock, CrossSectionProvenance
from statistical_sl.data_preparation.direct_pipeline.payload import build_canonical_dataset_payload
from statistical_sl.data_preparation.direct_pipeline.policies import MassDefinitionPolicy, UnitPolicy
from statistical_sl.data_preparation.direct_pipeline.provenance import DirectPipelineProvenance
from statistical_sl.data_preparation.direct_pipeline.records import (
    BaseLensRecord,
    CanonicalDatasetPayload,
    PreparedLensRecord,
    SigmaObservation,
)
from statistical_sl.data_preparation.direct_pipeline.records import MAX_SIGMA_OBSERVATIONS
from statistical_sl.data_preparation.direct_pipeline.grid_builders import LensingMassGridBlock, VelocityDispersionGridBlock
from statistical_sl.data_preparation.direct_pipeline.measurements import VelocityMeasurementReadResult, VelocityMeasurementRejectedRow
from statistical_sl.data_preparation.direct_pipeline.sigma_resolver import SigmaResolutionAudit
from statistical_sl.core.canonical_schema import (
    CAPABILITY_LENS_OBSERVATIONS_V1,
    CAPABILITY_LENSING_CROSS_SECTION_THETA_GAMMA_V1,
    CAPABILITY_LENSING_MASS_GRIDS_V1,
    CAPABILITY_VELOCITY_DISPERSION_PER_LENS_S2_V1,
)
from statistical_sl.data_preparation.models import AperturePolicy


def _prepared_record(
    lens_id: str,
    *,
    num_sigma: int,
    aperture_policy: AperturePolicy | None = None,
) -> PreparedLensRecord:
    """Create a prepared record with predictable values for payload assembly."""

    sigma_observations = ()
    if num_sigma:
        sigma_observations = (
            SigmaObservation(
                lens_id=lens_id,
                sigma_kms=240.0,
                sigma_err_kms=15.0,
            ),
        )
    resolved_aperture_policy = aperture_policy
    if resolved_aperture_policy is None and num_sigma:
        resolved_aperture_policy = AperturePolicy.rectangular(
            width_arcsec=1.6,
            height_arcsec=0.9,
            seeing_fwhm_arcsec=0.9,
        )
    return PreparedLensRecord(
        base_lens=BaseLensRecord(
            lens_id=lens_id,
            z_lens=0.5,
            z_source=1.5,
            theta_ein_arcsec=1.0,
            theta_ein_kpc=3.0,
            effective_radius_arcsec=0.8,
            effective_radius_kpc=5.0,
            log_stellar_mass=11.2,
            log_stellar_mass_err=0.08,
            profile_name="devauc",
            sersic_index=4.0,
        ),
        sigma_observations=sigma_observations,
        sigma_crit=2.0e9,
        aperture_policy=resolved_aperture_policy,
        observation_flavor="slit" if resolved_aperture_policy is not None else None,
        sigma_definition="observed_aperture" if resolved_aperture_policy is not None else None,
        unit_convention="h_units_v1",
        h_ref=0.7,
        theta_ein_kpc=3.0,
        effective_radius_kpc=5.0,
        active_log_stellar_mass=11.0,
        active_log_effective_radius=0.6,
    )


def test_payload_assembles_expected_blocks_and_provenance(tmp_path: Path) -> None:
    """The payload should expose the canonical blocks and the audit trail."""

    catalog_path = tmp_path / "summary_table_deV.txt"
    measurement_path = tmp_path / "ppxf_results_optimal.csv"
    cross_section_path = tmp_path / "cs_grid_power.h5"

    prepared_records = (
        _prepared_record("lens-a", num_sigma=1),
        _prepared_record("lens-b", num_sigma=0),
    )
    mass_block = LensingMassGridBlock(
        lens_ids=("lens-a", "lens-b"),
        gamma_axis=np.asarray([1.2, 2.0], dtype=float),
        log_enclosed_mass_grid=np.asarray([[10.0, 10.2], [10.1, 10.3]], dtype=float),
        dmass_dthetaein_grid=np.asarray([[0.9, 1.0], [1.1, 1.2]], dtype=float),
        mass_definition_label="m5_hinvkpc",
        mass_radius_kpc=5.0,
        unit_convention="h_units_v1",
        h_ref=0.7,
    )
    velocity_block = VelocityDispersionGridBlock(
        lens_ids=("lens-a", "lens-b"),
        gamma_axis=np.asarray([1.2, 2.0], dtype=float),
        s2_grid=np.asarray([[1.0e-5, 1.1e-5], [0.0, 0.0]], dtype=float),
        has_s2=np.asarray([True, False], dtype=bool),
        mass_definition_label="m5_hinvkpc",
        mass_radius_kpc=5.0,
        unit_convention="h_units_v1",
        h_ref=0.7,
    )
    cross_section_block = CrossSectionBlock(
        theta_e_axis=np.asarray([0.5, 1.0], dtype=float),
        gamma_axis=np.asarray([1.2, 2.0], dtype=float),
        cross_section_grid=np.asarray([[0.1, 0.2], [0.3, 0.4]], dtype=float),
        provenance=CrossSectionProvenance(
            source_path=cross_section_path,
            source_mode="cmass_power_law",
            source_dataset="compressed_grids/cs_over_theta_ein_grid",
        ),
    )
    rejected_row = VelocityMeasurementRejectedRow(
        lens_id="lens-b",
        reason="use_for_likelihood=false",
        source_path=measurement_path,
        raw_row={"lens_id": "lens-b"},
    )
    measurement_result = VelocityMeasurementReadResult(
        accepted=(
            SigmaObservation(
                lens_id="lens-a",
                obs_tag="A",
                sigma_kms=240.0,
                sigma_err_kms=15.0,
                source_system="lens-aA",
                source_file=str(measurement_path),
            ),
        ),
        rejected=(rejected_row,),
        provenance={
            "source_path": measurement_path,
            "schema_version": "velocity_measurements_v1",
            "row_count": 2,
        },
    )
    audit = SigmaResolutionAudit(
        source_type="ppxf_results_adapter",
        num_sigma_distribution={0: 1, 1: 1},
        missing_lens_ids=("lens-b",),
        rejected_measurements=(rejected_row,),
        extra={"accepted_measurement_count": 1},
    )
    catalog_provenance = DirectPipelineProvenance(
        catalog_path=catalog_path,
        catalog_type="cmass_summary_table",
        measurement_path=measurement_path,
        measurement_mode="ppxf_results_adapter",
        ignored_catalog_columns={
            "sigma": "untrusted_catalog_value",
            "sigma_err": "untrusted_catalog_value",
        },
        rejected_measurements=(rejected_row,),
        num_sigma_distribution={0: 1, 1: 1},
    )

    payload = build_canonical_dataset_payload(
        prepared_records=prepared_records,
        mass_block=mass_block,
        velocity_block=velocity_block,
        cross_section_block=cross_section_block,
        catalog_provenance=catalog_provenance,
        measurement_result=measurement_result,
        sigma_resolution_audit=audit,
        unit_policy=UnitPolicy(unit_convention="h_units_v1", h_ref=0.7),
        mass_policy=MassDefinitionPolicy(mass_definition_label="m5_hinvkpc", mass_radius_kpc=5.0),
    )

    assert isinstance(payload, CanonicalDatasetPayload)
    assert set(payload.__dict__) == {
        "metadata",
        "lenses",
        "lensing_mass_grids",
        "lensing_cross_section",
        "velocity_dispersion_grids",
        "provenance",
    }
    assert payload.metadata["schema_version"] == "canonical_dataset_payload_v1"
    assert payload.metadata["aperture_contract"] == "per_lens"
    assert set(payload.metadata["capabilities"]) == {
        CAPABILITY_LENS_OBSERVATIONS_V1,
        CAPABILITY_LENSING_MASS_GRIDS_V1,
        CAPABILITY_LENSING_CROSS_SECTION_THETA_GAMMA_V1,
        CAPABILITY_VELOCITY_DISPERSION_PER_LENS_S2_V1,
    }
    assert payload.lenses["num_sigma"].tolist() == [1, 0]
    assert payload.lenses["aperture_shape"].tolist() == ["rectangular", ""]
    np.testing.assert_allclose(payload.lenses["seeing_fwhm_arcsec"], np.asarray([0.9, 0.0]))
    assert payload.velocity_dispersion_grids["has_s2"].tolist() == [True, False]
    assert payload.provenance["catalog"]["source_path"] == str(catalog_path.resolve())
    assert payload.provenance["measurement"]["source_path"] == str(measurement_path.resolve())
    assert payload.provenance["catalog"]["ignored_columns"]["sigma"] == "untrusted_catalog_value"
    assert payload.provenance["measurement"]["rejected_rows"][0]["reason"] == "use_for_likelihood=false"
    assert payload.provenance["resolver"]["num_sigma_distribution"] == {0: 1, 1: 1}


def test_payload_allows_heterogeneous_per_lens_apertures(tmp_path: Path) -> None:
    """Root metadata should not force one aperture geometry for the whole file."""

    prepared_records = (
        _prepared_record(
            "lens-a",
            num_sigma=1,
            aperture_policy=AperturePolicy.rectangular(
                width_arcsec=1.6,
                height_arcsec=0.9,
                seeing_fwhm_arcsec=0.7,
            ),
        ),
        _prepared_record(
            "lens-b",
            num_sigma=1,
            aperture_policy=AperturePolicy.circular(
                radius_arcsec=1.5,
                seeing_fwhm_arcsec=1.2,
            ),
        ),
    )
    mass_block = LensingMassGridBlock(
        lens_ids=("lens-a", "lens-b"),
        gamma_axis=np.asarray([1.2, 2.0], dtype=float),
        log_enclosed_mass_grid=np.asarray([[10.0, 10.2], [10.1, 10.3]], dtype=float),
        dmass_dthetaein_grid=np.asarray([[0.9, 1.0], [1.1, 1.2]], dtype=float),
        mass_definition_label="m5_hinvkpc",
        mass_radius_kpc=5.0,
        unit_convention="h_units_v1",
        h_ref=0.7,
    )
    velocity_block = VelocityDispersionGridBlock(
        lens_ids=("lens-a", "lens-b"),
        gamma_axis=np.asarray([1.2, 2.0], dtype=float),
        s2_grid=np.asarray([[1.0e-5, 1.1e-5], [2.0e-5, 2.1e-5]], dtype=float),
        has_s2=np.asarray([True, True], dtype=bool),
        mass_definition_label="m5_hinvkpc",
        mass_radius_kpc=5.0,
        unit_convention="h_units_v1",
        h_ref=0.7,
    )
    cross_section_block = CrossSectionBlock(
        theta_e_axis=np.asarray([0.5, 1.0], dtype=float),
        gamma_axis=np.asarray([1.2, 2.0], dtype=float),
        cross_section_grid=np.asarray([[0.1, 0.2], [0.3, 0.4]], dtype=float),
        provenance=CrossSectionProvenance(
            source_path=tmp_path / "cs_grid_power.h5",
            source_mode="cmass_power_law",
            source_dataset="compressed_grids/cs_over_theta_ein_grid",
        ),
    )
    measurement_result = VelocityMeasurementReadResult(
        accepted=tuple(record.sigma_observations[0] for record in prepared_records),
        rejected=(),
        provenance={"source_path": tmp_path / "velocity_measurements_v1.csv"},
    )
    audit = SigmaResolutionAudit(
        source_type="velocity_measurements_v1",
        num_sigma_distribution={1: 2},
        missing_lens_ids=(),
        rejected_measurements=(),
        extra={"accepted_measurement_count": 2},
    )
    catalog_provenance = DirectPipelineProvenance(
        catalog_path=tmp_path / "summary_table_deV.txt",
        catalog_type="cmass_summary_table",
        measurement_path=tmp_path / "velocity_measurements_v1.csv",
        measurement_mode="velocity_measurements_v1",
        num_sigma_distribution={1: 2},
    )

    payload = build_canonical_dataset_payload(
        prepared_records=prepared_records,
        mass_block=mass_block,
        velocity_block=velocity_block,
        cross_section_block=cross_section_block,
        catalog_provenance=catalog_provenance,
        measurement_result=measurement_result,
        sigma_resolution_audit=audit,
        unit_policy=UnitPolicy(unit_convention="h_units_v1", h_ref=0.7),
        mass_policy=MassDefinitionPolicy(mass_definition_label="m5_hinvkpc", mass_radius_kpc=5.0),
    )

    assert payload.metadata["aperture_contract"] == "per_lens"
    assert "aperture_width_arcsec" not in payload.metadata
    assert payload.lenses["aperture_shape"].tolist() == ["rectangular", "circular"]
    np.testing.assert_allclose(payload.lenses["seeing_fwhm_arcsec"], np.asarray([0.7, 1.2]))
