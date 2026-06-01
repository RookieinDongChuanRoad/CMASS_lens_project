"""Tests for the direct canonical-pipeline domain records.

These tests define the first stable API layer for the direct pipeline.  The
objects under test are deliberately small: they carry catalog facts,
trusted velocity-dispersion measurements, and preparation policy decisions
without reading files or writing HDF5.
"""

from __future__ import annotations

import pytest

from statistical_sl.data_preparation.direct_pipeline.policies import (
    MassDefinitionPolicy,
    SigmaPolicy,
    UnitPolicy,
)
from statistical_sl.data_preparation.direct_pipeline.records import (
    BaseLensRecord,
    PreparedLensRecord,
    SigmaObservation,
)


def _base_lens_record() -> BaseLensRecord:
    """Build one valid catalog-only lens record used by record tests."""

    return BaseLensRecord(
        lens_id="023817-054555",
        z_lens=0.599,
        z_source=1.763,
        theta_ein_arcsec=0.929,
        effective_radius_arcsec=0.706,
        log_stellar_mass=11.51,
        log_stellar_mass_err=0.08,
        profile_name="devauc",
        sersic_index=4.0,
    )


def test_base_lens_record_accepts_catalog_facts_without_sigma() -> None:
    """Catalog records should not carry trusted velocity-dispersion data."""

    record = _base_lens_record()

    assert record.lens_id == "023817-054555"
    assert record.z_source > record.z_lens
    assert record.profile_name == "devauc"
    assert not hasattr(record, "sigma_kms")
    assert not hasattr(record, "sigma_err_kms")


def test_sigma_observation_normalizes_optional_tag_and_keeps_kms_units() -> None:
    """Trusted sigma rows should preserve km/s values and normalize tags."""

    observation = SigmaObservation(
        lens_id="023817-054555",
        obs_tag=" a ",
        sigma_kms=226.4,
        sigma_err_kms=9.6,
        sigma_error_kind="statistical",
        measurement_status="SUCCESS",
        source_system="HSCJ023817-054555A",
        source_file="/tmp/ppxf_results_optimal.csv",
    )

    assert observation.obs_tag == "A"
    assert observation.sigma_kms == pytest.approx(226.4)
    assert observation.sigma_err_kms == pytest.approx(9.6)


@pytest.mark.parametrize(
    ("sigma_kms", "sigma_err_kms"),
    [
        (0.0, 9.0),
        (-1.0, 9.0),
        (225.0, 0.0),
        (225.0, -1.0),
    ],
)
def test_sigma_observation_rejects_non_positive_values(
    sigma_kms: float,
    sigma_err_kms: float,
) -> None:
    """Sigma values and uncertainties must be positive finite km/s numbers."""

    with pytest.raises(ValueError, match="sigma"):
        SigmaObservation(
            lens_id="023817-054555",
            sigma_kms=sigma_kms,
            sigma_err_kms=sigma_err_kms,
        )


def test_prepared_lens_record_reports_num_sigma_from_observation_count() -> None:
    """`num_sigma` must be derived from accepted observations only."""

    observations = (
        SigmaObservation(lens_id="023817-054555", obs_tag="A", sigma_kms=226.0, sigma_err_kms=10.0),
        SigmaObservation(lens_id="023817-054555", obs_tag="B", sigma_kms=230.0, sigma_err_kms=11.0),
    )

    prepared = PreparedLensRecord(
        base_lens=_base_lens_record(),
        sigma_observations=observations,
        sigma_crit=2.2e9,
    )

    assert prepared.num_sigma == 2
    assert tuple(item.obs_tag for item in prepared.sigma_observations) == ("A", "B")


def test_prepared_lens_record_rejects_more_than_two_sigma_observations() -> None:
    """The canonical likelihood contract supports at most two sigma rows."""

    observations = tuple(
        SigmaObservation(
            lens_id="023817-054555",
            obs_tag=str(index),
            sigma_kms=220.0 + index,
            sigma_err_kms=10.0,
        )
        for index in range(3)
    )

    with pytest.raises(ValueError, match="at most two"):
        PreparedLensRecord(
            base_lens=_base_lens_record(),
            sigma_observations=observations,
        )


def test_prepared_lens_record_rejects_sigma_rows_for_other_lenses() -> None:
    """Joined records must not silently mix catalog and measurement identities."""

    with pytest.raises(ValueError, match="lens_id"):
        PreparedLensRecord(
            base_lens=_base_lens_record(),
            sigma_observations=(
                SigmaObservation(
                    lens_id="other-lens",
                    sigma_kms=220.0,
                    sigma_err_kms=10.0,
                ),
            ),
        )


def test_policy_objects_validate_core_direct_pipeline_choices() -> None:
    """Policies should fail early on nonsensical direct-pipeline config values."""

    unit_policy = UnitPolicy(
        unit_convention="h_units_v1",
        h_ref=0.7,
    )
    mass_policy = MassDefinitionPolicy(
        mass_definition_label="m5_hinvkpc",
        mass_radius_kpc=5.0,
    )
    sigma_policy = SigmaPolicy(
        source_type="ppxf_results_adapter",
        missing_policy="num_sigma_zero",
        max_observations_per_lens=2,
    )

    assert unit_policy.h_ref == pytest.approx(0.7)
    assert mass_policy.mass_radius_kpc == pytest.approx(5.0)
    assert sigma_policy.missing_policy == "num_sigma_zero"

    with pytest.raises(ValueError, match="missing_policy"):
        SigmaPolicy(source_type="catalog_columns", missing_policy="silently_drop")
