"""Tests for resolving catalog lenses against trusted sigma measurements."""

from __future__ import annotations

from collections.abc import Mapping

import pytest

from statistical_sl.data_preparation.direct_pipeline.policies import SigmaPolicy
from statistical_sl.data_preparation.direct_pipeline.records import BaseLensRecord, SigmaObservation
from statistical_sl.data_preparation.direct_pipeline.sigma_resolver import resolve_sigma_observations


def _base_lens(lens_id: str) -> BaseLensRecord:
    """Create one compact lens record for sigma resolution tests."""

    return BaseLensRecord(
        lens_id=lens_id,
        z_lens=0.5,
        z_source=1.5,
        theta_ein_arcsec=1.0,
        effective_radius_arcsec=0.8,
        log_stellar_mass=11.2,
        log_stellar_mass_err=0.08,
        profile_name="devauc",
        sersic_index=4.0,
    )


def _measurement(lens_id: str, *, obs_tag: str | None, sigma_kms: float, sigma_err_kms: float) -> SigmaObservation:
    """Build one trusted sigma observation for resolver tests."""

    return SigmaObservation(
        lens_id=lens_id,
        obs_tag=obs_tag,
        sigma_kms=sigma_kms,
        sigma_err_kms=sigma_err_kms,
        source_system=f"{lens_id}{obs_tag or ''}",
        source_file="/tmp/measurements.csv",
    )


def test_catalog_columns_mode_requires_one_sigma_observation_per_lens() -> None:
    """SLACS-like catalog columns should produce exactly one sigma row per lens."""

    lenses = (_base_lens("lens-a"), _base_lens("lens-b"))
    catalog_measurements: Mapping[str, tuple[SigmaObservation, ...]] = {
        "lens-a": (_measurement("lens-a", obs_tag=None, sigma_kms=240.0, sigma_err_kms=15.0),),
        "lens-b": (_measurement("lens-b", obs_tag=None, sigma_kms=250.0, sigma_err_kms=16.0),),
    }

    result = resolve_sigma_observations(
        lenses,
        sigma_policy=SigmaPolicy(
            source_type="catalog_columns",
            missing_policy="fail",
            trust_catalog_sigma=True,
        ),
        catalog_measurements=catalog_measurements,
    )

    assert [record.num_sigma for record in result.records] == [1, 1]
    assert [record.lens_id for record in result.records] == ["lens-a", "lens-b"]
    assert result.audit.num_sigma_distribution == {1: 2}


def test_external_mode_uses_num_sigma_zero_for_missing_lenses() -> None:
    """CMASS-like external measurements should keep unmatched lenses in the sample."""

    lenses = (_base_lens("lens-a"), _base_lens("lens-b"), _base_lens("lens-c"))
    accepted_measurements = (
        _measurement("lens-a", obs_tag=None, sigma_kms=240.0, sigma_err_kms=15.0),
        _measurement("lens-b", obs_tag="B", sigma_kms=251.0, sigma_err_kms=16.0),
        _measurement("lens-b", obs_tag="A", sigma_kms=249.0, sigma_err_kms=16.0),
    )

    result = resolve_sigma_observations(
        lenses,
        sigma_policy=SigmaPolicy(
            source_type="ppxf_results_adapter",
            missing_policy="num_sigma_zero",
        ),
        accepted_measurements=accepted_measurements,
    )

    assert [record.lens_id for record in result.records] == ["lens-a", "lens-b", "lens-c"]
    assert [record.num_sigma for record in result.records] == [1, 2, 0]
    assert [obs.obs_tag for obs in result.records[1].sigma_observations] == ["A", "B"]
    assert result.audit.num_sigma_distribution == {0: 1, 1: 1, 2: 1}
    assert result.audit.missing_lens_ids == ("lens-c",)


def test_external_mode_rejects_duplicate_tags() -> None:
    """Two accepted rows with the same tag must fail instead of being reordered."""

    with pytest.raises(ValueError, match="duplicate"):
        resolve_sigma_observations(
            (_base_lens("lens-a"),),
            sigma_policy=SigmaPolicy(source_type="ppxf_results_adapter", missing_policy="fail"),
            accepted_measurements=(
                _measurement("lens-a", obs_tag="A", sigma_kms=240.0, sigma_err_kms=15.0),
                _measurement("lens-a", obs_tag="A", sigma_kms=241.0, sigma_err_kms=16.0),
            ),
        )


def test_external_mode_rejects_more_than_two_measurements() -> None:
    """The canonical likelihood cannot store more than two accepted sigma rows."""

    with pytest.raises(ValueError, match="at most two"):
        resolve_sigma_observations(
            (_base_lens("lens-a"),),
            sigma_policy=SigmaPolicy(source_type="ppxf_results_adapter", missing_policy="fail"),
            accepted_measurements=(
                _measurement("lens-a", obs_tag=None, sigma_kms=240.0, sigma_err_kms=15.0),
                _measurement("lens-a", obs_tag="A", sigma_kms=241.0, sigma_err_kms=16.0),
                _measurement("lens-a", obs_tag="B", sigma_kms=242.0, sigma_err_kms=17.0),
            ),
        )


def test_external_mode_rejects_missing_lenses_when_policy_is_fail() -> None:
    """The fail policy should force every lens to have at least one measurement."""

    with pytest.raises(ValueError, match="missing"):
        resolve_sigma_observations(
            (_base_lens("lens-a"), _base_lens("lens-b")),
            sigma_policy=SigmaPolicy(source_type="ppxf_results_adapter", missing_policy="fail"),
            accepted_measurements=(
                _measurement("lens-a", obs_tag=None, sigma_kms=240.0, sigma_err_kms=15.0),
            ),
        )
