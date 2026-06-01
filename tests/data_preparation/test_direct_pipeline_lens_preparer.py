"""Tests for direct-pipeline lens preparation and unit normalization."""

from __future__ import annotations

import numpy as np
import pytest

from statistical_sl.data_preparation.direct_pipeline.policies import AperturePolicyRef, UnitPolicy
from statistical_sl.data_preparation.direct_pipeline.records import BaseLensRecord, PreparedLensRecord, SigmaObservation
from statistical_sl.data_preparation.direct_pipeline import lens_preparer
from statistical_sl.data_preparation.direct_pipeline.lens_preparer import prepare_lens_records
from statistical_sl.data_preparation.models import AperturePolicy
from statistical_sl.core.unit_conventions import logMstar_h2_from_legacy, logRe_hinv_from_legacy


def _base_lens() -> BaseLensRecord:
    """Build one lens with physical-size fields already supplied by the catalog."""

    return BaseLensRecord(
        lens_id="lens-a",
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
    )


def _base_lens_with_id(lens_id: str) -> BaseLensRecord:
    """Build a valid lens while varying only the canonical join key."""

    base = _base_lens()
    return BaseLensRecord(
        lens_id=lens_id,
        z_lens=base.z_lens,
        z_source=base.z_source,
        theta_ein_arcsec=base.theta_ein_arcsec,
        theta_ein_kpc=base.theta_ein_kpc,
        effective_radius_arcsec=base.effective_radius_arcsec,
        effective_radius_kpc=base.effective_radius_kpc,
        log_stellar_mass=base.log_stellar_mass,
        log_stellar_mass_err=base.log_stellar_mass_err,
        profile_name=base.profile_name,
        sersic_index=base.sersic_index,
    )


def _resolved_record(*, measurement_aperture: bool = False) -> PreparedLensRecord:
    """Build a sigma-resolved record before physical preparation."""

    aperture_kwargs = {}
    if measurement_aperture:
        aperture_kwargs = {
            "aperture_shape": "circular",
            "aperture_radius_arcsec": 1.5,
            "seeing_fwhm_arcsec": 1.2,
        }
    return PreparedLensRecord(
        base_lens=_base_lens(),
        sigma_observations=(
            SigmaObservation(
                lens_id="lens-a",
                sigma_kms=240.0,
                sigma_err_kms=15.0,
                **aperture_kwargs,
            ),
        ),
    )


def _default_aperture_ref() -> AperturePolicyRef:
    """Dataset-level slit aperture contract used by most preparation tests."""

    return AperturePolicyRef(
        observation_flavor="slit",
        sigma_definition="observed_aperture",
        aperture_policy=AperturePolicy.rectangular(
            width_arcsec=1.6,
            height_arcsec=0.9,
            seeing_fwhm_arcsec=0.9,
        ),
    )


def test_fixed_kpc_preparation_preserves_physical_mass_and_size(monkeypatch: pytest.MonkeyPatch) -> None:
    """Legacy fixed-kpc preparation should not apply h-dependent shifts."""

    monkeypatch.setattr(lens_preparer, "sigma_critical_surface_density", lambda _zd, _zs: 123.0)

    prepared = prepare_lens_records(
        (_resolved_record(),),
        unit_policy=UnitPolicy(unit_convention="legacy_fixed_kpc", h_ref=0.7),
        aperture_policy_ref=_default_aperture_ref(),
    )[0]

    assert prepared.sigma_crit == pytest.approx(123.0)
    assert prepared.theta_ein_kpc == pytest.approx(3.0)
    assert prepared.effective_radius_kpc == pytest.approx(5.0)
    assert prepared.active_log_stellar_mass == pytest.approx(11.2)
    assert prepared.active_log_effective_radius == pytest.approx(np.log10(5.0))
    assert prepared.unit_convention == "legacy_fixed_kpc"


def test_h_units_preparation_shifts_mass_and_size_with_existing_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    """h-units preparation should use the same algebra as existing helpers."""

    h_ref = 0.7
    monkeypatch.setattr(lens_preparer, "sigma_critical_surface_density", lambda _zd, _zs: 123.0)

    prepared = prepare_lens_records(
        (_resolved_record(),),
        unit_policy=UnitPolicy(unit_convention="h_units_v1", h_ref=h_ref),
        aperture_policy_ref=_default_aperture_ref(),
    )[0]

    physical_log_re = np.log10(5.0)
    assert prepared.active_log_stellar_mass == pytest.approx(float(logMstar_h2_from_legacy(11.2, h_ref=h_ref)))
    assert prepared.active_log_effective_radius == pytest.approx(float(logRe_hinv_from_legacy(physical_log_re, h_ref=h_ref)))
    assert prepared.unit_convention == "h_units_v1"
    assert prepared.h_ref == pytest.approx(h_ref)


def test_sigma_crit_is_computed_once_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Preparation should compute Sigma_crit once per unprepared record."""

    calls: list[tuple[float, float]] = []

    def fake_sigma_crit(z_lens: float, z_source: float) -> float:
        calls.append((z_lens, z_source))
        return 321.0

    monkeypatch.setattr(lens_preparer, "sigma_critical_surface_density", fake_sigma_crit)

    prepared = prepare_lens_records(
        (_resolved_record(),),
        unit_policy=UnitPolicy(unit_convention="legacy_fixed_kpc"),
        aperture_policy_ref=_default_aperture_ref(),
    )[0]

    assert prepared.sigma_crit == pytest.approx(321.0)
    assert calls == [(0.5, 1.5)]


def test_aperture_metadata_is_explicit_after_preparation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Dataset-level aperture policy should be copied onto prepared records."""

    monkeypatch.setattr(lens_preparer, "sigma_critical_surface_density", lambda _zd, _zs: 123.0)

    prepared = prepare_lens_records(
        (_resolved_record(),),
        unit_policy=UnitPolicy(unit_convention="legacy_fixed_kpc"),
        aperture_policy_ref=_default_aperture_ref(),
    )[0]

    assert prepared.observation_flavor == "slit"
    assert prepared.sigma_definition == "observed_aperture"
    assert prepared.aperture_policy is not None
    assert prepared.aperture_policy.shape == "rectangular"
    assert prepared.aperture_policy.width_arcsec == pytest.approx(1.6)
    assert prepared.aperture_policy.height_arcsec == pytest.approx(0.9)


def test_measurement_aperture_geometry_can_override_dataset_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Row-level aperture geometry should win when upstream measurements provide it."""

    monkeypatch.setattr(lens_preparer, "sigma_critical_surface_density", lambda _zd, _zs: 123.0)

    prepared = prepare_lens_records(
        (_resolved_record(measurement_aperture=True),),
        unit_policy=UnitPolicy(unit_convention="legacy_fixed_kpc"),
        aperture_policy_ref=_default_aperture_ref(),
    )[0]

    assert prepared.aperture_policy is not None
    assert prepared.aperture_policy.shape == "circular"
    assert prepared.aperture_policy.radius_arcsec == pytest.approx(1.5)
    assert prepared.aperture_policy.seeing_fwhm_arcsec == pytest.approx(1.2)


def test_missing_aperture_metadata_fails_when_no_default_exists(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prepared records must not leave aperture metadata implicit."""

    monkeypatch.setattr(lens_preparer, "sigma_critical_surface_density", lambda _zd, _zs: 123.0)

    with pytest.raises(ValueError, match="aperture"):
        prepare_lens_records(
            (_resolved_record(),),
            unit_policy=UnitPolicy(unit_convention="legacy_fixed_kpc"),
            aperture_policy_ref=None,
        )


def test_per_lens_measurement_apertures_may_differ(monkeypatch: pytest.MonkeyPatch) -> None:
    """Different lenses may carry different real measurement apertures."""

    monkeypatch.setattr(lens_preparer, "sigma_critical_surface_density", lambda _zd, _zs: 123.0)
    records = (
        PreparedLensRecord(
            base_lens=_base_lens_with_id("lens-a"),
            sigma_observations=(
                SigmaObservation(
                    lens_id="lens-a",
                    sigma_kms=240.0,
                    sigma_err_kms=15.0,
                    aperture_shape="rectangular",
                    aperture_width_arcsec=1.6,
                    aperture_height_arcsec=0.9,
                    seeing_fwhm_arcsec=0.7,
                ),
            ),
        ),
        PreparedLensRecord(
            base_lens=_base_lens_with_id("lens-b"),
            sigma_observations=(
                SigmaObservation(
                    lens_id="lens-b",
                    sigma_kms=250.0,
                    sigma_err_kms=16.0,
                    aperture_shape="circular",
                    aperture_radius_arcsec=1.5,
                    seeing_fwhm_arcsec=1.2,
                ),
            ),
        ),
    )

    prepared = prepare_lens_records(
        records,
        unit_policy=UnitPolicy(unit_convention="legacy_fixed_kpc"),
        aperture_policy_ref=None,
    )

    assert prepared[0].aperture_policy.shape == "rectangular"
    assert prepared[0].aperture_policy.width_arcsec == pytest.approx(1.6)
    assert prepared[0].aperture_policy.seeing_fwhm_arcsec == pytest.approx(0.7)
    assert prepared[1].aperture_policy.shape == "circular"
    assert prepared[1].aperture_policy.radius_arcsec == pytest.approx(1.5)
    assert prepared[1].aperture_policy.seeing_fwhm_arcsec == pytest.approx(1.2)


def test_same_lens_two_measurements_may_share_one_aperture(monkeypatch: pytest.MonkeyPatch) -> None:
    """The current per-lens s2 schema supports A/B rows only with one aperture."""

    monkeypatch.setattr(lens_preparer, "sigma_critical_surface_density", lambda _zd, _zs: 123.0)
    record = PreparedLensRecord(
        base_lens=_base_lens_with_id("lens-a"),
        sigma_observations=(
            SigmaObservation(
                lens_id="lens-a",
                obs_tag="A",
                sigma_kms=240.0,
                sigma_err_kms=15.0,
                aperture_shape="rectangular",
                aperture_width_arcsec=1.6,
                aperture_height_arcsec=0.9,
                seeing_fwhm_arcsec=0.7,
            ),
            SigmaObservation(
                lens_id="lens-a",
                obs_tag="B",
                sigma_kms=245.0,
                sigma_err_kms=16.0,
                aperture_shape="rectangular",
                aperture_width_arcsec=1.6,
                aperture_height_arcsec=0.9,
                seeing_fwhm_arcsec=0.7,
            ),
        ),
    )

    prepared = prepare_lens_records(
        (record,),
        unit_policy=UnitPolicy(unit_convention="legacy_fixed_kpc"),
        aperture_policy_ref=None,
    )[0]

    assert prepared.num_sigma == 2
    assert prepared.aperture_policy.shape == "rectangular"
    assert prepared.aperture_policy.seeing_fwhm_arcsec == pytest.approx(0.7)


def test_same_lens_two_measurements_reject_different_apertures(monkeypatch: pytest.MonkeyPatch) -> None:
    """A/B rows with different apertures require a future per-observation s2 schema."""

    monkeypatch.setattr(lens_preparer, "sigma_critical_surface_density", lambda _zd, _zs: 123.0)
    record = PreparedLensRecord(
        base_lens=_base_lens_with_id("lens-a"),
        sigma_observations=(
            SigmaObservation(
                lens_id="lens-a",
                obs_tag="A",
                sigma_kms=240.0,
                sigma_err_kms=15.0,
                aperture_shape="rectangular",
                aperture_width_arcsec=1.6,
                aperture_height_arcsec=0.9,
                seeing_fwhm_arcsec=0.7,
            ),
            SigmaObservation(
                lens_id="lens-a",
                obs_tag="B",
                sigma_kms=245.0,
                sigma_err_kms=16.0,
                aperture_shape="rectangular",
                aperture_width_arcsec=1.6,
                aperture_height_arcsec=0.9,
                seeing_fwhm_arcsec=0.9,
            ),
        ),
    )

    with pytest.raises(ValueError, match="inconsistent explicit aperture"):
        prepare_lens_records(
            (record,),
            unit_policy=UnitPolicy(unit_convention="legacy_fixed_kpc"),
            aperture_policy_ref=None,
        )
