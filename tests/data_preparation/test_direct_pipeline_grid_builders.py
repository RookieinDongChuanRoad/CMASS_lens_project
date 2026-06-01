"""Tests for direct-pipeline in-memory grid builders."""

from __future__ import annotations

import numpy as np
import pytest

from statistical_sl.data_preparation.direct_pipeline import grid_builders
from statistical_sl.data_preparation.direct_pipeline.grid_builders import build_derived_grid_blocks
from statistical_sl.data_preparation.direct_pipeline.policies import MassDefinitionPolicy, UnitPolicy
from statistical_sl.data_preparation.direct_pipeline.records import BaseLensRecord, PreparedLensRecord, SigmaObservation
from statistical_sl.data_preparation.models import AperturePolicy


def _base_lens(lens_id: str, *, profile_name: str = "devauc") -> BaseLensRecord:
    """Create one catalog record with values already normalized by the preparer."""

    return BaseLensRecord(
        lens_id=lens_id,
        z_lens=0.5,
        z_source=1.5,
        theta_ein_arcsec=1.0,
        theta_ein_kpc=3.0,
        effective_radius_arcsec=0.8,
        effective_radius_kpc=5.0,
        log_stellar_mass=11.2,
        log_stellar_mass_err=0.08,
        profile_name=profile_name,
        sersic_index=4.0 if profile_name == "devauc" else 5.0,
    )


def _prepared_record(lens_id: str, *, num_sigma: int) -> PreparedLensRecord:
    """Create one fully prepared record for grid-builder tests."""

    observations = ()
    if num_sigma:
        observations = (
            SigmaObservation(
                lens_id=lens_id,
                sigma_kms=240.0,
                sigma_err_kms=15.0,
            ),
        )
    return PreparedLensRecord(
        base_lens=_base_lens(lens_id),
        sigma_observations=observations,
        sigma_crit=2.0e9,
        aperture_policy=AperturePolicy.rectangular(
            width_arcsec=1.6,
            height_arcsec=0.9,
            seeing_fwhm_arcsec=0.9,
        ),
        observation_flavor="slit",
        sigma_definition="observed_aperture",
        unit_convention="h_units_v1",
        h_ref=0.7,
        theta_ein_kpc=3.0,
        effective_radius_kpc=5.0,
        active_log_stellar_mass=11.0,
        active_log_effective_radius=0.6,
    )


def test_build_derived_grid_blocks_uses_existing_physics_kernels(monkeypatch: pytest.MonkeyPatch) -> None:
    """The orchestration layer should reuse existing mass and Jeans kernels."""

    gamma_axis = np.asarray([1.2, 2.0], dtype=float)
    records = (
        _prepared_record("lens-with-sigma", num_sigma=1),
        _prepared_record("lens-without-sigma", num_sigma=0),
    )
    calls: dict[str, list[dict[str, object]]] = {"mass": [], "derivative": [], "sigma": []}

    def fake_compute_mass_grid(**kwargs: object) -> np.ndarray:
        calls["mass"].append(dict(kwargs))
        return np.full(gamma_axis.shape, 10.0 + len(calls["mass"]), dtype=float)

    def fake_compute_derivative_grid(**kwargs: object) -> np.ndarray:
        calls["derivative"].append(dict(kwargs))
        return np.full(gamma_axis.shape, 0.1 * len(calls["derivative"]), dtype=float)

    def fake_compute_sigma_unit_grid(**kwargs: object) -> np.ndarray:
        calls["sigma"].append(dict(kwargs))
        return np.full(gamma_axis.shape, 1.0e-5, dtype=float)

    monkeypatch.setattr(grid_builders, "compute_mass_grid", fake_compute_mass_grid)
    monkeypatch.setattr(grid_builders, "compute_dmass_dthetaein_grid", fake_compute_derivative_grid)
    monkeypatch.setattr(grid_builders, "compute_sigma_unit_grid", fake_compute_sigma_unit_grid)

    blocks = build_derived_grid_blocks(
        records,
        gamma_axis=gamma_axis,
        mass_policy=MassDefinitionPolicy(mass_definition_label="m5_hinvkpc", mass_radius_kpc=5.0),
        unit_policy=UnitPolicy(unit_convention="h_units_v1", h_ref=0.7),
        derivative_theta_samples=7,
    )

    assert blocks.mass.lens_ids == ("lens-with-sigma", "lens-without-sigma")
    assert blocks.mass.log_enclosed_mass_grid.shape == (2, 2)
    assert blocks.mass.dmass_dthetaein_grid.shape == (2, 2)
    assert blocks.velocity.s2_grid.shape == (2, 2)
    np.testing.assert_array_equal(blocks.velocity.has_s2, np.asarray([True, False]))
    np.testing.assert_allclose(blocks.velocity.s2_grid[0], np.asarray([1.0e-5, 1.0e-5]))
    np.testing.assert_allclose(blocks.velocity.s2_grid[1], np.zeros_like(gamma_axis))

    assert len(calls["mass"]) == 2
    assert len(calls["derivative"]) == 2
    assert len(calls["sigma"]) == 1
    assert calls["mass"][0]["unit_convention"] == "h_units_v1"
    assert calls["mass"][0]["h_ref"] == pytest.approx(0.7)
    assert calls["derivative"][0]["theta_samples"] == 7
    assert calls["sigma"][0]["profile_name"] == "devauc"
    assert calls["sigma"][0]["mass_radius_kpc"] == pytest.approx(5.0)


def test_s2_grid_is_not_required_for_sigma_free_lenses(monkeypatch: pytest.MonkeyPatch) -> None:
    """A sample with only num_sigma=0 lenses should not invoke the Jeans kernel."""

    gamma_axis = np.asarray([1.2, 2.0], dtype=float)
    records = (_prepared_record("lens-without-sigma", num_sigma=0),)

    monkeypatch.setattr(grid_builders, "compute_mass_grid", lambda **_kwargs: np.ones_like(gamma_axis))
    monkeypatch.setattr(grid_builders, "compute_dmass_dthetaein_grid", lambda **_kwargs: np.ones_like(gamma_axis))

    def fail_if_called(**_kwargs: object) -> np.ndarray:
        raise AssertionError("compute_sigma_unit_grid should not be called for num_sigma=0 lenses")

    monkeypatch.setattr(grid_builders, "compute_sigma_unit_grid", fail_if_called)

    blocks = build_derived_grid_blocks(
        records,
        gamma_axis=gamma_axis,
        mass_policy=MassDefinitionPolicy(mass_definition_label="m5_hinvkpc", mass_radius_kpc=5.0),
        unit_policy=UnitPolicy(unit_convention="h_units_v1", h_ref=0.7),
    )

    assert blocks.velocity.has_s2.tolist() == [False]
    np.testing.assert_allclose(blocks.velocity.s2_grid, np.zeros((1, 2)))
