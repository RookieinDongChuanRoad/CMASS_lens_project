from __future__ import annotations

from pathlib import Path

import pytest


def test_posterior_diagnostics_result_dir_uses_run_owned_workspace_layout(tmp_path: Path) -> None:
    """
    Posterior diagnostics are artifacts of a completed inference run.

    They should therefore live under the run directory's
    ``posterior_predictive/diagnostics/<diagnostic_run_id>`` subtree rather
    than the older flat ``ppc`` directory.  The test exercises only path
    materialization; it deliberately does not run the expensive diagnostics
    workflow.
    """

    from statistical_sl.posterior_predictive.predictive import _materialize_result_dir

    result_dir = _materialize_result_dir(
        output_root_dir=tmp_path,
        profile_name="sersic",
        run_id="20260528_120000_sersic_contract",
        diagnostic_run_id="diagnostic-smoke",
    )

    assert result_dir == (
        tmp_path
        / "sersic"
        / "20260528_120000_sersic_contract"
        / "posterior_predictive"
        / "diagnostics"
        / "diagnostic-smoke"
    )
    assert result_dir.is_dir()


def test_posterior_diagnostics_rejects_path_traversal_run_ids(tmp_path: Path) -> None:
    """
    A diagnostics run id is a directory name, not a user-controlled path.

    Rejecting traversal-like names keeps the new run-owned diagnostics layout
    from writing outside ``posterior_predictive/diagnostics`` even when a CLI
    caller passes a malformed ``--diagnostic-run-id`` value.
    """

    from statistical_sl.posterior_predictive.predictive import _materialize_result_dir

    for bad_run_id in ("..", ".", "nested/path"):
        with pytest.raises(ValueError):
            _materialize_result_dir(
                output_root_dir=tmp_path,
                profile_name="sersic",
                run_id="20260528_120000_sersic_contract",
                diagnostic_run_id=bad_run_id,
            )


def test_fig8_fixed_display_xlim_is_separate_from_diagnostic_bins() -> None:
    """Fig. 8 binning can be broad while the curated display window stays fixed."""

    from statistical_sl.core.mass_definition import H_UNITS_V1, get_mass_definition
    from statistical_sl.posterior_predictive.predictive import (
        _build_fixed_fig8_display_xlim_by_panel,
    )

    mass_definition = get_mass_definition(5, unit_convention=H_UNITS_V1)
    limits = _build_fixed_fig8_display_xlim_by_panel(
        mass_definition=mass_definition,
        profile_name="devauc",
        h_ref=0.7,
    )

    assert limits[mass_definition.label] == pytest.approx((10.690196080028514, 11.490196080028515))
    assert limits["gamma"] == pytest.approx((10.690196080028514, 11.490196080028515))
    assert limits["sigma_ap"] == pytest.approx((10.690196080028514, 11.490196080028515))
    assert limits["gamma_vs_sigma_star"] == (8.9, 9.6)
    assert limits["gamma_vs_logre_kpc"] == pytest.approx((0.2950980400142568, 0.7950980400142568))
