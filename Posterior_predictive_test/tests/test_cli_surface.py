"""
Smoke tests for the standalone posterior-predictive package surface.

These tests are intentionally minimal and exist to drive the migration:
the new package must expose its own CLI and stop relying on the
`cmass_lens_inference` command surface for PPT-only workflows.
"""

from __future__ import annotations

import pytest


def test_standalone_cli_exposes_only_ppt_family_commands() -> None:
    """The new standalone package should own the PPT command surface."""

    from lensing_posterior_predictive.cli import build_argument_parser

    parser = build_argument_parser()

    posterior_predictive_args = parser.parse_args(
        [
            "posterior-predictive",
            "--run-dir",
            "/tmp/run",
            "--sigma-table",
            "/tmp/table.h5",
        ]
    )
    posterior_trends_args = parser.parse_args(
        [
            "posterior-trends",
            "--run-dir",
            "/tmp/run",
            "--sigma-table",
            "/tmp/table.h5",
        ]
    )
    posterior_diagnostics_args = parser.parse_args(
        [
            "posterior-diagnostics",
            "--run-dir",
            "/tmp/run",
            "--sigma-table",
            "/tmp/table.h5",
        ]
    )
    monitor_args = parser.parse_args(
        [
            "posterior-predictive-monitor",
        ]
    )
    annotate_args = parser.parse_args(
        [
            "annotate-fig8-observations",
            "--run-dir",
            "/tmp/run-a",
            "--run-dir",
            "/tmp/run-b",
        ]
    )
    assert posterior_predictive_args.command == "posterior-predictive"
    assert posterior_trends_args.command == "posterior-trends"
    assert posterior_diagnostics_args.command == "posterior-diagnostics"
    assert posterior_diagnostics_args.parent_sample_size == 10000
    assert posterior_diagnostics_args.n_mass_bins == 19
    assert monitor_args.command == "posterior-predictive-monitor"
    assert annotate_args.command == "annotate-fig8-observations"
    assert annotate_args.run_dir == ["/tmp/run-a", "/tmp/run-b"]

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "notebook" + "-comparison",
                "--chain-path",
                "/tmp/chain.h5",
                "--population-model-path",
                "/tmp/Population_model.py",
                "--sigma-table-path",
                "/tmp/table.h5",
                "--cross-section-path",
                "/tmp/cs.h5",
                "--output-dir",
                "/tmp/out",
            ]
        )


def test_standalone_cli_output_dir_explicit_override_wins() -> None:
    """Explicit output roots must override the package-level defaults."""

    from lensing_posterior_predictive.cli import build_argument_parser

    parser = build_argument_parser()
    output_root = "/tmp/custom_output"

    posterior_predictive_args = parser.parse_args(
        [
            "posterior-predictive",
            "--run-dir",
            "/tmp/run",
            "--sigma-table",
            "/tmp/table.h5",
            "--output-dir",
            output_root,
        ]
    )
    posterior_diagnostics_args = parser.parse_args(
        [
            "posterior-diagnostics",
            "--run-dir",
            "/tmp/run",
            "--sigma-table",
            "/tmp/table.h5",
            "--output-dir",
            output_root,
        ]
    )
    posterior_trends_args = parser.parse_args(
        [
            "posterior-trends",
            "--run-dir",
            "/tmp/run",
            "--sigma-table",
            "/tmp/table.h5",
            "--output-dir",
            output_root,
        ]
    )
    monitor_args = parser.parse_args(
        [
            "posterior-predictive-monitor",
            "--output-dir",
            output_root,
        ]
    )
    annotate_args = parser.parse_args(
        [
            "annotate-fig8-observations",
            "--outputs-root",
            output_root,
        ]
    )

    assert posterior_predictive_args.output_dir == output_root
    assert posterior_trends_args.output_dir == output_root
    assert posterior_diagnostics_args.output_dir == output_root
    assert monitor_args.output_dir == output_root
    assert annotate_args.outputs_root == output_root


def test_posterior_diagnostics_cli_advertises_numba_backend() -> None:
    """
    The combined PPC + trend diagnostics command should advertise the production
    Numba backend, not the retired JAX implementation.

    This is deliberately a CLI-surface test rather than an implementation
    import check: users discover the backend contract from the command help,
    and downstream automation often snapshots that help text.
    """

    from lensing_posterior_predictive.cli import build_argument_parser

    parser = build_argument_parser()
    subparser_action = next(
        action for action in parser._actions if getattr(action, "choices", None)
    )
    diagnostics_help = next(
        choice.help
        for choice in subparser_action._choices_actions
        if choice.dest == "posterior-diagnostics"
    )

    assert "Numba" in diagnostics_help
    assert "JAX" not in diagnostics_help
