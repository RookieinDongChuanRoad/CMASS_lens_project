"""
Smoke tests for the standalone posterior-predictive package surface.

These tests are intentionally minimal and exist to drive the migration:
the new package must expose its own CLI and stop relying on the
`cmass_lens_inference` command surface for PPT-only workflows.
"""

from __future__ import annotations


def test_standalone_cli_exposes_only_ppt_family_commands() -> None:
    """The new standalone package should own the PPT command surface."""

    from cmass_posterior_predictive.cli import build_argument_parser

    parser = build_argument_parser()

    posterior_predictive_args = parser.parse_args(
        [
            "posterior-predictive",
            "--run-dir",
            "/tmp/run",
            "--sigma-table",
            "/tmp/table.h5",
            "--output-dir",
            "/tmp/out",
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
            "/tmp/out",
        ]
    )
    notebook_comparison_args = parser.parse_args(
        [
            "notebook-comparison",
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

    assert posterior_predictive_args.command == "posterior-predictive"
    assert posterior_trends_args.command == "posterior-trends"
    assert notebook_comparison_args.command == "notebook-comparison"
