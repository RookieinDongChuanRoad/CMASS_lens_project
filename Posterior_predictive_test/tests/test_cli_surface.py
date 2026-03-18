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
    assert monitor_args.command == "posterior-predictive-monitor"
    assert annotate_args.command == "annotate-fig8-observations"
    assert annotate_args.run_dir == ["/tmp/run-a", "/tmp/run-b"]
    assert notebook_comparison_args.command == "notebook-comparison"


def test_standalone_cli_output_dir_explicit_override_wins() -> None:
    """Explicit output roots must override the package-level defaults."""

    from cmass_posterior_predictive.cli import build_argument_parser

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
    assert monitor_args.output_dir == output_root
    assert annotate_args.outputs_root == output_root
