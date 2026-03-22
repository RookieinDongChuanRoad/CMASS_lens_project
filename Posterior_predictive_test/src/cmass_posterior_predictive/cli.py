"""
Standalone command-line interface for posterior-predictive workflows.

This CLI intentionally excludes the inference engine's `run` and `resume`
commands. Those remain in `cmass_lens_inference`, while all PPT-family
commands live here after the migration.
"""

from __future__ import annotations

import argparse
import json

from .notebook_comparison import run_notebook_pipeline_comparison
from .predictive import (
    DEFAULT_EXTERNAL_SIGMA_DIR,
    DEFAULT_MONITOR_NOT_BEFORE,
    DEFAULT_N_REPLICATES,
    DEFAULT_PPC_OUTPUT_ROOT_DIR,
    DEFAULT_TREND_MASS_BIN_COUNT,
    DEFAULT_TREND_MASS_BIN_MAX,
    DEFAULT_TREND_MASS_BIN_MIN,
    DEFAULT_TREND_PARENT_SAMPLE_SIZE,
    DEFAULT_TREND_POSTERIOR_DRAWS,
    annotate_existing_fig8_like_figures_with_observations,
    run_posterior_predictive,
    wait_for_external_sigma_tables_and_run,
)
from .trends import run_posterior_trends


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the top-level CLI parser and its PPT-family subcommands."""

    parser = argparse.ArgumentParser(description="CMASS posterior-predictive workflows")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ppc_parser = subparsers.add_parser("posterior-predictive", help="Run posterior predictive checks")
    ppc_parser.add_argument("--run-dir", required=True, help="Completed inference run directory")
    ppc_parser.add_argument("--sigma-table", required=True, help="Path to the Jeans sigma-unit interpolation table")
    ppc_parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_PPC_OUTPUT_ROOT_DIR),
        help=f"Root directory for PPC artifacts (default: {DEFAULT_PPC_OUTPUT_ROOT_DIR})",
    )
    ppc_parser.add_argument("--n-replicates", type=int, default=DEFAULT_N_REPLICATES)
    ppc_parser.add_argument("--burn-in", default="auto")
    ppc_parser.add_argument("--seed", type=int, default=20260309)
    ppc_parser.add_argument("--candidate-pool-size", type=int, default=None)
    ppc_parser.add_argument("--worker-processes", type=int, default=None)

    trend_parser = subparsers.add_parser("posterior-trends", help="Generate Fig. 8-like posterior trend figures")
    trend_parser.add_argument("--run-dir", required=True, help="Completed inference run directory")
    trend_parser.add_argument("--sigma-table", required=True, help="Path to the Jeans sigma-unit interpolation table")
    trend_parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_PPC_OUTPUT_ROOT_DIR),
        help=f"Root directory for trend artifacts (default: {DEFAULT_PPC_OUTPUT_ROOT_DIR})",
    )
    trend_parser.add_argument("--n-posterior-draws", type=int, default=DEFAULT_TREND_POSTERIOR_DRAWS)
    trend_parser.add_argument("--burn-in", default="auto")
    trend_parser.add_argument("--seed", type=int, default=20260310)
    trend_parser.add_argument("--n-parent-sample", type=int, default=DEFAULT_TREND_PARENT_SAMPLE_SIZE)
    trend_parser.add_argument("--worker-processes", type=int, default=None)
    trend_parser.add_argument("--n-mass-bins", type=int, default=DEFAULT_TREND_MASS_BIN_COUNT)
    trend_parser.add_argument("--mass-bin-min", type=float, default=DEFAULT_TREND_MASS_BIN_MIN)
    trend_parser.add_argument("--mass-bin-max", type=float, default=DEFAULT_TREND_MASS_BIN_MAX)
    trend_parser.add_argument("--n-mass-grid", type=int, default=None, help=argparse.SUPPRESS)
    trend_parser.add_argument("--logmstar-min", type=float, default=None, help=argparse.SUPPRESS)
    trend_parser.add_argument("--logmstar-max", type=float, default=None, help=argparse.SUPPRESS)
    trend_parser.add_argument("--n-candidate-per-mass", type=int, default=None, help=argparse.SUPPRESS)

    monitor_parser = subparsers.add_parser(
        "posterior-predictive-monitor",
        help="Wait for external sigma tables, validate them, and run devauc/sersic PPC",
    )
    monitor_parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_PPC_OUTPUT_ROOT_DIR),
        help=f"Root directory for PPC artifacts (default: {DEFAULT_PPC_OUTPUT_ROOT_DIR})",
    )
    monitor_parser.add_argument("--external-dir", default=str(DEFAULT_EXTERNAL_SIGMA_DIR))
    monitor_parser.add_argument(
        "--devauc-run-dir",
        default="/Users/liurongfu/Work/CMASS_lens_project/outputs/devauc/latest",
    )
    monitor_parser.add_argument(
        "--sersic-run-dir",
        default="/Users/liurongfu/Work/CMASS_lens_project/outputs/sersic/latest",
    )
    monitor_parser.add_argument("--not-before", default=DEFAULT_MONITOR_NOT_BEFORE.isoformat())
    monitor_parser.add_argument("--poll-interval-seconds", type=float, default=30.0)
    monitor_parser.add_argument("--timeout-seconds", type=float, default=None)
    monitor_parser.add_argument("--n-replicates", type=int, default=DEFAULT_N_REPLICATES)
    monitor_parser.add_argument("--burn-in", default="auto")
    monitor_parser.add_argument("--seed", type=int, default=20260309)
    monitor_parser.add_argument("--candidate-pool-size", type=int, default=None)
    monitor_parser.add_argument("--worker-processes", type=int, default=None)

    annotate_parser = subparsers.add_parser(
        "annotate-fig8-observations",
        help="Backup and rewrite existing Fig. 8-like PNGs with observed lens points",
    )
    annotate_parser.add_argument(
        "--outputs-root",
        default=str(DEFAULT_PPC_OUTPUT_ROOT_DIR),
        help=f"Root directory containing profile run trees (default: {DEFAULT_PPC_OUTPUT_ROOT_DIR})",
    )
    annotate_parser.add_argument(
        "--run-dir",
        action="append",
        default=None,
        help="Explicit run directory to process. May be provided multiple times.",
    )
    annotate_parser.add_argument(
        "--raw-devauc",
        default=None,
    )
    annotate_parser.add_argument(
        "--raw-sersic",
        default=None,
    )
    annotate_parser.add_argument("--backup-prefix", default="pre_observed_points")

    comparison_parser = subparsers.add_parser("notebook-comparison", help="Compare notebook and standalone PPT")
    comparison_parser.add_argument("--chain-path", required=True)
    comparison_parser.add_argument("--population-model-path", required=True)
    comparison_parser.add_argument("--sigma-table-path", required=True)
    comparison_parser.add_argument("--cross-section-path", required=True)
    comparison_parser.add_argument("--output-dir", required=True)
    comparison_parser.add_argument("--pipeline-config-path", required=False, default="/Users/liurongfu/Work/CMASS_lens_project/outputs/sersic/latest/config_snapshot.yaml")
    comparison_parser.add_argument("--observation-path", required=False, default=None)
    comparison_parser.add_argument("--discard", type=int, default=1000)
    comparison_parser.add_argument("--max-samples", type=int, default=None)
    comparison_parser.add_argument("--num-parents", type=int, default=10000)
    comparison_parser.add_argument("--theta-sample-size", type=int, default=22)
    comparison_parser.add_argument("--sigma-sample-size", type=int, default=7)
    comparison_parser.add_argument("--seed", type=int, default=20260310)

    return parser


def main() -> None:
    """Parse arguments, dispatch the selected workflow, and print JSON."""

    parser = build_argument_parser()
    args = parser.parse_args()

    if args.command == "posterior-predictive":
        burn_in: str | int = args.burn_in
        if burn_in != "auto":
            burn_in = int(burn_in)
        result = run_posterior_predictive(
            run_dir=args.run_dir,
            sigma_table_path=args.sigma_table,
            output_root_dir=args.output_dir,
            n_replicates=args.n_replicates,
            burn_in=burn_in,
            random_seed=args.seed,
            candidate_pool_size=args.candidate_pool_size,
            worker_processes=args.worker_processes,
        )
    elif args.command == "posterior-trends":
        removed_arguments = {
            "--n-mass-grid": args.n_mass_grid,
            "--logmstar-min": args.logmstar_min,
            "--logmstar-max": args.logmstar_max,
            "--n-candidate-per-mass": args.n_candidate_per_mass,
        }
        provided_removed_arguments = [name for name, value in removed_arguments.items() if value is not None]
        if provided_removed_arguments:
            parser.error(
                "posterior-trends is now bin-based; do not pass "
                + ", ".join(provided_removed_arguments)
                + ". Use --n-parent-sample, --n-mass-bins, --mass-bin-min, and --mass-bin-max instead."
            )
        burn_in = args.burn_in
        if burn_in != "auto":
            burn_in = int(burn_in)
        result = run_posterior_trends(
            run_dir=args.run_dir,
            sigma_table_path=args.sigma_table,
            output_root_dir=args.output_dir,
            n_posterior_draws=args.n_posterior_draws,
            burn_in=burn_in,
            random_seed=args.seed,
            n_parent_sample=args.n_parent_sample,
            worker_processes=args.worker_processes,
            n_mass_bins=args.n_mass_bins,
            mass_bin_min=args.mass_bin_min,
            mass_bin_max=args.mass_bin_max,
        )
    elif args.command == "posterior-predictive-monitor":
        burn_in = args.burn_in
        if burn_in != "auto":
            burn_in = int(burn_in)
        result = wait_for_external_sigma_tables_and_run(
            output_root_dir=args.output_dir,
            devauc_run_dir=args.devauc_run_dir,
            sersic_run_dir=args.sersic_run_dir,
            external_dir=args.external_dir,
            not_before=args.not_before,
            poll_interval_seconds=args.poll_interval_seconds,
            timeout_seconds=args.timeout_seconds,
            n_replicates=args.n_replicates,
            burn_in=burn_in,
            random_seed=args.seed,
            candidate_pool_size=args.candidate_pool_size,
            worker_processes=args.worker_processes,
        )
    elif args.command == "annotate-fig8-observations":
        result = annotate_existing_fig8_like_figures_with_observations(
            outputs_root=args.outputs_root,
            run_dirs=args.run_dir,
            raw_devauc_path=args.raw_devauc,
            raw_sersic_path=args.raw_sersic,
            backup_prefix=args.backup_prefix,
        )
    else:
        result = run_notebook_pipeline_comparison(
            chain_path=args.chain_path,
            pipeline_config_path=args.pipeline_config_path,
            population_model_path=args.population_model_path,
            sigma_table_path=args.sigma_table_path,
            cross_section_path=args.cross_section_path,
            output_dir=args.output_dir,
            discard=args.discard,
            max_samples=args.max_samples,
            num_parents=args.num_parents,
            theta_sample_size=args.theta_sample_size,
            sigma_sample_size=args.sigma_sample_size,
            random_seed=args.seed,
            observation_path=args.observation_path,
        )

    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
