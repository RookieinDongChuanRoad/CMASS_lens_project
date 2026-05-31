"""
Standalone command-line interface for posterior-predictive workflows.

This CLI intentionally excludes the inference engine's `run` and `resume`
commands. Those live under ``statistical_sl.inference``; this module owns the
posterior-predictive command family.
"""

from __future__ import annotations

import argparse
import json

from .config import load_posterior_diagnostics_config
from .predictive import (
    DEFAULT_EXTERNAL_SIGMA_DIR,
    DEFAULT_MONITOR_NOT_BEFORE,
    DEFAULT_N_REPLICATES,
    DEFAULT_DIAGNOSTICS_PARENT_SAMPLE_SIZE,
    DEFAULT_PPC_OUTPUT_ROOT_DIR,
    DEFAULT_TREND_MASS_BIN_COUNT,
    DEFAULT_TREND_MASS_BIN_MAX,
    DEFAULT_TREND_MASS_BIN_MIN,
    DEFAULT_TREND_PARENT_SAMPLE_SIZE,
    DEFAULT_TREND_POSTERIOR_DRAWS,
    annotate_existing_fig8_like_figures_with_observations,
    run_posterior_diagnostics,
    run_posterior_predictive,
    wait_for_external_sigma_tables_and_run,
)
from .trends import run_posterior_trends


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the top-level CLI parser and its PPT-family subcommands."""

    parser = argparse.ArgumentParser(description="Model-aware posterior-predictive workflows")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ppc_parser = subparsers.add_parser("posterior-predictive", help="Run posterior predictive checks")
    ppc_parser.add_argument("--run-dir", required=True, help="Completed inference run directory")
    ppc_parser.add_argument(
        "--sigma-table",
        default=None,
        help="Model-declared sigma-unit table path; required by current CMASS diagnostics",
    )
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
    trend_parser.add_argument(
        "--sigma-table",
        default=None,
        help="Model-declared sigma-unit table path; required by current CMASS diagnostics",
    )
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

    diagnostics_parser = subparsers.add_parser(
        "posterior-diagnostics",
        help="Run Numba shared-parent PPC and Fig. 8-like trend diagnostics",
    )
    diagnostics_parser.add_argument("--config", default=None, help="Posterior diagnostics YAML config")
    diagnostics_parser.add_argument("--run-dir", default=None, help="Completed inference run directory")
    diagnostics_parser.add_argument(
        "--sigma-table",
        default=None,
        help="Model-declared sigma-unit table path; required by current CMASS diagnostics",
    )
    diagnostics_parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_PPC_OUTPUT_ROOT_DIR),
        help=f"Root directory for diagnostics artifacts (default: {DEFAULT_PPC_OUTPUT_ROOT_DIR})",
    )
    diagnostics_parser.add_argument(
        "--diagnostic-run-id",
        default=None,
        help=(
            "Optional name for the diagnostics artifact directory under "
            "posterior_predictive/diagnostics/. Defaults to a timestamped id."
        ),
    )
    diagnostics_parser.add_argument("--n-posterior-draws", type=int, default=DEFAULT_TREND_POSTERIOR_DRAWS)
    diagnostics_parser.add_argument("--burn-in", default="auto")
    diagnostics_parser.add_argument("--seed", type=int, default=20260309)
    diagnostics_parser.add_argument("--parent-sample-size", type=int, default=DEFAULT_DIAGNOSTICS_PARENT_SAMPLE_SIZE)
    diagnostics_parser.add_argument("--worker-processes", type=int, default=None)
    diagnostics_parser.add_argument("--n-mass-bins", type=int, default=DEFAULT_TREND_MASS_BIN_COUNT)
    diagnostics_parser.add_argument("--mass-bin-min", type=float, default=DEFAULT_TREND_MASS_BIN_MIN)
    diagnostics_parser.add_argument("--mass-bin-max", type=float, default=DEFAULT_TREND_MASS_BIN_MAX)

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
        default="workspace/outputs/devauc/latest",
    )
    monitor_parser.add_argument(
        "--sersic-run-dir",
        default="workspace/outputs/sersic/latest",
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
    elif args.command == "posterior-diagnostics":
        burn_in = args.burn_in
        if burn_in != "auto":
            burn_in = int(burn_in)
        if args.config is not None:
            config = load_posterior_diagnostics_config(args.config)
            try:
                kwargs = config.to_run_kwargs(
                    run_dir_override=args.run_dir,
                    diagnostic_run_id=args.diagnostic_run_id,
                )
            except ValueError as exc:
                parser.error(str(exc))
            if args.sigma_table is not None:
                kwargs["sigma_table_path"] = args.sigma_table
            if args.output_dir != str(DEFAULT_PPC_OUTPUT_ROOT_DIR):
                kwargs["output_root_dir"] = args.output_dir
            if args.n_posterior_draws != DEFAULT_TREND_POSTERIOR_DRAWS:
                kwargs["n_posterior_draws"] = args.n_posterior_draws
            if args.burn_in != "auto":
                kwargs["burn_in"] = burn_in
            if args.seed != 20260309:
                kwargs["random_seed"] = args.seed
            if args.parent_sample_size != DEFAULT_DIAGNOSTICS_PARENT_SAMPLE_SIZE:
                kwargs["parent_sample_size"] = args.parent_sample_size
            if args.worker_processes is not None:
                kwargs["worker_processes"] = args.worker_processes
            if args.n_mass_bins != DEFAULT_TREND_MASS_BIN_COUNT:
                kwargs["n_mass_bins"] = args.n_mass_bins
            if args.mass_bin_min != DEFAULT_TREND_MASS_BIN_MIN:
                kwargs["mass_bin_min"] = args.mass_bin_min
            if args.mass_bin_max != DEFAULT_TREND_MASS_BIN_MAX:
                kwargs["mass_bin_max"] = args.mass_bin_max
            result = run_posterior_diagnostics(**kwargs)
        else:
            if args.run_dir is None:
                parser.error("posterior-diagnostics requires --run-dir unless --config provides inputs.inference_run_dir.")
            result = run_posterior_diagnostics(
                run_dir=args.run_dir,
                sigma_table_path=args.sigma_table,
                output_root_dir=args.output_dir,
                diagnostic_run_id=args.diagnostic_run_id,
                n_posterior_draws=args.n_posterior_draws,
                burn_in=burn_in,
                random_seed=args.seed,
                parent_sample_size=args.parent_sample_size,
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
    else:  # pragma: no cover - argparse prevents this for supported command sets.
        parser.error(f"Unsupported command: {args.command}")

    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
