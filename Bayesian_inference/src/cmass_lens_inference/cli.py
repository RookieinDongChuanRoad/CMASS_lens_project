"""
Command-line interface for launching or resuming inference runs.

The CLI intentionally mirrors the public Python API so automation can choose
between direct imports and shell execution without encountering divergent
behavior.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .posterior_predictive import (
    DEFAULT_EXTERNAL_SIGMA_DIR,
    DEFAULT_MONITOR_NOT_BEFORE,
    DEFAULT_N_REPLICATES,
    DEFAULT_TREND_MASS_BIN_COUNT,
    DEFAULT_TREND_MASS_BIN_MAX,
    DEFAULT_TREND_MASS_BIN_MIN,
    DEFAULT_TREND_PARENT_SAMPLE_SIZE,
    DEFAULT_TREND_POSTERIOR_DRAWS,
    run_posterior_trends,
    wait_for_external_sigma_tables_and_run,
    run_posterior_predictive,
)
from .runner import resume_inference, run_inference


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the top-level CLI parser and its subcommands."""

    parser = argparse.ArgumentParser(description="CMASS lens Bayesian inference runner")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Launch a new inference run")
    run_parser.add_argument("--config", required=True, help="Path to the YAML configuration file")
    run_parser.add_argument("--label", default=None, help="Optional run label overriding the config value")

    resume_parser = subparsers.add_parser("resume", help="Resume an existing run directory")
    resume_parser.add_argument("--run-dir", required=True, help="Existing run directory to resume")

    ppc_parser = subparsers.add_parser(
        "posterior-predictive",
        help="Run the posterior predictive test for a completed run directory",
    )
    ppc_parser.add_argument("--run-dir", required=True, help="Completed inference run directory")
    ppc_parser.add_argument(
        "--sigma-table",
        required=True,
        help="Path to the Jeans sigma-unit interpolation table consumed by the PPC step",
    )
    ppc_parser.add_argument(
        "--output-dir",
        required=True,
        help="Root directory under which PPC artifacts will be written",
    )
    ppc_parser.add_argument(
        "--n-replicates",
        type=int,
        default=DEFAULT_N_REPLICATES,
        help="Optional posterior-draw count; omit to use the canonical tail-capped full-chain policy",
    )
    ppc_parser.add_argument(
        "--burn-in",
        default="auto",
        help="Integer MCMC burn-in step count, or 'auto' to reuse the config warmup value",
    )
    ppc_parser.add_argument("--seed", type=int, default=20260309, help="Random seed for PPC sampling")
    ppc_parser.add_argument(
        "--candidate-pool-size",
        type=int,
        default=None,
        help="Number of candidate lenses materialized per posterior draw before weighted sampling",
    )
    ppc_parser.add_argument(
        "--worker-processes",
        type=int,
        default=None,
        help="Optional PPC process count; omit to auto-resolve from CPU count and reserve_cores",
    )

    trend_parser = subparsers.add_parser(
        "posterior-trends",
        help="Generate the Fig. 8-like posterior trend figure for a completed run directory",
    )
    trend_parser.add_argument("--run-dir", required=True, help="Completed inference run directory")
    trend_parser.add_argument(
        "--sigma-table",
        required=True,
        help="Path to the Jeans sigma-unit interpolation table consumed by the trend step",
    )
    trend_parser.add_argument(
        "--output-dir",
        required=True,
        help="Root directory under which the trend artifacts will be written",
    )
    trend_parser.add_argument(
        "--n-posterior-draws",
        type=int,
        default=DEFAULT_TREND_POSTERIOR_DRAWS,
        help="Number of posterior hyper-parameter draws used to build uncertainty bands",
    )
    trend_parser.add_argument(
        "--burn-in",
        default="auto",
        help="Integer MCMC burn-in step count, or 'auto' to reuse the config warmup value",
    )
    trend_parser.add_argument("--seed", type=int, default=20260310, help="Random seed for trend sampling")
    trend_parser.add_argument(
        "--n-parent-sample",
        type=int,
        default=DEFAULT_TREND_PARENT_SAMPLE_SIZE,
        help="Number of parent-population galaxies sampled for each posterior draw",
    )
    trend_parser.add_argument(
        "--n-mass-bins",
        type=int,
        default=DEFAULT_TREND_MASS_BIN_COUNT,
        help="Number of stellar-mass bins used to summarize the sampled parent population",
    )
    trend_parser.add_argument(
        "--mass-bin-min",
        type=float,
        default=DEFAULT_TREND_MASS_BIN_MIN,
        help="Lower edge of the stellar-mass binning range",
    )
    trend_parser.add_argument(
        "--mass-bin-max",
        type=float,
        default=DEFAULT_TREND_MASS_BIN_MAX,
        help="Upper edge of the stellar-mass binning range",
    )
    # Keep the retired conditional-curve arguments hidden but explicitly
    # rejected so callers get a targeted migration message instead of a generic
    # argparse "unrecognized arguments" failure.
    trend_parser.add_argument(
        "--n-mass-grid",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    trend_parser.add_argument("--logmstar-min", type=float, default=None, help=argparse.SUPPRESS)
    trend_parser.add_argument("--logmstar-max", type=float, default=None, help=argparse.SUPPRESS)
    trend_parser.add_argument("--n-candidate-per-mass", type=int, default=None, help=argparse.SUPPRESS)

    monitor_parser = subparsers.add_parser(
        "posterior-predictive-monitor",
        help="Wait for external sigma tables to be updated, validate them, then run devauc and sersic PPC",
    )
    monitor_parser.add_argument(
        "--output-dir",
        required=True,
        help="Root directory under which PPC artifacts will be written for both profiles",
    )
    monitor_parser.add_argument(
        "--external-dir",
        default=str(DEFAULT_EXTERNAL_SIGMA_DIR),
        help="Directory containing the externally produced `jeans_deV_grid.h5` and `jeans_sers_grid.h5` files",
    )
    monitor_parser.add_argument(
        "--devauc-run-dir",
        default="/Users/liurongfu/Work/CMASS_lens_project/outputs/devauc/latest",
        help="Completed devauc inference run directory",
    )
    monitor_parser.add_argument(
        "--sersic-run-dir",
        default="/Users/liurongfu/Work/CMASS_lens_project/outputs/sersic/latest",
        help="Completed sersic inference run directory",
    )
    monitor_parser.add_argument(
        "--not-before",
        default=DEFAULT_MONITOR_NOT_BEFORE.isoformat(),
        help="Only tables modified after this ISO-8601 timestamp are considered ready",
    )
    monitor_parser.add_argument(
        "--poll-interval-seconds",
        type=float,
        default=30.0,
        help="Sleep interval between readiness checks while waiting for external tables",
    )
    monitor_parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=None,
        help="Optional hard timeout for waiting on external tables",
    )
    monitor_parser.add_argument(
        "--n-replicates",
        type=int,
        default=DEFAULT_N_REPLICATES,
        help="Optional posterior-draw count; omit to use the canonical tail-capped full-chain policy",
    )
    monitor_parser.add_argument(
        "--burn-in",
        default="auto",
        help="Integer MCMC burn-in step count, or 'auto' to reuse the stored config warmup value",
    )
    monitor_parser.add_argument("--seed", type=int, default=20260309, help="Random seed for PPC sampling")
    monitor_parser.add_argument(
        "--candidate-pool-size",
        type=int,
        default=None,
        help="Number of candidate lenses materialized per posterior draw before weighted sampling",
    )
    monitor_parser.add_argument(
        "--worker-processes",
        type=int,
        default=None,
        help="Optional PPC process count; omit to auto-resolve from CPU count and reserve_cores",
    )

    return parser


def main() -> None:
    """Parse CLI arguments, dispatch the selected command, and print JSON."""

    parser = build_argument_parser()
    args = parser.parse_args()

    if args.command == "run":
        result = run_inference(args.config, label=args.label)
    elif args.command == "resume":
        result = resume_inference(args.run_dir)
    elif args.command == "posterior-predictive":
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
            n_mass_bins=args.n_mass_bins,
            mass_bin_min=args.mass_bin_min,
            mass_bin_max=args.mass_bin_max,
        )
    else:
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

    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
