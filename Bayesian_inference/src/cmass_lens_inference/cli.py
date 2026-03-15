"""
Command-line interface for launching or resuming inference runs.

Posterior-predictive workflows have moved to the standalone
`cmass_posterior_predictive` package. This CLI now only exposes the inference
engine's own responsibilities.
"""

from __future__ import annotations

import argparse
import json

from .posterior_corner import run_latest_profile_corner_plots
from .runner import resume_inference, run_inference


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the top-level CLI parser and its inference-only subcommands."""

    parser = argparse.ArgumentParser(description="CMASS lens Bayesian inference runner")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Launch a new inference run")
    run_parser.add_argument("--config", required=True, help="Path to the YAML configuration file")
    run_parser.add_argument("--label", default=None, help="Optional run label overriding the config value")

    resume_parser = subparsers.add_parser("resume", help="Resume an existing run directory")
    resume_parser.add_argument("--run-dir", required=True, help="Existing run directory to resume")

    corner_parser = subparsers.add_parser(
        "posterior-corner-latest",
        help="Generate corner plots for the current devauc and sersic latest runs",
    )
    corner_parser.add_argument(
        "--devauc-run-dir",
        default="/Users/liurongfu/Work/CMASS_lens_project/outputs/devauc/latest",
        help="Completed devauc run directory or latest symlink",
    )
    corner_parser.add_argument(
        "--sersic-run-dir",
        default="/Users/liurongfu/Work/CMASS_lens_project/outputs/sersic/latest",
        help="Completed sersic run directory or latest symlink",
    )
    corner_parser.add_argument(
        "--burn-in",
        default="auto",
        help="Burn-in steps to discard, or 'auto' to reuse each run's stored warmup",
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
    else:
        burn_in: str | int = args.burn_in
        if burn_in != "auto":
            burn_in = int(burn_in)
        result = run_latest_profile_corner_plots(
            devauc_run_dir=args.devauc_run_dir,
            sersic_run_dir=args.sersic_run_dir,
            burn_in=burn_in,
        )

    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
