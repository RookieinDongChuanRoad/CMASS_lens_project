#!/usr/bin/env python3
"""
Run the notebook-vs-pipeline apples-to-apples comparison on `full_0103.h5`.

This script is intentionally kept outside the installable package because it
is a one-off scientific comparison workflow, not a production CLI contract.
The heavy lifting lives in `cmass_lens_inference.notebook_comparison`; this
file only wires real default paths and exposes a small command-line surface.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path("/Users/liurongfu/Work/CMASS_lens_project/Bayesian_inference")
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from cmass_lens_inference.notebook_comparison import run_notebook_pipeline_comparison


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the argument parser with notebook-comparison defaults."""

    parser = argparse.ArgumentParser(description="Compare notebook PPT vs local pipeline on full_0103.h5")
    parser.add_argument(
        "--chain-path",
        default="/Users/liurongfu/Desktop/Spectrum_reduction/data/chains/full_0103.h5",
        help="Posterior chain used by both engines",
    )
    parser.add_argument(
        "--population-model-path",
        default="/Users/liurongfu/Desktop/Spectrum_reduction/Population_model.py",
        help="Notebook Population_model.py module",
    )
    parser.add_argument(
        "--sigma-table-path",
        default="/Users/liurongfu/Desktop/Spectrum_reduction/data/jeans_sers_grid.h5",
        help="Notebook-native sersic sigma interpolation table",
    )
    parser.add_argument(
        "--cross-section-path",
        default="/Users/liurongfu/Desktop/Spectrum_reduction/data/cs_grid_power.h5",
        help="Cross-section grid consumed by both engines",
    )
    parser.add_argument(
        "--observation-path",
        default="/Users/liurongfu/Desktop/Spectrum_reduction/data/observations_with_m5_grids.hdf5",
        help="Optional observation file override used when building the pipeline context",
    )
    parser.add_argument(
        "--pipeline-config-path",
        default="/Users/liurongfu/Work/CMASS_lens_project/outputs/sersic/latest/config_snapshot.yaml",
        help="Pipeline config snapshot used to build the local sersic context",
    )
    parser.add_argument(
        "--output-dir",
        default="/Users/liurongfu/Work/CMASS_lens_project/Posterior_predictive_test/results/notebook_vs_pipeline/full_0103",
        help="Directory that will receive comparison artifacts",
    )
    parser.add_argument("--discard", type=int, default=1000, help="Number of MCMC steps to discard before flattening")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap on post-burn-in posterior samples for quick debugging runs",
    )
    parser.add_argument("--num-parents", type=int, default=10000, help="Notebook parent-population size")
    parser.add_argument("--theta-sample-size", type=int, default=22, help="Replicated theta_E lens count")
    parser.add_argument("--sigma-sample-size", type=int, default=7, help="Replicated sigma lens count")
    parser.add_argument("--seed", type=int, default=20260310, help="Base random seed for deterministic per-sample runs")
    return parser


def main() -> None:
    """Parse arguments, run the comparison, and print a JSON result summary."""

    args = build_argument_parser().parse_args()
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
