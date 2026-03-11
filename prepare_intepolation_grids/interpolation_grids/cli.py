"""Command-line interface for interpolation-grid generation."""

from __future__ import annotations

import argparse
from pathlib import Path

from interpolation_grids.config import DEFAULT_INPUT_FILENAMES, EXTERNAL_DATA_DIRECTORY, RAW_DATA_DIRECTORY
from interpolation_grids.io.hdf5 import process_hdf5_file
from interpolation_grids.io.sigma_tables import build_default_sigma_unit_hdf5_tables


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser used by `python -m interpolation_grids`."""

    parser = argparse.ArgumentParser(
        description="Recompute m5, derivative, and velocity-dispersion interpolation grids.",
    )
    parser.add_argument(
        "--input",
        action="append",
        dest="inputs",
        help="Path to an HDF5 file to process. Can be passed multiple times.",
    )
    parser.add_argument(
        "--all-default-inputs",
        action="store_true",
        help="Process the two standard project HDF5 files under data/raw/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for non-in-place outputs. Defaults to the input file directory.",
    )
    parser.add_argument(
        "--overwrite-in-place",
        action="store_true",
        help="Replace the input file atomically after successful processing.",
    )
    parser.add_argument(
        "--group",
        action="append",
        dest="groups",
        help="Only process the named HDF5 group. Can be passed multiple times.",
    )
    parser.add_argument(
        "--build-sigma-unit-hdf5",
        action="store_true",
        help="Write the PPT sigma-unit interpolation HDF5 tables under the chosen output directory.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Worker-process count used by the sigma-unit HDF5 builder. Defaults to os.cpu_count().",
    )
    parser.add_argument(
        "--profile",
        choices=("devauc", "sersic", "all"),
        default="all",
        help="Limit sigma-unit HDF5 generation to one profile, or build both tables.",
    )
    return parser


def main() -> int:
    """Run the interpolation-grid batch processor."""

    parser = build_parser()
    args = parser.parse_args()

    if args.build_sigma_unit_hdf5:
        output_dir = args.output_dir or EXTERNAL_DATA_DIRECTORY
        requested_profiles = None if args.profile == "all" else (args.profile,)
        output_paths = build_default_sigma_unit_hdf5_tables(
            output_directory=output_dir,
            profiles=requested_profiles,
            workers=args.workers,
        )
        for profile_name, output_path in output_paths.items():
            print(f"{profile_name}: wrote {output_path}")
        return 0

    input_paths: list[Path] = []
    if args.all_default_inputs:
        input_paths.extend(RAW_DATA_DIRECTORY / name for name in DEFAULT_INPUT_FILENAMES)
    if args.inputs:
        input_paths.extend(Path(path) for path in args.inputs)
    if not input_paths:
        parser.error("Provide --input or --all-default-inputs.")

    for input_path in input_paths:
        output_dir = args.output_dir or input_path.parent
        output_path = output_dir / input_path.name
        if not args.overwrite_in_place:
            output_path = output_dir / f"{input_path.stem}.updated{input_path.suffix}"

        summary = process_hdf5_file(
            input_path=input_path,
            output_path=output_path,
            overwrite_in_place=args.overwrite_in_place,
            group_names=tuple(args.groups) if args.groups else None,
        )
        print(
            f"{input_path.name}: groups={summary.total_groups} "
            f"m5={summary.updated_m5} dm5={summary.updated_dm5} "
            f"s2={summary.updated_s2} failures={len(summary.failures)}"
        )
        for failure in summary.failures:
            print(f"  FAILURE {failure}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
