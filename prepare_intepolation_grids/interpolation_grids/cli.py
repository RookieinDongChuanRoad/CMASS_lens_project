"""Command-line interface for interpolation-grid generation."""

from __future__ import annotations

import argparse
from pathlib import Path

from interpolation_grids.config import (
    BOSS_SUMMARY_FILENAME,
    DEFAULT_INPUT_FILENAMES,
    EXTERNAL_DATA_DIRECTORY,
    RAW_DATA_DIRECTORY,
)
from interpolation_grids.io.boss_observations import build_boss_observation_hdf5_files
from interpolation_grids.io.hdf5 import process_hdf5_file
from interpolation_grids.io.slit_observation_updates import sync_slit_canonical_updates
from interpolation_grids.io.sigma_tables import (
    build_default_sigma_unit_hdf5_tables,
    repack_legacy_sigma_unit_hdf5_tables_into_bundles,
)


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
        "--build-boss-observation-hdf5",
        action="store_true",
        help="Build the two BOSS raw observation HDF5 files from the summary table.",
    )
    parser.add_argument(
        "--repack-legacy-sigma-unit-hdf5",
        action="store_true",
        help="Copy existing legacy slit sigma-unit HDF5 files into the new per-profile bundle schema without recomputing.",
    )
    parser.add_argument(
        "--sync-slit-canonical-sigma",
        action="store_true",
        help="Preview or apply the combined CSV sigma update plus SL2S merge workflow for the two slit canonical HDF5 files.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="PPXF CSV file used by --sync-slit-canonical-sigma.",
    )
    parser.add_argument(
        "--sl2s-source",
        type=Path,
        default=RAW_DATA_DIRECTORY / "observations_deV_with_SL2S_mass_grids.hdf5",
        help="SL2S source HDF5 used by --sync-slit-canonical-sigma.",
    )
    parser.add_argument(
        "--summary-table",
        type=Path,
        default=None,
        help=f"Override the BOSS summary-table path. Defaults to data/raw/{BOSS_SUMMARY_FILENAME}.",
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
    parser.add_argument(
        "--observation-flavor",
        choices=("slit", "boss", "all"),
        default="all",
        help="Limit sigma-bundle generation to one observation flavor, or build both.",
    )
    parser.add_argument(
        "--sigma-definition",
        choices=("observed_aperture", "within_re", "all"),
        default="all",
        help="Limit sigma-bundle generation to observed-aperture leaves, within-Re leaves, or build both.",
    )
    parser.add_argument(
        "--legacy-sigma-input-dir",
        type=Path,
        default=None,
        help="Directory containing the old flat sigma-unit HDF5 files when using --repack-legacy-sigma-unit-hdf5.",
    )
    parser.add_argument(
        "--unit-convention",
        choices=("h_units_v1", "legacy_fixed_kpc"),
        default="h_units_v1",
        help="Unit convention for rebuilt mass grids and sigma-unit tables.",
    )
    parser.add_argument(
        "--h-ref",
        type=float,
        default=0.7,
        help="Reference h used when --unit-convention h_units_v1 is selected.",
    )
    return parser


def main() -> int:
    """Run the interpolation-grid batch processor."""

    parser = build_parser()
    args = parser.parse_args()

    selected_build_modes = (
        int(args.build_sigma_unit_hdf5)
        + int(args.build_boss_observation_hdf5)
        + int(args.repack_legacy_sigma_unit_hdf5)
        + int(args.sync_slit_canonical_sigma)
    )
    if selected_build_modes > 1:
        parser.error("Choose only one special build mode at a time.")

    if args.build_boss_observation_hdf5:
        output_dir = args.output_dir or RAW_DATA_DIRECTORY
        summary_path = args.summary_table or RAW_DATA_DIRECTORY / BOSS_SUMMARY_FILENAME
        output_paths = build_boss_observation_hdf5_files(
            summary_path=summary_path,
            output_directory=output_dir,
        )
        for profile_name, output_path in output_paths.items():
            print(f"{profile_name}: wrote {output_path}")
        return 0

    if args.repack_legacy_sigma_unit_hdf5:
        if args.observation_flavor != "all":
            parser.error("--repack-legacy-sigma-unit-hdf5 only supports the existing slit legacy files, so do not pass --observation-flavor.")
        if args.sigma_definition != "all":
            parser.error("--repack-legacy-sigma-unit-hdf5 does not support --sigma-definition.")
        output_dir = args.output_dir or EXTERNAL_DATA_DIRECTORY
        # In worktree-based runs the canonical legacy files still live under the
        # main repository's external-data directory. Defaulting the repack input
        # to `output_dir` therefore matches the user's intent better than
        # blindly using this worktree's `EXTERNAL_DATA_DIRECTORY`.
        input_dir = args.legacy_sigma_input_dir or output_dir
        requested_profiles = None if args.profile == "all" else (args.profile,)
        output_paths = repack_legacy_sigma_unit_hdf5_tables_into_bundles(
            input_directory=input_dir,
            output_directory=output_dir,
            profiles=requested_profiles,
        )
        for profile_name, output_path in output_paths.items():
            print(f"{profile_name}: wrote {output_path}")
        return 0

    if args.build_sigma_unit_hdf5:
        if args.sigma_definition == "within_re" and args.observation_flavor != "all":
            parser.error("--sigma-definition within_re does not accept --observation-flavor because within_re is not an observation flavor.")
        output_dir = args.output_dir or EXTERNAL_DATA_DIRECTORY
        requested_profiles = None if args.profile == "all" else (args.profile,)
        requested_observation_flavors = None if args.observation_flavor == "all" else (args.observation_flavor,)
        requested_sigma_definitions = None if args.sigma_definition == "all" else (args.sigma_definition,)
        output_paths = build_default_sigma_unit_hdf5_tables(
            output_directory=output_dir,
            profiles=requested_profiles,
            observation_flavors=requested_observation_flavors,
            sigma_definitions=requested_sigma_definitions,
            workers=args.workers,
            unit_convention=args.unit_convention,
            h_ref=args.h_ref,
        )
        for profile_name, output_path in output_paths.items():
            print(f"{profile_name}: wrote {output_path}")
        return 0

    if args.sync_slit_canonical_sigma:
        if args.csv is None:
            parser.error("--sync-slit-canonical-sigma requires --csv.")
        slit_paths = [RAW_DATA_DIRECTORY / name for name in DEFAULT_INPUT_FILENAMES]
        results = sync_slit_canonical_updates(
            csv_path=args.csv,
            slit_hdf5_paths=slit_paths,
            sl2s_source_path=args.sl2s_source,
            overwrite_in_place=args.overwrite_in_place,
        )
        mode_text = "WRITE" if args.overwrite_in_place else "PREVIEW"
        for result in results:
            print(f"{mode_text} {result.input_path}")
            print(f"  csv_groups={len(result.csv_group_updates)}")
            for group_update in result.csv_group_updates:
                print(
                    f"    CSV {group_update.group_name}: "
                    f"{group_update.old_sigma.tolist()}->{group_update.new_sigma.tolist()} "
                    f"{group_update.old_sigma_err.tolist()}->{group_update.new_sigma_err.tolist()}"
                )
            print(f"  sl2s_groups={len(result.sl2s_group_updates)}")
            for group_update in result.sl2s_group_updates:
                print(
                    f"    SL2S {group_update.group_name}: num_sigma={group_update.num_sigma} "
                    f"shape={group_update.aperture_shape} seeing={group_update.seeing_fwhm_arcsec}"
                )
            if result.rebuilt_group_names:
                print(f"  rebuilt={','.join(result.rebuilt_group_names)}")
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
            unit_convention=args.unit_convention,
            h_ref=args.h_ref,
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
