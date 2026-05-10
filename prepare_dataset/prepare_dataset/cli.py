"""Command-line interface for dataset preparation."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from prepare_dataset.config import (
    BOSS_SUMMARY_FILENAME,
    DEFAULT_INPUT_FILENAMES,
    EXTERNAL_DATA_DIRECTORY,
    RAW_DATA_DIRECTORY,
)
from prepare_dataset.dataset_schema.writer import write_canonical_inference_dataset
from prepare_dataset.io.boss_observations import build_boss_observation_hdf5_files
from prepare_dataset.io.hdf5 import process_hdf5_file
from prepare_dataset.io.lensing_cross_sections import (
    write_fibre_cross_section_hdf5,
    write_power_law_cross_section_hdf5,
)
from prepare_dataset.io.slacs_observations import (
    write_slacs_observation_hdf5,
    write_slacs_population_sigma_unit_hdf5,
)
from prepare_dataset.io.slit_observation_updates import sync_slit_canonical_updates
from prepare_dataset.io.sigma_tables import (
    build_default_sigma_unit_hdf5_tables,
    repack_legacy_sigma_unit_hdf5_tables_into_bundles,
)
from prepare_dataset.physics.lensing_cross_section import (
    DEFAULT_CMASS_THETA_E_AXIS,
    DEFAULT_FIBRE_ARCSEC,
    DEFAULT_FIBRE_BETA_POINTS,
    DEFAULT_FIBRE_RADIAL_POINTS,
    DEFAULT_FIBRE_THETA_E_AXIS,
    DEFAULT_GAMMA_MAX,
    DEFAULT_GAMMA_MIN,
    DEFAULT_GAMMA_POINTS,
    DEFAULT_MUB_MIN,
    DEFAULT_SEEING_ARCSEC,
)


CANONICAL_DEFAULT_THETA_E_MIN = 0.0
CANONICAL_DEFAULT_THETA_E_MAX = 5.0
CANONICAL_DEFAULT_THETA_E_POINTS = 256


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser used by `python -m prepare_dataset`."""

    parser = argparse.ArgumentParser(
        description="Prepare CMASS lensing grids, sigma tables, and canonical inference datasets.",
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
        "--build-slacs-observation-hdf5",
        action="store_true",
        help="Build the Sonnenfeld/SLACS devauc fixed-m5 raw observation HDF5 from SLACS_table.cat.",
    )
    parser.add_argument(
        "--build-slacs-population-sigma-hdf5",
        action="store_true",
        help="Build the Sonnenfeld/SLACS population sigma-unit HDF5 table for fixed m5.",
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
        "--build-canonical-inference-dataset",
        action="store_true",
        help="Build one canonical inference_dataset.hdf5 from prepared observation and cross-section HDF5 inputs.",
    )
    parser.add_argument(
        "--build-power-law-cross-section-hdf5",
        action="store_true",
        help="Build the legacy CMASS power-law cs_grid_power.h5 cross-section table.",
    )
    parser.add_argument(
        "--build-fibre-cross-section-hdf5",
        action="store_true",
        help="Build the Sonnenfeld finite-fibre fibre_crosssect_grid.hdf5 table.",
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
    parser.add_argument(
        "--observation-hdf5",
        type=Path,
        default=None,
        help="Observation HDF5 input used by --build-canonical-inference-dataset.",
    )
    parser.add_argument(
        "--cross-section-hdf5",
        type=Path,
        default=None,
        help="Cross-section HDF5 input used by --build-canonical-inference-dataset.",
    )
    parser.add_argument(
        "--sigma-bundle-hdf5",
        type=Path,
        default=None,
        help="Optional sigma bundle HDF5 input copied into canonical velocity-dispersion blocks when available.",
    )
    parser.add_argument(
        "--population-sigma-hdf5",
        type=Path,
        default=None,
        help="Optional flat population sigma-unit HDF5 copied into canonical velocity-dispersion blocks.",
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=None,
        help="Raw catalog path used by --build-slacs-observation-hdf5.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for --build-canonical-inference-dataset.",
    )
    parser.add_argument(
        "--mass-definition-label",
        default="m5_hinvkpc",
        help="Mass-definition subgroup label to read when building a canonical inference dataset.",
    )
    parser.add_argument(
        "--theta-e-min",
        type=float,
        default=None,
        help="Minimum theta_E value for canonical or generated cross-section axes. Defaults depend on the selected mode.",
    )
    parser.add_argument(
        "--theta-e-max",
        type=float,
        default=None,
        help="Maximum theta_E value for canonical or generated cross-section axes. Defaults depend on the selected mode.",
    )
    parser.add_argument(
        "--theta-e-points",
        type=int,
        default=None,
        help="Number of theta_E samples for canonical or generated cross-section axes. Defaults depend on the selected mode.",
    )
    parser.add_argument(
        "--gamma-min",
        type=float,
        default=DEFAULT_GAMMA_MIN,
        help="Minimum gamma value for generated cross-section tables.",
    )
    parser.add_argument(
        "--gamma-max",
        type=float,
        default=DEFAULT_GAMMA_MAX,
        help="Maximum gamma value for generated cross-section tables.",
    )
    parser.add_argument(
        "--gamma-points",
        type=int,
        default=DEFAULT_GAMMA_POINTS,
        help="Number of gamma samples for generated cross-section tables.",
    )
    parser.add_argument(
        "--beta-points",
        type=int,
        default=DEFAULT_FIBRE_BETA_POINTS,
        help="Number of source-plane samples for --build-fibre-cross-section-hdf5.",
    )
    parser.add_argument(
        "--radial-points",
        type=int,
        default=DEFAULT_FIBRE_RADIAL_POINTS,
        help="Number of fibre-radius samples for --build-fibre-cross-section-hdf5.",
    )
    parser.add_argument(
        "--fibre-arcsec",
        type=float,
        default=DEFAULT_FIBRE_ARCSEC,
        help="Circular fibre radius in arcsec for --build-fibre-cross-section-hdf5.",
    )
    parser.add_argument(
        "--seeing-arcsec",
        type=float,
        default=DEFAULT_SEEING_ARCSEC,
        help="Seeing FWHM in arcsec for --build-fibre-cross-section-hdf5.",
    )
    parser.add_argument(
        "--muB-min",
        type=float,
        default=DEFAULT_MUB_MIN,
        help="Minimum faint-image magnification for --build-fibre-cross-section-hdf5.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the default progress display for --build-fibre-cross-section-hdf5.",
    )
    return parser


def _resolve_axis(
    *,
    minimum: float | None,
    maximum: float | None,
    points: int | None,
    default_axis: np.ndarray,
    axis_name: str,
) -> np.ndarray:
    """Resolve optional CLI axis pieces into a concrete NumPy axis."""

    default_values = np.asarray(default_axis, dtype=float)
    resolved_minimum = float(default_values[0] if minimum is None else minimum)
    resolved_maximum = float(default_values[-1] if maximum is None else maximum)
    resolved_points = int(default_values.size if points is None else points)
    if resolved_points <= 0:
        raise ValueError(f"{axis_name} points must be positive.")
    return np.linspace(resolved_minimum, resolved_maximum, resolved_points, dtype=float)


def main() -> int:
    """Run the interpolation-grid batch processor."""

    parser = build_parser()
    args = parser.parse_args()

    selected_build_modes = (
        int(args.build_sigma_unit_hdf5)
        + int(args.build_boss_observation_hdf5)
        + int(args.build_slacs_observation_hdf5)
        + int(args.build_slacs_population_sigma_hdf5)
        + int(args.repack_legacy_sigma_unit_hdf5)
        + int(args.sync_slit_canonical_sigma)
        + int(args.build_canonical_inference_dataset)
        + int(args.build_power_law_cross_section_hdf5)
        + int(args.build_fibre_cross_section_hdf5)
    )
    if selected_build_modes > 1:
        parser.error("Choose only one special build mode at a time.")

    if args.build_power_law_cross_section_hdf5:
        if args.output is None:
            parser.error("--build-power-law-cross-section-hdf5 requires --output.")
        if args.gamma_points <= 0:
            parser.error("--gamma-points must be positive.")
        try:
            gamma_axis = np.linspace(args.gamma_min, args.gamma_max, args.gamma_points, dtype=float)
            theta_e_axis = _resolve_axis(
                minimum=args.theta_e_min,
                maximum=args.theta_e_max,
                points=args.theta_e_points,
                default_axis=DEFAULT_CMASS_THETA_E_AXIS,
                axis_name="theta_E",
            )
            output_path = write_power_law_cross_section_hdf5(
                args.output,
                gamma_axis=gamma_axis,
                theta_e_axis=theta_e_axis,
                overwrite=args.overwrite_in_place,
            )
        except ValueError as error:
            parser.error(str(error))
        print(f"cross-section: wrote {output_path}")
        return 0

    if args.build_fibre_cross_section_hdf5:
        if args.output is None:
            parser.error("--build-fibre-cross-section-hdf5 requires --output.")
        if args.gamma_points <= 0:
            parser.error("--gamma-points must be positive.")
        try:
            gamma_axis = np.linspace(args.gamma_min, args.gamma_max, args.gamma_points, dtype=float)
            theta_e_axis = _resolve_axis(
                minimum=args.theta_e_min,
                maximum=args.theta_e_max,
                points=args.theta_e_points,
                default_axis=DEFAULT_FIBRE_THETA_E_AXIS,
                axis_name="theta_E",
            )
            output_path = write_fibre_cross_section_hdf5(
                args.output,
                gamma_axis=gamma_axis,
                theta_e_axis=theta_e_axis,
                fibre_arcsec=args.fibre_arcsec,
                seeing_arcsec=args.seeing_arcsec,
                muB_min=args.muB_min,
                beta_points=args.beta_points,
                radial_points=args.radial_points,
                overwrite=args.overwrite_in_place,
                progress=not args.no_progress,
            )
        except ValueError as error:
            parser.error(str(error))
        print(f"fibre-cross-section: wrote {output_path}")
        return 0

    if args.build_canonical_inference_dataset:
        if args.observation_hdf5 is None:
            parser.error("--build-canonical-inference-dataset requires --observation-hdf5.")
        if args.cross_section_hdf5 is None:
            parser.error("--build-canonical-inference-dataset requires --cross-section-hdf5.")
        if args.output is None:
            parser.error("--build-canonical-inference-dataset requires --output.")
        if args.profile == "all":
            parser.error("--build-canonical-inference-dataset requires --profile devauc or --profile sersic.")
        try:
            theta_e_axis = _resolve_axis(
                minimum=args.theta_e_min,
                maximum=args.theta_e_max,
                points=args.theta_e_points,
                default_axis=np.linspace(
                    CANONICAL_DEFAULT_THETA_E_MIN,
                    CANONICAL_DEFAULT_THETA_E_MAX,
                    CANONICAL_DEFAULT_THETA_E_POINTS,
                    dtype=float,
                ),
                axis_name="theta_E",
            )
        except ValueError as error:
            parser.error(str(error))
        output_path = write_canonical_inference_dataset(
            observation_path=args.observation_hdf5,
            cross_section_path=args.cross_section_hdf5,
            output_path=args.output,
            profile_name=args.profile,
            mass_definition_label=args.mass_definition_label,
            unit_convention=args.unit_convention,
            h_ref=args.h_ref,
            theta_e_axis=theta_e_axis,
            sigma_bundle_path=args.sigma_bundle_hdf5,
            population_sigma_path=args.population_sigma_hdf5,
            overwrite=args.overwrite_in_place,
        )
        print(f"canonical: wrote {output_path}")
        return 0

    if args.build_slacs_observation_hdf5:
        if args.output is None:
            parser.error("--build-slacs-observation-hdf5 requires --output.")
        if args.catalog is None:
            parser.error("--build-slacs-observation-hdf5 requires --catalog.")
        output_path = write_slacs_observation_hdf5(
            catalog_path=args.catalog,
            output_path=args.output,
            overwrite=args.overwrite_in_place,
        )
        print(f"slacs-observations: wrote {output_path}")
        return 0

    if args.build_slacs_population_sigma_hdf5:
        if args.output is None:
            parser.error("--build-slacs-population-sigma-hdf5 requires --output.")
        output_path = write_slacs_population_sigma_unit_hdf5(
            output_path=args.output,
            workers=args.workers,
            overwrite=args.overwrite_in_place,
        )
        print(f"slacs-population-sigma: wrote {output_path}")
        return 0

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
