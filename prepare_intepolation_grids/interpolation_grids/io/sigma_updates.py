"""CSV-driven updates for observed velocity-dispersion attributes in HDF5.

This module handles a narrower job than the main interpolation-grid pipeline:
it rewrites the observed ``sigma`` and ``sigma_err`` attributes stored on HDF5
galaxy groups from a vetted CSV export.

The implementation intentionally separates three concerns:

1. Parse and validate the CSV rows that describe each observation.
2. Build a complete per-file update plan before touching any HDF5 file.
3. Apply the validated plan atomically when the caller explicitly requests
   in-place replacement.

That separation matters because the user requirement is strict:

- only groups with ``num_sigma != 0`` may be updated
- ``num_sigma == 1`` must match exactly one CSV row
- ``num_sigma == 2`` must match exactly one ``A`` row and one ``B`` row
- any validation problem must stop the write phase before mutating files
"""

from __future__ import annotations

import argparse
import csv
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np


class SigmaUpdateValidationError(ValueError):
    """Raised when CSV content and HDF5 metadata cannot be reconciled safely."""


@dataclass(frozen=True)
class CsvSigmaObservation:
    """One observation row extracted from the CSV export.

    Attributes
    ----------
    base_name:
        Canonical galaxy identifier that must match an HDF5 group name.
    obs_tag:
        Optional observation tag. For two-observation systems this must be
        ``"A"`` or ``"B"`` so the HDF5 arrays can be written in stable order.
    sigma_value:
        Rounded observed velocity dispersion to write into ``group.attrs["sigma"]``.
    sigma_error:
        Rounded statistical uncertainty to write into ``group.attrs["sigma_err"]``.
    """

    base_name: str
    obs_tag: str | None
    sigma_value: int
    sigma_error: int


@dataclass(frozen=True)
class GroupSigmaUpdatePlan:
    """Validated update payload for one HDF5 galaxy group."""

    group_name: str
    num_sigma: int
    matched_row_count: int
    matched_obs_tags: list[str]
    old_sigma: np.ndarray
    old_sigma_err: np.ndarray
    new_sigma: np.ndarray
    new_sigma_err: np.ndarray


@dataclass(frozen=True)
class FileSigmaUpdatePlan:
    """Validated update plan for one HDF5 file."""

    input_path: Path
    group_updates: list[GroupSigmaUpdatePlan]


@dataclass(frozen=True)
class FileSigmaUpdateResult:
    """Outcome for one HDF5 file after preview or write execution."""

    input_path: Path
    group_updates: list[GroupSigmaUpdatePlan]
    was_written: bool
    backup_path: Path | None


def _normalize_obs_tag(raw_value: str | None) -> str | None:
    """Normalize CSV observation tags while preserving the single-observation case."""

    if raw_value is None:
        return None
    normalized = str(raw_value).strip().upper()
    if not normalized:
        return None
    return normalized


def _read_csv_observations(csv_path: Path) -> dict[str, list[CsvSigmaObservation]]:
    """Read and validate the subset of CSV columns required by the updater.

    The project environment does not declare pandas as a dependency, so the
    implementation stays on the standard library ``csv`` module.
    """

    required_columns = {
        "base_name",
        "obs_tag",
        "sigma_primary_kms",
        "sigma_stat_kms",
    }
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise SigmaUpdateValidationError(f"{csv_path} has no header row")

        missing_columns = sorted(required_columns.difference(reader.fieldnames))
        if missing_columns:
            missing_text = ", ".join(missing_columns)
            raise SigmaUpdateValidationError(f"{csv_path} is missing required columns: {missing_text}")

        observations_by_galaxy: dict[str, list[CsvSigmaObservation]] = {}
        for row in reader:
            base_name = str(row["base_name"]).strip()
            if not base_name:
                raise SigmaUpdateValidationError(f"{csv_path} contains an empty base_name row")

            try:
                sigma_value = int(np.rint(float(row["sigma_primary_kms"])))
                sigma_error = int(np.rint(float(row["sigma_stat_kms"])))
            except (TypeError, ValueError) as exc:
                raise SigmaUpdateValidationError(
                    f"{csv_path} contains a non-numeric sigma value for {base_name}"
                ) from exc

            observation = CsvSigmaObservation(
                base_name=base_name,
                obs_tag=_normalize_obs_tag(row.get("obs_tag")),
                sigma_value=sigma_value,
                sigma_error=sigma_error,
            )
            observations_by_galaxy.setdefault(base_name, []).append(observation)

    return observations_by_galaxy


def _read_existing_attr_array(group_handle: h5py.Group, attr_name: str, group_name: str) -> np.ndarray:
    """Read one required integer-vector attribute from an HDF5 group."""

    if attr_name not in group_handle.attrs:
        raise SigmaUpdateValidationError(f"{group_name} is missing required attribute {attr_name!r}")
    return np.asarray(group_handle.attrs[attr_name], dtype=np.int64)


def _build_group_update_plan(
    group_name: str,
    group_handle: h5py.Group,
    observations_by_galaxy: dict[str, list[CsvSigmaObservation]],
) -> GroupSigmaUpdatePlan | None:
    """Validate one HDF5 group and derive the exact arrays that will be written.

    Returning ``None`` means the group is intentionally skipped because
    ``num_sigma == 0``. That skip is a business rule, not an error.
    """

    num_sigma = int(group_handle.attrs.get("num_sigma", 0))
    if num_sigma == 0:
        return None
    if num_sigma not in (1, 2):
        raise SigmaUpdateValidationError(f"{group_name} has unsupported num_sigma={num_sigma}")

    matching_rows = observations_by_galaxy.get(group_name)
    if not matching_rows:
        raise SigmaUpdateValidationError(f"{group_name} is missing from the CSV input")

    if num_sigma == 1:
        if len(matching_rows) != 1:
            raise SigmaUpdateValidationError(
                f"{group_name} expects 1 CSV row because num_sigma=1, found {len(matching_rows)}"
            )
        row = matching_rows[0]
        new_sigma = np.asarray([row.sigma_value], dtype=np.int64)
        new_sigma_err = np.asarray([row.sigma_error], dtype=np.int64)
        matched_obs_tags: list[str] = []
    else:
        if len(matching_rows) != 2:
            raise SigmaUpdateValidationError(
                f"{group_name} expects 2 CSV rows because num_sigma=2, found {len(matching_rows)}"
            )

        rows_by_tag = {row.obs_tag: row for row in matching_rows}
        if set(rows_by_tag) != {"A", "B"} or any(row.obs_tag is None for row in matching_rows):
            raise SigmaUpdateValidationError(
                f"{group_name} requires exactly one A row and one B row when num_sigma=2"
            )

        ordered_rows = [rows_by_tag["A"], rows_by_tag["B"]]
        new_sigma = np.asarray([row.sigma_value for row in ordered_rows], dtype=np.int64)
        new_sigma_err = np.asarray([row.sigma_error for row in ordered_rows], dtype=np.int64)
        matched_obs_tags = ["A", "B"]

    old_sigma = _read_existing_attr_array(group_handle, "sigma", group_name)
    old_sigma_err = _read_existing_attr_array(group_handle, "sigma_err", group_name)
    if old_sigma.shape != new_sigma.shape or old_sigma_err.shape != new_sigma_err.shape:
        raise SigmaUpdateValidationError(
            f"{group_name} shape mismatch: old sigma arrays do not match num_sigma={num_sigma}"
        )

    return GroupSigmaUpdatePlan(
        group_name=group_name,
        num_sigma=num_sigma,
        matched_row_count=len(matching_rows),
        matched_obs_tags=matched_obs_tags,
        old_sigma=old_sigma,
        old_sigma_err=old_sigma_err,
        new_sigma=new_sigma,
        new_sigma_err=new_sigma_err,
    )


def plan_sigma_updates_for_files(csv_path: Path | str, hdf5_paths: Iterable[Path | str]) -> list[FileSigmaUpdatePlan]:
    """Build the complete update plan for every requested HDF5 file.

    This function performs all validation before any file mutation is allowed.
    Callers use it both for dry-run reporting and as the gate before actual
    writes.
    """

    csv_path = Path(csv_path)
    resolved_hdf5_paths = [Path(path) for path in hdf5_paths]
    observations_by_galaxy = _read_csv_observations(csv_path)

    plans: list[FileSigmaUpdatePlan] = []
    for hdf5_path in resolved_hdf5_paths:
        if not hdf5_path.exists():
            raise SigmaUpdateValidationError(f"HDF5 file does not exist: {hdf5_path}")

        group_updates: list[GroupSigmaUpdatePlan] = []
        with h5py.File(hdf5_path, "r") as handle:
            for group_name in handle.keys():
                group_plan = _build_group_update_plan(
                    group_name=group_name,
                    group_handle=handle[group_name],
                    observations_by_galaxy=observations_by_galaxy,
                )
                if group_plan is not None:
                    group_updates.append(group_plan)

        plans.append(FileSigmaUpdatePlan(input_path=hdf5_path, group_updates=group_updates))

    return plans


def _make_backup_path(input_path: Path, timestamp: str) -> Path:
    """Build a timestamped sibling backup path that keeps the original suffix."""

    return input_path.with_name(f"{input_path.stem}.{timestamp}.bak{input_path.suffix}")


def _apply_group_updates(group_handle: h5py.Group, group_update: GroupSigmaUpdatePlan) -> None:
    """Write the new sigma arrays into the existing HDF5 group attributes."""

    group_handle.attrs["sigma"] = np.asarray(group_update.new_sigma, dtype=np.int64)
    group_handle.attrs["sigma_err"] = np.asarray(group_update.new_sigma_err, dtype=np.int64)


def _write_single_hdf5_file(
    file_plan: FileSigmaUpdatePlan,
    create_backup: bool,
    timestamp: str,
) -> FileSigmaUpdateResult:
    """Apply one validated plan using atomic replacement of the target file."""

    input_path = file_plan.input_path
    backup_path = _make_backup_path(input_path, timestamp) if create_backup else None
    if backup_path is not None:
        shutil.copy2(input_path, backup_path)

    with tempfile.NamedTemporaryFile(
        prefix=f"{input_path.stem}.",
        suffix=input_path.suffix,
        dir=input_path.parent,
        delete=False,
    ) as temp_file:
        working_path = Path(temp_file.name)

    shutil.copy2(input_path, working_path)

    try:
        with h5py.File(working_path, "r+") as handle:
            for group_update in file_plan.group_updates:
                if group_update.group_name not in handle:
                    raise SigmaUpdateValidationError(
                        f"{group_update.group_name} disappeared from {input_path} during write"
                    )
                _apply_group_updates(handle[group_update.group_name], group_update)

        working_path.replace(input_path)
    finally:
        working_path.unlink(missing_ok=True)

    return FileSigmaUpdateResult(
        input_path=input_path,
        group_updates=file_plan.group_updates,
        was_written=True,
        backup_path=backup_path,
    )


def update_sigma_attributes_in_hdf5(
    csv_path: Path | str,
    hdf5_paths: Iterable[Path | str],
    overwrite_in_place: bool,
    create_backup: bool,
) -> list[FileSigmaUpdateResult]:
    """Preview or apply sigma-attribute updates for one or more HDF5 files.

    Parameters
    ----------
    csv_path:
        CSV file containing ``base_name``, ``obs_tag``, ``sigma_primary_kms``,
        and ``sigma_stat_kms``.
    hdf5_paths:
        Target HDF5 files whose group attributes will be inspected and,
        optionally, overwritten.
    overwrite_in_place:
        ``False`` means preview-only mode. The function still validates every
        input and returns the exact rows that *would* be written.
    create_backup:
        When ``overwrite_in_place`` is ``True``, create a timestamped backup in
        the same directory before replacing the original file.
    """

    if create_backup and not overwrite_in_place:
        raise SigmaUpdateValidationError("--backup requires --overwrite-in-place")

    file_plans = plan_sigma_updates_for_files(csv_path=csv_path, hdf5_paths=hdf5_paths)
    if not overwrite_in_place:
        return [
            FileSigmaUpdateResult(
                input_path=file_plan.input_path,
                group_updates=file_plan.group_updates,
                was_written=False,
                backup_path=None,
            )
            for file_plan in file_plans
        ]

    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    return [
        _write_single_hdf5_file(file_plan=file_plan, create_backup=create_backup, timestamp=timestamp)
        for file_plan in file_plans
    ]


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for the sigma-update utility."""

    parser = argparse.ArgumentParser(
        description=(
            "Preview or apply velocity-dispersion attribute updates from a CSV "
            "into one or more project HDF5 files."
        ),
    )
    parser.add_argument(
        "--csv",
        required=True,
        type=Path,
        help="CSV file containing base_name, obs_tag, sigma_primary_kms, and sigma_stat_kms.",
    )
    parser.add_argument(
        "--hdf5",
        action="append",
        dest="hdf5_paths",
        required=True,
        type=Path,
        help="Target HDF5 file. Can be passed multiple times.",
    )
    parser.add_argument(
        "--overwrite-in-place",
        action="store_true",
        help="Apply updates atomically to the provided HDF5 files. Default is preview-only.",
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Create a timestamped sibling backup before each in-place overwrite.",
    )
    return parser


def _format_group_preview(group_update: GroupSigmaUpdatePlan) -> str:
    """Format one update row for human inspection during preview and writes."""

    obs_tags_text = ",".join(group_update.matched_obs_tags) if group_update.matched_obs_tags else "NA"
    return (
        f"  {group_update.group_name}: rows={group_update.matched_row_count} "
        f"obs_tags={obs_tags_text} "
        f"sigma {group_update.old_sigma.tolist()} -> {group_update.new_sigma.tolist()} "
        f"sigma_err {group_update.old_sigma_err.tolist()} -> {group_update.new_sigma_err.tolist()}"
    )


def main(argv: list[str] | None = None) -> int:
    """Run the sigma-update CLI and print a full preview of the planned changes."""

    parser = build_parser()
    args = parser.parse_args(argv)

    results = update_sigma_attributes_in_hdf5(
        csv_path=args.csv,
        hdf5_paths=args.hdf5_paths,
        overwrite_in_place=args.overwrite_in_place,
        create_backup=args.backup,
    )

    mode_text = "WRITE" if args.overwrite_in_place else "PREVIEW"
    for result in results:
        print(f"{mode_text} {result.input_path}")
        print(f"  groups_to_update={len(result.group_updates)}")
        for group_update in result.group_updates:
            print(_format_group_preview(group_update))
        if result.backup_path is not None:
            print(f"  backup={result.backup_path}")

    return 0


__all__ = [
    "CsvSigmaObservation",
    "FileSigmaUpdatePlan",
    "FileSigmaUpdateResult",
    "GroupSigmaUpdatePlan",
    "SigmaUpdateValidationError",
    "build_parser",
    "main",
    "plan_sigma_updates_for_files",
    "update_sigma_attributes_in_hdf5",
]
