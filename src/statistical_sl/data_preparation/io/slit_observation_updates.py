"""High-level sync workflow for the two canonical slit observation HDF5 files.

This module combines two data-maintenance tasks that must stay in one strict
order:

1. update existing slit sigma attrs from the vetted PPXF CSV export
2. merge SL2S sigma plus explicit aperture metadata into the same canonical
   slit files and rebuild the affected `s2_grid` datasets

Keeping both stages behind one API prevents manual partial updates from leaving
the canonical files in an internally inconsistent state.
"""

from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np

from statistical_sl.data_preparation.io.hdf5 import process_hdf5_file, resolve_group_aperture_policy
from statistical_sl.data_preparation.io.sigma_updates import (
    GroupSigmaUpdatePlan,
    plan_sigma_updates_for_files,
    update_sigma_attributes_in_hdf5,
)


@dataclass(frozen=True)
class Sl2sGroupMergePlan:
    """Validated SL2S merge payload for one target group."""

    group_name: str
    num_sigma: int
    sigma: np.ndarray
    sigma_err: np.ndarray
    aperture_shape: str
    aperture_width_arcsec: float | None
    aperture_height_arcsec: float | None
    aperture_radius_arcsec: float | None
    seeing_fwhm_arcsec: float


@dataclass(frozen=True)
class FileSlitCanonicalUpdatePlan:
    """Preview payload for one canonical slit HDF5 file."""

    input_path: Path
    csv_group_updates: list[GroupSigmaUpdatePlan]
    sl2s_group_updates: list[Sl2sGroupMergePlan]


@dataclass(frozen=True)
class FileSlitCanonicalUpdateResult:
    """Outcome for one canonical slit HDF5 file after preview or write."""

    input_path: Path
    csv_group_updates: list[GroupSigmaUpdatePlan]
    sl2s_group_updates: list[Sl2sGroupMergePlan]
    rebuilt_group_names: list[str]
    was_written: bool


class SlitObservationUpdateValidationError(ValueError):
    """Raised when the slit-canonical sync workflow cannot be executed safely."""


def _normalize_path_list(paths: Iterable[Path | str]) -> list[Path]:
    """Resolve one user-supplied path iterable into stable `Path` objects."""

    return [Path(path).expanduser().resolve() for path in paths]


def _read_required_attr_array(group_handle: h5py.Group, attr_name: str, group_name: str) -> np.ndarray:
    """Read one required SL2S attr array and normalize it to `int64`."""

    if attr_name not in group_handle.attrs:
        raise SlitObservationUpdateValidationError(f"{group_name} is missing required attribute {attr_name!r}")
    return np.asarray(group_handle.attrs[attr_name], dtype=np.int64).reshape(-1)


def _build_sl2s_group_plan(group_name: str, source_group: h5py.Group) -> Sl2sGroupMergePlan:
    """Validate one SL2S source group and extract the merge payload."""

    num_sigma = int(source_group.attrs.get("num_sigma", 0))
    if num_sigma not in (1, 2):
        raise SlitObservationUpdateValidationError(
            f"{group_name} in the SL2S source file has unsupported num_sigma={num_sigma}"
        )

    sigma = _read_required_attr_array(source_group, "sigma", group_name)
    sigma_err = _read_required_attr_array(source_group, "sigma_err", group_name)
    if sigma.shape != sigma_err.shape or sigma.size != num_sigma:
        raise SlitObservationUpdateValidationError(
            f"{group_name} in the SL2S source file has sigma arrays inconsistent with num_sigma={num_sigma}"
        )

    explicit_policy = resolve_group_aperture_policy(source_group)
    if explicit_policy is None:
        raise SlitObservationUpdateValidationError(
            f"{group_name} in the SL2S source file is missing explicit aperture metadata"
        )

    return Sl2sGroupMergePlan(
        group_name=group_name,
        num_sigma=num_sigma,
        sigma=sigma,
        sigma_err=sigma_err,
        aperture_shape=explicit_policy.shape,
        aperture_width_arcsec=explicit_policy.width_arcsec,
        aperture_height_arcsec=explicit_policy.height_arcsec,
        aperture_radius_arcsec=explicit_policy.radius_arcsec,
        seeing_fwhm_arcsec=explicit_policy.seeing_fwhm_arcsec,
    )


def _load_sl2s_group_plans(sl2s_source_path: Path) -> list[Sl2sGroupMergePlan]:
    """Read all SL2S source groups into validated merge plans."""

    if not sl2s_source_path.exists():
        raise SlitObservationUpdateValidationError(f"SL2S source file does not exist: {sl2s_source_path}")

    with h5py.File(sl2s_source_path, "r") as handle:
        return [
            _build_sl2s_group_plan(group_name=group_name, source_group=handle[group_name])
            for group_name in sorted(handle.keys())
        ]


def _validate_sl2s_targets(
    input_path: Path,
    sl2s_group_plans: list[Sl2sGroupMergePlan],
) -> list[Sl2sGroupMergePlan]:
    """Validate that one canonical slit file can safely accept the SL2S groups."""

    validated_plans: list[Sl2sGroupMergePlan] = []
    with h5py.File(input_path, "r") as handle:
        for group_plan in sl2s_group_plans:
            if group_plan.group_name not in handle:
                raise SlitObservationUpdateValidationError(
                    f"{group_plan.group_name} is missing from target canonical file {input_path}"
                )
            target_group = handle[group_plan.group_name]
            existing_num_sigma = int(target_group.attrs.get("num_sigma", 0))
            if existing_num_sigma != 0:
                raise SlitObservationUpdateValidationError(
                    f"{group_plan.group_name} in {input_path.name} already has num_sigma={existing_num_sigma}; "
                    "SL2S merge only supports empty target groups."
                )
            validated_plans.append(group_plan)
    return validated_plans


def plan_slit_canonical_updates(
    csv_path: Path | str,
    slit_hdf5_paths: Iterable[Path | str],
    sl2s_source_path: Path | str,
) -> list[FileSlitCanonicalUpdatePlan]:
    """Build the full preview plan before any canonical slit file is mutated."""

    resolved_paths = _normalize_path_list(slit_hdf5_paths)
    csv_file_plans = {
        file_plan.input_path.resolve(): file_plan
        for file_plan in plan_sigma_updates_for_files(csv_path=csv_path, hdf5_paths=resolved_paths)
    }
    sl2s_group_plans = _load_sl2s_group_plans(Path(sl2s_source_path).expanduser().resolve())

    plans: list[FileSlitCanonicalUpdatePlan] = []
    for input_path in resolved_paths:
        if input_path.resolve() not in csv_file_plans:
            raise SlitObservationUpdateValidationError(f"CSV planning unexpectedly omitted {input_path}")
        plans.append(
            FileSlitCanonicalUpdatePlan(
                input_path=input_path,
                csv_group_updates=csv_file_plans[input_path.resolve()].group_updates,
                sl2s_group_updates=_validate_sl2s_targets(
                    input_path=input_path,
                    sl2s_group_plans=sl2s_group_plans,
                ),
            )
        )
    return plans


def _write_attr_or_delete(group_handle: h5py.Group, attr_name: str, value: object | None) -> None:
    """Write one attr when present or delete stale target attrs when absent."""

    if value is None:
        if attr_name in group_handle.attrs:
            del group_handle.attrs[attr_name]
        return
    group_handle.attrs[attr_name] = value


def _apply_sl2s_group_updates(input_path: Path, group_updates: list[Sl2sGroupMergePlan]) -> None:
    """Apply the validated SL2S attr payload to one canonical slit file atomically."""

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
            for group_update in group_updates:
                if group_update.group_name not in handle:
                    raise SlitObservationUpdateValidationError(
                        f"{group_update.group_name} disappeared from {input_path} during SL2S merge"
                    )
                group_handle = handle[group_update.group_name]
                group_handle.attrs["num_sigma"] = np.int64(group_update.num_sigma)
                group_handle.attrs["sigma"] = np.asarray(group_update.sigma, dtype=np.int64)
                group_handle.attrs["sigma_err"] = np.asarray(group_update.sigma_err, dtype=np.int64)
                group_handle.attrs["aperture_shape"] = group_update.aperture_shape
                _write_attr_or_delete(group_handle, "aperture_width_arcsec", group_update.aperture_width_arcsec)
                _write_attr_or_delete(group_handle, "aperture_height_arcsec", group_update.aperture_height_arcsec)
                _write_attr_or_delete(group_handle, "aperture_radius_arcsec", group_update.aperture_radius_arcsec)
                group_handle.attrs["seeing_fwhm_arcsec"] = float(group_update.seeing_fwhm_arcsec)
        working_path.replace(input_path)
    finally:
        working_path.unlink(missing_ok=True)


def sync_slit_canonical_updates(
    csv_path: Path | str,
    slit_hdf5_paths: Iterable[Path | str],
    sl2s_source_path: Path | str,
    overwrite_in_place: bool,
) -> list[FileSlitCanonicalUpdateResult]:
    """Preview or execute the full slit-canonical sync workflow.

    Preview mode performs all validation and returns the exact per-file plan.
    Write mode executes the workflow in this fixed order:

    1. apply CSV sigma attr updates
    2. merge SL2S sigma/aperture attrs
    3. rebuild the SL2S groups' `s2_grid` datasets in place
    """

    plans = plan_slit_canonical_updates(
        csv_path=csv_path,
        slit_hdf5_paths=slit_hdf5_paths,
        sl2s_source_path=sl2s_source_path,
    )
    if not overwrite_in_place:
        return [
            FileSlitCanonicalUpdateResult(
                input_path=plan.input_path,
                csv_group_updates=plan.csv_group_updates,
                sl2s_group_updates=plan.sl2s_group_updates,
                rebuilt_group_names=[],
                was_written=False,
            )
            for plan in plans
        ]

    resolved_paths = [plan.input_path for plan in plans]
    update_sigma_attributes_in_hdf5(
        csv_path=csv_path,
        hdf5_paths=resolved_paths,
        overwrite_in_place=True,
        create_backup=False,
    )

    results: list[FileSlitCanonicalUpdateResult] = []
    for plan in plans:
        _apply_sl2s_group_updates(
            input_path=plan.input_path,
            group_updates=plan.sl2s_group_updates,
        )
        rebuilt_group_names = [group_update.group_name for group_update in plan.sl2s_group_updates]
        summary = process_hdf5_file(
            input_path=plan.input_path,
            output_path=plan.input_path,
            overwrite_in_place=True,
            group_names=tuple(rebuilt_group_names),
        )
        if summary.failures:
            failure_text = "; ".join(summary.failures)
            raise SlitObservationUpdateValidationError(
                f"Failed to rebuild SL2S groups in {plan.input_path.name}: {failure_text}"
            )
        results.append(
            FileSlitCanonicalUpdateResult(
                input_path=plan.input_path,
                csv_group_updates=plan.csv_group_updates,
                sl2s_group_updates=plan.sl2s_group_updates,
                rebuilt_group_names=rebuilt_group_names,
                was_written=True,
            )
        )
    return results


__all__ = [
    "FileSlitCanonicalUpdatePlan",
    "FileSlitCanonicalUpdateResult",
    "Sl2sGroupMergePlan",
    "SlitObservationUpdateValidationError",
    "plan_slit_canonical_updates",
    "sync_slit_canonical_updates",
]
