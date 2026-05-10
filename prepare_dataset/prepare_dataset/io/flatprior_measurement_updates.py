"""Flat-prior measurement-attribute updates for project HDF5 files.

This module adds a second reusable updater alongside the existing sigma updater.
Its job is narrower than the main interpolation-grid pipeline:

- inspect only already-generated ``*_mass_grids`` HDF5 files
- derive per-group flat-prior measurement summaries from nested mass grids
- preview the exact attrs that would be written
- optionally apply those attrs via atomic in-place replacement

The user requirement is deliberately strict, so the implementation follows the
same three-stage safety model as ``sigma_updates.py``:

1. Validate every requested HDF5 file and build a full update plan in memory.
2. Report the planned attr values without mutating files when preview mode is
   selected.
3. Only after successful validation, write the new attrs into temporary file
   copies and atomically replace the originals.
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np

from prepare_dataset.config import (
    GAMMA_DATASET_NAME,
    H_UNITS_V1,
    LEGACY_FIXED_KPC,
    MASS_DEFINITIONS_GROUP_NAME,
    MASS_DERIVATIVE_DATASET_NAME,
    MASS_GRID_DATASET_NAME,
    S2_DATASET_NAME,
    mass_definition_label_for_convention,
)


TARGET_ATTR_NAMES = (
    "gamma_lower",
    "gamma_mid",
    "gamma_upper",
    "m5_lower",
    "m5_mid",
    "m5_upper",
    "m10_lower",
    "m10_mid",
    "m10_upper",
)
SUPPORTED_UNIT_CONVENTIONS = frozenset({LEGACY_FIXED_KPC, H_UNITS_V1})


def target_attr_names_for_convention(unit_convention: str) -> tuple[str, ...]:
    """
    Return the measurement attrs written for one unit convention.

    Legacy files keep the historical public attrs (`m5_*`, `m10_*`). H-units
    files must expose active public attrs (`m5_hinvkpc_*`, `m10_hinvkpc_*`) so
    downstream PPC code can read observed overlays without silently migrating
    legacy fixed-kpc summaries.
    """

    m5_prefix = mass_definition_label_for_convention(5.0, unit_convention)
    m10_prefix = mass_definition_label_for_convention(10.0, unit_convention)
    return (
        "gamma_lower",
        "gamma_mid",
        "gamma_upper",
        f"{m5_prefix}_lower",
        f"{m5_prefix}_mid",
        f"{m5_prefix}_upper",
        f"{m10_prefix}_lower",
        f"{m10_prefix}_mid",
        f"{m10_prefix}_upper",
    )


class FlatPriorMeasurementUpdateValidationError(ValueError):
    """Raised when an HDF5 file cannot safely support the requested update."""


@dataclass(frozen=True)
class GroupFlatPriorMeasurementUpdatePlan:
    """Validated measurement payload for one HDF5 group.

    Attributes
    ----------
    group_name:
        Name of the HDF5 group to update.
    num_sigma:
        Number of observed velocity-dispersion constraints used in the group.
    old_attrs:
        Existing values of the target attrs, if present. Keeping them in the
        preview plan makes dry-run output easier to audit.
    new_attrs:
        The exact root-level attrs that should be written for this group.
    """

    group_name: str
    num_sigma: int
    old_attrs: dict[str, float | None]
    new_attrs: dict[str, float]


@dataclass(frozen=True)
class FileFlatPriorMeasurementUpdatePlan:
    """Validated measurement-update plan for one HDF5 file."""

    input_path: Path
    group_updates: list[GroupFlatPriorMeasurementUpdatePlan]


@dataclass(frozen=True)
class FileFlatPriorMeasurementUpdateResult:
    """Outcome for one HDF5 file after preview or write execution."""

    input_path: Path
    group_updates: list[GroupFlatPriorMeasurementUpdatePlan]
    was_written: bool
    backup_path: Path | None


def _read_required_dataset(group_handle: h5py.Group, dataset_name: str, group_name: str) -> np.ndarray:
    """Read one required dataset and normalize it to a float array."""

    if dataset_name not in group_handle:
        raise FlatPriorMeasurementUpdateValidationError(
            f"{group_name} is missing required dataset {dataset_name!r}"
        )
    return np.asarray(group_handle[dataset_name], dtype=float)


def _read_required_attr_array(group_handle: h5py.Group, attr_name: str, group_name: str) -> np.ndarray:
    """Read one required root attribute as a float array."""

    if attr_name not in group_handle.attrs:
        raise FlatPriorMeasurementUpdateValidationError(
            f"{group_name} is missing required attribute {attr_name!r}"
        )
    return np.asarray(group_handle.attrs[attr_name], dtype=float)


def _validate_gamma_grid(gamma_grid: np.ndarray, group_name: str) -> np.ndarray:
    """Validate the inference axis because all later interpolation depends on it.

    Why this check matters:
    - Quantile extraction assumes a one-dimensional gamma axis.
    - Both quantile interpolation and mass-grid projection require the axis to
      be strictly increasing.
    """

    gamma_grid = np.asarray(gamma_grid, dtype=float)
    if gamma_grid.ndim != 1 or gamma_grid.size == 0:
        raise FlatPriorMeasurementUpdateValidationError(
            f"{group_name} gamma_grid must be a non-empty 1D dataset"
        )
    if gamma_grid.size > 1 and np.any(np.diff(gamma_grid) <= 0.0):
        raise FlatPriorMeasurementUpdateValidationError(
            f"{group_name} gamma_grid must be strictly increasing"
        )
    return gamma_grid


def _validate_grid_shape(values: np.ndarray, gamma_grid: np.ndarray, label: str, group_name: str) -> np.ndarray:
    """Ensure one auxiliary grid aligns exactly with ``gamma_grid``."""

    values = np.asarray(values, dtype=float)
    if values.shape != gamma_grid.shape:
        raise FlatPriorMeasurementUpdateValidationError(
            f"{group_name} dataset {label!r} shape {values.shape} does not match gamma_grid {gamma_grid.shape}"
        )
    return values


def _validate_positive_grid(values: np.ndarray, label: str, group_name: str) -> np.ndarray:
    """Reject non-positive grids that would break logarithms or square roots."""

    values = np.asarray(values, dtype=float)
    if np.any(values <= 0.0):
        raise FlatPriorMeasurementUpdateValidationError(
            f"{group_name} dataset {label!r} must be strictly positive"
        )
    return values


def _validate_sigma_arrays(
    sigma_values: np.ndarray,
    sigma_errors: np.ndarray,
    num_sigma: int,
    group_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate the observed sigma arrays against ``num_sigma``.

    The updater supports the same business rules as the rest of the project:
    one likelihood term when ``num_sigma == 1`` and a product of two Gaussian
    terms when ``num_sigma == 2``.
    """

    sigma_values = np.asarray(sigma_values, dtype=float).reshape(-1)
    sigma_errors = np.asarray(sigma_errors, dtype=float).reshape(-1)
    if sigma_values.shape != sigma_errors.shape:
        raise FlatPriorMeasurementUpdateValidationError(
            f"{group_name} sigma and sigma_err must have the same shape"
        )
    if sigma_values.size != num_sigma:
        raise FlatPriorMeasurementUpdateValidationError(
            f"{group_name} sigma arrays must contain exactly {num_sigma} values"
        )
    if np.any(sigma_errors <= 0.0):
        raise FlatPriorMeasurementUpdateValidationError(
            f"{group_name} sigma_err must be strictly positive"
        )
    return sigma_values, sigma_errors


def _decode_hdf5_string(raw_value) -> str:
    """Normalize scalar HDF5 string attrs returned as bytes or native strings."""

    if isinstance(raw_value, bytes):
        return raw_value.decode("utf-8")
    if isinstance(raw_value, np.ndarray) and raw_value.shape == ():
        return _decode_hdf5_string(raw_value.item())
    return str(raw_value)


def _resolve_group_unit_convention(group_handle: h5py.Group, group_name: str) -> str:
    """
    Resolve the active unit convention for one observation group.

    The h-units rebuild writes convention metadata both at file and group
    level. The updater accepts either location, but if both are present they
    must agree. This matters because the selected convention controls both the
    mass-definition leaves used for the calculation and the root attrs written
    for PPC overlays.
    """

    candidates: list[tuple[str, str]] = []
    if "unit_convention" in group_handle.file.attrs:
        candidates.append(("file attrs", _decode_hdf5_string(group_handle.file.attrs["unit_convention"])))
    if "unit_convention" in group_handle.attrs:
        candidates.append((f"group attrs {group_name}", _decode_hdf5_string(group_handle.attrs["unit_convention"])))
    if not candidates:
        return LEGACY_FIXED_KPC

    normalized_candidates = [(source, value.strip()) for source, value in candidates]
    reference_source, reference_value = normalized_candidates[0]
    if reference_value not in SUPPORTED_UNIT_CONVENTIONS:
        raise FlatPriorMeasurementUpdateValidationError(
            f"{group_name} has unsupported unit_convention={reference_value!r} in {reference_source}"
        )
    for source, value in normalized_candidates[1:]:
        if value not in SUPPORTED_UNIT_CONVENTIONS:
            raise FlatPriorMeasurementUpdateValidationError(
                f"{group_name} has unsupported unit_convention={value!r} in {source}"
            )
        if value != reference_value:
            raise FlatPriorMeasurementUpdateValidationError(
                f"{group_name} has inconsistent unit_convention metadata: "
                f"{reference_source}={reference_value!r}, {source}={value!r}"
            )
    return reference_value


def _extract_nested_mass_inputs(
    group_handle: h5py.Group,
    gamma_grid: np.ndarray,
    group_name: str,
    unit_convention: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load the nested mass-definition datasets required by the updater."""

    if MASS_DEFINITIONS_GROUP_NAME not in group_handle:
        raise FlatPriorMeasurementUpdateValidationError(
            f"{group_name} is missing required group {MASS_DEFINITIONS_GROUP_NAME!r}"
        )

    mass_definitions_handle = group_handle[MASS_DEFINITIONS_GROUP_NAME]
    m5_label = mass_definition_label_for_convention(5.0, unit_convention)
    m10_label = mass_definition_label_for_convention(10.0, unit_convention)
    if m5_label not in mass_definitions_handle:
        raise FlatPriorMeasurementUpdateValidationError(
            f"{group_name} is missing mass_definitions/{m5_label!r}"
        )
    if m10_label not in mass_definitions_handle:
        raise FlatPriorMeasurementUpdateValidationError(
            f"{group_name} is missing mass_definitions/{m10_label!r}"
        )

    m5_handle = mass_definitions_handle[m5_label]
    m10_handle = mass_definitions_handle[m10_label]

    m5_grid = _validate_grid_shape(
        _read_required_dataset(m5_handle, MASS_GRID_DATASET_NAME, group_name),
        gamma_grid=gamma_grid,
        label=f"mass_definitions/{m5_label}/mass_grid",
        group_name=group_name,
    )
    m10_grid = _validate_grid_shape(
        _read_required_dataset(m10_handle, MASS_GRID_DATASET_NAME, group_name),
        gamma_grid=gamma_grid,
        label=f"mass_definitions/{m10_label}/mass_grid",
        group_name=group_name,
    )
    derivative_grid = _validate_positive_grid(
        _validate_grid_shape(
            _read_required_dataset(m5_handle, MASS_DERIVATIVE_DATASET_NAME, group_name),
            gamma_grid=gamma_grid,
            label=f"mass_definitions/{m5_label}/dmass_dthetaein_grid",
            group_name=group_name,
        ),
        label=f"mass_definitions/{m5_label}/dmass_dthetaein_grid",
        group_name=group_name,
    )
    s2_grid = _validate_positive_grid(
        _validate_grid_shape(
            _read_required_dataset(m5_handle, S2_DATASET_NAME, group_name),
            gamma_grid=gamma_grid,
            label=f"mass_definitions/{m5_label}/s2_grid",
            group_name=group_name,
        ),
        label=f"mass_definitions/{m5_label}/s2_grid",
        group_name=group_name,
    )
    return m5_grid, m10_grid, derivative_grid, s2_grid


def _compute_normalized_log_posterior(
    m5_grid: np.ndarray,
    s2_grid: np.ndarray,
    derivative_grid: np.ndarray,
    sigma_values: np.ndarray,
    sigma_errors: np.ndarray,
) -> np.ndarray:
    """Return a numerically stable log-posterior grid.

    The flat-prior measurement construction is intentionally faithful to the
    documented external algorithm:

    ``logp = sum_j[-0.5 * ((sigma_model - sigma_j)/sigma_err_j)^2 - log(sigma_err_j)] - log(jacobian)``

    where ``sigma_model = sqrt(10**m5 * s2_grid)``.
    """

    sigma_model_grid = np.sqrt(np.maximum((10.0**m5_grid) * s2_grid, 1.0e-300))
    logp_grid = np.zeros_like(m5_grid, dtype=float)
    for sigma_value, sigma_error in zip(sigma_values, sigma_errors):
        logp_grid += -0.5 * ((sigma_model_grid - sigma_value) / sigma_error) ** 2 - np.log(sigma_error)
    logp_grid += -np.log(derivative_grid)
    logp_grid -= np.max(logp_grid)
    return logp_grid


def _cdf_quantile(axis: np.ndarray, cdf: np.ndarray, target: float) -> float:
    """Interpolate one quantile from a discrete cumulative distribution.

    Edge handling follows the user-approved rule:
    - clamp to the nearest edge when the target quantile falls outside the
      tabulated CDF support
    - otherwise linearly interpolate between neighboring CDF knots

    Repeated CDF values can appear when one grid point carries effectively zero
    posterior mass. In that degenerate case we return the upper axis value so
    the result stays deterministic without inventing support where none exists.
    """

    axis = np.asarray(axis, dtype=float)
    cdf = np.asarray(cdf, dtype=float)
    if target <= cdf[0]:
        return float(axis[0])
    if target >= cdf[-1]:
        return float(axis[-1])

    upper_index = int(np.searchsorted(cdf, target, side="left"))
    lower_index = upper_index - 1
    lower_cdf = float(cdf[lower_index])
    upper_cdf = float(cdf[upper_index])
    lower_axis = float(axis[lower_index])
    upper_axis = float(axis[upper_index])
    if upper_cdf <= lower_cdf:
        return upper_axis
    interpolation_weight = (target - lower_cdf) / (upper_cdf - lower_cdf)
    return lower_axis + interpolation_weight * (upper_axis - lower_axis)


def _project_mass_summary(
    gamma_grid: np.ndarray,
    mass_grid: np.ndarray,
    gamma_q16: float,
    gamma_q50: float,
    gamma_q84: float,
) -> tuple[float, float, float]:
    """Project gamma quantiles onto one mass definition.

    The mass grids can be non-monotonic functions of gamma. That means the
    projected 16% and 84% values are not guaranteed to arrive in numerical
    order, and the projected 50% value is not guaranteed to lie between them.
    To keep the stored attrs interpretable as positive error-bar amplitudes, we
    sort the projected edge values and take absolute distances from the median.
    """

    q16_value = float(np.interp(gamma_q16, gamma_grid, mass_grid))
    q50_value = float(np.interp(gamma_q50, gamma_grid, mass_grid))
    q84_value = float(np.interp(gamma_q84, gamma_grid, mass_grid))
    lower_value, upper_value = sorted((q16_value, q84_value))
    return (
        q50_value,
        abs(q50_value - lower_value),
        abs(upper_value - q50_value),
    )


def _build_group_update_plan(
    group_name: str,
    group_handle: h5py.Group,
) -> GroupFlatPriorMeasurementUpdatePlan | None:
    """Validate one HDF5 group and derive the exact attrs to write.

    Returning ``None`` means the group is intentionally skipped because
    ``num_sigma == 0``. That skip is a business rule, not an error.
    """

    num_sigma = int(group_handle.attrs.get("num_sigma", 0))
    if num_sigma == 0:
        return None
    if num_sigma not in (1, 2):
        raise FlatPriorMeasurementUpdateValidationError(
            f"{group_name} has unsupported num_sigma={num_sigma}"
        )
    unit_convention = _resolve_group_unit_convention(group_handle=group_handle, group_name=group_name)
    m5_attr_prefix = mass_definition_label_for_convention(5.0, unit_convention)
    m10_attr_prefix = mass_definition_label_for_convention(10.0, unit_convention)
    target_attr_names = target_attr_names_for_convention(unit_convention)

    gamma_grid = _validate_gamma_grid(
        _read_required_dataset(group_handle, GAMMA_DATASET_NAME, group_name),
        group_name=group_name,
    )
    sigma_values, sigma_errors = _validate_sigma_arrays(
        _read_required_attr_array(group_handle, "sigma", group_name),
        _read_required_attr_array(group_handle, "sigma_err", group_name),
        num_sigma=num_sigma,
        group_name=group_name,
    )
    m5_grid, m10_grid, derivative_grid, s2_grid = _extract_nested_mass_inputs(
        group_handle=group_handle,
        gamma_grid=gamma_grid,
        group_name=group_name,
        unit_convention=unit_convention,
    )

    posterior_grid = np.exp(
        _compute_normalized_log_posterior(
            m5_grid=m5_grid,
            s2_grid=s2_grid,
            derivative_grid=derivative_grid,
            sigma_values=sigma_values,
            sigma_errors=sigma_errors,
        )
    )
    total_probability = float(posterior_grid.sum())
    if total_probability <= 0.0 or not np.isfinite(total_probability):
        raise FlatPriorMeasurementUpdateValidationError(
            f"{group_name} posterior normalization is not finite"
        )
    posterior_grid /= total_probability
    cdf_grid = posterior_grid.cumsum()

    gamma_q16 = _cdf_quantile(gamma_grid, cdf_grid, 0.16)
    gamma_q50 = _cdf_quantile(gamma_grid, cdf_grid, 0.50)
    gamma_q84 = _cdf_quantile(gamma_grid, cdf_grid, 0.84)
    if gamma_q50 < gamma_q16 or gamma_q84 < gamma_q50:
        raise FlatPriorMeasurementUpdateValidationError(
            f"{group_name} produced non-monotonic gamma quantiles"
        )

    m5_mid, m5_lower, m5_upper = _project_mass_summary(
        gamma_grid=gamma_grid,
        mass_grid=m5_grid,
        gamma_q16=gamma_q16,
        gamma_q50=gamma_q50,
        gamma_q84=gamma_q84,
    )
    m10_mid, m10_lower, m10_upper = _project_mass_summary(
        gamma_grid=gamma_grid,
        mass_grid=m10_grid,
        gamma_q16=gamma_q16,
        gamma_q50=gamma_q50,
        gamma_q84=gamma_q84,
    )

    new_attrs = {
        "gamma_lower": float(gamma_q50 - gamma_q16),
        "gamma_mid": float(gamma_q50),
        "gamma_upper": float(gamma_q84 - gamma_q50),
        f"{m5_attr_prefix}_lower": float(m5_lower),
        f"{m5_attr_prefix}_mid": float(m5_mid),
        f"{m5_attr_prefix}_upper": float(m5_upper),
        f"{m10_attr_prefix}_lower": float(m10_lower),
        f"{m10_attr_prefix}_mid": float(m10_mid),
        f"{m10_attr_prefix}_upper": float(m10_upper),
    }
    old_attrs = {
        attr_name: (float(group_handle.attrs[attr_name]) if attr_name in group_handle.attrs else None)
        for attr_name in target_attr_names
    }

    return GroupFlatPriorMeasurementUpdatePlan(
        group_name=group_name,
        num_sigma=num_sigma,
        old_attrs=old_attrs,
        new_attrs=new_attrs,
    )


def plan_flatprior_measurement_updates_for_files(
    hdf5_paths: Iterable[Path | str],
) -> list[FileFlatPriorMeasurementUpdatePlan]:
    """Build validated update plans for every requested HDF5 file."""

    resolved_hdf5_paths = [Path(path) for path in hdf5_paths]
    plans: list[FileFlatPriorMeasurementUpdatePlan] = []
    for hdf5_path in resolved_hdf5_paths:
        if not hdf5_path.exists():
            raise FlatPriorMeasurementUpdateValidationError(f"HDF5 file does not exist: {hdf5_path}")

        group_updates: list[GroupFlatPriorMeasurementUpdatePlan] = []
        with h5py.File(hdf5_path, "r") as handle:
            for group_name in handle.keys():
                group_plan = _build_group_update_plan(group_name=group_name, group_handle=handle[group_name])
                if group_plan is not None:
                    group_updates.append(group_plan)

        plans.append(FileFlatPriorMeasurementUpdatePlan(input_path=hdf5_path, group_updates=group_updates))
    return plans


def _make_backup_path(input_path: Path, timestamp: str) -> Path:
    """Build a timestamped sibling backup path that keeps the original suffix."""

    return input_path.with_name(f"{input_path.stem}.{timestamp}.bak{input_path.suffix}")


def _apply_group_updates(
    group_handle: h5py.Group,
    group_update: GroupFlatPriorMeasurementUpdatePlan,
) -> None:
    """Write the planned measurement attrs into the existing HDF5 group."""

    for attr_name, attr_value in group_update.new_attrs.items():
        group_handle.attrs[attr_name] = float(attr_value)


def _write_single_hdf5_file(
    file_plan: FileFlatPriorMeasurementUpdatePlan,
    create_backup: bool,
    timestamp: str,
) -> FileFlatPriorMeasurementUpdateResult:
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
                    raise FlatPriorMeasurementUpdateValidationError(
                        f"{group_update.group_name} disappeared from {input_path} during write"
                    )
                _apply_group_updates(handle[group_update.group_name], group_update)
        working_path.replace(input_path)
    finally:
        working_path.unlink(missing_ok=True)

    return FileFlatPriorMeasurementUpdateResult(
        input_path=input_path,
        group_updates=file_plan.group_updates,
        was_written=True,
        backup_path=backup_path,
    )


def update_flatprior_measurement_attrs_in_hdf5(
    hdf5_paths: Iterable[Path | str],
    overwrite_in_place: bool,
    create_backup: bool,
) -> list[FileFlatPriorMeasurementUpdateResult]:
    """Preview or apply flat-prior measurement attrs for one or more HDF5 files."""

    if create_backup and not overwrite_in_place:
        raise FlatPriorMeasurementUpdateValidationError("--backup requires --overwrite-in-place")

    file_plans = plan_flatprior_measurement_updates_for_files(hdf5_paths=hdf5_paths)
    if not overwrite_in_place:
        return [
            FileFlatPriorMeasurementUpdateResult(
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
    """Create the CLI parser for the flat-prior measurement updater."""

    parser = argparse.ArgumentParser(
        description=(
            "Preview or apply flat-prior mass/gamma measurement attrs into "
            "one or more project HDF5 files."
        ),
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


def _format_group_preview(group_update: GroupFlatPriorMeasurementUpdatePlan) -> str:
    """Format one human-readable preview line for dry-run and write output."""

    mass_keys = tuple(
        attr_name.removesuffix("_mid")
        for attr_name in group_update.new_attrs
        if attr_name.endswith("_mid") and attr_name != "gamma_mid"
    )
    if len(mass_keys) != 2:
        raise FlatPriorMeasurementUpdateValidationError(
            f"{group_update.group_name} has unexpected mass summary attrs: {sorted(group_update.new_attrs)}"
        )

    return (
        f"  {group_update.group_name}: "
        f"gamma={group_update.new_attrs['gamma_mid']:.6f}"
        f" -{group_update.new_attrs['gamma_lower']:.6f}"
        f" +{group_update.new_attrs['gamma_upper']:.6f} "
        f"{mass_keys[0]}={group_update.new_attrs[f'{mass_keys[0]}_mid']:.6f}"
        f" -{group_update.new_attrs[f'{mass_keys[0]}_lower']:.6f}"
        f" +{group_update.new_attrs[f'{mass_keys[0]}_upper']:.6f} "
        f"{mass_keys[1]}={group_update.new_attrs[f'{mass_keys[1]}_mid']:.6f}"
        f" -{group_update.new_attrs[f'{mass_keys[1]}_lower']:.6f}"
        f" +{group_update.new_attrs[f'{mass_keys[1]}_upper']:.6f}"
    )


def main(argv: list[str] | None = None) -> int:
    """Run the measurement-update CLI and print a full preview of the plan."""

    parser = build_parser()
    args = parser.parse_args(argv)

    results = update_flatprior_measurement_attrs_in_hdf5(
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
    "FileFlatPriorMeasurementUpdatePlan",
    "FileFlatPriorMeasurementUpdateResult",
    "FlatPriorMeasurementUpdateValidationError",
    "GroupFlatPriorMeasurementUpdatePlan",
    "TARGET_ATTR_NAMES",
    "build_parser",
    "main",
    "plan_flatprior_measurement_updates_for_files",
    "update_flatprior_measurement_attrs_in_hdf5",
]


if __name__ == "__main__":
    raise SystemExit(main())
