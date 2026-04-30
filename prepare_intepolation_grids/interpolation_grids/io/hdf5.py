"""HDF5 reading and writing for interpolation-grid generation.

This module exists to keep file mutation concerns separate from numerical
calculations. The code here decides which groups to update, preserves the
existing schema, and applies the project's safe-write behavior.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import h5py
import numpy as np

from interpolation_grids.config import (
    DEFAULT_DERIVATIVE_THETA_SAMPLES,
    DERIVATIVE_DATASET_NAME,
    GAMMA_DATASET_NAME,
    GAMMA_GRID,
    H_UNITS_V1,
    LEGACY_DERIVATIVE_DATASET_NAME,
    LEGACY_FIXED_KPC,
    MASS_DEFINITIONS_GROUP_NAME,
    MASS_DERIVATIVE_DATASET_NAME,
    MASS_GRID_DATASET_NAME,
    S2_DATASET_NAME,
    SUPPORTED_MASS_RADII_KPC,
    UNIT_VERSION,
    mass_definition_label_for_convention,
    sigma_unit_units_for_radius,
)
from interpolation_grids.models import AperturePolicy, GalaxyInputs, ProcessingSummary
from interpolation_grids.physics.jeans import kpc_per_arcsec, compute_s2_grid
from interpolation_grids.physics.m5 import compute_dmass_dthetaein_grid, compute_mass_grid
from interpolation_grids.unit_conventions import logMstar_h2_from_legacy, logRe_hinv_from_legacy


def _read_optional_attr_string(group_handle: h5py.Group, key: str) -> str | None:
    """Read one optional string attr and normalize empty values to ``None``.

    Why the normalization matters:
    - HDF5 attrs may come back as Python strings, byte strings, or scalar
      NumPy arrays depending on how the file was written.
    - The aperture parser needs one consistent "missing vs present" boundary so
      it can distinguish explicit modern metadata from legacy files that never
      declared a shape.
    """

    if key not in group_handle.attrs:
        return None
    value = group_handle.attrs[key]
    if isinstance(value, np.ndarray):
        value = value.item()
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    normalized = str(value).strip()
    return normalized or None


def _read_optional_attr_float(group_handle: h5py.Group, key: str) -> float | None:
    """Read one optional float attr while treating NaN as missing metadata."""

    if key not in group_handle.attrs:
        return None
    value = group_handle.attrs[key]
    if isinstance(value, np.ndarray):
        value = value.item()
    numeric_value = float(value)
    if np.isnan(numeric_value):
        return None
    return numeric_value


def resolve_group_aperture_policy(group_handle: h5py.Group) -> AperturePolicy | None:
    """Resolve an explicit per-group aperture policy from modern HDF5 attrs.

    Resolution rules:
    - only the modern explicit attrs participate in policy resolution
    - legacy attrs such as `aperture_width` are intentionally ignored here
    - incomplete or contradictory modern metadata fails fast instead of silently
      falling back to the file-level default
    """

    aperture_shape = _read_optional_attr_string(group_handle, "aperture_shape")
    aperture_width_arcsec = _read_optional_attr_float(group_handle, "aperture_width_arcsec")
    aperture_height_arcsec = _read_optional_attr_float(group_handle, "aperture_height_arcsec")
    aperture_radius_arcsec = _read_optional_attr_float(group_handle, "aperture_radius_arcsec")
    seeing_fwhm_arcsec = _read_optional_attr_float(group_handle, "seeing_fwhm_arcsec")

    if aperture_shape is None:
        return None

    normalized_shape = aperture_shape.strip().lower()
    if seeing_fwhm_arcsec is None:
        raise ValueError("Explicit aperture metadata requires seeing_fwhm_arcsec.")

    if normalized_shape == "rectangular":
        if aperture_width_arcsec is None:
            raise ValueError("Rectangular aperture metadata requires aperture_width_arcsec.")
        if aperture_height_arcsec is None:
            raise ValueError("Rectangular aperture metadata requires aperture_height_arcsec.")
        if aperture_radius_arcsec is not None:
            raise ValueError("Rectangular aperture metadata must not define aperture_radius_arcsec.")
        return AperturePolicy.rectangular(
            width_arcsec=aperture_width_arcsec,
            height_arcsec=aperture_height_arcsec,
            seeing_fwhm_arcsec=seeing_fwhm_arcsec,
        )

    if normalized_shape == "circular":
        if aperture_radius_arcsec is None:
            raise ValueError("Circular aperture metadata requires aperture_radius_arcsec.")
        if aperture_width_arcsec is not None or aperture_height_arcsec is not None:
            raise ValueError("Circular aperture metadata must not define aperture_width_arcsec or aperture_height_arcsec.")
        return AperturePolicy.circular(
            radius_arcsec=aperture_radius_arcsec,
            seeing_fwhm_arcsec=seeing_fwhm_arcsec,
        )

    raise ValueError(f"Unsupported aperture_shape: {aperture_shape}")


def _read_attr_or_dataset(group_handle: h5py.Group, key: str) -> float | None:
    """Read a scalar either from attributes or datasets.

    Some legacy files store metadata as attributes while other workflows may
    write them as datasets. Supporting both reduces fragility.
    """

    if key in group_handle.attrs:
        return float(group_handle.attrs[key])
    if key in group_handle:
        return float(group_handle[key][()])
    return None


def _read_num_sigma(group_name: str, group_handle: h5py.Group) -> int:
    """Read and validate the velocity-dispersion multiplicity contract.

    Why this validation lives in the raw HDF5 loader:
    - `num_sigma` is the authoritative signal for whether `s2_grid` must be
      rebuilt for one galaxy
    - accepting unsupported values here would silently create partially updated
      files whose sigma branch no longer matches the inference contract
    """

    num_sigma_raw = group_handle.attrs.get("num_sigma", 0)
    num_sigma = int(num_sigma_raw)
    if num_sigma not in (0, 1, 2):
        raise ValueError(f"{group_name} has unsupported num_sigma={num_sigma}")
    return num_sigma


def build_galaxy_inputs(group_name: str, group_handle: h5py.Group, source_filename: str) -> GalaxyInputs:
    """Normalize one HDF5 group into a structured input object.

    The returned payload intentionally carries `num_sigma` instead of the old
    `has_s2_grid` flag. The rebuild policy should follow the scientific meaning
    of the data contract, not the presence of one legacy root-level dataset.
    """

    rein_arcsec = _read_attr_or_dataset(group_handle, "rein_arcsec")
    if rein_arcsec is None:
        raise ValueError(f"{group_name} is missing rein_arcsec")

    rein_kpc = _read_attr_or_dataset(group_handle, "r_ein_kpc")
    if rein_kpc is None:
        zd = _read_attr_or_dataset(group_handle, "zd")
        if zd is None:
            raise ValueError(f"{group_name} is missing both r_ein_kpc and zd")
        rein_kpc = rein_arcsec * kpc_per_arcsec(zd)

    return GalaxyInputs(
        group_name=group_name,
        source_filename=source_filename,
        zd=float(_read_attr_or_dataset(group_handle, "zd")),
        zs=float(_read_attr_or_dataset(group_handle, "zs")),
        sigma_crit=float(_read_attr_or_dataset(group_handle, "sigma_crit")),
        rein_arcsec=float(rein_arcsec),
        r_ein_kpc=float(rein_kpc),
        re_arcsec=_read_attr_or_dataset(group_handle, "re_arcsec"),
        reff_dev_arcsec=_read_attr_or_dataset(group_handle, "reff_deV"),
        nser=_read_attr_or_dataset(group_handle, "nser"),
        aperture_width_arcsec=_read_attr_or_dataset(group_handle, "aperture_width"),
        num_sigma=_read_num_sigma(group_name=group_name, group_handle=group_handle),
    )


def _write_or_replace_dataset(group_handle: h5py.Group, dataset_name: str, values: np.ndarray) -> None:
    """Replace or create a dataset while preserving a stable schema."""

    if dataset_name in group_handle:
        del group_handle[dataset_name]
    group_handle.create_dataset(dataset_name, data=values)


def _delete_dataset_if_present(group_handle: h5py.Group, dataset_name: str) -> None:
    """Remove one dataset when legacy inputs still carry a deprecated field."""

    if dataset_name in group_handle:
        del group_handle[dataset_name]


def _validate_unit_convention(unit_convention: str) -> str:
    """Normalize and validate the HDF5 builder's explicit unit convention."""

    normalized = str(unit_convention).strip()
    if normalized not in {LEGACY_FIXED_KPC, H_UNITS_V1}:
        raise ValueError(
            f"Unsupported unit_convention: {unit_convention}. "
            f"Expected '{LEGACY_FIXED_KPC}' or '{H_UNITS_V1}'."
        )
    return normalized


def _write_unit_metadata_attrs(
    attrs: h5py.AttributeManager,
    *,
    unit_convention: str,
    h_ref: float,
    mass_radius_kpc: float | None = None,
) -> None:
    """
    Write the common unit metadata contract onto one HDF5 object.

    The same attributes appear at root, galaxy-group, and mass-subgroup levels
    so downstream loaders can validate whichever boundary they touch first.
    """

    attrs["unit_version"] = UNIT_VERSION
    attrs["unit_convention"] = unit_convention
    attrs["h_ref"] = float(h_ref)
    attrs["stellar_mass_unit"] = "h^-2 Msun" if unit_convention == H_UNITS_V1 else "Msun"
    attrs["size_unit"] = "h^-1 kpc" if unit_convention == H_UNITS_V1 else "kpc"
    attrs["mass_unit"] = "h^-1 Msun" if unit_convention == H_UNITS_V1 else "Msun"
    attrs["mass_aperture_unit"] = "h^-1 kpc" if unit_convention == H_UNITS_V1 else "kpc"
    if mass_radius_kpc is not None:
        label = mass_definition_label_for_convention(mass_radius_kpc, unit_convention)
        attrs["mass_definition_label"] = label
        attrs["mass_radius_kpc"] = float(mass_radius_kpc)
        attrs["aperture_coefficient"] = float(mass_radius_kpc)
        attrs["aperture_h_power"] = -1 if unit_convention == H_UNITS_V1 else 0
        attrs["mass_h_power"] = -1 if unit_convention == H_UNITS_V1 else 0
        attrs["units"] = sigma_unit_units_for_radius(mass_radius_kpc, unit_convention)
        attrs["mass_log_definition"] = (
            f"log10[M_2D(<{float(mass_radius_kpc):g} {attrs['mass_aperture_unit']})/{attrs['mass_unit']}]"
        )


def _delete_unselected_mass_subgroups(
    mass_definitions_handle: h5py.Group,
    *,
    selected_labels: set[str],
) -> None:
    """Remove stale mass-definition groups from a processed galaxy."""

    for label in list(mass_definitions_handle.keys()):
        if label not in selected_labels:
            del mass_definitions_handle[label]


def _write_h_unit_observable_attrs(group_handle: h5py.Group, *, h_ref: float) -> None:
    """
    Materialize h-units stellar-mass and size attributes from legacy inputs.

    The raw builder may start from legacy observation files. Writing explicit
    h-unit attrs here makes the resulting file self-describing and prevents
    inference loaders from guessing how `logmchab` or angular sizes should be
    interpreted.
    """

    if "logmchab" in group_handle.attrs:
        group_handle.attrs["logmchab_h2"] = float(
            logMstar_h2_from_legacy(float(group_handle.attrs["logmchab"]), h_ref=h_ref)
        )
    if "logmchab_deV" in group_handle.attrs:
        group_handle.attrs["logmchab_deV_h2"] = float(
            logMstar_h2_from_legacy(float(group_handle.attrs["logmchab_deV"]), h_ref=h_ref)
        )

    zd = _read_attr_or_dataset(group_handle, "zd")
    if zd is None:
        return
    physical_kpc_per_arcsec = kpc_per_arcsec(float(zd))
    if "re_arcsec" in group_handle.attrs:
        re_kpc = max(float(group_handle.attrs["re_arcsec"]) * physical_kpc_per_arcsec, 1.0e-12)
        group_handle.attrs["log10_re_hinv_kpc"] = float(logRe_hinv_from_legacy(np.log10(re_kpc), h_ref=h_ref))
    if "reff_deV" in group_handle.attrs:
        reff_kpc = max(float(group_handle.attrs["reff_deV"]) * physical_kpc_per_arcsec, 1.0e-12)
        group_handle.attrs["log10_reff_deV_hinv_kpc"] = float(logRe_hinv_from_legacy(np.log10(reff_kpc), h_ref=h_ref))


def _process_group(
    group_name: str,
    group_handle: h5py.Group,
    source_filename: str,
    summary: ProcessingSummary,
    aperture_policy: AperturePolicy | None = None,
    unit_convention: str = LEGACY_FIXED_KPC,
    h_ref: float = 0.7,
) -> None:
    """Update all relevant grids for a single galaxy group."""

    galaxy = build_galaxy_inputs(
        group_name=group_name,
        group_handle=group_handle,
        source_filename=source_filename,
    )
    gamma_grid = group_handle[GAMMA_DATASET_NAME][:] if GAMMA_DATASET_NAME in group_handle else GAMMA_GRID
    effective_aperture_policy = resolve_group_aperture_policy(group_handle) or aperture_policy
    _write_unit_metadata_attrs(group_handle.attrs, unit_convention=unit_convention, h_ref=h_ref)
    if unit_convention == H_UNITS_V1:
        _write_h_unit_observable_attrs(group_handle, h_ref=h_ref)

    mass_definitions_handle = group_handle.require_group(MASS_DEFINITIONS_GROUP_NAME)
    selected_labels = {
        mass_definition_label_for_convention(radius_kpc, unit_convention)
        for radius_kpc in SUPPORTED_MASS_RADII_KPC
    }
    _delete_unselected_mass_subgroups(
        mass_definitions_handle,
        selected_labels=selected_labels,
    )
    theta_scale_kpc_per_arcsec = galaxy.r_ein_kpc / galaxy.rein_arcsec

    for mass_radius_kpc in SUPPORTED_MASS_RADII_KPC:
        mass_grid = compute_mass_grid(
            gamma_grid=gamma_grid,
            sigma_crit=galaxy.sigma_crit,
            rein_kpc=galaxy.r_ein_kpc,
            mass_radius_kpc=mass_radius_kpc,
            unit_convention=unit_convention,
            h_ref=h_ref,
        )
        derivative_grid = compute_dmass_dthetaein_grid(
            gamma_grid=gamma_grid,
            sigma_crit=galaxy.sigma_crit,
            theta_ein_arcsec=galaxy.rein_arcsec,
            kpc_per_arcsec=theta_scale_kpc_per_arcsec,
            theta_samples=DEFAULT_DERIVATIVE_THETA_SAMPLES,
            mass_radius_kpc=mass_radius_kpc,
        )
        label = mass_definition_label_for_convention(mass_radius_kpc, unit_convention)
        subgroup = mass_definitions_handle.require_group(label)
        _write_unit_metadata_attrs(
            subgroup.attrs,
            unit_convention=unit_convention,
            h_ref=h_ref,
            mass_radius_kpc=mass_radius_kpc,
        )
        _write_or_replace_dataset(subgroup, MASS_GRID_DATASET_NAME, mass_grid)
        _write_or_replace_dataset(subgroup, MASS_DERIVATIVE_DATASET_NAME, derivative_grid)

    # The migrated raw-observation contract keeps only `gamma_grid` at the root.
    # Remove any historical root-level mass/sigma datasets from processed groups
    # so rebuilds do not reintroduce the legacy schema.
    _delete_dataset_if_present(group_handle, "m5_grid")
    _delete_dataset_if_present(group_handle, DERIVATIVE_DATASET_NAME)
    _delete_dataset_if_present(group_handle, LEGACY_DERIVATIVE_DATASET_NAME)
    _delete_dataset_if_present(group_handle, S2_DATASET_NAME)

    summary.updated_m5 += 1
    summary.updated_dm5 += 1

    if galaxy.num_sigma > 0:
        for mass_radius_kpc in SUPPORTED_MASS_RADII_KPC:
            s2_grid = compute_s2_grid(
                galaxy=galaxy,
                gamma_grid=np.asarray(gamma_grid, dtype=float),
                mass_radius_kpc=mass_radius_kpc,
                aperture_policy=effective_aperture_policy,
                unit_convention=unit_convention,
                h_ref=h_ref,
            )
            subgroup = mass_definitions_handle[mass_definition_label_for_convention(mass_radius_kpc, unit_convention)]
            _write_or_replace_dataset(subgroup, S2_DATASET_NAME, s2_grid)
        summary.updated_s2 += 1


def process_hdf5_file(
    input_path: Path | str,
    output_path: Path | str,
    overwrite_in_place: bool = False,
    group_names: tuple[str, ...] | None = None,
    aperture_policy: AperturePolicy | None = None,
    unit_convention: str = LEGACY_FIXED_KPC,
    h_ref: float = 0.7,
) -> ProcessingSummary:
    """Process one HDF5 file and write the updated result.

    Parameters
    ----------
    input_path, output_path:
        Input and output file paths. When `overwrite_in_place` is false, the
        caller gets a new output file. When true, we still use a temporary file
        first and only replace the input once processing succeeds.
    overwrite_in_place:
        Enable atomic replacement of the input file.
    group_names:
        Optional whitelist of group names to process. This is primarily for
        debugging and targeted regression checks on one or a few galaxies.
    aperture_policy:
        Optional override for the aperture geometry used when refreshing
        ``s2_grid`` for every `num_sigma > 0` group. When omitted, the
        existing rectangular production policy remains in effect.
    """

    input_path = Path(input_path)
    output_path = Path(output_path)
    normalized_unit_convention = _validate_unit_convention(unit_convention)
    h_ref = float(h_ref)
    if not np.isfinite(h_ref) or h_ref <= 0.0:
        raise ValueError(f"h_ref must be a positive finite value, got {h_ref!r}.")
    final_output_path = input_path if overwrite_in_place else output_path
    summary = ProcessingSummary(input_path=input_path, output_path=final_output_path)

    if overwrite_in_place:
        temp_directory = output_path.parent if output_path.parent.exists() else input_path.parent
        with tempfile.NamedTemporaryFile(prefix=f"{input_path.stem}.", suffix=input_path.suffix, dir=temp_directory, delete=False) as temp_file:
            working_path = Path(temp_file.name)
    else:
        working_path = output_path

    shutil.copy2(input_path, working_path)

    try:
        with h5py.File(working_path, "r+") as handle:
            _write_unit_metadata_attrs(
                handle.attrs,
                unit_convention=normalized_unit_convention,
                h_ref=h_ref,
            )
            available_group_names = list(handle.keys())
            names_to_process = group_names or tuple(available_group_names)
            for group_name in names_to_process:
                if group_name not in handle:
                    summary.failures.append(f"{group_name}: group not found")
                    continue
                summary.total_groups += 1
                try:
                    _process_group(
                        group_name=group_name,
                        group_handle=handle[group_name],
                        source_filename=input_path.name,
                        summary=summary,
                        aperture_policy=aperture_policy,
                        unit_convention=normalized_unit_convention,
                        h_ref=h_ref,
                    )
                except Exception as exc:  # noqa: BLE001 - we want per-group resilience.
                    summary.failures.append(f"{group_name}: {exc}")
        if overwrite_in_place:
            working_path.replace(input_path)
    finally:
        if overwrite_in_place and working_path.exists():
            working_path.unlink(missing_ok=True)

    return summary
