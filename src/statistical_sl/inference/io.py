"""
HDF5 loading and schema compatibility helpers.

The scientific core should not know anything about alternate field names or
file-layout quirks. This module normalizes those variations into stable typed
records before the rest of the pipeline sees them.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from statistical_sl.core.mass_definition import (
    H_UNITS_V1,
    LEGACY_FIXED_KPC,
    MASS_DERIVATIVE_DATASET_NAME,
    MASS_GRID_DATASET_NAME,
    MASS_GROUP_ROOT_NAME,
    MASS_SIGMA_DATASET_NAME,
    MassDefinition,
    validate_h_ref,
    validate_unit_convention,
)
from statistical_sl.inference.types import CrossSectionGrid, ObservationRecord, ProfileSpec, SigmaUnitTable

SLIT_OBSERVATION_FLAVOR = "slit"
BOSS_OBSERVATION_FLAVOR = "boss"
SIGMA_UNIT_BUNDLE_SCHEMA_VERSION = "sigma_unit_bundle_hdf5_v2"
DEFAULT_SLIT_APERTURE_WIDTH_ARCSEC = 1.6
DEFAULT_SLIT_APERTURE_HEIGHT_ARCSEC = 0.9
DEFAULT_BOSS_APERTURE_RADIUS_ARCSEC = 1.0
DEFAULT_SLIT_SEEING_FWHM_ARCSEC = 0.9
DEFAULT_BOSS_SEEING_FWHM_ARCSEC = 1.5
OBSERVED_APERTURE_SIGMA_DEFINITION = "observed_aperture"
WITHIN_RE_SIGMA_DEFINITION = "within_re"


def _resolve_attribute(group: h5py.Group, aliases: tuple[str, ...]) -> float:
    """Return the first matching HDF5 attribute among the provided aliases."""

    for field_name in aliases:
        if field_name in group.attrs:
            return float(group.attrs[field_name])
    raise KeyError(f"None of the attribute aliases were found: {aliases}")


def _decode_hdf5_scalar_string(raw_value: object) -> str:
    """Normalize the scalar string encodings returned by HDF5."""

    if isinstance(raw_value, bytes):
        return raw_value.decode("utf-8")
    if isinstance(raw_value, np.ndarray) and raw_value.shape == ():
        return _decode_hdf5_scalar_string(raw_value.item())
    return str(raw_value)


def _optional_hdf5_float(raw_value: object) -> float | None:
    """Decode one optional HDF5 scalar into a Python float when possible."""

    if raw_value is None:
        return None
    if isinstance(raw_value, np.ndarray) and raw_value.shape == ():
        return _optional_hdf5_float(raw_value.item())
    try:
        return float(raw_value)
    except (TypeError, ValueError):
        return None


def _optional_resolve_attribute(group: h5py.Group, aliases: tuple[str, ...]) -> float | None:
    """Return the first matching attribute, or `None` when every alias is absent."""

    for field_name in aliases:
        if field_name in group.attrs:
            return float(group.attrs[field_name])
    return None


def _validate_unit_metadata(
    attrs: h5py.AttributeManager,
    *,
    expected_mass_definition: MassDefinition,
    expected_h_ref: float | None,
    context_label: str,
    allow_missing_legacy: bool = True,
) -> tuple[str, float | None]:
    """
    Validate unit metadata at one HDF5 boundary.

    Legacy files that predate this migration are allowed to omit the metadata
    only when the active run is explicitly legacy-compatible. H-unit files must
    carry an explicit convention and, when the caller supplies `h_ref`, a
    matching reference Hubble parameter.
    """

    raw_convention = attrs.get("unit_convention")
    if raw_convention is None:
        if allow_missing_legacy and expected_mass_definition.unit_convention == LEGACY_FIXED_KPC:
            return LEGACY_FIXED_KPC, None
        raise ValueError(
            f"{context_label} is missing unit_convention metadata required by "
            f"active convention '{expected_mass_definition.unit_convention}'."
        )

    actual_convention = validate_unit_convention(_decode_hdf5_scalar_string(raw_convention))
    if actual_convention != expected_mass_definition.unit_convention:
        raise ValueError(
            f"{context_label} unit_convention='{actual_convention}' does not match "
            f"active convention '{expected_mass_definition.unit_convention}'."
        )

    actual_h_ref = _optional_hdf5_float(attrs.get("h_ref"))
    if expected_h_ref is not None:
        expected_h_value = validate_h_ref(expected_h_ref)
        if actual_h_ref is None:
            raise ValueError(f"{context_label} is missing h_ref metadata required for convention validation.")
        if not np.isclose(actual_h_ref, expected_h_value):
            raise ValueError(
                f"{context_label} h_ref={actual_h_ref:g} does not match active h_ref={expected_h_value:g}."
            )
    return actual_convention, actual_h_ref


def _stellar_mass_aliases(profile_spec: ProfileSpec, mass_definition: MassDefinition) -> tuple[str, ...]:
    """Return the stellar-mass attribute aliases for the active unit convention."""

    if mass_definition.unit_convention == H_UNITS_V1:
        if profile_spec.name == "devauc":
            return ("logmchab_deV_h2", "logmchab_h2")
        return ("logmchab_h2",)
    return profile_spec.observation_field_aliases["stellar_mass"]


def _effective_radius_log_aliases(profile_spec: ProfileSpec) -> tuple[str, ...]:
    """Return h-units size-log aliases for the active profile."""

    if profile_spec.name == "devauc":
        return ("log10_reff_deV_hinv_kpc", "log10_re_hinv_kpc")
    return ("log10_re_hinv_kpc",)


def _load_observation_contract(file_path: str | Path) -> dict[str, float | str | None]:
    """
    Resolve the raw observation contract, validating BOSS files strictly.

    Historical slit raw files remain backward-compatible even when they omit
    aperture metadata. BOSS files are stricter because the 1.5 arcsec seeing
    is now part of the public physical contract and downstream bundle
    validation depends on it.
    """

    path = Path(file_path).expanduser().resolve()
    with h5py.File(path, "r") as handle:
        group_names = sorted(handle.keys())
        if not group_names:
            raise ValueError(f"Observation file '{path}' contains no lens groups.")

        sample_group = handle[group_names[0]]
        aperture_shape = _decode_hdf5_scalar_string(sample_group.attrs.get("aperture_shape", "")).strip().lower()
        aperture_radius_arcsec = _optional_hdf5_float(sample_group.attrs.get("aperture_radius_arcsec"))
        is_boss = aperture_shape == "circular" and aperture_radius_arcsec is not None and np.isclose(
            aperture_radius_arcsec,
            DEFAULT_BOSS_APERTURE_RADIUS_ARCSEC,
        )

        if not is_boss:
            return {
                "observation_flavor": SLIT_OBSERVATION_FLAVOR,
                "aperture_shape": "rectangular",
                "aperture_width_arcsec": DEFAULT_SLIT_APERTURE_WIDTH_ARCSEC,
                "aperture_height_arcsec": DEFAULT_SLIT_APERTURE_HEIGHT_ARCSEC,
                "aperture_radius_arcsec": None,
                "seeing_fwhm_arcsec": DEFAULT_SLIT_SEEING_FWHM_ARCSEC,
            }

        for group_name in group_names:
            group = handle[group_name]
            group_shape = _decode_hdf5_scalar_string(group.attrs.get("aperture_shape", "")).strip().lower()
            group_radius = _optional_hdf5_float(group.attrs.get("aperture_radius_arcsec"))
            group_seeing = _optional_hdf5_float(group.attrs.get("seeing_fwhm_arcsec"))
            if group_shape != "circular" or group_radius is None or not np.isclose(
                group_radius,
                DEFAULT_BOSS_APERTURE_RADIUS_ARCSEC,
            ):
                raise ValueError(
                    f"BOSS observation file '{path}' has inconsistent circular-aperture metadata in lens '{group_name}'."
                )
            if group_seeing is None or not np.isclose(group_seeing, DEFAULT_BOSS_SEEING_FWHM_ARCSEC):
                raise ValueError(
                    f"BOSS observation file '{path}' must record seeing_fwhm_arcsec={DEFAULT_BOSS_SEEING_FWHM_ARCSEC:.1f} "
                    f"for every lens group; lens '{group_name}' does not."
                )

        return {
            "observation_flavor": BOSS_OBSERVATION_FLAVOR,
            "aperture_shape": "circular",
            "aperture_width_arcsec": None,
            "aperture_height_arcsec": None,
            "aperture_radius_arcsec": DEFAULT_BOSS_APERTURE_RADIUS_ARCSEC,
            "seeing_fwhm_arcsec": DEFAULT_BOSS_SEEING_FWHM_ARCSEC,
        }


def load_observation_contract(file_path: str | Path) -> dict[str, float | str | None]:
    """Expose the raw observation contract to runtime callers."""

    return _load_observation_contract(file_path)


def load_cross_section_grid(file_path: str | Path) -> CrossSectionGrid:
    """
    Load the cross-section lookup grid with support for alias dataset names.
    """

    path = Path(file_path).expanduser().resolve()
    with h5py.File(path, "r") as handle:
        compressed = handle["compressed_grids"]
        gamma_dataset_name = "gamma_grid" if "gamma_grid" in compressed else "gamma_grids"
        cs_dataset_name = (
            "cs_over_theta_ein"
            if "cs_over_theta_ein" in compressed
            else "cs_over_theta_ein_grid"
        )
        return CrossSectionGrid(
            gamma_grid=np.asarray(compressed[gamma_dataset_name][()], dtype=float),
            cs_over_theta_ein=np.asarray(compressed[cs_dataset_name][()], dtype=float),
        )


def load_sigma_unit_table(
    file_path: str | Path,
    profile_spec: ProfileSpec,
    mass_definition: MassDefinition,
    observation_flavor: str | None = None,
    bundle_group: str | None = None,
    h_ref: float | None = None,
) -> SigmaUnitTable:
    """
    Load and validate the sigma-unit table used by the optional FP prior.

    Validation lives here because a mismatched profile or mass definition is a
    data-contract error, not a numerical-kernel concern.
    """

    path = Path(file_path).expanduser().resolve()
    with h5py.File(path, "r") as handle:
        schema_version = _decode_hdf5_scalar_string(handle.attrs.get("schema_version", ""))
        root_convention, root_h_ref = _validate_unit_metadata(
            handle.attrs,
            expected_mass_definition=mass_definition,
            expected_h_ref=h_ref,
            context_label=f"Sigma table '{path}'",
        )

        if schema_version == SIGMA_UNIT_BUNDLE_SCHEMA_VERSION:
            selected_bundle_group = bundle_group.strip().lower() if bundle_group is not None else None
            if selected_bundle_group is None:
                if observation_flavor is None:
                    raise ValueError(
                        f"Sigma bundle '{path}' requires either `bundle_group` or `observation_flavor` to select the correct leaf."
                    )
                selected_bundle_group = observation_flavor.strip().lower()

            if selected_bundle_group not in handle:
                raise ValueError(
                    f"Sigma bundle '{path}' does not contain the bundle group '{selected_bundle_group}'."
                )

            bundle_root_group = handle[selected_bundle_group]
            if mass_definition.label not in bundle_root_group:
                raise ValueError(
                    f"Sigma bundle '{path}' does not contain the mass-definition leaf "
                    f"'{selected_bundle_group}/{mass_definition.label}'."
                )

            leaf = bundle_root_group[mass_definition.label]
            leaf_convention, leaf_h_ref = _validate_unit_metadata(
                leaf.attrs,
                expected_mass_definition=mass_definition,
                expected_h_ref=h_ref,
                context_label=f"Sigma bundle leaf '{path}:{selected_bundle_group}/{mass_definition.label}'",
            )
            required_dataset_names = {"gamma_axis", "log_re_kpc_axis", "s_unit_grid"}
            if selected_bundle_group != WITHIN_RE_SIGMA_DEFINITION:
                required_dataset_names.add("zd_axis")
            missing = sorted(required_dataset_names.difference(leaf.keys()))
            if missing:
                raise ValueError(
                    f"Sigma bundle leaf '{path}:{selected_bundle_group}/{mass_definition.label}' "
                    f"is missing datasets: {missing}."
                )

            profile_name = _decode_hdf5_scalar_string(handle["profile_name"][()])
            if profile_name != profile_spec.name:
                raise ValueError(
                    f"Sigma table profile '{profile_name}' does not match active profile "
                    f"'{profile_spec.name}'."
                )

            mass_definition_label = _decode_hdf5_scalar_string(
                leaf.attrs.get("mass_definition_label", mass_definition.label)
            )
            mass_radius_kpc = float(leaf.attrs.get("mass_radius_kpc", float(mass_definition.radius_kpc)))
            if (
                mass_definition_label != mass_definition.label
                or not np.isclose(mass_radius_kpc, float(mass_definition.radius_kpc))
            ):
                raise ValueError(
                    f"Sigma table mass definition '{mass_definition_label}' ({mass_radius_kpc:g} kpc) "
                    f"does not match active mass definition '{mass_definition.label}' "
                    f"({mass_definition.radius_kpc:g} kpc)."
                )

            sigma_definition = _decode_hdf5_scalar_string(
                leaf.attrs.get(
                    "sigma_definition",
                    WITHIN_RE_SIGMA_DEFINITION
                    if selected_bundle_group == WITHIN_RE_SIGMA_DEFINITION
                    else OBSERVED_APERTURE_SIGMA_DEFINITION,
                )
            ).strip().lower()
            aperture_shape = _decode_hdf5_scalar_string(leaf.attrs.get("aperture_shape", "")).strip().lower()
            aperture_width_arcsec = _optional_hdf5_float(leaf.attrs.get("aperture_width_arcsec"))
            aperture_height_arcsec = _optional_hdf5_float(leaf.attrs.get("aperture_height_arcsec"))
            aperture_radius_arcsec = _optional_hdf5_float(leaf.attrs.get("aperture_radius_arcsec"))
            seeing_fwhm_arcsec = _optional_hdf5_float(leaf.attrs.get("seeing_fwhm_arcsec"))
            n_axis = np.asarray(leaf["n_axis"][()], dtype=float) if "n_axis" in leaf else None
            if profile_spec.uses_observed_n_in_likelihood and n_axis is None:
                raise ValueError(
                    f"Sigma table '{path}' is missing n_axis for the sersic profile schema."
                )

            if selected_bundle_group == WITHIN_RE_SIGMA_DEFINITION:
                aperture_radius_mode = _decode_hdf5_scalar_string(leaf.attrs.get("aperture_radius_mode", "")).strip().lower()
                seeing_mode = _decode_hdf5_scalar_string(leaf.attrs.get("seeing_mode", "")).strip().lower()
                if sigma_definition != WITHIN_RE_SIGMA_DEFINITION:
                    raise ValueError(
                        f"Sigma bundle leaf '{path}:{selected_bundle_group}/{mass_definition.label}' "
                        "does not declare sigma_definition='within_re'."
                    )
                if aperture_shape != "circular":
                    raise ValueError("Within-Re sigma bundle metadata must declare a circular aperture.")
                if aperture_radius_mode != "effective_radius":
                    raise ValueError("Within-Re sigma bundle metadata must declare aperture_radius_mode='effective_radius'.")
                if seeing_mode != "none":
                    raise ValueError("Within-Re sigma bundle metadata must declare seeing_mode='none'.")
                if any(value is not None for value in (aperture_width_arcsec, aperture_height_arcsec, aperture_radius_arcsec)):
                    raise ValueError("Within-Re sigma bundle metadata must not carry fixed angular aperture sizes.")
                if seeing_fwhm_arcsec is not None:
                    raise ValueError("Within-Re sigma bundle metadata must not carry a seeing_fwhm_arcsec value.")

                return SigmaUnitTable(
                    profile_name=profile_name,
                    mass_definition_label=mass_definition_label,
                    mass_radius_kpc=mass_radius_kpc,
                    unit_convention=leaf_convention,
                    h_ref=leaf_h_ref if leaf_h_ref is not None else root_h_ref,
                    units=_decode_hdf5_scalar_string(leaf.attrs.get("units", mass_definition.sigma_unit_units)),
                    gamma_axis=np.asarray(leaf["gamma_axis"][()], dtype=float),
                    zd_axis=None,
                    log_re_kpc_axis=np.asarray(leaf["log_re_kpc_axis"][()], dtype=float),
                    sigma_unit_grid=np.asarray(leaf["s_unit_grid"][()], dtype=float),
                    n_axis=n_axis,
                    sigma_definition=sigma_definition,
                    bundle_group_name=selected_bundle_group,
                    observation_flavor=None,
                    aperture_shape=aperture_shape,
                    aperture_width_arcsec=None,
                    aperture_height_arcsec=None,
                    aperture_radius_arcsec=None,
                    seeing_fwhm_arcsec=None,
                    bundle_leaf_path=f"/{selected_bundle_group}/{mass_definition.label}",
                )

            normalized_observation_flavor = selected_bundle_group
            if normalized_observation_flavor not in {SLIT_OBSERVATION_FLAVOR, BOSS_OBSERVATION_FLAVOR}:
                raise ValueError(
                    f"Unsupported observation flavor '{normalized_observation_flavor}' for sigma bundle '{path}'."
                )
            if sigma_definition != OBSERVED_APERTURE_SIGMA_DEFINITION:
                raise ValueError(
                    f"Sigma bundle leaf '{path}:{selected_bundle_group}/{mass_definition.label}' "
                    "does not declare sigma_definition='observed_aperture'."
                )
            if normalized_observation_flavor == BOSS_OBSERVATION_FLAVOR:
                if aperture_shape != "circular" or aperture_radius_arcsec is None or not np.isclose(
                    aperture_radius_arcsec,
                    DEFAULT_BOSS_APERTURE_RADIUS_ARCSEC,
                ):
                    raise ValueError("Sigma bundle aperture metadata does not match the BOSS circular-aperture contract.")
                if seeing_fwhm_arcsec is None or not np.isclose(seeing_fwhm_arcsec, DEFAULT_BOSS_SEEING_FWHM_ARCSEC):
                    raise ValueError("Sigma bundle seeing metadata does not match the BOSS 1.5 arcsec contract.")
            else:
                if aperture_shape != "rectangular" or aperture_width_arcsec is None or not np.isclose(
                    aperture_width_arcsec,
                    DEFAULT_SLIT_APERTURE_WIDTH_ARCSEC,
                ) or aperture_height_arcsec is None or not np.isclose(
                    aperture_height_arcsec,
                    DEFAULT_SLIT_APERTURE_HEIGHT_ARCSEC,
                ):
                    raise ValueError("Sigma bundle aperture metadata does not match the slit rectangular-aperture contract.")
                if seeing_fwhm_arcsec is None or not np.isclose(seeing_fwhm_arcsec, DEFAULT_SLIT_SEEING_FWHM_ARCSEC):
                    raise ValueError("Sigma bundle seeing metadata does not match the slit 0.9 arcsec contract.")

            return SigmaUnitTable(
                profile_name=profile_name,
                mass_definition_label=mass_definition_label,
                mass_radius_kpc=mass_radius_kpc,
                unit_convention=leaf_convention,
                h_ref=leaf_h_ref if leaf_h_ref is not None else root_h_ref,
                units=_decode_hdf5_scalar_string(leaf.attrs.get("units", mass_definition.sigma_unit_units)),
                gamma_axis=np.asarray(leaf["gamma_axis"][()], dtype=float),
                zd_axis=np.asarray(leaf["zd_axis"][()], dtype=float),
                log_re_kpc_axis=np.asarray(leaf["log_re_kpc_axis"][()], dtype=float),
                sigma_unit_grid=np.asarray(leaf["s_unit_grid"][()], dtype=float),
                n_axis=n_axis,
                sigma_definition=sigma_definition,
                bundle_group_name=selected_bundle_group,
                observation_flavor=normalized_observation_flavor,
                aperture_shape=aperture_shape,
                aperture_width_arcsec=aperture_width_arcsec,
                aperture_height_arcsec=aperture_height_arcsec,
                aperture_radius_arcsec=aperture_radius_arcsec,
                seeing_fwhm_arcsec=float(seeing_fwhm_arcsec),
                bundle_leaf_path=f"/{selected_bundle_group}/{mass_definition.label}",
            )

        required_dataset_names = {
            "profile_name",
            "gamma_axis",
            "zd_axis",
            "log_re_kpc_axis",
            "s_unit_grid",
        }
        missing = sorted(required_dataset_names.difference(handle.keys()))
        if missing:
            raise ValueError(
                f"Sigma table '{path}' does not match the required HDF5 schema. "
                f"Missing datasets: {missing}."
            )

        profile_name = _decode_hdf5_scalar_string(handle["profile_name"][()])
        if profile_name != profile_spec.name:
            raise ValueError(
                f"Sigma table profile '{profile_name}' does not match active profile "
                f"'{profile_spec.name}'."
            )

        mass_definition_label = _decode_hdf5_scalar_string(
            handle.attrs.get("mass_definition_label", mass_definition.label)
        )
        mass_radius_kpc = float(handle.attrs.get("mass_radius_kpc", float(mass_definition.radius_kpc)))
        if (
            mass_definition_label != mass_definition.label
            or not np.isclose(mass_radius_kpc, float(mass_definition.radius_kpc))
        ):
            raise ValueError(
                f"Sigma table mass definition '{mass_definition_label}' ({mass_radius_kpc:g} kpc) "
                f"does not match active mass definition '{mass_definition.label}' "
                f"({mass_definition.radius_kpc:g} kpc)."
            )

        n_axis = np.asarray(handle["n_axis"][()], dtype=float) if "n_axis" in handle else None
        if profile_spec.uses_observed_n_in_likelihood and n_axis is None:
            raise ValueError(
                f"Sigma table '{path}' is missing n_axis for the sersic profile schema."
            )

        table_observation_flavor = _decode_hdf5_scalar_string(
            handle.attrs.get("observation_flavor", SLIT_OBSERVATION_FLAVOR)
        ).strip().lower()
        if observation_flavor is not None and table_observation_flavor != observation_flavor.strip().lower():
            raise ValueError(
                f"Sigma table observation flavor '{table_observation_flavor}' does not match "
                f"active observation flavor '{observation_flavor.strip().lower()}'."
            )

        aperture_shape = _decode_hdf5_scalar_string(handle.attrs.get("aperture_shape", ""))
        aperture_width_arcsec = _optional_hdf5_float(handle.attrs.get("aperture_width_arcsec"))
        aperture_height_arcsec = _optional_hdf5_float(handle.attrs.get("aperture_height_arcsec"))
        aperture_radius_arcsec = _optional_hdf5_float(handle.attrs.get("aperture_radius_arcsec"))
        seeing_fwhm_arcsec = _optional_hdf5_float(handle.attrs.get("seeing_fwhm_arcsec"))

        return SigmaUnitTable(
            profile_name=profile_name,
            mass_definition_label=mass_definition_label,
            mass_radius_kpc=mass_radius_kpc,
            unit_convention=root_convention,
            h_ref=root_h_ref,
            units=_decode_hdf5_scalar_string(
                handle.attrs.get("units", mass_definition.sigma_unit_units)
            ),
            gamma_axis=np.asarray(handle["gamma_axis"][()], dtype=float),
            zd_axis=np.asarray(handle["zd_axis"][()], dtype=float),
            log_re_kpc_axis=np.asarray(handle["log_re_kpc_axis"][()], dtype=float),
            sigma_unit_grid=np.asarray(handle["s_unit_grid"][()], dtype=float),
            n_axis=n_axis,
            sigma_definition=OBSERVED_APERTURE_SIGMA_DEFINITION,
            bundle_group_name=table_observation_flavor,
            observation_flavor=table_observation_flavor,
            aperture_shape=aperture_shape,
            aperture_width_arcsec=aperture_width_arcsec,
            aperture_height_arcsec=aperture_height_arcsec,
            aperture_radius_arcsec=aperture_radius_arcsec,
            seeing_fwhm_arcsec=float(
                seeing_fwhm_arcsec
                if seeing_fwhm_arcsec is not None
                else (
                    DEFAULT_BOSS_SEEING_FWHM_ARCSEC
                    if table_observation_flavor == BOSS_OBSERVATION_FLAVOR
                    else DEFAULT_SLIT_SEEING_FWHM_ARCSEC
                )
            ),
        )


def _load_mass_dependent_grids(
    group: h5py.Group,
    gamma_grid: np.ndarray,
    mass_definition: MassDefinition,
    h_ref: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """
    Resolve the mass-dependent grids for the selected definition.

    Supported layout:
    - files store mass-dependent data under `<lens>/mass_definitions/<label>/`
    """

    if MASS_GROUP_ROOT_NAME not in group or mass_definition.subgroup_name not in group[MASS_GROUP_ROOT_NAME]:
        raise KeyError(
            "Observation group is missing the required mass-definition subgroup "
            f"schema for '{mass_definition.subgroup_name}'."
        )

    selected_group = group[MASS_GROUP_ROOT_NAME][mass_definition.subgroup_name]
    _validate_unit_metadata(
        selected_group.attrs,
        expected_mass_definition=mass_definition,
        expected_h_ref=h_ref,
        context_label=(
            f"Observation mass-definition subgroup "
            f"'{group.name}/{MASS_GROUP_ROOT_NAME}/{mass_definition.subgroup_name}'"
        ),
    )
    if MASS_GRID_DATASET_NAME not in selected_group:
        raise KeyError(
            "Observation mass-definition subgroup is missing required dataset "
            f"'{mass_definition.subgroup_name}/{MASS_GRID_DATASET_NAME}'."
        )
    if MASS_DERIVATIVE_DATASET_NAME not in selected_group:
        raise KeyError(
            "Observation mass-definition subgroup is missing required dataset "
            f"'{mass_definition.subgroup_name}/{MASS_DERIVATIVE_DATASET_NAME}'."
        )

    mass_grid = np.asarray(selected_group[MASS_GRID_DATASET_NAME][()], dtype=float)
    dmass_grid = np.asarray(selected_group[MASS_DERIVATIVE_DATASET_NAME][()], dtype=float)
    s2_grid = (
        np.asarray(selected_group[MASS_SIGMA_DATASET_NAME][()], dtype=float)
        if MASS_SIGMA_DATASET_NAME in selected_group
        else None
    )
    return mass_grid, dmass_grid, s2_grid


def load_observations(
    file_path: str | Path,
    profile_spec: ProfileSpec,
    mass_definition: MassDefinition,
    h_ref: float | None = None,
) -> list[ObservationRecord]:
    """
    Read the observation HDF5 file and normalize profile-specific aliases.

    The return type is intentionally explicit so downstream code can depend on
    stable field names regardless of how the raw files were produced.
    """

    path = Path(file_path).expanduser().resolve()
    _load_observation_contract(path)
    observations: list[ObservationRecord] = []

    with h5py.File(path, "r") as handle:
        _validate_unit_metadata(
            handle.attrs,
            expected_mass_definition=mass_definition,
            expected_h_ref=h_ref,
            context_label=f"Observation file '{path}'",
        )
        for lens_id in sorted(handle.keys()):
            group = handle[lens_id]
            _validate_unit_metadata(
                group.attrs,
                expected_mass_definition=mass_definition,
                expected_h_ref=h_ref,
                context_label=f"Observation group '{path}:{lens_id}'",
            )
            log_stellar_mass_obs = _resolve_attribute(
                group, _stellar_mass_aliases(profile_spec, mass_definition)
            )
            log_effective_radius_obs: float | None = None
            if mass_definition.unit_convention == H_UNITS_V1:
                log_effective_radius_obs = _resolve_attribute(
                    group,
                    _effective_radius_log_aliases(profile_spec),
                )
                effective_radius_arcsec = _optional_resolve_attribute(
                    group,
                    profile_spec.observation_field_aliases["effective_radius_arcsec"],
                )
                if effective_radius_arcsec is None:
                    effective_radius_arcsec = float("nan")
            else:
                effective_radius_arcsec = _resolve_attribute(
                    group, profile_spec.observation_field_aliases["effective_radius_arcsec"]
                )
            sigma_observed = np.array([], dtype=float)
            sigma_error = np.array([], dtype=float)
            num_sigma = int(group.attrs["num_sigma"])
            if num_sigma > 0:
                sigma_values = np.atleast_1d(group.attrs["sigma"]).astype(float)
                sigma_error_values = np.atleast_1d(group.attrs["sigma_err"]).astype(float)
                sigma_observed = sigma_values
                sigma_error = sigma_error_values

            n_observed = (
                float(profile_spec.fixed_n)
                if profile_spec.fixed_n is not None
                else _resolve_attribute(group, profile_spec.observation_field_aliases["nser"])
            )
            gamma_grid = np.asarray(group["gamma_grid"][()], dtype=float)
            mass_grid, dmass_grid, s2_grid = _load_mass_dependent_grids(
                group=group,
                gamma_grid=gamma_grid,
                mass_definition=mass_definition,
                h_ref=h_ref,
            )

            observations.append(
                ObservationRecord(
                    lens_id=lens_id,
                    z_d=float(group.attrs["zd"]),
                    z_s=float(group.attrs["zs"]),
                    log_stellar_mass_obs=log_stellar_mass_obs,
                    log_stellar_mass_err=_resolve_attribute(
                        group, profile_spec.observation_field_aliases["stellar_mass_error"]
                    ),
                    n_observed=n_observed,
                    effective_radius_arcsec=effective_radius_arcsec,
                    log_effective_radius_obs=log_effective_radius_obs,
                    einstein_radius_arcsec=_resolve_attribute(
                        group, profile_spec.observation_field_aliases["einstein_radius_arcsec"]
                    ),
                    num_sigma=num_sigma,
                    sigma_observed=sigma_observed,
                    sigma_error=sigma_error,
                    gamma_grid_17=gamma_grid,
                    mass_grid_17=mass_grid,
                    dmass_dthetaein_grid_17=dmass_grid,
                    s2_grid_17=s2_grid,
                )
            )

    return observations
