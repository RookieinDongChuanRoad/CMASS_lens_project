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

from .mass_definition import (
    LEGACY_M5_DERIVATIVE_DATASET_NAME,
    LEGACY_M5_GRID_DATASET_NAME,
    LEGACY_M5_SIGMA_DATASET_NAME,
    MASS_DERIVATIVE_DATASET_NAME,
    MASS_GRID_DATASET_NAME,
    MASS_GROUP_ROOT_NAME,
    MASS_SIGMA_DATASET_NAME,
    MassDefinition,
    convert_log_enclosed_mass,
    convert_sigma_unit_grid,
    get_mass_definition,
)
from .types import CrossSectionGrid, ObservationRecord, ProfileSpec, SigmaUnitTable

SLIT_OBSERVATION_FLAVOR = "slit"
BOSS_OBSERVATION_FLAVOR = "boss"
SIGMA_UNIT_BUNDLE_SCHEMA_VERSION = "sigma_unit_bundle_hdf5_v2"
DEFAULT_SLIT_APERTURE_WIDTH_ARCSEC = 1.6
DEFAULT_SLIT_APERTURE_HEIGHT_ARCSEC = 0.9
DEFAULT_BOSS_APERTURE_RADIUS_ARCSEC = 1.0
DEFAULT_SLIT_SEEING_FWHM_ARCSEC = 0.9
DEFAULT_BOSS_SEEING_FWHM_ARCSEC = 1.5


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
) -> SigmaUnitTable:
    """
    Load and validate the sigma-unit table used by the optional FP prior.

    Validation lives here because a mismatched profile or mass definition is a
    data-contract error, not a numerical-kernel concern.
    """

    path = Path(file_path).expanduser().resolve()
    with h5py.File(path, "r") as handle:
        schema_version = _decode_hdf5_scalar_string(handle.attrs.get("schema_version", ""))

        if schema_version == SIGMA_UNIT_BUNDLE_SCHEMA_VERSION:
            if observation_flavor is None:
                raise ValueError(
                    f"Sigma bundle '{path}' requires an explicit observation_flavor to select the correct leaf."
                )

            normalized_observation_flavor = observation_flavor.strip().lower()
            if normalized_observation_flavor not in {SLIT_OBSERVATION_FLAVOR, BOSS_OBSERVATION_FLAVOR}:
                raise ValueError(f"Unsupported observation flavor '{observation_flavor}' for sigma bundle '{path}'.")
            if normalized_observation_flavor not in handle:
                raise ValueError(
                    f"Sigma bundle '{path}' does not contain the observation flavor '{normalized_observation_flavor}'."
                )

            flavor_group = handle[normalized_observation_flavor]
            if mass_definition.label not in flavor_group:
                raise ValueError(
                    f"Sigma bundle '{path}' does not contain the mass-definition leaf "
                    f"'{normalized_observation_flavor}/{mass_definition.label}'."
                )

            leaf = flavor_group[mass_definition.label]
            required_dataset_names = {"gamma_axis", "zd_axis", "log_re_kpc_axis", "s_unit_grid"}
            missing = sorted(required_dataset_names.difference(leaf.keys()))
            if missing:
                raise ValueError(
                    f"Sigma bundle leaf '{path}:{normalized_observation_flavor}/{mass_definition.label}' "
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

            aperture_shape = _decode_hdf5_scalar_string(leaf.attrs.get("aperture_shape", "")).strip().lower()
            aperture_width_arcsec = _optional_hdf5_float(leaf.attrs.get("aperture_width_arcsec"))
            aperture_height_arcsec = _optional_hdf5_float(leaf.attrs.get("aperture_height_arcsec"))
            aperture_radius_arcsec = _optional_hdf5_float(leaf.attrs.get("aperture_radius_arcsec"))
            seeing_fwhm_arcsec = _optional_hdf5_float(leaf.attrs.get("seeing_fwhm_arcsec"))
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

            n_axis = np.asarray(leaf["n_axis"][()], dtype=float) if "n_axis" in leaf else None
            if profile_spec.uses_observed_n_in_likelihood and n_axis is None:
                raise ValueError(
                    f"Sigma table '{path}' is missing n_axis for the sersic profile schema."
                )

            return SigmaUnitTable(
                profile_name=profile_name,
                mass_definition_label=mass_definition_label,
                mass_radius_kpc=mass_radius_kpc,
                units=_decode_hdf5_scalar_string(leaf.attrs.get("units", mass_definition.sigma_unit_units)),
                gamma_axis=np.asarray(leaf["gamma_axis"][()], dtype=float),
                zd_axis=np.asarray(leaf["zd_axis"][()], dtype=float),
                log_re_kpc_axis=np.asarray(leaf["log_re_kpc_axis"][()], dtype=float),
                sigma_unit_grid=np.asarray(leaf["s_unit_grid"][()], dtype=float),
                n_axis=n_axis,
                observation_flavor=normalized_observation_flavor,
                aperture_shape=aperture_shape,
                aperture_width_arcsec=aperture_width_arcsec,
                aperture_height_arcsec=aperture_height_arcsec,
                aperture_radius_arcsec=aperture_radius_arcsec,
                seeing_fwhm_arcsec=float(seeing_fwhm_arcsec),
                bundle_leaf_path=f"/{normalized_observation_flavor}/{mass_definition.label}",
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
            units=_decode_hdf5_scalar_string(
                handle.attrs.get("units", mass_definition.sigma_unit_units)
            ),
            gamma_axis=np.asarray(handle["gamma_axis"][()], dtype=float),
            zd_axis=np.asarray(handle["zd_axis"][()], dtype=float),
            log_re_kpc_axis=np.asarray(handle["log_re_kpc_axis"][()], dtype=float),
            sigma_unit_grid=np.asarray(handle["s_unit_grid"][()], dtype=float),
            n_axis=n_axis,
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """
    Resolve the mass-dependent grids for the selected definition.

    The migration needs to support two physical layouts:
    - new files storing data under `<lens>/mass_definitions/<label>/`
    - legacy files exposing only root-level `m5_*` datasets
    """

    if MASS_GROUP_ROOT_NAME in group and mass_definition.subgroup_name in group[MASS_GROUP_ROOT_NAME]:
        selected_group = group[MASS_GROUP_ROOT_NAME][mass_definition.subgroup_name]
        mass_grid = np.asarray(selected_group[MASS_GRID_DATASET_NAME][()], dtype=float)
        dmass_grid = np.asarray(selected_group[MASS_DERIVATIVE_DATASET_NAME][()], dtype=float)
        s2_grid = (
            np.asarray(selected_group[MASS_SIGMA_DATASET_NAME][()], dtype=float)
            if MASS_SIGMA_DATASET_NAME in selected_group
            else None
        )
        return mass_grid, dmass_grid, s2_grid

    legacy_m5 = get_mass_definition(5)
    if LEGACY_M5_GRID_DATASET_NAME not in group or LEGACY_M5_DERIVATIVE_DATASET_NAME not in group:
        raise KeyError(
            "Observation group is missing both the new mass-definition subgroup "
            "schema and the legacy root-level m5 datasets."
        )

    legacy_mass_grid = np.asarray(group[LEGACY_M5_GRID_DATASET_NAME][()], dtype=float)
    legacy_dmass_grid = np.asarray(group[LEGACY_M5_DERIVATIVE_DATASET_NAME][()], dtype=float)
    legacy_s2_grid = (
        np.asarray(group[LEGACY_M5_SIGMA_DATASET_NAME][()], dtype=float)
        if LEGACY_M5_SIGMA_DATASET_NAME in group
        else None
    )
    if mass_definition.radius_kpc == legacy_m5.radius_kpc:
        return legacy_mass_grid, legacy_dmass_grid, legacy_s2_grid

    converted_mass_grid = convert_log_enclosed_mass(
        log_mass=legacy_mass_grid,
        gamma=gamma_grid,
        from_radius_kpc=legacy_m5.radius_kpc,
        to_radius_kpc=mass_definition.radius_kpc,
    )
    converted_s2_grid = None
    if legacy_s2_grid is not None:
        converted_s2_grid = convert_sigma_unit_grid(
            sigma_unit_grid=legacy_s2_grid,
            gamma=gamma_grid,
            from_radius_kpc=legacy_m5.radius_kpc,
            to_radius_kpc=mass_definition.radius_kpc,
        )
    return converted_mass_grid, legacy_dmass_grid, converted_s2_grid


def load_observations(
    file_path: str | Path,
    profile_spec: ProfileSpec,
    mass_definition: MassDefinition,
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
        for lens_id in sorted(handle.keys()):
            group = handle[lens_id]
            log_stellar_mass_obs = _resolve_attribute(
                group, profile_spec.observation_field_aliases["stellar_mass"]
            )
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
