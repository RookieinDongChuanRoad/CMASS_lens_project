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
) -> SigmaUnitTable:
    """
    Load and validate the sigma-unit table used by the optional FP prior.

    Validation lives here because a mismatched profile or mass definition is a
    data-contract error, not a numerical-kernel concern.
    """

    path = Path(file_path).expanduser().resolve()
    with h5py.File(path, "r") as handle:
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
