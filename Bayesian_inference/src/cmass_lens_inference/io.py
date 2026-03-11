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

from .types import CrossSectionGrid, ObservationRecord, ProfileSpec


def _resolve_attribute(group: h5py.Group, aliases: tuple[str, ...]) -> float:
    """Return the first matching HDF5 attribute among the provided aliases."""

    for field_name in aliases:
        if field_name in group.attrs:
            return float(group.attrs[field_name])
    raise KeyError(f"None of the attribute aliases were found: {aliases}")


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


def load_observations(file_path: str | Path, profile_spec: ProfileSpec) -> list[ObservationRecord]:
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
                    gamma_grid_17=np.asarray(group["gamma_grid"][()], dtype=float),
                    m5_grid_17=np.asarray(group["m5_grid"][()], dtype=float),
                    dm5_dthetaein_grid_17=np.asarray(group["dm5_dthetaein_grid"][()], dtype=float),
                    s2_grid_17=np.asarray(group["s2_grid"][()], dtype=float) if "s2_grid" in group else None,
                )
            )

    return observations
