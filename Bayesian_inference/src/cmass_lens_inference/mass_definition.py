"""
Shared mass-definition helpers for the CMASS lens pipeline.

Why this module exists:
- the scientific model now needs to switch between `m5` and `m10` without
  duplicating the full inference and posterior-predictive implementations
- external interfaces still need definition-specific names such as `mu5_0`
  and `mu10_0`
- the exact algebra linking different enclosed-mass radii should live in one
  auditable place instead of being re-derived ad hoc across the codebase
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import numpy as np


INTERNAL_MASS_PARAMETER_NAMES: tuple[str, ...] = (
    "mu5_0",
    "beta5",
    "xi5",
    "sigma5",
)

SHARED_PARAMETER_NAMES: tuple[str, ...] = (
    "mu_gamma_0",
    "beta_gamma",
    "xi_gamma",
    "sigma_gamma",
    "mu_zs",
    "sigma_zs",
    "theta0",
    "loga",
)

MASS_GROUP_ROOT_NAME = "mass_definitions"
MASS_GRID_DATASET_NAME = "mass_grid"
MASS_DERIVATIVE_DATASET_NAME = "dmass_dthetaein_grid"
MASS_SIGMA_DATASET_NAME = "s2_grid"
LEGACY_M5_GRID_DATASET_NAME = "m5_grid"
LEGACY_M5_DERIVATIVE_DATASET_NAME = "dm5_dthetaein_grid"
LEGACY_M5_SIGMA_DATASET_NAME = "s2_grid"


@dataclass(frozen=True)
class MassDefinition:
    """
    Runtime description of one enclosed-mass convention.

    Attributes
    ----------
    radius_kpc:
        Physical aperture radius used to define the enclosed projected mass.
    label:
        Public short label such as `m5` or `m10`. This is used in config
        names, metadata, plot labels, and serialized result keys.
    public_parameter_names:
        The public names for the four mass-population hyper-parameters under
        this definition.
    sigma_unit_units:
        Human-readable units string for sigma-unit interpolation tables.
    subgroup_name:
        HDF5 subgroup name used under `<lens>/mass_definitions/`.
    """

    radius_kpc: float
    label: str
    public_parameter_names: tuple[str, str, str, str]
    sigma_unit_units: str
    subgroup_name: str

    @property
    def mass_radius_power_label(self) -> str:
        """Return the exponent label used in text outputs and metadata."""

        return self.label

    def sigma_table_filename(self, profile_name: str) -> str:
        """Return the canonical sigma-table filename for one profile branch."""

        normalized_profile = profile_name.strip().lower()
        if normalized_profile == "devauc":
            profile_stem = "jeans_deV"
        elif normalized_profile == "sersic":
            profile_stem = "jeans_sers"
        else:
            raise ValueError(f"Unsupported sigma-table profile: {profile_name}")
        return f"{profile_stem}_{self.label}_grid.h5"


_MASS_DEFINITIONS_BY_RADIUS = {
    5.0: MassDefinition(
        radius_kpc=5.0,
        label="m5",
        public_parameter_names=("mu5_0", "beta5", "xi5", "sigma5"),
        sigma_unit_units="km2 s-2 per 10**m5",
        subgroup_name="m5",
    ),
    10.0: MassDefinition(
        radius_kpc=10.0,
        label="m10",
        public_parameter_names=("mu10_0", "beta10", "xi10", "sigma10"),
        sigma_unit_units="km2 s-2 per 10**m10",
        subgroup_name="m10",
    ),
}

_SIGMA_BUNDLE_FILENAMES = {
    "devauc": "jeans_deV_sigma_bundle.h5",
    "sersic": "jeans_sers_sigma_bundle.h5",
}


def get_mass_definition(radius_kpc: float | int) -> MassDefinition:
    """Return the supported mass definition matching `radius_kpc`."""

    normalized_radius = float(radius_kpc)
    try:
        return _MASS_DEFINITIONS_BY_RADIUS[normalized_radius]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported enclosed mass radius {radius_kpc}. Supported radii are 5 and 10 kpc."
        ) from exc


def sigma_bundle_filename(profile_name: str) -> str:
    """Return the canonical per-profile sigma bundle filename."""

    normalized_profile = profile_name.strip().lower()
    try:
        return _SIGMA_BUNDLE_FILENAMES[normalized_profile]
    except KeyError as exc:
        raise ValueError(f"Unsupported sigma-bundle profile: {profile_name}") from exc


def convert_log_enclosed_mass(
    log_mass: np.ndarray | float,
    gamma: np.ndarray | float,
    from_radius_kpc: float | int,
    to_radius_kpc: float | int,
) -> np.ndarray:
    """
    Convert `log10(M_2D(<R))` between two enclosed-mass radii.

    For a power-law profile,

    `10**m_R = pi * Sigma_c * r_ein**(gamma-1) * R**(3-gamma)`.

    Holding the physical lens fixed means changing only the definition radius,
    so the two logarithmic mass observables differ by the analytic term

    `(3 - gamma) * log10(R_to / R_from)`.
    """

    log_mass_array = np.asarray(log_mass, dtype=float)
    gamma_array = np.asarray(gamma, dtype=float)
    if float(from_radius_kpc) == float(to_radius_kpc):
        return np.array(log_mass_array, dtype=float, copy=True)
    radius_ratio = float(to_radius_kpc) / float(from_radius_kpc)
    return np.asarray(log_mass_array + (3.0 - gamma_array) * math.log10(radius_ratio), dtype=float)


def convert_sigma_unit_grid(
    sigma_unit_grid: np.ndarray | float,
    gamma: np.ndarray | float,
    from_radius_kpc: float | int,
    to_radius_kpc: float | int,
) -> np.ndarray:
    """
    Convert `S_unit`/`s2_grid` between enclosed-mass definitions.

    Why this relation is exact:
    - the Jeans interpolation tables store `sigma^2 / 10**m_R`
    - for the same physical lens, changing only the aperture definition from
      `R_from` to `R_to` changes the mass normalization by the analytic factor
      encoded in `convert_log_enclosed_mass`
    - therefore the sigma-unit grid must scale by `10**(m_from - m_to)`
    """

    sigma_unit_array = np.asarray(sigma_unit_grid, dtype=float)
    gamma_array = np.asarray(gamma, dtype=float)
    if float(from_radius_kpc) == float(to_radius_kpc):
        return np.array(sigma_unit_array, dtype=float, copy=True)
    radius_ratio = float(from_radius_kpc) / float(to_radius_kpc)
    return np.asarray(sigma_unit_array * (radius_ratio ** (3.0 - gamma_array)), dtype=float)


def normalize_public_initial_center(
    initial_center_raw: Mapping[str, float],
    mass_definition: MassDefinition,
) -> dict[str, float]:
    """
    Map definition-specific public config keys onto the internal parameter set.

    The internal inference machinery still consumes one fixed 12-parameter
    vector. This helper keeps that vector stable while allowing the YAML
    surface to expose `mu5_*` or `mu10_*` according to the selected mass
    definition.
    """

    normalized: dict[str, float] = {}
    for internal_name, public_name in zip(
        INTERNAL_MASS_PARAMETER_NAMES,
        mass_definition.public_parameter_names,
        strict=True,
    ):
        normalized[internal_name] = float(initial_center_raw[public_name])

    for name in SHARED_PARAMETER_NAMES:
        normalized[name] = float(initial_center_raw[name])
    return normalized


def serialize_public_initial_center(
    internal_initial_center: Mapping[str, float],
    mass_definition: MassDefinition,
) -> dict[str, float]:
    """
    Expose one internal hyper-parameter set under the selected public names.

    The sampler and kernels keep a stable internal order, but anything written
    to metadata, JSON, or downstream PPC inputs should use the run's public
    naming surface so users do not need to remember the internal aliasing.
    """

    serialized: dict[str, float] = {}
    for internal_name, public_name in zip(
        INTERNAL_MASS_PARAMETER_NAMES,
        mass_definition.public_parameter_names,
        strict=True,
    ):
        serialized[public_name] = float(internal_initial_center[internal_name])

    for name in SHARED_PARAMETER_NAMES:
        serialized[name] = float(internal_initial_center[name])
    return serialized


def mass_definition_metadata(mass_definition: MassDefinition) -> dict[str, float | str]:
    """Return a stable metadata payload describing one mass definition."""

    return {
        "label": mass_definition.label,
        "enclosed_radius_kpc": float(mass_definition.radius_kpc),
        "sigma_unit_units": mass_definition.sigma_unit_units,
        "subgroup_name": mass_definition.subgroup_name,
    }


def public_mass_keys(mass_definition: MassDefinition, stem: str) -> tuple[str, str]:
    """
    Return a pair of result keys for summary serialization.

    This small helper keeps downstream code honest when it needs a definition-
    specific key such as `theta_sample_m10` instead of `theta_sample_m5`.
    """

    return stem, f"{stem}_{mass_definition.label}"
