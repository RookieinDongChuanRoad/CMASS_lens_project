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


H_UNITS_V1 = "h_units_v1"
LEGACY_FIXED_KPC = "legacy_fixed_kpc"
UNIT_VERSION = "cmass_dual_units_v1"
SUPPORTED_UNIT_CONVENTIONS = frozenset({H_UNITS_V1, LEGACY_FIXED_KPC})

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
        Public aperture coefficient for the selected convention. In legacy
        mode this is also the physical kpc radius. In h-units mode this is the
        coefficient in `R h^-1 kpc`; use `physical_radius_kpc(h_ref)` whenever
        a physical kpc aperture is required by a numerical kernel.
    unit_convention:
        Explicit unit contract that selected this definition. This value is
        carried into config parsing, HDF5 validation, sigma-table validation,
        PPC metadata, and plot labeling so mixed conventions fail early.
    label:
        Public short label such as `m5` or `m10`. This is used in config
        names, metadata, plot labels, and serialized result keys.
    aperture_h_power:
        Power of `h` in the aperture unit. Legacy fixed-kpc apertures have
        power zero, while `5 h^-1 kpc` has power -1.
    mass_h_power:
        Power of `h` in the mass unit. Legacy fixed-kpc masses are physical
        `Msun` logs, while h-units masses are `h^-1 Msun` logs.
    public_parameter_names:
        The public names for the four mass-population hyper-parameters under
        this definition.
    sigma_unit_units:
        Human-readable units string for sigma-unit interpolation tables.
    subgroup_name:
        HDF5 subgroup name used under `<lens>/mass_definitions/`.
    """

    radius_kpc: float
    unit_convention: str
    label: str
    aperture_h_power: int
    mass_h_power: int
    public_parameter_names: tuple[str, str, str, str]
    sigma_unit_units: str
    subgroup_name: str

    @property
    def mass_radius_power_label(self) -> str:
        """Return the exponent label used in text outputs and metadata."""

        return self.label

    @property
    def aperture_coefficient(self) -> float:
        """Return the numeric aperture coefficient exposed in config files."""

        return float(self.radius_kpc)

    @property
    def mass_aperture_unit(self) -> str:
        """Return the human-readable aperture unit used in metadata."""

        if self.unit_convention == H_UNITS_V1:
            return "h^-1 kpc"
        return "kpc"

    @property
    def mass_unit(self) -> str:
        """Return the human-readable enclosed-mass unit used in metadata."""

        if self.unit_convention == H_UNITS_V1:
            return "h^-1 Msun"
        return "Msun"

    @property
    def mass_log_definition(self) -> str:
        """Describe the logarithmic enclosed-mass observable."""

        return f"log10[M_2D(<{self.radius_kpc:g} {self.mass_aperture_unit})/{self.mass_unit}]"

    def physical_radius_kpc(self, h_ref: float) -> float:
        """
        Convert the configured aperture into the physical kpc radius.

        The likelihood kernels solve lensing and Jeans equations in physical
        units. Keeping this conversion on the MassDefinition object prevents
        call sites from re-deriving whether `radius_kpc` is a fixed kpc value
        or an h-dependent coefficient.
        """

        validate_h_ref(h_ref)
        return float(self.radius_kpc) * float(h_ref) ** int(self.aperture_h_power)

    def log_mass_physical_offset(self, h_ref: float) -> float:
        """
        Return the additive conversion from convention mass log to physical Msun.

        Example: a h-units mass is stored as `log10[M/(h^-1 Msun)]`, so the
        physical-Msun log is larger by `log10(h_ref^-1)`.
        """

        validate_h_ref(h_ref)
        return math.log10(float(h_ref) ** int(self.mass_h_power))

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


_LEGACY_MASS_DEFINITIONS_BY_RADIUS = {
    5.0: MassDefinition(
        radius_kpc=5.0,
        unit_convention=LEGACY_FIXED_KPC,
        label="m5",
        aperture_h_power=0,
        mass_h_power=0,
        public_parameter_names=("mu5_0", "beta5", "xi5", "sigma5"),
        sigma_unit_units="km2 s-2 per 10**m5",
        subgroup_name="m5",
    ),
    10.0: MassDefinition(
        radius_kpc=10.0,
        unit_convention=LEGACY_FIXED_KPC,
        label="m10",
        aperture_h_power=0,
        mass_h_power=0,
        public_parameter_names=("mu10_0", "beta10", "xi10", "sigma10"),
        sigma_unit_units="km2 s-2 per 10**m10",
        subgroup_name="m10",
    ),
}

_H_UNIT_MASS_DEFINITIONS_BY_RADIUS = {
    5.0: MassDefinition(
        radius_kpc=5.0,
        unit_convention=H_UNITS_V1,
        label="m5_hinvkpc",
        aperture_h_power=-1,
        mass_h_power=-1,
        public_parameter_names=("mu5h_0", "beta5h", "xi5h", "sigma5h"),
        sigma_unit_units="km2 s-2 per 10**m5_hinvkpc",
        subgroup_name="m5_hinvkpc",
    ),
    10.0: MassDefinition(
        radius_kpc=10.0,
        unit_convention=H_UNITS_V1,
        label="m10_hinvkpc",
        aperture_h_power=-1,
        mass_h_power=-1,
        public_parameter_names=("mu10h_0", "beta10h", "xi10h", "sigma10h"),
        sigma_unit_units="km2 s-2 per 10**m10_hinvkpc",
        subgroup_name="m10_hinvkpc",
    ),
}

_MASS_DEFINITIONS_BY_CONVENTION = {
    LEGACY_FIXED_KPC: _LEGACY_MASS_DEFINITIONS_BY_RADIUS,
    H_UNITS_V1: _H_UNIT_MASS_DEFINITIONS_BY_RADIUS,
}

_SIGMA_BUNDLE_FILENAMES = {
    "devauc": "jeans_deV_sigma_bundle.h5",
    "sersic": "jeans_sers_sigma_bundle.h5",
}


def validate_unit_convention(unit_convention: str) -> str:
    """
    Normalize and validate the public unit-convention string.

    This small guard is used by config, HDF5, sigma-table, and PPC loaders so
    every boundary reports the same error when a convention is missing or
    misspelled.
    """

    normalized = str(unit_convention).strip()
    if normalized not in SUPPORTED_UNIT_CONVENTIONS:
        raise ValueError(
            f"Unsupported unit_convention '{unit_convention}'. Expected one of: "
            f"{', '.join(sorted(SUPPORTED_UNIT_CONVENTIONS))}."
        )
    return normalized


def validate_h_ref(h_ref: float) -> float:
    """Validate the positive finite reference Hubble parameter `h_ref`."""

    normalized = float(h_ref)
    if not math.isfinite(normalized) or normalized <= 0.0:
        raise ValueError(f"h_ref must be a positive finite value, got {h_ref!r}.")
    return normalized


def get_mass_definition(
    radius_kpc: float | int,
    *,
    unit_convention: str = LEGACY_FIXED_KPC,
) -> MassDefinition:
    """
    Return the supported mass definition matching the aperture and convention.

    The positional argument name is kept as `radius_kpc` for backward source
    compatibility with legacy callers. Under `h_units_v1` the same value is the
    `R` coefficient in `R h^-1 kpc`; call `physical_radius_kpc(h_ref)` before
    passing the aperture into physical-unit kernels.
    """

    normalized_radius = float(radius_kpc)
    normalized_convention = validate_unit_convention(unit_convention)
    try:
        return _MASS_DEFINITIONS_BY_CONVENTION[normalized_convention][normalized_radius]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported enclosed mass aperture {radius_kpc} for convention "
            f"'{normalized_convention}'. Supported apertures are 5 and 10."
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


def convert_log_mass_fixed_kpc_to_hinv(
    log_mass_fixed_kpc: np.ndarray | float,
    gamma: np.ndarray | float,
    h_ref: float,
) -> np.ndarray:
    """
    Convert a legacy fixed-kpc power-law mass log to the matching h-units log.

    The aperture changes from `R kpc` to `R h^-1 kpc` and the mass unit changes
    from `Msun` to `h^-1 Msun`. For a power-law projected mass this gives the
    analytic migration term `-(2 - gamma) log10(h_ref)`. This helper is for
    tests, debug comparisons, and controlled migration checks; production
    h-units science products should be regenerated directly in h-units.
    """

    validate_h_ref(h_ref)
    return np.asarray(
        np.asarray(log_mass_fixed_kpc, dtype=float)
        - (2.0 - np.asarray(gamma, dtype=float)) * math.log10(float(h_ref)),
        dtype=float,
    )


def convert_sigma_unit_fixed_kpc_to_hinv(
    sigma_unit_fixed_kpc: np.ndarray | float,
    gamma: np.ndarray | float,
    h_ref: float,
) -> np.ndarray:
    """
    Convert legacy fixed-kpc sigma-unit values into h-units normalization.

    Since the Jeans table stores `sigma^2 / 10**m_R`, the sigma-unit factor
    scales opposite to the mass-log migration term:
    `S_unit_h = S_unit_fixed * h_ref**(2 - gamma)`.
    """

    validate_h_ref(h_ref)
    return np.asarray(
        np.asarray(sigma_unit_fixed_kpc, dtype=float)
        * float(h_ref) ** (2.0 - np.asarray(gamma, dtype=float)),
        dtype=float,
    )


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
        "unit_version": UNIT_VERSION,
        "unit_convention": mass_definition.unit_convention,
        "label": mass_definition.label,
        "enclosed_radius_kpc": float(mass_definition.radius_kpc),
        "aperture_coefficient": float(mass_definition.aperture_coefficient),
        "aperture_h_power": int(mass_definition.aperture_h_power),
        "mass_h_power": int(mass_definition.mass_h_power),
        "mass_unit": mass_definition.mass_unit,
        "mass_aperture_unit": mass_definition.mass_aperture_unit,
        "mass_log_definition": mass_definition.mass_log_definition,
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
