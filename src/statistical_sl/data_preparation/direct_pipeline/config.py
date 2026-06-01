"""YAML configuration for the direct source-to-canonical pipeline.

The direct pipeline needs a schema-checked configuration layer because it owns
more than one source contract:

- catalog facts can come from either CMASS or SLACS-style tables
- velocity-dispersion measurements may arrive from an external CSV adapter or
  from trusted catalog columns in a compatibility mode
- cross-section products come from two different HDF5 layouts with different
  semantics

This module keeps those concerns explicit.  It parses the YAML file into typed
dataclasses, validates the important boundaries early, and exposes small helper
methods that return the policy objects used by the rest of the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from statistical_sl.data_preparation.direct_pipeline.policies import (
    AperturePolicyRef,
    MassDefinitionPolicy,
    SUPPORTED_PROFILE_NAMES,
    SigmaPolicy,
    UnitPolicy,
)
from statistical_sl.data_preparation.config import (
    OBSERVED_APERTURE_SIGMA_DEFINITION,
    SUPPORTED_OBSERVATION_FLAVORS,
    SUPPORTED_SIGMA_DEFINITIONS,
    mass_definition_label_for_convention,
)
from statistical_sl.data_preparation.models import AperturePolicy


DIRECT_PIPELINE_SCHEMA_VERSION = "statistical_sl_direct_data_preparation_v1"
SUPPORTED_CATALOG_TYPES = frozenset({"cmass_summary_table", "slacs_table"})
SUPPORTED_MEASUREMENT_TYPES = frozenset({"ppxf_results_adapter", "velocity_measurements_v1", "catalog_columns"})
SUPPORTED_CROSS_SECTION_TYPES = frozenset({"cmass_power_law", "sonnenfeld_fibre"})
DIRECT_PIPELINE_OBSERVATION_FLAVORS = frozenset((*SUPPORTED_OBSERVATION_FLAVORS, "slacs_fibre"))


def _normalize_text(value: Any, field_name: str, *, lowercase: bool = False) -> str:
    """Return a stripped string and reject empty configuration tokens."""

    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{field_name} must be a non-empty string.")
    return normalized.lower() if lowercase else normalized


def _coerce_bool(value: Any, field_name: str) -> bool:
    """Parse a permissive boolean token from YAML or a command-line override."""

    if isinstance(value, bool):
        return value
    normalized = _normalize_text(value, field_name).lower()
    if normalized in {"1", "true", "t", "yes", "y"}:
        return True
    if normalized in {"0", "false", "f", "no", "n"}:
        return False
    raise ValueError(f"{field_name} must be a boolean-like value, got {value!r}.")


def _resolve_path(value: Any, *, field_name: str, base_directory: Path) -> Path:
    """Resolve one configured path relative to the YAML file location."""

    if value is None:
        raise ValueError(f"{field_name} is required.")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_directory / path
    return path.resolve()


def _parse_axis(values: Any, field_name: str) -> np.ndarray:
    """Convert a YAML axis spec into a strictly increasing one-dimensional axis."""

    if isinstance(values, dict):
        minimum = float(values.get("min"))
        maximum = float(values.get("max"))
        points = int(values.get("points"))
        if not np.isfinite(minimum) or not np.isfinite(maximum):
            raise ValueError(f"{field_name}.min and {field_name}.max must be finite.")
        if maximum <= minimum:
            raise ValueError(f"{field_name}.max must be greater than {field_name}.min.")
        if points <= 0:
            raise ValueError(f"{field_name}.points must be positive.")
        values = np.linspace(minimum, maximum, points, dtype=float)

    axis = np.asarray(values, dtype=float)
    if axis.ndim != 1 or axis.size == 0:
        raise ValueError(f"{field_name} must be a non-empty one-dimensional sequence.")
    if not np.all(np.isfinite(axis)):
        raise ValueError(f"{field_name} must contain only finite numbers.")
    if np.any(np.diff(axis) <= 0.0):
        raise ValueError(f"{field_name} must be strictly increasing.")
    return axis


def _require_mapping(value: Any, field_name: str) -> dict[str, Any]:
    """Return one YAML mapping as a plain dictionary."""

    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be a mapping.")
    return dict(value)


@dataclass(frozen=True)
class OutputConfig:
    """Destination paths for the direct canonical build."""

    canonical_hdf5: Path
    audit_json: Path | None = None

    def __post_init__(self) -> None:
        """Normalize output paths so later stages never depend on cwd."""

        object.__setattr__(self, "canonical_hdf5", Path(self.canonical_hdf5).expanduser().resolve())
        if self.audit_json is not None:
            object.__setattr__(self, "audit_json", Path(self.audit_json).expanduser().resolve())


@dataclass(frozen=True)
class CatalogConfig:
    """Source catalog selection for one direct canonical build."""

    type: str
    path: Path
    profile_name: str

    def __post_init__(self) -> None:
        """Normalize the catalog type and validate the supported profile name."""

        normalized_type = _normalize_text(self.type, "catalog.type", lowercase=True)
        if normalized_type not in SUPPORTED_CATALOG_TYPES:
            raise ValueError(f"catalog.type must be one of {sorted(SUPPORTED_CATALOG_TYPES)}, got {normalized_type!r}.")
        normalized_profile_name = _normalize_text(self.profile_name, "catalog.profile_name", lowercase=True)
        if normalized_profile_name not in SUPPORTED_PROFILE_NAMES:
            raise ValueError(
                f"catalog.profile_name must be one of {sorted(SUPPORTED_PROFILE_NAMES)}, "
                f"got {normalized_profile_name!r}."
            )

        object.__setattr__(self, "type", normalized_type)
        object.__setattr__(self, "profile_name", normalized_profile_name)
        object.__setattr__(self, "path", Path(self.path).expanduser().resolve())


@dataclass(frozen=True)
class VelocityMeasurementConfig:
    """Trusted sigma-measurement source selection."""

    type: str
    path: Path | None = None
    error_column: str = "sigma_stat_kms"
    missing_policy: str = "fail"
    max_observations_per_lens: int = 2
    trust_catalog_sigma: bool = False

    def __post_init__(self) -> None:
        """Normalize the measurement source contract."""

        normalized_type = _normalize_text(self.type, "velocity_measurements.type", lowercase=True)
        if normalized_type not in SUPPORTED_MEASUREMENT_TYPES:
            raise ValueError(
                f"velocity_measurements.type must be one of {sorted(SUPPORTED_MEASUREMENT_TYPES)}, "
                f"got {normalized_type!r}."
            )

        normalized_error_column = _normalize_text(self.error_column, "velocity_measurements.error_column")
        normalized_missing_policy = _normalize_text(self.missing_policy, "velocity_measurements.missing_policy", lowercase=True)
        if normalized_missing_policy not in {"fail", "num_sigma_zero"}:
            raise ValueError(
                "velocity_measurements.missing_policy must be one of ['fail', 'num_sigma_zero'], "
                f"got {normalized_missing_policy!r}."
            )

        max_observations = int(self.max_observations_per_lens)
        if max_observations < 1 or max_observations > 2:
            raise ValueError("velocity_measurements.max_observations_per_lens must be 1 or 2.")

        if self.path is None and normalized_type != "catalog_columns":
            raise ValueError("velocity_measurements.path is required for external measurement modes.")

        object.__setattr__(self, "type", normalized_type)
        object.__setattr__(self, "error_column", normalized_error_column)
        object.__setattr__(self, "missing_policy", normalized_missing_policy)
        object.__setattr__(self, "max_observations_per_lens", max_observations)
        object.__setattr__(self, "trust_catalog_sigma", bool(self.trust_catalog_sigma))
        if self.path is not None:
            object.__setattr__(self, "path", Path(self.path).expanduser().resolve())


@dataclass(frozen=True)
class UnitConfig:
    """Mass and length unit convention for the direct build."""

    unit_convention: str
    h_ref: float = 0.7
    mass_definition_label: str = "m5_hinvkpc"
    mass_radius_kpc: float = 5.0

    def __post_init__(self) -> None:
        """Validate the unit convention and keep the mass label consistent."""

        unit_policy = UnitPolicy(self.unit_convention, self.h_ref)
        mass_policy = MassDefinitionPolicy(self.mass_definition_label, self.mass_radius_kpc)
        try:
            expected_label = mass_definition_label_for_convention(mass_policy.mass_radius_kpc, unit_policy.unit_convention)
        except KeyError as exc:
            raise ValueError(
                "mass_radius_kpc must be one of the supported radii for the configured unit convention."
            ) from exc
        if mass_policy.mass_definition_label != expected_label:
            raise ValueError(
                f"mass_definition_label must be {expected_label!r} for mass_radius_kpc={mass_policy.mass_radius_kpc!r} "
                f"and unit_convention={unit_policy.unit_convention!r}."
            )

        object.__setattr__(self, "unit_convention", unit_policy.unit_convention)
        object.__setattr__(self, "h_ref", unit_policy.h_ref)
        object.__setattr__(self, "mass_definition_label", mass_policy.mass_definition_label)
        object.__setattr__(self, "mass_radius_kpc", mass_policy.mass_radius_kpc)


@dataclass(frozen=True)
class ApertureConfig:
    """Dataset-level aperture semantics and concrete geometry."""

    observation_flavor: str
    sigma_definition: str
    shape: str
    width_arcsec: float | None = None
    height_arcsec: float | None = None
    radius_arcsec: float | None = None
    seeing_fwhm_arcsec: float | None = None

    def __post_init__(self) -> None:
        """Validate both the semantic labels and the physical geometry."""

        normalized_flavor = _normalize_text(self.observation_flavor, "aperture.observation_flavor", lowercase=True)
        if normalized_flavor not in DIRECT_PIPELINE_OBSERVATION_FLAVORS:
            raise ValueError(
                f"aperture.observation_flavor must be one of {sorted(DIRECT_PIPELINE_OBSERVATION_FLAVORS)}, "
                f"got {normalized_flavor!r}."
            )

        normalized_sigma_definition = _normalize_text(
            self.sigma_definition,
            "aperture.sigma_definition",
            lowercase=True,
        )
        if normalized_sigma_definition not in SUPPORTED_SIGMA_DEFINITIONS:
            raise ValueError(
                f"aperture.sigma_definition must be one of {sorted(SUPPORTED_SIGMA_DEFINITIONS)}, "
                f"got {normalized_sigma_definition!r}."
            )

        normalized_shape = _normalize_text(self.shape, "aperture.shape", lowercase=True)
        object.__setattr__(self, "observation_flavor", normalized_flavor)
        object.__setattr__(self, "sigma_definition", normalized_sigma_definition)
        object.__setattr__(self, "shape", normalized_shape)
        object.__setattr__(self, "width_arcsec", None if self.width_arcsec is None else float(self.width_arcsec))
        object.__setattr__(self, "height_arcsec", None if self.height_arcsec is None else float(self.height_arcsec))
        object.__setattr__(self, "radius_arcsec", None if self.radius_arcsec is None else float(self.radius_arcsec))
        object.__setattr__(
            self,
            "seeing_fwhm_arcsec",
            None if self.seeing_fwhm_arcsec is None else float(self.seeing_fwhm_arcsec),
        )

        # Constructing an AperturePolicy is the clearest way to validate the
        # geometry because the downstream physics layer already depends on that
        # exact contract.
        self.to_aperture_policy()

    def to_aperture_policy(self) -> AperturePolicy:
        """Return the concrete aperture geometry used by the Jeans kernels."""

        if self.shape == "rectangular":
            return AperturePolicy.rectangular(
                width_arcsec=float(self.width_arcsec),
                height_arcsec=float(self.height_arcsec),
                seeing_fwhm_arcsec=float(self.seeing_fwhm_arcsec),
            )
        if self.shape == "circular":
            return AperturePolicy.circular(
                radius_arcsec=float(self.radius_arcsec),
                seeing_fwhm_arcsec=float(self.seeing_fwhm_arcsec),
            )
        raise ValueError(f"Unsupported aperture.shape: {self.shape!r}.")


@dataclass(frozen=True)
class GridConfig:
    """Explicit numeric axes used by the in-memory builders."""

    gamma_axis: np.ndarray
    theta_e_axis: np.ndarray

    def __post_init__(self) -> None:
        """Normalize the axes so downstream code receives concrete arrays."""

        object.__setattr__(self, "gamma_axis", _parse_axis(self.gamma_axis, "grids.gamma_axis"))
        object.__setattr__(self, "theta_e_axis", _parse_axis(self.theta_e_axis, "grids.theta_e_axis"))


@dataclass(frozen=True)
class CrossSectionConfig:
    """Cross-section source product used by the direct canonical build."""

    type: str
    source_hdf5: Path

    def __post_init__(self) -> None:
        """Validate the source mode and normalize the resolved path."""

        normalized_type = _normalize_text(self.type, "cross_section.type", lowercase=True)
        if normalized_type not in SUPPORTED_CROSS_SECTION_TYPES:
            raise ValueError(
                f"cross_section.type must be one of {sorted(SUPPORTED_CROSS_SECTION_TYPES)}, got {normalized_type!r}."
            )
        object.__setattr__(self, "type", normalized_type)
        object.__setattr__(self, "source_hdf5", Path(self.source_hdf5).expanduser().resolve())


@dataclass(frozen=True)
class DirectPipelineConfig:
    """Complete direct-pipeline configuration parsed from YAML."""

    schema_version: str
    output: OutputConfig
    catalog: CatalogConfig
    velocity_measurements: VelocityMeasurementConfig
    units: UnitConfig
    aperture: ApertureConfig
    grids: GridConfig
    cross_section: CrossSectionConfig

    def __post_init__(self) -> None:
        """Reject unknown schema versions early and keep the object explicit."""

        normalized_schema_version = _normalize_text(self.schema_version, "schema_version")
        if normalized_schema_version != DIRECT_PIPELINE_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must be {DIRECT_PIPELINE_SCHEMA_VERSION!r}, got {normalized_schema_version!r}."
            )
        object.__setattr__(self, "schema_version", normalized_schema_version)

    def unit_policy(self) -> UnitPolicy:
        """Return the unit-policy object consumed by the physics builders."""

        return UnitPolicy(unit_convention=self.units.unit_convention, h_ref=self.units.h_ref)

    def mass_policy(self) -> MassDefinitionPolicy:
        """Return the mass-definition policy consumed by the grid builders."""

        return MassDefinitionPolicy(
            mass_definition_label=self.units.mass_definition_label,
            mass_radius_kpc=self.units.mass_radius_kpc,
        )

    def aperture_policy_ref(self) -> AperturePolicyRef:
        """Return the semantic aperture wrapper used by lens preparation."""

        return AperturePolicyRef(
            observation_flavor=self.aperture.observation_flavor,
            sigma_definition=self.aperture.sigma_definition,
            aperture_policy=self.aperture.to_aperture_policy(),
        )

    def sigma_policy(self) -> SigmaPolicy:
        """Return the sigma-resolution policy used to resolve trusted measurements."""

        return SigmaPolicy(
            source_type=self.velocity_measurements.type,
            missing_policy=self.velocity_measurements.missing_policy,
            max_observations_per_lens=self.velocity_measurements.max_observations_per_lens,
            error_column=self.velocity_measurements.error_column,
            trust_catalog_sigma=self.velocity_measurements.trust_catalog_sigma,
        )


def load_direct_pipeline_config(config_path: Path | str) -> DirectPipelineConfig:
    """Parse one YAML configuration file into a validated typed config object."""

    resolved_config_path = Path(config_path).expanduser().resolve()
    raw_config = yaml.safe_load(resolved_config_path.read_text(encoding="utf-8"))
    if not isinstance(raw_config, dict):
        raise ValueError(f"{resolved_config_path} must contain a YAML mapping at the top level.")

    base_directory = resolved_config_path.parent

    output_block = _require_mapping(raw_config.get("output"), "output")
    catalog_block = _require_mapping(raw_config.get("catalog"), "catalog")
    velocity_block = _require_mapping(raw_config.get("velocity_measurements"), "velocity_measurements")
    units_block = _require_mapping(raw_config.get("units"), "units")
    aperture_block = _require_mapping(raw_config.get("aperture"), "aperture")
    grids_block = _require_mapping(raw_config.get("grids"), "grids")
    cross_section_block = _require_mapping(raw_config.get("cross_section"), "cross_section")

    output = OutputConfig(
        canonical_hdf5=_resolve_path(output_block.get("canonical_hdf5"), field_name="output.canonical_hdf5", base_directory=base_directory),
        audit_json=(
            None
            if output_block.get("audit_json") is None
            else _resolve_path(output_block.get("audit_json"), field_name="output.audit_json", base_directory=base_directory)
        ),
    )

    catalog = CatalogConfig(
        type=catalog_block.get("type"),
        path=_resolve_path(catalog_block.get("path"), field_name="catalog.path", base_directory=base_directory),
        profile_name=catalog_block.get("profile_name"),
    )
    if not catalog.path.exists():
        raise ValueError(f"catalog.path does not exist: {catalog.path}")

    velocity_type = _normalize_text(velocity_block.get("type"), "velocity_measurements.type", lowercase=True)
    velocity_path_value = velocity_block.get("path")
    velocity_path = None
    if velocity_path_value is not None:
        velocity_path = _resolve_path(
            velocity_path_value,
            field_name="velocity_measurements.path",
            base_directory=base_directory,
        )
    if velocity_type != "catalog_columns":
        if velocity_path is None:
            raise ValueError("velocity_measurements.path is required for external measurement modes.")
        if not velocity_path.exists():
            raise ValueError(f"velocity_measurements.path does not exist: {velocity_path}")

    velocity_measurements = VelocityMeasurementConfig(
        type=velocity_type,
        path=velocity_path,
        error_column=velocity_block.get("error_column", "sigma_stat_kms"),
        missing_policy=velocity_block.get("missing_policy", "fail"),
        max_observations_per_lens=velocity_block.get("max_observations_per_lens", 2),
        trust_catalog_sigma=_coerce_bool(velocity_block.get("trust_catalog_sigma", False), "velocity_measurements.trust_catalog_sigma"),
    )
    if velocity_measurements.type == "catalog_columns" and not velocity_measurements.trust_catalog_sigma:
        raise ValueError("catalog_columns mode requires velocity_measurements.trust_catalog_sigma=true.")

    units = UnitConfig(
        unit_convention=units_block.get("unit_convention"),
        h_ref=units_block.get("h_ref", 0.7),
        mass_definition_label=units_block.get("mass_definition_label", "m5_hinvkpc"),
        mass_radius_kpc=units_block.get("mass_radius_kpc", 5.0),
    )

    aperture = ApertureConfig(
        observation_flavor=aperture_block.get("observation_flavor"),
        sigma_definition=aperture_block.get("sigma_definition", OBSERVED_APERTURE_SIGMA_DEFINITION),
        shape=aperture_block.get("shape"),
        width_arcsec=aperture_block.get("width_arcsec"),
        height_arcsec=aperture_block.get("height_arcsec"),
        radius_arcsec=aperture_block.get("radius_arcsec"),
        seeing_fwhm_arcsec=aperture_block.get("seeing_fwhm_arcsec"),
    )

    grids = GridConfig(
        gamma_axis=grids_block.get("gamma_axis"),
        theta_e_axis=grids_block.get("theta_e_axis"),
    )

    cross_section = CrossSectionConfig(
        type=cross_section_block.get("type"),
        source_hdf5=_resolve_path(
            cross_section_block.get("source_hdf5"),
            field_name="cross_section.source_hdf5",
            base_directory=base_directory,
        ),
    )
    if not cross_section.source_hdf5.exists():
        raise ValueError(f"cross_section.source_hdf5 does not exist: {cross_section.source_hdf5}")

    config = DirectPipelineConfig(
        schema_version=raw_config.get("schema_version"),
        output=output,
        catalog=catalog,
        velocity_measurements=velocity_measurements,
        units=units,
        aperture=aperture,
        grids=grids,
        cross_section=cross_section,
    )

    if config.catalog.type == "cmass_summary_table" and config.velocity_measurements.type == "catalog_columns" and not config.velocity_measurements.trust_catalog_sigma:
        raise ValueError("CMASS catalog_columns mode requires velocity_measurements.trust_catalog_sigma=true.")

    return config


__all__ = [
    "ApertureConfig",
    "CatalogConfig",
    "CrossSectionConfig",
    "DIRECT_PIPELINE_SCHEMA_VERSION",
    "DIRECT_PIPELINE_OBSERVATION_FLAVORS",
    "DirectPipelineConfig",
    "GridConfig",
    "OutputConfig",
    "UnitConfig",
    "VelocityMeasurementConfig",
    "load_direct_pipeline_config",
]
