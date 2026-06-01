"""Domain records for the direct source-to-canonical pipeline.

The direct pipeline uses these records as stable handoff points between stages:

1. ``BaseLensRecord`` carries catalog facts and deliberately excludes trusted
   velocity-dispersion measurements.
2. ``SigmaObservation`` carries one accepted likelihood measurement in km/s.
3. ``PreparedLensRecord`` joins catalog facts with accepted measurements and
   derived preparation state.
4. ``CanonicalDatasetPayload`` is the in-memory object handed to the HDF5 writer.

Keeping these boundaries explicit prevents the old failure mode where a table
column named ``sigma`` was accidentally treated as a trusted measurement source.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from statistical_sl.data_preparation.models import AperturePolicy


MAX_SIGMA_OBSERVATIONS = 2


def _normalize_required_text(value: str, field_name: str) -> str:
    """Return a stripped string and reject empty domain identifiers."""

    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{field_name} must be a non-empty string.")
    return normalized


def _normalize_optional_text(value: str | None) -> str | None:
    """Normalize optional text fields while preserving explicit absence."""

    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _require_finite(value: float, field_name: str) -> float:
    """Validate one scalar that must be finite but may be signed."""

    numeric_value = float(value)
    if not np.isfinite(numeric_value):
        raise ValueError(f"{field_name} must be finite.")
    return numeric_value


def _require_positive_finite(value: float, field_name: str) -> float:
    """Validate one scalar that must be strictly positive and finite."""

    numeric_value = _require_finite(value, field_name)
    if numeric_value <= 0.0:
        raise ValueError(f"{field_name} must be positive.")
    return numeric_value


def _optional_positive_finite(value: float | None, field_name: str) -> float | None:
    """Validate an optional positive scalar without inventing missing values."""

    if value is None:
        return None
    return _require_positive_finite(value, field_name)


def _freeze_mapping(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    """Return a read-only shallow copy for audit metadata fields."""

    return MappingProxyType(dict(value or {}))


def _validate_optional_aperture_metadata(observation: "SigmaObservation") -> None:
    """Validate row-level aperture metadata when a measurement provides it.

    Measurement rows are allowed to omit aperture fields because many datasets
    use a dataset-level default aperture policy.  If a row starts declaring a
    geometry, however, the geometry must be internally complete; otherwise a
    later preparation stage could produce a canonical file that only appears
    self-describing.
    """

    if observation.aperture_shape is None:
        present_geometry = (
            observation.aperture_width_arcsec,
            observation.aperture_height_arcsec,
            observation.aperture_radius_arcsec,
            observation.seeing_fwhm_arcsec,
        )
        if any(value is not None for value in present_geometry):
            raise ValueError("aperture_shape is required when aperture dimensions or seeing are provided.")
        return

    if observation.seeing_fwhm_arcsec is None:
        raise ValueError("seeing_fwhm_arcsec is required when aperture_shape is provided.")

    if observation.aperture_shape == "rectangular":
        if observation.aperture_width_arcsec is None or observation.aperture_height_arcsec is None:
            raise ValueError("rectangular aperture metadata requires width and height.")
        if observation.aperture_radius_arcsec is not None:
            raise ValueError("rectangular aperture metadata must not define a radius.")
        return

    if observation.aperture_shape == "circular":
        if observation.aperture_radius_arcsec is None:
            raise ValueError("circular aperture metadata requires a radius.")
        if observation.aperture_width_arcsec is not None or observation.aperture_height_arcsec is not None:
            raise ValueError("circular aperture metadata must not define width or height.")
        return

    raise ValueError(f"Unsupported aperture_shape: {observation.aperture_shape!r}.")


@dataclass(frozen=True)
class BaseLensRecord:
    """Catalog-only facts for one lens before measurement resolution.

    Parameters
    ----------
    lens_id:
        Canonical lens identifier used as the join key across catalog,
        measurement, and canonical output blocks.
    z_lens, z_source:
        Lens and source redshifts.  The source redshift must be greater than the
        lens redshift because every downstream distance calculation assumes a
        physical lens-source geometry.
    theta_ein_arcsec:
        Einstein radius in arcsec.
    effective_radius_arcsec, effective_radius_kpc:
        Tracer effective radius.  The angular value is required by source
        catalogs; the physical value may be supplied by low-redshift catalogs or
        computed later by the preparation stage.
    log_stellar_mass, log_stellar_mass_err:
        Stellar-mass estimate and uncertainty in the active catalog convention.
    profile_name, sersic_index:
        Tracer profile branch and optional Sersic index.  deV records normally
        carry ``sersic_index=4`` for auditability.
    source_metadata:
        Optional immutable audit metadata such as catalog line number.  It is
        not used as trusted scientific data.
    """

    lens_id: str
    z_lens: float
    z_source: float
    theta_ein_arcsec: float
    effective_radius_arcsec: float
    log_stellar_mass: float
    log_stellar_mass_err: float
    profile_name: str
    sersic_index: float | None = None
    effective_radius_kpc: float | None = None
    theta_ein_kpc: float | None = None
    ra_deg: float | None = None
    dec_deg: float | None = None
    source_metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate catalog facts without adding any measurement semantics."""

        object.__setattr__(self, "lens_id", _normalize_required_text(self.lens_id, "lens_id"))
        object.__setattr__(self, "profile_name", _normalize_required_text(self.profile_name, "profile_name").lower())

        z_lens = _require_finite(self.z_lens, "z_lens")
        z_source = _require_finite(self.z_source, "z_source")
        if z_lens < 0.0 or z_source <= z_lens:
            raise ValueError("BaseLensRecord requires non-negative z_lens and z_source > z_lens.")
        object.__setattr__(self, "z_lens", z_lens)
        object.__setattr__(self, "z_source", z_source)

        object.__setattr__(
            self,
            "theta_ein_arcsec",
            _require_positive_finite(self.theta_ein_arcsec, "theta_ein_arcsec"),
        )
        object.__setattr__(
            self,
            "effective_radius_arcsec",
            _require_positive_finite(self.effective_radius_arcsec, "effective_radius_arcsec"),
        )
        object.__setattr__(
            self,
            "effective_radius_kpc",
            _optional_positive_finite(self.effective_radius_kpc, "effective_radius_kpc"),
        )
        object.__setattr__(
            self,
            "theta_ein_kpc",
            _optional_positive_finite(self.theta_ein_kpc, "theta_ein_kpc"),
        )

        object.__setattr__(self, "log_stellar_mass", _require_finite(self.log_stellar_mass, "log_stellar_mass"))
        object.__setattr__(
            self,
            "log_stellar_mass_err",
            _require_positive_finite(self.log_stellar_mass_err, "log_stellar_mass_err"),
        )
        object.__setattr__(
            self,
            "sersic_index",
            _optional_positive_finite(self.sersic_index, "sersic_index"),
        )
        object.__setattr__(self, "ra_deg", None if self.ra_deg is None else _require_finite(self.ra_deg, "ra_deg"))
        object.__setattr__(self, "dec_deg", None if self.dec_deg is None else _require_finite(self.dec_deg, "dec_deg"))
        object.__setattr__(self, "source_metadata", _freeze_mapping(self.source_metadata))


@dataclass(frozen=True)
class SigmaObservation:
    """One trusted velocity-dispersion observation used by the likelihood.

    All sigma values are in km/s by contract.  Rejected or failed upstream rows
    should live in provenance/audit records, not in this accepted-observation
    dataclass.
    """

    lens_id: str
    sigma_kms: float
    sigma_err_kms: float
    obs_tag: str | None = None
    sigma_error_kind: str = "statistical"
    measurement_status: str = "success"
    use_for_likelihood: bool = True
    source_system: str | None = None
    source_file: str | None = None
    aperture_shape: str | None = None
    aperture_width_arcsec: float | None = None
    aperture_height_arcsec: float | None = None
    aperture_radius_arcsec: float | None = None
    seeing_fwhm_arcsec: float | None = None
    source_metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalize identifiers and validate physical measurement values."""

        object.__setattr__(self, "lens_id", _normalize_required_text(self.lens_id, "lens_id"))

        normalized_tag = _normalize_optional_text(self.obs_tag)
        object.__setattr__(self, "obs_tag", None if normalized_tag is None else normalized_tag.upper())

        object.__setattr__(self, "sigma_kms", _require_positive_finite(self.sigma_kms, "sigma_kms"))
        object.__setattr__(
            self,
            "sigma_err_kms",
            _require_positive_finite(self.sigma_err_kms, "sigma_err_kms"),
        )
        object.__setattr__(
            self,
            "sigma_error_kind",
            _normalize_required_text(self.sigma_error_kind, "sigma_error_kind").lower(),
        )
        object.__setattr__(
            self,
            "measurement_status",
            _normalize_required_text(self.measurement_status, "measurement_status"),
        )
        object.__setattr__(self, "use_for_likelihood", bool(self.use_for_likelihood))
        object.__setattr__(self, "source_system", _normalize_optional_text(self.source_system))
        object.__setattr__(self, "source_file", _normalize_optional_text(self.source_file))

        normalized_aperture_shape = _normalize_optional_text(self.aperture_shape)
        object.__setattr__(
            self,
            "aperture_shape",
            None if normalized_aperture_shape is None else normalized_aperture_shape.lower(),
        )
        object.__setattr__(
            self,
            "aperture_width_arcsec",
            _optional_positive_finite(self.aperture_width_arcsec, "aperture_width_arcsec"),
        )
        object.__setattr__(
            self,
            "aperture_height_arcsec",
            _optional_positive_finite(self.aperture_height_arcsec, "aperture_height_arcsec"),
        )
        object.__setattr__(
            self,
            "aperture_radius_arcsec",
            _optional_positive_finite(self.aperture_radius_arcsec, "aperture_radius_arcsec"),
        )
        object.__setattr__(
            self,
            "seeing_fwhm_arcsec",
            _optional_positive_finite(self.seeing_fwhm_arcsec, "seeing_fwhm_arcsec"),
        )
        object.__setattr__(self, "source_metadata", _freeze_mapping(self.source_metadata))

        _validate_optional_aperture_metadata(self)


@dataclass(frozen=True)
class PreparedLensRecord:
    """A catalog lens after sigma resolution and preparation-state attachment."""

    base_lens: BaseLensRecord
    sigma_observations: tuple[SigmaObservation, ...] = ()
    sigma_crit: float | None = None
    aperture_policy: AperturePolicy | None = None
    observation_flavor: str | None = None
    sigma_definition: str | None = None
    unit_convention: str | None = None
    h_ref: float | None = None
    theta_ein_kpc: float | None = None
    effective_radius_kpc: float | None = None
    active_log_stellar_mass: float | None = None
    active_log_effective_radius: float | None = None
    preparation_metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the joined record that later grid builders will consume."""

        if not isinstance(self.base_lens, BaseLensRecord):
            raise TypeError("base_lens must be a BaseLensRecord.")

        observations = tuple(self.sigma_observations)
        if len(observations) > MAX_SIGMA_OBSERVATIONS:
            raise ValueError(f"PreparedLensRecord supports at most two sigma observations, got {len(observations)}.")
        for observation in observations:
            if not isinstance(observation, SigmaObservation):
                raise TypeError("sigma_observations must contain SigmaObservation instances.")
            if observation.lens_id != self.base_lens.lens_id:
                raise ValueError(
                    f"sigma_observations lens_id {observation.lens_id!r} does not match "
                    f"base lens_id {self.base_lens.lens_id!r}."
                )
        object.__setattr__(self, "sigma_observations", observations)

        if self.sigma_crit is not None:
            object.__setattr__(self, "sigma_crit", _require_positive_finite(self.sigma_crit, "sigma_crit"))
        if self.aperture_policy is not None and not isinstance(self.aperture_policy, AperturePolicy):
            raise TypeError("aperture_policy must be an AperturePolicy instance when provided.")
        object.__setattr__(
            self,
            "observation_flavor",
            None if self.observation_flavor is None else _normalize_required_text(self.observation_flavor, "observation_flavor"),
        )
        object.__setattr__(
            self,
            "sigma_definition",
            None if self.sigma_definition is None else _normalize_required_text(self.sigma_definition, "sigma_definition"),
        )
        object.__setattr__(
            self,
            "unit_convention",
            None if self.unit_convention is None else _normalize_required_text(self.unit_convention, "unit_convention"),
        )
        if self.h_ref is not None:
            object.__setattr__(self, "h_ref", _require_positive_finite(self.h_ref, "h_ref"))
        if self.theta_ein_kpc is not None:
            object.__setattr__(self, "theta_ein_kpc", _require_positive_finite(self.theta_ein_kpc, "theta_ein_kpc"))
        if self.effective_radius_kpc is not None:
            object.__setattr__(
                self,
                "effective_radius_kpc",
                _require_positive_finite(self.effective_radius_kpc, "effective_radius_kpc"),
            )
        if self.active_log_stellar_mass is not None:
            object.__setattr__(
                self,
                "active_log_stellar_mass",
                _require_finite(self.active_log_stellar_mass, "active_log_stellar_mass"),
            )
        if self.active_log_effective_radius is not None:
            object.__setattr__(
                self,
                "active_log_effective_radius",
                _require_finite(self.active_log_effective_radius, "active_log_effective_radius"),
            )
        object.__setattr__(self, "preparation_metadata", _freeze_mapping(self.preparation_metadata))

    @property
    def lens_id(self) -> str:
        """Expose the canonical join key directly on the prepared record."""

        return self.base_lens.lens_id

    @property
    def num_sigma(self) -> int:
        """Return the number of accepted likelihood sigma observations."""

        return len(self.sigma_observations)


@dataclass(frozen=True)
class CanonicalDatasetPayload:
    """In-memory representation of the canonical HDF5 top-level blocks.

    The writer should be a serialization boundary over this object.  It should
    not need to rediscover measurement provenance, unit policy, or per-lens grid
    availability from intermediate HDF5 files.
    """

    metadata: Mapping[str, Any]
    lenses: Mapping[str, Any]
    lensing_mass_grids: Mapping[str, Any]
    lensing_cross_section: Mapping[str, Any]
    velocity_dispersion_grids: Mapping[str, Any]
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Freeze top-level mappings so later stages cannot mutate silently."""

        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))
        object.__setattr__(self, "lenses", _freeze_mapping(self.lenses))
        object.__setattr__(self, "lensing_mass_grids", _freeze_mapping(self.lensing_mass_grids))
        object.__setattr__(self, "lensing_cross_section", _freeze_mapping(self.lensing_cross_section))
        object.__setattr__(
            self,
            "velocity_dispersion_grids",
            _freeze_mapping(self.velocity_dispersion_grids),
        )
        object.__setattr__(self, "provenance", _freeze_mapping(self.provenance))


__all__ = [
    "BaseLensRecord",
    "CanonicalDatasetPayload",
    "MAX_SIGMA_OBSERVATIONS",
    "PreparedLensRecord",
    "SigmaObservation",
]
