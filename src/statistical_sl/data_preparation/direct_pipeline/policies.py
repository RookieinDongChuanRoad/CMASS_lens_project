"""Policy records for the direct canonical-pipeline configuration layer.

The direct pipeline needs to keep two categories of information separate:

- *facts* that come from catalogs or measurements
- *policies* that describe how those facts should be interpreted

This module carries the second category only.  The objects are deliberately
lightweight dataclasses because they are created from YAML configuration and
then passed through the pipeline as explicit, inspectable decisions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from statistical_sl.data_preparation.config import H_UNITS_V1, LEGACY_FIXED_KPC
from statistical_sl.data_preparation.models import AperturePolicy


SUPPORTED_UNIT_CONVENTIONS = frozenset({LEGACY_FIXED_KPC, H_UNITS_V1})
SUPPORTED_PROFILE_NAMES = frozenset({"devauc", "sersic"})
SUPPORTED_MISSING_POLICIES = frozenset({"fail", "num_sigma_zero"})


def _normalize_required_text(value: str, field_name: str) -> str:
    """Return a stripped string and reject empty configuration tokens."""

    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{field_name} must be a non-empty string.")
    return normalized


def _require_positive_finite(value: float, field_name: str) -> float:
    """Validate a scalar config value that must be positive and finite."""

    numeric_value = float(value)
    if not np.isfinite(numeric_value) or numeric_value <= 0.0:
        raise ValueError(f"{field_name} must be a positive finite number.")
    return numeric_value


@dataclass(frozen=True)
class UnitPolicy:
    """Unit convention selected for one direct canonical build.

    Parameters
    ----------
    unit_convention:
        Public convention label.  The first implementation supports the two
        conventions already used by this repository: ``legacy_fixed_kpc`` and
        ``h_units_v1``.
    h_ref:
        Reference Hubble parameter used when ``unit_convention`` is
        ``h_units_v1``.  It is still carried for fixed-kpc builds so payload
        metadata has one consistent shape.
    """

    unit_convention: str
    h_ref: float = 0.7

    def __post_init__(self) -> None:
        """Normalize and validate the selected unit convention."""

        normalized = _normalize_required_text(self.unit_convention, "unit_convention")
        if normalized not in SUPPORTED_UNIT_CONVENTIONS:
            raise ValueError(
                f"unit_convention must be one of {sorted(SUPPORTED_UNIT_CONVENTIONS)}, got {normalized!r}."
            )
        object.__setattr__(self, "unit_convention", normalized)
        object.__setattr__(self, "h_ref", _require_positive_finite(self.h_ref, "h_ref"))


@dataclass(frozen=True)
class ProfilePolicy:
    """Tracer-profile branch selected for preparing lens-level derived grids."""

    profile_name: str

    def __post_init__(self) -> None:
        """Reject unsupported profile labels before numerical builders run."""

        normalized = _normalize_required_text(self.profile_name, "profile_name").lower()
        if normalized not in SUPPORTED_PROFILE_NAMES:
            raise ValueError(f"profile_name must be one of {sorted(SUPPORTED_PROFILE_NAMES)}, got {normalized!r}.")
        object.__setattr__(self, "profile_name", normalized)


@dataclass(frozen=True)
class MassDefinitionPolicy:
    """Enclosed-mass definition used for mass grids and velocity grids."""

    mass_definition_label: str
    mass_radius_kpc: float

    def __post_init__(self) -> None:
        """Validate the public label and physical radius together."""

        object.__setattr__(
            self,
            "mass_definition_label",
            _normalize_required_text(self.mass_definition_label, "mass_definition_label"),
        )
        object.__setattr__(
            self,
            "mass_radius_kpc",
            _require_positive_finite(self.mass_radius_kpc, "mass_radius_kpc"),
        )


@dataclass(frozen=True)
class AperturePolicyRef:
    """Semantic wrapper around a concrete :class:`AperturePolicy`.

    The low-level Jeans code only needs geometry.  The canonical dataset also
    needs to know *what the geometry means*, for example whether it describes a
    slit observation, a fibre observation, or a within-effective-radius
    quantity.  This wrapper keeps those labels attached to the geometry.
    """

    observation_flavor: str
    sigma_definition: str
    aperture_policy: AperturePolicy

    def __post_init__(self) -> None:
        """Validate the semantic labels while delegating geometry validation."""

        object.__setattr__(
            self,
            "observation_flavor",
            _normalize_required_text(self.observation_flavor, "observation_flavor").lower(),
        )
        object.__setattr__(
            self,
            "sigma_definition",
            _normalize_required_text(self.sigma_definition, "sigma_definition").lower(),
        )
        if not isinstance(self.aperture_policy, AperturePolicy):
            raise TypeError("aperture_policy must be an AperturePolicy instance.")


@dataclass(frozen=True)
class SigmaPolicy:
    """Policy for resolving trusted velocity-dispersion measurements.

    Parameters
    ----------
    source_type:
        Logical measurement source, such as ``catalog_columns`` or
        ``ppxf_results_adapter``.
    missing_policy:
        ``fail`` means every catalog lens must receive at least one trusted
        measurement.  ``num_sigma_zero`` keeps unmatched lenses in the sample
        and marks them as sigma-free.
    max_observations_per_lens:
        Upper bound supported by the canonical likelihood arrays.  The current
        science contract accepts one or two observations per lens.
    error_column:
        Optional upstream uncertainty column selected by adapters.
    trust_catalog_sigma:
        Explicit guard for modes that intentionally use catalog sigma columns.
        CMASS-style summary readers should leave this false.
    """

    source_type: str
    missing_policy: str = "fail"
    max_observations_per_lens: int = 2
    error_column: str | None = None
    trust_catalog_sigma: bool = False

    def __post_init__(self) -> None:
        """Normalize policy text and reject resolver choices the writer cannot store."""

        object.__setattr__(self, "source_type", _normalize_required_text(self.source_type, "source_type"))

        normalized_missing_policy = _normalize_required_text(self.missing_policy, "missing_policy").lower()
        if normalized_missing_policy not in SUPPORTED_MISSING_POLICIES:
            raise ValueError(
                f"missing_policy must be one of {sorted(SUPPORTED_MISSING_POLICIES)}, "
                f"got {normalized_missing_policy!r}."
            )
        object.__setattr__(self, "missing_policy", normalized_missing_policy)

        max_count = int(self.max_observations_per_lens)
        if max_count < 1 or max_count > 2:
            raise ValueError("max_observations_per_lens must be 1 or 2 for the canonical sigma slots.")
        object.__setattr__(self, "max_observations_per_lens", max_count)

        if self.error_column is not None:
            object.__setattr__(
                self,
                "error_column",
                _normalize_required_text(self.error_column, "error_column"),
            )


__all__ = [
    "AperturePolicyRef",
    "MassDefinitionPolicy",
    "ProfilePolicy",
    "SigmaPolicy",
    "UnitPolicy",
]
