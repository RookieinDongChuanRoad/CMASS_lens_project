"""Prepare resolved lens records for derived-grid builders.

The resolver stage already decided which sigma rows belong to which lens.  This
module performs the physical and unit normalization that the numerical builders
need:

- compute or preserve the lens-source ``Sigma_crit``
- resolve a concrete aperture policy
- attach explicit active-unit stellar-mass and size scalars
- carry the prepared values forward in ``PreparedLensRecord``

It deliberately avoids grid generation.  The only output is a new prepared
record, ready for mass-grid and Jeans builders.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import astropy.units as u
from astropy.constants import G, c

from statistical_sl.data_preparation.direct_pipeline.policies import AperturePolicyRef, UnitPolicy
from statistical_sl.data_preparation.direct_pipeline.records import BaseLensRecord, PreparedLensRecord
from statistical_sl.data_preparation.models import AperturePolicy
from statistical_sl.data_preparation.physics.jeans import COSMOLOGY, kpc_per_arcsec
from statistical_sl.core.unit_conventions import logMstar_h2_from_legacy, logRe_hinv_from_legacy


def sigma_critical_surface_density(z_lens: float, z_source: float) -> float:
    """Compute the lensing critical surface density in ``Msun / kpc^2``."""

    angular_diameter_lens = COSMOLOGY.angular_diameter_distance(z_lens)
    angular_diameter_source = COSMOLOGY.angular_diameter_distance(z_source)
    angular_diameter_lens_source = COSMOLOGY.angular_diameter_distance_z1z2(z_lens, z_source)
    sigma_crit = (
        (c**2 / (4.0 * math.pi * G))
        * angular_diameter_source
        / (angular_diameter_lens * angular_diameter_lens_source)
    )
    return float(sigma_crit.to(u.Msun / u.kpc**2).value)


def _validate_aperture_geometry_consistency(observations, lens_id: str) -> AperturePolicy | None:
    """Build one aperture policy from explicit measurement-row geometry."""

    explicit_rows = [observation for observation in observations if observation.aperture_shape is not None]
    if not explicit_rows:
        return None

    first = explicit_rows[0]
    for observation in explicit_rows[1:]:
        if (
            observation.aperture_shape != first.aperture_shape
            or observation.aperture_width_arcsec != first.aperture_width_arcsec
            or observation.aperture_height_arcsec != first.aperture_height_arcsec
            or observation.aperture_radius_arcsec != first.aperture_radius_arcsec
            or observation.seeing_fwhm_arcsec != first.seeing_fwhm_arcsec
        ):
            raise ValueError(f"{lens_id} has inconsistent explicit aperture metadata across sigma observations.")

    if len(explicit_rows) != len(tuple(observations)):
        raise ValueError(
            f"{lens_id} cannot mix explicit and implicit aperture metadata without a dataset-level default."
        )

    if first.aperture_shape == "rectangular":
        return AperturePolicy.rectangular(
            width_arcsec=float(first.aperture_width_arcsec),
            height_arcsec=float(first.aperture_height_arcsec),
            seeing_fwhm_arcsec=float(first.seeing_fwhm_arcsec),
        )
    if first.aperture_shape == "circular":
        return AperturePolicy.circular(
            radius_arcsec=float(first.aperture_radius_arcsec),
            seeing_fwhm_arcsec=float(first.seeing_fwhm_arcsec),
        )
    raise ValueError(f"{lens_id} has unsupported aperture_shape={first.aperture_shape!r}.")


def _resolve_aperture_policy(
    base_lens: BaseLensRecord,
    sigma_observations,
    aperture_policy_ref: AperturePolicyRef | None,
) -> tuple[AperturePolicy, str, str]:
    """Resolve a concrete aperture policy plus its semantic labels."""

    row_policy = _validate_aperture_geometry_consistency(sigma_observations, base_lens.lens_id)
    if row_policy is not None:
        if aperture_policy_ref is not None:
            return row_policy, aperture_policy_ref.observation_flavor, aperture_policy_ref.sigma_definition
        return row_policy, "measurement_row", "observed_aperture"

    if aperture_policy_ref is None:
        raise ValueError(
            f"{base_lens.lens_id} has no explicit aperture metadata and no dataset-level default aperture policy."
        )
    return aperture_policy_ref.aperture_policy, aperture_policy_ref.observation_flavor, aperture_policy_ref.sigma_definition


def _resolve_physical_scales(base_lens: BaseLensRecord, unit_policy: UnitPolicy) -> tuple[float, float, float, float]:
    """Return physical Einstein/size scales and the active-unit log scalars."""

    physical_kpc_per_arcsec = kpc_per_arcsec(base_lens.z_lens)

    theta_ein_kpc = base_lens.theta_ein_kpc
    if theta_ein_kpc is None:
        theta_ein_kpc = float(base_lens.theta_ein_arcsec * physical_kpc_per_arcsec)

    effective_radius_kpc = base_lens.effective_radius_kpc
    if effective_radius_kpc is None:
        effective_radius_kpc = float(base_lens.effective_radius_arcsec * physical_kpc_per_arcsec)

    if unit_policy.unit_convention == "h_units_v1":
        active_log_stellar_mass = float(logMstar_h2_from_legacy(base_lens.log_stellar_mass, h_ref=unit_policy.h_ref))
        active_log_effective_radius = float(
            logRe_hinv_from_legacy(math.log10(effective_radius_kpc), h_ref=unit_policy.h_ref)
        )
    else:
        active_log_stellar_mass = float(base_lens.log_stellar_mass)
        active_log_effective_radius = float(math.log10(effective_radius_kpc))

    return theta_ein_kpc, effective_radius_kpc, active_log_stellar_mass, active_log_effective_radius


def _prepare_single_record(
    record: PreparedLensRecord,
    *,
    unit_policy: UnitPolicy,
    aperture_policy_ref: AperturePolicyRef | None,
) -> PreparedLensRecord:
    """Return one fully prepared lens record."""

    theta_ein_kpc, effective_radius_kpc, active_log_stellar_mass, active_log_effective_radius = _resolve_physical_scales(
        record.base_lens,
        unit_policy,
    )
    sigma_crit = record.sigma_crit
    if sigma_crit is None:
        sigma_crit = sigma_critical_surface_density(record.base_lens.z_lens, record.base_lens.z_source)

    aperture_policy, observation_flavor, sigma_definition = _resolve_aperture_policy(
        record.base_lens,
        record.sigma_observations,
        aperture_policy_ref,
    )

    return PreparedLensRecord(
        base_lens=record.base_lens,
        sigma_observations=record.sigma_observations,
        sigma_crit=sigma_crit,
        aperture_policy=aperture_policy,
        observation_flavor=observation_flavor,
        sigma_definition=sigma_definition,
        unit_convention=unit_policy.unit_convention,
        h_ref=unit_policy.h_ref,
        theta_ein_kpc=theta_ein_kpc,
        effective_radius_kpc=effective_radius_kpc,
        active_log_stellar_mass=active_log_stellar_mass,
        active_log_effective_radius=active_log_effective_radius,
        preparation_metadata={
            **dict(record.preparation_metadata),
            "prepared_by": "lens_preparer",
        },
    )


def prepare_lens_records(
    records: Sequence[PreparedLensRecord],
    *,
    unit_policy: UnitPolicy,
    aperture_policy_ref: AperturePolicyRef | None = None,
) -> tuple[PreparedLensRecord, ...]:
    """Prepare one sequence of sigma-resolved lens records for later builders."""

    return tuple(
        _prepare_single_record(
            record,
            unit_policy=unit_policy,
            aperture_policy_ref=aperture_policy_ref,
        )
        for record in records
    )


__all__ = [
    "prepare_lens_records",
    "sigma_critical_surface_density",
]
