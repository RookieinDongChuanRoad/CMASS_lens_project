"""Canonical-data capability audit for Sonnenfeld 2024 SLACS.

Sonnenfeld needs more than the CMASS likelihood inputs.  In particular, its
selection function uses a velocity-dispersion proxy for `theta_E_est`, so the
normalization integral needs a population-level sigma-unit grid.  This module
keeps that startup contract explicit while the numerical likelihood remains
disabled.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from ....canonical_dataset import (
    CAPABILITY_LENSING_CROSS_SECTION_THETA_GAMMA_V1,
    CAPABILITY_LENSING_MASS_GRIDS_V1,
    CAPABILITY_LENS_OBSERVATIONS_V1,
    CAPABILITY_VELOCITY_DISPERSION_PER_LENS_S2_V1,
    CAPABILITY_VELOCITY_DISPERSION_POPULATION_SIGMA_UNIT_V1,
    CanonicalInferenceDataset,
)


REQUIRED_CAPABILITIES: tuple[str, ...] = (
    CAPABILITY_LENS_OBSERVATIONS_V1,
    CAPABILITY_LENSING_MASS_GRIDS_V1,
    CAPABILITY_LENSING_CROSS_SECTION_THETA_GAMMA_V1,
    CAPABILITY_VELOCITY_DISPERSION_PER_LENS_S2_V1,
    CAPABILITY_VELOCITY_DISPERSION_POPULATION_SIGMA_UNIT_V1,
)


@dataclass(frozen=True)
class SonnenfeldCapabilityAudit:
    """Result of checking one canonical dataset against Sonnenfeld needs."""

    available_capabilities: tuple[str, ...]
    required_capabilities: tuple[str, ...]
    missing_capabilities: tuple[str, ...]
    ready: bool
    blocking_reason: str


def _capability_set(source: Iterable[str] | CanonicalInferenceDataset) -> frozenset[str]:
    """Normalize either a dataset object or an iterable of capability names."""

    if isinstance(source, CanonicalInferenceDataset):
        return frozenset(source.metadata.capabilities)
    return frozenset(str(capability) for capability in source)


def audit_capabilities(source: Iterable[str] | CanonicalInferenceDataset) -> SonnenfeldCapabilityAudit:
    """
    Check whether canonical data can support a future Sonnenfeld runtime.

    The audit is intentionally capability-level.  Formula correctness, Table-1
    constants, and reference-code numerical equivalence are separate model
    tests that should be added only when the likelihood is implemented.
    """

    available = _capability_set(source)
    missing = tuple(capability for capability in REQUIRED_CAPABILITIES if capability not in available)
    if not missing:
        blocking_reason = ""
    elif CAPABILITY_VELOCITY_DISPERSION_POPULATION_SIGMA_UNIT_V1 in missing:
        blocking_reason = (
            "Sonnenfeld selection requires population_sigma_unit to build the "
            "velocity-dispersion proxy theta_E_est during normalization."
        )
    else:
        blocking_reason = (
            "Sonnenfeld canonical input is missing required lens observations, "
            "lensing grids, cross-section grids, or per-lens sigma grids."
        )
    return SonnenfeldCapabilityAudit(
        available_capabilities=tuple(sorted(available)),
        required_capabilities=REQUIRED_CAPABILITIES,
        missing_capabilities=missing,
        ready=not missing,
        blocking_reason=blocking_reason,
    )


__all__ = [
    "REQUIRED_CAPABILITIES",
    "SonnenfeldCapabilityAudit",
    "audit_capabilities",
]

