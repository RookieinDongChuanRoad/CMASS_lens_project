"""Unit-convention helpers shared by Statistical_SL workflow stages.

Data preparation is currently the main writer of h-unit products, while
inference and posterior-predictive code validate and consume those products.
Keeping the h-dependent algebra in ``core`` prevents each stage from
re-deriving the same conversion rules.
"""

from __future__ import annotations

import math

import numpy as np


H_UNITS_V1 = "h_units_v1"
LEGACY_FIXED_KPC = "legacy_fixed_kpc"


def _validate_h_ref(h_ref: float) -> float:
    """Return a positive finite reference `h`, or raise a clear error."""

    normalized = float(h_ref)
    if not math.isfinite(normalized) or normalized <= 0.0:
        raise ValueError(f"h_ref must be a positive finite value, got {h_ref!r}.")
    return normalized


def logMstar_h2_from_legacy(log_mstar_msun: np.ndarray | float, *, h_ref: float) -> np.ndarray:
    """
    Convert legacy physical stellar mass logs into `h^-2 Msun` logs.

    A value stored as `log10[M*/Msun]` becomes
    `log10[M*/(h^-2 Msun)] = log10[M*/Msun] + 2 log10(h_ref)`.
    """

    h_value = _validate_h_ref(h_ref)
    return np.asarray(np.asarray(log_mstar_msun, dtype=float) + 2.0 * math.log10(h_value), dtype=float)


def logRe_hinv_from_legacy(log_re_kpc: np.ndarray | float, *, h_ref: float) -> np.ndarray:
    """
    Convert legacy physical-size logs into `h^-1 kpc` logs.

    A physical size `Re[kpc]` is represented in h-units as
    `Re/(h^-1 kpc) = Re * h_ref`, hence the additive `log10(h_ref)` shift.
    """

    h_value = _validate_h_ref(h_ref)
    return np.asarray(np.asarray(log_re_kpc, dtype=float) + math.log10(h_value), dtype=float)


def logSigmaStar_from_h_units(
    log_mstar_h2: np.ndarray | float,
    log_re_hinv: np.ndarray | float,
) -> np.ndarray:
    """
    Compute the effective stellar surface-density log from h-unit variables.

    The h powers cancel exactly:
    `M*/Re^2` has `(h^-2 Msun)/(h^-1 kpc)^2`, so the resulting surface density
    is the same physical `Msun/kpc^2` quantity used by the legacy model.
    """

    return np.asarray(
        np.asarray(log_mstar_h2, dtype=float)
        - math.log10(2.0 * math.pi)
        - 2.0 * np.asarray(log_re_hinv, dtype=float),
        dtype=float,
    )


def mR_hinv_from_fixed_kpc(
    log_mass_fixed_kpc: np.ndarray | float,
    gamma: np.ndarray | float,
    *,
    h_ref: float,
) -> np.ndarray:
    """
    Convert fixed-kpc enclosed-mass logs to the h-units aperture/mass contract.

    The aperture changes from `R kpc` to `R h^-1 kpc` and the mass unit changes
    from `Msun` to `h^-1 Msun`, giving the exact power-law migration term
    `-(2 - gamma) log10(h_ref)`.
    """

    h_value = _validate_h_ref(h_ref)
    return np.asarray(
        np.asarray(log_mass_fixed_kpc, dtype=float)
        - (2.0 - np.asarray(gamma, dtype=float)) * math.log10(h_value),
        dtype=float,
    )


def Sunit_hinv_from_fixed_kpc(
    sigma_unit_fixed_kpc: np.ndarray | float,
    gamma: np.ndarray | float,
    *,
    h_ref: float,
) -> np.ndarray:
    """
    Convert fixed-kpc Jeans sigma-unit tables to the h-units mass definition.

    Because sigma-unit tables store `sigma^2 / 10**m_R`, they scale opposite
    to the enclosed-mass normalization:
    `Sunit_h = Sunit_fixed * h_ref**(2 - gamma)`.
    """

    h_value = _validate_h_ref(h_ref)
    return np.asarray(
        np.asarray(sigma_unit_fixed_kpc, dtype=float)
        * np.power(h_value, 2.0 - np.asarray(gamma, dtype=float)),
        dtype=float,
    )
