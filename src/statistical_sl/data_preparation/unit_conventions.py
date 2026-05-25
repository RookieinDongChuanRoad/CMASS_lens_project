"""Data-preparation access to shared unit-convention helpers.

The unit-convention contract now lives in :mod:`statistical_sl.core` so every
workflow stage uses the same h-unit algebra.  This module keeps data-prep
imports local while still sourcing the actual contract from ``core``.
"""

from __future__ import annotations

from statistical_sl.core.unit_conventions import (
    H_UNITS_V1,
    LEGACY_FIXED_KPC,
    Sunit_hinv_from_fixed_kpc,
    logMstar_h2_from_legacy,
    logRe_hinv_from_legacy,
    logSigmaStar_from_h_units,
    mR_hinv_from_fixed_kpc,
)

__all__ = [
    "H_UNITS_V1",
    "LEGACY_FIXED_KPC",
    "Sunit_hinv_from_fixed_kpc",
    "logMstar_h2_from_legacy",
    "logRe_hinv_from_legacy",
    "logSigmaStar_from_h_units",
    "mR_hinv_from_fixed_kpc",
]
