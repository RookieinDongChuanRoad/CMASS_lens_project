"""Numerical constants shared by Numba kernels."""

from __future__ import annotations

import math

from astropy.constants import G, c
import astropy.units as u


SQRT2 = math.sqrt(2.0)
SQRT2PI = math.sqrt(2.0 * math.pi)
LOG10_2PI = math.log10(2.0 * math.pi)
LOG10_4 = math.log10(4.0)
C_KM_S = float(c.to("km/s").value)
G_KPC_KMS2_MSUN = float(G.to(u.kpc * u.km**2 / (u.s**2 * u.Msun)).value)

__all__ = [
    "C_KM_S",
    "G_KPC_KMS2_MSUN",
    "LOG10_2PI",
    "LOG10_4",
    "SQRT2",
    "SQRT2PI",
]
