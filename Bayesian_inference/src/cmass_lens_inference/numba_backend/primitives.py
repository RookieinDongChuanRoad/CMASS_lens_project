"""Compatibility re-export for Numba kernel primitives.

New production code should import from ``numba_backend.kernels`` modules by
numerical responsibility.  This module remains as a stable compatibility layer
for older tests and transitional imports.
"""

from __future__ import annotations

from .kernels.constants import C_KM_S, LOG10_2PI, LOG10_4
from .kernels.distributions import (
    normal_pdf,
    phi_standard,
    skewnorm_sample,
    truncated_normal_pdf,
    truncated_normal_pdf_nonneg,
    truncnorm_sample,
)
from .kernels.integration import trapezoid_1d
from .kernels.interpolation import (
    interp_cross_section_theta_gamma,
    interp_sigma_unit_clip,
)
from .kernels.lensing import theta_ein_arcsec
from .kernels.population import linear_size_relation_mean
from .kernels.selection import p_find

__all__ = [
    "C_KM_S",
    "LOG10_2PI",
    "LOG10_4",
    "interp_cross_section_theta_gamma",
    "interp_sigma_unit_clip",
    "linear_size_relation_mean",
    "normal_pdf",
    "p_find",
    "phi_standard",
    "skewnorm_sample",
    "theta_ein_arcsec",
    "trapezoid_1d",
    "truncated_normal_pdf",
    "truncated_normal_pdf_nonneg",
    "truncnorm_sample",
]
