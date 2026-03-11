"""
Centralized numerical kernels for the CMASS inference hot path.

The performance refactor deliberately moves reusable `numba` primitives and the
two production kernels into one package so future speed work has a single
obvious home instead of being scattered across wrapper modules.
"""

from .likelihood import log_likelihood_lenses_numba
from .normalization import normalization_mc_numba
from .primitives import (
    interp1d_clip,
    mu_r,
    normal_pdf,
    normal_ppf,
    p_find,
    skewnorm_pdf,
    skewnorm_sample,
    theta_ein_arcsec,
    truncnorm_sample,
    truncated_normal_pdf_nonneg,
)

__all__ = [
    "interp1d_clip",
    "log_likelihood_lenses_numba",
    "mu_r",
    "normal_pdf",
    "normal_ppf",
    "normalization_mc_numba",
    "p_find",
    "skewnorm_pdf",
    "skewnorm_sample",
    "theta_ein_arcsec",
    "truncnorm_sample",
    "truncated_normal_pdf_nonneg",
]
