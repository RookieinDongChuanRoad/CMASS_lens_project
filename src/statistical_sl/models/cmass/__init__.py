"""Default CMASS production model package.

The package keeps the public ``models.cmass`` import stable while separating
three model-owned responsibilities:

* ``assembly`` declares the scientific model contract.
* ``runtime`` builds the parameter-independent context from canonical data.
* ``posterior`` wires CMASS kernels into a posterior log-probability.
"""

from __future__ import annotations

from .assembly import (
    COMPONENTS,
    GAMMA_DISTRIBUTION_SIGMA_STAR_DEPENDENT,
    GAMMA_MODE_SIGMA_STAR_DEPENDENT_CODE,
    INTERNAL_PARAMETER_NAMES,
    MODEL_NAME,
    PARAMETER_SPECS,
    PARAMETERS,
    PUBLIC_PARAMETER_NAMES,
)
from .assembly import get_model_spec as _get_model_spec


def get_model_spec():
    """Return the public CMASS model specification from the assembly module."""

    return _get_model_spec()

__all__ = [
    "GAMMA_DISTRIBUTION_SIGMA_STAR_DEPENDENT",
    "GAMMA_MODE_SIGMA_STAR_DEPENDENT_CODE",
    "INTERNAL_PARAMETER_NAMES",
    "MODEL_NAME",
    "COMPONENTS",
    "PARAMETER_SPECS",
    "PARAMETERS",
    "PUBLIC_PARAMETER_NAMES",
    "get_model_spec",
]
