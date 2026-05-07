"""Parameter declarations for the default CMASS model.

This component is the single source of truth for the fixed 11-dimensional
CMASS parameter vector.  Keeping the schema separate from the assembly file
makes it clear which names are scientific/internal (`mu5_0`) and which names
are public/config-facing (`mu5h_0`).
"""

from __future__ import annotations

import math
from typing import NamedTuple

from ....model_interfaces import ParameterSpec


INTERNAL_MASS_PARAMETER_NAMES: tuple[str, ...] = (
    "mu5_0",
    "beta5",
    "xi5",
    "sigma5",
)

SIGMA_STAR_DEPENDENT_GAMMA_PARAMETER_NAMES: tuple[str, ...] = (
    "mu_gamma_0",
    "beta_sigma_star_gamma",
    "sigma_gamma",
)

TAIL_PARAMETER_NAMES: tuple[str, ...] = (
    "mu_zs",
    "sigma_zs",
    "theta0",
    "loga",
)
INTERNAL_PARAMETER_NAMES: tuple[str, ...] = (
    INTERNAL_MASS_PARAMETER_NAMES
    + SIGMA_STAR_DEPENDENT_GAMMA_PARAMETER_NAMES
    + TAIL_PARAMETER_NAMES
)
PUBLIC_PARAMETER_NAMES: tuple[str, ...] = (
    ("mu5h_0", "beta5h", "xi5h", "sigma5h")
    + SIGMA_STAR_DEPENDENT_GAMMA_PARAMETER_NAMES
    + TAIL_PARAMETER_NAMES
)

DEFAULT_BOX_PRIOR_BOUNDS_BY_INTERNAL_NAME: dict[str, tuple[float, float]] = {
    "mu5_0": (9.0, 12.0),
    "beta5": (-3.0, 3.0),
    "xi5": (-3.0, 3.0),
    "sigma5": (1.0e-2, 0.2),
    "mu_gamma_0": (1.5, 2.5),
    "beta_sigma_star_gamma": (-3.0, 3.0),
    "sigma_gamma": (0.0, 0.5),
    "mu_zs": (1.0, 3.0),
    "sigma_zs": (0.0, 2.0),
    "theta0": (0.0, 3.0),
    "loga": (-1.0, 3.0),
}
PARAMETER_SPECS: tuple[ParameterSpec, ...] = tuple(
    ParameterSpec(
        internal_name=internal_name,
        public_name=public_name,
        bounds=DEFAULT_BOX_PRIOR_BOUNDS_BY_INTERNAL_NAME[internal_name],
    )
    for internal_name, public_name in zip(
        INTERNAL_PARAMETER_NAMES,
        PUBLIC_PARAMETER_NAMES,
        strict=True,
    )
)


class CMASSTheta(NamedTuple):
    """Named view over the fixed 11D CMASS parameter vector."""

    mu5_0: object
    beta5: object
    xi5: object
    sigma5: object
    mu_gamma_0: object
    beta_sigma_star_gamma: object
    sigma_gamma: object
    mu_zs: object
    sigma_zs: object
    theta0: object
    loga: object


def unpack_theta(theta) -> CMASSTheta:
    """
    Unpack the fixed 11D CMASS parameter vector.

    This helper is retained for reference/oracle utilities.  Production Numba
    kernels use their own typed scalar unpacker so importing this declaration
    module does not require JAX.
    """

    return CMASSTheta(
        mu5_0=theta[0],
        beta5=theta[1],
        xi5=theta[2],
        sigma5=theta[3],
        mu_gamma_0=theta[4],
        beta_sigma_star_gamma=theta[5],
        sigma_gamma=theta[6],
        mu_zs=theta[7],
        sigma_zs=theta[8],
        theta0=theta[9],
        loga=theta[10],
    )


def validate_theta(
    theta,
    theta_parts: CMASSTheta,
    context: object,
    static: dict[str, int],
) -> bool:
    """
    Return the differentiable validity mask for one CMASS theta vector.

    The backend combines this mask with likelihood and normalization validity.
    The context/static arguments are accepted for the common model-hook
    signature, even though the current parameter constraints are context-free.
    """

    del context, static
    if float(theta_parts.sigma_zs) <= 0.0:
        trunc_den = 0.0
    else:
        z0 = (0.0 - float(theta_parts.mu_zs)) / float(theta_parts.sigma_zs)
        trunc_den = 1.0 - 0.5 * (1.0 + math.erf(z0 / math.sqrt(2.0)))
    return (
        (theta.shape[0] == len(INTERNAL_PARAMETER_NAMES))
        and (float(theta_parts.sigma5) > 0.0)
        and (float(theta_parts.sigma_gamma) > 0.0)
        and (float(theta_parts.sigma_zs) > 0.0)
        and (trunc_den > 0.0)
    )


__all__ = [
    "CMASSTheta",
    "INTERNAL_PARAMETER_NAMES",
    "PARAMETER_SPECS",
    "PUBLIC_PARAMETER_NAMES",
    "SIGMA_STAR_DEPENDENT_GAMMA_PARAMETER_NAMES",
    "unpack_theta",
    "validate_theta",
]
