"""Paper constants and sampled-parameter declarations for Sonnenfeld 2024.

The constants in this module are taken from the paper/source tables in physical
stellar-mass coordinates, `m* = log10(M*/Msun)`.  Runtime preprocessing must
convert location-like mass constants if a future implementation consumes a
different canonical mass coordinate, such as `h_units_v1`.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from ....model_interfaces import ParameterSpec


MSTAR_PIVOT_PHYSICAL = 11.3
MBAR_PHYSICAL = 11.06
PARENT_ALPHA = -1.207
SIZE_MU0_PHYSICAL = 7.55
SIZE_MU1_PHYSICAL = -1.84
SIZE_MU2_PHYSICAL = 0.11
SIZE_SCATTER = 0.112
TRUNCATION_MASS_POLYNOMIAL_COEFFICIENTS = np.asarray(
    [9.388, 7.855, 48.34, -312.5, 535.7, -274.2],
    dtype=np.float64,
)
TRUNCATION_MASS_SCATTER = 0.0007
GAMMA_TRUNC_LOW = 1.2
GAMMA_TRUNC_HIGH = 2.8
PARENT_ZD_MIN = 0.05
PARENT_ZD_MAX = 0.95
PARENT_MSTAR_MIN_OFFSET = -0.8
PARENT_MSTAR_MAX_OFFSET = 1.0
SIGMA_PROXY_FRACTIONAL_SCATTER = 0.0625

INTERNAL_PARAMETER_NAMES: tuple[str, ...] = (
    "mu5_0",
    "beta5",
    "xi5",
    "sigma5",
    "mu_gamma_0",
    "beta_gamma",
    "xi_gamma",
    "sigma_gamma",
    "mu_zs",
    "sigma_zs",
    "theta0",
    "loga",
)
PUBLIC_PARAMETER_NAMES: tuple[str, ...] = INTERNAL_PARAMETER_NAMES

DEFAULT_BOX_PRIOR_BOUNDS_BY_INTERNAL_NAME: dict[str, tuple[float, float]] = {
    "mu5_0": (10.5, 12.2),
    "beta5": (-3.0, 3.0),
    "xi5": (-3.0, 3.0),
    "sigma5": (1.0e-2, 0.3),
    "mu_gamma_0": (1.2, 2.8),
    "beta_gamma": (-3.0, 3.0),
    "xi_gamma": (-3.0, 3.0),
    "sigma_gamma": (1.0e-2, 0.8),
    "mu_zs": (0.0, 2.0),
    "sigma_zs": (1.0e-3, 1.0),
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


class SonnenfeldTheta(NamedTuple):
    """Named view over the fixed 12D Sonnenfeld parameter vector."""

    mu5_0: jnp.ndarray
    beta5: jnp.ndarray
    xi5: jnp.ndarray
    sigma5: jnp.ndarray
    mu_gamma_0: jnp.ndarray
    beta_gamma: jnp.ndarray
    xi_gamma: jnp.ndarray
    sigma_gamma: jnp.ndarray
    mu_zs: jnp.ndarray
    sigma_zs: jnp.ndarray
    theta0: jnp.ndarray
    loga: jnp.ndarray


def unpack_theta(theta: jnp.ndarray) -> SonnenfeldTheta:
    """Unpack the fixed 12D Sonnenfeld parameter vector."""

    return SonnenfeldTheta(
        mu5_0=theta[0],
        beta5=theta[1],
        xi5=theta[2],
        sigma5=theta[3],
        mu_gamma_0=theta[4],
        beta_gamma=theta[5],
        xi_gamma=theta[6],
        sigma_gamma=theta[7],
        mu_zs=theta[8],
        sigma_zs=theta[9],
        theta0=theta[10],
        loga=theta[11],
    )


def validate_theta(
    theta: jnp.ndarray,
    theta_parts: SonnenfeldTheta,
    context: object,
    static: dict[str, int],
) -> jnp.ndarray:
    """
    Return the differentiable validity mask for one Sonnenfeld theta vector.

    Box-prior bounds are checked on the host before this hook runs.  This hook
    only enforces constraints that matter inside the continuous likelihood,
    such as positive scatters and a normalizable non-negative source-redshift
    distribution.
    """

    del context, static
    return (
        (theta.shape[0] == len(INTERNAL_PARAMETER_NAMES))
        & (theta_parts.sigma5 > 0.0)
        & (theta_parts.sigma_gamma > 0.0)
        & (theta_parts.sigma_zs > 0.0)
    )


def shift_physical_mass_location_to_hunits(value: float, h_ref: float) -> float:
    """
    Convert a physical log-stellar-mass location to the h-units coordinate.

    For `m*_h = log10(M*/(h^-2 Msun))`, every location-like stellar-mass
    constant shifts by `2 log10(h_ref)`.  Keeping this helper here prevents the
    Sonnenfeld runtime from silently mixing physical Table-1 constants with a
    canonical h-dependent data coordinate.
    """

    return float(value) + 2.0 * float(np.log10(h_ref))


def active_size_relation_coefficients(
    *,
    h_ref: float,
    unit_convention: str,
) -> tuple[float, float, float]:
    """
    Return Equation 29 coefficients in the active size/mass coordinate.

    The paper relation is written in physical coordinates:

    ``log10(R_e/kpc) = a + b m_phys + c m_phys^2``.

    The explicit hunit variant evaluates the same physical relation on
    canonical h-dependent coordinates:

    - ``m_h = m_phys + 2 log10(h_ref)``
    - ``r_h = log10(R_e/(h^-1 kpc)) = r_phys + log10(h_ref)``

    Substituting ``m_phys = m_h - 2 log10(h_ref)`` gives a new quadratic in
    ``m_h``.  Keeping this algebra in one helper makes hunit shifts auditable
    and prevents JAX kernels from carrying paper/native conversion branches.
    """

    if unit_convention == "legacy_fixed_kpc":
        return SIZE_MU0_PHYSICAL, SIZE_MU1_PHYSICAL, SIZE_MU2_PHYSICAL
    if unit_convention == "h_units_v1":
        log10_h = float(np.log10(h_ref))
        return (
            SIZE_MU0_PHYSICAL
            - 2.0 * SIZE_MU1_PHYSICAL * log10_h
            + 4.0 * SIZE_MU2_PHYSICAL * log10_h * log10_h
            + log10_h,
            SIZE_MU1_PHYSICAL - 4.0 * SIZE_MU2_PHYSICAL * log10_h,
            SIZE_MU2_PHYSICAL,
        )
    raise ValueError(
        "Sonnenfeld size relation supports unit_convention "
        f"'legacy_fixed_kpc' or 'h_units_v1', got '{unit_convention}'."
    )


__all__ = [
    "DEFAULT_BOX_PRIOR_BOUNDS_BY_INTERNAL_NAME",
    "GAMMA_TRUNC_HIGH",
    "GAMMA_TRUNC_LOW",
    "INTERNAL_PARAMETER_NAMES",
    "MBAR_PHYSICAL",
    "MSTAR_PIVOT_PHYSICAL",
    "PARAMETER_SPECS",
    "PARENT_ALPHA",
    "PARENT_MSTAR_MAX_OFFSET",
    "PARENT_MSTAR_MIN_OFFSET",
    "PARENT_ZD_MAX",
    "PARENT_ZD_MIN",
    "PUBLIC_PARAMETER_NAMES",
    "SIGMA_PROXY_FRACTIONAL_SCATTER",
    "SIZE_MU0_PHYSICAL",
    "SIZE_MU1_PHYSICAL",
    "SIZE_MU2_PHYSICAL",
    "SIZE_SCATTER",
    "SonnenfeldTheta",
    "TRUNCATION_MASS_SCATTER",
    "TRUNCATION_MASS_POLYNOMIAL_COEFFICIENTS",
    "active_size_relation_coefficients",
    "shift_physical_mass_location_to_hunits",
    "unpack_theta",
    "validate_theta",
]
