"""
Backward-compatible normalization wrappers.

The production implementation now lives in `kernels/normalization.py` and is
fed by the compiled array context. This module remains as the public wrapper
layer so existing callers and tests do not need to know about the kernel
package or the internal random-basis representation.
"""

from __future__ import annotations

import numpy as np

from .compiled_context import build_random_basis
from .kernels.normalization import normalization_mc_numba
from .mass_definition import MassDefinition, get_mass_definition
from .types import HyperParams, ProfileSpec, RandomBasis


def apply_normalization_guard(z_norm: float) -> float:
    """Reject parameter vectors whose normalization violates the hard cutoff."""

    return -np.inf if z_norm <= 1.0e-10 else 0.0


def estimate_normalization(
    hyper_params: HyperParams,
    profile_spec: ProfileSpec,
    random_basis: RandomBasis,
    cosmology,
    cross_section_grid,
    mass_definition: MassDefinition | None = None,
) -> float:
    """
    Public wrapper around the production normalization kernel.

    Keeping this wrapper allows legacy tests and helper code to ask for a
    standalone normalization value without having to build the full compiled
    model object first.
    """

    selected_mass_definition = mass_definition or get_mass_definition(5)
    return float(
        normalization_mc_numba(
            theta=hyper_params.to_array(),
            base_normals=random_basis.base_normals,
            cs_gamma_grid=cross_section_grid.gamma_grid,
            cs_over_theta=cross_section_grid.cs_over_theta_ein,
            z_grid=np.asarray(cosmology.z_table, dtype=np.float64),
            chi_kpc_grid=np.asarray(cosmology.comoving_distance_table_mpc * 1000.0, dtype=np.float64),
            mu_d=0.558,
            sigma_d=0.085,
            mass_function_loc=profile_spec.mass_function_loc,
            mass_function_scale=profile_spec.mass_function_scale,
            mass_function_alpha=profile_spec.mass_function_alpha,
            mu_r0=profile_spec.mu_r0,
            beta_r=profile_spec.beta_r,
            sigma_r=profile_spec.sigma_r,
            nu_r=profile_spec.nu_r if profile_spec.nu_r is not None else 0.0,
            use_sersic_index=1 if profile_spec.uses_observed_n_in_likelihood else 0,
            n_fixed=profile_spec.fixed_n if profile_spec.fixed_n is not None else 4.0,
            mu_n0=profile_spec.mu_n0 if profile_spec.mu_n0 is not None else 0.0,
            beta_n=profile_spec.beta_n if profile_spec.beta_n is not None else 0.0,
            sigma_n=profile_spec.sigma_n if profile_spec.sigma_n is not None else 0.0,
            gamma_trunc_low=1.2,
            gamma_trunc_high=2.8,
            mass_radius_kpc=float(selected_mass_definition.radius_kpc),
        )
    )


__all__ = [
    "apply_normalization_guard",
    "build_random_basis",
    "estimate_normalization",
    "normalization_mc_numba",
]
