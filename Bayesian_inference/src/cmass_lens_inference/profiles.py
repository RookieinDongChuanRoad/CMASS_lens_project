"""Compatibility profile-spec builder for CMASS profile branches.

The model-owned constants now live in ``models/cmass/constants.py``.  This
module stays as a thin compatibility layer because many readers and tests still
ask for a ``ProfileSpec`` by profile name.  It should not grow new scientific
defaults of its own.
"""

from __future__ import annotations

from .types import ProfileSpec


def build_profile_spec(profile_name: str) -> ProfileSpec:
    """Return the fixed profile specification for the requested branch.

    ``ProfileSpec`` is still the transport type expected by legacy readers and
    canonical validation helpers.  The numeric source, however, is the CMASS
    model package, so this wrapper simply copies those constants into the
    existing dataclass shape.
    """

    # Import lazily to avoid a package-initialization cycle: ``models`` imports
    # runtime adapters, runtime adapters import this compatibility module, and
    # this wrapper ultimately reads the CMASS model-owned constants.
    from .models.cmass.constants import get_cmass_profile_constants

    constants = get_cmass_profile_constants(profile_name)
    return ProfileSpec(
        name=constants.name,
        fixed_n=constants.fixed_n,
        uses_observed_n_in_likelihood=constants.uses_observed_n_in_likelihood,
        observation_field_aliases=dict(constants.observation_field_aliases),
        mass_function_loc=constants.mass_function_loc,
        mass_function_scale=constants.mass_function_scale,
        mass_function_alpha=constants.mass_function_alpha,
        mu_r0=constants.mu_r0,
        beta_r=constants.beta_r,
        sigma_r=constants.sigma_r,
        nu_r=constants.nu_r,
        mu_n0=constants.mu_n0,
        beta_n=constants.beta_n,
        sigma_n=constants.sigma_n,
    )
