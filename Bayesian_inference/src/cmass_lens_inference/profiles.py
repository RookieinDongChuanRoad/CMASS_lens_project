"""
Profile-specific constants and lookup helpers.

The requirements are explicit that `devauc` and `sersic` share the same
statistical skeleton. This module is therefore the only place where their
fixed constants and field aliases are allowed to diverge.
"""

from __future__ import annotations

from .types import ProfileSpec


def build_profile_spec(profile_name: str) -> ProfileSpec:
    """Return the fixed profile specification for the requested branch."""

    normalized_name = profile_name.strip().lower()
    if normalized_name == "devauc":
        return ProfileSpec(
            name="devauc",
            fixed_n=4.0,
            uses_observed_n_in_likelihood=False,
            observation_field_aliases={
                "stellar_mass": ("logmchab_deV", "logmchab"),
                "stellar_mass_error": ("logmchab_err",),
                "effective_radius_arcsec": ("reff_deV", "re_arcsec"),
                "einstein_radius_arcsec": ("rein_arcsec",),
                "nser": ("nser",),
            },
            mass_function_loc=11.252,
            mass_function_scale=0.202,
            mass_function_alpha=10.0**0.17,
            mu_r0=0.774,
            beta_r=0.977,
            sigma_r=0.112,
            nu_r=None,
            mu_n0=None,
            beta_n=None,
            sigma_n=None,
        )
    if normalized_name == "sersic":
        return ProfileSpec(
            name="sersic",
            fixed_n=None,
            uses_observed_n_in_likelihood=True,
            observation_field_aliases={
                "stellar_mass": ("logmchab",),
                "stellar_mass_error": ("logmchab_err",),
                "effective_radius_arcsec": ("re_arcsec",),
                "einstein_radius_arcsec": ("rein_arcsec",),
                "nser": ("nser",),
            },
            mass_function_loc=11.249,
            mass_function_scale=0.285,
            mass_function_alpha=10.0**0.43,
            mu_r0=0.817,
            beta_r=1.184,
            sigma_r=0.133,
            nu_r=0.383,
            mu_n0=0.704,
            beta_n=0.464,
            sigma_n=0.163,
        )
    raise ValueError(f"Unsupported profile: {profile_name}")
