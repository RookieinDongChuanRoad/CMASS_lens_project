"""Fixed scientific constants for the default CMASS model.

This module is the model-owned source of CMASS constants.  Generic config and
runtime code may transport these values, but they should not independently
define CMASS scientific defaults.  Keeping the constants here makes later
audits answerable from the model package itself, matching the same ownership
pattern used by the Sonnenfeld model's ``paper_constants.py``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CMASSProfileConstants:
    """Fixed constants for one CMASS profile branch.

    The branch name, observed-field aliases, and Sersic-index behavior are
    compatibility metadata needed by existing readers.  The population fields
    are scientific constants used by the CMASS stellar-mass and size-relation
    terms, so their source of truth belongs in this model package.
    """

    name: str
    fixed_n: float | None
    uses_observed_n_in_likelihood: bool
    observation_field_aliases: dict[str, tuple[str, ...]]
    mass_function_loc: float
    mass_function_scale: float
    mass_function_alpha: float
    mu_r0: float
    beta_r: float
    sigma_r: float
    nu_r: float | None
    mu_n0: float | None
    beta_n: float | None
    sigma_n: float | None


@dataclass(frozen=True)
class FPPriorDefaults:
    """Default FP-prior constants for one concrete scientific model.

    ``enabled`` is intentionally not part of the defaults.  Enabling or
    disabling the prior remains a run configuration choice; this object only
    defines which numeric prior constants a model uses when the YAML section
    omits explicit overrides.
    """

    fit_mstar_min: float
    pivot_mstar: float
    fiducial_scatter: float
    scatter_error: float
    mu_v_prior: float
    mu_v_error: float
    beta_v_prior: float
    beta_v_error: float

    def to_config_defaults(self) -> dict[str, float]:
        """Return a plain mapping consumed by the generic config parser."""

        return {
            "fit_mstar_min": self.fit_mstar_min,
            "pivot_mstar": self.pivot_mstar,
            "fiducial_scatter": self.fiducial_scatter,
            "scatter_error": self.scatter_error,
            "mu_v_prior": self.mu_v_prior,
            "mu_v_error": self.mu_v_error,
            "beta_v_prior": self.beta_v_prior,
            "beta_v_error": self.beta_v_error,
        }


CMASS_STELLAR_MASS_PIVOT = 11.4
CMASS_LENS_REDSHIFT_MEAN = 0.558
CMASS_LENS_REDSHIFT_SCATTER = 0.085
CMASS_GAMMA_TRUNC_LOW = 1.2
CMASS_GAMMA_TRUNC_HIGH = 2.8
CMASS_NORMALIZATION_MIN_VALUE = 1.0e-10

DEVAUC_PROFILE_CONSTANTS = CMASSProfileConstants(
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

SERSIC_PROFILE_CONSTANTS = CMASSProfileConstants(
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

CMASS_PROFILE_CONSTANTS_BY_NAME = {
    DEVAUC_PROFILE_CONSTANTS.name: DEVAUC_PROFILE_CONSTANTS,
    SERSIC_PROFILE_CONSTANTS.name: SERSIC_PROFILE_CONSTANTS,
}

# These values are locked to the 2026-04-29 CMASS FP-enabled run snapshot.
# The fit cut and pivot are included so the whole FP-prior default contract has
# one source, even though the 2026-04-29 issue was driven by the six tighter
# prior terms below.
CMASS_FP_PRIOR_DEFAULTS_20260429 = FPPriorDefaults(
    fit_mstar_min=11.0,
    pivot_mstar=11.3,
    fiducial_scatter=0.075,
    scatter_error=0.003,
    mu_v_prior=2.34548,
    mu_v_error=0.00611,
    beta_v_prior=0.176,
    beta_v_error=0.011,
)


def get_cmass_profile_constants(profile_name: str) -> CMASSProfileConstants:
    """Return CMASS profile constants for a public profile name."""

    normalized_name = profile_name.strip().lower()
    try:
        return CMASS_PROFILE_CONSTANTS_BY_NAME[normalized_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported profile: {profile_name}") from exc


__all__ = [
    "CMASS_FP_PRIOR_DEFAULTS_20260429",
    "CMASS_GAMMA_TRUNC_HIGH",
    "CMASS_GAMMA_TRUNC_LOW",
    "CMASS_LENS_REDSHIFT_MEAN",
    "CMASS_LENS_REDSHIFT_SCATTER",
    "CMASS_NORMALIZATION_MIN_VALUE",
    "CMASS_PROFILE_CONSTANTS_BY_NAME",
    "CMASS_STELLAR_MASS_PIVOT",
    "CMASSProfileConstants",
    "DEVAUC_PROFILE_CONSTANTS",
    "FPPriorDefaults",
    "SERSIC_PROFILE_CONSTANTS",
    "get_cmass_profile_constants",
]
