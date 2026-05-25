"""Assembly layer for the default CMASS lens-population model."""

from __future__ import annotations

from statistical_sl.inference.canonical_dataset import (
    CAPABILITY_LENSING_CROSS_SECTION_THETA_GAMMA_V1,
    CAPABILITY_LENSING_MASS_GRIDS_V1,
    CAPABILITY_LENS_OBSERVATIONS_V1,
    CAPABILITY_VELOCITY_DISPERSION_FP_WITHIN_RE_V1,
    CAPABILITY_VELOCITY_DISPERSION_PER_LENS_S2_V1,
)
from statistical_sl.models.components import (
    aggregate_optional_capabilities,
    aggregate_parameters,
    aggregate_required_capabilities,
)
from statistical_sl.models.components.lensing.cross_section import theta_gamma_cross_section_component
from statistical_sl.models.components.lensing.powerlaw import powerlaw_lensing_component
from statistical_sl.models.components.observations.lens_sample import lens_sample_component
from statistical_sl.models.components.population.aperture_mass_relation.gaussian_linear import (
    gaussian_linear_aperture_mass_component,
)
from statistical_sl.models.components.population.gamma_relation.sigma_star_linear import (
    sigma_star_linear_gamma_component,
)
from statistical_sl.models.components.population.size_relation.linear import linear_size_relation_component
from statistical_sl.models.components.population.source_redshift.truncated_nonnegative_gaussian import (
    truncated_nonnegative_gaussian_source_redshift_component,
)
from statistical_sl.models.components.population.stellar_mass_function.skewnormal import (
    skewnormal_stellar_mass_function_component,
)
from statistical_sl.models.components.selection.discovery_probability import discovery_probability_component
from statistical_sl.core.mass_definition import H_UNITS_V1
from statistical_sl.models.interfaces import ModelSpec, ParameterSpec
from .constants import CMASS_FP_PRIOR_DEFAULTS_20260429


GAMMA_MODE_SIGMA_STAR_DEPENDENT_CODE = 2
GAMMA_DISTRIBUTION_SIGMA_STAR_DEPENDENT = "sigma_star_dependent"
MODEL_NAME = "cmass"
MODEL_COMPONENT_KEY = "default"
MASS_APERTURE_KPC = 5

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
SOURCE_REDSHIFT_PARAMETER_NAMES: tuple[str, ...] = ("mu_zs", "sigma_zs")
DISCOVERY_PARAMETER_NAMES: tuple[str, ...] = ("theta0", "loga")
INTERNAL_PARAMETER_NAMES: tuple[str, ...] = (
    INTERNAL_MASS_PARAMETER_NAMES
    + SIGMA_STAR_DEPENDENT_GAMMA_PARAMETER_NAMES
    + SOURCE_REDSHIFT_PARAMETER_NAMES
    + DISCOVERY_PARAMETER_NAMES
)
PUBLIC_PARAMETER_NAMES: tuple[str, ...] = (
    ("mu5h_0", "beta5h", "xi5h", "sigma5h")
    + SIGMA_STAR_DEPENDENT_GAMMA_PARAMETER_NAMES
    + SOURCE_REDSHIFT_PARAMETER_NAMES
    + DISCOVERY_PARAMETER_NAMES
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
PARAMETER_SPEC_BY_INTERNAL_NAME: dict[str, ParameterSpec] = {
    parameter.internal_name: parameter for parameter in PARAMETER_SPECS
}

COMPONENTS = (
    lens_sample_component(
        required_context_fields=("zd", "zs", "sigma_obs", "sigma_err"),
        required_capabilities=(CAPABILITY_LENS_OBSERVATIONS_V1,),
    ),
    skewnormal_stellar_mass_function_component(
        required_context_fields=("base_normals", "mstar_integrand_base"),
    ),
    linear_size_relation_component(
        required_context_fields=("delta_r_grid",),
    ),
    gaussian_linear_aperture_mass_component(
        parameters=tuple(
            PARAMETER_SPEC_BY_INTERNAL_NAME[name] for name in INTERNAL_MASS_PARAMETER_NAMES
        ),
        required_context_fields=("base_normals", "mass_grid_int"),
        required_capabilities=(CAPABILITY_LENSING_MASS_GRIDS_V1,),
    ),
    sigma_star_linear_gamma_component(
        parameters=tuple(
            PARAMETER_SPEC_BY_INTERNAL_NAME[name]
            for name in SIGMA_STAR_DEPENDENT_GAMMA_PARAMETER_NAMES
        ),
        required_context_fields=("base_normals", "gamma_grid_int"),
    ),
    truncated_nonnegative_gaussian_source_redshift_component(
        parameters=tuple(
            PARAMETER_SPEC_BY_INTERNAL_NAME[name] for name in SOURCE_REDSHIFT_PARAMETER_NAMES
        ),
        required_context_fields=("zs",),
    ),
    powerlaw_lensing_component(
        required_context_fields=("z_grid", "chi_kpc_grid", "mass_radius_kpc"),
    ),
    theta_gamma_cross_section_component(
        required_context_fields=("cs_theta_e_axis", "cs_gamma_grid", "cs_cross_section_grid"),
        required_capabilities=(CAPABILITY_LENSING_CROSS_SECTION_THETA_GAMMA_V1,),
    ),
    discovery_probability_component(
        parameters=tuple(PARAMETER_SPEC_BY_INTERNAL_NAME[name] for name in DISCOVERY_PARAMETER_NAMES),
        required_context_fields=("cs_theta_e_axis", "cs_gamma_grid", "cs_cross_section_grid"),
    ),
)

PARAMETERS = aggregate_parameters(COMPONENTS)
if PARAMETERS != PARAMETER_SPECS:
    raise ValueError("CMASS component parameter blocks no longer match the public parameter schema.")


def get_model_spec() -> ModelSpec:
    """
    Return the human-authored scientific specification for CMASS.

    The registry pairs this spec with `models/cmass/runtime.py` to build the
    compiled context.  This function is the first layer where the generic
    component repository becomes the concrete CMASS model.
    """

    return ModelSpec(
        name=MODEL_NAME,
        component_key=MODEL_COMPONENT_KEY,
        required_unit_convention=H_UNITS_V1,
        mass_aperture_kpc=MASS_APERTURE_KPC,
        parameters=PARAMETERS,
        metadata={
            "component_assembly": (
                "lens_observations -> enclosed_mass_population -> "
                "gamma_population -> source_redshift -> theta_gamma_selection "
                "-> observed_sigma_likelihood -> fundamental_plane_prior"
            ),
            "gamma_distribution": GAMMA_DISTRIBUTION_SIGMA_STAR_DEPENDENT,
            "mass_definition": "m5_hinvkpc",
            "unit_convention": H_UNITS_V1,
        },
        required_capabilities=aggregate_required_capabilities(COMPONENTS),
        optional_capabilities=aggregate_optional_capabilities(
            COMPONENTS,
            extra=(
                CAPABILITY_VELOCITY_DISPERSION_PER_LENS_S2_V1,
                CAPABILITY_VELOCITY_DISPERSION_FP_WITHIN_RE_V1,
            ),
        ),
        static_codes={"gamma_mode": GAMMA_MODE_SIGMA_STAR_DEPENDENT_CODE},
        backend_kernel=MODEL_NAME,
        fp_prior_defaults=CMASS_FP_PRIOR_DEFAULTS_20260429.to_config_defaults(),
    )


__all__ = [
    "GAMMA_DISTRIBUTION_SIGMA_STAR_DEPENDENT",
    "INTERNAL_PARAMETER_NAMES",
    "MODEL_NAME",
    "COMPONENTS",
    "PARAMETER_SPECS",
    "PARAMETERS",
    "PUBLIC_PARAMETER_NAMES",
    "get_model_spec",
]
