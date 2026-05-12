"""Assembly layer for the CMASS lens-only model."""

from __future__ import annotations

from ...canonical_dataset import (
    CAPABILITY_LENSING_MASS_GRIDS_V1,
    CAPABILITY_LENS_OBSERVATIONS_V1,
    CAPABILITY_VELOCITY_DISPERSION_PER_LENS_S2_V1,
)
from ...components import (
    aggregate_optional_capabilities,
    aggregate_parameters,
    aggregate_required_capabilities,
)
from ...components.observations.lens_sample import lens_sample_component
from ...components.population.aperture_mass_relation.gaussian_linear import (
    gaussian_linear_aperture_mass_component,
)
from ...components.population.gamma_relation.sigma_star_linear import (
    sigma_star_linear_gamma_component,
)
from ...components.population.size_relation.linear import linear_size_relation_component
from ...components.population.stellar_mass_function.gaussian_lens_sample import (
    gaussian_lens_sample_stellar_mass_component,
)
from ...mass_definition import H_UNITS_V1
from ...model_interfaces import ModelSpec, ParameterSpec


GAMMA_MODE_SIGMA_STAR_DEPENDENT_CODE = 2
GAMMA_DISTRIBUTION_SIGMA_STAR_DEPENDENT = "sigma_star_dependent"
MODEL_NAME = "cmass_lens_only"
MODEL_COMPONENT_KEY = "default"
BACKEND_KERNEL = "cmass_lens_only"
MASS_APERTURE_KPC = 5

LENS_STELLAR_MASS_PARAMETER_NAMES: tuple[str, ...] = (
    "mu_mstar_lens",
    "sigma_mstar_lens",
)
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
INTERNAL_PARAMETER_NAMES: tuple[str, ...] = (
    LENS_STELLAR_MASS_PARAMETER_NAMES
    + INTERNAL_MASS_PARAMETER_NAMES
    + SIGMA_STAR_DEPENDENT_GAMMA_PARAMETER_NAMES
)
PUBLIC_PARAMETER_NAMES: tuple[str, ...] = (
    LENS_STELLAR_MASS_PARAMETER_NAMES
    + ("mu5h_0", "beta5h", "xi5h", "sigma5h")
    + SIGMA_STAR_DEPENDENT_GAMMA_PARAMETER_NAMES
)
DEFAULT_BOX_PRIOR_BOUNDS_BY_INTERNAL_NAME: dict[str, tuple[float, float]] = {
    "mu_mstar_lens": (10.0, 12.5),
    "sigma_mstar_lens": (1.0e-3, 1.0),
    "mu5_0": (9.0, 12.0),
    "beta5": (-3.0, 3.0),
    "xi5": (-3.0, 3.0),
    "sigma5": (1.0e-2, 0.2),
    "mu_gamma_0": (1.5, 2.5),
    "beta_sigma_star_gamma": (-3.0, 3.0),
    "sigma_gamma": (1.0e-3, 0.5),
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
        required_context_fields=(
            "log_mstar_obs",
            "log_mstar_err",
            "log_re_obs",
            "n_obs",
            "num_sigma",
            "sigma_obs",
            "sigma_err",
        ),
        required_capabilities=(CAPABILITY_LENS_OBSERVATIONS_V1,),
    ),
    gaussian_lens_sample_stellar_mass_component(
        parameters=tuple(
            PARAMETER_SPEC_BY_INTERNAL_NAME[name] for name in LENS_STELLAR_MASS_PARAMETER_NAMES
        ),
        required_context_fields=("mstar_grid", "mstar_observation_density"),
    ),
    linear_size_relation_component(
        required_context_fields=("delta_r_grid",),
    ),
    gaussian_linear_aperture_mass_component(
        parameters=tuple(
            PARAMETER_SPEC_BY_INTERNAL_NAME[name] for name in INTERNAL_MASS_PARAMETER_NAMES
        ),
        required_context_fields=(
            "mstar_shift11p4",
            "delta_r_grid",
            "mass_grid_int",
            "dmass_dthetaein_grid_int",
        ),
        required_capabilities=(CAPABILITY_LENSING_MASS_GRIDS_V1,),
    ),
    sigma_star_linear_gamma_component(
        parameters=tuple(
            PARAMETER_SPEC_BY_INTERNAL_NAME[name]
            for name in SIGMA_STAR_DEPENDENT_GAMMA_PARAMETER_NAMES
        ),
        required_context_fields=(
            "gamma_grid_int",
            "sigma_star_shift9p0_grid",
        ),
    ),
)

PARAMETERS = aggregate_parameters(COMPONENTS)
if PARAMETERS != PARAMETER_SPECS:
    raise ValueError("CMASS lens-only component parameter blocks do not match the schema.")


def get_model_spec() -> ModelSpec:
    """
    Return the scientific specification for the CMASS lens-only model.

    This concrete model fits the already-observed lens sample directly.  It
    intentionally excludes source-redshift population parameters,
    discovery-probability parameters, lensing cross-section weights, and
    selection normalization.  The observed velocity-dispersion likelihood is
    not a standalone component here; the dedicated posterior kernel assembles
    it explicitly so the hot likelihood path remains auditable in one place.
    """

    return ModelSpec(
        name=MODEL_NAME,
        component_key=MODEL_COMPONENT_KEY,
        required_unit_convention=H_UNITS_V1,
        mass_aperture_kpc=MASS_APERTURE_KPC,
        parameters=PARAMETERS,
        metadata={
            "component_assembly": (
                "lens_observations -> lens_sample_stellar_mass_distribution -> "
                "linear_size_relation -> enclosed_mass_population -> gamma_population; "
                "posterior_kernel adds observed_sigma_likelihood"
            ),
            "gamma_distribution": GAMMA_DISTRIBUTION_SIGMA_STAR_DEPENDENT,
            "mass_definition": "m5_hinvkpc",
            "unit_convention": H_UNITS_V1,
            "selection_correction": False,
            "fp_prior_supported": False,
            "observed_velocity_dispersion_component": False,
            "target_population": "observed_cmass_lenses",
        },
        required_capabilities=aggregate_required_capabilities(
            COMPONENTS,
            extra=(CAPABILITY_VELOCITY_DISPERSION_PER_LENS_S2_V1,),
        ),
        optional_capabilities=aggregate_optional_capabilities(COMPONENTS),
        static_codes={"gamma_mode": GAMMA_MODE_SIGMA_STAR_DEPENDENT_CODE},
        backend_kernel=BACKEND_KERNEL,
    )


__all__ = [
    "BACKEND_KERNEL",
    "COMPONENTS",
    "GAMMA_DISTRIBUTION_SIGMA_STAR_DEPENDENT",
    "GAMMA_MODE_SIGMA_STAR_DEPENDENT_CODE",
    "INTERNAL_PARAMETER_NAMES",
    "MODEL_NAME",
    "PARAMETER_SPECS",
    "PARAMETERS",
    "PUBLIC_PARAMETER_NAMES",
    "get_model_spec",
]
