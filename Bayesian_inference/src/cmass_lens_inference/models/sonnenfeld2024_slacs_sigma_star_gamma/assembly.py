"""Assembly layer for the Sonnenfeld 2024 SLACS sigma-star-gamma model.

This package is a peer of ``models.sonnenfeld2024_slacs`` rather than an
in-place option inside that package.  The scientific target is intentionally
narrow: keep the Sonnenfeld parent population, canonical data contract, source
model, finite-fibre selection, and FP-prior semantics, but replace the
mass-and-size-residual gamma relation with the CMASS-style sigma-star relation.
"""

from __future__ import annotations

from ...canonical_dataset import (
    CAPABILITY_LENSING_CROSS_SECTION_THETA_GAMMA_V1,
    CAPABILITY_LENSING_MASS_GRIDS_V1,
    CAPABILITY_LENS_OBSERVATIONS_V1,
    CAPABILITY_VELOCITY_DISPERSION_PER_LENS_S2_V1,
    CAPABILITY_VELOCITY_DISPERSION_POPULATION_SIGMA_UNIT_V1,
)
from ...components import aggregate_parameters, aggregate_required_capabilities
from ...components.lensing.cross_section import theta_gamma_cross_section_component
from ...components.lensing.powerlaw import powerlaw_lensing_component
from ...components.observations.lens_sample import lens_sample_component
from ...components.population.aperture_mass_relation.gaussian_linear import (
    gaussian_linear_aperture_mass_component,
)
from ...components.population.gamma_relation.sigma_star_linear import (
    sigma_star_linear_gamma_component,
)
from ...components.population.size_relation.quadratic import quadratic_size_relation_component
from ...components.population.source_redshift.gaussian import gaussian_source_redshift_component
from ...components.population.stellar_mass_function.smooth_truncated_schechter import (
    smooth_truncated_schechter_component,
)
from ...components.selection.discovery_probability import discovery_probability_component
from ...components.selection.velocity_proxy import velocity_proxy_selection_component
from ...mass_definition import H_UNITS_V1, LEGACY_FIXED_KPC
from ...model_interfaces import ModelSpec, ParameterSpec
from ..sonnenfeld2024_slacs.paper_constants import (
    MBAR_PHYSICAL,
    MSTAR_PIVOT_PHYSICAL,
)


MODEL_NAME = "sonnenfeld2024_slacs_sigma_star_gamma"
HUNIT_MODEL_NAME = "sonnenfeld2024_slacs_sigma_star_gamma_hunit"
MODEL_COMPONENT_KEY = "table1_velocity_proxy_sigma_star_gamma"
MASS_APERTURE_KPC = 5
GAMMA_DISTRIBUTION_SIGMA_STAR_DEPENDENT = "sigma_star_dependent"

INTERNAL_MASS_PARAMETER_NAMES: tuple[str, ...] = (
    "mu5_0",
    "beta5",
    "xi5",
    "sigma5",
)
SIGMA_STAR_GAMMA_PARAMETER_NAMES: tuple[str, ...] = (
    "mu_gamma_0",
    "beta_sigma_star_gamma",
    "sigma_gamma",
)
SOURCE_REDSHIFT_PARAMETER_NAMES: tuple[str, ...] = ("mu_zs", "sigma_zs")
DISCOVERY_PARAMETER_NAMES: tuple[str, ...] = ("theta0", "loga")
INTERNAL_PARAMETER_NAMES: tuple[str, ...] = (
    INTERNAL_MASS_PARAMETER_NAMES
    + SIGMA_STAR_GAMMA_PARAMETER_NAMES
    + SOURCE_REDSHIFT_PARAMETER_NAMES
    + DISCOVERY_PARAMETER_NAMES
)
PUBLIC_PARAMETER_NAMES: tuple[str, ...] = INTERNAL_PARAMETER_NAMES

DEFAULT_BOX_PRIOR_BOUNDS_BY_INTERNAL_NAME: dict[str, tuple[float, float]] = {
    "mu5_0": (10.5, 12.2),
    "beta5": (-3.0, 3.0),
    "xi5": (-3.0, 3.0),
    "sigma5": (1.0e-2, 0.3),
    "mu_gamma_0": (1.2, 2.8),
    "beta_sigma_star_gamma": (-3.0, 3.0),
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
PARAMETER_SPEC_BY_INTERNAL_NAME: dict[str, ParameterSpec] = {
    parameter.internal_name: parameter for parameter in PARAMETER_SPECS
}

REQUIRED_CAPABILITIES: tuple[str, ...] = (
    CAPABILITY_LENS_OBSERVATIONS_V1,
    CAPABILITY_LENSING_MASS_GRIDS_V1,
    CAPABILITY_LENSING_CROSS_SECTION_THETA_GAMMA_V1,
    CAPABILITY_VELOCITY_DISPERSION_PER_LENS_S2_V1,
    CAPABILITY_VELOCITY_DISPERSION_POPULATION_SIGMA_UNIT_V1,
)
COMPONENTS = (
    lens_sample_component(
        required_context_fields=("zd", "zs", "sigma_obs", "sigma_err"),
        required_capabilities=(CAPABILITY_LENS_OBSERVATIONS_V1,),
    ),
    smooth_truncated_schechter_component(
        required_context_fields=("zd", "parent_mstar_density_grid"),
    ),
    quadratic_size_relation_component(
        required_context_fields=("size_density_grid", "delta_r_grid"),
    ),
    gaussian_linear_aperture_mass_component(
        parameters=tuple(
            PARAMETER_SPEC_BY_INTERNAL_NAME[name] for name in INTERNAL_MASS_PARAMETER_NAMES
        ),
        required_context_fields=("mass_grid_int",),
        required_capabilities=(CAPABILITY_LENSING_MASS_GRIDS_V1,),
    ),
    sigma_star_linear_gamma_component(
        parameters=tuple(
            PARAMETER_SPEC_BY_INTERNAL_NAME[name] for name in SIGMA_STAR_GAMMA_PARAMETER_NAMES
        ),
        required_context_fields=(
            "gamma_grid_int",
            "mstar_grid",
            "log_re_obs",
            "parent_sample_mstar",
            "parent_sample_log_re",
        ),
    ),
    gaussian_source_redshift_component(
        parameters=tuple(PARAMETER_SPEC_BY_INTERNAL_NAME[name] for name in SOURCE_REDSHIFT_PARAMETER_NAMES),
        required_context_fields=("zs",),
    ),
    powerlaw_lensing_component(
        required_context_fields=("z_grid", "chi_kpc_grid", "mass_radius_kpc"),
    ),
    theta_gamma_cross_section_component(
        required_context_fields=("cs_theta_e_axis", "cs_gamma_axis", "cs_cross_section_grid"),
        required_capabilities=(CAPABILITY_LENSING_CROSS_SECTION_THETA_GAMMA_V1,),
    ),
    discovery_probability_component(
        parameters=tuple(PARAMETER_SPEC_BY_INTERNAL_NAME[name] for name in DISCOVERY_PARAMETER_NAMES),
        required_context_fields=("cs_theta_e_axis", "cs_gamma_axis", "cs_cross_section_grid"),
    ),
    velocity_proxy_selection_component(
        required_context_fields=(
            "base_normals",
            "population_sigma_unit_grid",
            "s2_grid_int",
            "has_s2",
            "num_sigma",
            "sigma_obs",
            "sigma_err",
        ),
        required_capabilities=(
            CAPABILITY_VELOCITY_DISPERSION_PER_LENS_S2_V1,
            CAPABILITY_VELOCITY_DISPERSION_POPULATION_SIGMA_UNIT_V1,
        ),
    ),
)
PARAMETERS = aggregate_parameters(COMPONENTS)
if PARAMETERS != PARAMETER_SPECS:
    raise ValueError("Sonnenfeld sigma-star-gamma component parameters do not match the public schema.")


def _build_model_spec(
    *,
    model_name: str,
    unit_convention: str,
    mass_definition_label: str,
    mass_coordinate: str,
) -> ModelSpec:
    """
    Build one concrete unit-convention variant of the new Sonnenfeld peer model.

    The package follows the existing Sonnenfeld unit boundary: paper-native and
    h-unit semantics share one implementation package, but each is exposed as a
    separate registry model name so source YAML cannot silently mix units.
    """

    return ModelSpec(
        name=model_name,
        component_key=MODEL_COMPONENT_KEY,
        required_unit_convention=unit_convention,
        mass_aperture_kpc=MASS_APERTURE_KPC,
        parameters=PARAMETERS,
        metadata={
            "component_assembly": (
                "table1_parent_density -> quadratic_size_relation -> "
                "enclosed_mass_population -> sigma_star_gamma_population -> "
                "source_redshift -> finite_fibre_selection -> "
                "velocity_proxy_likelihood"
            ),
            "foreground_population": "sonnenfeld2024_table1",
            "gamma_distribution": GAMMA_DISTRIBUTION_SIGMA_STAR_DEPENDENT,
            "selection": "velocity_dispersion_proxy_theta_e_est",
            "cross_section": "theta_gamma_finite_fibre",
            "mass_definition": mass_definition_label,
            "unit_convention": unit_convention,
            "mass_coordinate": mass_coordinate,
            "mstar_pivot_physical": MSTAR_PIVOT_PHYSICAL,
            "mbar_physical": MBAR_PHYSICAL,
        },
        required_capabilities=aggregate_required_capabilities(COMPONENTS),
        optional_capabilities=(),
        static_codes={},
        backend_kernel=MODEL_NAME,
    )


def get_model_spec() -> ModelSpec:
    """Return the paper-native fixed-5-kpc sigma-star-gamma model specification."""

    return _build_model_spec(
        model_name=MODEL_NAME,
        unit_convention=LEGACY_FIXED_KPC,
        mass_definition_label="m5",
        mass_coordinate="physical_fixed_5kpc",
    )


def get_hunit_model_spec() -> ModelSpec:
    """Return the explicit h-unit sigma-star-gamma model specification."""

    return _build_model_spec(
        model_name=HUNIT_MODEL_NAME,
        unit_convention=H_UNITS_V1,
        mass_definition_label="m5_hinvkpc",
        mass_coordinate="h_units_v1_m5_hinvkpc",
    )


__all__ = [
    "COMPONENTS",
    "GAMMA_DISTRIBUTION_SIGMA_STAR_DEPENDENT",
    "HUNIT_MODEL_NAME",
    "INTERNAL_PARAMETER_NAMES",
    "MODEL_NAME",
    "PARAMETER_SPECS",
    "PARAMETERS",
    "PUBLIC_PARAMETER_NAMES",
    "REQUIRED_CAPABILITIES",
    "get_hunit_model_spec",
    "get_model_spec",
]
