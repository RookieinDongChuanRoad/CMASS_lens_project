"""Assembly layer for the default CMASS lens-population model.

This module intentionally does not implement scientific formulas.  It names
the concrete model, records the fixed CMASS choices, and wires dedicated
components into `ModelSpec`.  The component split keeps this file readable and
gives future models, such as Sonnenfeld 2024, the same assembly pattern without
copying the JAX execution framework.
"""

from __future__ import annotations

from ..canonical_dataset import (
    CAPABILITY_LENSING_CROSS_SECTION_THETA_GAMMA_V1,
    CAPABILITY_LENSING_MASS_GRIDS_V1,
    CAPABILITY_LENS_OBSERVATIONS_V1,
    CAPABILITY_VELOCITY_DISPERSION_FP_WITHIN_RE_V1,
    CAPABILITY_VELOCITY_DISPERSION_PER_LENS_S2_V1,
)
from ..mass_definition import H_UNITS_V1
from ..model_interfaces import ModelSpec
from .components.cmass import likelihood, parameters, population, selection, summaries


GAMMA_MODE_SIGMA_STAR_DEPENDENT_CODE = 2
GAMMA_DISTRIBUTION_SIGMA_STAR_DEPENDENT = "sigma_star_dependent"
MODEL_NAME = "cmass"
MODEL_COMPONENT_KEY = "default"
MASS_APERTURE_KPC = 5


def get_model_spec() -> ModelSpec:
    """
    Return the human-authored scientific specification for CMASS.

    The registry pairs this spec with `cmass_runtime.get_runtime_adapter()` to
    build the lower-level backend definition.  Adding or replacing a CMASS
    formula should happen inside `models.components.cmass`, while this function
    remains the single place that assembles those components into a runnable
    model.
    """

    return ModelSpec(
        name=MODEL_NAME,
        component_key=MODEL_COMPONENT_KEY,
        required_unit_convention=H_UNITS_V1,
        mass_aperture_kpc=MASS_APERTURE_KPC,
        parameters=parameters.PARAMETER_SPECS,
        metadata={
            "gamma_distribution": GAMMA_DISTRIBUTION_SIGMA_STAR_DEPENDENT,
            "mass_definition": "m5_hinvkpc",
            "unit_convention": H_UNITS_V1,
        },
        required_capabilities=(
            CAPABILITY_LENS_OBSERVATIONS_V1,
            CAPABILITY_LENSING_MASS_GRIDS_V1,
            CAPABILITY_LENSING_CROSS_SECTION_THETA_GAMMA_V1,
        ),
        optional_capabilities=(
            CAPABILITY_VELOCITY_DISPERSION_PER_LENS_S2_V1,
            CAPABILITY_VELOCITY_DISPERSION_FP_WITHIN_RE_V1,
        ),
        static_codes={"gamma_mode": GAMMA_MODE_SIGMA_STAR_DEPENDENT_CODE},
        unpack_theta=parameters.unpack_theta,
        validate_theta=parameters.validate_theta,
        draw_population=population.draw_population,
        selection_weight=selection.selection_weight_from_normal,
        summary_row=summaries.summary_row,
        lens_integrals=likelihood.lens_integrals,
        extra_prior=summaries.extra_prior,
    )


__all__ = [
    "GAMMA_DISTRIBUTION_SIGMA_STAR_DEPENDENT",
    "MODEL_NAME",
    "get_model_spec",
]
