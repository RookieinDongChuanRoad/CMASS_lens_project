"""Assembly layer for the toy hierarchical model.

The model is deliberately synthetic.  Its purpose is architectural: demonstrate
that a new model can provide component declarations, runtime context, and a
production log-probability without changing runner, sampler, output, or
posterior-reader code.
"""

from __future__ import annotations

from ...components import ComponentSpec, aggregate_parameters, aggregate_required_capabilities
from ...mass_definition import H_UNITS_V1
from ...model_interfaces import ModelSpec, ParameterSpec


MODEL_NAME = "toy_hierarchical"
MODEL_COMPONENT_KEY = "gaussian_population"
MASS_APERTURE_KPC = 5
PARAMETER_SPECS: tuple[ParameterSpec, ...] = (
    ParameterSpec("population_mean", "population_mean", (-5.0, 5.0)),
    ParameterSpec("log_population_scatter", "log_population_scatter", (-5.0, 1.0)),
)
COMPONENTS = (
    ComponentSpec(
        name="synthetic.gaussian_population",
        kind="synthetic_population",
        parameters=PARAMETER_SPECS,
        required_context_fields=("observed_values", "observed_errors"),
        metadata={"density": "gaussian_population_with_measurement_error"},
    ),
)


def get_model_spec() -> ModelSpec:
    """
    Return the synthetic two-parameter hierarchical model specification.

    The unit and mass-aperture fields satisfy the generic production config
    contract.  The toy posterior itself ignores lensing mass definitions because
    it models a small synthetic population vector, not a physical lens sample.
    """

    return ModelSpec(
        name=MODEL_NAME,
        component_key=MODEL_COMPONENT_KEY,
        required_unit_convention=H_UNITS_V1,
        mass_aperture_kpc=MASS_APERTURE_KPC,
        parameters=aggregate_parameters(COMPONENTS),
        metadata={
            "component_assembly": "gaussian_population",
            "purpose": "architecture_acceptance_test",
            "data_source": "synthetic_runtime_context",
        },
        required_capabilities=aggregate_required_capabilities(COMPONENTS),
        optional_capabilities=(),
        static_codes={},
        backend_kernel=MODEL_NAME,
    )


__all__ = ["COMPONENTS", "MODEL_NAME", "PARAMETER_SPECS", "get_model_spec"]
