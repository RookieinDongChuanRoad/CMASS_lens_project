"""Boundary tests for the default CMASS production model.

These tests intentionally focus on module ownership rather than scientific
values.  The numerical regression suite already checks the CMASS likelihood;
this file makes sure the model file stays as a small assembly layer and does
not grow sampler/backend responsibilities again.
"""

from __future__ import annotations

import ast
import inspect

from cmass_lens_inference.model_registry import get_model_definition
from cmass_lens_inference.models import cmass


def test_cmass_file_is_assembly_only() -> None:
    """`models.cmass` should not define sampler or backend implementations."""

    source_tree = ast.parse(inspect.getsource(cmass))
    function_names = {
        node.name
        for node in source_tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    class_names = {
        node.name
        for node in source_tree.body
        if isinstance(node, ast.ClassDef)
    }

    assert function_names == {"get_model_spec"}
    assert "CMASSTheta" not in class_names
    assert "CMASSPopulationDraw" not in class_names
    assert not {
        "log_prob",
        "run_emcee_sampler",
        "build_compiled_model",
        "normalization_mc_numba",
        "lens_log_likelihood_numba",
    } & function_names


def test_cmass_model_spec_is_assembled_from_components() -> None:
    """The public CMASS spec should expose only declarative model metadata."""

    model_spec = cmass.get_model_spec()

    assert model_spec.parameters == cmass.PARAMETER_SPECS
    assert model_spec.name == "cmass"
    assert model_spec.component_key == "default"
    assert model_spec.backend_kernel == "cmass"
    assert model_spec.required_unit_convention == "h_units_v1"
    assert model_spec.mass_aperture_kpc == 5
    assert model_spec.static_codes == {"gamma_mode": cmass.GAMMA_MODE_SIGMA_STAR_DEPENDENT_CODE}
    assert "lensing_cross_section.theta_gamma_grid.v1" in model_spec.required_capabilities


def test_cmass_parameter_component_exposes_fixed_schema() -> None:
    """The componentized CMASS model should keep the fixed 11D schema."""

    assert cmass.INTERNAL_PARAMETER_NAMES == (
        "mu5_0",
        "beta5",
        "xi5",
        "sigma5",
        "mu_gamma_0",
        "beta_sigma_star_gamma",
        "sigma_gamma",
        "mu_zs",
        "sigma_zs",
        "theta0",
        "loga",
    )
    assert tuple(parameter.public_name for parameter in cmass.PARAMETER_SPECS) == (
        "mu5h_0",
        "beta5h",
        "xi5h",
        "sigma5h",
        "mu_gamma_0",
        "beta_sigma_star_gamma",
        "sigma_gamma",
        "mu_zs",
        "sigma_zs",
        "theta0",
        "loga",
    )


def test_cmass_registry_definition_is_numba_kernel_backed() -> None:
    """The registry should expose CMASS through the model-owned posterior."""

    model_definition = get_model_definition("cmass")

    assert model_definition.name == "cmass"
    assert model_definition.backend_kernel == "cmass"
    assert model_definition.required_capabilities == cmass.get_model_spec().required_capabilities
    assert not hasattr(model_definition, "to_backend_context")
