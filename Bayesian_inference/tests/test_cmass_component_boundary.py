"""Component-boundary tests for the default CMASS model.

These tests intentionally focus on module ownership rather than scientific
values.  The numerical regression suite already checks the CMASS likelihood;
this file makes sure the model file stays as a small assembly layer after the
component split.
"""

from __future__ import annotations

import ast
import inspect

import jax.numpy as jnp
import numpy as np

from cmass_lens_inference.models import cmass
from cmass_lens_inference.models.components.cmass import (
    likelihood,
    parameters,
    population,
    selection,
    summaries,
)
from cmass_lens_inference.models.components.common import fp_prior


def test_cmass_file_is_assembly_only() -> None:
    """`models.cmass` should not define scientific hook implementations."""

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
        "draw_population",
        "selection_weight_from_normal",
        "lens_integrals",
        "summary_row",
        "extra_prior",
    } & function_names


def test_cmass_model_spec_is_assembled_from_components() -> None:
    """The public CMASS spec should wire hooks from dedicated components."""

    model_spec = cmass.get_model_spec()

    assert model_spec.parameters is parameters.PARAMETER_SPECS
    assert model_spec.unpack_theta is parameters.unpack_theta
    assert model_spec.validate_theta is parameters.validate_theta
    assert model_spec.draw_population is population.draw_population
    assert model_spec.selection_weight is selection.selection_weight_from_normal
    assert model_spec.summary_row is summaries.summary_row
    assert model_spec.lens_integrals is likelihood.lens_integrals
    assert model_spec.extra_prior is summaries.extra_prior


def test_cmass_parameter_component_exposes_fixed_schema() -> None:
    """The componentized CMASS model should keep the fixed 11D schema."""

    assert parameters.INTERNAL_PARAMETER_NAMES == (
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
    assert tuple(parameter.public_name for parameter in parameters.PARAMETER_SPECS) == (
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


def test_common_fp_prior_disabled_path_returns_neutral_prior_and_nan_diagnostics() -> None:
    """Disabled FP prior should be a neutral log term with empty diagnostics."""

    log_prior, fpfit_mu, fpfit_beta, fpfit_xi, fpfit_scatter = fp_prior.fp_prior_value(
        jnp.zeros(fp_prior.FP_OLS_SUMMARY_SIZE, dtype=jnp.float64),
        fp_enabled=0,
        fp_fiducial_scatter=0.075,
        fp_scatter_error=0.003,
        fp_mu_v_prior=2.34548,
        fp_mu_v_error=0.00611,
        fp_beta_v_prior=0.176,
        fp_beta_v_error=0.011,
    )

    assert float(log_prior) == 0.0
    assert np.isnan(float(fpfit_mu))
    assert np.isnan(float(fpfit_beta))
    assert np.isnan(float(fpfit_xi))
    assert np.isnan(float(fpfit_scatter))
