"""Tests for model-aware posterior-predictive dispatch.

The current refactor goal is to stop treating CMASS as an implicit global
inside the generic PPT workflow.  These tests pin the smallest dispatch
contract first: the PPT package should expose a predictive registry, CMASS
should register the existing Numba shared-parent diagnostics, and unsupported
models should fail with a model-specific error before any CMASS context builder
is touched.
"""

from __future__ import annotations

import inspect

import pytest


def test_cmass_predictive_registry_entry_declares_existing_diagnostics() -> None:
    """CMASS should register the current Numba diagnostics through a thin contract."""

    from cmass_posterior_predictive.registry import get_predictive_definition

    definition = get_predictive_definition("cmass")

    assert definition.model_name == "cmass"
    assert definition.backend == "numba_shared_parent"
    assert definition.required_external_inputs == ("sigma_table",)
    assert "posterior_diagnostics" in definition.supported_diagnostics
    assert callable(definition.build_context)
    assert definition.trend_category_names == ("parent", "detectable", "selected")
    assert definition.build_trend_panel_order(type("Mass", (), {"label": "m10"})()) == (
        "m10",
        "gamma",
        "sigma_ap",
        "gamma_vs_sigma_star",
        "gamma_vs_logre_kpc",
    )


def test_unsupported_model_predictive_dispatch_fails_before_cmass_fallback() -> None:
    """Unsupported models should fail explicitly instead of silently using CMASS logic."""

    from cmass_posterior_predictive.registry import (
        UnsupportedPredictiveModelError,
        get_predictive_definition,
    )

    with pytest.raises(UnsupportedPredictiveModelError, match="toy_hierarchical"):
        get_predictive_definition("toy_hierarchical")


@pytest.mark.parametrize("model_name", ["sonnenfeld2024_slacs", "sonnenfeld2024_slacs_hunit"])
def test_sonnenfeld_predictive_registry_entry_declares_independent_schema(model_name: str) -> None:
    """Sonnenfeld models should expose a non-CMASS predictive contract."""

    from cmass_posterior_predictive.registry import get_predictive_definition

    definition = get_predictive_definition(model_name)

    assert definition.model_name == model_name
    assert definition.backend == "numba_sonnenfeld_parent"
    assert definition.required_external_inputs == ()
    assert definition.artifact_schema_version == "sonnenfeld2024_slacs_ppt_diagnostics_v1"
    assert "posterior_diagnostics" in definition.supported_diagnostics
    assert definition.build_trend_panel_order(type("Mass", (), {"label": "m5"})()) == (
        "theta_ein",
        "sigma_ap",
        "gamma",
        "gamma_vs_logre_kpc",
    )


def test_ppc_context_builder_uses_predictive_registry_instead_of_model_name_branch() -> None:
    """The generic context builder should dispatch through the predictive registry."""

    from cmass_posterior_predictive import predictive

    source = inspect.getsource(predictive._build_ppc_context)

    assert "get_predictive_definition" in source
    assert 'runtime_config.model.name != "cmass"' not in source


def test_generic_predictive_workflow_does_not_import_cmass_posterior_helpers() -> None:
    """CMASS posterior helpers should live behind the model-specific boundary."""

    from cmass_posterior_predictive import predictive

    source = inspect.getsource(predictive)

    assert "from cmass_lens_inference.models.cmass.posterior import" not in source
    assert "cmass_gamma_population_mean" not in source
    assert "unpack_cmass_theta" not in source


def test_diagnostics_runner_is_model_definition_owned() -> None:
    """The joint diagnostics workflow should call the active predictive definition."""

    from cmass_posterior_predictive import predictive

    source = inspect.getsource(predictive.run_posterior_diagnostics)

    assert "predictive_definition.run_diagnostics" in source
    assert "_run_shared_parent_diagnostics_numba(" not in source


def test_cmass_context_builder_reuses_inference_model_registry() -> None:
    """CMASS predictive context construction should reuse the inference registry."""

    from cmass_posterior_predictive.adapters import cmass

    source = inspect.getsource(cmass.build_context)

    assert "get_model_definition" in source
    assert ".build_compiled_model(" in source
    assert "load_cmass_canonical_dataset" not in source
    assert "build_cmass_context_from_canonical_dataset" not in source


def test_legacy_raw_config_parser_is_cmass_only() -> None:
    """Raw pre-registry snapshots should not become a non-CMASS fallback path."""

    from pathlib import Path

    from cmass_posterior_predictive.legacy import load_legacy_ppc_runtime_config

    with pytest.raises(ValueError, match="only supports model.name='cmass'"):
        load_legacy_ppc_runtime_config(
            Path("config_snapshot.yaml"),
            {"model": {"name": "sonnenfeld2024_slacs"}},
        )


def test_legacy_parser_is_quarantined_outside_generic_predictive_module() -> None:
    """Generic predictive workflow should not define the legacy CMASS parser."""

    from cmass_posterior_predictive import predictive

    source = inspect.getsource(predictive)

    assert "def _legacy_ppc_parameter_order" not in source
    assert "def _load_legacy_ppc_runtime_config" not in source
