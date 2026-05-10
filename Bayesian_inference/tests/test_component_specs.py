"""Tests for reusable component declarations and assembly aggregation."""

from __future__ import annotations

import importlib

import pytest

from cmass_lens_inference.components import (
    ComponentSpec,
    KernelRef,
    aggregate_optional_capabilities,
    aggregate_parameters,
    aggregate_required_capabilities,
)
from cmass_lens_inference.model_interfaces import ParameterSpec
from cmass_lens_inference.models import cmass, sonnenfeld2024_slacs


def test_component_spec_rejects_duplicate_parameter_names() -> None:
    """Assembly should fail before duplicate theta/config names reach backend code."""

    first = ComponentSpec(
        name="first",
        kind="test",
        parameters=(ParameterSpec("alpha", "alpha", (0.0, 1.0)),),
    )
    duplicate_internal = ComponentSpec(
        name="duplicate-internal",
        kind="test",
        parameters=(ParameterSpec("alpha", "beta", (0.0, 1.0)),),
    )
    duplicate_public = ComponentSpec(
        name="duplicate-public",
        kind="test",
        parameters=(ParameterSpec("gamma", "alpha", (0.0, 1.0)),),
    )

    with pytest.raises(ValueError, match="Duplicate internal parameter"):
        aggregate_parameters((first, duplicate_internal))
    with pytest.raises(ValueError, match="Duplicate public parameter"):
        aggregate_parameters((first, duplicate_public))


def test_component_capability_aggregation_preserves_first_seen_order() -> None:
    """Capability aggregation should be deterministic and duplicate-free."""

    first = ComponentSpec(
        name="first",
        kind="test",
        required_capabilities=("a", "b"),
        optional_capabilities=("optional-a",),
    )
    second = ComponentSpec(
        name="second",
        kind="test",
        required_capabilities=("b", "c"),
        optional_capabilities=("optional-a", "optional-b"),
    )

    assert aggregate_required_capabilities((first, second), extra=("c", "d")) == ("a", "b", "c", "d")
    assert aggregate_optional_capabilities((first, second), extra=("optional-b", "optional-c")) == (
        "optional-a",
        "optional-b",
        "optional-c",
    )


def test_component_spec_records_required_kernel_refs_without_dispatch() -> None:
    """Kernel refs should be auditable declarations, not runtime callables."""

    component = ComponentSpec(
        name="population.source_redshift.gaussian",
        kind="source_redshift",
        required_kernels=(KernelRef("population.source_redshift", "gaussian_density"),),
    )

    assert component.required_kernels == (
        KernelRef("population.source_redshift", "gaussian_density"),
    )


def test_cmass_assembly_uses_component_parameter_and_capability_sources() -> None:
    """CMASS ModelSpec should be aggregated from selected component declarations."""

    model_spec = cmass.get_model_spec()

    assert model_spec.parameters == aggregate_parameters(cmass.COMPONENTS)
    assert model_spec.parameters == cmass.PARAMETER_SPECS
    assert model_spec.required_capabilities == aggregate_required_capabilities(cmass.COMPONENTS)
    assert model_spec.optional_capabilities == aggregate_optional_capabilities(
        cmass.COMPONENTS,
        extra=tuple(model_spec.optional_capabilities),
    )
    assert all(not component.name.startswith("cmass.") for component in cmass.COMPONENTS)


def test_sonnenfeld_assembly_uses_component_parameter_and_capability_sources() -> None:
    """Sonnenfeld ModelSpec should derive its contract from the selected component."""

    model_spec = sonnenfeld2024_slacs.get_model_spec()

    assert model_spec.parameters == aggregate_parameters(sonnenfeld2024_slacs.COMPONENTS)
    assert model_spec.parameters == sonnenfeld2024_slacs.PARAMETER_SPECS
    assert model_spec.required_capabilities == sonnenfeld2024_slacs.REQUIRED_CAPABILITIES
    assert model_spec.required_capabilities == aggregate_required_capabilities(
        sonnenfeld2024_slacs.COMPONENTS
    )
    assert all(
        not component.name.startswith("sonnenfeld2024_slacs.")
        for component in sonnenfeld2024_slacs.COMPONENTS
    )


def test_sonnenfeld_sigma_star_gamma_assembly_uses_component_sources() -> None:
    """The new Sonnenfeld peer model should also be component-assembled."""

    sigma_star_model = importlib.import_module(
        "cmass_lens_inference.models.sonnenfeld2024_slacs_sigma_star_gamma"
    )
    model_spec = sigma_star_model.get_model_spec()

    assert model_spec.parameters == aggregate_parameters(sigma_star_model.COMPONENTS)
    assert model_spec.parameters == sigma_star_model.PARAMETER_SPECS
    assert model_spec.required_capabilities == sigma_star_model.REQUIRED_CAPABILITIES
    assert model_spec.required_capabilities == aggregate_required_capabilities(
        sigma_star_model.COMPONENTS
    )
    assert model_spec.metadata["gamma_distribution"] == "sigma_star_dependent"
    assert "beta_sigma_star_gamma" in [
        parameter.public_name for parameter in model_spec.parameters
    ]
    assert all(
        not component.name.startswith("sonnenfeld2024_slacs_sigma_star_gamma.")
        for component in sigma_star_model.COMPONENTS
    )


def test_selected_component_kernel_refs_resolve_to_shared_numba_kernels() -> None:
    """Selected component declarations should point at real shared kernel functions."""

    for component in (*cmass.COMPONENTS, *sonnenfeld2024_slacs.COMPONENTS):
        for kernel_ref in component.required_kernels:
            module = importlib.import_module(
                f"cmass_lens_inference.numba_backend.kernels.{kernel_ref.module}"
            )
            assert hasattr(module, kernel_ref.name)
