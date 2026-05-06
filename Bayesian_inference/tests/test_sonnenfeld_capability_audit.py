"""Tests for the Sonnenfeld 2024 pre-implementation capability audit.

The first Sonnenfeld step is deliberately not a runnable likelihood.  These
tests lock down the data contract and paper constants that must be satisfied
before the model can be enabled in the registry.
"""

from __future__ import annotations

import numpy as np
import pytest

from cmass_lens_inference.canonical_dataset import (
    CAPABILITY_LENSING_CROSS_SECTION_THETA_GAMMA_V1,
    CAPABILITY_LENSING_MASS_GRIDS_V1,
    CAPABILITY_LENS_OBSERVATIONS_V1,
    CAPABILITY_VELOCITY_DISPERSION_PER_LENS_S2_V1,
    CAPABILITY_VELOCITY_DISPERSION_POPULATION_SIGMA_UNIT_V1,
)
from cmass_lens_inference.models import sonnenfeld2024_slacs
from cmass_lens_inference.models.components.sonnenfeld2024_slacs import (
    capabilities,
    parameters,
)
from test_canonical_dataset import _write_canonical_dataset


def test_sonnenfeld_required_capabilities_match_canonical_schema() -> None:
    """Sonnenfeld should declare the full canonical data contract it needs."""

    assert capabilities.REQUIRED_CAPABILITIES == (
        CAPABILITY_LENS_OBSERVATIONS_V1,
        CAPABILITY_LENSING_MASS_GRIDS_V1,
        CAPABILITY_LENSING_CROSS_SECTION_THETA_GAMMA_V1,
        CAPABILITY_VELOCITY_DISPERSION_PER_LENS_S2_V1,
        CAPABILITY_VELOCITY_DISPERSION_POPULATION_SIGMA_UNIT_V1,
    )


def test_sonnenfeld_capability_audit_reports_missing_population_sigma_unit() -> None:
    """Current CMASS-like canonical inputs should fail the Sonnenfeld audit."""

    available = {
        CAPABILITY_LENS_OBSERVATIONS_V1,
        CAPABILITY_LENSING_MASS_GRIDS_V1,
        CAPABILITY_LENSING_CROSS_SECTION_THETA_GAMMA_V1,
        CAPABILITY_VELOCITY_DISPERSION_PER_LENS_S2_V1,
    }

    audit = capabilities.audit_capabilities(available)

    assert audit.ready is False
    assert audit.missing_capabilities == (
        CAPABILITY_VELOCITY_DISPERSION_POPULATION_SIGMA_UNIT_V1,
    )
    assert "theta_E_est" in audit.blocking_reason


def test_sonnenfeld_capability_audit_accepts_complete_capability_set() -> None:
    """A canonical dataset with every required capability should pass audit."""

    audit = capabilities.audit_capabilities(capabilities.REQUIRED_CAPABILITIES)

    assert audit.ready is True
    assert audit.missing_capabilities == ()
    assert audit.blocking_reason == ""


def test_sonnenfeld_capability_audit_accepts_loaded_canonical_dataset(tmp_path) -> None:
    """The audit should work on a loaded dataset, not only raw capability sets."""

    from cmass_lens_inference.canonical_dataset import load_canonical_inference_dataset

    dataset_path = _write_canonical_dataset(
        tmp_path / "sonnenfeld_ready_canonical.hdf5",
        declare_population_sigma_unit=True,
        write_population_sigma_unit=True,
    )
    dataset = load_canonical_inference_dataset(
        dataset_path,
        expected_unit_convention="h_units_v1",
        expected_h_ref=0.7,
        expected_profile_name="sersic",
        expected_mass_definition_label="m5_hinvkpc",
        required_capabilities=capabilities.REQUIRED_CAPABILITIES,
    )

    audit = capabilities.audit_capabilities(dataset)

    assert audit.ready is True
    assert audit.missing_capabilities == ()


def test_sonnenfeld_parameter_constants_preserve_paper_mass_scale() -> None:
    """The initial constants should preserve the paper's physical-mass pivots."""

    assert parameters.MSTAR_PIVOT_PHYSICAL == pytest.approx(11.3)
    assert parameters.MBAR_PHYSICAL == pytest.approx(11.06)
    assert parameters.PARENT_ALPHA == pytest.approx(-1.207)
    np.testing.assert_allclose(
        parameters.TRUNCATION_MASS_POLYNOMIAL_COEFFICIENTS,
        np.asarray([9.388, 7.855, 48.34, -312.5, 535.7, -274.2]),
    )
    assert tuple(parameter.internal_name for parameter in parameters.PARAMETER_SPECS) == (
        "mu5_0",
        "beta5",
        "xi5",
        "sigma5",
        "mu_gamma_0",
        "beta_gamma",
        "xi_gamma",
        "sigma_gamma",
        "mu_zs",
        "sigma_zs",
        "theta0",
        "loga",
    )


def test_sonnenfeld_model_spec_keeps_capability_audit_context() -> None:
    """The enabled model should still publish the same canonical data needs."""

    model_spec = sonnenfeld2024_slacs.get_model_spec()

    assert model_spec.name == "sonnenfeld2024_slacs"
    assert model_spec.required_capabilities == capabilities.REQUIRED_CAPABILITIES
    assert model_spec.metadata["selection"] == "velocity_dispersion_proxy_theta_e_est"
