"""Assembly layer for the Sonnenfeld 2024 SLACS debiased model.

The backend kernels live outside this assembly layer.  This file mirrors the
CMASS pattern: it names each concrete model, records fixed metadata, and exposes
the parameter/capability contract consumed by the production backend.
"""

from __future__ import annotations

from ..mass_definition import H_UNITS_V1, LEGACY_FIXED_KPC
from ..model_interfaces import ModelSpec
from .components.sonnenfeld2024_slacs.capabilities import REQUIRED_CAPABILITIES
from .components.sonnenfeld2024_slacs import parameters


MODEL_NAME = "sonnenfeld2024_slacs"
HUNIT_MODEL_NAME = "sonnenfeld2024_slacs_hunit"
MODEL_COMPONENT_KEY = "table1_velocity_proxy"
MASS_APERTURE_KPC = 5


def _build_model_spec(
    *,
    model_name: str,
    unit_convention: str,
    mass_definition_label: str,
    mass_coordinate: str,
) -> ModelSpec:
    """
    Build one concrete Sonnenfeld unit-convention variant.

    ``sonnenfeld2024_slacs`` is reserved for the paper-native fixed-kpc mass
    convention.  ``sonnenfeld2024_slacs_hunit`` is the explicit h-units variant
    that runs on the current hunit canonical backend.  Both variants share the
    same backend kernel; the runtime context decides whether paper mass-location
    constants are shifted before numerical evaluation.
    """

    return ModelSpec(
        name=model_name,
        component_key=MODEL_COMPONENT_KEY,
        required_unit_convention=unit_convention,
        mass_aperture_kpc=MASS_APERTURE_KPC,
        parameters=parameters.PARAMETER_SPECS,
        metadata={
            "foreground_population": "sonnenfeld2024_table1",
            "selection": "velocity_dispersion_proxy_theta_e_est",
            "cross_section": "theta_gamma_finite_fibre",
            "mass_definition": mass_definition_label,
            "unit_convention": unit_convention,
            "mass_coordinate": mass_coordinate,
            "mstar_pivot_physical": parameters.MSTAR_PIVOT_PHYSICAL,
            "mbar_physical": parameters.MBAR_PHYSICAL,
        },
        required_capabilities=REQUIRED_CAPABILITIES,
        optional_capabilities=(),
        static_codes={},
        backend_kernel="sonnenfeld2024_slacs",
    )


def get_model_spec() -> ModelSpec:
    """Return the paper-native fixed-5-kpc Sonnenfeld model specification."""

    return _build_model_spec(
        model_name=MODEL_NAME,
        unit_convention=LEGACY_FIXED_KPC,
        mass_definition_label="m5",
        mass_coordinate="physical_fixed_5kpc",
    )


def get_hunit_model_spec() -> ModelSpec:
    """Return the explicit h-units Sonnenfeld model specification."""

    return _build_model_spec(
        model_name=HUNIT_MODEL_NAME,
        unit_convention=H_UNITS_V1,
        mass_definition_label="m5_hinvkpc",
        mass_coordinate="h_units_v1_m5_hinvkpc",
    )


__all__ = ["HUNIT_MODEL_NAME", "MODEL_NAME", "get_hunit_model_spec", "get_model_spec"]
