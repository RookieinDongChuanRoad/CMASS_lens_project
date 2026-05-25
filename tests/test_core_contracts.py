from __future__ import annotations

import math
from pathlib import Path

import numpy as np


def test_core_exposes_canonical_dataset_schema_constants() -> None:
    """Core schema constants should match the existing canonical HDF5 contract."""

    from statistical_sl.core import canonical_schema

    assert canonical_schema.CANONICAL_SCHEMA_VERSION == "canonical_inference_dataset_v1"
    assert canonical_schema.TOP_LEVEL_BLOCKS == (
        "metadata",
        "lenses",
        "lensing_mass_grids",
        "lensing_cross_section",
        "velocity_dispersion_grids",
    )
    assert canonical_schema.CAPABILITY_LENS_OBSERVATIONS_V1 == "lens_observations.v1"


def test_core_exposes_unit_conversion_helpers() -> None:
    """Core unit helpers should preserve the h-units algebra used by data prep."""

    from statistical_sl.core import unit_conventions

    log_mstar = unit_conventions.logMstar_h2_from_legacy(11.0, h_ref=0.7)
    log_re = unit_conventions.logRe_hinv_from_legacy(1.0, h_ref=0.7)

    assert unit_conventions.H_UNITS_V1 == "h_units_v1"
    assert np.isclose(log_mstar, 11.0 + 2.0 * math.log10(0.7))
    assert np.isclose(log_re, 1.0 + math.log10(0.7))


def test_core_exposes_mass_definition_contract() -> None:
    """Core mass definitions should be available without importing inference paths."""

    from statistical_sl.core import mass_definition

    definition = mass_definition.get_mass_definition(
        5,
        unit_convention=mass_definition.H_UNITS_V1,
    )

    assert definition.label == "m5_hinvkpc"
    assert definition.public_parameter_names == ("mu5h_0", "beta5h", "xi5h", "sigma5h")
    assert np.isclose(definition.physical_radius_kpc(0.7), 5.0 / 0.7)


def test_core_exposes_manifest_and_artifact_names() -> None:
    """Shared filenames should live in core rather than in one workflow package."""

    from statistical_sl.core import artifacts, manifests

    assert manifests.RUN_MANIFEST_FILENAME == "run_manifest.json"
    assert manifests.POSTERIOR_PREDICTIVE_DIRNAME == "posterior_predictive"
    assert manifests.DIAGNOSTICS_DIRNAME == "diagnostics"
    assert artifacts.INFERENCE_CHAIN_FILENAME == "chain.h5"
    assert artifacts.PPC_SUMMARY_FILENAME == "ppc_summary.json"


def test_core_contract_modules_do_not_import_legacy_workflows() -> None:
    """Core contracts must not depend on data-prep, inference, or PPC packages."""

    repository_root = Path(__file__).resolve().parents[1]
    core_root = repository_root / "src" / "statistical_sl" / "core"
    forbidden_import_fragments = (
        "statistical_sl.data_preparation",
        "statistical_sl.inference",
        "statistical_sl.posterior_predictive",
    )

    for module_path in core_root.glob("*.py"):
        module_source = module_path.read_text(encoding="utf-8")
        for forbidden_fragment in forbidden_import_fragments:
            assert forbidden_fragment not in module_source, (module_path, forbidden_fragment)
