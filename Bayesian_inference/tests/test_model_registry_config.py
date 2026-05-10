"""Tests for the concrete-model registry configuration contract."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from cmass_lens_inference.canonical_dataset import (
    CAPABILITY_LENSING_CROSS_SECTION_THETA_GAMMA_V1,
    CAPABILITY_LENSING_MASS_GRIDS_V1,
    CAPABILITY_LENS_OBSERVATIONS_V1,
)
from cmass_lens_inference.config import load_runtime_config
from cmass_lens_inference.models import cmass, cmass_runtime
from cmass_lens_inference.models.cmass.context import CMASSModelContext
from cmass_lens_inference.mass_definition import H_UNITS_V1, LEGACY_FIXED_KPC, get_mass_definition
from cmass_lens_inference.model_interfaces import (
    CompiledContextBundle,
    ModelRuntimeAdapter,
    ModelSpec,
    ParameterSpec,
)
from cmass_lens_inference.model_registry import get_model_definition
from cmass_lens_inference.numba_backend.compiled_model_factory import build_model_definition


SONNENFELD_SIGMA_STAR_PUBLIC_PARAMETERS = (
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
SONNENFELD_SIGMA_STAR_BOX_PRIOR = {
    "mu5_0": [10.5, 12.2],
    "beta5": [-3.0, 3.0],
    "xi5": [-3.0, 3.0],
    "sigma5": [1.0e-2, 0.3],
    "mu_gamma_0": [1.2, 2.8],
    "beta_sigma_star_gamma": [-3.0, 3.0],
    "sigma_gamma": [1.0e-2, 0.8],
    "mu_zs": [0.0, 2.0],
    "sigma_zs": [1.0e-3, 1.0],
    "theta0": [0.0, 3.0],
    "loga": [-1.0, 3.0],
}


def _minimal_cmass_config(tmp_path: Path) -> dict:
    """Build a config payload that can be parsed without opening HDF5 files."""

    return {
        "profile": {"name": "sersic"},
        "unit_convention": H_UNITS_V1,
        "model": {"name": "cmass"},
        "data": {
            "inference_dataset_path": str(tmp_path / "canonical_inference_dataset.hdf5"),
        },
        "box_prior": {
            "mu5h_0": [9.0, 12.0],
            "beta5h": [-3.0, 3.0],
            "xi5h": [-3.0, 3.0],
            "sigma5h": [1.0e-2, 0.2],
            "mu_gamma_0": [1.5, 2.5],
            "beta_sigma_star_gamma": [-3.0, 3.0],
            "sigma_gamma": [0.0, 0.5],
            "mu_zs": [1.0, 3.0],
            "sigma_zs": [0.0, 2.0],
            "theta0": [0.0, 3.0],
            "loga": [-1.0, 3.0],
        },
        "sampling": {
            "random_seed": 7,
            "n_walkers": 24,
            "n_steps": 3,
            "burn_in": 1,
            "initial_center": {
                "mu5h_0": 11.17,
                "beta5h": 0.59,
                "xi5h": -0.11,
                "sigma5h": 0.06,
                "mu_gamma_0": 1.99,
                "beta_sigma_star_gamma": 0.24,
                "sigma_gamma": 0.149,
                "mu_zs": 1.8,
                "sigma_zs": 0.215,
                "theta0": 0.93,
                "loga": 1.0,
            },
            "initial_jitter_scale": 1.0e-3,
        },
        "integration": {
            "gamma_points": 200,
            "mstar_points": 200,
            "normalization_samples": 128,
        },
        "cosmology": {"h0": 70.0, "omega_m": 0.3},
        "runtime": {
            "checkpoint_every": 1,
            "parallel_strategy": "auto",
            "progress": False,
            "progress_summary_every": 1,
            "show_stage_timing": True,
            "disable_hdf5_file_locking": False,
            "num_threads": 0,
            "reserve_cores": 2,
        },
        "output": {
            "root_dir": str(tmp_path / "outputs"),
            "run_label": "registry-contract",
            "overwrite_latest": True,
        },
    }


def test_runtime_config_resolves_cmass_as_one_concrete_model(tmp_path: Path) -> None:
    """The default CMASS model should not require component switches."""

    config_path = tmp_path / "cmass.yaml"
    config_path.write_text(
        yaml.safe_dump(_minimal_cmass_config(tmp_path), sort_keys=False),
        encoding="utf-8",
    )

    runtime_config = load_runtime_config(config_path)

    assert runtime_config.model.name == "cmass"
    assert runtime_config.mass_definition == get_mass_definition(5, unit_convention=H_UNITS_V1)
    assert runtime_config.parameter_schema.model_name == "cmass"
    assert runtime_config.parameter_schema.model_metadata["gamma_distribution"] == "sigma_star_dependent"
    assert runtime_config.parameter_schema.model_metadata["mass_definition"] == "m5_hinvkpc"
    assert not hasattr(runtime_config.model, "components")


def test_runtime_config_accepts_canonical_inference_dataset_path(tmp_path: Path) -> None:
    """New configs should be able to point at one canonical inference dataset."""

    payload = _minimal_cmass_config(tmp_path)
    dataset_path = tmp_path / "canonical_inference_dataset.hdf5"
    payload["data"] = {"inference_dataset_path": str(dataset_path)}
    config_path = tmp_path / "cmass_canonical.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    runtime_config = load_runtime_config(config_path)

    assert runtime_config.data.inference_dataset_path == dataset_path.resolve()
    assert runtime_config.data.observation_path is None
    assert runtime_config.data.cross_section_path is None


def test_runtime_config_rejects_raw_input_data_paths(tmp_path: Path) -> None:
    """Production inference configs must start from one canonical dataset."""

    payload = _minimal_cmass_config(tmp_path)
    payload["data"] = {
        "observation_path": str(tmp_path / "observations.hdf5"),
        "cross_section_path": str(tmp_path / "cross_section.h5"),
    }
    config_path = tmp_path / "raw_input_paths.yaml"
    config_path.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="inference_dataset_path"):
        load_runtime_config(config_path)


def test_model_components_are_rejected(tmp_path: Path) -> None:
    """The parser should fail fast on the removed component-selection surface."""

    payload = _minimal_cmass_config(tmp_path)
    payload["model"]["components"] = {
        "mass_definition": "m5_hinvkpc",
        "gamma_distribution": "sigma_star_dependent",
    }
    config_path = tmp_path / "components.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="model.components"):
        load_runtime_config(config_path)


def test_legacy_model_names_and_top_level_sections_are_rejected(tmp_path: Path) -> None:
    """Old registry keys and old top-level model sections should not parse."""

    legacy_name_payload = _minimal_cmass_config(tmp_path)
    legacy_name_payload["model"]["name"] = "cmass_current"
    legacy_name_path = tmp_path / "legacy_model_name.yaml"
    legacy_name_path.write_text(yaml.safe_dump(legacy_name_payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="cmass_current"):
        load_runtime_config(legacy_name_path)

    legacy_sections_payload = _minimal_cmass_config(tmp_path)
    legacy_sections_payload["mass_definition"] = {"enclosed_radius_kpc": 5}
    legacy_sections_payload["gamma_model"] = {"mode": "dependent"}
    legacy_sections_path = tmp_path / "legacy_sections.yaml"
    legacy_sections_path.write_text(yaml.safe_dump(legacy_sections_payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="mass_definition.*gamma_model"):
        load_runtime_config(legacy_sections_path)


def test_cmass_rejects_legacy_fixed_kpc_unit_convention(tmp_path: Path) -> None:
    """The default CMASS model is fixed to h_units_v1."""

    payload = _minimal_cmass_config(tmp_path)
    payload["unit_convention"] = "legacy_fixed_kpc"
    config_path = tmp_path / "legacy_units.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="h_units_v1"):
        load_runtime_config(config_path)


def test_legacy_sampling_fields_are_rejected(tmp_path: Path) -> None:
    """The emcee-only config parser should reject retired sampling names."""

    payload = _minimal_cmass_config(tmp_path)
    payload["sampling"].update(
        {
            "num_chains": 24,
            "num_samples": 3,
            "num_warmup": 1,
            "chain_method": "sequential",
            "thinning": 1,
            "warmup": 1,
        }
    )
    config_path = tmp_path / "legacy_sampling.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="num_chains.*num_samples.*num_warmup.*warmup"):
        load_runtime_config(config_path)


def test_model_registry_exposes_cmass_and_sonnenfeld() -> None:
    """Registry dispatch should expose every implemented concrete model."""

    cmass_model = get_model_definition("cmass")
    assert cmass_model.name == "cmass"

    sonnenfeld_model = get_model_definition("sonnenfeld2024_slacs")
    sonnenfeld_hunit_model = get_model_definition("sonnenfeld2024_slacs_hunit")

    assert sonnenfeld_model.name == "sonnenfeld2024_slacs"
    assert sonnenfeld_model.resolve_mass_definition(LEGACY_FIXED_KPC).label == "m5"
    assert sonnenfeld_model.required_capabilities[-1] == "velocity_dispersion.population_sigma_unit.v1"
    assert sonnenfeld_hunit_model.name == "sonnenfeld2024_slacs_hunit"
    assert sonnenfeld_hunit_model.resolve_mass_definition(H_UNITS_V1).label == "m5_hinvkpc"
    assert sonnenfeld_hunit_model.required_capabilities == sonnenfeld_model.required_capabilities

    toy_model = get_model_definition("toy_hierarchical")
    assert toy_model.name == "toy_hierarchical"
    assert toy_model.backend_kernel == "toy_hierarchical"
    assert callable(toy_model.evaluate_log_prob)


def test_model_registry_exposes_sonnenfeld_sigma_star_gamma_variants() -> None:
    """
    The sigma-star-gamma model should be a concrete peer of existing models.

    The test deliberately checks both unit-convention entrypoints because the
    implementation package must mirror the current Sonnenfeld pattern: one
    package supports paper-native and h-unit semantics, while the registry
    exposes two unambiguous model names.
    """

    paper_model = get_model_definition("sonnenfeld2024_slacs_sigma_star_gamma")
    hunit_model = get_model_definition("sonnenfeld2024_slacs_sigma_star_gamma_hunit")

    paper_schema = paper_model.build_parameter_schema(
        mass_definition=get_mass_definition(5, unit_convention=LEGACY_FIXED_KPC),
        public_box_prior=SONNENFELD_SIGMA_STAR_BOX_PRIOR,
    )
    hunit_schema = hunit_model.build_parameter_schema(
        mass_definition=get_mass_definition(5, unit_convention=H_UNITS_V1),
        public_box_prior=SONNENFELD_SIGMA_STAR_BOX_PRIOR,
    )
    original_schema = get_model_definition("sonnenfeld2024_slacs").build_parameter_schema(
        mass_definition=get_mass_definition(5, unit_convention=LEGACY_FIXED_KPC),
        public_box_prior=None,
    )
    original_hunit_schema = get_model_definition("sonnenfeld2024_slacs_hunit").build_parameter_schema(
        mass_definition=get_mass_definition(5, unit_convention=H_UNITS_V1),
        public_box_prior=None,
    )

    assert paper_model.name == "sonnenfeld2024_slacs_sigma_star_gamma"
    assert paper_model.resolve_mass_definition(LEGACY_FIXED_KPC).label == "m5"
    assert hunit_model.name == "sonnenfeld2024_slacs_sigma_star_gamma_hunit"
    assert hunit_model.resolve_mass_definition(H_UNITS_V1).label == "m5_hinvkpc"
    assert hunit_model.required_capabilities == paper_model.required_capabilities

    assert paper_schema.public_parameter_names == SONNENFELD_SIGMA_STAR_PUBLIC_PARAMETERS
    assert hunit_schema.public_parameter_names == SONNENFELD_SIGMA_STAR_PUBLIC_PARAMETERS
    assert paper_schema.n_dim == 11
    assert hunit_schema.n_dim == 11
    assert "beta_sigma_star_gamma" in paper_schema.public_parameter_names
    assert "beta_gamma" not in paper_schema.public_parameter_names
    assert "xi_gamma" not in paper_schema.public_parameter_names
    assert paper_schema.model_metadata["gamma_distribution"] == "sigma_star_dependent"
    assert hunit_schema.model_metadata["gamma_distribution"] == "sigma_star_dependent"

    assert original_schema.n_dim == 12
    assert original_hunit_schema.n_dim == 12
    assert "beta_sigma_star_gamma" not in original_schema.public_parameter_names
    assert "beta_gamma" in original_schema.public_parameter_names
    assert "xi_gamma" in original_schema.public_parameter_names


def test_cmass_model_file_exposes_high_level_model_spec() -> None:
    """CMASS should expose a human-authored spec instead of a backend definition."""

    model_spec = cmass.get_model_spec()
    runtime_definition = get_model_definition("cmass")

    assert model_spec.name == "cmass"
    assert model_spec.required_unit_convention == H_UNITS_V1
    assert model_spec.mass_aperture_kpc == 5
    assert model_spec.metadata["gamma_distribution"] == "sigma_star_dependent"
    assert model_spec.required_capabilities == (
        CAPABILITY_LENS_OBSERVATIONS_V1,
        CAPABILITY_LENSING_MASS_GRIDS_V1,
        CAPABILITY_LENSING_CROSS_SECTION_THETA_GAMMA_V1,
    )
    assert [parameter.public_name for parameter in model_spec.parameters][:4] == [
        "mu5h_0",
        "beta5h",
        "xi5h",
        "sigma5h",
    ]
    assert runtime_definition.name == model_spec.name


def test_generic_compiled_model_factory_builds_schema_without_cmass_fields() -> None:
    """The spec adapter should not depend on CMASS-specific parameter names."""

    model_spec = ModelSpec(
        name="toy_model",
        component_key="toy",
        required_unit_convention=H_UNITS_V1,
        mass_aperture_kpc=5,
        parameters=(
            ParameterSpec("alpha_internal", "alpha", (-1.0, 1.0)),
            ParameterSpec("scale_internal", "scale", (0.1, 3.0)),
        ),
        metadata={"purpose": "adapter-test"},
        required_capabilities=("toy.capability.v1",),
        optional_capabilities=(),
        static_codes={},
        backend_kernel="toy_kernel",
    )
    runtime_adapter = ModelRuntimeAdapter(
        build_context_bundle=lambda runtime_config: CompiledContextBundle(
            context=runtime_config,
            profile=None,
            cross_section_grid=None,
            cosmology=None,
            random_basis=None,
            observations=(),
        ),
        data_spec=cmass_runtime.get_data_spec(),
    )

    model_definition = build_model_definition(
        model_spec,
        runtime_adapter,
        lambda theta, compiled_model, total_start: (0.0, None),
    )
    parameter_schema = model_definition.build_parameter_schema(
        mass_definition=get_mass_definition(5, unit_convention=H_UNITS_V1),
        public_box_prior={"alpha": [-0.5, 0.5], "scale": [0.2, 2.0]},
    )

    assert model_definition.name == "toy_model"
    assert parameter_schema.internal_parameter_names == ("alpha_internal", "scale_internal")
    assert parameter_schema.public_parameter_names == ("alpha", "scale")
    assert parameter_schema.prior_bounds == ((-0.5, 0.5), (0.2, 2.0))
    assert parameter_schema.model_metadata == {"purpose": "adapter-test"}
    assert model_definition.required_capabilities == ("toy.capability.v1",)
    assert model_definition.backend_kernel == "toy_kernel"
    assert model_definition.evaluate_log_prob is not None


def _minimal_cmass_model_context() -> CMASSModelContext:
    """Build a tiny CMASS context for testing scalar packing only."""

    one = np.ones(1, dtype=np.float64)
    two_by_two = np.ones((1, 1), dtype=np.float64)
    return CMASSModelContext(
        z_grid=one,
        chi_kpc_grid=one,
        cs_gamma_grid=one,
        cs_over_theta_grid=one,
        cs_theta_e_axis=one,
        cs_cross_section_grid=two_by_two,
        cs_over_theta_int=one,
        gamma_grid_int=one,
        mass_grid_int=two_by_two,
        dmass_dthetaein_grid_int=two_by_two,
        s2_grid_int=two_by_two,
        has_s2=np.ones(1, dtype=np.int64),
        num_sigma=np.ones(1, dtype=np.int64),
        sigma_obs=np.ones((1, 2), dtype=np.float64),
        sigma_err=np.ones((1, 2), dtype=np.float64),
        zd=one,
        zs=one,
        p_zd_fixed=one,
        mstar_grid=two_by_two,
        mstar_shift11p4=two_by_two,
        stellar_mass_pivot=11.092783,
        sigma_star_shift9p0_grid=two_by_two,
        mstar_integrand_base=two_by_two,
        delta_r_grid=two_by_two,
        base_normals=np.ones((2, 8), dtype=np.float64),
        mass_radius_kpc=7.142857,
        mass_log_physical_offset=0.309804,
        use_sersic_index=1,
        n_fixed=4.0,
        mu_n0=0.1,
        beta_n=0.2,
        sigma_n=0.3,
        mass_function_loc=11.0,
        mass_function_scale=0.4,
        mass_function_alpha=-2.0,
        mu_r0=0.5,
        beta_r=0.6,
        sigma_r=0.7,
        nu_r=0.8,
        mu_d=0.558,
        sigma_d=0.085,
        gamma_trunc_low=1.2,
        gamma_trunc_high=2.8,
        normalization_min_value=1.0e-10,
        gamma_mode_code=2,
        fp_enabled=1,
        fp_fit_mstar_min=11.0,
        fp_pivot_mstar=11.3,
        fp_fiducial_scatter=0.075,
        fp_scatter_error=0.003,
        fp_mu_v_prior=2.34548,
        fp_mu_v_error=0.00611,
        fp_beta_v_prior=0.176,
        fp_beta_v_error=0.011,
        fp_gamma_axis=one,
        fp_zd_axis=one,
        fp_log_re_kpc_axis=one,
        fp_n_axis=one,
        fp_sigma_unit_grid=np.ones((1, 1, 1, 1), dtype=np.float64),
        fp_has_n_axis=1,
    )


def test_cmass_data_spec_preserves_scalar_field_order_and_values() -> None:
    """CMASS runtime declaration must preserve scalar field order and values."""

    raw_context = _minimal_cmass_model_context()
    data_spec = cmass_runtime.get_data_spec()
    scalar_values = np.asarray(
        [
            getattr(raw_context, scalar_spec.source_name)
            for scalar_spec in data_spec.scalar_fields
        ],
        dtype=np.float64,
    )

    expected = np.asarray(
        [
            raw_context.mass_radius_kpc,
            raw_context.n_fixed,
            raw_context.mu_n0,
            raw_context.beta_n,
            raw_context.sigma_n,
            raw_context.mass_function_loc,
            raw_context.mass_function_scale,
            raw_context.mass_function_alpha,
            raw_context.mu_r0,
            raw_context.beta_r,
            raw_context.sigma_r,
            raw_context.nu_r,
            raw_context.mu_d,
            raw_context.sigma_d,
            raw_context.gamma_trunc_low,
            raw_context.gamma_trunc_high,
            raw_context.normalization_min_value,
            raw_context.fp_fit_mstar_min,
            raw_context.fp_pivot_mstar,
            raw_context.fp_fiducial_scatter,
            raw_context.fp_scatter_error,
            raw_context.fp_mu_v_prior,
            raw_context.fp_mu_v_error,
            raw_context.fp_beta_v_prior,
            raw_context.fp_beta_v_error,
            raw_context.stellar_mass_pivot,
            raw_context.mass_log_physical_offset,
        ],
        dtype=np.float64,
    )
    np.testing.assert_allclose(scalar_values, expected)


def test_runtime_adapter_rejects_old_manual_backend_context_hooks() -> None:
    """Runtime adapters should no longer ask models to hand-write backend packing."""

    with pytest.raises(TypeError, match="static_backend_kwargs|to_backend_context|unexpected"):
        ModelRuntimeAdapter(
            build_compiled_model=lambda runtime_config: runtime_config,
            static_backend_kwargs=lambda compiled_model: {},
            to_backend_context=lambda compiled_model: compiled_model.context,
        )


def test_generic_compiled_model_factory_uses_spec_bounds_when_box_prior_is_absent() -> None:
    """A model spec should be able to provide its own default box prior."""

    model_definition = get_model_definition("cmass")
    parameter_schema = model_definition.build_parameter_schema(
        mass_definition=get_mass_definition(5, unit_convention=H_UNITS_V1),
        public_box_prior=None,
    )

    assert parameter_schema.public_parameter_names[:4] == ("mu5h_0", "beta5h", "xi5h", "sigma5h")
    assert parameter_schema.prior_bounds[:4] == (
        (9.0, 12.0),
        (-3.0, 3.0),
        (-3.0, 3.0),
        (1.0e-2, 0.2),
    )
