"""Tests for the concrete-model registry configuration contract."""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import numpy as np
import pytest
import yaml

from cmass_lens_inference.jax_backend.model_adapter import build_model_definition
from cmass_lens_inference.jax_backend.context_builder import (
    build_jax_context_from_data_spec,
    static_jit_kwargs_from_data_spec,
)
from cmass_lens_inference.canonical_dataset import (
    CAPABILITY_LENSING_CROSS_SECTION_THETA_GAMMA_V1,
    CAPABILITY_LENSING_MASS_GRIDS_V1,
    CAPABILITY_LENS_OBSERVATIONS_V1,
)
from cmass_lens_inference.config import load_runtime_config
from cmass_lens_inference.models import cmass, cmass_runtime
from cmass_lens_inference.models.cmass_context import CMASSModelContext
from cmass_lens_inference.mass_definition import H_UNITS_V1, get_mass_definition
from cmass_lens_inference.model_interfaces import (
    CompiledContextBundle,
    ContextArraySpec,
    ContextScalarSpec,
    DataSpec,
    ModelRuntimeAdapter,
    ModelSpec,
    ParameterSpec,
    StaticContextSpec,
)
from cmass_lens_inference.model_registry import get_model_definition


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
            "num_chains": 24,
            "num_samples": 3,
            "num_warmup": 1,
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
    """The NumPyro-only config parser should reject emcee-era sampling names."""

    payload = _minimal_cmass_config(tmp_path)
    payload["sampling"].update({"n_walkers": 24, "n_steps": 3, "warmup": 1})
    config_path = tmp_path / "legacy_sampling.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="n_walkers.*n_steps.*warmup"):
        load_runtime_config(config_path)


def test_model_registry_exposes_cmass_and_blocks_unimplemented_sonnenfeld() -> None:
    """Registry dispatch should be explicit about implemented models."""

    cmass_model = get_model_definition("cmass")
    assert cmass_model.name == "cmass"

    with pytest.raises(NotImplementedError, match="sonnenfeld2024_slacs"):
        get_model_definition("sonnenfeld2024_slacs")


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


def test_generic_model_adapter_builds_schema_without_cmass_fields() -> None:
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
        unpack_theta=lambda theta: theta,
        validate_theta=lambda theta, theta_parts, context, static: True,
        draw_population=lambda theta_parts, nrm, context, static: nrm,
        selection_weight=lambda theta_parts, draw, nrm, context, static: 1.0,
        summary_row=lambda theta_parts, draw, context, static: draw,
        lens_integrals=lambda theta_parts, context, static: context.lens_integrals,
        extra_prior=lambda fp_summary, context, static: (0.0, 0.0, 0.0, 0.0, 0.0),
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
        data_spec=DataSpec(
            jax_context_type=ToyJaxContext,
            array_fields=(),
            scalar_fields=(),
            static_fields=(),
            normalization_samples_field="base_normals",
            normalization_min_value_field="normalization_min_value",
        ),
    )

    model_definition = build_model_definition(model_spec, runtime_adapter)
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


class ToyJaxContext(NamedTuple):
    """Small context type used to prove the generic builder is model-agnostic."""

    signal: object
    scalar_context: object


class ToyRawContext(NamedTuple):
    """Raw NumPy context with deliberately non-CMASS field names."""

    signal_numpy: np.ndarray
    scale: float
    offset: float
    use_feature: int
    base_normals: np.ndarray
    normalization_floor: float


def test_data_spec_builds_jax_context_and_static_flags_without_cmass_fields() -> None:
    """The generic context builder should only depend on declarative specs."""

    raw_context = ToyRawContext(
        signal_numpy=np.asarray([1.0, 2.0, 3.0], dtype=np.float64),
        scale=4.0,
        offset=-1.5,
        use_feature=1,
        base_normals=np.zeros((2, 3), dtype=np.float64),
        normalization_floor=1.0e-8,
    )
    data_spec = DataSpec(
        jax_context_type=ToyJaxContext,
        array_fields=(ContextArraySpec(source_name="signal_numpy", target_name="signal"),),
        scalar_fields=(
            ContextScalarSpec(source_name="scale"),
            ContextScalarSpec(source_name="offset"),
        ),
        static_fields=(StaticContextSpec(source_name="use_feature", target_name="feature_enabled"),),
        normalization_samples_field="base_normals",
        normalization_min_value_field="normalization_floor",
    )

    jax_context = build_jax_context_from_data_spec(raw_context, data_spec)
    static_kwargs = static_jit_kwargs_from_data_spec(raw_context, data_spec)

    np.testing.assert_allclose(np.asarray(jax_context.signal), np.asarray([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(np.asarray(jax_context.scalar_context), np.asarray([4.0, -1.5]))
    assert static_kwargs == {"feature_enabled": 1}


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


def test_cmass_data_spec_preserves_scalar_context_order_and_values() -> None:
    """CMASS scalar packing must stay byte-for-byte compatible with old hooks."""

    raw_context = _minimal_cmass_model_context()
    data_spec = cmass_runtime.get_data_spec()
    jax_context = build_jax_context_from_data_spec(raw_context, data_spec)

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
    np.testing.assert_allclose(np.asarray(jax_context.scalar_context), expected)


def test_runtime_adapter_rejects_old_manual_context_hooks() -> None:
    """Runtime adapters should no longer ask models to hand-write JAX packing."""

    with pytest.raises(TypeError, match="static_jit_kwargs|to_jax_context|unexpected"):
        ModelRuntimeAdapter(
            build_compiled_model=lambda runtime_config: runtime_config,
            static_jit_kwargs=lambda compiled_model: {},
            to_jax_context=lambda compiled_model: compiled_model.context,
        )


def test_generic_model_adapter_uses_spec_bounds_when_box_prior_is_absent() -> None:
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
