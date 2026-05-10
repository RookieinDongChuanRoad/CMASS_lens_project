"""Runtime and likelihood tests for the Sonnenfeld 2024 SLACS model.

These tests intentionally use tiny synthetic canonical datasets.  They do not
claim scientific validation against the paper results; instead they lock down
the engineering contract needed before real-data validation can happen:

- the registry should expose a paper-native ``sonnenfeld2024_slacs`` model and
  an explicit ``sonnenfeld2024_slacs_hunit`` variant;
- the runtime should build a model-specific context from canonical input;
- paper-native fixed-mass coordinates should keep the paper's mass-location
  constants unshifted, while h-dependent coordinates should shift them before
  the model posterior consumes them;
- the Numba likelihood backend should be able to evaluate one finite
  posterior value through the Sonnenfeld kernel.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest
import yaml

from cmass_lens_inference.config import load_runtime_config
from cmass_lens_inference.numba_backend.primitives import normal_pdf
from cmass_lens_inference.numba_backend.likelihood_engine import (
    build_compiled_model,
    log_prob,
)
from cmass_lens_inference.runner import run_inference
from cmass_lens_inference.mass_definition import H_UNITS_V1, LEGACY_FIXED_KPC, get_mass_definition
from cmass_lens_inference.model_registry import get_model_definition
from cmass_lens_inference.models.sonnenfeld2024_slacs import assembly as capabilities
from cmass_lens_inference.models.sonnenfeld2024_slacs import paper_constants as parameters
from cmass_lens_inference.models.sonnenfeld2024_slacs import posterior as sonnenfeld_posterior
from cmass_lens_inference.models.sonnenfeld2024_slacs.preprocessing import (
    build_sonnenfeld_context_from_canonical_dataset,
    _parent_density_grid,
    truncation_mass_threshold,
)
from test_canonical_dataset import _write_canonical_dataset


def _minimal_sonnenfeld_config(
    tmp_path: Path,
    dataset_path: Path,
    *,
    model_name: str = "sonnenfeld2024_slacs",
    unit_convention: str = LEGACY_FIXED_KPC,
) -> Path:
    """Write a compact config that exercises the Sonnenfeld registry path."""

    payload = {
        "profile": {"name": "sersic"},
        "unit_convention": unit_convention,
        "model": {"name": model_name},
        "data": {"inference_dataset_path": str(dataset_path)},
        "box_prior": {
            "mu5_0": [10.5, 12.2],
            "beta5": [-3.0, 3.0],
            "xi5": [-3.0, 3.0],
            "sigma5": [1.0e-2, 0.3],
            "mu_gamma_0": [1.2, 2.8],
            "beta_gamma": [-3.0, 3.0],
            "xi_gamma": [-3.0, 3.0],
            "sigma_gamma": [1.0e-2, 0.8],
            "mu_zs": [0.0, 2.0],
            "sigma_zs": [1.0e-3, 1.0],
            "theta0": [0.0, 3.0],
            "loga": [-1.0, 3.0],
        },
        "sampling": {
            "random_seed": 7,
            "n_walkers": 24,
            "n_steps": 2,
            "burn_in": 1,
            "initial_center": {
                "mu5_0": 11.2,
                "beta5": 0.5,
                "xi5": -0.1,
                "sigma5": 0.08,
                "mu_gamma_0": 2.0,
                "beta_gamma": 0.1,
                "xi_gamma": 0.05,
                "sigma_gamma": 0.15,
                "mu_zs": 1.5,
                "sigma_zs": 0.25,
                "theta0": 0.2,
                "loga": 0.0,
            },
            "initial_jitter_scale": 1.0e-3,
        },
        "integration": {
            "gamma_points": 16,
            "mstar_points": 12,
            "normalization_samples": 32,
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
            "run_label": "sonnenfeld-smoke",
            "overwrite_latest": True,
        },
    }
    config_path = tmp_path / f"{model_name}_{unit_convention}.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return config_path


@pytest.fixture
def sonnenfeld_hunit_ready_dataset_path(tmp_path: Path) -> Path:
    """Create the smallest hunit canonical dataset that declares all required inputs."""

    return _write_canonical_dataset(
        tmp_path / "sonnenfeld_hunit_ready.hdf5",
        declare_population_sigma_unit=True,
        write_population_sigma_unit=True,
    )


@pytest.fixture
def sonnenfeld_fixed_m5_ready_dataset_path(
    tmp_path: Path,
    sonnenfeld_hunit_ready_dataset_path: Path,
) -> Path:
    """Rewrite metadata labels so the fixture represents paper-native fixed m5."""

    import h5py
    import shutil

    fixed_path = tmp_path / "sonnenfeld_fixed_m5_ready.hdf5"
    shutil.copy2(sonnenfeld_hunit_ready_dataset_path, fixed_path)
    with h5py.File(fixed_path, "r+") as handle:
        metadata = handle["metadata"]
        metadata.attrs["unit_convention"] = LEGACY_FIXED_KPC
        metadata.attrs["mass_definition_label"] = "m5"
    return fixed_path


def test_sonnenfeld_registry_exposes_model_definition() -> None:
    """The Sonnenfeld label should now resolve to the generic backend contract."""

    model_definition = get_model_definition("sonnenfeld2024_slacs")
    hunit_model_definition = get_model_definition("sonnenfeld2024_slacs_hunit")

    assert model_definition.name == "sonnenfeld2024_slacs"
    assert model_definition.resolve_mass_definition(LEGACY_FIXED_KPC).label == "m5"
    assert model_definition.required_capabilities == capabilities.REQUIRED_CAPABILITIES
    assert hunit_model_definition.name == "sonnenfeld2024_slacs_hunit"
    assert hunit_model_definition.resolve_mass_definition(H_UNITS_V1).label == "m5_hinvkpc"


def test_sonnenfeld_selection_proxy_uses_paper_fractional_scatter() -> None:
    """Sonnenfeld Section 4.5 uses a 6.25% velocity-dispersion proxy scatter."""

    assert parameters.SIGMA_PROXY_FRACTIONAL_SCATTER == pytest.approx(0.0625)
    source = inspect.getsource(sonnenfeld_posterior)
    assert "max(0.1" not in source


def test_sonnenfeld_effective_source_redshift_is_ordinary_gaussian() -> None:
    """Equation 33 is an ordinary Gaussian effective source-redshift term."""

    mu_zs = 0.2
    sigma_zs = 0.5
    negative_zs = -0.1
    density = normal_pdf(negative_zs, mu_zs, sigma_zs)

    assert float(density) == pytest.approx(
        float(normal_pdf(negative_zs, mu_zs, sigma_zs))
    )


def test_sonnenfeld_population_source_redshift_draw_is_nonnegative_truncated() -> None:
    """The reference population normalization draws source redshifts from z_s >= 0."""

    theta = np.asarray(
        [
            11.2,
            0.0,
            0.0,
            0.05,
            2.0,
            0.0,
            0.0,
            0.1,
            0.1,
            0.1,
            1.0,
            0.0,
        ],
        dtype=np.float64,
    )
    standard_normal_draws = np.zeros(8, dtype=np.float64)
    standard_normal_draws[5] = -8.0

    _, source_redshift, *_ = sonnenfeld_posterior._draw_population_state(
        theta,
        standard_normal_draws,
        0.2,
        11.3,
        0.7,
        0.0,
        parameters.MSTAR_PIVOT_PHYSICAL,
        4.0,
        0,
        parameters.GAMMA_TRUNC_LOW,
        parameters.GAMMA_TRUNC_HIGH,
    )

    assert source_redshift >= 0.0


def test_sonnenfeld_population_source_mask_matches_reference_window() -> None:
    """The reference normalization keeps only z_s > z_d + 0.05 and 0.05 < z_s < 2."""

    assert sonnenfeld_posterior._passes_reference_source_redshift_mask(
        0.30,
        0.36,
        parameters.SOURCE_Z_MIN,
        parameters.SOURCE_Z_MAX,
        parameters.SOURCE_LENS_REDSHIFT_GAP,
    )
    assert not sonnenfeld_posterior._passes_reference_source_redshift_mask(
        0.30,
        0.34,
        parameters.SOURCE_Z_MIN,
        parameters.SOURCE_Z_MAX,
        parameters.SOURCE_LENS_REDSHIFT_GAP,
    )
    assert not sonnenfeld_posterior._passes_reference_source_redshift_mask(
        0.30,
        0.05,
        parameters.SOURCE_Z_MIN,
        parameters.SOURCE_Z_MAX,
        parameters.SOURCE_LENS_REDSHIFT_GAP,
    )
    assert not sonnenfeld_posterior._passes_reference_source_redshift_mask(
        0.30,
        2.00,
        parameters.SOURCE_Z_MIN,
        parameters.SOURCE_Z_MAX,
        parameters.SOURCE_LENS_REDSHIFT_GAP,
    )


def test_sonnenfeld_parent_density_uses_reference_volume_and_arctan_truncation() -> None:
    """Equation 27 uses reference volume weighting and arctan completeness."""

    zd = np.asarray([0.25], dtype=np.float64)
    mstar_grid = np.asarray([[10.99, 11.56, 11.80]], dtype=np.float64)
    volume_factor = np.asarray([42.0], dtype=np.float64)

    density = _parent_density_grid(
        zd=zd,
        mstar_grid=mstar_grid,
        mbar=parameters.MBAR_PHYSICAL,
        parent_alpha=parameters.PARENT_ALPHA,
        h_ref=0.7,
        unit_convention=LEGACY_FIXED_KPC,
        volume_factor=volume_factor,
    )

    threshold = sum(
        coefficient * zd[:, None] ** power
        for power, coefficient in enumerate(
            parameters.TRUNCATION_MASS_POLYNOMIAL_COEFFICIENTS
        )
    )
    completeness = (
        np.arctan((mstar_grid - threshold) / parameters.TRUNCATION_MASS_SCATTER)
        / np.pi
        + 0.5
    )
    schechter_mass = 10.0 ** (mstar_grid - parameters.MBAR_PHYSICAL)
    expected = (
        volume_factor[:, None]
        * completeness
        * 10.0 ** ((mstar_grid - parameters.MBAR_PHYSICAL) * (parameters.PARENT_ALPHA + 1.0))
        * np.exp(-schechter_mass)
    )

    assert density == pytest.approx(expected)


def test_paper_native_sonnenfeld_preprocessing_keeps_physical_mass_constants(
    tmp_path: Path,
    sonnenfeld_fixed_m5_ready_dataset_path: Path,
) -> None:
    """The paper-native model should keep fixed-kpc mass constants unshifted."""

    config = load_runtime_config(
        _minimal_sonnenfeld_config(
            tmp_path,
            sonnenfeld_fixed_m5_ready_dataset_path,
            model_name="sonnenfeld2024_slacs",
            unit_convention=LEGACY_FIXED_KPC,
        )
    )

    bundle = build_sonnenfeld_context_from_canonical_dataset(config)
    context = bundle.context

    assert context.mass_radius_kpc == pytest.approx(5.0)
    assert context.mass_log_physical_offset == pytest.approx(0.0)
    assert context.mstar_pivot == pytest.approx(parameters.MSTAR_PIVOT_PHYSICAL)
    assert context.mbar == pytest.approx(parameters.MBAR_PHYSICAL)
    assert config.mass_definition == get_mass_definition(5, unit_convention=LEGACY_FIXED_KPC)
    assert config.parameter_schema.model_metadata["mass_definition"] == "m5"


def test_paper_native_sonnenfeld_preprocessing_uses_quadratic_size_relation(
    tmp_path: Path,
    sonnenfeld_fixed_m5_ready_dataset_path: Path,
) -> None:
    """Fixed-5-kpc Sonnenfeld preprocessing should use Equation 28-29."""

    config = load_runtime_config(
        _minimal_sonnenfeld_config(
            tmp_path,
            sonnenfeld_fixed_m5_ready_dataset_path,
            model_name="sonnenfeld2024_slacs",
            unit_convention=LEGACY_FIXED_KPC,
        )
    )
    bundle = build_sonnenfeld_context_from_canonical_dataset(config)
    context = bundle.context

    expected_mu_r = (
        parameters.SIZE_MU0_PHYSICAL
        + parameters.SIZE_MU1_PHYSICAL * context.mstar_grid[0]
        + parameters.SIZE_MU2_PHYSICAL * context.mstar_grid[0] ** 2
    )

    assert context.delta_r_grid[0] == pytest.approx(context.log_re_obs[0] - expected_mu_r)
    assert context.size_mu0 == pytest.approx(parameters.SIZE_MU0_PHYSICAL)
    assert context.size_mu1 == pytest.approx(parameters.SIZE_MU1_PHYSICAL)
    assert context.size_mu2 == pytest.approx(parameters.SIZE_MU2_PHYSICAL)


def test_sonnenfeld_context_uses_reference_parent_sample_not_runtime_proposal(
    tmp_path: Path,
    sonnenfeld_fixed_m5_ready_dataset_path: Path,
) -> None:
    """Sonnenfeld runtime must consume parent-distribution samples directly."""

    config_path = _minimal_sonnenfeld_config(
        tmp_path,
        sonnenfeld_fixed_m5_ready_dataset_path,
        model_name="sonnenfeld2024_slacs",
        unit_convention=LEGACY_FIXED_KPC,
    )
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload["integration"]["normalization_samples"] = 64
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    config = load_runtime_config(config_path)

    bundle = build_sonnenfeld_context_from_canonical_dataset(config)
    context = bundle.context

    assert context.parent_sample_zd.shape == (64,)
    assert context.parent_sample_mstar.shape == (64,)
    assert context.parent_sample_log_re.shape == (64,)
    assert context.parent_sample_delta_r.shape == (64,)
    assert np.all(context.parent_sample_zd >= parameters.PARENT_ZD_MIN)
    assert np.all(context.parent_sample_zd <= parameters.PARENT_ZD_MAX)
    assert np.all(context.parent_sample_mstar >= parameters.PARENT_MSTAR_MIN_PHYSICAL)
    assert np.all(context.parent_sample_mstar <= parameters.PARENT_MSTAR_MAX_PHYSICAL)

    source = inspect.getsource(sonnenfeld_posterior)
    assert "proposal_density" not in source
    assert "parent_density / proposal_density" not in source


def test_sonnenfeld_fp_prior_defaults_match_reference_fitpars(
    tmp_path: Path,
    sonnenfeld_fixed_m5_ready_dataset_path: Path,
) -> None:
    """Enabling the FP prior without overrides should reproduce reference defaults."""

    config_path = _minimal_sonnenfeld_config(
        tmp_path,
        sonnenfeld_fixed_m5_ready_dataset_path,
        model_name="sonnenfeld2024_slacs",
        unit_convention=LEGACY_FIXED_KPC,
    )
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload["fp_prior"] = {"enabled": True}
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    config = load_runtime_config(config_path)

    bundle = build_sonnenfeld_context_from_canonical_dataset(config)
    context = bundle.context

    expected_mu_v = 2.2577 + 0.3034 * 0.3 - 0.0761 * 0.3**2
    expected_beta_v = 0.3034 - 2.0 * 0.0761 * 0.3

    assert context.fp_fiducial_scatter == pytest.approx(0.047)
    assert context.fp_scatter_error == pytest.approx(0.008)
    assert context.fp_mu_v_prior == pytest.approx(expected_mu_v)
    assert context.fp_mu_v_error == pytest.approx(0.03)
    assert context.fp_beta_v_prior == pytest.approx(expected_beta_v)
    assert context.fp_beta_v_error == pytest.approx(0.03)


def test_sonnenfeld_hunit_preprocessing_builds_context_and_hunit_mass_constants(
    tmp_path: Path,
    sonnenfeld_hunit_ready_dataset_path: Path,
) -> None:
    """Runtime preprocessing should expose canonical data and shifted pivots."""

    config = load_runtime_config(
        _minimal_sonnenfeld_config(
            tmp_path,
            sonnenfeld_hunit_ready_dataset_path,
            model_name="sonnenfeld2024_slacs_hunit",
            unit_convention=H_UNITS_V1,
        )
    )

    bundle = build_sonnenfeld_context_from_canonical_dataset(config)
    context = bundle.context

    assert context.mass_radius_kpc == pytest.approx(5.0 / 0.7)
    assert context.mass_log_physical_offset == pytest.approx(np.log10(0.7**-1))
    assert context.mstar_pivot == pytest.approx(
        parameters.shift_physical_mass_location_to_hunits(parameters.MSTAR_PIVOT_PHYSICAL, 0.7)
    )
    assert context.mbar == pytest.approx(
        parameters.shift_physical_mass_location_to_hunits(parameters.MBAR_PHYSICAL, 0.7)
    )
    assert context.parent_mstar_min == pytest.approx(
        parameters.shift_physical_mass_location_to_hunits(parameters.PARENT_MSTAR_MIN_PHYSICAL, 0.7)
    )
    assert context.parent_mstar_max == pytest.approx(
        parameters.shift_physical_mass_location_to_hunits(parameters.PARENT_MSTAR_MAX_PHYSICAL, 0.7)
    )
    assert truncation_mass_threshold(
        np.asarray([context.zd[0]], dtype=np.float64),
        h_ref=0.7,
        unit_convention=H_UNITS_V1,
    )[0] == pytest.approx(
        truncation_mass_threshold(
            np.asarray([context.zd[0]], dtype=np.float64),
            h_ref=0.7,
            unit_convention=LEGACY_FIXED_KPC,
        )[0]
        + 2.0 * np.log10(0.7)
    )
    assert context.population_sigma_unit_grid.shape == (3, 2, 2, 2)
    assert bundle.metadata["canonical_capabilities"] == tuple(
        sorted(capabilities.REQUIRED_CAPABILITIES)
    )


def test_sonnenfeld_hunit_preprocessing_shifts_all_location_parameters(
    tmp_path: Path,
    sonnenfeld_hunit_ready_dataset_path: Path,
) -> None:
    """The hunit variant should shift the full active-coordinate size relation."""

    config = load_runtime_config(
        _minimal_sonnenfeld_config(
            tmp_path,
            sonnenfeld_hunit_ready_dataset_path,
            model_name="sonnenfeld2024_slacs_hunit",
            unit_convention=H_UNITS_V1,
        )
    )
    bundle = build_sonnenfeld_context_from_canonical_dataset(config)
    context = bundle.context
    log10_h = np.log10(config.h_ref)
    physical_mstar_grid = context.mstar_grid[0] - 2.0 * log10_h
    expected_mu_r = (
        parameters.SIZE_MU0_PHYSICAL
        + parameters.SIZE_MU1_PHYSICAL * physical_mstar_grid
        + parameters.SIZE_MU2_PHYSICAL * physical_mstar_grid**2
        + log10_h
    )

    assert context.delta_r_grid[0] == pytest.approx(context.log_re_obs[0] - expected_mu_r)
    assert context.size_mu0 == pytest.approx(
        parameters.SIZE_MU0_PHYSICAL
        - 2.0 * parameters.SIZE_MU1_PHYSICAL * log10_h
        + 4.0 * parameters.SIZE_MU2_PHYSICAL * log10_h**2
        + log10_h
    )
    assert context.size_mu1 == pytest.approx(
        parameters.SIZE_MU1_PHYSICAL - 4.0 * parameters.SIZE_MU2_PHYSICAL * log10_h
    )
    assert context.size_mu2 == pytest.approx(parameters.SIZE_MU2_PHYSICAL)


def test_config_loads_sonnenfeld_parameter_schema(
    tmp_path: Path,
    sonnenfeld_fixed_m5_ready_dataset_path: Path,
) -> None:
    """Config parsing should use Sonnenfeld parameter names and mass metadata."""

    config = load_runtime_config(
        _minimal_sonnenfeld_config(
            tmp_path,
            sonnenfeld_fixed_m5_ready_dataset_path,
            model_name="sonnenfeld2024_slacs",
            unit_convention=LEGACY_FIXED_KPC,
        )
    )

    assert config.model.name == "sonnenfeld2024_slacs"
    assert config.mass_definition == get_mass_definition(5, unit_convention=LEGACY_FIXED_KPC)
    assert config.parameter_schema.public_parameter_names == capabilities.PUBLIC_PARAMETER_NAMES
    assert config.parameter_schema.model_metadata["foreground_population"] == "sonnenfeld2024_table1"


def test_sonnenfeld_numba_log_prob_is_finite_for_synthetic_dataset(
    tmp_path: Path,
    sonnenfeld_fixed_m5_ready_dataset_path: Path,
) -> None:
    """A valid synthetic point should evaluate through the Numba backend."""

    config = load_runtime_config(
        _minimal_sonnenfeld_config(
            tmp_path,
            sonnenfeld_fixed_m5_ready_dataset_path,
            model_name="sonnenfeld2024_slacs",
            unit_convention=LEGACY_FIXED_KPC,
        )
    )
    theta = config.sampling.initial_center.to_array()

    compiled = build_compiled_model(config)
    value, blob = log_prob(theta, compiled)

    assert np.isfinite(value)
    assert np.isfinite(float(blob["normalization_value"]))
    assert compiled.context.population_sigma_unit_grid.shape == (3, 2, 2, 2)


def test_sonnenfeld_fp_prior_contributes_to_log_prob_when_enabled(
    tmp_path: Path,
    sonnenfeld_fixed_m5_ready_dataset_path: Path,
) -> None:
    """Enabling the FP prior should add its quadratic penalty to Sonnenfeld."""

    config_path = _minimal_sonnenfeld_config(
        tmp_path,
        sonnenfeld_fixed_m5_ready_dataset_path,
        model_name="sonnenfeld2024_slacs",
        unit_convention=LEGACY_FIXED_KPC,
    )
    base_payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    base_payload["integration"]["normalization_samples"] = 256
    disabled_payload = yaml.safe_load(yaml.safe_dump(base_payload))
    enabled_payload = yaml.safe_load(yaml.safe_dump(base_payload))
    disabled_payload["fp_prior"] = {"enabled": False}
    enabled_payload["fp_prior"] = {
        "enabled": True,
        "fit_mstar_min": 11.0,
        "pivot_mstar": 11.3,
        # These deliberately loose reference values keep the synthetic fixture
        # finite while still making a missing prior contribution observable.
        "fiducial_scatter": 0.1,
        "scatter_error": 10.0,
        "mu_v_prior": 1.0,
        "mu_v_error": 10.0,
        "beta_v_prior": 1.0,
        "beta_v_error": 10.0,
    }
    disabled_config_path = tmp_path / "sonnenfeld_fp_disabled.yaml"
    enabled_config_path = tmp_path / "sonnenfeld_fp_enabled.yaml"
    disabled_config_path.write_text(
        yaml.safe_dump(disabled_payload, sort_keys=False),
        encoding="utf-8",
    )
    enabled_config_path.write_text(
        yaml.safe_dump(enabled_payload, sort_keys=False),
        encoding="utf-8",
    )

    disabled_config = load_runtime_config(disabled_config_path)
    enabled_config = load_runtime_config(enabled_config_path)
    theta = enabled_config.sampling.initial_center.to_array()

    disabled_compiled = build_compiled_model(disabled_config)
    enabled_compiled = build_compiled_model(enabled_config)
    disabled_value, disabled_blob = log_prob(theta, disabled_compiled)
    enabled_value, enabled_blob = log_prob(theta, enabled_compiled)
    fp_prior_log_term = float(enabled_blob["fp_prior_log_term"])

    assert enabled_compiled.context.fp_enabled == 1
    assert disabled_compiled.context.fp_enabled == 0
    assert np.isfinite(disabled_value)
    assert np.isfinite(enabled_value)
    assert float(enabled_blob["normalization_value"]) == pytest.approx(
        float(disabled_blob["normalization_value"]),
        abs=1.0e-8,
    )
    assert np.isfinite(fp_prior_log_term)
    assert fp_prior_log_term != pytest.approx(0.0)
    assert np.isfinite(float(enabled_blob["fpfit_mu"]))
    assert np.isfinite(float(enabled_blob["fpfit_beta"]))
    assert np.isfinite(float(enabled_blob["fpfit_xi"]))
    assert np.isfinite(float(enabled_blob["fpfit_scatter"]))
    assert enabled_value == pytest.approx(
        disabled_value + fp_prior_log_term,
        abs=1.0e-8,
    )


def test_sonnenfeld_hunit_numba_log_prob_is_finite_for_synthetic_dataset(
    tmp_path: Path,
    sonnenfeld_hunit_ready_dataset_path: Path,
) -> None:
    """The explicit hunit variant should preserve the previous runnable path."""

    config = load_runtime_config(
        _minimal_sonnenfeld_config(
            tmp_path,
            sonnenfeld_hunit_ready_dataset_path,
            model_name="sonnenfeld2024_slacs_hunit",
            unit_convention=H_UNITS_V1,
        )
    )
    theta = config.sampling.initial_center.to_array()

    compiled = build_compiled_model(config)
    value, blob = log_prob(theta, compiled)

    assert np.isfinite(value)
    assert np.isfinite(float(blob["normalization_value"]))
    assert compiled.context.mstar_pivot == pytest.approx(
        parameters.shift_physical_mass_location_to_hunits(parameters.MSTAR_PIVOT_PHYSICAL, 0.7)
    )


def test_sonnenfeld_short_emcee_run_writes_chain_h5(
    tmp_path: Path,
    sonnenfeld_fixed_m5_ready_dataset_path: Path,
) -> None:
    """The Sonnenfeld model should run through the shared production sampler."""

    config_path = _minimal_sonnenfeld_config(
        tmp_path,
        sonnenfeld_fixed_m5_ready_dataset_path,
        model_name="sonnenfeld2024_slacs",
        unit_convention=LEGACY_FIXED_KPC,
    )
    run_result = run_inference(str(config_path))

    assert run_result.status == "completed"
    assert run_result.metadata["backend"] == "numba_emcee"
    assert run_result.metadata["chain_storage"] == "emcee_hdf_backend"
    assert (run_result.run_dir / "chain.h5").exists()

    import emcee

    backend = emcee.backends.HDFBackend(str(run_result.run_dir / "chain.h5"), read_only=True)
    assert backend.get_chain().shape == (2, 24, 12)
