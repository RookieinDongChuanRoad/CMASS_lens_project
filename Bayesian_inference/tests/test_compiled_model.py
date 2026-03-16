"""
Tests for the second-stage performance refactor.

These tests deliberately lock the architectural shape before implementation:
- public numerical primitives must live in `kernels/primitives.py`
- the production log-probability path must build a compiled array context
- likelihood and normalization must execute through monolithic numba kernels

The scientific model is still exercised on tiny synthetic data. That keeps the
tests fast while proving that the new performance-oriented structure is really
in use.
"""

from __future__ import annotations

from pathlib import Path

from numba.core.registry import CPUDispatcher
import numpy as np
import pytest
import yaml

from cmass_lens_inference.config import load_runtime_config
from cmass_lens_inference.model import build_compiled_model, log_prob
from cmass_lens_inference.kernels.likelihood import log_likelihood_lenses_numba
from cmass_lens_inference.kernels.normalization import normalization_mc_numba
from cmass_lens_inference.kernels.primitives import (
    normal_pdf,
    p_find,
    theta_ein_arcsec,
    truncated_normal_pdf_nonneg,
)
from cmass_lens_inference.kernels.primitives import (
    interp1d_clip,
    normal_ppf,
    skewnorm_sample,
    truncnorm_sample,
)


def test_compiled_model_builds_contiguous_array_context(synthetic_config_path) -> None:
    """
    The new compiled model builder must convert the observation list into
    contiguous arrays so that numba kernels can consume the full dataset in one
    pass without Python object dispatch.
    """

    runtime_config = load_runtime_config(synthetic_config_path)
    compiled_model = build_compiled_model(runtime_config)

    assert compiled_model.context.zd.ndim == 1
    assert compiled_model.context.mass_grid_int.ndim == 2
    assert compiled_model.context.dmass_dthetaein_grid_int.ndim == 2
    assert compiled_model.context.mstar_integrand_base.ndim == 2
    assert compiled_model.context.base_normals.ndim == 2
    assert compiled_model.context.mass_radius_kpc == 5.0
    assert compiled_model.context.zd.flags.c_contiguous
    assert compiled_model.context.mass_grid_int.flags.c_contiguous
    assert compiled_model.context.dmass_dthetaein_grid_int.flags.c_contiguous
    assert compiled_model.context.mstar_integrand_base.flags.c_contiguous
    assert compiled_model.context.base_normals.flags.c_contiguous
    assert compiled_model.context.mstar_integrand_base.dtype == np.float64
    assert compiled_model.context.z_grid.dtype == np.float64
    assert compiled_model.context.chi_kpc_grid.dtype == np.float64
    assert not hasattr(compiled_model.context.z_grid, "unit")
    assert not hasattr(compiled_model.context.chi_kpc_grid, "unit")
    assert not hasattr(compiled_model.context, "gamma_w")
    assert not hasattr(compiled_model.context, "n_obs_for_model")
    assert compiled_model.context.gamma_mode_code == 0


def test_compiled_model_tracks_selected_mass_definition_in_context(synthetic_m10_config_path) -> None:
    """
    The compiled context must carry the chosen mass definition into the hot path.

    This keeps the kernels generic: they receive one mass grid and one radius
    value instead of a special-case `m5` implementation.
    """

    runtime_config = load_runtime_config(synthetic_m10_config_path)
    compiled_model = build_compiled_model(runtime_config)

    assert compiled_model.context.mass_radius_kpc == 10.0
    assert compiled_model.context.mass_grid_int.shape[0] == 1
    assert compiled_model.context.dmass_dthetaein_grid_int.shape == compiled_model.context.mass_grid_int.shape


def test_model_log_prob_runs_through_monolithic_numba_kernels(synthetic_config_path) -> None:
    """
    The production `log_prob` entrypoint must compile and execute both the
    all-lens likelihood kernel and the normalization kernel.
    """

    runtime_config = load_runtime_config(synthetic_config_path)
    compiled_model = build_compiled_model(runtime_config)

    log_prob_value, blob = log_prob(
        runtime_config.sampling.initial_center.to_array(),
        compiled_model,
    )

    assert isinstance(log_likelihood_lenses_numba, CPUDispatcher)
    assert isinstance(normalization_mc_numba, CPUDispatcher)
    assert log_likelihood_lenses_numba.signatures
    assert normalization_mc_numba.signatures
    assert np.isfinite(log_prob_value)
    assert blob.dtype.names is not None
    assert set(blob.dtype.names) >= {
        "total_log_prob_seconds",
        "likelihood_seconds",
        "normalization_seconds",
        "normalization_value",
        "parallel_strategy",
    }
    assert blob["parallel_strategy"].decode("utf-8").rstrip("\x00") == compiled_model.parallelism.strategy


def test_independent_gamma_mode_uses_ten_dimensional_theta_vector(
    synthetic_independent_config_path,
) -> None:
    """
    The independent gamma mode should flow through the compiled model as 10-D.

    This test locks the runtime contract so the compiled model, not just the
    config loader, honors the reduced sampled parameter space.
    """

    runtime_config = load_runtime_config(synthetic_independent_config_path)
    compiled_model = build_compiled_model(runtime_config)
    theta = runtime_config.sampling.initial_center.to_array()

    assert theta.shape == (10,)
    assert compiled_model.context.gamma_mode_code == 1

    log_prob_value, _ = log_prob(theta, compiled_model)
    assert np.isfinite(log_prob_value)


def test_independent_gamma_mode_matches_zero_slope_dependent_log_prob(
    synthetic_config_path,
    synthetic_independent_config_path,
    tmp_path,
) -> None:
    """
    Independent gamma mode should match dependent mode with zero gamma slopes.

    This is the key scientific regression guard for the new parameterization:
    removing the sampled gamma slopes should change the theta dimension, but it
    must not change the implied likelihood when the dependent model slopes are
    explicitly set to zero.
    """

    dependent_payload = yaml.safe_load(Path(synthetic_config_path).read_text(encoding="utf-8"))
    dependent_payload["sampling"]["initial_center"]["beta_gamma"] = 0.0
    dependent_payload["sampling"]["initial_center"]["xi_gamma"] = 0.0
    dependent_zero_path = tmp_path / "dependent_zero_slopes.yaml"
    dependent_zero_path.write_text(
        yaml.safe_dump(dependent_payload, sort_keys=False),
        encoding="utf-8",
    )

    dependent_runtime_config = load_runtime_config(dependent_zero_path)
    independent_runtime_config = load_runtime_config(synthetic_independent_config_path)
    dependent_compiled_model = build_compiled_model(dependent_runtime_config)
    independent_compiled_model = build_compiled_model(independent_runtime_config)

    dependent_log_prob, _ = log_prob(
        dependent_runtime_config.sampling.initial_center.to_array(),
        dependent_compiled_model,
    )
    independent_log_prob, _ = log_prob(
        independent_runtime_config.sampling.initial_center.to_array(),
        independent_compiled_model,
    )

    assert np.isfinite(dependent_log_prob)
    assert np.isfinite(independent_log_prob)
    assert independent_log_prob == pytest.approx(dependent_log_prob, rel=0.0, abs=1.0e-12)


def test_likelihood_kernel_matches_numpy_trapezoid_reference(synthetic_config_path) -> None:
    """
    The production likelihood kernel should now mirror the mathematical
    two-stage trapezoid quadrature exactly: one explicit integral over `m*`
    nested inside one explicit integral over `gamma`.

    This test recomputes the same one-lens likelihood in plain NumPy so future
    refactors cannot silently reintroduce pre-weighted bases or double-apply
    quadrature weights.
    """

    runtime_config = load_runtime_config(synthetic_config_path)
    compiled_model = build_compiled_model(runtime_config)
    context = compiled_model.context
    theta = runtime_config.sampling.initial_center.to_array()
    gamma_mode_code = context.gamma_mode_code

    mu5_0 = float(theta[0])
    beta5 = float(theta[1])
    xi5 = float(theta[2])
    sigma5 = float(theta[3])
    mu_gamma_0 = float(theta[4])
    beta_gamma = float(theta[5])
    xi_gamma = float(theta[6])
    sigma_gamma = float(theta[7])
    mu_zs = float(theta[8])
    sigma_zs = float(theta[9])
    theta0 = float(theta[10])
    loga = float(theta[11])

    gamma_integrand = np.zeros_like(context.gamma_grid_int)
    lens_index = 0
    p_zd = float(context.p_zd_fixed[lens_index])
    p_zs = float(truncated_normal_pdf_nonneg(context.zs[lens_index], mu_zs, sigma_zs))

    for gamma_index, gamma_value in enumerate(context.gamma_grid_int):
        log_enclosed_mass = float(context.mass_grid_int[lens_index, gamma_index])
        jacobian = abs(float(context.dmass_dthetaein_grid_int[lens_index, gamma_index]))
        if jacobian <= 0.0:
            continue

        theta_ein = float(
            theta_ein_arcsec(
                float(context.zd[lens_index]),
                float(context.zs[lens_index]),
                log_enclosed_mass,
                float(gamma_value),
                context.z_grid,
                context.chi_kpc_grid,
                float(context.mass_radius_kpc),
            )
        )
        if theta_ein <= 0.0:
            continue

        cross_section_area = np.pi * (float(context.cs_over_theta_int[gamma_index]) * theta_ein) ** 2
        find_probability = float(p_find(theta_ein, theta0, loga))
        if cross_section_area <= 0.0 or find_probability <= 0.0:
            continue

        sigma_likelihood = 1.0
        if int(context.num_sigma[lens_index]) > 0:
            if int(context.has_s2[lens_index]) == 0:
                sigma_likelihood = 0.0
            else:
                sigma_model = np.sqrt(
                    max(float(context.s2_grid_int[lens_index, gamma_index]) * (10.0**log_enclosed_mass), 1.0e-30)
                )
                for sigma_index in range(int(context.num_sigma[lens_index])):
                    sigma_likelihood *= float(
                        normal_pdf(
                            float(context.sigma_obs[lens_index, sigma_index]),
                            sigma_model,
                            float(context.sigma_err[lens_index, sigma_index]),
                        )
                    )
        if sigma_likelihood <= 0.0:
            continue

        mstar_integrand = np.zeros_like(context.mstar_grid[lens_index])
        for mstar_index, _ in enumerate(context.mstar_grid[lens_index]):
            mu5 = (
                mu5_0
                + beta5 * float(context.mstar_shift11p4[lens_index, mstar_index])
                + xi5 * float(context.delta_r_grid[lens_index, mstar_index])
            )
            mu_gamma = (
                mu_gamma_0
                + beta_gamma * float(context.mstar_shift11p4[lens_index, mstar_index])
                + xi_gamma * float(context.delta_r_grid[lens_index, mstar_index])
            )
            mstar_integrand[mstar_index] = (
                float(context.mstar_integrand_base[lens_index, mstar_index])
                * float(normal_pdf(log_enclosed_mass, mu5, sigma5))
                * float(normal_pdf(float(gamma_value), mu_gamma, sigma_gamma))
            )

        integrated_mstar = float(np.trapezoid(mstar_integrand, context.mstar_grid[lens_index]))
        gamma_integrand[gamma_index] = (
            integrated_mstar * p_zd * p_zs * find_probability * cross_section_area * jacobian * sigma_likelihood
        )

    reference_value = float(np.log(np.trapezoid(gamma_integrand, context.gamma_grid_int)))
    kernel_value = float(
        log_likelihood_lenses_numba(
            theta=theta,
            z_grid=context.z_grid,
            chi_kpc_grid=context.chi_kpc_grid,
            cs_over_theta_int=context.cs_over_theta_int,
            mass_grid_int=context.mass_grid_int,
            dmass_dthetaein_grid_int=context.dmass_dthetaein_grid_int,
            s2_grid_int=context.s2_grid_int,
            has_s2=context.has_s2,
            num_sigma=context.num_sigma,
            sigma_obs=context.sigma_obs,
            sigma_err=context.sigma_err,
            zd=context.zd,
            zs=context.zs,
            p_zd_fixed=context.p_zd_fixed,
            mstar_grid=context.mstar_grid,
            mstar_shift11p4=context.mstar_shift11p4,
            mstar_integrand_base=context.mstar_integrand_base,
            delta_r_grid=context.delta_r_grid,
            gamma_grid_int=context.gamma_grid_int,
            mass_radius_kpc=context.mass_radius_kpc,
            gamma_mode_code=gamma_mode_code,
        )
    )

    assert np.isfinite(reference_value)
    assert np.isfinite(kernel_value)
    np.testing.assert_allclose(reference_value, kernel_value, rtol=1.0e-12, atol=1.0e-12)


def test_kernel_primitives_live_in_shared_module_and_compile() -> None:
    """
    Shared numerical primitives should compile from one place so likelihood and
    normalization reuse identical approximations and sampling transforms.
    """

    interp_value = interp1d_clip(0.5, np.array([0.0, 1.0]), np.array([10.0, 20.0]))
    ppf_value = normal_ppf(0.84)
    skew_value = skewnorm_sample(11.0, 0.2, 1.5, 0.1, -0.2)
    trunc_value = truncnorm_sample(2.0, 0.2, 1.2, 2.8, 0.3)
    theta_value = theta_ein_arcsec(
        0.55,
        1.8,
        11.2,
        2.0,
        np.array([0.0, 1.0, 2.0, 3.0]),
        np.array([0.0, 1000.0, 1800.0, 2400.0]),
    )

    assert isinstance(interp1d_clip, CPUDispatcher)
    assert isinstance(normal_ppf, CPUDispatcher)
    assert isinstance(skewnorm_sample, CPUDispatcher)
    assert isinstance(truncnorm_sample, CPUDispatcher)
    assert isinstance(theta_ein_arcsec, CPUDispatcher)
    assert interp1d_clip.signatures
    assert normal_ppf.signatures
    assert skewnorm_sample.signatures
    assert truncnorm_sample.signatures
    assert theta_ein_arcsec.signatures
    assert interp_value == 15.0
    assert np.isfinite(ppf_value)
    assert np.isfinite(skew_value)
    assert np.isfinite(trunc_value)
    assert theta_value >= 0.0
