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

import cmass_lens_inference.kernels.normalization as normalization_kernels
from numba.core.registry import CPUDispatcher
import numpy as np
import pytest
import yaml

from cmass_lens_inference.config import load_runtime_config
from cmass_lens_inference.model import (
    build_compiled_model,
    log_prob,
    solve_fundamental_plane_ols,
)
from cmass_lens_inference.kernels.likelihood import log_likelihood_lenses_numba
from cmass_lens_inference.kernels.normalization import normalization_mc_numba, population_summary_mc_numba
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


def _log_likelihood_from_context(theta: np.ndarray, compiled_model) -> float:
    """
    Evaluate the compiled all-lens likelihood outside `model.log_prob`.

    This helper keeps the tests focused on one moving part at a time. Several
    regression checks need the production likelihood value while swapping only
    the normalization or FP-summary implementation under test.
    """

    context = compiled_model.context
    return float(
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
            sigma_star_shift9p0_grid=context.sigma_star_shift9p0_grid,
            mstar_integrand_base=context.mstar_integrand_base,
            delta_r_grid=context.delta_r_grid,
            gamma_grid_int=context.gamma_grid_int,
            mass_radius_kpc=context.mass_radius_kpc,
            gamma_mode_code=context.gamma_mode_code,
        )
    )


def _population_summary_kwargs(theta: np.ndarray, compiled_model) -> dict[str, object]:
    """
    Build the complete argument map for the FP population-summary kernels.

    The production code passes a large flat argument list into numba to avoid
    Python object dispatch in the hot path. Tests centralize the same mapping
    here so the serial reference and parallel implementation are exercised with
    exactly the same scientific inputs.
    """

    context = compiled_model.context
    return {
        "theta": theta,
        "base_normals": context.base_normals,
        "cs_gamma_grid": context.cs_gamma_grid,
        "cs_over_theta": context.cs_over_theta_grid,
        "z_grid": context.z_grid,
        "chi_kpc_grid": context.chi_kpc_grid,
        "mu_d": context.mu_d,
        "sigma_d": context.sigma_d,
        "mass_function_loc": context.mass_function_loc,
        "mass_function_scale": context.mass_function_scale,
        "mass_function_alpha": context.mass_function_alpha,
        "mu_r0": context.mu_r0,
        "beta_r": context.beta_r,
        "sigma_r": context.sigma_r,
        "nu_r": context.nu_r,
        "use_sersic_index": context.use_sersic_index,
        "n_fixed": context.n_fixed,
        "mu_n0": context.mu_n0,
        "beta_n": context.beta_n,
        "sigma_n": context.sigma_n,
        "gamma_trunc_low": context.gamma_trunc_low,
        "gamma_trunc_high": context.gamma_trunc_high,
        "mass_radius_kpc": context.mass_radius_kpc,
        "gamma_mode_code": context.gamma_mode_code,
        "fp_fit_mstar_min": context.fp_fit_mstar_min,
        "fp_pivot_mstar": context.fp_pivot_mstar,
        "fp_gamma_axis": context.fp_gamma_axis,
        "fp_zd_axis": context.fp_zd_axis,
        "fp_log_re_kpc_axis": context.fp_log_re_kpc_axis,
        "fp_n_axis": context.fp_n_axis,
        "fp_sigma_unit_grid": context.fp_sigma_unit_grid,
        "fp_has_n_axis": context.fp_has_n_axis,
    }


def _solve_fp_from_summary(fp_summary: np.ndarray) -> tuple[float, float, float, float]:
    """
    Convert a summary vector into the fitted FP coefficients used by the prior.

    The tests intentionally call the public OLS helper instead of duplicating
    the linear algebra. That keeps the regression focused on whether the kernel
    exports the correct sufficient statistics.
    """

    return solve_fundamental_plane_ols(
        sample_count=float(fp_summary[0]),
        sum_x1=float(fp_summary[1]),
        sum_x2=float(fp_summary[2]),
        sum_x1x1=float(fp_summary[3]),
        sum_x1x2=float(fp_summary[4]),
        sum_x2x2=float(fp_summary[5]),
        sum_y=float(fp_summary[6]),
        sum_x1y=float(fp_summary[7]),
        sum_x2y=float(fp_summary[8]),
        sum_yy=float(fp_summary[9]),
    )


def _legacy_log_prob_with_serial_fp_prior(theta: np.ndarray, compiled_model) -> float:
    """
    Recompute the FP-enabled posterior using the legacy serial summary kernel.

    This is the scientific reference for the parallel refactor: the production
    path is free to change execution strategy, but not the implied posterior.
    """

    serial_summary_kernel = getattr(
        normalization_kernels,
        "_population_summary_mc_serial_reference_numba",
        None,
    )
    assert serial_summary_kernel is not None

    context = compiled_model.context
    z_norm, fp_summary = serial_summary_kernel(**_population_summary_kwargs(theta, compiled_model))
    likelihood_value = _log_likelihood_from_context(theta, compiled_model)
    intercept, beta_mass, _beta_radius, scatter = _solve_fp_from_summary(fp_summary)
    if not np.isfinite([z_norm, likelihood_value, intercept, beta_mass, scatter]).all():
        return -np.inf

    log_fp_prior = 0.0
    log_fp_prior += -0.5 * ((scatter - context.fp_fiducial_scatter) / context.fp_scatter_error) ** 2
    log_fp_prior += -0.5 * ((intercept - context.fp_mu_v_prior) / context.fp_mu_v_error) ** 2
    log_fp_prior += -0.5 * ((beta_mass - context.fp_beta_v_prior) / context.fp_beta_v_error) ** 2
    return float(likelihood_value - context.zd.shape[0] * np.log(z_norm) + log_fp_prior)


def _legacy_log_prob_without_fp_prior(theta: np.ndarray, compiled_model) -> float:
    """
    Recompute the pre-FP-prior posterior formula directly from the kernels.

    This helper exists to lock one compatibility promise in the test suite:
    disabling the optional FP prior must leave the original posterior formula
    exactly unchanged.
    """

    context = compiled_model.context
    z_norm = normalization_mc_numba(
        theta=theta,
        base_normals=context.base_normals,
        cs_gamma_grid=context.cs_gamma_grid,
        cs_over_theta=context.cs_over_theta_grid,
        z_grid=context.z_grid,
        chi_kpc_grid=context.chi_kpc_grid,
        mu_d=context.mu_d,
        sigma_d=context.sigma_d,
        mass_function_loc=context.mass_function_loc,
        mass_function_scale=context.mass_function_scale,
        mass_function_alpha=context.mass_function_alpha,
        mu_r0=context.mu_r0,
        beta_r=context.beta_r,
        sigma_r=context.sigma_r,
        nu_r=context.nu_r,
        use_sersic_index=context.use_sersic_index,
        n_fixed=context.n_fixed,
        mu_n0=context.mu_n0,
        beta_n=context.beta_n,
        sigma_n=context.sigma_n,
        gamma_trunc_low=context.gamma_trunc_low,
        gamma_trunc_high=context.gamma_trunc_high,
        mass_radius_kpc=context.mass_radius_kpc,
        gamma_mode_code=context.gamma_mode_code,
    )
    likelihood_value = _log_likelihood_from_context(theta, compiled_model)
    return float(likelihood_value - context.zd.shape[0] * np.log(z_norm))


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


def test_compiled_model_exposes_fp_prior_sigma_table_context(
    synthetic_fp_prior_config_path,
) -> None:
    """Enabling the FP prior should compile the sigma-table arrays into context."""

    runtime_config = load_runtime_config(synthetic_fp_prior_config_path)
    compiled_model = build_compiled_model(runtime_config)
    context = compiled_model.context

    assert context.fp_enabled == 1
    assert context.fp_fit_mstar_min == pytest.approx(11.0)
    assert context.fp_pivot_mstar == pytest.approx(11.3)
    assert context.fp_gamma_axis.shape == (5,)
    assert context.fp_zd_axis.shape == (4,)
    assert context.fp_log_re_kpc_axis.shape == (3,)
    assert context.fp_n_axis.shape == (4,)
    assert context.fp_sigma_unit_grid.shape == (5, 4, 3, 4)
    assert context.fp_has_n_axis == 1


def test_compiled_model_fp_prior_uses_degenerate_n_axis_for_devauc(
    synthetic_devauc_fp_prior_config_path,
) -> None:
    """Devauc FP-prior context should collapse the missing n-axis to length one."""

    runtime_config = load_runtime_config(synthetic_devauc_fp_prior_config_path)
    compiled_model = build_compiled_model(runtime_config)
    context = compiled_model.context

    assert context.fp_enabled == 1
    assert context.fp_has_n_axis == 0
    assert context.fp_n_axis.shape == (1,)
    assert context.fp_sigma_unit_grid.shape == (5, 4, 3, 1)


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


def test_fp_population_summary_parallel_kernel_matches_legacy_serial_reference(
    synthetic_fp_prior_config_path,
) -> None:
    """
    The FP summary kernel should stay scientifically identical after parallelization.

    This test locks two contracts at once:
    - the production FP summary path must really be compiled with numba parallel
    - its normalization value and OLS sufficient statistics must match the
      retained serial reference implementation on the same random basis
    """

    runtime_config = load_runtime_config(synthetic_fp_prior_config_path)
    compiled_model = build_compiled_model(runtime_config)
    theta = runtime_config.sampling.initial_center.to_array()
    summary_kwargs = _population_summary_kwargs(theta, compiled_model)

    serial_summary_kernel = getattr(
        normalization_kernels,
        "_population_summary_mc_serial_reference_numba",
        None,
    )
    assert serial_summary_kernel is not None
    assert population_summary_mc_numba.targetoptions.get("parallel") is True

    z_norm_parallel, fp_summary_parallel = population_summary_mc_numba(**summary_kwargs)
    z_norm_serial, fp_summary_serial = serial_summary_kernel(**summary_kwargs)

    assert isinstance(population_summary_mc_numba, CPUDispatcher)
    assert population_summary_mc_numba.signatures
    assert z_norm_parallel == pytest.approx(z_norm_serial)
    np.testing.assert_allclose(fp_summary_parallel, fp_summary_serial)
    np.testing.assert_allclose(
        np.asarray(_solve_fp_from_summary(fp_summary_parallel)),
        np.asarray(_solve_fp_from_summary(fp_summary_serial)),
    )


def test_log_prob_matches_legacy_formula_when_fp_prior_disabled(synthetic_config_path) -> None:
    """Disabling the optional FP prior must preserve the original posterior exactly."""

    runtime_config = load_runtime_config(synthetic_config_path)
    compiled_model = build_compiled_model(runtime_config)
    theta = runtime_config.sampling.initial_center.to_array()

    log_prob_value, _ = log_prob(theta, compiled_model)
    legacy_value = _legacy_log_prob_without_fp_prior(theta, compiled_model)

    assert log_prob_value == pytest.approx(legacy_value, rel=0.0, abs=0.0)


def test_fp_prior_log_prob_matches_serial_reference_in_dependent_gamma_mode(
    synthetic_fp_prior_config_path,
) -> None:
    """FP-enabled dependent mode should keep the same posterior after parallelization."""

    runtime_config = load_runtime_config(synthetic_fp_prior_config_path)
    compiled_model = build_compiled_model(runtime_config)
    theta = runtime_config.sampling.initial_center.to_array()

    log_prob_value, _ = log_prob(theta, compiled_model)
    legacy_value = _legacy_log_prob_with_serial_fp_prior(theta, compiled_model)

    assert np.isfinite(log_prob_value)
    assert log_prob_value == pytest.approx(legacy_value)


def test_solve_fundamental_plane_ols_matches_numpy_reference() -> None:
    """The FP OLS helper should recover the same fit as a direct NumPy solve."""

    x1 = np.array([-0.3, 0.2, 0.7, -0.5, 1.1], dtype=np.float64)
    x2 = np.array([0.4, -0.6, 0.1, 0.8, -0.2], dtype=np.float64)
    design_matrix = np.column_stack([np.ones_like(x1), x1, x2])
    y = np.array([2.15, 1.82, 2.41, 1.96, 2.63], dtype=np.float64)

    xtx = design_matrix.T @ design_matrix
    xty = design_matrix.T @ y
    yty = float(y @ y)
    coeff_reference = np.linalg.solve(xtx, xty)
    residual_reference = y - design_matrix @ coeff_reference
    scatter_reference = float(np.sqrt(np.mean(residual_reference**2)))

    intercept, beta_mass, beta_radius, scatter = solve_fundamental_plane_ols(
        sample_count=float(y.shape[0]),
        sum_x1=float(design_matrix[:, 1].sum()),
        sum_x2=float(design_matrix[:, 2].sum()),
        sum_x1x1=float(np.sum(design_matrix[:, 1] ** 2)),
        sum_x1x2=float(np.sum(design_matrix[:, 1] * design_matrix[:, 2])),
        sum_x2x2=float(np.sum(design_matrix[:, 2] ** 2)),
        sum_y=float(y.sum()),
        sum_x1y=float(np.sum(design_matrix[:, 1] * y)),
        sum_x2y=float(np.sum(design_matrix[:, 2] * y)),
        sum_yy=yty,
    )

    assert intercept == pytest.approx(float(coeff_reference[0]))
    assert beta_mass == pytest.approx(float(coeff_reference[1]))
    assert beta_radius == pytest.approx(float(coeff_reference[2]))
    assert scatter == pytest.approx(scatter_reference)


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


def test_fp_prior_changes_log_prob_in_independent_gamma_mode(
    synthetic_fp_prior_independent_config_path,
) -> None:
    """
    Independent gamma mode should still receive a nontrivial FP-prior term.

    This guards the contract that the optional FP prior follows the active
    gamma parameterization instead of silently dropping out when the mass and
    size gamma slopes are absent from the sampled vector.
    """

    runtime_config = load_runtime_config(synthetic_fp_prior_independent_config_path)
    compiled_model = build_compiled_model(runtime_config)
    theta = runtime_config.sampling.initial_center.to_array()

    legacy_value = _legacy_log_prob_with_serial_fp_prior(theta, compiled_model)
    log_prob_value, _ = log_prob(theta, compiled_model)

    assert np.isfinite(log_prob_value)
    assert log_prob_value == pytest.approx(legacy_value)


def test_sigma_star_gamma_mode_uses_eleven_dimensional_theta_vector(
    synthetic_sigma_star_dependent_config_path,
) -> None:
    """
    Sigma-star gamma mode should flow through the compiled model as 11-D.

    The third mode keeps the reduced evaluation vector all the way into the
    kernels and also requires a dedicated precomputed `Sigma_*` grid for the
    likelihood path.
    """

    runtime_config = load_runtime_config(synthetic_sigma_star_dependent_config_path)
    compiled_model = build_compiled_model(runtime_config)
    theta = runtime_config.sampling.initial_center.to_array()

    assert theta.shape == (11,)
    assert compiled_model.context.gamma_mode_code == 2
    assert compiled_model.context.sigma_star_shift9p0_grid.shape == compiled_model.context.mstar_grid.shape

    log_prob_value, _ = log_prob(theta, compiled_model)
    assert np.isfinite(log_prob_value)


def test_sigma_star_gamma_mode_matches_independent_log_prob_when_sigma_slope_is_zero(
    synthetic_sigma_star_dependent_config_path,
    synthetic_independent_config_path,
    tmp_path,
) -> None:
    """
    Sigma-star gamma mode should collapse to the independent mode at zero slope.

    This is the key regression guard for the new parameterization: removing the
    `Sigma_*` slope should leave the scientific model identical to the
    previously implemented `independent` gamma mode.
    """

    sigma_payload = yaml.safe_load(Path(synthetic_sigma_star_dependent_config_path).read_text(encoding="utf-8"))
    sigma_payload["sampling"]["initial_center"]["beta_sigma_star_gamma"] = 0.0
    sigma_zero_path = tmp_path / "sigma_star_zero_slope.yaml"
    sigma_zero_path.write_text(
        yaml.safe_dump(sigma_payload, sort_keys=False),
        encoding="utf-8",
    )

    sigma_runtime_config = load_runtime_config(sigma_zero_path)
    independent_runtime_config = load_runtime_config(synthetic_independent_config_path)
    sigma_compiled_model = build_compiled_model(sigma_runtime_config)
    independent_compiled_model = build_compiled_model(independent_runtime_config)

    sigma_log_prob, _ = log_prob(
        sigma_runtime_config.sampling.initial_center.to_array(),
        sigma_compiled_model,
    )
    independent_log_prob, _ = log_prob(
        independent_runtime_config.sampling.initial_center.to_array(),
        independent_compiled_model,
    )

    assert np.isfinite(sigma_log_prob)
    assert np.isfinite(independent_log_prob)
    assert sigma_log_prob == pytest.approx(independent_log_prob, rel=0.0, abs=1.0e-12)


def test_fp_prior_changes_log_prob_in_zero_slope_sigma_star_mode(
    synthetic_fp_prior_sigma_star_zero_slope_config_path,
) -> None:
    """
    Zero-slope sigma-star mode should still receive a nontrivial FP-prior term.

    The scientific reason for this check is simple: even when `gamma` loses
    its explicit `Sigma_*` dependence, the FP prior still constrains the
    population implied by the remaining hyper-parameters.
    """

    runtime_config = load_runtime_config(synthetic_fp_prior_sigma_star_zero_slope_config_path)
    compiled_model = build_compiled_model(runtime_config)
    theta = runtime_config.sampling.initial_center.to_array()

    legacy_value = _legacy_log_prob_with_serial_fp_prior(theta, compiled_model)
    log_prob_value, _ = log_prob(theta, compiled_model)

    assert np.isfinite(log_prob_value)
    assert log_prob_value == pytest.approx(legacy_value)


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
            sigma_star_shift9p0_grid=context.sigma_star_shift9p0_grid,
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
