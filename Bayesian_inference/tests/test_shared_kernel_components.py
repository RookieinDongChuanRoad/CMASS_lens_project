"""
Focused tests for shared mid-level Numba kernel components.

These tests protect the Phase-4 refactor boundary: the shared helpers must
factor out repeated scientific fragments without changing the scalar
probabilities that CMASS and Sonnenfeld already used in their own kernels.
"""

from __future__ import annotations

import numpy as np

import pytest

from cmass_lens_inference.numba_backend.kernels.distributions import (
    normal_pdf,
    truncated_normal_pdf_nonneg,
)
from cmass_lens_inference.numba_backend.kernels.fundamental_plane import (
    FP_OLS_COUNT_INDEX,
    FP_OLS_SUM_X1X1_INDEX,
    FP_OLS_SUM_X1Y_INDEX,
    FP_OLS_SUM_X1_INDEX,
    FP_OLS_SUM_Y_INDEX,
    FP_OLS_SUM_YY_INDEX,
    FP_OLS_SUMMARY_SIZE,
    accumulate_fp_ols_summary,
)
from cmass_lens_inference.numba_backend.kernels.interpolation import (
    interp_cross_section_theta_gamma,
)
from cmass_lens_inference.numba_backend.kernels.selection import p_find
from cmass_lens_inference.numba_backend.kernels.selection_likelihood import (
    cross_section_find_weight,
    gaussian_source_redshift_density,
    observed_sigma_likelihood,
    sigma_model_from_s2,
    truncated_nonnegative_source_redshift_density,
)


def test_cross_section_find_weight_matches_explicit_scalar_product() -> None:
    """
    Shared selection weighting must remain exactly the old scalar product.

    The helper combines two already existing operations, so the reference value
    deliberately calls those operations separately instead of using a new
    closed-form expectation.
    """

    theta_e_axis = np.array([1.0, 2.0], dtype=np.float64)
    gamma_axis = np.array([1.5, 2.5], dtype=np.float64)
    cross_section_grid = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float64)
    theta_e = 1.25
    gamma = 2.0
    theta_for_detection = 1.1
    theta0 = 0.7
    loga = -0.2

    expected = interp_cross_section_theta_gamma(
        theta_e,
        gamma,
        theta_e_axis,
        gamma_axis,
        cross_section_grid,
    ) * p_find(theta_for_detection, theta0, loga)

    observed = cross_section_find_weight(
        theta_e,
        gamma,
        theta_for_detection,
        theta0,
        loga,
        theta_e_axis,
        gamma_axis,
        cross_section_grid,
    )

    assert observed == pytest.approx(expected)


def test_source_redshift_wrappers_preserve_model_specific_density_choices() -> None:
    """
    CMASS and Sonnenfeld intentionally use different effective source-z priors.

    The wrappers make that model-level choice visible while still delegating to
    the same low-level density implementations used before the refactor.
    """

    assert gaussian_source_redshift_density(1.4, 1.3, 0.2) == pytest.approx(normal_pdf(1.4, 1.3, 0.2))
    assert truncated_nonnegative_source_redshift_density(1.4, 1.3, 0.2) == pytest.approx(
        truncated_normal_pdf_nonneg(1.4, 1.3, 0.2)
    )
    assert truncated_nonnegative_source_redshift_density(-0.1, 1.3, 0.2) == 0.0


def test_observed_sigma_likelihood_handles_present_missing_and_absent_observations() -> None:
    """
    Observed velocity-dispersion likelihood has three distinct row states.

    Rows with observations need a valid S2 model grid, rows with observations
    and no grid reject to zero, and rows without observations contribute a
    neutral factor so they do not alter the lens integral.
    """

    observed_sigma_count = np.array([2, 1, 0], dtype=np.int64)
    has_s2 = np.array([1, 0, 1], dtype=np.int64)
    sigma_obs = np.array(
        [
            [210.0, 230.0],
            [200.0, 0.0],
            [0.0, 0.0],
        ],
        dtype=np.float64,
    )
    sigma_err = np.array(
        [
            [10.0, 20.0],
            [15.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=np.float64,
    )
    sigma_model = 220.0

    expected_present = normal_pdf(210.0, sigma_model, 10.0) * normal_pdf(230.0, sigma_model, 20.0)

    likelihood_args = (observed_sigma_count, has_s2, sigma_obs, sigma_err, sigma_model)

    assert observed_sigma_likelihood(0, *likelihood_args) == pytest.approx(expected_present)
    assert observed_sigma_likelihood(1, *likelihood_args) == 0.0
    assert observed_sigma_likelihood(2, *likelihood_args) == 1.0


def test_sigma_model_from_s2_keeps_historical_positive_floor() -> None:
    """The shared sigma converter must keep the old finite floor behavior."""

    assert sigma_model_from_s2(4.0, 4.0) == pytest.approx(200.0)
    assert sigma_model_from_s2(-1.0, 4.0) == pytest.approx(1.0e-15)


def test_fp_ols_summary_accumulates_expected_sufficient_statistics() -> None:
    """
    FP summary accumulation must keep the positional schema stable.

    Production code later solves the OLS relation from these six entries, so the
    test checks both the index constants and the arithmetic written by the
    shared reducer.
    """

    summary = np.zeros(FP_OLS_SUMMARY_SIZE, dtype=np.float64)
    accumulate_fp_ols_summary(summary, mstar=11.8, log_sigma_model=2.30, pivot_mstar=11.4)
    accumulate_fp_ols_summary(summary, mstar=11.1, log_sigma_model=2.15, pivot_mstar=11.4)

    x_values = np.array([0.4, -0.3], dtype=np.float64)
    y_values = np.array([2.30, 2.15], dtype=np.float64)

    assert summary[FP_OLS_COUNT_INDEX] == 2.0
    assert summary[FP_OLS_SUM_X1_INDEX] == pytest.approx(np.sum(x_values))
    assert summary[FP_OLS_SUM_X1X1_INDEX] == pytest.approx(np.sum(x_values * x_values))
    assert summary[FP_OLS_SUM_Y_INDEX] == pytest.approx(np.sum(y_values))
    assert summary[FP_OLS_SUM_X1Y_INDEX] == pytest.approx(np.sum(x_values * y_values))
    assert summary[FP_OLS_SUM_YY_INDEX] == pytest.approx(np.sum(y_values * y_values))
