"""Regression tests for flat-prior measurement-attribute HDF5 updates.

These tests lock down the contract requested for the new updater:

- only groups with ``num_sigma != 0`` are eligible for measurement attrs
- the authoritative inputs live under ``mass_definitions``
- quantiles must stay defined even when the discrete CDF jumps over 16%/84%
- projected mass error bars must remain positive for non-monotonic mass curves
- preview mode must not mutate the input file
- in-place mode must preserve datasets while writing only root attrs
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

from prepare_dataset.io.flatprior_measurement_updates import (
    FlatPriorMeasurementUpdateValidationError,
    plan_flatprior_measurement_updates_for_files,
    update_flatprior_measurement_attrs_in_hdf5,
)


TARGET_ATTR_NAMES = (
    "gamma_lower",
    "gamma_mid",
    "gamma_upper",
    "m5_lower",
    "m5_mid",
    "m5_upper",
    "m10_lower",
    "m10_mid",
    "m10_upper",
)


def _sigma_unit_grid_from_model(log_mass_grid: np.ndarray, sigma_model_grid: np.ndarray) -> np.ndarray:
    """Back-compute the unit-mass ``s2_grid`` that yields a chosen sigma model.

    The production updater reconstructs ``sigma_model`` through
    ``sqrt(10**m5 * s2_grid)``. Building fixtures from the desired sigma-model
    values keeps the tests easy to reason about while still exercising the same
    physical formula as production code.
    """

    return np.asarray(sigma_model_grid, dtype=float) ** 2 / (10.0 ** np.asarray(log_mass_grid, dtype=float))


def _write_mass_definition_group(
    mass_definitions_handle: h5py.Group,
    label: str,
    mass_grid: np.ndarray,
    derivative_grid: np.ndarray | None = None,
    s2_grid: np.ndarray | None = None,
) -> None:
    """Write one nested mass-definition subgroup used by the updater."""

    subgroup = mass_definitions_handle.create_group(label)
    subgroup.create_dataset("mass_grid", data=np.asarray(mass_grid, dtype=float))
    if derivative_grid is not None:
        subgroup.create_dataset("dmass_dthetaein_grid", data=np.asarray(derivative_grid, dtype=float))
    if s2_grid is not None:
        subgroup.create_dataset("s2_grid", data=np.asarray(s2_grid, dtype=float))


def _create_hdf5_fixture(path: Path) -> None:
    """Create a compact HDF5 fixture that mirrors the production schema surface.

    The fixture intentionally includes misleading top-level ``m5_grid`` and
    ``s2_grid`` datasets for the eligible groups. That makes the test fail if
    the updater accidentally reads the legacy root-level grids instead of the
    nested ``mass_definitions`` inputs mandated by the plan.
    """

    gamma_grid = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0], dtype=float)
    derivative_grid = np.asarray([0.8, 0.9, 1.0, 0.9, 0.8], dtype=float)

    reversed_m5_grid = np.asarray([10.0, 12.0, 13.0, 11.0, 9.0], dtype=float)
    reversed_m10_grid = np.asarray([10.7, 12.7, 13.7, 11.7, 9.7], dtype=float)
    reversed_sigma_model = np.asarray([110.0, 180.0, 220.0, 175.0, 105.0], dtype=float)
    reversed_s2_grid = _sigma_unit_grid_from_model(reversed_m5_grid, reversed_sigma_model)

    delta_m5_grid = np.asarray([10.0, 10.2, 10.4, 10.6, 10.8], dtype=float)
    delta_m10_grid = np.asarray([10.6, 10.8, 11.0, 11.2, 11.4], dtype=float)
    delta_sigma_model = np.asarray([120.0, 150.0, 190.0, 300.0, 160.0], dtype=float)
    delta_s2_grid = _sigma_unit_grid_from_model(delta_m5_grid, delta_sigma_model)

    with h5py.File(path, "w") as handle:
        skipped = handle.create_group("skip-me")
        skipped.attrs["num_sigma"] = np.int64(0)
        skipped.create_dataset("gamma_grid", data=gamma_grid)

        reversed_group = handle.create_group("reversed-mass")
        reversed_group.attrs["num_sigma"] = np.int64(1)
        reversed_group.attrs["sigma"] = np.asarray([210.0], dtype=float)
        reversed_group.attrs["sigma_err"] = np.asarray([35.0], dtype=float)
        reversed_group.create_dataset("gamma_grid", data=gamma_grid)
        reversed_group.create_dataset("m5_grid", data=np.zeros_like(gamma_grid))
        reversed_group.create_dataset("s2_grid", data=np.ones_like(gamma_grid))
        reversed_mass_definitions = reversed_group.create_group("mass_definitions")
        _write_mass_definition_group(
            reversed_mass_definitions,
            label="m5",
            mass_grid=reversed_m5_grid,
            derivative_grid=derivative_grid,
            s2_grid=reversed_s2_grid,
        )
        _write_mass_definition_group(
            reversed_mass_definitions,
            label="m10",
            mass_grid=reversed_m10_grid,
        )

        delta_group = handle.create_group("delta-like")
        delta_group.attrs["num_sigma"] = np.int64(2)
        delta_group.attrs["sigma"] = np.asarray([300.0, 295.0], dtype=float)
        delta_group.attrs["sigma_err"] = np.asarray([6.0, 6.0], dtype=float)
        delta_group.create_dataset("gamma_grid", data=gamma_grid)
        delta_group.create_dataset("m5_grid", data=-5.0 * np.ones_like(gamma_grid))
        delta_group.create_dataset("s2_grid", data=3.0 * np.ones_like(gamma_grid))
        delta_mass_definitions = delta_group.create_group("mass_definitions")
        _write_mass_definition_group(
            delta_mass_definitions,
            label="m5",
            mass_grid=delta_m5_grid,
            derivative_grid=derivative_grid,
            s2_grid=delta_s2_grid,
        )
        _write_mass_definition_group(
            delta_mass_definitions,
            label="m10",
            mass_grid=delta_m10_grid,
        )


def _interpolate_quantile_from_cdf(axis: np.ndarray, cdf: np.ndarray, target: float) -> float:
    """Evaluate one quantile on a discrete CDF using linear interpolation.

    This helper mirrors the requested production behavior, including edge
    clamping. It exists in the tests so we can compare the updater output with
    an independently computed expectation rather than hard-coding many floating
    point literals.
    """

    axis = np.asarray(axis, dtype=float)
    cdf = np.asarray(cdf, dtype=float)
    if target <= cdf[0]:
        return float(axis[0])
    if target >= cdf[-1]:
        return float(axis[-1])

    upper_index = int(np.searchsorted(cdf, target, side="left"))
    lower_index = upper_index - 1
    lower_cdf = float(cdf[lower_index])
    upper_cdf = float(cdf[upper_index])
    lower_axis = float(axis[lower_index])
    upper_axis = float(axis[upper_index])

    if upper_cdf <= lower_cdf:
        return upper_axis

    interpolation_weight = (target - lower_cdf) / (upper_cdf - lower_cdf)
    return lower_axis + interpolation_weight * (upper_axis - lower_axis)


def _expected_measurement_attrs(group_handle: h5py.Group) -> dict[str, float]:
    """Compute the expected root attrs for one eligible fixture group."""

    gamma_grid = np.asarray(group_handle["gamma_grid"], dtype=float)
    sigma_values = np.asarray(group_handle.attrs["sigma"], dtype=float)
    sigma_errors = np.asarray(group_handle.attrs["sigma_err"], dtype=float)

    m5_group = group_handle["mass_definitions"]["m5"]
    m10_group = group_handle["mass_definitions"]["m10"]
    m5_grid = np.asarray(m5_group["mass_grid"], dtype=float)
    m10_grid = np.asarray(m10_group["mass_grid"], dtype=float)
    derivative_grid = np.asarray(m5_group["dmass_dthetaein_grid"], dtype=float)
    s2_grid = np.asarray(m5_group["s2_grid"], dtype=float)

    sigma_model_grid = np.sqrt(np.maximum((10.0**m5_grid) * s2_grid, 1.0e-300))
    logp_grid = np.zeros_like(gamma_grid)
    for sigma_value, sigma_error in zip(sigma_values, sigma_errors):
        logp_grid += -0.5 * ((sigma_model_grid - sigma_value) / sigma_error) ** 2 - np.log(sigma_error)
    logp_grid += -np.log(derivative_grid)
    logp_grid -= np.max(logp_grid)

    posterior_grid = np.exp(logp_grid)
    posterior_grid /= posterior_grid.sum()
    cdf_grid = posterior_grid.cumsum()

    gamma_q16 = _interpolate_quantile_from_cdf(gamma_grid, cdf_grid, 0.16)
    gamma_q50 = _interpolate_quantile_from_cdf(gamma_grid, cdf_grid, 0.50)
    gamma_q84 = _interpolate_quantile_from_cdf(gamma_grid, cdf_grid, 0.84)

    m5_q16 = float(np.interp(gamma_q16, gamma_grid, m5_grid))
    m5_q50 = float(np.interp(gamma_q50, gamma_grid, m5_grid))
    m5_q84 = float(np.interp(gamma_q84, gamma_grid, m5_grid))

    m10_q16 = float(np.interp(gamma_q16, gamma_grid, m10_grid))
    m10_q50 = float(np.interp(gamma_q50, gamma_grid, m10_grid))
    m10_q84 = float(np.interp(gamma_q84, gamma_grid, m10_grid))

    m5_low_value, m5_high_value = sorted((m5_q16, m5_q84))
    m10_low_value, m10_high_value = sorted((m10_q16, m10_q84))

    return {
        "gamma_lower": gamma_q50 - gamma_q16,
        "gamma_mid": gamma_q50,
        "gamma_upper": gamma_q84 - gamma_q50,
        "m5_lower": abs(m5_q50 - m5_low_value),
        "m5_mid": m5_q50,
        "m5_upper": abs(m5_high_value - m5_q50),
        "m10_lower": abs(m10_q50 - m10_low_value),
        "m10_mid": m10_q50,
        "m10_upper": abs(m10_high_value - m10_q50),
    }


def test_plan_flatprior_updates_uses_nested_mass_definitions_and_returns_expected_attrs(
    tmp_path: Path,
) -> None:
    """Planning should compute one measurement summary per eligible group."""

    hdf5_path = tmp_path / "input.hdf5"
    _create_hdf5_fixture(hdf5_path)

    plans = plan_flatprior_measurement_updates_for_files(hdf5_paths=[hdf5_path])

    assert len(plans) == 1
    planned_groups = {item.group_name: item for item in plans[0].group_updates}
    assert sorted(planned_groups) == ["delta-like", "reversed-mass"]

    with h5py.File(hdf5_path, "r") as handle:
        for group_name, group_plan in planned_groups.items():
            expected_attrs = _expected_measurement_attrs(handle[group_name])
            for attr_name, expected_value in expected_attrs.items():
                assert group_plan.new_attrs[attr_name] == pytest.approx(expected_value)

    # The "reversed-mass" fixture is intentionally non-monotonic. This asserts
    # the updater returns positive error bars instead of carrying through a
    # negative sign from the projection order.
    assert planned_groups["reversed-mass"].new_attrs["m5_lower"] > 0.0
    assert planned_groups["reversed-mass"].new_attrs["m5_upper"] > 0.0


def test_update_flatprior_measurements_preview_mode_leaves_source_file_unchanged(tmp_path: Path) -> None:
    """Preview mode must validate everything without mutating the input file."""

    hdf5_path = tmp_path / "input.hdf5"
    _create_hdf5_fixture(hdf5_path)
    original_bytes = hdf5_path.read_bytes()

    result = update_flatprior_measurement_attrs_in_hdf5(
        hdf5_paths=[hdf5_path],
        overwrite_in_place=False,
        create_backup=False,
    )

    assert hdf5_path.read_bytes() == original_bytes
    assert result[0].was_written is False
    assert result[0].backup_path is None


def test_update_flatprior_measurements_in_place_writes_root_attrs_and_preserves_datasets(
    tmp_path: Path,
) -> None:
    """In-place writes should add only the measurement attrs at the group root."""

    hdf5_path = tmp_path / "input.hdf5"
    _create_hdf5_fixture(hdf5_path)

    with h5py.File(hdf5_path, "r") as handle:
        preserved_dataset = handle["delta-like"]["mass_definitions"]["m5"]["s2_grid"][:]
        original_skip_attrs = set(handle["skip-me"].attrs.keys())

    result = update_flatprior_measurement_attrs_in_hdf5(
        hdf5_paths=[hdf5_path],
        overwrite_in_place=True,
        create_backup=True,
    )

    assert result[0].was_written is True
    assert result[0].backup_path is not None
    assert result[0].backup_path.exists()

    with h5py.File(hdf5_path, "r") as handle:
        assert np.array_equal(
            handle["delta-like"]["mass_definitions"]["m5"]["s2_grid"][:],
            preserved_dataset,
        )
        assert set(handle["skip-me"].attrs.keys()) == original_skip_attrs

        for group_name in ("reversed-mass", "delta-like"):
            group_handle = handle[group_name]
            for attr_name in TARGET_ATTR_NAMES:
                assert attr_name in group_handle.attrs
                assert np.isfinite(group_handle.attrs[attr_name])


def test_plan_flatprior_updates_rejects_missing_nested_mass_grid(tmp_path: Path) -> None:
    """Missing nested mass-definition datasets must fail validation early."""

    hdf5_path = tmp_path / "input.hdf5"
    _create_hdf5_fixture(hdf5_path)

    with h5py.File(hdf5_path, "r+") as handle:
        del handle["delta-like"]["mass_definitions"]["m10"]["mass_grid"]

    with pytest.raises(FlatPriorMeasurementUpdateValidationError, match="delta-like"):
        plan_flatprior_measurement_updates_for_files(hdf5_paths=[hdf5_path])


def test_plan_flatprior_updates_rejects_sigma_shape_mismatch(tmp_path: Path) -> None:
    """The updater must reject groups whose sigma arrays disagree with ``num_sigma``."""

    hdf5_path = tmp_path / "input.hdf5"
    _create_hdf5_fixture(hdf5_path)

    with h5py.File(hdf5_path, "r+") as handle:
        handle["delta-like"].attrs["sigma"] = np.asarray([300.0], dtype=float)

    with pytest.raises(FlatPriorMeasurementUpdateValidationError, match="delta-like"):
        plan_flatprior_measurement_updates_for_files(hdf5_paths=[hdf5_path])


def test_module_cli_entrypoint_prints_preview_for_python_dash_m(tmp_path: Path) -> None:
    """The module should behave like a real standalone CLI under ``python -m``."""

    hdf5_path = tmp_path / "input.hdf5"
    _create_hdf5_fixture(hdf5_path)

    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    project_pythonpath = ".:/Users/liurongfu/tools"
    env["PYTHONPATH"] = (
        f"{project_pythonpath}:{existing_pythonpath}" if existing_pythonpath else project_pythonpath
    )

    completed_process = subprocess.run(
        [
            sys.executable,
            "-m",
            "prepare_dataset.io.flatprior_measurement_updates",
            "--hdf5",
            str(hdf5_path),
        ],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed_process.returncode == 0, completed_process.stderr
    assert "PREVIEW" in completed_process.stdout
    assert "groups_to_update=2" in completed_process.stdout
