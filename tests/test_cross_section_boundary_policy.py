from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from statistical_sl.core.cross_section_policy import (
    BOUNDARY_THETA_SQUARED_EXTRAPOLATE_CLIP_GAMMA,
    BOUNDARY_ZERO_OUTSIDE_THETA_CLIP_GAMMA,
    CROSS_SECTION_MODE_GRID_ZERO_OUTSIDE,
    CROSS_SECTION_MODE_SEPARABLE_THETA_SQUARED,
    SOURCE_MUFIBRE3_CS_GRID,
    SOURCE_SEPARABLE_CS_OVER_THETA_EIN,
    resolve_cross_section_mode,
)
from statistical_sl.inference.config import load_runtime_config
from statistical_sl.models.cmass.preprocessing import load_cmass_canonical_dataset
from statistical_sl.models.cmass.runtime import build_context_bundle
from statistical_sl.numerics.numba.kernels.interpolation import interp_cross_section_theta_gamma
from statistical_sl.numerics.numba.kernels.selection import p_find
from statistical_sl.numerics.numba.kernels.selection_likelihood import (
    policy_cross_section_find_weight,
    separable_theta_squared_cross_section,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CMASS_DEVAUC_DATASET = REPOSITORY_ROOT / "workspace/data/canonical/inference_dataset_devauc_slit_m5_hunits_v1.hdf5"
SONNENFELD_FIXED_DATASET = (
    REPOSITORY_ROOT / "workspace/data/canonical/inference_dataset_sonnenfeld2024_slacs_m5_fixed_v1.hdf5"
)
SONNENFELD_HUNITS_DATASET = (
    REPOSITORY_ROOT / "workspace/data/canonical/inference_dataset_sonnenfeld2024_slacs_m5_hunits_v1.hdf5"
)
CMASS_DEVAUC_CONFIG = REPOSITORY_ROOT / "workspace/configs/inference/cmass/devauc.yaml"
LEGACY_CMASS_CROSS_SECTION = REPOSITORY_ROOT / "workspace/data/external/cs_grid_power.h5"


def _decode_hdf5_attr(value: object) -> str:
    """Return an HDF5 scalar string attr as a normal Python string."""

    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray) and value.shape == ():
        return _decode_hdf5_attr(value.item())
    return str(value)


def _cross_section_attrs(path: Path) -> tuple[str, str]:
    """Read the self-describing cross-section source and boundary policy."""

    with h5py.File(path, "r") as handle:
        group = handle["lensing_cross_section"]
        return (
            _decode_hdf5_attr(group.attrs["source"]),
            _decode_hdf5_attr(group.attrs["boundary_policy"]),
        )


def test_workspace_canonical_cross_section_metadata_matches_policy_contract() -> None:
    """Tracked canonical datasets must advertise the boundary policy runtime will use."""

    assert _cross_section_attrs(CMASS_DEVAUC_DATASET) == (
        SOURCE_SEPARABLE_CS_OVER_THETA_EIN,
        BOUNDARY_THETA_SQUARED_EXTRAPOLATE_CLIP_GAMMA,
    )
    assert _cross_section_attrs(SONNENFELD_FIXED_DATASET) == (
        SOURCE_MUFIBRE3_CS_GRID,
        BOUNDARY_ZERO_OUTSIDE_THETA_CLIP_GAMMA,
    )
    assert _cross_section_attrs(SONNENFELD_HUNITS_DATASET) == (
        SOURCE_MUFIBRE3_CS_GRID,
        BOUNDARY_ZERO_OUTSIDE_THETA_CLIP_GAMMA,
    )


def test_cross_section_policy_resolver_rejects_invalid_combinations() -> None:
    """The resolver is the guardrail between HDF5 metadata and Numba modes."""

    assert (
        resolve_cross_section_mode(
            SOURCE_SEPARABLE_CS_OVER_THETA_EIN,
            BOUNDARY_THETA_SQUARED_EXTRAPOLATE_CLIP_GAMMA,
        )
        == CROSS_SECTION_MODE_SEPARABLE_THETA_SQUARED
    )
    assert (
        resolve_cross_section_mode(
            SOURCE_MUFIBRE3_CS_GRID,
            BOUNDARY_ZERO_OUTSIDE_THETA_CLIP_GAMMA,
        )
        == CROSS_SECTION_MODE_GRID_ZERO_OUTSIDE
    )
    assert (
        resolve_cross_section_mode(
            SOURCE_SEPARABLE_CS_OVER_THETA_EIN,
            BOUNDARY_ZERO_OUTSIDE_THETA_CLIP_GAMMA,
        )
        == CROSS_SECTION_MODE_GRID_ZERO_OUTSIDE
    )
    with pytest.raises(ValueError, match=SOURCE_MUFIBRE3_CS_GRID):
        resolve_cross_section_mode(
            SOURCE_MUFIBRE3_CS_GRID,
            BOUNDARY_THETA_SQUARED_EXTRAPOLATE_CLIP_GAMMA,
        )


def test_canonical_reader_exposes_cross_section_source() -> None:
    """The reader should preserve source metadata instead of reducing it to a grid."""

    runtime_config = load_runtime_config(CMASS_DEVAUC_CONFIG)
    dataset = load_cmass_canonical_dataset(runtime_config)

    assert dataset.cross_section.source == SOURCE_SEPARABLE_CS_OVER_THETA_EIN
    assert dataset.cross_section.boundary_policy == BOUNDARY_THETA_SQUARED_EXTRAPOLATE_CLIP_GAMMA


def test_cmass_separable_cross_section_extrapolates_as_theta_squared() -> None:
    """CMASS separable cross-sections have a valid analytic theta_E extension."""

    gamma_axis = np.asarray([1.5, 2.0, 2.5], dtype=np.float64)
    cs_over_theta_grid = np.asarray([0.25, 0.50, 1.00], dtype=np.float64)
    theta_e = 6.0
    gamma = 2.0

    cross_section = separable_theta_squared_cross_section(
        theta_e,
        gamma,
        gamma_axis,
        cs_over_theta_grid,
    )

    assert np.isclose(cross_section, np.pi * (theta_e * 0.50) ** 2)


def test_policy_cross_section_find_weight_uses_detection_probability() -> None:
    """The policy-aware helper should preserve the existing discovery weighting."""

    gamma_axis = np.asarray([1.5, 2.0, 2.5], dtype=np.float64)
    theta_axis = np.asarray([0.0, 5.0], dtype=np.float64)
    grid = np.ones((theta_axis.size, gamma_axis.size), dtype=np.float64)
    cs_over_theta_grid = np.asarray([0.25, 0.50, 1.00], dtype=np.float64)
    theta_e = 6.0
    theta0 = 1.0
    loga = 1.0

    weighted = policy_cross_section_find_weight(
        theta_e,
        2.0,
        theta_e,
        theta0,
        loga,
        CROSS_SECTION_MODE_SEPARABLE_THETA_SQUARED,
        theta_axis,
        gamma_axis,
        grid,
        cs_over_theta_grid,
    )

    expected_cross_section = np.pi * (theta_e * 0.50) ** 2
    assert np.isclose(weighted, expected_cross_section * p_find(theta_e, theta0, loga))


def test_generic_grid_cross_section_still_zeroes_theta_outside() -> None:
    """Finite two-dimensional grids must not inherit the CMASS analytic extension."""

    theta_axis = np.asarray([0.0, 5.0], dtype=np.float64)
    gamma_axis = np.asarray([1.5, 2.0, 2.5], dtype=np.float64)
    grid = np.ones((theta_axis.size, gamma_axis.size), dtype=np.float64)

    assert interp_cross_section_theta_gamma(6.0, 2.0, theta_axis, gamma_axis, grid) == 0.0
    assert (
        policy_cross_section_find_weight(
            6.0,
            2.0,
            6.0,
            1.0,
            1.0,
            CROSS_SECTION_MODE_GRID_ZERO_OUTSIDE,
            theta_axis,
            gamma_axis,
            grid,
            np.zeros_like(gamma_axis),
        )
        == 0.0
    )


def test_sonnenfeld_grid_policy_does_not_extrapolate() -> None:
    """Sonnenfeld finite-fibre tables remain finite-domain cross-section grids."""

    assert (
        resolve_cross_section_mode(SOURCE_MUFIBRE3_CS_GRID, BOUNDARY_ZERO_OUTSIDE_THETA_CLIP_GAMMA)
        == CROSS_SECTION_MODE_GRID_ZERO_OUTSIDE
    )
    with pytest.raises(ValueError):
        resolve_cross_section_mode(SOURCE_MUFIBRE3_CS_GRID, BOUNDARY_THETA_SQUARED_EXTRAPOLATE_CLIP_GAMMA)


def test_cmass_context_recovers_separable_cross_section_factor() -> None:
    """CMASS runtime context should recover the legacy separable gamma factor."""

    runtime_config = load_runtime_config(CMASS_DEVAUC_CONFIG)
    context = build_context_bundle(runtime_config).context

    assert context.cross_section_mode_code == CROSS_SECTION_MODE_SEPARABLE_THETA_SQUARED

    with h5py.File(LEGACY_CMASS_CROSS_SECTION, "r") as legacy_cross_section:
        legacy_group = legacy_cross_section["compressed_grids"]
        np.testing.assert_allclose(
            context.cs_gamma_grid,
            legacy_group["gamma_grids"][...],
            rtol=0.0,
            atol=0.0,
        )
        np.testing.assert_allclose(
            context.cs_over_theta_grid,
            legacy_group["cs_over_theta_ein_grid"][...],
            rtol=1.0e-12,
            atol=1.0e-12,
        )
