"""Regression tests for lensing cross-section grid generation.

The project has two different cross-section products:

- The historical CMASS table stores a separable source-plane radius
  ``beta_max / theta_E`` for a power-law lens.
- The Sonnenfeld SLACS table stores the already integrated finite-fibre
  source-plane area, including seeing convolution and flux thresholds.

These tests keep the two conventions explicit so future inference work does not
accidentally mix the old CMASS approximation with the Sonnenfeld fibre product.
"""

from __future__ import annotations

from pathlib import Path
import sys

import h5py
import numpy as np
import pytest

from prepare_dataset.cli import build_parser, main
from prepare_dataset.io.lensing_cross_sections import (
    write_fibre_cross_section_hdf5,
    write_power_law_cross_section_hdf5,
)
from prepare_dataset.physics.lensing_cross_section import (
    DEFAULT_CMASS_GAMMA_AXIS,
    DEFAULT_FIBRE_GAMMA_AXIS,
    compute_fibre_cross_section_grid,
    compute_power_law_cross_section_grid,
)


def test_power_law_cross_section_matches_legacy_key_values() -> None:
    """The migrated CMASS generator must reproduce the historical lookup table."""

    gamma_axis = np.asarray([1.2, 1.6, 2.0, 2.4, 2.8], dtype=float)
    theta_e_axis = np.asarray([0.1, 1.0, 5.0], dtype=float)
    result = compute_power_law_cross_section_grid(
        gamma_axis=gamma_axis,
        theta_e_axis=theta_e_axis,
    )

    expected_ratio = np.asarray(
        [
            0.0819199999237061,
            0.3257301136880289,
            0.4999999999999953,
            0.433325759768486,
            0.373494891524315,
        ],
        dtype=float,
    )
    np.testing.assert_allclose(result.cs_over_theta_ein_grid, expected_ratio, rtol=0.0, atol=5.0e-8)


def test_power_law_cross_section_is_scale_free_in_theta_e() -> None:
    """For the legacy power-law product, ``beta_max / theta_E`` is gamma-only."""

    result = compute_power_law_cross_section_grid(
        gamma_axis=np.asarray([1.4, 2.0, 2.6], dtype=float),
        theta_e_axis=np.asarray([0.1, 1.0, 5.0], dtype=float),
    )

    ratio_by_theta = result.cs_grid / result.theta_e_axis[None, :]
    np.testing.assert_allclose(
        ratio_by_theta,
        np.repeat(result.cs_over_theta_ein_grid[:, None], ratio_by_theta.shape[1], axis=1),
        rtol=0.0,
        atol=1.0e-8,
    )


def test_power_law_writer_emits_legacy_hdf5_schema(tmp_path: Path) -> None:
    """The CMASS writer should produce the exact group/dataset surface expected by legacy readers."""

    output_path = write_power_law_cross_section_hdf5(
        tmp_path / "cs_grid_power.h5",
        gamma_axis=np.asarray([2.0], dtype=float),
        theta_e_axis=np.asarray([0.1, 1.0], dtype=float),
    )

    with h5py.File(output_path, "r") as handle:
        assert set(handle.keys()) == {"full_grids", "compressed_grids"}
        assert set(handle["full_grids"].keys()) == {"gamma_grids", "theta_ein_grids", "cs_grid"}
        assert set(handle["compressed_grids"].keys()) == {"gamma_grids", "cs_over_theta_ein_grid"}
        assert handle["full_grids"]["cs_grid"].shape == (1, 2)
        assert handle["compressed_grids"]["cs_over_theta_ein_grid"].shape == (1,)
        assert handle.attrs["generator_name"] == "prepare_dataset.power_law_cross_section"

    with pytest.raises(FileExistsError):
        write_power_law_cross_section_hdf5(output_path, gamma_axis=np.asarray([2.0]), theta_e_axis=np.asarray([1.0]))


def test_fibre_cross_section_small_grid_matches_reference_path() -> None:
    """A tiny Sonnenfeld grid should follow the reference finite-fibre numerical path."""

    result = compute_fibre_cross_section_grid(
        gamma_axis=np.asarray([2.0], dtype=float),
        theta_e_axis=np.asarray([0.0, 1.0], dtype=float),
        beta_points=17,
        radial_points=5,
    )

    assert result.mufibre2_cs_grid.shape == (2, 1)
    assert result.mufibre3_cs_grid.shape == (2, 1)
    assert result.ycaust_grid.shape == (2, 1)
    np.testing.assert_array_equal(result.mufibre2_cs_grid[0], np.asarray([0.0]))
    np.testing.assert_array_equal(result.mufibre3_cs_grid[0], np.asarray([0.0]))
    np.testing.assert_allclose(result.ycaust_grid[1, 0], 0.99, rtol=0.0, atol=1.0e-12)
    assert np.all(result.mufibre2_cs_grid >= 0.0)
    assert np.all(result.mufibre3_cs_grid >= 0.0)
    assert np.all(result.mufibre3_cs_grid <= result.mufibre2_cs_grid)
    np.testing.assert_allclose(result.mufibre2_cs_grid[1, 0], 0.8659898324390689, rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(result.mufibre3_cs_grid[1, 0], 0.5051607355894567, rtol=0.0, atol=1.0e-12)


def test_fibre_writer_emits_sonnenfeld_hdf5_schema(tmp_path: Path) -> None:
    """The fibre writer should expose the datasets expected by Sonnenfeld-style readers."""

    output_path = write_fibre_cross_section_hdf5(
        tmp_path / "fibre_crosssect_grid.hdf5",
        gamma_axis=np.asarray([2.0], dtype=float),
        theta_e_axis=np.asarray([0.0, 1.0], dtype=float),
        beta_points=9,
        radial_points=5,
    )

    with h5py.File(output_path, "r") as handle:
        assert set(handle.keys()) == {
            "tein_grid",
            "gamma_grid",
            "mufibre2_cs_grid",
            "mufibre3_cs_grid",
            "ycaust_grid",
        }
        assert handle["mufibre3_cs_grid"].shape == (2, 1)
        assert handle.attrs["muB_min"] == pytest.approx(1.0)
        assert handle.attrs["fibre_arcsec"] == pytest.approx(1.5)
        assert handle.attrs["seeing_arcsec"] == pytest.approx(1.5)
        assert handle.attrs["source_reference_sha"] == "25873a873a5ecbd61b272e61f1da9a62edada7b5"


def test_cross_section_cli_exposes_new_build_modes() -> None:
    """The public parser should expose both cross-section build entrypoints."""

    parser = build_parser()

    power_law_args = parser.parse_args(
        [
            "--build-power-law-cross-section-hdf5",
            "--output",
            "cs_grid_power.h5",
        ]
    )
    assert power_law_args.build_power_law_cross_section_hdf5 is True
    assert power_law_args.gamma_points == DEFAULT_CMASS_GAMMA_AXIS.size

    fibre_args = parser.parse_args(
        [
            "--build-fibre-cross-section-hdf5",
            "--output",
            "fibre_crosssect_grid.hdf5",
        ]
    )
    assert fibre_args.build_fibre_cross_section_hdf5 is True
    assert fibre_args.gamma_points == DEFAULT_FIBRE_GAMMA_AXIS.size
    assert fibre_args.theta_e_points is None


def test_cross_section_cli_generates_small_power_law_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The command-line build mode should generate a small legacy CMASS table."""

    output_path = tmp_path / "small_power_law.h5"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_dataset",
            "--build-power-law-cross-section-hdf5",
            "--output",
            str(output_path),
            "--gamma-points",
            "1",
            "--theta-e-points",
            "2",
        ],
    )

    assert main() == 0
    with h5py.File(output_path, "r") as handle:
        assert handle["full_grids"]["cs_grid"].shape == (1, 2)


def test_cross_section_cli_generates_small_fibre_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The command-line build mode should generate a small Sonnenfeld fibre table."""

    output_path = tmp_path / "small_fibre.h5"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_dataset",
            "--build-fibre-cross-section-hdf5",
            "--output",
            str(output_path),
            "--gamma-points",
            "1",
            "--theta-e-points",
            "2",
            "--beta-points",
            "9",
            "--radial-points",
            "5",
        ],
    )

    assert main() == 0
    with h5py.File(output_path, "r") as handle:
        assert handle["mufibre3_cs_grid"].shape == (2, 1)
