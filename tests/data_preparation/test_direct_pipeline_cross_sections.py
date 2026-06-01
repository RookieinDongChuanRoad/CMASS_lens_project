"""Tests for direct-pipeline cross-section providers."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from statistical_sl.data_preparation.direct_pipeline.cross_sections import (
    CmassPowerLawCrossSectionProvider,
    SonnenfeldFibreCrossSectionProvider,
)


def _write_cmass_power_law_file(path: Path) -> Path:
    """Create a compact CMASS legacy cross-section ratio fixture."""

    with h5py.File(path, "w") as handle:
        compressed = handle.create_group("compressed_grids")
        compressed.create_dataset("gamma_grids", data=np.asarray([1.2, 2.0], dtype=float))
        compressed.create_dataset("cs_over_theta_ein_grid", data=np.asarray([0.2, 0.3], dtype=float))
        handle.attrs["generator_name"] = "statistical_sl.data_preparation.power_law_cross_section"
    return path


def _write_sonnenfeld_fibre_file(path: Path) -> Path:
    """Create a compact Sonnenfeld finite-fibre area-grid fixture."""

    with h5py.File(path, "w") as handle:
        handle.create_dataset("tein_grid", data=np.asarray([0.0, 1.0], dtype=float))
        handle.create_dataset("gamma_grid", data=np.asarray([1.2, 2.0], dtype=float))
        handle.create_dataset("mufibre3_cs_grid", data=np.asarray([[0.0, 0.0], [0.5, 1.5]], dtype=float))
        handle.attrs["generator_name"] = "statistical_sl.data_preparation.fibre_cross_section"
    return path


def test_cmass_power_law_provider_converts_ratio_to_area_grid(tmp_path: Path) -> None:
    """CMASS compressed ratios should become theta_E x gamma source-plane areas."""

    source_path = _write_cmass_power_law_file(tmp_path / "cs_grid_power.h5")
    theta_axis = np.asarray([0.5, 1.0], dtype=float)

    block = CmassPowerLawCrossSectionProvider(source_path).load(theta_e_axis=theta_axis)

    expected = np.pi * (theta_axis[:, None] * np.asarray([0.2, 0.3])[None, :]) ** 2
    np.testing.assert_allclose(block.theta_e_axis, theta_axis)
    np.testing.assert_allclose(block.gamma_axis, np.asarray([1.2, 2.0]))
    np.testing.assert_allclose(block.cross_section_grid, expected)
    assert block.provenance.source_path == source_path.resolve()
    assert block.provenance.source_mode == "cmass_power_law"
    assert block.provenance.source_dataset == "compressed_grids/cs_over_theta_ein_grid"


def test_sonnenfeld_fibre_provider_preserves_area_grid_without_cmass_formula(tmp_path: Path) -> None:
    """Sonnenfeld fibre grids already store finite-fibre source-plane areas."""

    source_path = _write_sonnenfeld_fibre_file(tmp_path / "fibre_crosssect_grid.hdf5")

    block = SonnenfeldFibreCrossSectionProvider(source_path).load()

    np.testing.assert_allclose(block.theta_e_axis, np.asarray([0.0, 1.0]))
    np.testing.assert_allclose(block.gamma_axis, np.asarray([1.2, 2.0]))
    np.testing.assert_allclose(block.cross_section_grid, np.asarray([[0.0, 0.0], [0.5, 1.5]]))
    assert block.provenance.source_path == source_path.resolve()
    assert block.provenance.source_mode == "sonnenfeld_fibre"
    assert block.provenance.source_dataset == "mufibre3_cs_grid"
