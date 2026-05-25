"""
Regression tests for the standalone Einstein-radius histogram script.

The script is intentionally kept outside the main Python packages because the
user asked for a small utility living directly under `data/`. These tests load
that script by file path so we can still enforce TDD and verify the plotting
contract without repackaging the repository.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import h5py
import numpy as np
import pytest


WORKTREE_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = WORKTREE_ROOT / "data" / "plot_einstein_radius_hist.py"
EXPECTED_MEAN_R_EIN_KPC = 8.463024096936982
EXPECTED_SAMPLE_SIZE = 23


def _load_script_module():
    """
    Import the standalone script as a module for direct function-level tests.

    The file lives outside a package, so we cannot rely on normal import
    resolution. Loading by absolute file path keeps the production layout
    unchanged while still giving the tests direct access to helper functions.
    """

    module_spec = importlib.util.spec_from_file_location("plot_einstein_radius_hist", SCRIPT_PATH)
    if module_spec is None or module_spec.loader is None:
        raise RuntimeError(f"Unable to create module spec for {SCRIPT_PATH}")

    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def _write_einstein_radius_fixture(path: Path) -> Path:
    """
    Write a minimal observation HDF5 file for the standalone plotting script.

    The production script only depends on each galaxy group exposing an
    ``r_ein_kpc`` attribute, so the test fixture should encode exactly that
    public contract instead of depending on a large ignored raw-data file from a
    particular developer checkout.  The symmetric offsets keep the expected mean
    analytically fixed while still exercising histogram rendering on a non-empty
    distribution rather than on one repeated value.
    """

    radius_offsets = np.linspace(-1.1, 1.1, EXPECTED_SAMPLE_SIZE)
    radii_kpc = EXPECTED_MEAN_R_EIN_KPC + radius_offsets

    with h5py.File(path, "w") as observation_file:
        for index, radius_kpc in enumerate(radii_kpc):
            galaxy_group = observation_file.create_group(f"lens_{index:02d}")
            galaxy_group.attrs["r_ein_kpc"] = float(radius_kpc)

    return path


def test_load_compute_and_plot_histogram_with_mean_annotation(tmp_path: Path) -> None:
    """
    The script must expose a reproducible path from raw data to annotated PNG.

    This single test covers the user-visible contract end to end:
    - load exactly the 23 per-galaxy Einstein radii stored in HDF5
    - compute the correct sample mean from those values
    - render a histogram that contains both a vertical mean marker and a text
      annotation describing that mean
    """

    histogram_module = _load_script_module()
    input_path = _write_einstein_radius_fixture(tmp_path / "observations_with_m5_grids_all.hdf5")

    radii_kpc = histogram_module.load_einstein_radii_kpc(input_path)
    assert len(radii_kpc) == EXPECTED_SAMPLE_SIZE

    summary_statistics = histogram_module.compute_summary_statistics(radii_kpc)
    assert summary_statistics["count"] == EXPECTED_SAMPLE_SIZE
    assert summary_statistics["mean_r_ein_kpc"] == pytest.approx(EXPECTED_MEAN_R_EIN_KPC)

    output_path = tmp_path / "einstein_radius_hist_kpc.png"
    figure = histogram_module.save_histogram(radii_kpc, summary_statistics, output_path)

    assert output_path.exists()
    assert output_path.stat().st_size > 0

    axis = figure.axes[0]
    mean_marker_positions = [
        line.get_xdata()[0]
        for line in axis.lines
        if len(line.get_xdata()) == 2 and line.get_xdata()[0] == pytest.approx(line.get_xdata()[1])
    ]
    assert mean_marker_positions
    assert mean_marker_positions[0] == pytest.approx(EXPECTED_MEAN_R_EIN_KPC)

    annotation_texts = [text.get_text() for text in axis.texts]
    assert any("Mean r_ein_kpc" in text for text in annotation_texts)
    assert any(f"{EXPECTED_MEAN_R_EIN_KPC:.3f}" in text for text in annotation_texts)
