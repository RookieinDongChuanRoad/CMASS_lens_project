"""
Create a histogram of observed Einstein radii measured in kpc.

Why this script exists:
- the user asked for a small utility that lives directly under `data/`
- the raw observation HDF5 already stores per-galaxy `r_ein_kpc`, so the
  shortest and safest implementation is to read that field directly instead of
  recomputing from `rein_arcsec` and cosmology
- keeping the logic in a standalone script makes the output easy to reproduce
  later with a single command
"""

from __future__ import annotations

from pathlib import Path
import subprocess

import h5py
import matplotlib
import numpy as np


# Force a non-interactive backend so the script behaves the same on local
# machines, remote terminals, and CI-style environments without a display.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.figure import Figure


DATA_DIRECTORY = Path(__file__).resolve().parent


def resolve_project_data_directory() -> Path:
    """
    Resolve the canonical `data/` directory for both normal and worktree runs.

    In the main workspace the raw HDF5 files live right next to this script, so
    `DATA_DIRECTORY` is already correct. In git worktrees the ignored `raw/`
    payloads are not copied, so we fall back to the shared repository root
    discovered from `git rev-parse --git-common-dir`.
    """

    local_raw_directory = DATA_DIRECTORY / "raw"
    if local_raw_directory.exists():
        return DATA_DIRECTORY

    completed_process = subprocess.run(
        ["git", "rev-parse", "--git-common-dir"],
        cwd=DATA_DIRECTORY,
        check=True,
        capture_output=True,
        text=True,
    )
    common_git_directory = Path(completed_process.stdout.strip()).resolve()
    shared_data_directory = common_git_directory.parent / "data"

    if not shared_data_directory.exists():
        raise FileNotFoundError(f"Unable to locate a usable data directory from {DATA_DIRECTORY}")

    return shared_data_directory


PROJECT_DATA_DIRECTORY = resolve_project_data_directory()
DEFAULT_INPUT_PATH = PROJECT_DATA_DIRECTORY / "raw" / "observations_with_m5_grids_all.hdf5"
DEFAULT_OUTPUT_PATH = PROJECT_DATA_DIRECTORY / "einstein_radius_hist_kpc.png"


def load_einstein_radii_kpc(input_path: Path) -> np.ndarray:
    """
    Load the observed Einstein radii in physical kpc units from HDF5.

    Parameters
    ----------
    input_path:
        HDF5 file containing one group per galaxy. Each group is expected to
        expose `r_ein_kpc` as an attribute.

    Returns
    -------
    np.ndarray
        One-dimensional float array of Einstein radii in kpc, sorted by group
        name to keep the extraction order deterministic.

    Raises
    ------
    ValueError
        If the file has no groups or any group is missing `r_ein_kpc`.
    """

    radii_kpc: list[float] = []

    with h5py.File(input_path, "r") as observation_file:
        group_names = sorted(observation_file.keys())
        if not group_names:
            raise ValueError(f"No galaxy groups were found in {input_path}")

        for group_name in group_names:
            galaxy_group = observation_file[group_name]
            if "r_ein_kpc" not in galaxy_group.attrs:
                raise ValueError(f"Group {group_name!r} is missing required attribute 'r_ein_kpc'")

            radii_kpc.append(float(galaxy_group.attrs["r_ein_kpc"]))

    if not radii_kpc:
        raise ValueError(f"No Einstein radii were loaded from {input_path}")

    return np.asarray(radii_kpc, dtype=np.float64)


def compute_summary_statistics(radii_kpc: np.ndarray) -> dict[str, float]:
    """
    Compute compact summary statistics needed for reporting and plotting.

    The plotting contract only requires the sample mean to be marked on the
    figure, but carrying a few extra descriptive values makes the script easier
    to inspect and extend without re-reading the array each time.
    """

    if radii_kpc.size == 0:
        raise ValueError("Cannot compute summary statistics for an empty sample")

    return {
        "count": float(radii_kpc.size),
        "mean_r_ein_kpc": float(np.mean(radii_kpc)),
        "median_r_ein_kpc": float(np.median(radii_kpc)),
        "min_r_ein_kpc": float(np.min(radii_kpc)),
        "max_r_ein_kpc": float(np.max(radii_kpc)),
    }


def save_histogram(radii_kpc: np.ndarray, summary_statistics: dict[str, float], output_path: Path) -> Figure:
    """
    Render and save the Einstein-radius histogram.

    Parameters
    ----------
    radii_kpc:
        Physical Einstein radii for the observed galaxy sample.
    summary_statistics:
        Precomputed statistics from `compute_summary_statistics`. The mean value
        is used both for the vertical marker line and the text annotation.
    output_path:
        Destination PNG file.

    Returns
    -------
    Figure
        The live matplotlib figure so tests can inspect plot elements directly.
    """

    mean_radius_kpc = float(summary_statistics["mean_r_ein_kpc"])

    figure, axis = plt.subplots(figsize=(8, 5.5), constrained_layout=True)
    axis.hist(
        radii_kpc,
        bins=8,
        color="#4C78A8",
        edgecolor="white",
        alpha=0.9,
    )

    # The mean is the quantity the user explicitly asked to highlight, so we
    # represent it twice: a precise vertical locator line and a human-readable
    # label anchored inside the axes.
    axis.axvline(
        mean_radius_kpc,
        color="#D62728",
        linestyle="--",
        linewidth=2.0,
        label=f"Mean = {mean_radius_kpc:.3f} kpc",
    )
    axis.text(
        0.98,
        0.95,
        f"Mean r_ein_kpc = {mean_radius_kpc:.3f}",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#D62728", "alpha": 0.9},
    )

    axis.set_title("Observed Einstein Radius Distribution")
    axis.set_xlabel("Einstein Radius r_ein (kpc)")
    axis.set_ylabel("Number of Galaxies")
    axis.legend(loc="upper left")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    return figure


def main() -> None:
    """
    Run the standalone extraction and plotting workflow.

    The function prints a short summary so a terminal user can confirm the
    number of galaxies and the sample mean without opening the image first.
    """

    radii_kpc = load_einstein_radii_kpc(DEFAULT_INPUT_PATH)
    summary_statistics = compute_summary_statistics(radii_kpc)
    figure = save_histogram(radii_kpc, summary_statistics, DEFAULT_OUTPUT_PATH)
    plt.close(figure)

    print(f"Loaded {int(summary_statistics['count'])} galaxies from {DEFAULT_INPUT_PATH}")
    print(f"Mean r_ein_kpc = {summary_statistics['mean_r_ein_kpc']:.15f}")
    print(f"Saved histogram to {DEFAULT_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
