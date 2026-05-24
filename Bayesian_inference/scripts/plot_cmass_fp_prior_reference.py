"""Plot the CMASS FP-prior reference line together with observed sigma points.

This utility is intentionally narrow:
- it does not touch inference code
- it reads one canonical CMASS h-unit dataset
- it visualizes the reference FP relation and the lenses with sigma
  measurements on the same axes

The figure is meant to answer one question only: where do the observed
velocity-dispersion lenses sit relative to the current CMASS FP-prior
reference line?
"""

from __future__ import annotations

import math
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path("/Users/liurongfu/Work/CMASS_lens_project")
DATASET_PATH = REPO_ROOT / "data/external/inference_dataset_devauc_slit_m5_hunits_v1.hdf5"
OUTPUT_DIR = REPO_ROOT / "outputs/diagnostics/cmass_fp_prior"
OUTPUT_PATH = OUTPUT_DIR / "cmass_fp_prior_vs_sigma_observations_devauc_hunit.png"


def _load_sigma_observations(dataset_path: Path) -> dict[str, np.ndarray | float]:
    """Load the handful of fields needed for the diagnostic scatter plot.

    We deliberately keep this loader small instead of going through the full
    runtime config path.  The plot only needs the lens stellar masses, the
    observed sigma values, and the dataset-level h reference to place the FP
    prior in the active h-unit coordinate system.
    """

    with h5py.File(dataset_path, "r") as handle:
        lenses = handle["lenses"]
        metadata = handle["metadata"]
        return {
            "lens_id": np.asarray(lenses["lens_id"][()]),
            "log_mstar_obs": np.asarray(lenses["log_mstar_obs"][()], dtype=np.float64),
            "num_sigma": np.asarray(lenses["num_sigma"][()], dtype=np.int64),
            "sigma_obs": np.asarray(lenses["sigma_obs"][()], dtype=np.float64),
            "sigma_err": np.asarray(lenses["sigma_err"][()], dtype=np.float64),
            "h_ref": float(metadata.attrs["h_ref"]),
            "unit_convention": str(metadata.attrs["unit_convention"]),
            "mass_definition_label": str(metadata.attrs["mass_definition_label"]),
        }


def _build_fp_reference_curve(
    x_values: np.ndarray,
    h_ref: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Return the FP-prior mean line and its intrinsic-scatter envelope.

    The current CMASS FP-prior reference is defined on the h-unit stellar-mass
    coordinate.  The numeric target values themselves are not shifted; only the
    pivot location moves by `2 log10(h_ref)` so the relation is expressed in
    the active coordinate system used by the CMASS h-unit model.
    """

    fit_mstar_min_phys = 11.0
    pivot_mstar_phys = 11.3
    fiducial_scatter = 0.075
    mu_v_prior = 2.34548
    beta_v_prior = 0.176

    mass_axis_shift = 2.0 * math.log10(h_ref)
    fit_mstar_min_h = fit_mstar_min_phys + mass_axis_shift
    pivot_mstar_h = pivot_mstar_phys + mass_axis_shift

    y_mean = mu_v_prior + beta_v_prior * (x_values - pivot_mstar_h)
    y_upper = y_mean + fiducial_scatter
    y_lower = y_mean - fiducial_scatter

    return y_mean, y_lower, y_upper, fit_mstar_min_h, pivot_mstar_h


def main() -> None:
    """Render the FP-prior diagnostic figure to disk."""

    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Canonical dataset not found: {DATASET_PATH}")

    payload = _load_sigma_observations(DATASET_PATH)
    log_mstar_obs = payload["log_mstar_obs"]
    num_sigma = payload["num_sigma"]
    sigma_obs = payload["sigma_obs"]
    sigma_err = payload["sigma_err"]
    h_ref = float(payload["h_ref"])

    # The canonical CMASS h-unit dataset stores the active stellar-mass
    # coordinate directly as log10[M*/(h^-2 Msun)], so it can be plotted as-is.
    sigma_mask = num_sigma > 0
    sigma_one_mask = sigma_mask & (num_sigma == 1)
    sigma_two_mask = sigma_mask & (num_sigma >= 2)

    # Main points use the first sigma slot for every lens with an observation.
    # A few lenses have a second sigma slot; those are shown as a lighter
    # overlay so the plot still communicates that the canonical dataset carries
    # more than one sigma value for those systems.
    x_main = log_mstar_obs[sigma_mask]
    y_main = np.log10(sigma_obs[sigma_mask, 0])
    yerr_main = sigma_err[sigma_mask, 0] / (sigma_obs[sigma_mask, 0] * math.log(10.0))

    x_double = log_mstar_obs[sigma_two_mask]
    y_double = np.log10(sigma_obs[sigma_two_mask, 1])
    yerr_double = sigma_err[sigma_two_mask, 1] / (sigma_obs[sigma_two_mask, 1] * math.log(10.0))

    x_min = min(float(x_main.min()) - 0.08, 10.62)
    x_max = max(float(x_main.max()) + 0.08, 11.38)
    x_grid = np.linspace(x_min, x_max, 400)
    y_mean, y_lower, y_upper, fit_cut_h, pivot_h = _build_fp_reference_curve(x_grid, h_ref)

    fig, ax = plt.subplots(figsize=(9.2, 6.0), constrained_layout=True)

    # The reference line is the current CMASS FP-prior mean relation.
    ax.plot(
        x_grid,
        y_mean,
        color="black",
        linewidth=2.2,
        label=r"FP prior mean: $\log_{10}\sigma = \mu_v + \beta_v(m_\star - m_{\star,\mathrm{piv}})$",
    )
    ax.fill_between(
        x_grid,
        y_lower,
        y_upper,
        color="black",
        alpha=0.12,
        linewidth=0,
        label=r"Intrinsic scatter $\pm 0.075$ dex",
    )

    # Main observed points: one point per lens with sigma data.
    ax.errorbar(
        x_main,
        y_main,
        yerr=yerr_main,
        fmt="o",
        ms=6.5,
        mfc="#1f77b4",
        mec="white",
        mew=0.8,
        ecolor="#9ecae1",
        elinewidth=1.2,
        capsize=2.5,
        alpha=0.95,
        label=f"Observed lenses with sigma (n={int(sigma_mask.sum())})",
        zorder=3,
    )

    # Secondary sigma measurements, only for the three lenses with num_sigma=2.
    if x_double.size:
        ax.errorbar(
            x_double,
            y_double,
            yerr=yerr_double,
            fmt="D",
            ms=5.5,
            mfc="none",
            mec="#d95f0e",
            mew=1.2,
            ecolor="#fdae6b",
            elinewidth=1.0,
            capsize=2.0,
            alpha=0.9,
            label="Secondary sigma slot (num_sigma=2)",
            zorder=4,
        )

    ax.axvline(fit_cut_h, color="#6a3d9a", linestyle="--", linewidth=1.4, alpha=0.85, label=rf"FP fit cut = {fit_cut_h:.3f}")
    ax.axvline(pivot_h, color="#6a3d9a", linestyle=":", linewidth=1.4, alpha=0.85, label=rf"FP pivot = {pivot_h:.3f}")

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(2.18, 2.52)
    ax.set_xlabel(r"$\log_{10}[M_\star / (h^{-2} M_\odot)]$")
    ax.set_ylabel(r"$\log_{10}(\sigma / \mathrm{km\,s^{-1}})$")
    ax.set_title("CMASS h-unit FP prior and observed sigma lenses")
    ax.grid(True, which="major", color="#d9d9d9", linewidth=0.8)
    ax.grid(True, which="minor", color="#eeeeee", linewidth=0.5)
    ax.minorticks_on()

    annotation = (
        r"$\mu_v=2.34548,\ \beta_v=0.176$" "\n"
        rf"$m_{{\star,\mathrm{{piv}}}}^{{(h)}}={pivot_h:.3f}$" "\n"
        r"$\sigma_{\rm FP}=0.075$"
    )
    ax.text(
        0.03,
        0.97,
        annotation,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#cccccc", alpha=0.95),
    )

    ax.legend(loc="lower right", frameon=True, framealpha=0.95, fontsize=9)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=220)
    plt.close(fig)

    print(f"Saved figure to: {OUTPUT_PATH}")
    print(f"Dataset: {DATASET_PATH}")
    print(f"h_ref={h_ref:.3f}, sigma_lenses={int(sigma_mask.sum())}, double_sigma_lenses={int(sigma_two_mask.sum())}")
    print(f"fit_cut_h={fit_cut_h:.6f}, pivot_h={pivot_h:.6f}")


if __name__ == "__main__":
    main()
