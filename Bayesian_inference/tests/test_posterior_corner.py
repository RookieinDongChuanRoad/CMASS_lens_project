"""
Integration tests for posterior corner-plot generation.

Why these tests exist:
- the standard pipeline now treats posterior corner plots as required output,
  so the workflow must be exercised by the automated suite rather than being
  regenerated manually from an ad hoc notebook
- the mass-definition migration changed the user-visible parameter names for
  the first four hyper-parameters, so the corner plot must expose `m10` labels
  when the run configuration requests `10 kpc`
- the CLI surface should let the operator regenerate both profile plots in one
  machine-readable call after inference finishes
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import corner
import emcee
import matplotlib.pyplot as plt
import numpy as np
import yaml

from cmass_lens_inference.config import load_runtime_config
from cmass_lens_inference.model import LOG_PROB_BLOB_DTYPE


def _box_prior_for_gamma_mode(
    mass_radius_kpc: int,
    gamma_mode: str,
) -> dict[str, list[float]]:
    """Return the explicit box-prior payload matching one corner-test schema."""

    if mass_radius_kpc == 10:
        mass_bounds = {
            "mu10_0": [9.0, 12.0],
            "beta10": [-3.0, 3.0],
            "xi10": [-3.0, 3.0],
            "sigma10": [1.0e-2, 0.2],
        }
    else:
        mass_bounds = {
            "mu5_0": [9.0, 12.0],
            "beta5": [-3.0, 3.0],
            "xi5": [-3.0, 3.0],
            "sigma5": [1.0e-2, 0.2],
        }

    if gamma_mode == "dependent":
        gamma_bounds = {
            "mu_gamma_0": [1.5, 2.5],
            "beta_gamma": [-3.0, 3.0],
            "xi_gamma": [-3.0, 3.0],
            "sigma_gamma": [0.0, 0.5],
        }
    elif gamma_mode == "independent":
        gamma_bounds = {
            "mu_gamma_0": [1.5, 2.5],
            "sigma_gamma": [0.0, 0.5],
        }
    elif gamma_mode == "sigma_star_dependent":
        gamma_bounds = {
            "mu_gamma_0": [1.5, 2.5],
            "beta_sigma_star_gamma": [-3.0, 3.0],
            "sigma_gamma": [0.0, 0.5],
        }
    else:
        raise ValueError(f"Unsupported corner-test gamma mode '{gamma_mode}'.")

    return {
        **mass_bounds,
        **gamma_bounds,
        "mu_zs": [1.0, 3.0],
        "sigma_zs": [0.0, 2.0],
        "theta0": [0.0, 3.0],
        "loga": [-1.0, 3.0],
    }


def _initial_center_for_gamma_mode(
    mass_initial_center: dict[str, float],
    gamma_mode: str,
) -> dict[str, float]:
    """Build the public initial-center payload for one gamma parameterization."""

    if gamma_mode == "dependent":
        return {
            **mass_initial_center,
            "mu_gamma_0": 1.99,
            "beta_gamma": 0.10,
            "xi_gamma": -0.67,
            "sigma_gamma": 0.149,
            "mu_zs": 1.8,
            "sigma_zs": 0.215,
            "theta0": 0.93,
            "loga": 1.0,
        }
    if gamma_mode == "independent":
        return {
            **mass_initial_center,
            "mu_gamma_0": 1.99,
            "sigma_gamma": 0.149,
            "mu_zs": 1.8,
            "sigma_zs": 0.215,
            "theta0": 0.93,
            "loga": 1.0,
        }
    if gamma_mode == "sigma_star_dependent":
        return {
            **mass_initial_center,
            "mu_gamma_0": 1.99,
            "beta_sigma_star_gamma": 0.24,
            "sigma_gamma": 0.149,
            "mu_zs": 1.8,
            "sigma_zs": 0.215,
            "theta0": 0.93,
            "loga": 1.0,
        }
    raise ValueError(f"Unsupported corner-test gamma mode '{gamma_mode}'.")


def _parameter_center_for_gamma_mode(
    mass_radius_kpc: int,
    gamma_mode: str,
) -> np.ndarray:
    """Return one deterministic posterior center matching the config schema."""

    if mass_radius_kpc == 10:
        mass_theta = [11.42, 0.49, -0.21, 0.08]
    else:
        mass_theta = [11.32, 0.59, -0.11, 0.06]

    if gamma_mode == "dependent":
        return np.array([*mass_theta, 1.99, 0.10, -0.67, 0.149, 1.8, 0.215, 0.93, 1.0], dtype=float)
    if gamma_mode == "independent":
        return np.array([*mass_theta, 1.99, 0.149, 1.8, 0.215, 0.93, 1.0], dtype=float)
    if gamma_mode == "sigma_star_dependent":
        return np.array([*mass_theta, 1.99, 0.24, 0.149, 1.8, 0.215, 0.93, 1.0], dtype=float)
    raise ValueError(f"Unsupported corner-test gamma mode '{gamma_mode}'.")


def _write_corner_config(
    path: Path,
    profile_name: str,
    output_root: Path,
    mass_radius_kpc: int = 10,
    warmup: int = 2,
    gamma_mode: str = "dependent",
) -> Path:
    """
    Create the minimal config snapshot needed by the corner-plot workflow.

    The corner post-processing code only needs a small subset of the original
    run contract:
    - the selected profile name for metadata and figure titles
    - the selected mass definition so the first four public labels are correct
    - the stored warmup length so `burn_in=\"auto\"` matches production
    """

    if mass_radius_kpc == 10:
        mass_initial_center = {
            "mu10_0": 11.42,
            "beta10": 0.49,
            "xi10": -0.21,
            "sigma10": 0.08,
        }
    else:
        mass_initial_center = {
            "mu5_0": 11.32,
            "beta5": 0.59,
            "xi5": -0.11,
            "sigma5": 0.06,
        }

    config = {
        "profile": {"name": profile_name},
        "mass_definition": {"enclosed_radius_kpc": mass_radius_kpc},
        "gamma_model": {"mode": gamma_mode},
        "data": {
            "observation_path": str(output_root / f"{profile_name}_observations.hdf5"),
            "cross_section_path": str(output_root / "cs_grid_power.h5"),
        },
        "box_prior": _box_prior_for_gamma_mode(mass_radius_kpc, gamma_mode),
        "sampling": {
            "n_walkers": 24,
            "n_steps": 5,
            "warmup": warmup,
            "random_seed": 7,
            "initial_center": _initial_center_for_gamma_mode(mass_initial_center, gamma_mode),
            "initial_jitter_scale": 1.0e-3,
        },
        "integration": {
            "gamma_points": 200,
            "mstar_points": 200,
            "normalization_samples": 128,
        },
        "cosmology": {
            "h0": 70.0,
            "omega_m": 0.3,
        },
        "runtime": {
            "checkpoint_every": 1,
            "parallel_strategy": "off",
            "progress": False,
            "progress_summary_every": 1,
            "show_stage_timing": True,
            "disable_hdf5_file_locking": False,
            "num_threads": 0,
            "reserve_cores": 2,
        },
        "output": {
            "root_dir": str(output_root),
            "run_label": "synthetic_corner",
            "overwrite_latest": True,
        },
    }
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def _seed_corner_backend(
    chain_path: Path,
    parameter_center: np.ndarray,
    n_steps: int = 5,
    n_walkers: int = 24,
) -> None:
    """
    Create a deterministic backend chain with small but non-degenerate spread.

    Using a real `emcee` backend keeps the fixture faithful to the production
    storage format and avoids hand-writing fragile HDF5 structures.
    """

    backend = emcee.backends.HDFBackend(str(chain_path))
    backend.reset(n_walkers, parameter_center.shape[0])
    blobs = np.zeros(n_walkers, dtype=LOG_PROB_BLOB_DTYPE)
    blobs["parallel_strategy"] = b"off"
    backend.grow(n_steps, blobs)
    random_state = np.random.RandomState(123).get_state()

    for step_index in range(n_steps):
        coords = np.tile(parameter_center, (n_walkers, 1))
        coords += 1.0e-3 * step_index
        coords += np.linspace(0.0, 2.0e-4, n_walkers, dtype=float)[:, None]
        log_prob = np.full(n_walkers, -5.0 + 0.1 * step_index, dtype=float)
        state = emcee.State(coords, log_prob=log_prob, blobs=blobs.copy(), random_state=random_state)
        backend.save_step(state, np.ones(n_walkers, dtype=bool))


def _build_corner_run(
    tmp_path: Path,
    profile_name: str,
    mass_radius_kpc: int = 10,
    warmup: int = 2,
    n_steps: int = 5,
    gamma_mode: str = "dependent",
) -> Path:
    """Create a minimal completed run directory for the corner-plot tests."""

    run_dir = tmp_path / "runs" / profile_name / f"20260315_090000_{profile_name}_synthetic_corner"
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_corner_config(
        run_dir / "config_snapshot.yaml",
        profile_name=profile_name,
        output_root=tmp_path,
        mass_radius_kpc=mass_radius_kpc,
        warmup=warmup,
        gamma_mode=gamma_mode,
    )
    _seed_corner_backend(
        run_dir / "chain.h5",
        parameter_center=_parameter_center_for_gamma_mode(mass_radius_kpc, gamma_mode),
        n_steps=n_steps,
    )
    return run_dir


def test_run_posterior_corner_generates_expected_artifacts_for_m10(tmp_path: Path) -> None:
    """A completed `m10` run should produce figure and JSON artifacts in-place."""

    from cmass_lens_inference.posterior_corner import run_posterior_corner

    run_dir = _build_corner_run(tmp_path, profile_name="sersic", mass_radius_kpc=10)
    result = run_posterior_corner(run_dir=str(run_dir), burn_in="auto")

    assert result.profile_name == "sersic"
    assert result.status == "completed"
    assert result.input_run_dir == run_dir
    assert result.figure_path == run_dir / "posterior_corner.png"
    assert result.figure_path.exists()
    assert result.result_path == run_dir / "posterior_corner_result.json"
    assert result.result_path.exists()
    assert result.burn_in_applied == 2
    assert result.n_posterior_samples == 3 * 24
    assert result.metadata["parameter_order"][:4] == ["mu10_0", "beta10", "xi10", "sigma10"]
    assert result.metadata["mass_definition"]["label"] == "m10"

    payload = json.loads(result.result_path.read_text(encoding="utf-8"))
    assert payload["profile_name"] == "sersic"
    assert payload["status"] == "completed"
    assert payload["burn_in_applied"] == 2
    assert payload["n_posterior_samples"] == 72
    assert payload["metadata"]["parameter_order"][:4] == ["mu10_0", "beta10", "xi10", "sigma10"]


def test_run_posterior_corner_calls_corner_with_public_m10_labels(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """The corner renderer should expose the selected mass definition in labels."""

    from matplotlib.figure import Figure

    from cmass_lens_inference.posterior_corner import run_posterior_corner

    captured: dict[str, object] = {}

    def _fake_corner(samples, **kwargs):
        captured["samples_shape"] = np.asarray(samples).shape
        captured["kwargs"] = kwargs
        return plt.figure(figsize=(4, 4))

    monkeypatch.setattr(corner, "corner", _fake_corner)
    monkeypatch.setattr(Figure, "savefig", lambda self, path, *args, **kwargs: None)
    monkeypatch.setattr(plt, "close", lambda figure: None)

    run_dir = _build_corner_run(tmp_path, profile_name="devauc", mass_radius_kpc=10)
    runtime_config = load_runtime_config(run_dir / "config_snapshot.yaml")
    run_posterior_corner(run_dir=str(run_dir), burn_in="auto")

    expected_labels = [
        r"$\mu_{10,0}$",
        r"$\beta_{10}$",
        r"$\xi_{10}$",
        r"$\sigma_{10}$",
    ]
    assert captured["samples_shape"] == (72, runtime_config.parameter_schema.n_dim)
    assert captured["kwargs"]["labels"][:4] == expected_labels
    assert captured["kwargs"]["titles"][:4] == expected_labels
    assert captured["kwargs"]["show_titles"] is True
    assert captured["kwargs"]["title_fmt"] == ".2f"
    assert captured["kwargs"]["quantiles"] == [0.16, 0.5, 0.84]
    assert captured["kwargs"]["plot_datapoints"] is False
    assert captured["kwargs"]["levels"] == [0.68, 0.95]


def test_run_posterior_corner_uses_schema_driven_parameter_order_for_sigma_star_mode(
    tmp_path: Path,
) -> None:
    """Corner metadata should follow the active 11D sigma-star gamma schema."""

    from cmass_lens_inference.posterior_corner import run_posterior_corner

    run_dir = _build_corner_run(
        tmp_path,
        profile_name="sersic",
        mass_radius_kpc=10,
        gamma_mode="sigma_star_dependent",
    )
    result = run_posterior_corner(run_dir=str(run_dir), burn_in="auto")

    assert result.metadata["parameter_order"] == [
        "mu10_0",
        "beta10",
        "xi10",
        "sigma10",
        "mu_gamma_0",
        "beta_sigma_star_gamma",
        "sigma_gamma",
        "mu_zs",
        "sigma_zs",
        "theta0",
        "loga",
    ]
    assert len(result.metadata["parameter_labels"]) == 11


def test_cli_posterior_corner_latest_command_generates_both_profiles(tmp_path: Path) -> None:
    """The CLI should regenerate both profile plots and emit machine-readable JSON."""

    devauc_run_dir = _build_corner_run(tmp_path, profile_name="devauc", mass_radius_kpc=10)
    sersic_run_dir = _build_corner_run(tmp_path, profile_name="sersic", mass_radius_kpc=10)
    project_source_root = Path(__file__).resolve().parents[1] / "src"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "cmass_lens_inference.cli",
            "posterior-corner-latest",
            "--devauc-run-dir",
            str(devauc_run_dir),
            "--sersic-run-dir",
            str(sersic_run_dir),
            "--burn-in",
            "auto",
        ],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(project_source_root)},
    )

    payload = json.loads(completed.stdout)
    assert payload["status"] == "completed"
    assert payload["devauc_result"]["profile_name"] == "devauc"
    assert payload["sersic_result"]["profile_name"] == "sersic"
    assert payload["devauc_result"]["burn_in_applied"] == 2
    assert payload["sersic_result"]["burn_in_applied"] == 2
    assert payload["devauc_result"]["n_posterior_samples"] == 72
    assert payload["sersic_result"]["n_posterior_samples"] == 72
    assert payload["devauc_result"]["metadata"]["parameter_order"][:4] == ["mu10_0", "beta10", "xi10", "sigma10"]
    assert payload["sersic_result"]["metadata"]["parameter_order"][:4] == ["mu10_0", "beta10", "xi10", "sigma10"]
