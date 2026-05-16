"""
Integration tests for posterior corner-plot generation.

Why these tests exist:
- the standard pipeline now treats posterior corner plots as required output
- the default CMASS model exposes h-unit public parameter names
- the CLI surface should regenerate both profile plots in one JSON-producing call
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


LEGACY_EMCEE_BLOB_DTYPE = np.dtype(
    [
        ("total_log_prob_seconds", np.float64),
        ("likelihood_seconds", np.float64),
        ("normalization_seconds", np.float64),
        ("fp_prior_seconds", np.float64),
        ("normalization_value", np.float64),
        ("fp_prior_log_term", np.float64),
        ("fpfit_mu", np.float64),
        ("fpfit_beta", np.float64),
        ("fpfit_xi", np.float64),
        ("fpfit_scatter", np.float64),
        ("parallel_strategy", "S16"),
    ]
)


def _box_prior_payload() -> dict[str, list[float]]:
    """Return the fixed CMASS h-unit box-prior payload."""

    return {
        "mu5h_0": [9.0, 12.0],
        "beta5h": [-3.0, 3.0],
        "xi5h": [-3.0, 3.0],
        "sigma5h": [1.0e-2, 0.2],
        "mu_gamma_0": [1.5, 2.5],
        "beta_sigma_star_gamma": [-3.0, 3.0],
        "sigma_gamma": [0.0, 0.5],
        "mu_zs": [1.0, 3.0],
        "sigma_zs": [0.0, 2.0],
        "theta0": [0.0, 3.0],
        "loga": [-1.0, 3.0],
    }


def _initial_center_payload() -> dict[str, float]:
    """Build the fixed CMASS public initial-center payload."""

    return {
        "mu5h_0": 11.17,
        "beta5h": 0.59,
        "xi5h": -0.11,
        "sigma5h": 0.06,
        "mu_gamma_0": 1.99,
        "beta_sigma_star_gamma": 0.24,
        "sigma_gamma": 0.149,
        "mu_zs": 1.8,
        "sigma_zs": 0.215,
        "theta0": 0.93,
        "loga": 1.0,
    }


def _parameter_center() -> np.ndarray:
    """Return one deterministic posterior center matching the fixed schema."""

    return np.array([11.17, 0.59, -0.11, 0.06, 1.99, 0.24, 0.149, 1.8, 0.215, 0.93, 1.0], dtype=float)


def _lens_only_parameter_center() -> np.ndarray:
    """Return one deterministic posterior center matching `cmass_lens_only`."""

    return np.array([11.0, 0.15, 11.17, 0.59, -0.11, 0.06, 1.99, 0.24, 0.149], dtype=float)


def _write_corner_config(
    path: Path,
    profile_name: str,
    output_root: Path,
    warmup: int = 2,
) -> Path:
    """
    Create the minimal config snapshot needed by the corner-plot workflow.

    The corner post-processing code only needs the selected profile, fixed
    model schema, and stored warmup length.
    """

    config = {
        "profile": {"name": profile_name},
        "unit_convention": "h_units_v1",
        "model": {"name": "cmass"},
        "data": {
            "inference_dataset_path": str(output_root / f"{profile_name}_canonical_inference_dataset.hdf5"),
        },
        "box_prior": _box_prior_payload(),
        "sampling": {
            "n_walkers": 24,
            "n_steps": 5,
            "burn_in": warmup,
            "random_seed": 7,
            "initial_center": _initial_center_payload(),
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
    sample_count: int = 5,
    walker_count: int = 24,
) -> None:
    """
    Create a deterministic backend chain with small but non-degenerate spread.

    Using a real `emcee` backend keeps the fixture faithful to the production
    storage format and avoids hand-writing fragile HDF5 structures.
    """

    backend = emcee.backends.HDFBackend(str(chain_path))
    backend.reset(walker_count, parameter_center.shape[0])
    blobs = np.zeros(walker_count, dtype=LEGACY_EMCEE_BLOB_DTYPE)
    blobs["parallel_strategy"] = b"off"
    backend.grow(sample_count, blobs)
    random_state = np.random.RandomState(123).get_state()

    for step_index in range(sample_count):
        coords = np.tile(parameter_center, (walker_count, 1))
        coords += 1.0e-3 * step_index
        coords += np.linspace(0.0, 2.0e-4, walker_count, dtype=float)[:, None]
        log_prob = np.full(walker_count, -5.0 + 0.1 * step_index, dtype=float)
        state = emcee.State(coords, log_prob=log_prob, blobs=blobs.copy(), random_state=random_state)
        backend.save_step(state, np.ones(walker_count, dtype=bool))


def _build_corner_run(
    tmp_path: Path,
    profile_name: str,
    warmup: int = 2,
    sample_count: int = 5,
) -> Path:
    """Create a minimal completed run directory for the corner-plot tests."""

    run_dir = tmp_path / "runs" / profile_name / f"20260315_090000_{profile_name}_synthetic_corner"
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_corner_config(
        run_dir / "config_snapshot.yaml",
        profile_name=profile_name,
        output_root=tmp_path,
        warmup=warmup,
    )
    _seed_corner_backend(
        run_dir / "chain.h5",
        parameter_center=_parameter_center(),
        sample_count=sample_count,
    )
    return run_dir


def _build_lens_only_corner_run(
    tmp_path: Path,
    synthetic_lens_only_config_path: Path,
    warmup: int = 1,
    sample_count: int = 5,
) -> Path:
    """
    Create a completed `cmass_lens_only` run directory for corner-plot tests.

    This fixture deliberately reuses the production-style lens-only YAML from
    `conftest.py`. The regression we need to catch lives in the post-processing
    boundary: the chain is valid, but the corner renderer must not assume that
    every model starts with the four aperture-mass population parameters.
    """

    run_dir = tmp_path / "runs" / "devauc" / "20260512_202812_devauc_cmass-lens-only"
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = yaml.safe_load(synthetic_lens_only_config_path.read_text(encoding="utf-8"))
    payload["profile"] = {"name": "devauc"}
    payload["sampling"]["burn_in"] = warmup
    (run_dir / "config_snapshot.yaml").write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    _seed_corner_backend(
        run_dir / "chain.h5",
        parameter_center=_lens_only_parameter_center(),
        sample_count=sample_count,
    )
    return run_dir


def test_run_posterior_corner_generates_expected_artifacts_for_cmass(tmp_path: Path) -> None:
    """A completed default CMASS run should produce figure and JSON artifacts in-place."""

    from cmass_lens_inference.posterior_corner import run_posterior_corner

    run_dir = _build_corner_run(tmp_path, profile_name="sersic")
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
    assert result.metadata["parameter_order"][:4] == ["mu5h_0", "beta5h", "xi5h", "sigma5h"]
    assert result.metadata["mass_definition"]["label"] == "m5_hinvkpc"

    payload = json.loads(result.result_path.read_text(encoding="utf-8"))
    assert payload["profile_name"] == "sersic"
    assert payload["status"] == "completed"
    assert payload["burn_in_applied"] == 2
    assert payload["n_posterior_samples"] == 72
    assert payload["metadata"]["parameter_order"][:4] == ["mu5h_0", "beta5h", "xi5h", "sigma5h"]


def test_run_posterior_corner_calls_corner_with_public_hunit_labels(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """The corner renderer should expose h-unit mass labels."""

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

    run_dir = _build_corner_run(tmp_path, profile_name="devauc")
    runtime_config = load_runtime_config(run_dir / "config_snapshot.yaml")
    run_posterior_corner(run_dir=str(run_dir), burn_in="auto")

    expected_labels = [
        r"$\mu_{5,0}$",
        r"$\beta_{5}$",
        r"$\xi_{5}$",
        r"$\sigma_{5}$",
    ]
    assert captured["samples_shape"] == (72, runtime_config.parameter_schema.n_dim)
    assert captured["kwargs"]["labels"][:4] == expected_labels
    assert captured["kwargs"]["titles"][:4] == expected_labels
    assert captured["kwargs"]["show_titles"] is True
    assert captured["kwargs"]["title_fmt"] == ".2f"
    assert captured["kwargs"]["quantiles"] == [0.16, 0.5, 0.84]
    assert captured["kwargs"]["plot_datapoints"] is False
    assert captured["kwargs"]["levels"] == [0.68, 0.95]


def test_run_posterior_corner_uses_schema_driven_parameter_order_for_cmass(
    tmp_path: Path,
) -> None:
    """Corner metadata should follow the fixed CMASS schema."""

    from cmass_lens_inference.posterior_corner import run_posterior_corner

    run_dir = _build_corner_run(tmp_path, profile_name="sersic")
    result = run_posterior_corner(run_dir=str(run_dir), burn_in="auto")

    assert result.metadata["parameter_order"] == [
        "mu5h_0",
        "beta5h",
        "xi5h",
        "sigma5h",
        "mu_gamma_0",
        "beta_sigma_star_gamma",
        "sigma_gamma",
        "mu_zs",
        "sigma_zs",
        "theta0",
        "loga",
    ]
    assert len(result.metadata["parameter_labels"]) == 11


def test_run_posterior_corner_labels_cmass_lens_only_schema_by_name(
    tmp_path: Path,
    synthetic_lens_only_config_path: Path,
    monkeypatch,
) -> None:
    """Lens-only corner labels should follow the active model schema by name."""

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

    run_dir = _build_lens_only_corner_run(tmp_path, synthetic_lens_only_config_path)
    result = run_posterior_corner(run_dir=str(run_dir), burn_in="auto")

    expected_order = [
        "mu_mstar_lens",
        "sigma_mstar_lens",
        "mu5h_0",
        "beta5h",
        "xi5h",
        "sigma5h",
        "mu_gamma_0",
        "beta_sigma_star_gamma",
        "sigma_gamma",
    ]
    expected_labels = [
        r"$\mu_{\log M_{\ast,\mathrm{lens}}}$",
        r"$\sigma_{\log M_{\ast,\mathrm{lens}}}$",
        r"$\mu_{5,0}$",
        r"$\beta_{5}$",
        r"$\xi_{5}$",
        r"$\sigma_{5}$",
        r"$\mu_{\gamma,0}$",
        r"$\beta_{\Sigma_\ast,\gamma}$",
        r"$\sigma_{\gamma}$",
    ]

    assert result.metadata["parameter_order"] == expected_order
    assert result.metadata["parameter_labels"] == expected_labels
    assert captured["samples_shape"] == (96, len(expected_order))
    assert captured["kwargs"]["labels"] == expected_labels


def test_cli_posterior_corner_latest_command_generates_both_profiles(tmp_path: Path) -> None:
    """The CLI should regenerate both profile plots and emit machine-readable JSON."""

    devauc_run_dir = _build_corner_run(tmp_path, profile_name="devauc")
    sersic_run_dir = _build_corner_run(tmp_path, profile_name="sersic")
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
    assert payload["devauc_result"]["metadata"]["parameter_order"][:4] == ["mu5h_0", "beta5h", "xi5h", "sigma5h"]
    assert payload["sersic_result"]["metadata"]["parameter_order"][:4] == ["mu5h_0", "beta5h", "xi5h", "sigma5h"]
