from __future__ import annotations

from pathlib import Path

import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPOSITORY_ROOT / "workspace"
INFERENCE_CONFIG_ROOT = WORKSPACE_ROOT / "configs" / "inference"
POSTERIOR_CONFIG_ROOT = WORKSPACE_ROOT / "configs" / "posterior_predictive"
RECIPE_ROOT = WORKSPACE_ROOT / "recipes"


EXPECTED_RUNNABLE_CONFIGS = {
    Path("workspace/configs/inference/cmass/devauc.yaml"),
    Path("workspace/configs/inference/sonnenfeld2024_slacs/sonnenfeld2024_slacs.yaml"),
    Path("workspace/configs/inference/sonnenfeld2024_slacs/sonnenfeld2024_slacs_hunit.yaml"),
    Path("workspace/configs/inference/sonnenfeld2024_slacs/sonnenfeld2024_slacs_sigma_star_gamma.yaml"),
    Path("workspace/configs/inference/sonnenfeld2024_slacs/sonnenfeld2024_slacs_sigma_star_gamma_hunit.yaml"),
}

EXPECTED_BLOCKED_CONFIGS = {
    Path("workspace/configs/inference/cmass/sersic.yaml"): (
        "workspace/data/canonical/inference_dataset_sersic_slit_m5_hunits_v1.hdf5"
    ),
}


def _load_yaml(path: Path) -> dict:
    """Load a YAML mapping and fail loudly if the file is malformed."""

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict), f"{path} must contain a YAML mapping."
    return payload


def _dataset_path_for_config(config_path: Path) -> Path:
    """Return the canonical dataset path declared by one inference config."""

    payload = _load_yaml(config_path)
    return REPOSITORY_ROOT / payload["data"]["inference_dataset_path"]


def _runnable_post_canonical_configs() -> list[Path]:
    """Return inference configs whose canonical input exists right now.

    This helper encodes the current audit slice. It does not say configs with
    missing canonical datasets are invalid forever; it only excludes them from
    the after-data-preparation workflow being repaired here.
    """

    runnable: list[Path] = []
    for relative_config_path in sorted(EXPECTED_RUNNABLE_CONFIGS):
        config_path = REPOSITORY_ROOT / relative_config_path
        assert config_path.is_file(), config_path
        assert _dataset_path_for_config(config_path).is_file(), config_path
        runnable.append(config_path)
    return runnable


def _diagnostics_config_for_inference_config(inference_config_path: Path) -> Path:
    """Map one runnable inference config to its required diagnostics config path."""

    relative_path = inference_config_path.relative_to(INFERENCE_CONFIG_ROOT)
    return POSTERIOR_CONFIG_ROOT / relative_path.with_name(f"{relative_path.stem}_diagnostics.yaml")


def _post_canonical_recipe_for_inference_config(inference_config_path: Path) -> Path:
    """Map one runnable inference config to its required post-canonical recipe."""

    relative_path = inference_config_path.relative_to(INFERENCE_CONFIG_ROOT)
    return RECIPE_ROOT / relative_path.with_name(f"{relative_path.stem}_diagnostics_from_canonical.yaml")


def test_expected_runnable_and_blocked_inference_configs_are_classified() -> None:
    """The current workflow slice must classify missing data as blocked, not invalid."""

    for relative_config_path in EXPECTED_RUNNABLE_CONFIGS:
        config_path = REPOSITORY_ROOT / relative_config_path
        assert _dataset_path_for_config(config_path).is_file(), relative_config_path

    for relative_config_path, missing_dataset in EXPECTED_BLOCKED_CONFIGS.items():
        config_path = REPOSITORY_ROOT / relative_config_path
        assert config_path.is_file(), relative_config_path
        assert str(_dataset_path_for_config(config_path).relative_to(REPOSITORY_ROOT)) == missing_dataset
        assert not _dataset_path_for_config(config_path).is_file(), relative_config_path


def test_runnable_configs_have_matching_posterior_diagnostics_configs() -> None:
    """Every runnable post-canonical inference path needs diagnostics config coverage."""

    from statistical_sl.posterior_predictive.registry import get_predictive_definition

    missing: list[str] = []
    for inference_config_path in _runnable_post_canonical_configs():
        diagnostics_config_path = _diagnostics_config_for_inference_config(inference_config_path)
        if not diagnostics_config_path.is_file():
            missing.append(str(diagnostics_config_path.relative_to(REPOSITORY_ROOT)))
            continue

        payload = _load_yaml(diagnostics_config_path)
        assert payload["schema_version"] == "statistical_sl_posterior_predictive_config_v1"
        assert payload["workflow"] == "posterior_diagnostics"
        assert payload["inputs"]["inference_run_dir"] is None
        assert Path(payload["outputs"]["output_root_dir"]) == Path("workspace/outputs")
        get_predictive_definition(payload["model"]["name"])

        sigma_table_path = payload["inputs"]["sigma_table_path"]
        if payload["model"]["name"] == "cmass":
            assert sigma_table_path == "workspace/data/external/hunits_v1/jeans_deV_sigma_bundle.h5"
            assert (REPOSITORY_ROOT / sigma_table_path).is_file()
        else:
            assert sigma_table_path is None

    assert not missing, "\n".join(missing)


def test_runnable_configs_have_post_canonical_recipes() -> None:
    """Current post-canonical recipes should not include data-preparation steps."""

    missing: list[str] = []
    for inference_config_path in _runnable_post_canonical_configs():
        recipe_path = _post_canonical_recipe_for_inference_config(inference_config_path)
        if not recipe_path.is_file():
            missing.append(str(recipe_path.relative_to(REPOSITORY_ROOT)))
            continue

        payload = _load_yaml(recipe_path)
        steps = payload["steps"]
        assert payload["schema_version"] == "statistical_sl_pipeline_v1"
        assert payload["workspace_root"] == "../.."
        assert set(steps) == {"inference", "posterior_predictive"}
        assert "data_preparation" not in steps

        recipe_dir = recipe_path.parent
        inference_config = (recipe_dir / steps["inference"]["config"]).resolve()
        posterior_config = (recipe_dir / steps["posterior_predictive"]["config"]).resolve()
        dataset_path = (WORKSPACE_ROOT / steps["inference"]["dataset"]).resolve()

        assert inference_config == inference_config_path.resolve()
        assert posterior_config == _diagnostics_config_for_inference_config(inference_config_path).resolve()
        assert dataset_path.is_file()
        assert steps["posterior_predictive"]["run_dir"] == "${steps.inference.output_run_dir}"
        assert steps["posterior_predictive"]["result_dir"] == (
            "${steps.inference.output_run_dir}/posterior_predictive/diagnostics/${diagnostic_run_id}"
        )

    assert not missing, "\n".join(missing)
