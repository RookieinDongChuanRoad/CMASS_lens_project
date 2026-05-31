# Post-Canonical Inference Diagnostics Workflow Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Repair the workflow slice that starts after canonical datasets already exist, so inference and posterior diagnostics can be configured, validated, dry-run, and executed consistently with the repository integration plan.

**Architecture:** This plan does not redefine the whole repository as canonical-only and does not remove data preparation from the long-term architecture. It treats the three existing canonical HDF5 files as the available inputs for this audit slice, adds missing posterior diagnostics configs and post-canonical recipes for the configs that can run from those files, and adds a lightweight pipeline layer that can validate and dry-run this post-canonical slice without starting expensive scientific jobs. Existing inference artifacts remain at the run root for compatibility; posterior diagnostics remain nested under `posterior_predictive/diagnostics/<diagnostic_run_id>/`.

**Tech Stack:** Python 3.11 in the `cmass_lens` conda environment, `PyYAML`, dataclasses, pathlib, pytest, the existing `statistical_sl.inference.runner.run_inference` API, the existing `statistical_sl.inference.posterior_corner.run_posterior_corner` API, and the existing `statistical_sl.posterior_predictive.predictive.run_posterior_diagnostics` API.

---

## Scope, Constraints, And Completion Gate

This plan is an after-data-preparation repair. It deliberately ignores data-preparation execution while checking what remains broken between existing canonical datasets, inference configs, posterior diagnostics configs, recipes, CLI entrypoints, and output contracts.

Important distinction:

- In this plan, "existing canonical dataset" is a workflow slice boundary for review and repair.
- It is not a permanent repository constraint.
- It must not be used as a reason to delete data-preparation code, data-preparation configs, or model support whose canonical input is not currently present.
- A config whose canonical dataset is absent is "blocked for this slice", not necessarily invalid forever.

Execution constraints:

- Run all commands through `conda run -n cmass_lens --no-capture-output`.
- Do not run long inference or posterior diagnostics during implementation verification; tests must use validation, dry-run, or monkeypatched execution.
- Do not write new production artifacts to root-level `data/` or `outputs/`.
- Do not make `src/`, `tests/`, `workspace/`, README current workflow, or pipeline recipes depend on `legacy/`.
- Do not reintroduce `cmass_lens_inference`, `lensing_posterior_predictive`, or old top-level workflow directories as production imports.
- Keep CMASS de Vaucouleurs diagnostics tied to `workspace/data/external/hunits_v1/jeans_deV_sigma_bundle.h5`.
- Keep Sonnenfeld diagnostics sigma-table-free unless the model adapter explicitly starts requiring one.
- Keep `workspace/configs/inference/cmass/sersic.yaml` as a legitimate config file if it is still useful, but classify it as blocked for this after-data-preparation slice while `workspace/data/canonical/inference_dataset_sersic_slit_m5_hunits_v1.hdf5` is absent.

Definition of done:

- Runnable and blocked configs are classified by an explicit rule:
  a runnable post-canonical config is listed in this plan's runnable inventory
  and its `data.inference_dataset_path` exists under `workspace/data/canonical`;
  a blocked config is listed in the blocked inventory and its missing dataset
  is reported without deleting or invalidating the config.
- Every runnable post-canonical inference config has exactly one matching
  posterior diagnostics config at the path derived from its inference config
  stem, and every such diagnostics config has a `model.name` accepted by the
  posterior predictive registry.
- CMASS de Vaucouleurs diagnostics use the existing
  `workspace/data/external/hunits_v1/jeans_deV_sigma_bundle.h5`; Sonnenfeld
  diagnostics explicitly use `sigma_table_path: null`.
- Every runnable post-canonical inference config has exactly one matching
  recipe named `*_diagnostics_from_canonical.yaml`; those recipes are marked
  `mode: post_canonical`, contain only `inference` and `posterior_predictive`
  steps, and point at existing canonical HDF5 inputs.
- Existing data-preparation configs and full-pipeline recipes are not treated
  as errors merely because this plan does not execute them. Validation for this
  plan must target `mode: post_canonical` recipes or `*_from_canonical.yaml`
  recipes; full recipes are outside this completion gate unless explicitly
  requested.
- `statistical-sl pipeline validate --recipe <post-canonical-recipe>` validates
  every post-canonical recipe without starting scientific jobs.
- `statistical-sl pipeline run --recipe <post-canonical-recipe> --dry-run`
  prints the planned `run_inference`, `run_posterior_corner`, and
  `run_posterior_diagnostics` actions,
  is tested with monkeypatched scientific runners that would fail if called,
  and does not create new run directories or root-level output trees.
- `statistical-sl posterior-predictive posterior-diagnostics --config <config>
  --run-dir <run_dir>` is a tested config-driven entrypoint. Tests must verify
  YAML defaults, CLI overrides, missing-run-dir failure, and the final kwargs
  passed to `run_posterior_diagnostics`.
- The artifact contract is both documented and tested at the path-contract
  level: inference readers still expect `config_snapshot.yaml` and `chain.h5`
  at the run root, pipeline runs write inference-owned `posterior_corner.png`
  and `posterior_corner_result.json` at the run root, and posterior diagnostics
  materialize results under `posterior_predictive/diagnostics/<diagnostic_run_id>/`,
  not under flat `ppc/`.
- The final verification command set covers import smoke, CLI smoke, recipe
  validation, dry-run no-side-effect checks, focused tests, static forbidden
  dependency search, output-tree hygiene, and `git diff --check`.

Non-goals:

- No canonical dataset generation.
- No deletion of data-preparation code or configs.
- No deletion of sersic model code or sersic inference config solely because the sersic canonical HDF5 is absent in the current workspace.
- No migration of existing run directories.
- No change to scientific model semantics, priors, units, parameter order, or posterior diagnostics algorithms.
- No real long-running inference or posterior diagnostics execution unless the user explicitly approves it after dry-run validation.

## Current Post-Canonical Inventory

The implementation must treat these as runnable in this workflow slice because their canonical datasets exist:

```text
workspace/configs/inference/cmass/devauc.yaml
  dataset: workspace/data/canonical/inference_dataset_devauc_slit_m5_hunits_v1.hdf5
  posterior config to add: workspace/configs/posterior_predictive/cmass/devauc_diagnostics.yaml
  post-canonical recipe to add: workspace/recipes/cmass/devauc_diagnostics_from_canonical.yaml

workspace/configs/inference/sonnenfeld2024_slacs/sonnenfeld2024_slacs.yaml
  dataset: workspace/data/canonical/inference_dataset_sonnenfeld2024_slacs_m5_fixed_v1.hdf5
  posterior config to add: workspace/configs/posterior_predictive/sonnenfeld2024_slacs/sonnenfeld2024_slacs_diagnostics.yaml
  post-canonical recipe to add: workspace/recipes/sonnenfeld2024_slacs/sonnenfeld2024_slacs_diagnostics_from_canonical.yaml

workspace/configs/inference/sonnenfeld2024_slacs/sonnenfeld2024_slacs_hunit.yaml
  dataset: workspace/data/canonical/inference_dataset_sonnenfeld2024_slacs_m5_hunits_v1.hdf5
  posterior config to add: workspace/configs/posterior_predictive/sonnenfeld2024_slacs/sonnenfeld2024_slacs_hunit_diagnostics.yaml
  post-canonical recipe to add: workspace/recipes/sonnenfeld2024_slacs/sonnenfeld2024_slacs_hunit_diagnostics_from_canonical.yaml

workspace/configs/inference/sonnenfeld2024_slacs/sonnenfeld2024_slacs_sigma_star_gamma.yaml
  dataset: workspace/data/canonical/inference_dataset_sonnenfeld2024_slacs_m5_fixed_v1.hdf5
  posterior config to add: workspace/configs/posterior_predictive/sonnenfeld2024_slacs/sonnenfeld2024_slacs_sigma_star_gamma_diagnostics.yaml
  post-canonical recipe to add: workspace/recipes/sonnenfeld2024_slacs/sonnenfeld2024_slacs_sigma_star_gamma_diagnostics_from_canonical.yaml

workspace/configs/inference/sonnenfeld2024_slacs/sonnenfeld2024_slacs_sigma_star_gamma_hunit.yaml
  dataset: workspace/data/canonical/inference_dataset_sonnenfeld2024_slacs_m5_hunits_v1.hdf5
  posterior config to add: workspace/configs/posterior_predictive/sonnenfeld2024_slacs/sonnenfeld2024_slacs_sigma_star_gamma_hunit_diagnostics.yaml
  post-canonical recipe to add: workspace/recipes/sonnenfeld2024_slacs/sonnenfeld2024_slacs_sigma_star_gamma_hunit_diagnostics_from_canonical.yaml
```

The implementation must treat this as blocked in this workflow slice, not as a deletion target:

```text
workspace/configs/inference/cmass/sersic.yaml
  missing dataset: workspace/data/canonical/inference_dataset_sersic_slit_m5_hunits_v1.hdf5
  current status: blocked until data preparation or data sync provides the canonical HDF5
```

The existing `workspace/recipes/cmass/sersic_diagnostics.yaml` may remain as a full or pre-existing recipe. This plan must not use it as the proof that the post-canonical workflow is complete because it includes `data_preparation` and targets a currently missing canonical dataset.

## File Structure

Create:

- `src/statistical_sl/pipeline/__init__.py`: public exports for pipeline recipe loading, validation, dry-run planning, and execution.
- `src/statistical_sl/pipeline/recipe.py`: typed recipe parser and validator. It must support post-canonical recipes and must not globally reject recipes that contain `data_preparation`.
- `src/statistical_sl/pipeline/runner.py`: dry-run and real execution orchestration using existing inference and posterior diagnostics APIs.
- `src/statistical_sl/pipeline/cli.py`: `statistical-sl pipeline validate` and `statistical-sl pipeline run`.
- `src/statistical_sl/posterior_predictive/config.py`: typed loader for posterior diagnostics YAML configs.
- `tests/test_post_canonical_workflow_contract.py`: workspace coverage tests for runnable post-canonical inference configs, posterior configs, and recipes.
- `tests/test_posterior_predictive_config.py`: unit tests for posterior diagnostics config loading and CLI merge behavior.
- `tests/test_pipeline_recipe.py`: unit tests for recipe validation and dry-run planning.
- `tests/test_pipeline_cli.py`: CLI smoke tests for `pipeline validate` and `pipeline run --dry-run`.
- `workspace/configs/posterior_predictive/cmass/devauc_diagnostics.yaml`: CMASS de Vaucouleurs diagnostics config.
- `workspace/configs/posterior_predictive/sonnenfeld2024_slacs/`: posterior diagnostics configs for Sonnenfeld variants.
- `workspace/recipes/cmass/devauc_diagnostics_from_canonical.yaml`: current post-canonical CMASS de Vaucouleurs recipe.
- `workspace/recipes/sonnenfeld2024_slacs/`: current post-canonical recipes for Sonnenfeld variants.

Modify:

- `src/statistical_sl/cli.py`: add `pipeline` to the root command router.
- `src/statistical_sl/posterior_predictive/cli.py`: add `--config` support to `posterior-diagnostics`.
- `workspace/outputs/README.md`: document current artifact contract with inference artifacts in run root.
- `workspace/README.md`: document the post-canonical recipes as the current audited inference + diagnostics slice.
- `README.md`: update workflow examples.
- `docs/superpowers/plans/2026-05-24-repository-integration-structure.md`: add a short amendment noting the accepted current artifact contract and the difference between post-canonical recipes and full pipeline recipes.
- `tests/test_workspace_layout.py`: stop hard-coding the old sersic recipe as the only recipe contract; validate post-canonical recipes generically while allowing full recipes to exist separately.
- `tests/test_dependency_boundaries.py`: keep the existing boundary gate; no functional change expected unless new files need to be included automatically.

Do not delete:

- `workspace/configs/inference/cmass/sersic.yaml`
- `workspace/configs/posterior_predictive/cmass/sersic_diagnostics.yaml`
- `workspace/recipes/cmass/sersic_diagnostics.yaml`

If a later implementation finds those files actively misleading, the correct repair is to mark them with explicit metadata or move them only after user approval. This plan itself does not require deleting them.

## Task 1: Add Failing Post-Canonical Workflow Contract Tests

**Files:**

- Create: `tests/test_post_canonical_workflow_contract.py`
- Modify: none
- Test: `tests/test_post_canonical_workflow_contract.py`

- [ ] **Step 1: Create contract tests that fail on the current workspace**

Create `tests/test_post_canonical_workflow_contract.py` with:

```python
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
```

- [ ] **Step 2: Run the new tests and verify they fail for the right reasons**

Run:

```bash
conda run -n cmass_lens --no-capture-output \
  env PYTHONPATH="$PWD/src:/Users/liurongfu/tools" \
  python -m pytest tests/test_post_canonical_workflow_contract.py -q
```

Expected before implementation:

```text
FAILED tests/test_post_canonical_workflow_contract.py::test_runnable_configs_have_matching_posterior_diagnostics_configs
FAILED tests/test_post_canonical_workflow_contract.py::test_runnable_configs_have_post_canonical_recipes
```

The failure must mention missing CMASS devauc and Sonnenfeld posterior diagnostics configs or post-canonical recipes. The classification test should pass if sersic is present but blocked by its missing canonical HDF5.

- [ ] **Step 3: Commit the failing tests**

```bash
git add tests/test_post_canonical_workflow_contract.py
git commit -m "test: define post-canonical workflow contract"
```

## Task 2: Add Posterior Diagnostics Configs And Post-Canonical Recipes

**Files:**

- Create: `workspace/configs/posterior_predictive/cmass/devauc_diagnostics.yaml`
- Create: `workspace/configs/posterior_predictive/sonnenfeld2024_slacs/sonnenfeld2024_slacs_diagnostics.yaml`
- Create: `workspace/configs/posterior_predictive/sonnenfeld2024_slacs/sonnenfeld2024_slacs_hunit_diagnostics.yaml`
- Create: `workspace/configs/posterior_predictive/sonnenfeld2024_slacs/sonnenfeld2024_slacs_sigma_star_gamma_diagnostics.yaml`
- Create: `workspace/configs/posterior_predictive/sonnenfeld2024_slacs/sonnenfeld2024_slacs_sigma_star_gamma_hunit_diagnostics.yaml`
- Create: `workspace/recipes/cmass/devauc_diagnostics_from_canonical.yaml`
- Create: `workspace/recipes/sonnenfeld2024_slacs/sonnenfeld2024_slacs_diagnostics_from_canonical.yaml`
- Create: `workspace/recipes/sonnenfeld2024_slacs/sonnenfeld2024_slacs_hunit_diagnostics_from_canonical.yaml`
- Create: `workspace/recipes/sonnenfeld2024_slacs/sonnenfeld2024_slacs_sigma_star_gamma_diagnostics_from_canonical.yaml`
- Create: `workspace/recipes/sonnenfeld2024_slacs/sonnenfeld2024_slacs_sigma_star_gamma_hunit_diagnostics_from_canonical.yaml`
- Test: `tests/test_post_canonical_workflow_contract.py`

- [ ] **Step 1: Add CMASS devauc posterior diagnostics config**

Create `workspace/configs/posterior_predictive/cmass/devauc_diagnostics.yaml` with:

```yaml
schema_version: statistical_sl_posterior_predictive_config_v1
workflow: posterior_diagnostics
model:
  name: cmass
profile:
  name: devauc

inputs:
  inference_run_dir: null
  sigma_table_path: workspace/data/external/hunits_v1/jeans_deV_sigma_bundle.h5

execution:
  n_posterior_draws: 1000
  burn_in: auto
  random_seed: 20260309
  parent_sample_size: 100000
  worker_processes: null
  n_mass_bins: 19
  mass_bin_min: 10.15
  mass_bin_max: 12.05

outputs:
  output_root_dir: workspace/outputs
  result_dir: ${steps.inference.output_run_dir}/posterior_predictive/diagnostics/${diagnostic_run_id}

notes:
  purpose: >
    Post-canonical CMASS de Vaucouleurs posterior diagnostics.  The concrete
    inference run directory is supplied by a pipeline recipe or by the CLI
    --run-dir option.
```

- [ ] **Step 2: Add Sonnenfeld posterior diagnostics configs**

Create `workspace/configs/posterior_predictive/sonnenfeld2024_slacs/sonnenfeld2024_slacs_diagnostics.yaml` with:

```yaml
schema_version: statistical_sl_posterior_predictive_config_v1
workflow: posterior_diagnostics
model:
  name: sonnenfeld2024_slacs
profile:
  name: devauc

inputs:
  inference_run_dir: null
  sigma_table_path: null

execution:
  n_posterior_draws: 1000
  burn_in: auto
  random_seed: 20260309
  parent_sample_size: 100000
  worker_processes: null
  n_mass_bins: 19
  mass_bin_min: 10.15
  mass_bin_max: 12.05

outputs:
  output_root_dir: workspace/outputs
  result_dir: ${steps.inference.output_run_dir}/posterior_predictive/diagnostics/${diagnostic_run_id}

notes:
  purpose: >
    Post-canonical Sonnenfeld 2024 SLACS paper-native diagnostics. The model
    adapter does not require a CMASS observed-aperture sigma bundle.
```

Create `workspace/configs/posterior_predictive/sonnenfeld2024_slacs/sonnenfeld2024_slacs_hunit_diagnostics.yaml` with:

```yaml
schema_version: statistical_sl_posterior_predictive_config_v1
workflow: posterior_diagnostics
model:
  name: sonnenfeld2024_slacs_hunit
profile:
  name: devauc

inputs:
  inference_run_dir: null
  sigma_table_path: null

execution:
  n_posterior_draws: 1000
  burn_in: auto
  random_seed: 20260309
  parent_sample_size: 100000
  worker_processes: null
  n_mass_bins: 19
  mass_bin_min: 10.15
  mass_bin_max: 12.05

outputs:
  output_root_dir: workspace/outputs
  result_dir: ${steps.inference.output_run_dir}/posterior_predictive/diagnostics/${diagnostic_run_id}

notes:
  purpose: >
    Post-canonical Sonnenfeld 2024 SLACS h-unit diagnostics. The model adapter
    does not require a CMASS observed-aperture sigma bundle.
```

Create `workspace/configs/posterior_predictive/sonnenfeld2024_slacs/sonnenfeld2024_slacs_sigma_star_gamma_diagnostics.yaml` with:

```yaml
schema_version: statistical_sl_posterior_predictive_config_v1
workflow: posterior_diagnostics
model:
  name: sonnenfeld2024_slacs_sigma_star_gamma
profile:
  name: devauc

inputs:
  inference_run_dir: null
  sigma_table_path: null

execution:
  n_posterior_draws: 1000
  burn_in: auto
  random_seed: 20260309
  parent_sample_size: 100000
  worker_processes: null
  n_mass_bins: 19
  mass_bin_min: 10.15
  mass_bin_max: 12.05

outputs:
  output_root_dir: workspace/outputs
  result_dir: ${steps.inference.output_run_dir}/posterior_predictive/diagnostics/${diagnostic_run_id}

notes:
  purpose: >
    Post-canonical Sonnenfeld 2024 SLACS sigma-star gamma diagnostics. The
    model adapter does not require a CMASS observed-aperture sigma bundle.
```

Create `workspace/configs/posterior_predictive/sonnenfeld2024_slacs/sonnenfeld2024_slacs_sigma_star_gamma_hunit_diagnostics.yaml` with:

```yaml
schema_version: statistical_sl_posterior_predictive_config_v1
workflow: posterior_diagnostics
model:
  name: sonnenfeld2024_slacs_sigma_star_gamma_hunit
profile:
  name: devauc

inputs:
  inference_run_dir: null
  sigma_table_path: null

execution:
  n_posterior_draws: 1000
  burn_in: auto
  random_seed: 20260309
  parent_sample_size: 100000
  worker_processes: null
  n_mass_bins: 19
  mass_bin_min: 10.15
  mass_bin_max: 12.05

outputs:
  output_root_dir: workspace/outputs
  result_dir: ${steps.inference.output_run_dir}/posterior_predictive/diagnostics/${diagnostic_run_id}

notes:
  purpose: >
    Post-canonical Sonnenfeld 2024 SLACS sigma-star gamma h-unit diagnostics.
    The model adapter does not require a CMASS observed-aperture sigma bundle.
```

- [ ] **Step 3: Add CMASS devauc post-canonical recipe**

Create `workspace/recipes/cmass/devauc_diagnostics_from_canonical.yaml` with:

```yaml
schema_version: statistical_sl_pipeline_v1
name: cmass_devauc_diagnostics_from_canonical
workspace_root: ../..
mode: post_canonical

steps:
  inference:
    config: ../../configs/inference/cmass/devauc.yaml
    dataset: data/canonical/inference_dataset_devauc_slit_m5_hunits_v1.hdf5
    output_run_dir: outputs/devauc/${run_id}

  posterior_predictive:
    config: ../../configs/posterior_predictive/cmass/devauc_diagnostics.yaml
    run_dir: ${steps.inference.output_run_dir}
    output_root_dir: outputs
    result_dir: ${steps.inference.output_run_dir}/posterior_predictive/diagnostics/${diagnostic_run_id}
```

- [ ] **Step 4: Add Sonnenfeld post-canonical recipes**

Create `workspace/recipes/sonnenfeld2024_slacs/sonnenfeld2024_slacs_diagnostics_from_canonical.yaml` with:

```yaml
schema_version: statistical_sl_pipeline_v1
name: sonnenfeld2024_slacs_diagnostics_from_canonical
workspace_root: ../..
mode: post_canonical

steps:
  inference:
    config: ../../configs/inference/sonnenfeld2024_slacs/sonnenfeld2024_slacs.yaml
    dataset: data/canonical/inference_dataset_sonnenfeld2024_slacs_m5_fixed_v1.hdf5
    output_run_dir: outputs/devauc/${run_id}

  posterior_predictive:
    config: ../../configs/posterior_predictive/sonnenfeld2024_slacs/sonnenfeld2024_slacs_diagnostics.yaml
    run_dir: ${steps.inference.output_run_dir}
    output_root_dir: outputs
    result_dir: ${steps.inference.output_run_dir}/posterior_predictive/diagnostics/${diagnostic_run_id}
```

Create `workspace/recipes/sonnenfeld2024_slacs/sonnenfeld2024_slacs_hunit_diagnostics_from_canonical.yaml` with:

```yaml
schema_version: statistical_sl_pipeline_v1
name: sonnenfeld2024_slacs_hunit_diagnostics_from_canonical
workspace_root: ../..
mode: post_canonical

steps:
  inference:
    config: ../../configs/inference/sonnenfeld2024_slacs/sonnenfeld2024_slacs_hunit.yaml
    dataset: data/canonical/inference_dataset_sonnenfeld2024_slacs_m5_hunits_v1.hdf5
    output_run_dir: outputs/devauc/${run_id}

  posterior_predictive:
    config: ../../configs/posterior_predictive/sonnenfeld2024_slacs/sonnenfeld2024_slacs_hunit_diagnostics.yaml
    run_dir: ${steps.inference.output_run_dir}
    output_root_dir: outputs
    result_dir: ${steps.inference.output_run_dir}/posterior_predictive/diagnostics/${diagnostic_run_id}
```

Create `workspace/recipes/sonnenfeld2024_slacs/sonnenfeld2024_slacs_sigma_star_gamma_diagnostics_from_canonical.yaml` with:

```yaml
schema_version: statistical_sl_pipeline_v1
name: sonnenfeld2024_slacs_sigma_star_gamma_diagnostics_from_canonical
workspace_root: ../..
mode: post_canonical

steps:
  inference:
    config: ../../configs/inference/sonnenfeld2024_slacs/sonnenfeld2024_slacs_sigma_star_gamma.yaml
    dataset: data/canonical/inference_dataset_sonnenfeld2024_slacs_m5_fixed_v1.hdf5
    output_run_dir: outputs/devauc/${run_id}

  posterior_predictive:
    config: ../../configs/posterior_predictive/sonnenfeld2024_slacs/sonnenfeld2024_slacs_sigma_star_gamma_diagnostics.yaml
    run_dir: ${steps.inference.output_run_dir}
    output_root_dir: outputs
    result_dir: ${steps.inference.output_run_dir}/posterior_predictive/diagnostics/${diagnostic_run_id}
```

Create `workspace/recipes/sonnenfeld2024_slacs/sonnenfeld2024_slacs_sigma_star_gamma_hunit_diagnostics_from_canonical.yaml` with:

```yaml
schema_version: statistical_sl_pipeline_v1
name: sonnenfeld2024_slacs_sigma_star_gamma_hunit_diagnostics_from_canonical
workspace_root: ../..
mode: post_canonical

steps:
  inference:
    config: ../../configs/inference/sonnenfeld2024_slacs/sonnenfeld2024_slacs_sigma_star_gamma_hunit.yaml
    dataset: data/canonical/inference_dataset_sonnenfeld2024_slacs_m5_hunits_v1.hdf5
    output_run_dir: outputs/devauc/${run_id}

  posterior_predictive:
    config: ../../configs/posterior_predictive/sonnenfeld2024_slacs/sonnenfeld2024_slacs_sigma_star_gamma_hunit_diagnostics.yaml
    run_dir: ${steps.inference.output_run_dir}
    output_root_dir: outputs
    result_dir: ${steps.inference.output_run_dir}/posterior_predictive/diagnostics/${diagnostic_run_id}
```

- [ ] **Step 5: Run contract tests**

Run:

```bash
conda run -n cmass_lens --no-capture-output \
  env PYTHONPATH="$PWD/src:/Users/liurongfu/tools" \
  python -m pytest tests/test_post_canonical_workflow_contract.py -q
```

Expected:

```text
3 passed
```

- [ ] **Step 6: Commit workspace workflow coverage**

```bash
git add \
  workspace/configs/posterior_predictive/cmass/devauc_diagnostics.yaml \
  workspace/configs/posterior_predictive/sonnenfeld2024_slacs \
  workspace/recipes/cmass/devauc_diagnostics_from_canonical.yaml \
  workspace/recipes/sonnenfeld2024_slacs \
  tests/test_post_canonical_workflow_contract.py
git commit -m "fix: add post-canonical diagnostics recipes"
```

## Task 3: Add Posterior Diagnostics Config Loading And CLI Merge

**Files:**

- Create: `src/statistical_sl/posterior_predictive/config.py`
- Modify: `src/statistical_sl/posterior_predictive/cli.py`
- Create: `tests/test_posterior_predictive_config.py`
- Test: `tests/test_posterior_predictive_config.py`

- [ ] **Step 1: Add failing tests for posterior diagnostics config loading**

Create `tests/test_posterior_predictive_config.py` with:

```python
from __future__ import annotations

from pathlib import Path

import pytest


def test_load_posterior_diagnostics_config_normalizes_paths() -> None:
    """The workspace YAML should become explicit run kwargs without running PPC."""

    from statistical_sl.posterior_predictive.config import load_posterior_diagnostics_config

    config = load_posterior_diagnostics_config(
        "workspace/configs/posterior_predictive/cmass/devauc_diagnostics.yaml"
    )

    assert config.model_name == "cmass"
    assert config.profile_name == "devauc"
    assert config.inference_run_dir is None
    assert config.sigma_table_path == Path("workspace/data/external/hunits_v1/jeans_deV_sigma_bundle.h5").resolve()
    assert config.output_root_dir == Path("workspace/outputs").resolve()
    assert config.n_posterior_draws == 1000
    assert config.burn_in == "auto"
    assert config.random_seed == 20260309
    assert config.parent_sample_size == 100000
    assert config.worker_processes is None
    assert config.n_mass_bins == 19
    assert config.mass_bin_min == 10.15
    assert config.mass_bin_max == 12.05


def test_posterior_diagnostics_config_requires_run_dir_from_yaml_or_cli() -> None:
    """A reusable config may omit run_dir, but execution kwargs may not."""

    from statistical_sl.posterior_predictive.config import load_posterior_diagnostics_config

    config = load_posterior_diagnostics_config(
        "workspace/configs/posterior_predictive/cmass/devauc_diagnostics.yaml"
    )

    with pytest.raises(ValueError, match="inference run directory"):
        config.to_run_kwargs()

    kwargs = config.to_run_kwargs(run_dir_override="workspace/outputs/devauc/latest")
    assert kwargs["run_dir"] == str(Path("workspace/outputs/devauc/latest").resolve())
    assert kwargs["sigma_table_path"] == Path("workspace/data/external/hunits_v1/jeans_deV_sigma_bundle.h5").resolve()
    assert kwargs["output_root_dir"] == Path("workspace/outputs").resolve()


def test_posterior_diagnostics_cli_merges_config_and_cli_overrides(monkeypatch, capsys) -> None:
    """The CLI should merge YAML defaults and explicit CLI overrides before dispatch."""

    import sys

    import statistical_sl.posterior_predictive.cli as posterior_cli

    captured_kwargs: dict[str, object] = {}

    class FakeDiagnosticsResult:
        def to_dict(self) -> dict[str, object]:
            return {"status": "completed", "result_dir": "unused"}

    def fake_run_posterior_diagnostics(**kwargs):
        captured_kwargs.update(kwargs)
        return FakeDiagnosticsResult()

    monkeypatch.setattr(posterior_cli, "run_posterior_diagnostics", fake_run_posterior_diagnostics)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "statistical-sl posterior-predictive",
            "posterior-diagnostics",
            "--config",
            "workspace/configs/posterior_predictive/cmass/devauc_diagnostics.yaml",
            "--run-dir",
            "workspace/outputs/devauc/latest",
            "--diagnostic-run-id",
            "diagnostic-smoke",
            "--n-posterior-draws",
            "17",
            "--seed",
            "123",
        ],
    )

    posterior_cli.main()

    captured = capsys.readouterr()
    assert '"status": "completed"' in captured.out
    assert captured_kwargs["run_dir"] == str(Path("workspace/outputs/devauc/latest").resolve())
    assert captured_kwargs["diagnostic_run_id"] == "diagnostic-smoke"
    assert captured_kwargs["n_posterior_draws"] == 17
    assert captured_kwargs["random_seed"] == 123
    assert captured_kwargs["sigma_table_path"] == Path("workspace/data/external/hunits_v1/jeans_deV_sigma_bundle.h5").resolve()
```

- [ ] **Step 2: Run tests and verify they fail because the loader does not exist**

Run:

```bash
conda run -n cmass_lens --no-capture-output \
  env PYTHONPATH="$PWD/src:/Users/liurongfu/tools" \
  python -m pytest tests/test_posterior_predictive_config.py -q
```

Expected:

```text
ModuleNotFoundError: No module named 'statistical_sl.posterior_predictive.config'
```

- [ ] **Step 3: Implement the typed config loader**

Create `src/statistical_sl/posterior_predictive/config.py` with:

```python
"""
Configuration loading for posterior diagnostics workflows.

This module gives posterior diagnostics the same durable config surface that
inference already has. The config intentionally stores reusable execution
settings separately from the concrete inference run directory: a run directory
is produced by inference and is often known only after a pipeline run starts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


SUPPORTED_SCHEMA_VERSION = "statistical_sl_posterior_predictive_config_v1"
SUPPORTED_WORKFLOW = "posterior_diagnostics"


def _require_mapping(payload: dict[str, Any], section_name: str) -> dict[str, Any]:
    """Return a required YAML mapping section with a direct error message."""

    section = payload.get(section_name)
    if not isinstance(section, dict):
        raise TypeError(f"Posterior diagnostics config section '{section_name}' must be a mapping.")
    return section


def _optional_path(raw_value: object) -> Path | None:
    """Normalize optional path values while preserving omitted inputs as None."""

    if raw_value is None:
        return None
    text = str(raw_value)
    if "${" in text:
        return None
    return Path(text).expanduser().resolve()


@dataclass(frozen=True)
class PosteriorDiagnosticsConfig:
    """Typed settings needed to run one posterior diagnostics workflow."""

    path: Path
    model_name: str
    profile_name: str
    inference_run_dir: Path | None
    sigma_table_path: Path | None
    output_root_dir: Path
    result_dir_template: str | None
    n_posterior_draws: int | None
    burn_in: str | int
    random_seed: int
    parent_sample_size: int
    worker_processes: int | None
    n_mass_bins: int
    mass_bin_min: float
    mass_bin_max: float

    def to_run_kwargs(
        self,
        *,
        run_dir_override: str | Path | None = None,
        diagnostic_run_id: str | None = None,
    ) -> dict[str, object]:
        """Return kwargs accepted by `run_posterior_diagnostics`."""

        run_dir = Path(run_dir_override).expanduser().resolve() if run_dir_override is not None else self.inference_run_dir
        if run_dir is None:
            raise ValueError(
                "Posterior diagnostics require an inference run directory. "
                "Set inputs.inference_run_dir in the config or pass --run-dir."
            )

        return {
            "run_dir": str(run_dir),
            "sigma_table_path": self.sigma_table_path,
            "output_root_dir": self.output_root_dir,
            "diagnostic_run_id": diagnostic_run_id,
            "n_posterior_draws": self.n_posterior_draws,
            "burn_in": self.burn_in,
            "random_seed": self.random_seed,
            "parent_sample_size": self.parent_sample_size,
            "worker_processes": self.worker_processes,
            "n_mass_bins": self.n_mass_bins,
            "mass_bin_min": self.mass_bin_min,
            "mass_bin_max": self.mass_bin_max,
        }


def load_posterior_diagnostics_config(config_path: str | Path) -> PosteriorDiagnosticsConfig:
    """Load and validate one posterior diagnostics YAML config."""

    path = Path(config_path).expanduser().resolve()
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("Posterior diagnostics config must be a YAML mapping.")
    if payload.get("schema_version") != SUPPORTED_SCHEMA_VERSION:
        raise ValueError(f"Unsupported posterior diagnostics config schema: {payload.get('schema_version')!r}")
    if payload.get("workflow") != SUPPORTED_WORKFLOW:
        raise ValueError(f"Unsupported posterior diagnostics workflow: {payload.get('workflow')!r}")

    model = _require_mapping(payload, "model")
    profile = _require_mapping(payload, "profile")
    inputs = _require_mapping(payload, "inputs")
    execution = _require_mapping(payload, "execution")
    outputs = _require_mapping(payload, "outputs")

    return PosteriorDiagnosticsConfig(
        path=path,
        model_name=str(model["name"]),
        profile_name=str(profile["name"]),
        inference_run_dir=_optional_path(inputs.get("inference_run_dir")),
        sigma_table_path=_optional_path(inputs.get("sigma_table_path")),
        output_root_dir=Path(outputs.get("output_root_dir", "workspace/outputs")).expanduser().resolve(),
        result_dir_template=None if outputs.get("result_dir") is None else str(outputs["result_dir"]),
        n_posterior_draws=(
            None
            if execution.get("n_posterior_draws") is None
            else int(execution["n_posterior_draws"])
        ),
        burn_in=execution.get("burn_in", "auto"),
        random_seed=int(execution.get("random_seed", 20260309)),
        parent_sample_size=int(execution.get("parent_sample_size", 100000)),
        worker_processes=(
            None
            if execution.get("worker_processes") is None
            else int(execution["worker_processes"])
        ),
        n_mass_bins=int(execution.get("n_mass_bins", 10)),
        mass_bin_min=float(execution.get("mass_bin_min", 10.9)),
        mass_bin_max=float(execution.get("mass_bin_max", 11.9)),
    )
```

- [ ] **Step 4: Make posterior-diagnostics CLI consume the config**

In `src/statistical_sl/posterior_predictive/cli.py`, add this import:

```python
from .config import load_posterior_diagnostics_config
```

Change the diagnostics parser arguments so `--run-dir` is optional and `--config` is available:

```python
diagnostics_parser.add_argument("--config", default=None, help="Posterior diagnostics YAML config")
diagnostics_parser.add_argument("--run-dir", default=None, help="Completed inference run directory")
```

Replace the `elif args.command == "posterior-diagnostics":` block with:

```python
    elif args.command == "posterior-diagnostics":
        burn_in = args.burn_in
        if burn_in != "auto":
            burn_in = int(burn_in)

        if args.config is not None:
            config = load_posterior_diagnostics_config(args.config)
            kwargs = config.to_run_kwargs(
                run_dir_override=args.run_dir,
                diagnostic_run_id=args.diagnostic_run_id,
            )
            if args.sigma_table is not None:
                kwargs["sigma_table_path"] = args.sigma_table
            if args.output_dir != str(DEFAULT_PPC_OUTPUT_ROOT_DIR):
                kwargs["output_root_dir"] = args.output_dir
            if args.n_posterior_draws != DEFAULT_TREND_POSTERIOR_DRAWS:
                kwargs["n_posterior_draws"] = args.n_posterior_draws
            if args.burn_in != "auto":
                kwargs["burn_in"] = burn_in
            if args.seed != 20260309:
                kwargs["random_seed"] = args.seed
            if args.parent_sample_size != DEFAULT_DIAGNOSTICS_PARENT_SAMPLE_SIZE:
                kwargs["parent_sample_size"] = args.parent_sample_size
            if args.worker_processes is not None:
                kwargs["worker_processes"] = args.worker_processes
            if args.n_mass_bins != DEFAULT_TREND_MASS_BIN_COUNT:
                kwargs["n_mass_bins"] = args.n_mass_bins
            if args.mass_bin_min != DEFAULT_TREND_MASS_BIN_MIN:
                kwargs["mass_bin_min"] = args.mass_bin_min
            if args.mass_bin_max != DEFAULT_TREND_MASS_BIN_MAX:
                kwargs["mass_bin_max"] = args.mass_bin_max
            result = run_posterior_diagnostics(**kwargs)
        else:
            if args.run_dir is None:
                parser.error("posterior-diagnostics requires --run-dir unless --config provides inputs.inference_run_dir.")
            result = run_posterior_diagnostics(
                run_dir=args.run_dir,
                sigma_table_path=args.sigma_table,
                output_root_dir=args.output_dir,
                diagnostic_run_id=args.diagnostic_run_id,
                n_posterior_draws=args.n_posterior_draws,
                burn_in=burn_in,
                random_seed=args.seed,
                parent_sample_size=args.parent_sample_size,
                worker_processes=args.worker_processes,
                n_mass_bins=args.n_mass_bins,
                mass_bin_min=args.mass_bin_min,
                mass_bin_max=args.mass_bin_max,
            )
```

- [ ] **Step 5: Run posterior config tests**

Run:

```bash
conda run -n cmass_lens --no-capture-output \
  env PYTHONPATH="$PWD/src:/Users/liurongfu/tools" \
  python -m pytest tests/test_posterior_predictive_config.py -q
```

Expected:

```text
3 passed
```

- [ ] **Step 6: Commit config-driven diagnostics**

```bash
git add \
  src/statistical_sl/posterior_predictive/config.py \
  src/statistical_sl/posterior_predictive/cli.py \
  tests/test_posterior_predictive_config.py
git commit -m "feat: load posterior diagnostics configs"
```

## Task 4: Add Pipeline Recipe Validation, Dry-Run, And Execution

**Files:**

- Create: `src/statistical_sl/pipeline/__init__.py`
- Create: `src/statistical_sl/pipeline/recipe.py`
- Create: `src/statistical_sl/pipeline/runner.py`
- Create: `src/statistical_sl/pipeline/cli.py`
- Modify: `src/statistical_sl/cli.py`
- Create: `tests/test_pipeline_recipe.py`
- Create: `tests/test_pipeline_cli.py`
- Test: `tests/test_pipeline_recipe.py`, `tests/test_pipeline_cli.py`

- [ ] **Step 1: Add failing tests for recipe loading and dry-run planning**

Create `tests/test_pipeline_recipe.py` with:

```python
from __future__ import annotations

from pathlib import Path


def test_load_pipeline_recipe_resolves_post_canonical_steps() -> None:
    """A post-canonical recipe should resolve configs without running data prep."""

    from statistical_sl.pipeline.recipe import load_pipeline_recipe

    recipe = load_pipeline_recipe("workspace/recipes/cmass/devauc_diagnostics_from_canonical.yaml")

    assert recipe.name == "cmass_devauc_diagnostics_from_canonical"
    assert recipe.mode == "post_canonical"
    assert recipe.workspace_root == Path("workspace").resolve()
    assert set(recipe.steps) == {"inference", "posterior_predictive"}
    assert recipe.inference_config_path == Path("workspace/configs/inference/cmass/devauc.yaml").resolve()
    assert recipe.posterior_diagnostics_config_path == Path(
        "workspace/configs/posterior_predictive/cmass/devauc_diagnostics.yaml"
    ).resolve()
    assert recipe.dataset_path == Path(
        "workspace/data/canonical/inference_dataset_devauc_slit_m5_hunits_v1.hdf5"
    ).resolve()


def test_pipeline_dry_run_describes_existing_post_canonical_workflow() -> None:
    """Dry-run must show concrete actions while avoiding scientific execution."""

    from statistical_sl.pipeline.recipe import load_pipeline_recipe
    from statistical_sl.pipeline.runner import plan_pipeline_run

    recipe = load_pipeline_recipe("workspace/recipes/cmass/devauc_diagnostics_from_canonical.yaml")
    plan = plan_pipeline_run(recipe, diagnostic_run_id="diagnostic-smoke")

    assert plan.recipe_name == "cmass_devauc_diagnostics_from_canonical"
    assert plan.inference_config_path == Path("workspace/configs/inference/cmass/devauc.yaml").resolve()
    assert plan.posterior_diagnostics_config_path == Path(
        "workspace/configs/posterior_predictive/cmass/devauc_diagnostics.yaml"
    ).resolve()
    assert plan.diagnostic_run_id == "diagnostic-smoke"
    assert "run_inference" in plan.actions[0]
    assert "run_posterior_corner" in plan.actions[1]
    assert "run_posterior_diagnostics" in plan.actions[2]
```

Create `tests/test_pipeline_cli.py` with:

```python
from __future__ import annotations


def test_pipeline_validate_cli_accepts_devauc_post_canonical_recipe(capsys) -> None:
    """The public CLI should validate recipes without launching inference."""

    from statistical_sl.cli import main

    exit_code = main([
        "pipeline",
        "validate",
        "--recipe",
        "workspace/recipes/cmass/devauc_diagnostics_from_canonical.yaml",
    ])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "cmass_devauc_diagnostics_from_canonical" in captured.out
    assert "valid" in captured.out


def test_pipeline_run_dry_run_cli_prints_planned_actions(capsys) -> None:
    """Dry-run is the safe default verification path for long scientific jobs."""

    from statistical_sl.cli import main

    exit_code = main([
        "pipeline",
        "run",
        "--recipe",
        "workspace/recipes/cmass/devauc_diagnostics_from_canonical.yaml",
        "--diagnostic-run-id",
        "diagnostic-smoke",
        "--dry-run",
    ])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "run_inference" in captured.out
    assert "run_posterior_corner" in captured.out
    assert "run_posterior_diagnostics" in captured.out
    assert "diagnostic-smoke" in captured.out


def test_pipeline_run_dry_run_does_not_call_scientific_runners(monkeypatch, capsys) -> None:
    """Dry-run must stay side-effect free even if the real runners are importable."""

    import statistical_sl.pipeline.runner as pipeline_runner
    from statistical_sl.cli import main

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("dry-run must not call scientific runners")

    monkeypatch.setattr(pipeline_runner, "run_inference", fail_if_called)
    monkeypatch.setattr(pipeline_runner, "run_posterior_corner", fail_if_called)
    monkeypatch.setattr(pipeline_runner, "run_posterior_diagnostics", fail_if_called)

    exit_code = main([
        "pipeline",
        "run",
        "--recipe",
        "workspace/recipes/cmass/devauc_diagnostics_from_canonical.yaml",
        "--diagnostic-run-id",
        "diagnostic-smoke",
        "--dry-run",
    ])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "run_inference" in captured.out
    assert "run_posterior_corner" in captured.out
    assert "run_posterior_diagnostics" in captured.out
```

- [ ] **Step 2: Run tests and verify they fail because pipeline package does not exist**

Run:

```bash
conda run -n cmass_lens --no-capture-output \
  env PYTHONPATH="$PWD/src:/Users/liurongfu/tools" \
  python -m pytest tests/test_pipeline_recipe.py tests/test_pipeline_cli.py -q
```

Expected:

```text
ModuleNotFoundError: No module named 'statistical_sl.pipeline'
```

- [ ] **Step 3: Implement recipe parser**

Create `src/statistical_sl/pipeline/recipe.py` with:

```python
"""
Pipeline recipe parsing for Statistical_SL workflows.

Recipes are the user-facing orchestration contract. This parser supports the
post-canonical recipes repaired in this plan, while deliberately not banning
full recipes that include data-preparation steps for future end-to-end runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


SUPPORTED_SCHEMA_VERSION = "statistical_sl_pipeline_v1"


@dataclass(frozen=True)
class PipelineRecipe:
    """Resolved pipeline recipe."""

    path: Path
    name: str
    mode: str
    workspace_root: Path
    steps: dict[str, dict[str, Any]]
    inference_config_path: Path
    posterior_diagnostics_config_path: Path
    dataset_path: Path
    inference_output_run_dir_template: str
    diagnostics_result_dir_template: str


def _resolve_workspace_root(recipe_path: Path, raw_workspace_root: object) -> Path:
    """Resolve recipe-local workspace roots such as '../..'."""

    return (recipe_path.parent / str(raw_workspace_root)).resolve()


def _resolve_recipe_relative_path(recipe_path: Path, raw_path: object) -> Path:
    """Resolve config paths relative to the recipe file location."""

    return (recipe_path.parent / str(raw_path)).resolve()


def _require_step(steps: dict[str, Any], step_name: str) -> dict[str, Any]:
    """Return a required step mapping with a clear error message."""

    step = steps.get(step_name)
    if not isinstance(step, dict):
        raise ValueError(f"Pipeline recipe requires a mapping step named '{step_name}'.")
    return step


def load_pipeline_recipe(recipe_path: str | Path) -> PipelineRecipe:
    """Load and validate one pipeline recipe."""

    path = Path(recipe_path).expanduser().resolve()
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("Pipeline recipe must be a YAML mapping.")
    if payload.get("schema_version") != SUPPORTED_SCHEMA_VERSION:
        raise ValueError(f"Unsupported pipeline recipe schema: {payload.get('schema_version')!r}")

    steps = payload.get("steps")
    if not isinstance(steps, dict):
        raise ValueError("Pipeline recipe requires a mapping 'steps' section.")

    inference_step = _require_step(steps, "inference")
    posterior_step = _require_step(steps, "posterior_predictive")
    workspace_root = _resolve_workspace_root(path, payload["workspace_root"])
    inference_config_path = _resolve_recipe_relative_path(path, inference_step["config"])
    posterior_config_path = _resolve_recipe_relative_path(path, posterior_step["config"])
    dataset_path = workspace_root / str(inference_step["dataset"])

    missing_paths = [
        candidate
        for candidate in (workspace_root, inference_config_path, posterior_config_path, dataset_path)
        if not candidate.exists()
    ]
    if missing_paths:
        formatted = "\n".join(str(candidate) for candidate in missing_paths)
        raise FileNotFoundError(f"Pipeline recipe references missing paths:\n{formatted}")

    return PipelineRecipe(
        path=path,
        name=str(payload["name"]),
        mode=str(payload.get("mode", "full")),
        workspace_root=workspace_root,
        steps={key: dict(value) for key, value in steps.items()},
        inference_config_path=inference_config_path,
        posterior_diagnostics_config_path=posterior_config_path,
        dataset_path=dataset_path,
        inference_output_run_dir_template=str(inference_step["output_run_dir"]),
        diagnostics_result_dir_template=str(posterior_step["result_dir"]),
    )
```

- [ ] **Step 4: Implement pipeline runner and dry-run plan**

Create `src/statistical_sl/pipeline/runner.py` with:

```python
"""
Pipeline execution for inference plus posterior diagnostics.

The runner delegates scientific work to existing workflow APIs. Its narrow
responsibility is orchestration: validate the recipe, call inference, then pass
the completed run directory into posterior diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from statistical_sl.inference.posterior_corner import run_posterior_corner
from statistical_sl.inference.runner import run_inference
from statistical_sl.posterior_predictive.config import load_posterior_diagnostics_config
from statistical_sl.posterior_predictive.predictive import run_posterior_diagnostics
from statistical_sl.pipeline.recipe import PipelineRecipe


@dataclass(frozen=True)
class PipelineDryRunPlan:
    """Human-readable description of a pipeline run that has not executed."""

    recipe_name: str
    inference_config_path: Path
    posterior_diagnostics_config_path: Path
    diagnostic_run_id: str | None
    actions: tuple[str, ...]


@dataclass(frozen=True)
class PipelineRunResult:
    """Structured result of one executed pipeline."""

    recipe_name: str
    inference_run_dir: Path
    posterior_corner_figure_path: Path
    posterior_corner_result_path: Path
    diagnostics_result_dir: Path
    status: str


def plan_pipeline_run(
    recipe: PipelineRecipe,
    *,
    diagnostic_run_id: str | None = None,
) -> PipelineDryRunPlan:
    """Return the actions a pipeline run would execute."""

    diagnostics_config = load_posterior_diagnostics_config(recipe.posterior_diagnostics_config_path)
    actions = (
        f"run_inference(config_path={recipe.inference_config_path})",
        (
            "run_posterior_corner("
            "run_dir=<inference_result.run_dir>, "
            f"burn_in={diagnostics_config.burn_in!r})"
        ),
        (
            "run_posterior_diagnostics("
            "run_dir=<inference_result.run_dir>, "
            f"config_path={diagnostics_config.path}, "
            f"diagnostic_run_id={diagnostic_run_id!r})"
        ),
    )
    return PipelineDryRunPlan(
        recipe_name=recipe.name,
        inference_config_path=recipe.inference_config_path,
        posterior_diagnostics_config_path=recipe.posterior_diagnostics_config_path,
        diagnostic_run_id=diagnostic_run_id,
        actions=actions,
    )


def run_pipeline(
    recipe: PipelineRecipe,
    *,
    diagnostic_run_id: str | None = None,
) -> PipelineRunResult:
    """Execute inference followed by posterior diagnostics for one recipe."""

    inference_result = run_inference(str(recipe.inference_config_path))
    diagnostics_config = load_posterior_diagnostics_config(recipe.posterior_diagnostics_config_path)
    posterior_corner_result = run_posterior_corner(
        run_dir=inference_result.run_dir,
        burn_in=diagnostics_config.burn_in,
    )
    diagnostics_result = run_posterior_diagnostics(
        **diagnostics_config.to_run_kwargs(
            run_dir_override=inference_result.run_dir,
            diagnostic_run_id=diagnostic_run_id,
        )
    )
    return PipelineRunResult(
        recipe_name=recipe.name,
        inference_run_dir=Path(inference_result.run_dir),
        posterior_corner_figure_path=Path(posterior_corner_result.figure_path),
        posterior_corner_result_path=Path(posterior_corner_result.result_path),
        diagnostics_result_dir=Path(diagnostics_result.result_dir),
        status="completed",
    )
```

- [ ] **Step 5: Implement pipeline CLI**

Create `src/statistical_sl/pipeline/cli.py` with:

```python
"""Command-line interface for Statistical_SL pipeline recipes."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from statistical_sl.pipeline.recipe import load_pipeline_recipe
from statistical_sl.pipeline.runner import plan_pipeline_run, run_pipeline


def _json_ready(payload: object) -> object:
    """Convert dataclass and Path values into JSON-friendly structures."""

    if isinstance(payload, Path):
        return str(payload)
    if isinstance(payload, dict):
        return {key: _json_ready(value) for key, value in payload.items()}
    if isinstance(payload, (list, tuple)):
        return [_json_ready(value) for value in payload]
    return payload


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the pipeline recipe CLI parser."""

    parser = argparse.ArgumentParser(description="Run or validate Statistical_SL pipeline recipes")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate", help="Validate a pipeline recipe")
    validate_parser.add_argument("--recipe", required=True, help="Pipeline recipe YAML path")

    run_parser = subparsers.add_parser("run", help="Run a pipeline recipe")
    run_parser.add_argument("--recipe", required=True, help="Pipeline recipe YAML path")
    run_parser.add_argument("--diagnostic-run-id", default=None, help="Optional diagnostics artifact directory name")
    run_parser.add_argument("--dry-run", action="store_true", help="Print planned actions without running inference or diagnostics")

    return parser


def main() -> int:
    """Dispatch the selected pipeline command."""

    parser = build_argument_parser()
    args = parser.parse_args()
    recipe = load_pipeline_recipe(args.recipe)

    if args.command == "validate":
        print(json.dumps({"status": "valid", "recipe": recipe.name, "mode": recipe.mode}, indent=2, sort_keys=True))
        return 0

    if args.command == "run":
        if args.dry_run:
            plan = plan_pipeline_run(recipe, diagnostic_run_id=args.diagnostic_run_id)
            print(json.dumps(_json_ready(asdict(plan)), indent=2, sort_keys=True))
            return 0
        result = run_pipeline(recipe, diagnostic_run_id=args.diagnostic_run_id)
        print(json.dumps(_json_ready(asdict(result)), indent=2, sort_keys=True))
        return 0

    parser.error(f"Unsupported pipeline command: {args.command}")
    return 2
```

Create `src/statistical_sl/pipeline/__init__.py` with:

```python
"""Pipeline recipe support for Statistical_SL."""

from statistical_sl.pipeline.recipe import PipelineRecipe, load_pipeline_recipe
from statistical_sl.pipeline.runner import PipelineDryRunPlan, PipelineRunResult, plan_pipeline_run, run_pipeline

__all__ = [
    "PipelineDryRunPlan",
    "PipelineRecipe",
    "PipelineRunResult",
    "load_pipeline_recipe",
    "plan_pipeline_run",
    "run_pipeline",
]
```

- [ ] **Step 6: Add the pipeline command to the root CLI**

In `src/statistical_sl/cli.py`, add this entry to `WORKFLOW_COMMANDS`:

```python
    "pipeline": ("statistical_sl.pipeline.cli", "statistical-sl pipeline"),
```

- [ ] **Step 7: Run pipeline tests**

Run:

```bash
conda run -n cmass_lens --no-capture-output \
  env PYTHONPATH="$PWD/src:/Users/liurongfu/tools" \
  python -m pytest tests/test_pipeline_recipe.py tests/test_pipeline_cli.py -q
```

Expected:

```text
5 passed
```

- [ ] **Step 8: Commit pipeline runner**

```bash
git add \
  src/statistical_sl/cli.py \
  src/statistical_sl/pipeline \
  tests/test_pipeline_recipe.py \
  tests/test_pipeline_cli.py
git commit -m "feat: add post-canonical pipeline recipes"
```

## Task 5: Update Workspace Layout Tests And Documentation Contracts

**Files:**

- Modify: `tests/test_workspace_layout.py`
- Modify: `workspace/outputs/README.md`
- Modify: `workspace/README.md`
- Modify: `README.md`
- Modify: `docs/superpowers/plans/2026-05-24-repository-integration-structure.md`
- Test: `tests/test_workspace_layout.py`

- [ ] **Step 1: Replace hard-coded sersic recipe test with generic post-canonical recipe validation**

In `tests/test_workspace_layout.py`, add or replace with:

```python
def test_workspace_post_canonical_recipes_reference_existing_step_configs() -> None:
    """
    Post-canonical recipes are the current audited inference + diagnostics slice.

    Full recipes that include data preparation may exist separately. This test
    only validates recipes explicitly marked `mode: post_canonical`.
    """

    repository_root = Path(__file__).resolve().parents[1]
    recipe_root = repository_root / "workspace" / "recipes"
    recipe_paths = []
    for candidate in sorted(recipe_root.glob("**/*.yaml")):
        payload = yaml.safe_load(candidate.read_text(encoding="utf-8"))
        if payload.get("mode") == "post_canonical":
            recipe_paths.append(candidate)

    assert recipe_paths, "No post-canonical workspace pipeline recipes found."

    for recipe_path in recipe_paths:
        recipe = yaml.safe_load(recipe_path.read_text(encoding="utf-8"))
        assert recipe["schema_version"] == "statistical_sl_pipeline_v1"
        assert recipe["workspace_root"] == "../.."
        assert set(recipe["steps"]) == {"inference", "posterior_predictive"}

        recipe_dir = recipe_path.parent
        inference_step = recipe["steps"]["inference"]
        posterior_step = recipe["steps"]["posterior_predictive"]

        inference_config_path = (recipe_dir / inference_step["config"]).resolve()
        posterior_config_path = (recipe_dir / posterior_step["config"]).resolve()
        dataset_path = (repository_root / "workspace" / inference_step["dataset"]).resolve()

        assert inference_config_path.is_file(), inference_config_path
        assert posterior_config_path.is_file(), posterior_config_path
        assert dataset_path.is_file(), dataset_path
        assert posterior_step["run_dir"] == "${steps.inference.output_run_dir}"
        assert posterior_step["output_root_dir"] == "outputs"
        assert posterior_step["result_dir"] == (
            "${steps.inference.output_run_dir}/posterior_predictive/diagnostics/${diagnostic_run_id}"
        )
```

Do not remove sersic from general inference config coverage unless that test claims every config must have an existing dataset. If it does, split the test into "config exists" and "post-canonical runnable config has existing dataset" assertions.

- [ ] **Step 2: Update `workspace/outputs/README.md`**

Replace the layout block with:

```markdown
```text
outputs/
  <profile>/
    <run_id>/
      run_manifest.json
      config_snapshot.yaml
      metadata.json
      run_result.json
      chain.h5
      checkpoints/
      logs/
      posterior_predictive/
        diagnostics/
          <diagnostic_run_id>/
            run_manifest.json
            ppc_summary.json
            replicated_statistics.npz
            fig8_like.png
            fig8_like_summary.json
            fig8_like_curves.npz
            gamma_vs_*.png
            gamma_vs_*_summary.json
            gamma_vs_*_curves.npz
```
```

Add this paragraph below the block:

```markdown
Inference artifacts currently live at the run root because `posterior_corner`
and posterior diagnostics read `config_snapshot.yaml` and `chain.h5` from that
location. A future artifact-layout migration may introduce an `inference/`
subdirectory, but that requires compatibility handling for existing readers and
saved runs.
```

- [ ] **Step 3: Update `workspace/README.md`**

Add this section:

```markdown
## Current Post-Canonical Workflow Slice

The currently audited workflow slice starts after canonical datasets already
exist under `data/canonical/` and then runs:

1. inference from `configs/inference/...`
2. posterior diagnostics from `configs/posterior_predictive/...`

Recipes with `mode: post_canonical` connect those two steps. They do not imply
that data preparation is deprecated or removed; they only provide a focused run
surface for validating inference and diagnostics from existing canonical data.
```

- [ ] **Step 4: Update root `README.md`**

In the CLI section, add:

```markdown
Post-canonical pipeline recipe commands:

```bash
statistical-sl pipeline validate --recipe workspace/recipes/cmass/devauc_diagnostics_from_canonical.yaml
statistical-sl pipeline run --recipe workspace/recipes/cmass/devauc_diagnostics_from_canonical.yaml --dry-run
```

Run a posterior diagnostics config against an existing inference run:

```bash
statistical-sl posterior-predictive posterior-diagnostics \
  --config workspace/configs/posterior_predictive/cmass/devauc_diagnostics.yaml \
  --run-dir workspace/outputs/devauc/latest
```
```

- [ ] **Step 5: Add an amendment to the old integration plan**

Append this section near the output-layout discussion in `docs/superpowers/plans/2026-05-24-repository-integration-structure.md`:

```markdown
#### 2026-05-30 post-canonical workflow amendment

The post-canonical workflow repair validates the slice that starts after
canonical datasets already exist. This is not a decision to remove or deprecate
data preparation. Full recipes may still contain a data-preparation step; the
new `mode: post_canonical` recipes are focused entrypoints for inference plus
posterior diagnostics from already available canonical HDF5 files.

The same repair keeps inference artifacts at the run root:
`config_snapshot.yaml`, `chain.h5`, `metadata.json`, and `run_result.json`.
This is an intentional compatibility decision because current posterior corner
and posterior diagnostics readers load those files from the run root.

The retained invariant is still "one run directory owns all artifacts":
posterior diagnostics are nested below
`posterior_predictive/diagnostics/<diagnostic_run_id>/`. Moving inference
artifacts into an `inference/` subdirectory is deferred to a separate artifact
compatibility migration.
```

- [ ] **Step 6: Run workspace tests**

Run:

```bash
conda run -n cmass_lens --no-capture-output \
  env PYTHONPATH="$PWD/src:/Users/liurongfu/tools" \
  python -m pytest tests/test_workspace_layout.py tests/test_post_canonical_workflow_contract.py -q
```

Expected:

```text
all selected tests passed
```

- [ ] **Step 7: Commit docs and layout tests**

```bash
git add \
  README.md \
  workspace/README.md \
  workspace/outputs/README.md \
  docs/superpowers/plans/2026-05-24-repository-integration-structure.md \
  tests/test_workspace_layout.py
git commit -m "docs: document post-canonical workflow slice"
```

## Task 6: Full Static Boundary And Smoke Verification

**Files:**

- Modify: none unless verification reveals a defect.
- Test: full selected verification suite.

- [ ] **Step 1: Run import smoke**

Run:

```bash
conda run -n cmass_lens --no-capture-output \
  env PYTHONPATH="$PWD/src:/Users/liurongfu/tools" \
  python - <<'PY'
from pathlib import Path

import statistical_sl
import statistical_sl.inference
import statistical_sl.pipeline
import statistical_sl.posterior_predictive

repo = Path.cwd().resolve()
for module in (
    statistical_sl,
    statistical_sl.inference,
    statistical_sl.pipeline,
    statistical_sl.posterior_predictive,
):
    path = Path(module.__file__).resolve()
    print(f"{module.__name__}: {path}")
    if repo not in path.parents:
        raise SystemExit(f"{module.__name__} resolves outside checkout: {path}")
PY
```

Expected: all printed module paths are under the current checkout.

- [ ] **Step 2: Run CLI smoke**

Run:

```bash
conda run -n cmass_lens --no-capture-output \
  env PYTHONPATH="$PWD/src:/Users/liurongfu/tools" \
  statistical-sl --help

conda run -n cmass_lens --no-capture-output \
  env PYTHONPATH="$PWD/src:/Users/liurongfu/tools" \
  statistical-sl pipeline --help

conda run -n cmass_lens --no-capture-output \
  env PYTHONPATH="$PWD/src:/Users/liurongfu/tools" \
  statistical-sl posterior-predictive posterior-diagnostics --help
```

Expected: command help exits with status 0 and documents the new `pipeline` command plus diagnostics `--config`.

- [ ] **Step 3: Validate every post-canonical recipe**

Run:

```bash
for recipe in workspace/recipes/**/*_from_canonical.yaml; do
  conda run -n cmass_lens --no-capture-output \
    env PYTHONPATH="$PWD/src:/Users/liurongfu/tools" \
    statistical-sl pipeline validate --recipe "$recipe"
done
```

Expected: every recipe prints JSON with `"status": "valid"`.

- [ ] **Step 4: Dry-run every post-canonical recipe**

Run:

```bash
before_dirs=$(mktemp)
after_dirs=$(mktemp)
find workspace/outputs -mindepth 1 -maxdepth 3 -type d -print | sort > "$before_dirs"

for recipe in workspace/recipes/**/*_from_canonical.yaml; do
  conda run -n cmass_lens --no-capture-output \
    env PYTHONPATH="$PWD/src:/Users/liurongfu/tools" \
    statistical-sl pipeline run --recipe "$recipe" --diagnostic-run-id diagnostic-smoke --dry-run
done

find workspace/outputs -mindepth 1 -maxdepth 3 -type d -print | sort > "$after_dirs"
diff -u "$before_dirs" "$after_dirs"
```

Expected: every dry-run prints `run_inference`, `run_posterior_corner`, and `run_posterior_diagnostics` actions, and `diff` prints no output because dry-run did not create or remove output directories.

- [ ] **Step 5: Run focused tests**

Run:

```bash
conda run -n cmass_lens --no-capture-output \
  env PYTHONPATH="$PWD/src:/Users/liurongfu/tools" \
  python -m pytest \
    tests/test_post_canonical_workflow_contract.py \
    tests/test_posterior_predictive_config.py \
    tests/test_pipeline_recipe.py \
    tests/test_pipeline_cli.py \
    tests/test_workspace_layout.py \
    tests/test_dependency_boundaries.py \
    tests/test_posterior_predictive_layout.py \
    tests/test_core_contracts.py \
    tests/test_inference_config.py \
    -q
```

Expected: all selected tests pass.

- [ ] **Step 6: Run static forbidden-dependency search**

Run:

```bash
rg -n \
  "cmass_lens_inference|lensing_posterior_predictive|from prepare_dataset|import prepare_dataset|Bayesian_inference|Posterior_predictive_test|_legacy_paths|_legacy_imports|reexport_module|legacy/" \
  src tests workspace pyproject.toml README.md
```

Expected: no output from current production code, tests, workspace, pyproject, or root README.

- [ ] **Step 7: Check for accidental root outputs or source-tree outputs**

Run:

```bash
find src -maxdepth 4 -type d -name outputs -print
find . -maxdepth 2 -type d \( -name data -o -name outputs \) ! -path './workspace/*' -print
```

Expected:

```text
```

The second command intentionally excludes the official `workspace/data` and
`workspace/outputs` directories. Any remaining output must be an explicitly
explained historical ignored directory that existed before this repair. New
workflow dry-runs must not create root-level `data/`, root-level `outputs/`, or
`src/outputs`.

- [ ] **Step 8: Check diff hygiene**

Run:

```bash
git diff --check
git status --short
```

Expected:

```text
```

from `git diff --check`, and `git status --short` shows only the intended tracked changes plus any pre-existing unrelated dirty files already known before this plan execution.

- [ ] **Step 9: Commit verification fixes if any**

If verification required small fixes:

```bash
git add <fixed-files>
git commit -m "test: verify post-canonical workflow slice"
```

If no fixes were needed, do not create an empty commit.

## Final Acceptance Checklist

Before declaring this plan complete, verify these exact conditions:

- [ ] `workspace/configs/inference/cmass/sersic.yaml` still exists unless the user separately approves removal.
- [ ] `workspace/configs/posterior_predictive/cmass/sersic_diagnostics.yaml` and `workspace/recipes/cmass/sersic_diagnostics.yaml` still exist unless the user separately approves removal or replacement.
- [ ] `workspace/configs/posterior_predictive/cmass/devauc_diagnostics.yaml` exists and points to `jeans_deV_sigma_bundle.h5`.
- [ ] Four Sonnenfeld posterior diagnostics configs exist and have `sigma_table_path: null`.
- [ ] Five post-canonical recipes exist: one CMASS devauc plus four Sonnenfeld variants.
- [ ] Post-canonical recipes are marked `mode: post_canonical`.
- [ ] Post-canonical recipes contain only `inference` and `posterior_predictive` steps.
- [ ] The post-canonical workflow contract test proves sersic is classified as blocked for this slice, not invalid or deleted.
- [ ] The posterior predictive registry accepts every `model.name` declared by new diagnostics configs.
- [ ] The CMASS de Vaucouleurs sigma bundle file exists at `workspace/data/external/hunits_v1/jeans_deV_sigma_bundle.h5`.
- [ ] Existing full recipes or data-preparation configs are not deleted merely because this plan ignores data preparation execution.
- [ ] `statistical-sl pipeline validate --recipe workspace/recipes/cmass/devauc_diagnostics_from_canonical.yaml` passes.
- [ ] `statistical-sl pipeline run --recipe workspace/recipes/cmass/devauc_diagnostics_from_canonical.yaml --dry-run` does not start inference.
- [ ] Dry-run verification shows no new output directories under `workspace/outputs`.
- [ ] `statistical-sl posterior-predictive posterior-diagnostics --help` documents `--config`.
- [ ] Tests verify posterior diagnostics `--config` merges YAML defaults and CLI overrides into the kwargs passed to `run_posterior_diagnostics`.
- [ ] Tests verify missing posterior diagnostics run directory fails before any expensive diagnostic execution.
- [ ] Artifact-layout tests or path-contract tests verify posterior diagnostics use `posterior_predictive/diagnostics/<diagnostic_run_id>` and not flat `ppc/`.
- [ ] Tests listed in Task 6 Step 5 pass.
- [ ] Static forbidden-dependency search does not show current workflow dependencies on old packages, old directories, or `legacy/`.
- [ ] Final response reports any pre-existing unrelated dirty files separately from this repair.

## Execution Notes

Recommended execution mode: inline execution in this thread, because the same design correction affects configs, recipe semantics, CLI, and tests.

Safe checkpoint sequence:

1. Execute Tasks 1-2 and stop for review of the workspace contract.
2. Execute Tasks 3-4 and stop for review of the new CLI / pipeline runner.
3. Execute Tasks 5-6 and stop for final verification.

Do not run a real inference or posterior diagnostics job unless the user explicitly approves that expensive execution after the dry-run and validation steps pass.
