# CMASS Lens Project Workspace

This repository is a workstation-bound CMASS strong-lens research workspace,
not a generic plug-and-play package. It combines production-oriented inference
code, interpolation-grid preparation, comparison harnesses, and research-side
validation scripts that currently assume the local filesystem layout under
`/Users/liurongfu/Work/CMASS_lens_project`.

If you are taking over this workspace, treat this README as an operator manual:
it explains what each major area does, which assumptions are hard-coded today,
and what order of operations is expected when running the project.

## What Lives Here

### `prepare_intepolation_grids/`

Prepares or refreshes the HDF5 interpolation content consumed by the inference
and posterior-predictive workflows.

- Updates the raw observation HDF5 files under `data/raw/`
- Builds the external sigma-unit tables under `data/external/`
- Depends on the external `spherical_jeans` package
- Has its own environment notes in
  [`prepare_intepolation_grids/README.md`](prepare_intepolation_grids/README.md)

### `Bayesian_inference/`

Main hierarchical Bayesian inference package for the CMASS strong-lens study.

- Package source lives under `Bayesian_inference/src/cmass_lens_inference/`
- Runtime configs live under `Bayesian_inference/configs/`
- Provides CLI entrypoints for new runs and resume
- Writes main run artifacts under `outputs/`
- Scientific and modeling requirements are documented in
  [`Bayesian_inference/PROJECT_REQUIREMENTS.md`](Bayesian_inference/PROJECT_REQUIREMENTS.md)

### `key_tests/`

An isolated comparison workspace for running the current implementation against
the legacy reference implementation.

- Contains wrappers, generated configs, copied reference code, notebook assets,
  and comparison reports
- Uses `key_tests/run_comparison.py` as the orchestrator for the smoke/compare
  workflow
- Writes comparison artifacts under `key_tests/output/`
- Current comparison report lives at
  [`key_tests/reports/pipeline_comparison.md`](key_tests/reports/pipeline_comparison.md)

### `Posterior_predictive_test/`

Standalone posterior-predictive, trend, monitor, and notebook-comparison package.

- Package source lives under `Posterior_predictive_test/src/cmass_posterior_predictive/`
- Provides the `cmass-posterior-predictive` CLI for PPC, trends, and monitor workflows
- Keeps the one-off comparison driver:
  `Posterior_predictive_test/compare_full0103_notebook_vs_pipeline.py`
- Writes its artifacts under `Posterior_predictive_test/results/`
- Some defaults in this area point outside the repository to
  `/Users/liurongfu/Desktop/Spectrum_reduction/...`

### `data/`

Canonical local data layout for this workspace.

- `data/raw/`: immutable observation HDF5 inputs
- `data/external/`: external cross-section and sigma-unit grids
- `data/derived/`: reproducible transformed data products
- `data/caches/`: disposable caches

See [`data/README.md`](data/README.md) for the current layout summary.

### `outputs/`

Canonical output location for primary inference runs.

- `outputs/devauc/`: de Vaucouleurs-profile runs
- `outputs/sersic/`: Sersic-profile runs
- `outputs/*/latest`: current canonical latest run links
- `outputs/benchmarks/`: benchmark logs and timing outputs

## Hard Assumptions You Must Not Ignore

### Environment

The supported runtime environment is the conda environment `cmass_lens`.

The repository only tracks the baseline environment file in
[`prepare_intepolation_grids/environment.yml`](prepare_intepolation_grids/environment.yml):

```bash
conda env update -n cmass_lens -f /Users/liurongfu/Work/CMASS_lens_project/prepare_intepolation_grids/environment.yml
```

That environment file covers baseline scientific packages, but the inference
package itself is defined in `Bayesian_inference/pyproject.toml`. Install it
into the same conda environment before using the CLI:

```bash
cd /Users/liurongfu/Work/CMASS_lens_project/Bayesian_inference
conda run -n cmass_lens python -m pip install -e .
```

### External dependency

`prepare_intepolation_grids` depends on `spherical_jeans`, which is not
vendored in this repository. The environment must already be able to import it.

### Absolute-path assumptions

Many configs, scripts, and reports assume this exact root path:

```text
/Users/liurongfu/Work/CMASS_lens_project
```

This is especially true for:

- `Bayesian_inference/configs/*.yaml`
- `Posterior_predictive_test/compare_full0103_notebook_vs_pipeline.py`
- reports and manifests under `key_tests/`

### Missing large artifacts from Git

The repository does not track the full HDF5 inputs, sigma tables, chains, or
generated result trees. Full scientific runs require local copies of these
artifacts to exist under the expected directories, especially:

- `data/raw/*.hdf5`
- `data/external/*.h5`
- `outputs/`
- `key_tests/output/`
- `Posterior_predictive_test/results/`

## Standard Workflow

The safest way to work in this repository is to follow the same order the code
already assumes.

### 1. Validate the environment

First update the conda environment, then run the grid-preparation environment
check from the `prepare_intepolation_grids` directory:

```bash
cd /Users/liurongfu/Work/CMASS_lens_project/prepare_intepolation_grids
conda run -n cmass_lens python -m interpolation_grids.env_check
```

If you need the inference CLI in that environment, install the editable package
once from `Bayesian_inference/`.

### 2. Refresh raw interpolation grids or build external sigma tables

Run these commands from `prepare_intepolation_grids/`.

Process both standard raw observation files:

```bash
cd /Users/liurongfu/Work/CMASS_lens_project/prepare_intepolation_grids
conda run -n cmass_lens python -m interpolation_grids --all-default-inputs
```

Process a single file:

```bash
cd /Users/liurongfu/Work/CMASS_lens_project/prepare_intepolation_grids
conda run -n cmass_lens python -m interpolation_grids --input /Users/liurongfu/Work/CMASS_lens_project/data/raw/observations_with_m5_grids_all.hdf5
```

Build both posterior-predictive sigma-unit tables:

```bash
cd /Users/liurongfu/Work/CMASS_lens_project/prepare_intepolation_grids
conda run -n cmass_lens python -m interpolation_grids --build-sigma-unit-hdf5 --profile all --workers 14
```

These commands populate or refresh:

- `data/raw/*.updated.hdf5` or in-place replacements
- `data/external/jeans_deV_m5_grid.h5`
- `data/external/jeans_deV_m10_grid.h5`
- `data/external/jeans_sers_m5_grid.h5`
- `data/external/jeans_sers_m10_grid.h5`

### 3. Run the main inference

After installing `Bayesian_inference` into `cmass_lens`, use the packaged CLI.
The tracked configs already point at the canonical data paths and `outputs/`.
Every inference config now must include:

```yaml
mass_definition:
  enclosed_radius_kpc: 5  # or 10
```

The public hyper-parameter names follow that choice:
- `5 kpc`: `mu5_0`, `beta5`, `xi5`, `sigma5`
- `10 kpc`: `mu10_0`, `beta10`, `xi10`, `sigma10`

Run the de Vaucouleurs branch:

```bash
cd /Users/liurongfu/Work/CMASS_lens_project/Bayesian_inference
conda run -n cmass_lens cmass-lens-inference run --config /Users/liurongfu/Work/CMASS_lens_project/Bayesian_inference/configs/devauc.yaml
```

Run the Sersic branch:

```bash
cd /Users/liurongfu/Work/CMASS_lens_project/Bayesian_inference
conda run -n cmass_lens cmass-lens-inference run --config /Users/liurongfu/Work/CMASS_lens_project/Bayesian_inference/configs/sersic.yaml
```

Resume an existing run:

```bash
cd /Users/liurongfu/Work/CMASS_lens_project/Bayesian_inference
conda run -n cmass_lens cmass-lens-inference resume --run-dir /Users/liurongfu/Work/CMASS_lens_project/outputs/devauc/latest
```

### 4. Run posterior predictive tests or posterior trends

These commands use the `cmass-posterior-predictive` CLI from
`Posterior_predictive_test/` after that package has been installed into the
same `cmass_lens` environment.

Example devauc posterior predictive run:

```bash
cd /Users/liurongfu/Work/CMASS_lens_project/Posterior_predictive_test
conda run -n cmass_lens cmass-posterior-predictive posterior-predictive \
  --run-dir /Users/liurongfu/Work/CMASS_lens_project/outputs/devauc/latest \
  --sigma-table /Users/liurongfu/Work/CMASS_lens_project/data/external/jeans_deV_m5_grid.h5 \
  --output-dir /Users/liurongfu/Work/CMASS_lens_project/Posterior_predictive_test/results/devauc
```

Example sersic posterior-trend run:

```bash
cd /Users/liurongfu/Work/CMASS_lens_project/Posterior_predictive_test
conda run -n cmass_lens cmass-posterior-predictive posterior-trends \
  --run-dir /Users/liurongfu/Work/CMASS_lens_project/outputs/sersic/latest \
  --sigma-table /Users/liurongfu/Work/CMASS_lens_project/data/external/jeans_sers_m5_grid.h5 \
  --output-dir /Users/liurongfu/Work/CMASS_lens_project/Posterior_predictive_test/results/sersic
```

The CLI also exposes `posterior-predictive-monitor` if you need to wait for
fresh external sigma tables before running both profile branches. The monitor
now resolves the expected external filenames from each run's recorded mass
definition, so an `m10` run waits for `jeans_*_m10_grid.h5` rather than the
historical `m5` tables.

### 5. Run current-vs-reference comparison checks

The isolated comparison harness lives in `key_tests/` and should be run from
there because the scripts use sibling imports and workspace-relative paths.

```bash
cd /Users/liurongfu/Work/CMASS_lens_project/key_tests
conda run -n cmass_lens python run_comparison.py
```

This workflow prepares the workspace, refreshes copied reference files, runs
smoke and compare jobs for both profiles, and rewrites the comparison report.

### 6. Run notebook-vs-pipeline comparison scripts only when the external defaults are valid

`Posterior_predictive_test/compare_full0103_notebook_vs_pipeline.py` is a
research-side script, not a stable package CLI. Its default paths point to
desktop-side notebooks and chain files outside this repository. Override those
arguments if your local setup differs.

## Data and Output Conventions

- Canonical raw observation inputs live under `data/raw/`
- Canonical external grids live under `data/external/`
- Main inference outputs live under `outputs/`
- Current-vs-reference comparison artifacts live under `key_tests/output/`
- Notebook-vs-pipeline comparison artifacts live under
  `Posterior_predictive_test/results/`

Git intentionally ignores most heavy runtime artifacts, so absence from Git
history does not mean those files are optional for local scientific runs.

## Verification Entry Points

Useful checks already defined in the workspace:

- Grid-preparation environment check:

  ```bash
  cd /Users/liurongfu/Work/CMASS_lens_project/prepare_intepolation_grids
  conda run -n cmass_lens python -m interpolation_grids.env_check
  ```

- Grid-generation tests:

  ```bash
  cd /Users/liurongfu/Work/CMASS_lens_project/prepare_intepolation_grids
  conda run -n cmass_lens pytest -q tests/test_jeans_regression.py tests/test_hdf5_processing.py tests/test_sigma_unit_tables.py
  ```

- Inference / PPC tests:

  ```bash
  cd /Users/liurongfu/Work/CMASS_lens_project/Bayesian_inference
  conda run -n cmass_lens pytest -q tests/test_runner_cli.py tests/test_posterior_predictive.py
  ```

- Key-test workspace checks:

  ```bash
  cd /Users/liurongfu/Work/CMASS_lens_project/key_tests
  conda run -n cmass_lens pytest -q
  ```

## Important Limitations

- This workspace is path-bound. Moving it to another root directory will break
  configs and scripts unless you update hard-coded paths.
- `spherical_jeans` is external and not version-pinned here beyond “must be
  importable in `cmass_lens`”.
- Many large data and result artifacts are intentionally excluded from Git, so
  cloning the repository alone is not enough for full reproducibility.
- Some files in the tree are operator notes rather than polished product
  documentation, especially `task_plan.md`, `progress.md`, and `findings.md`.
- `Posterior_predictive_test/` and parts of `key_tests/` are mixed
  research/prototype tooling, not hardened public interfaces.

## Deeper References

- [`Bayesian_inference/PROJECT_REQUIREMENTS.md`](Bayesian_inference/PROJECT_REQUIREMENTS.md)
- [`prepare_intepolation_grids/README.md`](prepare_intepolation_grids/README.md)
- [`data/README.md`](data/README.md)
- [`key_tests/reports/pipeline_comparison.md`](key_tests/reports/pipeline_comparison.md)
