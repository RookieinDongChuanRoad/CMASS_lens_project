# Model Refactor Progress

## Final Target

The production inference architecture should let a scientist add or inspect a
model without understanding sampler internals, output artifact conventions, or
CMASS-specific data plumbing.  The production backend for this worktree is now
intentionally:

```text
canonical inference dataset
  -> thin ModelSpec + ModelRuntimeAdapter
  -> model-specific Numba kernels
  -> backend-owned posterior reduction and diagnostics
  -> emcee sampler
  -> emcee HDFBackend chain.h5 outputs
```

In final form, a model assembly file should define only:

- sampled parameters and their public config names;
- model-specific data capability requirements;
- model metadata and unit/mass-coordinate contract;
- the backend kernel key that owns the numerical hot path.

Framework code owns config validation, schema construction, canonical HDF5
loading, runtime context dispatch, emcee sampling, output metadata,
checkpoints, diagnostics, and production backend execution.

JAX/NumPyro are no longer production dependencies or production run-path
requirements.  Their backend modules, NumPyro sampler, hook/oracle-only
component modules, and package extras have been physically removed from this
worktree.

## Current State After Numba/Emcee Production Migration

The production data entry remains canonical-only:

```yaml
data:
  inference_dataset_path: /path/to/inference_dataset.hdf5
```

The old observation/cross-section/sigma-table YAML paths are not accepted by
`load_runtime_config()`.  Raw files remain useful for data preparation and
explicitly constructed oracle tests, but they are not part of production
inference.

The public model entry can select these concrete production models:

```yaml
unit_convention: h_units_v1
model:
  name: cmass
```

```yaml
unit_convention: legacy_fixed_kpc
model:
  name: sonnenfeld2024_slacs
```

```yaml
unit_convention: h_units_v1
model:
  name: sonnenfeld2024_slacs_hunit
```

Implemented production boundaries:

- `ModelSpec` and `ParameterSpec` describe the human-authored scientific model
  surface.
- `ModelRuntimeAdapter` builds validated NumPy source contexts from canonical
  datasets and exposes a backend-neutral `DataSpec`.
- `DataSpec` describes context arrays, scalar packing order, static flags,
  normalization samples, and normalization floors without binding model
  declarations to a specific sampler.
- `model_registry.py` uses `numba_backend.model_adapter` as the production
  adapter for CMASS and both Sonnenfeld labels.
- `numba_backend/likelihood_engine.py` owns box-prior rejection, backend
  dispatch, posterior reduction, timing diagnostics, and the emcee blob dtype.
- `numba_backend/cmass_kernels.py` owns CMASS normalization, per-lens
  likelihood, FP summary, FP prior support, and canonical
  `theta_E x gamma` cross-section interpolation.
- `numba_backend/sonnenfeld_kernels.py` owns Sonnenfeld normalization,
  per-lens likelihood, velocity-dispersion proxy selection, finite-fibre
  cross-section interpolation, and parent/proposal correction.
- `emcee_sampler.py` owns walker initialization, HDFBackend setup, log-prob
  wrapping, optional process-pool execution, and acceptance summaries.
- `runner.py` calls only the Numba compiled model builder and emcee sampler on
  the production path.
- `outputs.py` writes production run layout, metadata, run summaries, and
  emcee checkpoints; it no longer writes NumPyro checkpoint or ArviZ artifacts.
- `posterior_corner.py` reads production `chain.h5`.
- `scripts/benchmark_log_prob.py` benchmarks the production Numba log-prob
  path and optionally runs an explicitly supplied short emcee config.
- `canonical_dataset.py` reads and validates canonical HDF5 blocks,
  capabilities, unit metadata, cross-block shapes, and velocity-dispersion
  capability-to-block consistency.
- `canonical_context.py` owns backend-neutral canonical metadata, gamma-axis
  handling, mass-grid interpolation, and sigma-grid normalization helpers.
- `models/cmass.py` and `models/sonnenfeld2024_slacs.py` are assembly layers;
  sampler/backend/output logic stays outside model files.
- `models/*_runtime.py` and `models/components/*/preprocessing.py` own
  model-specific deterministic context construction.
- `tests/test_no_jax_numpyro_backend.py` prevents the retired JAX/NumPyro stack
  from reappearing as production imports or packaging extras.

The following public surfaces remain intentionally unsupported:

- `model.name: cmass_current`
- `model.components`
- top-level `gamma_model`
- top-level `mass_definition`
- `data.observation_path`
- `data.cross_section_path`
- `data.sigma_table_path`
- `sampling.num_chains`
- `sampling.num_samples`
- `sampling.num_warmup`
- `sampling.chain_method`
- `sampling.thinning`
- `sampling.warmup`
- `unit_convention: legacy_fixed_kpc` for the default CMASS model

## Completed Migration Phases

### Phase A: Backend Skeleton

Completed:

- added `numba_backend/` and `emcee_sampler.py`;
- switched `runner.py` from the retired sampler path to
  `build_numba_compiled_model() + run_emcee_sampler()`;
- changed production sampling config to `n_walkers`, `n_steps`, and `burn_in`;
- rejected retired sampling field names at config-load time;
- switched production artifacts to `chain.h5` plus emcee checkpoint arrays;
- removed production NumPyro output helpers and ArviZ dependency.

### Phase B: CMASS Numba Production

Completed:

- implemented CMASS Numba kernels for normalization, likelihood, FP summary,
  FP prior diagnostics, and canonical cross-section interpolation;
- verified synthetic CMASS finite log-probability, bounds rejection, h-unit
  context values, FP-disabled path, FP-enabled path, and short emcee output;
- verified real devauc canonical slit log-probability against the legacy raw
  oracle with zero log-probability difference and normalization difference
  `1.1102230246251565e-16`;
- verified real devauc slit/BOSS FP diagnostics are finite;
- confirmed real devauc short emcee run writes `chain.h5` and metadata records
  `backend=numba_emcee`, `sampler=emcee`, and
  `chain_storage=emcee_hdf_backend`;
- replaced the benchmark script with a production Numba benchmark.

### Phase C: Sonnenfeld Numba Production

Completed:

- implemented Sonnenfeld Numba normalization and per-lens likelihood kernels;
- implemented the velocity-dispersion proxy selection path using
  `population_sigma_unit`, `theta_E_est`, finite-fibre cross-section lookup,
  parent/proposal correction, and source-redshift density;
- kept both `sonnenfeld2024_slacs` and `sonnenfeld2024_slacs_hunit` wired
  through the same production backend framework;
- verified synthetic paper-native and hunit Sonnenfeld log-probability paths;
- verified missing canonical capabilities fail at the model/data boundary;
- verified a short Sonnenfeld emcee run writes `chain.h5`.

### Phase D: Physical JAX/NumPyro Removal

Completed:

- deleted `src/cmass_lens_inference/jax_backend/`;
- deleted `src/cmass_lens_inference/numpyro_sampler.py`;
- deleted JAX hook/oracle-only component modules under
  `models/components/{common,cmass,sonnenfeld2024_slacs}/`;
- removed the `jax-oracle` optional dependency group from `pyproject.toml`;
- removed backward-compatible JAX naming aliases from `DataSpec`;
- deleted the obsolete source-tree Sonnenfeld JAX implementation note;
- added and verified a regression test that scans production source and package
  metadata for retired JAX/NumPyro imports or extras.

## Remaining Work

Backend migration blockers: none.

Future scientific validation work remains separate from the backend migration:

1. Run paper-level Sonnenfeld validation against a realistic SLACS canonical
   dataset and a trusted Sonnenfeld reference implementation.
2. Promote more utilities into `models/components/common/` only when a second
   implemented production model needs the same algebra.
3. Add long-chain production benchmarks and posterior-predictive validation for
   final scientific runs.

## Verification Record

All commands below were run from
`/Users/liurongfu/.codex/worktrees/b283/CMASS_lens_project` with the
`cmass_lens` conda environment.

Full test suite:

```bash
conda run -n cmass_lens python -m pytest Bayesian_inference/tests -q
```

Latest result:

```text
........................................................................ [ 61%]
...............sss............................                           [100%]
```

The three skipped tests are real-data tests whose hardcoded b283
`data/external` files are absent in this worktree.  Equivalent real-data checks
were run manually against locally available canonical HDF5 files under the 1ca5
worktree data directory.

Production import boundary:

```bash
conda run -n cmass_lens bash -lc 'python - <<'"'"'PY'"'"'
import sys
sys.path.insert(0, "Bayesian_inference/src")
from cmass_lens_inference.model_registry import get_model_definition
from cmass_lens_inference.runner import run_inference
print(get_model_definition("cmass").backend_kernel)
print(get_model_definition("sonnenfeld2024_slacs").backend_kernel)
print(get_model_definition("sonnenfeld2024_slacs_hunit").backend_kernel)
print("jax" in sys.modules, "numpyro" in sys.modules)
PY'
```

Observed result:

```text
cmass
sonnenfeld2024_slacs
sonnenfeld2024_slacs
False False
```

Physical JAX/NumPyro removal boundary:

```bash
conda run -n cmass_lens python -m pytest Bayesian_inference/tests/test_no_jax_numpyro_backend.py -q
```

Observed result:

```text
..                                                                       [100%]
```

Compilation and patch hygiene:

```bash
conda run -n cmass_lens python -m py_compile $(rg --files Bayesian_inference/src/cmass_lens_inference -g '*.py')
git diff --check
```

Observed result: both commands exited with code 0.

Real devauc canonical-vs-legacy oracle check:

```text
canonical_log_prob -129.39561603870516
legacy_log_prob -129.39561603870516
delta_log_prob 0.0
canonical_norm 0.4871499982066181
legacy_norm 0.487149998206618
delta_norm 1.1102230246251565e-16
```

Real devauc FP diagnostics:

```text
CMASS_REAL_DEVAUC_FP slit
log_prob -233.26842187564836
normalization 0.48714999820661803
fp_prior -103.8728058369432
fpfit_mu 2.3824739595552935
fpfit_beta 0.30186483077170184
fpfit_xi_is_nan True
fpfit_scatter 0.05598807504759378

CMASS_REAL_DEVAUC_FP boss
log_prob -290.39710660067954
normalization 0.487149998206618
fp_prior -103.8728058369432
fpfit_mu 2.3824739595552935
fpfit_beta 0.30186483077170184
fpfit_xi_is_nan True
fpfit_scatter 0.05598807504759378
```

Real devauc short emcee artifact check:

```text
run_dir /tmp/cmass_lens_numba_acceptance_outputs/devauc/20260507_130306_devauc_devauc-real-short-emcee
chain_exists True
metadata_backend numba_emcee
metadata_sampler emcee
metadata_chain_storage emcee_hdf_backend
status completed
completed_steps 2
acceptance_fraction_mean 0.7083333333333334
```

Benchmark command:

```bash
conda run -n cmass_lens python Bayesian_inference/scripts/benchmark_log_prob.py --config /tmp/cmass_numba_devauc_acceptance.yaml --repeats 3 --output-dir /tmp/cmass_numba_benchmarks
```

Benchmark result:

```text
benchmark_path /private/tmp/cmass_numba_benchmarks/20260507_130130_numba_benchmark.json
config_path /private/tmp/cmass_numba_devauc_acceptance.yaml
normalization_samples 1024
gamma_points 80
mstar_points 80
n_walkers 24
n_steps 2
first_call_seconds 0.1836464999942109
steady_log_prob_median_seconds 0.00052370794583112
steady_log_prob_mean_seconds 0.000525083354053398
steady_log_prob_value -129.39561603870513
```
