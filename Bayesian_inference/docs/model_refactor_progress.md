# Model Refactor Progress

## Final Target

The inference architecture should let a scientist add a new model without
understanding the sampler, JAX compilation, output writer, or CMASS-specific
data plumbing.  In the final form, a model file should only define:

- sampled parameters and their public config names;
- model-specific data requirements;
- latent-population, selection, per-lens likelihood, and optional prior formulas;
- model-owned diagnostics.

Framework code should own config validation, schema construction, HDF5 field
loading, context packing, JAX `jit`/`vmap`, selection-normalization reduction,
NumPyro/NUTS integration, and output metadata.

## Current State After Canonical Input Closure

The codebase now uses the inference-side canonical dataset as the only
production data entry, roughly 92% of the final target.  The required YAML data
entry is:

```yaml
data:
  inference_dataset_path: /path/to/inference_dataset.hdf5
```

The old observation/cross-section/sigma-table paths are no longer accepted by
`load_runtime_config()`.  They remain available only to data-preparation code
and explicitly constructed legacy oracle tests.

The public model entry remains:

```yaml
unit_convention: h_units_v1
model:
  name: cmass
```

Implemented boundaries:

- `ModelSpec` and `ParameterSpec` describe the human-authored scientific model.
- `DataSpec` describes how a validated NumPy source context is packed into a
  JAX context, including traced arrays, ordered scalar packing, static JIT
  flags, normalization samples, and normalization floor.
- `ModelRuntimeAdapter` now only provides source-context construction plus the
  model's `DataSpec`; it no longer hand-writes JAX packing functions.
- `jax_backend/model_adapter.py` converts `ModelSpec + ModelRuntimeAdapter` into
  the low-level `ModelDefinition` consumed by the existing JAX backend.
- `jax_backend/context_builder.py` owns generic JAX-context construction,
  static-flag extraction, normalization-field access, and `CompiledModel`
  construction.
- `models/cmass.py` now focuses on CMASS parameters and scientific hooks.
- `models/cmass_runtime.py` owns only canonical CMASS source-context
  construction and the declarative `DataSpec`; it no longer falls back to raw
  observation/cross-section inputs.
- `models/cmass_context.py` contains both the NumPy source context and JAX
  context type for CMASS, making the context boundary explicit.
- `canonical_dataset.py` reads and validates the canonical HDF5 blocks,
  capabilities, unit metadata, and cross-block shape constraints.
- `cmass_runtime.py` can now build `CMASSModelContext` directly from
  `data.inference_dataset_path`.
- Runner metadata records `input_inference_dataset_path` and canonical
  capabilities instead of raw observation or sigma-table paths.
- CMASS selection and likelihood hooks now consume the canonical two-dimensional
  `theta_E x gamma` cross-section interface.  The legacy separable
  cross-section path is converted into the same representation for migration
  testing.

The old public surfaces remain intentionally unsupported:

- `model.name: cmass_current`
- `model.components`
- top-level `gamma_model`
- top-level `mass_definition`
- `data.observation_path`
- `data.cross_section_path`
- `data.sigma_table_path`
- `unit_convention: legacy_fixed_kpc` for the default CMASS model

## Remaining Work

The next milestones are:

1. Move reusable scientific pieces, such as FP prior summaries and common
   selection functions, into component modules.
2. Thin `models/cmass_runtime.py` further by moving deterministic canonical
   preprocessing into reusable backend/data-preparation helpers.
3. Add population-sigma-unit runtime tests when that canonical capability is
   needed by an implemented model.
4. Implement the real `sonnenfeld2024_slacs` model using the same interface.
5. Run scientific equivalence, performance, and short-chain NumPyro validation
   for CMASS and Sonnenfeld.

## Current Refactor Step: Canonical Input Closure

This step is the planned bridge before thinning `models/cmass_runtime.py`
further.  The goal is to make the production inference entry accept exactly one
prepared canonical dataset:

```yaml
data:
  inference_dataset_path: /path/to/inference_dataset.hdf5
```

The raw observation/cross-section/sigma-table paths remain useful for data
preparation and legacy numerical oracles, but they should no longer be accepted
by `load_runtime_config()` for production inference.  That separation matters
because the final target assumes inference starts from one stable canonical
schema instead of each model knowing how to normalize historical raw products.

Implemented in this step:

- added repeatable `real_data`/`slow` tests for the existing devauc canonical
  datasets under `data/external`;
- proved the canonical devauc path matches the legacy raw-input oracle for
  log-probability and selection normalization;
- proved canonical FP-within-Re diagnostics remain finite for the devauc slit
  and BOSS datasets;
- rejected `data.observation_path`, `data.cross_section_path`, and
  `data.sigma_table_path` in production configs;
- updated runner metadata to record `inference_dataset_path` and canonical
  capabilities rather than raw input paths.

Known input coverage:

- real canonical datasets currently exist for `devauc + m5_hinvkpc +
  h_units_v1`;
- sersic remains covered by synthetic canonical tests until a real sersic
  canonical dataset is prepared.

## Verification Record

```bash
conda run -n cmass_lens python -m pytest Bayesian_inference/tests -q
```

Latest result after the canonical reader v1 milestone:

```text
........................................................................ [ 86%]
...........                                                              [100%]
```

Additional focused checks after the DataSpec v1 milestone:

```bash
conda run -n cmass_lens python -m pytest Bayesian_inference/tests/test_model_registry_config.py -q
conda run -n cmass_lens python -m pytest Bayesian_inference/tests/test_jax_numpyro_inference.py -q
```

Results:

```text
............                                                             [100%]
......                                                                   [100%]
```

Additional focused checks after the canonical reader v1 milestone:

```bash
conda run -n cmass_lens python -m pytest Bayesian_inference/tests/test_canonical_dataset.py Bayesian_inference/tests/test_model_registry_config.py Bayesian_inference/tests/test_jax_numpyro_inference.py -q
```

Result:

```text
.......................                                                  [100%]
```

Additional checks after the canonical input-closure milestone:

```bash
conda run -n cmass_lens python -m pytest Bayesian_inference/tests/test_canonical_dataset.py Bayesian_inference/tests/test_model_registry_config.py Bayesian_inference/tests/test_jax_numpyro_inference.py -q
conda run -n cmass_lens python -m pytest Bayesian_inference/tests -q
conda run -n cmass_lens python -m pytest Bayesian_inference/tests -q -m real_data
```

Results:

```text
........................                                                 [100%]
........................................................................ [ 82%]
...............                                                          [100%]
...                                                                      [100%]
```
