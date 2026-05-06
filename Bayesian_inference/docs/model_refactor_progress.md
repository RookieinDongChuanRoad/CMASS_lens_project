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

## Current State After Sonnenfeld Paper-Native / Hunit Split

The codebase now uses the inference-side canonical dataset as the only
production data entry, roughly 98% of the architectural target.  The required YAML data
entry is:

```yaml
data:
  inference_dataset_path: /path/to/inference_dataset.hdf5
```

The old observation/cross-section/sigma-table paths are no longer accepted by
`load_runtime_config()`.  They remain available only to data-preparation code
and explicitly constructed legacy oracle tests.

The public model entry can now select one of three concrete models.  CMASS
remains the default hunit production model:

```yaml
unit_convention: h_units_v1
model:
  name: cmass
```

The paper-native Sonnenfeld entry is now fixed-kpc / physical-mass `m5`,
matching the intended Sonnenfeld 2024 coordinate contract:

```yaml
unit_convention: legacy_fixed_kpc
model:
  name: sonnenfeld2024_slacs
```

The previous hunit runnable path remains available, but it is now explicit in
the model name so its results cannot be mistaken for paper-native fixed `m5`:

```yaml
unit_convention: h_units_v1
model:
  name: sonnenfeld2024_slacs_hunit
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
- `models/cmass.py` is now a pure assembly layer: it names the concrete model,
  records fixed metadata, and wires component hooks into `ModelSpec`.
- `models/components/cmass/` owns the CMASS scientific components:
  parameters, latent population draw, selection weight, per-lens likelihood,
  FP-prior summaries, context types, and deterministic preprocessing.
- `models/components/common/fp_prior.py` owns the shared hunit-aware 1D FP OLS
  sufficient statistics, optional prior value, and diagnostics convention.
- `models/components/sonnenfeld2024_slacs/` now owns the shared Sonnenfeld
  implementation used by both concrete Sonnenfeld entries: capability audit,
  sampled parameters, context types, deterministic preprocessing,
  latent-population draw, velocity-proxy selection, per-lens likelihood, and
  neutral diagnostics/prior hooks.
- `models/sonnenfeld2024_slacs.py` is now a pure assembly layer that exposes
  two explicit `ModelSpec` objects: `sonnenfeld2024_slacs` for paper-native
  fixed `m5`, and `sonnenfeld2024_slacs_hunit` for the hunit canonical-backend
  variant.
- `models/sonnenfeld2024_slacs_runtime.py` declares the Sonnenfeld `DataSpec`
  and builds source context from canonical input.
- `jax_backend/canonical_context.py` owns reusable canonical metadata, gamma
  axis, per-lens mass-grid interpolation, and sigma-grid normalization helpers.
- `models/components/cmass/preprocessing.py` owns CMASS deterministic
  preprocessing: hunit population pivots, stellar-mass quadrature arrays, fixed
  redshift weights, random-basis construction, FP-grid adaptation, and
  `CMASSModelContext` assembly.
- `models/cmass_runtime.py` is now a thin glue module: it checks the canonical
  input path, loads the CMASS canonical dataset, calls CMASS preprocessing, and
  exposes the declarative `DataSpec`.
- `models/components/cmass/context.py` contains both the NumPy source context
  and JAX context type for CMASS, making the context boundary explicit.
- `canonical_dataset.py` reads and validates the canonical HDF5 blocks,
  capabilities, unit metadata, cross-block shape constraints, and
  velocity-dispersion capability-to-block consistency.
- `canonical_dataset.py` now validates
  `velocity_dispersion_grids/population_sigma_unit` as the canonical data
  capability needed by Sonnenfeld's velocity-dispersion proxy selection path.
- `cmass_runtime.py` can now build `CMASSModelContext` directly from
  `data.inference_dataset_path`.
- Runner metadata records `input_inference_dataset_path` and canonical
  capabilities instead of raw observation or sigma-table paths.
- CMASS selection and likelihood hooks now consume the canonical two-dimensional
  `theta_E x gamma` cross-section interface.  The legacy separable
  cross-section path is converted into the same representation for migration
  testing.
- Sonnenfeld selection and likelihood hooks consume the canonical two-dimensional
  `theta_E x gamma` cross-section and use `population_sigma_unit` to construct
  the velocity-dispersion proxy `theta_E_est` for `P_find`.

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

1. Validate Sonnenfeld numerical semantics against the paper/reference
   implementation, especially the Table-1 parent population, selection proxy
   scatter convention, and finite-fibre cross-section interpolation.
2. Prepare or validate realistic canonical SLACS datasets for both relevant
   coordinate contracts: paper-native `legacy_fixed_kpc + m5` for direct
   paper comparison, and optional `h_units_v1 + m5_hinvkpc` for hunit backend
   continuity checks.
3. Run Sonnenfeld short-chain NumPyro smoke tests once a realistic canonical
   SLACS dataset with `population_sigma_unit` and finite-fibre cross-section is
   available in `data/external`.
4. Promote additional shared utilities from CMASS and Sonnenfeld components into
   `models/components/common/` only when a second implemented model needs the
   same algebra or diagnostics.
5. Run scientific equivalence, performance, and posterior-predictive validation
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

## Refactor Step: Canonical Preprocessing Split

This step moved preprocessing responsibilities out of `models/cmass_runtime.py`
without changing the CMASS numerical model.

Implemented in this step:

- added `jax_backend/canonical_context.py` for model-neutral canonical helpers;
- added `models/components/cmass/preprocessing.py` for CMASS-only deterministic
  context construction;
- reduced `models/cmass_runtime.py` to runtime glue plus `DataSpec`;
- added unit tests for canonical helper behavior and direct CMASS preprocessing;
- kept existing CMASS log-probability, FP prior, NumPyro, and real-data tests
  as behavior-preservation checks.

## Refactor Step: CMASS Componentization

This step made CMASS follow the target model-file pattern: the model file
assembles components, while the components own scientific formulas and
model-specific context definitions.  The backend interface, YAML entry,
canonical dataset contract, NumPyro output format, and CMASS numerical formulas
were intentionally left unchanged.

Implemented in this step:

- added `models/components/common/fp_prior.py` for shared 1D FP prior algebra;
- added `models/components/cmass/parameters.py` for the fixed 11D CMASS schema;
- added `models/components/cmass/population.py`, `selection.py`,
  `likelihood.py`, and `summaries.py` for CMASS scientific hooks;
- moved CMASS context and deterministic preprocessing into
  `models/components/cmass/context.py` and
  `models/components/cmass/preprocessing.py`;
- rewrote `models/cmass.py` so `get_model_spec()` only wires component hooks;
- added component-boundary tests proving `models/cmass.py` no longer defines
  formula hooks or CMASS context/draw types.

## Refactor Step: Sonnenfeld Capability Audit

This step starts the Sonnenfeld implementation path without enabling a runnable
likelihood.  The goal is to prevent accidental CMASS reuse under a Sonnenfeld
label while making the next data-preparation requirements testable.

Implemented in this step:

- added `models/components/sonnenfeld2024_slacs/capabilities.py` with the
  required canonical capability tuple:
  `lens_observations.v1`, `lensing_mass_grids.v1`,
  `lensing_cross_section.theta_gamma_grid.v1`,
  `velocity_dispersion.per_lens_s2.v1`, and
  `velocity_dispersion.population_sigma_unit.v1`;
- added a capability audit object that reports missing inputs and explicitly
  calls out `population_sigma_unit` as blocking `theta_E_est` construction;
- added `models/components/sonnenfeld2024_slacs/parameters.py` with the paper
  constants in physical stellar-mass coordinates:
  `mstar_pivot = 11.3`, `mbar = 11.06`, `alpha = -1.207`,
  `m_t` polynomial coefficients, and `sigma_t = 0.0007`;
- kept `models/sonnenfeld2024_slacs.py` disabled, but updated its
  `NotImplementedError` to point at the missing canonical capability boundary;
- added tests that prove the audit contract, parameter constants, and disabled
  registry boundary.

## Current Refactor Step: Population Sigma-Unit Canonical Validation

This step makes the inference-side canonical reader capable of expressing and
validating the data needed by Sonnenfeld's `theta_E_est` selection proxy.  It
still does not implement the Sonnenfeld likelihood or normalization formula.

Implemented in this step:

- added synthetic canonical reader tests that write
  `velocity_dispersion_grids/population_sigma_unit` with `gamma_axis`,
  `zd_axis`, `log_re_kpc_axis`, optional `n_axis`, and `s_unit_grid`;
- made declared velocity-dispersion capabilities fail fast when the matching
  HDF5 block is absent;
- validated sigma-unit axes as non-empty one-dimensional arrays;
- validated `population_sigma_unit.s_unit_grid` shape against
  `(N_gamma, N_zd, N_log_re, N_n)` when `n_axis` is present, and against the
  corresponding lower-rank shape when optional axes are absent;
- added a Sonnenfeld audit test that consumes a loaded
  `CanonicalInferenceDataset`, proving future runtime code can audit the real
  dataset object directly.

## Refactor Step: Sonnenfeld Runtime Model v1

This step implements the first runnable Sonnenfeld model while preserving the
same registry, `ModelSpec`, `DataSpec`, JAX backend, NumPyro sampler, and
canonical dataset boundaries used by CMASS.

Implemented in this step:

- added `models/components/sonnenfeld2024_slacs/context.py` with separate
  NumPy source and JAX pytree context types;
- added `models/components/sonnenfeld2024_slacs/preprocessing.py` to load the
  complete canonical dataset, normalize the `population_sigma_unit` grid,
  interpolate mass tracks, build stellar-mass quadrature arrays, and place the
  paper constants `mstar_pivot = 11.3` and `mbar = 11.06` into the active
  model coordinate;
- added `models/sonnenfeld2024_slacs_runtime.py` with a declarative `DataSpec`
  for generic JAX context packing;
- extended `models/components/sonnenfeld2024_slacs/parameters.py` with the 12D
  parameter unpack/validation hooks;
- added Sonnenfeld `population.py`, `selection.py`, `likelihood.py`, and
  `summaries.py` components;
- rewrote `models/sonnenfeld2024_slacs.py` as a pure assembly layer returning
  `ModelSpec`;
- enabled `model_registry.py` to build a real `sonnenfeld2024_slacs`
  `ModelDefinition`;
- added synthetic runtime tests proving config parsing, context construction,
  hunit mass-location shifts, registry dispatch, and finite JAX log-probability.

## Refactor Step: Sonnenfeld Paper-Native / Hunit Split

This step separates the paper-native Sonnenfeld name from the hunit-compatible
engineering variant.  The goal is to keep scientific interpretation explicit:
`sonnenfeld2024_slacs` now means fixed `M_2D(<5 kpc)` / physical stellar-mass
coordinates, while `sonnenfeld2024_slacs_hunit` means the hunit canonical
backend variant.

Implemented in this step:

- changed `models/sonnenfeld2024_slacs.py` so `get_model_spec()` returns the
  paper-native `legacy_fixed_kpc + m5` model;
- added `get_hunit_model_spec()` and registered
  `model.name: sonnenfeld2024_slacs_hunit` for the existing
  `h_units_v1 + m5_hinvkpc` runnable path;
- kept both variants on the same component package and runtime adapter, so
  formula changes remain centralized;
- updated Sonnenfeld preprocessing so Table-1 mass-location constants and the
  low-mass truncation threshold stay unshifted for the paper-native model and
  shift by `2 log10(h_ref)` only for the explicit hunit model;
- updated the size-relation intercept handling so the hunit context gets the
  hunit `log R_e` intercept while the paper-native context keeps the physical
  intercept;
- added runtime tests for both valid model/unit combinations and registry tests
  for the two Sonnenfeld labels.

Known scientific caveats:

- this is a runnable v1, not yet a paper-level reproduction;
- the Table-1 parent density and velocity-proxy scatter convention still need
  direct comparison against the Sonnenfeld reference implementation;
- real-data Sonnenfeld validation requires a canonical SLACS dataset with the
  complete finite-fibre cross-section and population sigma-unit products;
- the synthetic paper-native fixed `m5` test fixture rewrites metadata from a
  minimal hunit fixture only to exercise the code path.  It is not a scientific
  fixed-kpc SLACS dataset.

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

Additional checks after the canonical preprocessing-split milestone:

```bash
conda run -n cmass_lens python -m pytest Bayesian_inference/tests/test_canonical_context_helpers.py Bayesian_inference/tests/test_cmass_preprocessing.py -q
conda run -n cmass_lens python -m pytest Bayesian_inference/tests/test_canonical_context_helpers.py Bayesian_inference/tests/test_cmass_preprocessing.py Bayesian_inference/tests/test_canonical_dataset.py Bayesian_inference/tests/test_model_registry_config.py Bayesian_inference/tests/test_jax_numpyro_inference.py -q
conda run -n cmass_lens python -m pytest Bayesian_inference/tests -q
conda run -n cmass_lens python -m pytest Bayesian_inference/tests -q -m real_data
```

Results:

```text
....                                                                     [100%]
............................                                             [100%]
........................................................................ [ 79%]
...................                                                      [100%]
...                                                                      [100%]
```

Additional checks after the CMASS componentization milestone:

```bash
conda run -n cmass_lens python -m pytest Bayesian_inference/tests/test_cmass_component_boundary.py -q
conda run -n cmass_lens python -m pytest Bayesian_inference/tests/test_cmass_component_boundary.py Bayesian_inference/tests/test_model_registry_config.py Bayesian_inference/tests/test_cmass_preprocessing.py -q
conda run -n cmass_lens python -m pytest Bayesian_inference/tests/test_jax_numpyro_inference.py -q
conda run -n cmass_lens python -m pytest Bayesian_inference/tests -q
conda run -n cmass_lens python -m pytest Bayesian_inference/tests -q -m real_data
git diff --check
```

Results:

```text
....                                                                     [100%]
...................                                                      [100%]
.......                                                                  [100%]
........................................................................ [ 75%]
.......................                                                  [100%]
...                                                                      [100%]
git diff --check: passed
```

Additional checks after the Sonnenfeld capability-audit milestone:

```bash
conda run -n cmass_lens python -m pytest Bayesian_inference/tests/test_sonnenfeld_capability_audit.py -q
conda run -n cmass_lens python -m pytest Bayesian_inference/tests/test_sonnenfeld_capability_audit.py Bayesian_inference/tests/test_model_registry_config.py -q
conda run -n cmass_lens python -m pytest Bayesian_inference/tests -q
conda run -n cmass_lens python -m pytest Bayesian_inference/tests -q -m real_data
```

Results:

```text
.....                                                                    [100%]
...................                                                      [100%]
........................................................................ [ 72%]
............................                                             [100%]
...                                                                      [100%]
```

Additional checks after the population sigma-unit canonical-validation
milestone:

```bash
conda run -n cmass_lens python -m pytest Bayesian_inference/tests/test_canonical_dataset.py Bayesian_inference/tests/test_sonnenfeld_capability_audit.py -q
conda run -n cmass_lens python -m pytest Bayesian_inference/tests/test_canonical_dataset.py Bayesian_inference/tests/test_sonnenfeld_capability_audit.py Bayesian_inference/tests/test_jax_numpyro_inference.py -q
conda run -n cmass_lens python -m pytest Bayesian_inference/tests -q
conda run -n cmass_lens python -m pytest Bayesian_inference/tests -q -m real_data
```

Results:

```text
............                                                             [100%]
...................                                                      [100%]
........................................................................ [ 69%]
................................                                         [100%]
...                                                                      [100%]
```

Additional checks after the Sonnenfeld runtime-model v1 milestone:

```bash
conda run -n cmass_lens python -m pytest Bayesian_inference/tests/test_sonnenfeld_runtime_model.py -q
conda run -n cmass_lens python -m pytest Bayesian_inference/tests/test_sonnenfeld_capability_audit.py Bayesian_inference/tests/test_model_registry_config.py Bayesian_inference/tests/test_sonnenfeld_runtime_model.py -q
conda run -n cmass_lens python -m pytest Bayesian_inference/tests -q
conda run -n cmass_lens python -m pytest Bayesian_inference/tests -q -m real_data
conda run -n cmass_lens python -m compileall -q Bayesian_inference/src/cmass_lens_inference/models/components/sonnenfeld2024_slacs Bayesian_inference/src/cmass_lens_inference/models/sonnenfeld2024_slacs.py Bayesian_inference/src/cmass_lens_inference/models/sonnenfeld2024_slacs_runtime.py
git diff --check
```

Results:

```text
....                                                                     [100%]
........................                                                 [100%]
........................................................................ [ 66%]
....................................                                     [100%]
...                                                                      [100%]
compileall: passed
git diff --check: passed
```

Additional checks after the Sonnenfeld paper-native / hunit split:

```bash
conda run -n cmass_lens python -m pytest Bayesian_inference/tests/test_sonnenfeld_runtime_model.py -q
conda run -n cmass_lens python -m pytest Bayesian_inference/tests/test_model_registry_config.py Bayesian_inference/tests/test_sonnenfeld_capability_audit.py Bayesian_inference/tests/test_sonnenfeld_runtime_model.py -q
conda run -n cmass_lens python -m pytest Bayesian_inference/tests -q
conda run -n cmass_lens python -m pytest Bayesian_inference/tests -q -m real_data
conda run -n cmass_lens python -m compileall -q Bayesian_inference/src/cmass_lens_inference/models/components/sonnenfeld2024_slacs Bayesian_inference/src/cmass_lens_inference/models/sonnenfeld2024_slacs.py Bayesian_inference/src/cmass_lens_inference/models/sonnenfeld2024_slacs_runtime.py Bayesian_inference/src/cmass_lens_inference/model_registry.py
git diff --check
```

Results:

```text
......                                                                   [100%]
..........................                                               [100%]
........................................................................ [ 65%]
......................................                                   [100%]
...                                                                      [100%]
compileall: passed
git diff --check: passed
```
