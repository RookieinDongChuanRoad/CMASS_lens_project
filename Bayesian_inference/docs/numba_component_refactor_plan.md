# Numba Component Refactor Plan

## Final Target

The production inference architecture should keep `numba/emcee` as the only
production backend while making future hierarchical models easier to add,
inspect, and validate.

The target production flow is:

```text
canonical inference dataset
  -> model assembly
  -> runtime context
  -> model-owned production adapter
  -> shared Numba kernels
  -> backend-owned execution, diagnostics, and emcee bridge
  -> emcee HDFBackend chain.h5 outputs
```

The final architecture should enforce these ownership rules:

1. `components/` owns reusable scientific component declarations.
2. `ComponentSpec` owns sampled parameter names, public names, bounds, block
   order, required context fields, and required capabilities for each component.
3. `models/<model>/assembly.py` chooses components, orders component parameter
   blocks, aggregates capabilities, declares unit/mass contracts, and builds the
   final `ModelSpec`.
4. `models/<model>/runtime.py` builds parameter-independent context from the
   canonical dataset.
5. `models/<model>/production.py` owns the explicit posterior assembly for that
   model.
6. `models/<model>/constants.py` or `paper_constants.py` may store fixed
   scientific constants only; sampled parameters should not be duplicated there.
7. `numba_backend/kernels/` owns reusable compiled numerical kernels.
8. `numba_backend/` owns only backend execution mechanics: compiled-model
   factory, common diagnostics, box-prior rejection, dispatch, and sampler bridge.
9. `runner.py`, `emcee_sampler.py`, `outputs.py`, and `posterior_corner.py` must
   stay model-agnostic.

In final form, adding a production model should usually require changes only in:

```text
components/                 # only when a new reusable component is needed
models/<new_model>/
model_registry.py
tests/
```

It should not require changes to the runner, sampler, output writer, or posterior
reader.

## Implementation Phases

### Phase 1: Move Model-Specific Posterior Assembly Into `models/`

Goal:

```text
Make the backend engine generic and move per-model posterior wiring to the model
package.
```

Work:

1. Create `models/cmass/production.py`.
2. Create `models/sonnenfeld2024_slacs/production.py`.
3. Move CMASS log-prob posterior assembly out of `numba_backend/likelihood_engine.py`.
4. Move Sonnenfeld log-prob posterior assembly out of `numba_backend/likelihood_engine.py`.
5. Keep shared timing, box-prior rejection, diagnostic blob creation, and dispatch
   in the backend.
6. Rename or conceptually narrow `numba_backend/model_adapter.py` into a
   backend-owned compiled-model factory.

Acceptance:

1. `likelihood_engine.py` no longer contains large CMASS or Sonnenfeld posterior
   bodies.
2. CMASS and Sonnenfeld production log-prob values remain unchanged.
3. Existing short emcee tests still write readable `chain.h5`.

Status: complete.

Completed:

1. Converted `models/cmass.py` into `models/cmass/assembly.py`.
2. Converted `models/cmass_runtime.py` into `models/cmass/runtime.py`.
3. Converted `models/sonnenfeld2024_slacs.py` into
   `models/sonnenfeld2024_slacs/assembly.py`.
4. Converted `models/sonnenfeld2024_slacs_runtime.py` into
   `models/sonnenfeld2024_slacs/runtime.py`.
5. Added `models/cmass/production.py` for CMASS posterior assembly.
6. Added `models/sonnenfeld2024_slacs/production.py` for Sonnenfeld posterior
   assembly.
7. Added `numba_backend/diagnostics.py` for backend-owned emcee blob dtype and
   timing/rejection blob construction.
8. Renamed `numba_backend/model_adapter.py` to
   `numba_backend/compiled_model_factory.py`.
9. Updated `likelihood_engine.py` so it performs host-side bounds rejection and
   dispatches to model-owned production adapters.

Verification:

```bash
conda run -n cmass_lens python -m py_compile \
  Bayesian_inference/src/cmass_lens_inference/model_registry.py \
  Bayesian_inference/src/cmass_lens_inference/numba_backend/compiled_model_factory.py \
  Bayesian_inference/src/cmass_lens_inference/numba_backend/diagnostics.py \
  Bayesian_inference/src/cmass_lens_inference/numba_backend/likelihood_engine.py

conda run -n cmass_lens python -m pytest \
  Bayesian_inference/tests/test_cmass_component_boundary.py \
  Bayesian_inference/tests/test_model_registry_config.py \
  Bayesian_inference/tests/test_sonnenfeld_capability_audit.py \
  Bayesian_inference/tests/test_numba_emcee_inference.py -q
```

Observed result:

```text
...............................                                          [100%]
```

Remaining if following the original plan:

1. Introduce `ComponentSpec`.
2. Move parameter/capability ownership from model-level parameter/capability
   files into component declarations and assembly aggregation.

### Phase 2: Introduce `ComponentSpec` As the Parameter and Capability Source

Goal:

```text
Make component declarations the single source of sampled parameter blocks and
component-level capabilities.
```

Work:

1. Add `components/interfaces.py` with a minimal `ComponentSpec`.
2. Represent each component's sampled parameters through `ComponentSpec`.
3. Represent each component's required capabilities through `ComponentSpec`.
4. Update model assembly code to aggregate component parameter blocks.
5. Update model assembly code to aggregate component capabilities.
6. Keep model-specific constants in `constants.py` or `paper_constants.py`.
7. Avoid creating new model-level `parameters.py` or `capabilities.py` files as
   second sources of truth.

Acceptance:

1. Parameter order has one auditable source.
2. Required capabilities are derived from selected components plus explicit
   model-level additions.
3. Existing config parameter names and bounds remain backward compatible unless
   deliberately changed.

Status: complete.

Completed:

1. Added top-level `components/` with `ComponentSpec` and aggregation helpers.
2. Added `components/cmass.py` as the CMASS component parameter/capability source.
3. Added `components/sonnenfeld2024_slacs.py` as the Sonnenfeld component
   parameter/capability source and capability audit source.
4. Updated CMASS and Sonnenfeld assembly files to aggregate parameters and
   capabilities from selected components.
5. Converted old `models/components/*/parameters.py` and
   `models/components/sonnenfeld2024_slacs/capabilities.py` into compatibility
   re-export modules.
6. Updated Sonnenfeld preprocessing to use the top-level component declaration.
7. Added `tests/test_component_specs.py` to lock down duplicate-parameter
   rejection, capability aggregation, and assembly/component ownership.

Verification:

```bash
conda run -n cmass_lens python -m py_compile \
  Bayesian_inference/src/cmass_lens_inference/components/interfaces.py \
  Bayesian_inference/src/cmass_lens_inference/components/cmass.py \
  Bayesian_inference/src/cmass_lens_inference/components/sonnenfeld2024_slacs.py \
  Bayesian_inference/src/cmass_lens_inference/models/components/sonnenfeld2024_slacs/preprocessing.py \
  Bayesian_inference/tests/test_component_specs.py

conda run -n cmass_lens python -m pytest \
  Bayesian_inference/tests/test_component_specs.py \
  Bayesian_inference/tests/test_cmass_component_boundary.py \
  Bayesian_inference/tests/test_model_registry_config.py \
  Bayesian_inference/tests/test_sonnenfeld_capability_audit.py \
  Bayesian_inference/tests/test_sonnenfeld_runtime_model.py -q
```

Observed result:

```text
.......................................                                  [100%]
```

Remaining if following the original plan:

1. Split shared low-level Numba primitives by numerical responsibility.
2. Keep compatibility re-export modules only as long as downstream code still
   imports old component paths.

### Phase 3: Split Low-Level Numba Primitives by Numerical Responsibility

Goal:

```text
Turn the current shared primitive file into a small Numba kernel library with
clear numerical responsibilities.
```

Work:

1. Create `numba_backend/kernels/distributions.py`.
2. Create `numba_backend/kernels/interpolation.py`.
3. Create `numba_backend/kernels/lensing.py`.
4. Create `numba_backend/kernels/selection.py`.
5. Create `numba_backend/kernels/integration.py`.
6. Move existing primitive functions without changing behavior.
7. Update imports in CMASS and Sonnenfeld kernels.

Acceptance:

1. Full test suite remains unchanged.
2. `git diff --check` passes.
3. Numba steady-state log-prob benchmark does not show material regression.

Status: structural split complete; final benchmark verification deferred to
Phase 7.

Completed:

1. Added `numba_backend/kernels/constants.py`.
2. Added `numba_backend/kernels/distributions.py`.
3. Added `numba_backend/kernels/interpolation.py`.
4. Added `numba_backend/kernels/integration.py`.
5. Added `numba_backend/kernels/lensing.py`.
6. Added `numba_backend/kernels/selection.py`.
7. Added `numba_backend/kernels/population.py` for current CMASS population
   helper kernels that are not yet promoted into full scientific components.
8. Replaced `numba_backend/primitives.py` with a compatibility re-export module.
9. Updated CMASS and Sonnenfeld kernels to import directly from
   `numba_backend/kernels/`.

Verification:

```bash
conda run -n cmass_lens python -m py_compile \
  $(conda run -n cmass_lens rg --files Bayesian_inference/src/cmass_lens_inference/numba_backend -g '*.py')

conda run -n cmass_lens python -m pytest \
  Bayesian_inference/tests/test_math_core.py \
  Bayesian_inference/tests/test_component_specs.py \
  Bayesian_inference/tests/test_numba_emcee_inference.py \
  Bayesian_inference/tests/test_sonnenfeld_runtime_model.py -q

conda run -n cmass_lens git diff --check
```

Observed targeted test result:

```text
.............................                                            [100%]
```

Remaining if following the original plan:

1. Run the fixed production benchmark in Phase 7 and compare against the latest
   recorded steady-state expectation.
2. Extract real shared mid-level scientific components.

### Phase 4: Extract Shared Mid-Level Scientific Components

Goal:

```text
Promote real shared science from model-specific kernels into reusable components
only where reuse already exists or is immediately needed.
```

Candidate components:

1. Power-law lensing geometry.
2. `theta_E x gamma` cross-section selection.
3. Sigma-unit interpolation.
4. Velocity-dispersion likelihood.
5. Source-redshift density.
6. Truncated proposal correction.
7. Fundamental Plane sufficient statistics.

Work:

1. Extract one component at a time.
2. Add focused reference tests for each extracted component.
3. Update CMASS and Sonnenfeld production paths to call shared component kernels.
4. Avoid abstracting formulas used by only one model unless they clarify a model
   boundary.

Acceptance:

1. Each shared component has an isolated numerical test.
2. CMASS real-data equivalence remains unchanged.
3. Sonnenfeld synthetic log-prob remains finite and stable.

Status: complete.

Completed:

1. Added `numba_backend/kernels/selection_likelihood.py` for shared
   cross-section/detection weighting, source-redshift density wrappers,
   sigma-model conversion, and observed velocity-dispersion likelihood factors.
2. Added `numba_backend/kernels/fundamental_plane.py` for FP OLS summary schema
   constants and sufficient-statistic accumulation.
3. Updated CMASS production kernels to call the shared selection/source-z/sigma
   likelihood helpers and FP summary reducer.
4. Updated Sonnenfeld production kernels to call the shared selection/source-z
   and observed-sigma helpers while keeping the Sonnenfeld-specific
   velocity-proxy `theta_E` estimate in its own model kernel.
5. Updated CMASS production posterior code to import FP summary schema from the
   shared FP kernel module.
6. Added `tests/test_shared_kernel_components.py` with isolated tests for the
   extracted shared components.

Verification:

```bash
conda run -n cmass_lens python -m py_compile \
  Bayesian_inference/src/cmass_lens_inference/numba_backend/kernels/selection_likelihood.py \
  Bayesian_inference/src/cmass_lens_inference/numba_backend/kernels/fundamental_plane.py \
  Bayesian_inference/src/cmass_lens_inference/numba_backend/cmass_kernels.py \
  Bayesian_inference/src/cmass_lens_inference/numba_backend/sonnenfeld_kernels.py \
  Bayesian_inference/src/cmass_lens_inference/models/cmass/production.py \
  Bayesian_inference/tests/test_shared_kernel_components.py

conda run -n cmass_lens python -m pytest \
  Bayesian_inference/tests/test_shared_kernel_components.py \
  Bayesian_inference/tests/test_numba_emcee_inference.py \
  Bayesian_inference/tests/test_sonnenfeld_runtime_model.py \
  Bayesian_inference/tests/test_component_specs.py -q
```

Observed targeted test result:

```text
.............................                                            [100%]
```

Remaining if following the original plan:

1. Make CMASS and Sonnenfeld assembly files more explicitly read as component
   selections rather than just parameter/capability aggregators.
2. Prove the extension boundary with a toy production model.

### Phase 5: Rewrite CMASS and Sonnenfeld Assembly Around Components

Goal:

```text
Make the two existing production models serve as examples for future
component-driven model assembly.
```

Work:

1. Convert CMASS assembly to select component specs for population, lensing,
   selection, likelihood, and optional FP prior behavior.
2. Convert Sonnenfeld assembly to select component specs for parent density,
   size relation, lensing, velocity-proxy selection, finite-fibre selection, and
   likelihood behavior.
3. Make each assembly file aggregate parameter order and capabilities from its
   selected components.
4. Keep model-specific constants separate from sampled parameter declarations.

Acceptance:

1. Reading `models/<model>/assembly.py` explains what scientific components the
   model uses.
2. Adding or removing a component changes parameter/capability contracts through
   assembly, not through duplicate model-level files.
3. Existing CMASS and Sonnenfeld production tests still pass.

Status: complete.

Completed:

1. Split CMASS declarations into lens-observation, enclosed-mass population,
   gamma population, source-redshift, theta-gamma selection, observed-sigma
   likelihood, and optional FP prior components.
2. Split Sonnenfeld declarations into Table-1 parent density, quadratic size
   relation, enclosed-mass population, gamma population, source-redshift,
   finite-fibre selection, and velocity-proxy likelihood components.
3. Kept `CMASS_PRODUCTION_COMPONENT` and `SONNENFELD_PRODUCTION_COMPONENT` as
   compatibility envelopes for transitional imports and stable parameter tuple
   identity.
4. Updated `models/cmass/assembly.py` and
   `models/sonnenfeld2024_slacs/assembly.py` to select component tuples and
   aggregate parameter/capability contracts from those selections.
5. Added `component_assembly` metadata strings so run metadata exposes the
   intended component order without requiring readers to inspect Python source.
6. Updated component tests to assert assembly derives its contracts from the new
   component tuples.

Verification:

```bash
conda run -n cmass_lens python -m py_compile \
  Bayesian_inference/src/cmass_lens_inference/components/cmass.py \
  Bayesian_inference/src/cmass_lens_inference/components/sonnenfeld2024_slacs.py \
  Bayesian_inference/src/cmass_lens_inference/models/cmass/assembly.py \
  Bayesian_inference/src/cmass_lens_inference/models/sonnenfeld2024_slacs/assembly.py \
  Bayesian_inference/tests/test_component_specs.py

conda run -n cmass_lens python -m pytest \
  Bayesian_inference/tests/test_component_specs.py \
  Bayesian_inference/tests/test_model_registry_config.py \
  Bayesian_inference/tests/test_cmass_component_boundary.py \
  Bayesian_inference/tests/test_sonnenfeld_capability_audit.py -q

conda run -n cmass_lens python -m pytest \
  Bayesian_inference/tests/test_numba_emcee_inference.py \
  Bayesian_inference/tests/test_sonnenfeld_runtime_model.py \
  Bayesian_inference/tests/test_shared_kernel_components.py -q
```

Observed targeted test results:

```text
...........................                                              [100%]
.........................                                                [100%]
```

Remaining if following the original plan:

1. Add a toy hierarchical model to prove the extension path.
2. Remove the remaining backend-side model-name dispatch so adding a model only
   requires registration plus model-owned files.

### Phase 6: Add a Toy Hierarchical Model as an Architecture Acceptance Test

Goal:

```text
Prove that the refactored boundary actually supports adding a new model without
touching framework code.
```

Work:

1. Add a small synthetic model that reuses existing components.
2. Add `models/toy_hierarchical/assembly.py`.
3. Add `models/toy_hierarchical/runtime.py`.
4. Add `models/toy_hierarchical/production.py`.
5. Register the model once in `model_registry.py`.
6. Add synthetic log-prob and short emcee smoke tests.

Acceptance:

1. The toy model does not require changes to `runner.py`, `emcee_sampler.py`,
   `outputs.py`, or `posterior_corner.py`.
2. The toy model can build context, evaluate finite log-prob, reject invalid
   theta, and write `chain.h5`.

Status: complete.

Completed:

1. Added `components/toy_hierarchical.py` with a two-parameter synthetic
   Gaussian-population component.
2. Added `models/toy_hierarchical/assembly.py`, `runtime.py`, and
   `production.py`.
3. Extended `ModelDefinition` with a model-owned `evaluate_log_prob` callable.
4. Updated `numba_backend/compiled_model_factory.py` to bind the production
   callable at registration time.
5. Updated `numba_backend/likelihood_engine.py` so it only performs host-side
   box-prior rejection and then calls `model_definition.evaluate_log_prob`.
6. Registered `toy_hierarchical` once in `model_registry.py`.
7. Added `tests/test_toy_hierarchical_extension.py` to verify registry binding,
   finite log-prob, out-of-bounds rejection, and short emcee `chain.h5` output.

Verification:

```bash
conda run -n cmass_lens python -m py_compile \
  Bayesian_inference/src/cmass_lens_inference/model_interfaces.py \
  Bayesian_inference/src/cmass_lens_inference/numba_backend/compiled_model_factory.py \
  Bayesian_inference/src/cmass_lens_inference/numba_backend/likelihood_engine.py \
  Bayesian_inference/src/cmass_lens_inference/model_registry.py \
  Bayesian_inference/src/cmass_lens_inference/components/toy_hierarchical.py \
  Bayesian_inference/src/cmass_lens_inference/models/toy_hierarchical/assembly.py \
  Bayesian_inference/src/cmass_lens_inference/models/toy_hierarchical/runtime.py \
  Bayesian_inference/src/cmass_lens_inference/models/toy_hierarchical/production.py \
  Bayesian_inference/tests/test_toy_hierarchical_extension.py

conda run -n cmass_lens python -m pytest \
  Bayesian_inference/tests/test_toy_hierarchical_extension.py -q

conda run -n cmass_lens python -m pytest \
  Bayesian_inference/tests/test_model_registry_config.py \
  Bayesian_inference/tests/test_numba_emcee_inference.py \
  Bayesian_inference/tests/test_sonnenfeld_runtime_model.py \
  Bayesian_inference/tests/test_component_specs.py -q
```

Observed targeted test results:

```text
...                                                                      [100%]
.....................................                                    [100%]
```

Remaining if following the original plan:

1. Run full-suite verification, diff checks, and benchmark checks.
2. Update `docs/model_refactor_progress.md`.

### Phase 7: Final Verification and Progress Update

Goal:

```text
Verify that the refactor changed ownership boundaries without changing production
science or sampler behavior.
```

Work:

1. Run the full test suite in the `cmass_lens` environment.
2. Run targeted CMASS real-data equivalence checks.
3. Run targeted Sonnenfeld synthetic/reference checks.
4. Run the production log-prob benchmark.
5. Run a short CMASS emcee smoke test.
6. Run a short Sonnenfeld emcee smoke test.
7. Update `docs/model_refactor_progress.md`.

Acceptance:

1. Full tests pass, except explicitly documented data-dependent skips.
2. CMASS real-data equivalence is unchanged.
3. Sonnenfeld synthetic/reference checks pass.
4. Benchmark shows no material steady-state regression.
5. `model_refactor_progress.md` records the completed component-refactor state.

Status: complete.

Completed:

1. Ran the full test suite in the `cmass_lens` environment.
2. Re-ran CMASS real devauc canonical-vs-legacy oracle verification against
   locally available 1ca5 canonical/raw files.
3. Re-ran benchmark timing with the fixed real devauc acceptance config.
4. Fixed `scripts/benchmark_log_prob.py` so direct script execution sets OpenMP
   defaults before importing Numba-backed project modules.
5. Re-ran full tests and `git diff --check` after the benchmark-script fix.
6. Updated `docs/model_refactor_progress.md` with the completed
   component-refactor state.

Verification:

```bash
conda run -n cmass_lens python -m pytest Bayesian_inference/tests -q

conda run -n cmass_lens bash -lc \
  'python -m py_compile $(rg --files Bayesian_inference/src/cmass_lens_inference -g "*.py")'

git diff --check

conda run -n cmass_lens python Bayesian_inference/scripts/benchmark_log_prob.py \
  --config /tmp/cmass_numba_devauc_acceptance.yaml \
  --repeats 3 \
  --output-dir /tmp/cmass_numba_component_refactor_benchmarks
```

Observed full test result:

```text
........................................................................ [ 55%]
...................sss....................................               [100%]
```

The three skips are the existing real-data tests whose b283-local `data/`
fixtures are absent.  The equivalent CMASS real devauc oracle check was run
manually against the locally available 1ca5 canonical/raw files:

```text
canonical_log_prob -129.39561603870516
legacy_log_prob -129.39561603870516
delta_log_prob 0.0
canonical_norm 0.48714999820661803
legacy_norm 0.48714999820661786
delta_norm 1.6653345369377348e-16
```

Observed benchmark result:

```text
benchmark_path /private/tmp/cmass_numba_component_refactor_benchmarks/20260507_222919_numba_benchmark.json
steady_log_prob_median_seconds 0.0005239590536803007
steady_log_prob_mean_seconds 0.0005660003516823053
steady_log_prob_value -129.39561603870513
```

Compared with the prior recorded median `0.00052370794583112 s`, this shows no
material steady-state regression.

## Completion Definition

This refactor is complete when:

1. Done: model-owned production adapters live under
   `models/<model>/production.py`.
2. Done: component specs own sampled parameter blocks and component
   capabilities.
3. Done: model assembly aggregates component parameters and capabilities.
4. Done: backend code no longer grows a new model-specific adapter file or
   dispatch branch for each new model; `ModelDefinition.evaluate_log_prob` is
   model-owned and registry-bound.
5. Done: shared Numba kernels are organized by numerical/scientific
   responsibility.
6. Done: `toy_hierarchical` demonstrates the intended extension path and writes
   an emcee `chain.h5`.
7. Done: existing CMASS and Sonnenfeld production behavior remains verified by
   full tests, targeted tests, real devauc oracle equivalence, and benchmark
   evidence above.
