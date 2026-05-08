# Component And Kernel Refactor Progress

Target document: `Bayesian_inference/docs/component_repository_refactor_target.md`

The final target, locked design, phase plan, and completion definition are
recorded in that target document.  The target document is frozen for this work.
This progress file records phase progress for the refactor.

## 2026-05-08 Start

Status:

- Refactor started from the locked target.
- Current worktree still contains model-named component modules and model-named
  backend kernel modules.
- Next step under the target plan is Phase 1: extend component interfaces with
  `KernelRef` and `ComponentSpec.required_kernels`.

Remaining work by the locked plan:

- Phase 1: extend component interfaces.
- Phase 2: rebuild `components/` as scientific modules.
- Phase 3: rebuild shared kernel library and remove model-named backend kernel
  modules.
- Phase 4: rename model production adapters to posterior modules.
- Phase 5: reassemble CMASS and Sonnenfeld under the locked boundary.
- Phase 6: run verification and record completion status.

## 2026-05-08 21:18 CST Phase 1 Complete

Status:

- Added `KernelRef` to `components/interfaces.py`.
- Added `ComponentSpec.required_kernels` as an audit-only declaration field.
- Exported `KernelRef` from `components`.
- Added a component-spec test proving kernel refs are recorded as declarations.

Verification:

- `conda run -n cmass_lens python -m pytest Bayesian_inference/tests/test_component_specs.py -q`
  passed.

Remaining work by the locked plan:

- Phase 2: rebuild `components/` as scientific modules.
- Phase 3: rebuild shared kernel library and remove model-named backend kernel
  modules.
- Phase 4: rename model production adapters to posterior modules.
- Phase 5: reassemble CMASS and Sonnenfeld under the locked boundary.
- Phase 6: run verification and record completion status.

## 2026-05-08 21:31 CST Phase 2 Complete

Status:

- Rebuilt `components/` as scientific modules organized by observations,
  population, lensing, and selection.
- Removed production-model-named component modules:
  `components/cmass.py`, `components/sonnenfeld2024_slacs.py`, and
  `components/toy_hierarchical.py`.
- Updated CMASS, Sonnenfeld, and toy assemblies so model-specific parameter
  names, public names, bounds, and component bundles are owned under
  `models/<model>/`.
- Moved model-specific context and preprocessing modules from
  `models/components/<model>/` into `models/<model>/`.
- Removed the historical `models/components` Python compatibility layer.

Verification:

- `conda run -n cmass_lens python -m py_compile ...` passed for component and
  touched model modules.
- `conda run -n cmass_lens python -m pytest Bayesian_inference/tests/test_component_specs.py Bayesian_inference/tests/test_cmass_component_boundary.py Bayesian_inference/tests/test_cmass_preprocessing.py Bayesian_inference/tests/test_sonnenfeld_capability_audit.py -q`
  passed.

Remaining work by the locked plan:

- Phase 3: rebuild shared kernel library and remove model-named backend kernel
  modules.
- Phase 4: rename model production adapters to posterior modules.
- Phase 5: reassemble CMASS and Sonnenfeld under the locked boundary.
- Phase 6: run verification and record completion status.

## 2026-05-08 21:40 CST Phase 3 Complete

Status:

- Moved model-specific fused Numba loops out of `numba_backend/` into
  `models/cmass/posterior_kernels.py` and
  `models/sonnenfeld2024_slacs/posterior_kernels.py`.
- Removed `numba_backend/cmass_kernels.py` and
  `numba_backend/sonnenfeld_kernels.py`.
- Removed CMASS-specific theta unpacking and gamma-mode helpers from shared
  `numba_backend/kernels/population.py`.
- Removed model-specific imports and wording from `numba_backend`.
- Added reusable shared kernels for generic population relations and
  velocity-proxy Einstein-radius estimates.
- Updated component `required_kernels` refs to point at real shared kernel
  functions, with a test that resolves those refs.

Verification:

- `conda run -n cmass_lens python -m py_compile ...` passed for touched backend
  and model kernel modules.
- `conda run -n cmass_lens python -m pytest Bayesian_inference/tests/test_component_specs.py Bayesian_inference/tests/test_shared_kernel_components.py Bayesian_inference/tests/test_cmass_component_boundary.py Bayesian_inference/tests/test_cmass_preprocessing.py Bayesian_inference/tests/test_sonnenfeld_capability_audit.py -q`
  passed.
- Static search for `cmass`, `sonnenfeld`, `CMASS`, or `Sonnenfeld` in
  `src/cmass_lens_inference/numba_backend` returned no matches.

Remaining work by the locked plan:

- Phase 4: rename model production adapters to posterior modules.
- Phase 5: reassemble CMASS and Sonnenfeld under the locked boundary.
- Phase 6: run verification and record completion status.

## 2026-05-08 21:46 CST Phase 4 Complete

Status:

- Renamed model-owned posterior adapter modules from `production.py` to
  `posterior.py` for CMASS, Sonnenfeld, and the toy model.
- Updated `model_registry.py` to bind assembly, runtime, and posterior modules.
- Removed stale references to production adapter modules in touched code and
  tests.

Verification:

- `find Bayesian_inference/src/cmass_lens_inference/models -name 'production.py' -print`
  returned no files.
- `conda run -n cmass_lens python -m py_compile ...` passed for the registry and
  renamed posterior modules.
- `conda run -n cmass_lens python -m pytest Bayesian_inference/tests/test_model_registry_config.py Bayesian_inference/tests/test_toy_hierarchical_extension.py Bayesian_inference/tests/test_numba_emcee_inference.py -q`
  passed.

Remaining work by the locked plan:

- Phase 5: reassemble CMASS and Sonnenfeld under the locked boundary.
- Phase 6: run verification and record completion status.

## 2026-05-08 21:50 CST Phase 5 Complete

Status:

- Confirmed CMASS and Sonnenfeld assemblies now select generic component specs
  and own model-specific parameter names, public names, bounds, metadata, and
  capability aggregation.
- Confirmed runtime/context/preprocessing live under concrete model packages.
- Confirmed posterior structure and model-specific fused loops live under
  `models/<model>/posterior.py` and `models/<model>/posterior_kernels.py`.
- Added static boundary tests for the locked split.

Verification:

- `conda run -n cmass_lens python -m pytest Bayesian_inference/tests/test_component_kernel_boundary.py -q`
  passed.
- `conda run -n cmass_lens python -m pytest Bayesian_inference/tests/test_component_kernel_boundary.py Bayesian_inference/tests/test_component_specs.py Bayesian_inference/tests/test_model_registry_config.py Bayesian_inference/tests/test_sonnenfeld_runtime_model.py -q`
  passed.

Remaining work by the locked plan:

- Phase 6: run final verification and record completion status.

## 2026-05-08 21:58 CST Phase 6 Complete

Status:

- Completion definition in the locked target has been checked.
- `components/` contains reusable scientific declarations and no production
  model names.
- `numba_backend/` contains backend mechanics plus reusable shared kernels and
  no production model names.
- CMASS/Sonnenfeld-specific parameter naming, constants, theta order,
  posterior reductions, and fused loops live under `models/<model>/`.
- `models/<model>/posterior.py` is now the model-owned posterior assembly
  layer.
- Runner, sampler, output writer, and posterior reader stayed model-agnostic.

Verification:

- `rg -n "cmass|sonnenfeld|CMASS|Sonnenfeld|SLACS" Bayesian_inference/src/cmass_lens_inference/components Bayesian_inference/src/cmass_lens_inference/numba_backend -g '*.py'`
  returned no matches.
- `find Bayesian_inference/src/cmass_lens_inference/models -name 'production.py' -print`
  returned no files.
- `find Bayesian_inference/src/cmass_lens_inference/models/components -type f -name '*.py' -print 2>/dev/null`
  returned no files.
- `git diff --check` passed.
- `conda run -n cmass_lens python -m pytest Bayesian_inference/tests -q -rs`
  passed with 3 skips for missing locally synced real-data products.
- A synthetic CMASS log-prob benchmark using
  `Bayesian_inference/scripts/benchmark_log_prob.py --repeats 2` passed:
  steady-state median `log_prob` time was `0.000535478990059346` seconds and
  the steady value was `-13.521043006908734`.

Remaining work by the locked plan:

- None for the locked component/kernel/model boundary.
- Real-data CMASS equivalence tests remain data-gated in this worktree because
  the required local HDF5 products are not present.

## 2026-05-08 22:09 CST Final Sanity Check

Status:

- Re-ran the final boundary and regression checks before closing this work item.
- The locked target document remains the source of truth for the final goal;
  this progress document records only implementation status.

Verification:

- `rg -n "cmass|sonnenfeld|CMASS|Sonnenfeld|SLACS" Bayesian_inference/src/cmass_lens_inference/components Bayesian_inference/src/cmass_lens_inference/numba_backend -g '*.py'`
  returned no matches.
- `find Bayesian_inference/src/cmass_lens_inference/models -name 'production.py' -print`
  returned no files.
- `git diff --check` passed.
- `conda run -n cmass_lens python -m pytest Bayesian_inference/tests -q -rs`
  passed with 3 skips for missing locally synced real-data products.

Remaining work by the locked plan:

- None for the locked component/kernel/model boundary.
- Real-data CMASS equivalence tests can be re-run after the missing local HDF5
  products are synced into this worktree.

## 2026-05-08 22:24 CST Posterior Single-File Cleanup

Status:

- Collapsed model-owned private fused kernels into each model's `posterior.py`.
- Removed `models/cmass/posterior_kernels.py`.
- Removed `models/sonnenfeld2024_slacs/posterior_kernels.py`.
- Added a static boundary test requiring model posteriors to be single-file
  assemblies without sibling `posterior_kernels.py` modules.

Verification:

- The new boundary test first failed while the two `posterior_kernels.py` files
  still existed.
- `conda run -n cmass_lens python -m py_compile Bayesian_inference/src/cmass_lens_inference/models/cmass/posterior.py Bayesian_inference/src/cmass_lens_inference/models/sonnenfeld2024_slacs/posterior.py Bayesian_inference/tests/test_component_kernel_boundary.py`
  passed after the merge.
- `conda run -n cmass_lens python -m pytest Bayesian_inference/tests/test_component_kernel_boundary.py::test_model_posteriors_are_single_file_assemblies -q`
  passed after the merge.
- `find Bayesian_inference/src/cmass_lens_inference/models -name 'posterior_kernels.py' -print`
  returned no files.
- `conda run -n cmass_lens python -m pytest Bayesian_inference/tests/test_component_kernel_boundary.py Bayesian_inference/tests/test_model_registry_config.py Bayesian_inference/tests/test_sonnenfeld_runtime_model.py Bayesian_inference/tests/test_numba_emcee_inference.py -q`
  passed.

Remaining work by the locked plan:

- None for the locked component/kernel/model boundary.
- A full test suite re-run is pending for this cleanup entry.

## 2026-05-08 22:31 CST Posterior Single-File Cleanup Verified

Status:

- Full test-suite verification for the posterior single-file cleanup is now
  complete.

Verification:

- `conda run -n cmass_lens python -m pytest Bayesian_inference/tests -q -rs`
  passed with 3 skips for missing locally synced real-data products.
- `git diff --check` passed.
- `find Bayesian_inference/src/cmass_lens_inference/models -name 'posterior_kernels.py' -print`
  returned no files.

Remaining work by the locked plan:

- None for the locked component/kernel/model boundary.
- Real-data CMASS equivalence tests can be re-run after the missing local HDF5
  products are synced into this worktree.
