# PPC Diagnostics Parallelism Repair Progress

Plan file: `docs/superpowers/plans/2026-05-17-ppc-diagnostics-parallelism-repair.md`

Rule: the plan file is treated as read-only during execution. This document records implementation progress, verification output, review status, and blockers.

Baseline:
- Workspace: `/Users/liurongfu/Work/CMASS_lens_project`
- Branch: `main`
- Baseline HEAD: `025452c10f55d5ed28e64e4e86a7da6cf4e306c0`
- Required environment: `cmass_lens`
- Pre-existing dirty files are present outside the PPC repair scope; they must not be reverted.

## Task Status

- [x] Task 1: Add Diagnostics Execution Contract
- [x] Task 2: Resolve Diagnostics Compute Width In Generic PPC
- [x] Task 3: Make CMASS The Explicit Adapter Pattern
- [x] Task 4: Replace Sonnenfeld Python Loops With Numba Diagnostics Kernel
- [x] Task 5: Correct Stale Metadata Expectations
- [x] Task 6: Real-Run Smoke Validation
- [x] Task 7: Medium Benchmark Before Full-Chain PPC
- [x] Task 8: Documentation Update

## Log

### 2026-05-17 Task 1 Start

- Started subagent-driven execution.
- Original plan remains unchanged.
- Next step: dispatch a fresh implementer subagent for Task 1.
- Implementer subagent: `019e3431-b870-7113-a7a3-8188e22f7868`.
- Implementer completed Task 1.
- Reported files changed:
  - `Posterior_predictive_test/src/lensing_posterior_predictive/interfaces.py`
  - `Posterior_predictive_test/src/lensing_posterior_predictive/adapters/cmass.py`
  - `Posterior_predictive_test/src/lensing_posterior_predictive/adapters/sonnenfeld.py`
  - `Posterior_predictive_test/tests/test_predictive_registry.py`
- Reported verification:
  - `python -m pytest Posterior_predictive_test/tests/test_predictive_registry.py::test_predictive_definition_diagnostics_hook_accepts_execution_context -q`
  - Result: `1 passed`
- Next step: dispatch Task 1 spec compliance reviewer.

### 2026-05-17 Task 1 Spec Review

- Spec reviewer subagent: `019e3435-cd8b-7561-af54-5ac9c6fecb90`.
- Result: issues found.
- Real issue to fix:
  - Making `execution` mandatory breaks the current `run_posterior_diagnostics()` adapter call before Task 2 wires the resolver.
  - Fix direction: make `execution` a keyword-only optional contract argument for Task 1 so current behavior stays intact; Task 2 will pass a concrete object.
- Non-issue clarified:
  - The plan file existed before Task 1.
  - This progress file is explicitly authorized by the user and is outside Task 1 implementation scope.

### 2026-05-17 Task 1 Fix

- Implementer updated adapter signatures so `execution` is keyword-only optional.
- Reported verification:
  - Contract test: `1 passed`
  - Temporary signature-bind check: all registered models bind both with and without `execution=`.
  - `git diff --check`: passed.
- Next step: spec re-review.

### 2026-05-17 Task 1 Spec Re-review

- Spec re-reviewer subagent: `019e343c-84a3-7643-8054-33e526026df4`.
- Result: Task 1 code contract is compliant.
- Reviewer still noted the untracked plan file; controller ruling:
  - The plan file was created before Task 1 and must remain unchanged per user instruction.
  - It is not counted as a Task 1 implementation issue.
- Next step: code quality review.

### 2026-05-17 Task 1 Code Quality Review

- Code quality reviewer subagent: `019e3440-7b2c-76e1-ab10-3260b01cfc02`.
- Result: with fixes.
- Important fixes requested:
  - Strengthen the contract test so `execution` must be keyword-only and optional.
  - Expand `DiagnosticsExecution` field semantics so artifact metadata is not ambiguous.
- Minor fix:
  - Remove duplicate local `inspect` import from the new test.

### 2026-05-17 Task 1 Quality Fix

- Implementer strengthened contract test and expanded `DiagnosticsExecution` field semantics.
- Reported verification:
  - `python -m pytest Posterior_predictive_test/tests/test_predictive_registry.py -q`
  - Result: `13 passed`
  - `git diff --check` for Task 1 files: passed.
- Next step: quality re-review.

### 2026-05-17 Task 1 Complete

- Code quality re-reviewer subagent: `019e3445-2b3d-7e30-84ea-6d2441896b07`.
- Result: ready to proceed.
- Task 1 accepted.
- Next step: dispatch Task 2 implementer.

### 2026-05-17 Task 2 Implementation

- Implementer subagent: `019e3446-d323-7070-9fc7-ea3a5d4771ed`.
- Reported files changed:
  - `Posterior_predictive_test/src/lensing_posterior_predictive/predictive.py`
  - `Posterior_predictive_test/tests/test_posterior_predictive.py`
- Reported verification:
  - `python -m pytest Posterior_predictive_test/tests/test_posterior_predictive.py::test_run_posterior_diagnostics_records_requested_compute_width -q`
  - Result: `1 passed`
  - `git diff --check`: passed.
- Next step: dispatch Task 2 spec compliance reviewer.

### 2026-05-17 Task 2 Spec Review

- Spec reviewer subagent: `019e344c-641c-71a3-afd6-44f4cfcc73fa`.
- Result: reviewer found cumulative-diff scope issues.
- Controller ruling:
  - Reported scope issues are Task 1 files that were already accepted before Task 2 started.
  - Since no Task 1 commit was created, whole-worktree diff is not a valid Task 2 scope boundary.
  - Reviewer confirmed Task 2 core behavior in `predictive.py` and `test_posterior_predictive.py` is basically compliant.
- Next step: run Task 2 spec re-review with Task 1 accepted changes treated as baseline.

### 2026-05-17 Task 2 Spec Re-review

- Spec re-reviewer subagent: `019e344f-9b08-70c0-9abe-66c1f28875a8`.
- Result: spec compliant.
- Next step: code quality review.

### 2026-05-17 Task 2 Code Quality Review

- Code quality reviewer subagent: `019e3451-9869-70a2-b907-da7c496ce081`.
- Result: with fixes.
- Important fixes requested:
  - Do not silently rewrite explicit `worker_processes <= 0`.
  - Add early failure for `n_posterior_draws_used == 0`.
- Minor improvements accepted:
  - Add focused tests for invalid worker count and zero posterior draw request.
  - Avoid shared-dict aliasing by copying `parallelism_payload` when embedding it in artifact payloads.

### 2026-05-17 Task 2 Quality Fix

- Implementer added validation for explicit non-positive worker counts and zero posterior draws.
- Implementer copied `parallelism_payload` when embedding metadata in artifact payloads.
- Reported verification:
  - Positive metadata test plus two negative tests: passed.
  - `git diff --check` for Task 2 files: passed.
- Next step: quality re-review.

### 2026-05-17 Task 2 Complete

- Code quality re-reviewer subagent: `019e345a-79e7-7d60-b58b-7ae2adc1aac2`.
- Result: ready to proceed.
- Task 2 accepted.
- Next step: dispatch Task 3 implementer.

### 2026-05-17 Task 3 Implementation

- Implementer subagent: `019e345d-7488-7a72-8fb7-8cb3d090069c`.
- Reported files changed:
  - `Posterior_predictive_test/src/lensing_posterior_predictive/adapters/cmass.py`
  - `Posterior_predictive_test/tests/test_posterior_predictive.py`
- Reported verification:
  - `test_cmass_diagnostics_uses_numba_kernel_threads`: passed.
  - `test_run_posterior_diagnostics_records_requested_compute_width` plus CMASS thread test: passed.
  - `git diff --check` for Task 3 files: passed.
- Next step: dispatch Task 3 spec compliance reviewer.

### 2026-05-17 Task 3 Spec Review

- Spec reviewer subagent: `019e3461-929e-7030-ba01-5a970c75a724`.
- Result: spec compliant.
- Next step: code quality review.

### 2026-05-17 Task 3 Code Quality Review

- Code quality reviewer subagent: `019e3463-e098-7d31-9d65-a7d70b29bbe1`.
- Result: with fixes.
- Important fix requested:
  - Avoid test pollution from real `apply_thread_limits()` calls in diagnostics integration tests.
- Minor fix requested:
  - Use strict monkeypatching for `cmass_adapter.apply_thread_limits` so missing imports are not hidden.

### 2026-05-17 Task 3 Quality Fix

- Implementer added a targeted no-op thread-limit helper for ordinary diagnostics tests.
- Dedicated CMASS handoff test now uses strict monkeypatching.
- Reported verification:
  - Normal diagnostics test, metadata test, and CMASS thread handoff test: passed.
  - `git diff --check` for Task 3 files: passed.
- Next step: quality re-review.

### 2026-05-17 Task 3 Complete

- Code quality re-reviewer subagent: `019e346a-ad07-7b21-8c4b-256020ab775d`.
- Result: ready to proceed.
- Task 3 accepted.
- Next step: dispatch Task 4 implementer.

### 2026-05-17 Task 4 Implementation

- Implementer subagent: `019e3470-1e3e-7123-bab1-7216a3781b4b`.
- Reported files changed:
  - `Posterior_predictive_test/src/lensing_posterior_predictive/adapters/sonnenfeld.py`
  - `Posterior_predictive_test/tests/test_predictive_registry.py`
- Reported verification:
  - New Sonnenfeld Numba schema test: passed.
  - `Posterior_predictive_test/tests/test_predictive_registry.py`: `14 passed`.
  - `git diff --check`: passed.
- Next step: dispatch Task 4 spec compliance reviewer.

### 2026-05-17 Task 4 Spec Review

- Spec reviewer subagent: `019e347e-d066-7520-8aa6-f8f389129250`.
- Result: spec compliant.
- Next step: code quality review.

### 2026-05-17 Task 4 Code Quality Review

- Code quality reviewer subagent: `019e3483-4492-71a2-8fa7-612c646e4615`.
- Result: with fixes.
- Important fixes requested:
  - Shape-only tests are not enough; add schema/order and parity coverage.
  - Add a small deterministic Python reference/golden fixture for core Sonnenfeld semantics.
  - Use strict monkeypatching for `sonnenfeld.apply_thread_limits`.
  - Preserve old NaN percentile semantics in Numba summary statistics.
- Minor issue:
  - Source-string test for old loop is brittle; prefer kernel-call/schema/parity assertions.

### 2026-05-17 Task 4 Quality Fix

- Implementer added strict monkeypatching, NaN propagation in Numba summary stats, exact schema assertions, kernel spy, and Python reference/parity coverage.
- Reported verification:
  - NaN summary helper test: passed.
  - Sonnenfeld Numba schema/parity test: passed.
  - `Posterior_predictive_test/tests/test_predictive_registry.py`: `15 passed`.
  - `git diff --check` for Task 4 files: passed.
- Next step: quality re-review.

### 2026-05-17 Task 4 Quality Re-review

- Code quality re-reviewer subagent: `019e348f-1b48-71b3-9713-06e2dd70c237`.
- Result: with fixes.
- Remaining fixes requested:
  - Extend reference parity to `trend_draws["sigma_ap"]`.
  - Extend reference parity to `gamma_vs_sigma_star_*` and `gamma_vs_delta_r_*` trends/counts/sums.
  - Use a non-degenerate fake context so scatter/source-redshift/Sersic-related branches are actually constrained.

### 2026-05-17 Task 4 Parity Fix

- Implementer made fake context non-degenerate and enabled Sersic branch.
- Reference parity now covers all latent keys, all trend quantities/categories, all gamma trend payloads, and all count/sum arrays.
- Reported verification:
  - Sonnenfeld schema/parity test: passed.
  - `Posterior_predictive_test/tests/test_predictive_registry.py`: `15 passed`.
  - `git diff --check` for Task 4 files: passed.
  - `rg "raising=False|ProcessPool|multiprocessing"` in relevant files: no matches.
- Next step: final quality re-review.

### 2026-05-17 Task 4 Complete

- Final code quality re-reviewer subagent: `019e3497-56a5-7292-8c34-6a6a270e908a`.
- Result: ready to proceed.
- Minor residual note: parity test does not cross the 16-draw chunk boundary; not a blocker for this task.
- Task 4 accepted.
- Next step: dispatch Task 5 implementer.

### 2026-05-17 Task 5 Implementation

- Implementer subagent: `019e349c-607f-78d2-8e65-a56a50c94702`.
- Reported file changed:
  - `Posterior_predictive_test/tests/test_posterior_predictive.py`
- Reported changes:
  - Added `_assert_kernel_only_parallelism_metadata()`.
  - Replaced stale naked legacy `worker_processes == 0` assertions in trend/CLI metadata tests with full `parallelism` semantics.
- Reported verification:
  - Related 4-test subset: passed.
  - 5-test subset: 4 passed, 1 failed with existing `KeyError: 'display_xlim'`.
  - Full file: 6 failures, including 5 legacy import failures and the same `display_xlim` failure.
  - Remaining `worker_processes == 0` matches only in full-context helper or Task 2/3 diagnostics assertions.
  - `git diff --check`: passed.
- Next step: dispatch Task 5 spec compliance reviewer.

### 2026-05-17 Task 5 Spec Review

- Spec reviewer subagent: `019e34a3-48d8-7820-aadb-99c818c91546`.
- Result: Task 5 assertion cleanup is compliant.
- Reviewer noted cumulative production diffs from Tasks 1-4; controller ruling:
  - These are accepted prior-task changes, not Task 5 scope violations.
  - Task 5 itself only changed `test_posterior_predictive.py`.
- Reviewer noted `display_xlim` failure does not appear introduced by Task 5.
- Next step: code quality review.

### 2026-05-17 Task 5 Complete

- Code quality reviewer subagent: `019e34a6-634d-7f52-99de-0ff79143ca59`.
- Result: ready to proceed.
- Minor residual note: helper only checks `kernel_threads_per_process >= 1`, not exact width; not a blocker for this task.
- Task 5 accepted.
- Next step: Task 6 real-run smoke validation.

### 2026-05-17 Task 6 Smoke Validation

- First CMASS candidate failed because it was an old 2026-04-29 config with removed top-level `mass_definition` / `gamma_model`; no code changed.
- CMASS smoke rerun used:
  - Run: `outputs/devauc/20260512_204813_devauc_cmass-no-fp-standard-devauc`
  - Sigma table: `data/external/hunits_v1/jeans_deV_sigma_bundle.h5`
  - Output: `/tmp/ppc_numba_parallel_smoke/cmass/devauc/20260512_204813_devauc_cmass-no-fp-standard-devauc/ppc`
  - Result: completed.
- Sonnenfeld smoke used:
  - Run: `outputs/devauc/20260511_130710_devauc_sonnenfeld2024-slacs-sigma-star-gamma-hunit`
  - Output: `/tmp/ppc_numba_parallel_smoke/sonnenfeld/devauc/20260511_130710_devauc_sonnenfeld2024-slacs-sigma-star-gamma-hunit/ppc`
  - Result: completed.
- Artifact verification:
  - CMASS: `backend=numba_shared_parent`, `n_posterior_draws_used=4`, `parallelism.strategy=kernel_only`, `requested_worker_processes=2`, `worker_processes=0`, `kernel_threads_per_process=2`.
  - Sonnenfeld: `backend=numba_sonnenfeld_parent`, `n_posterior_draws_used=4`, `parallelism.strategy=kernel_only`, `requested_worker_processes=2`, `worker_processes=0`, `kernel_threads_per_process=2`.
- Task 6 accepted.
- Next step: Task 7 medium benchmark.

### 2026-05-17 Task 7 Medium Benchmark

- Benchmark target:
  - Run: `outputs/devauc/20260511_130710_devauc_sonnenfeld2024-slacs-sigma-star-gamma-hunit`
  - Draws: `64`
  - Parent sample size: `512`
  - Burn-in: `auto`
- Thread-1 command completed:
  - Output: `/tmp/ppc_numba_parallel_benchmark/thread1/devauc/20260511_130710_devauc_sonnenfeld2024-slacs-sigma-star-gamma-hunit/ppc`
  - Wall time: `real 8.19`
  - Metadata: `strategy=kernel_only`, `requested_worker_processes=1`, `worker_processes=0`, `kernel_threads_per_process=1`
- Thread-4 command completed:
  - Output: `/tmp/ppc_numba_parallel_benchmark/thread4/devauc/20260511_130710_devauc_sonnenfeld2024-slacs-sigma-star-gamma-hunit/ppc`
  - Wall time: `real 7.72`
  - Metadata: `strategy=kernel_only`, `requested_worker_processes=4`, `worker_processes=0`, `kernel_threads_per_process=4`
- Interpretation:
  - This medium benchmark validates the repaired execution contract and artifact metadata on a real Sonnenfeld run.
  - The selected size is still small enough that wall-time scaling is not a robust full-chain performance claim.
- Task 7 accepted.
- Next step: Task 8 documentation update.

### 2026-05-17 Task 8 Documentation Update

- Updated `Posterior_predictive_test/README.md`.
- Documented:
  - PPC diagnostics use adapter-owned Numba kernels by default.
  - `--worker-processes` is interpreted as requested diagnostics compute width under `kernel_only`.
  - Artifact metadata records `requested_worker_processes`, `worker_processes=0`, and `kernel_threads_per_process`.
  - New adapters must accept keyword-only `execution`, own model-specific predictive semantics, and apply the resolved kernel thread budget before entering compiled kernels.
- Task 8 accepted.
- Next step: final verification.

### 2026-05-17 Final Verification

- Benchmark artifact metadata check:
  - Command: JSON verification script over `/tmp/ppc_numba_parallel_benchmark/thread1/**/ppc_summary.json` and `/tmp/ppc_numba_parallel_benchmark/thread4/**/ppc_summary.json`
  - Result: passed.
  - Confirmed backend `numba_sonnenfeld_parent`, `n_posterior_draws_used=64`, `worker_processes=0`, and requested/kernel widths of `1` and `4`.
- Registry tests:
  - Command: `conda run -n cmass_lens --no-capture-output env PYTHONPATH="$PWD/Bayesian_inference/src:$PWD/Posterior_predictive_test/src:$PWD/prepare_dataset:/Users/liurongfu/tools" python -m pytest Posterior_predictive_test/tests/test_predictive_registry.py -q`
  - Result: `15 passed`.
- PPC parallelism/metadata targeted tests:
  - Command: targeted `test_posterior_predictive.py` subset covering diagnostics, execution metadata, CMASS thread handoff, invalid width, zero-draw rejection, trend metadata, deterministic worker-width behavior, and CLI entry points.
  - Result: `9 passed`.
- Broader selected subset:
  - Command: same subset plus `test_run_posterior_trends_generates_expected_artifacts_for_sersic`.
  - Result: `9 passed, 1 failed`.
  - Failure: existing `KeyError: 'display_xlim'` in `test_run_posterior_trends_generates_expected_artifacts_for_sersic`; this was already recorded during Task 5 review as unrelated to the PPC parallelism repair.
- Whitespace check:
  - Command: `git diff --check -- ...` over PPC repair files and docs.
  - Result: passed.
