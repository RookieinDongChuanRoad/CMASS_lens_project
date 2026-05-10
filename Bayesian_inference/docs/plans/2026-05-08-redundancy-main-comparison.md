# Redundancy Cleanup And Main Comparison Implementation Plan

> **For Codex:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` before executing this plan task-by-task.

**Goal:** 在不改变当前科学语义和 numba/emcee production 后端的前提下，完成去冗余清理，并用可复现证据对比当前工作树与 `main` 分支的 CMASS log-prob 数值一致性和速度。

**Architecture:** 先冻结当前工作树的性能和数值基线，再做去冗余，最后用同一套 branch-aware harness 对比 `main`、清理前 HEAD、清理后 HEAD。去冗余只清掉重复 glue、陈旧命名和可证明无用的层；不得把 CMASS/Sonnenfeld 私有 posterior 逻辑重新推回 shared `components/` 或 `numba_backend/`。

**Tech Stack:** Python, pytest, Numba, emcee, HDF5, git worktree, `conda run -n cmass_lens`, JSON benchmark artifacts.

---

## Non-Negotiable Constraints

- 所有命令都在 `conda run -n cmass_lens ...` 下执行。
- `components/` 和 `numba_backend/` 继续保持无 production model-specific 内容。
- `models/<model>/posterior.py` 继续是单文件 posterior assembly，不恢复 `posterior_kernels.py`。
- 性能比较必须区分 first-call/JIT 时间和 steady-state log-prob 时间。
- 数值比较必须使用相同 theta、相同积分规模、相同随机种子和同一份科学数据内容。
- 如果真实 HDF5 数据缺失，先完成 synthetic/medium benchmark；真实数据 benchmark 标为 data-gated，不伪造结论。

## Acceptance Definition

这两件事完成的标准是：

- 冗余清理后，`git diff --check` 通过。
- `find Bayesian_inference/src/cmass_lens_inference/models -name 'posterior_kernels.py' -print` 无输出。
- `rg -n "cmass|sonnenfeld|CMASS|Sonnenfeld|SLACS" Bayesian_inference/src/cmass_lens_inference/components Bayesian_inference/src/cmass_lens_inference/numba_backend -g '*.py'` 无输出。
- `conda run -n cmass_lens python -m pytest Bayesian_inference/tests -q -rs` 通过；真实数据缺失导致的 skip 必须单独记录。
- main 对照报告写出两个 git hash、数据 case、theta、积分规模、线程设置、first-call 时间、steady-state median/min/mean、log-prob 值、normalization 值。
- 对 synthetic deterministic case，当前分支和 `main` 的 log-prob 与 normalization 差异必须落在明确容差内；建议初始容差 `rtol=1e-10, atol=1e-8`，如果失败必须解释是科学语义差异、数据 contract 差异，还是 bug。
- 对速度，steady-state median 不得比 `main` 慢超过 10%；如果慢超过 10%，必须附 profiling 或 timing breakdown 后才能进入下一步清理。

---

## Phase 0: Freeze Current Baseline Before Cleanup

**Files:**

- Read: `Bayesian_inference/scripts/benchmark_log_prob.py`
- Read: `Bayesian_inference/tests/test_numba_emcee_inference.py`
- Read: `Bayesian_inference/tests/test_real_data_canonical_equivalence.py`
- Create later if needed: `Bayesian_inference/docs/reports/main_comparison/`

**Step 1: Confirm repository state and branch hashes**

Run:

```bash
git status --short -uall
git rev-parse main
git rev-parse HEAD
git merge-base main HEAD
```

Expected:

- No unexpected dirty files outside the planned docs/scripts/tests/code.
- `main` and `HEAD` hashes are recorded.

**Step 2: Run current full regression suite**

Run:

```bash
conda run -n cmass_lens python -m pytest Bayesian_inference/tests -q -rs
```

Expected:

- Pass.
- Any skip is recorded with the exact missing data paths.

**Step 3: Run current synthetic log-prob benchmark**

Use the existing benchmark entrypoint with a synthetic config generated from test fixtures or a dedicated benchmark fixture script.

Run shape:

```bash
conda run -n cmass_lens python Bayesian_inference/scripts/benchmark_log_prob.py \
  --config /tmp/<case>/cmass_synthetic.yaml \
  --repeats 20 \
  --output-dir Bayesian_inference/docs/reports/main_comparison/current_pre_cleanup
```

Expected JSON fields:

- `first_call_seconds`
- `steady_log_prob_median_seconds`
- `steady_log_prob_min_seconds`
- `steady_log_prob_mean_seconds`
- `steady_log_prob_value`
- `steady_normalization_value`

---

## Phase 1: Redundancy Audit

**Files:**

- Read: `Bayesian_inference/src/cmass_lens_inference/models/cmass/posterior.py`
- Read: `Bayesian_inference/src/cmass_lens_inference/models/sonnenfeld2024_slacs/posterior.py`
- Read: `Bayesian_inference/src/cmass_lens_inference/components/`
- Read: `Bayesian_inference/src/cmass_lens_inference/numba_backend/`
- Read: `Bayesian_inference/tests/`
- Create: `Bayesian_inference/docs/reports/redundancy_audit.md`

**Step 1: Static search for stale file names and old architecture words**

Run:

```bash
rg -n "posterior_kernels|production.py|models/components|cmass_kernels|sonnenfeld_kernels|jax|numpyro" \
  Bayesian_inference/src Bayesian_inference/tests Bayesian_inference/docs
```

Expected:

- Source and tests should not contain stale implementation references except intentional historical docs.
- Historical progress-doc references should be classified as history, not implementation debt.

**Step 2: Static search for duplicated helper names**

Run:

```bash
rg -n "def _unpack_theta|def unpack_|def .*_density|def .*_relation|__all__|KernelRef\\(" \
  Bayesian_inference/src/cmass_lens_inference
```

Expected:

- Produce a short audit table:
  - safe to remove;
  - safe to rename;
  - intentionally duplicated because model-specific;
  - candidate for shared model-agnostic kernel promotion.

**Step 3: Decide cleanup boundaries**

Rules:

- Promote code to `numba_backend/kernels/` only if it is genuinely model-agnostic and has no paper/model constants.
- Keep model-specific theta unpacking, posterior reduction, likelihood composition, and fused loops inside `models/<model>/posterior.py`.
- Do not split `posterior.py` into a sibling kernel file.

---

## Phase 2: Low-Risk Redundancy Cleanup

**Files:**

- Modify as audit justifies:
  - `Bayesian_inference/src/cmass_lens_inference/models/cmass/posterior.py`
  - `Bayesian_inference/src/cmass_lens_inference/models/sonnenfeld2024_slacs/posterior.py`
  - `Bayesian_inference/src/cmass_lens_inference/models/*/__init__.py`
  - `Bayesian_inference/tests/*`
  - docs only if they are active target/progress docs for this work

**Step 1: Write/extend static tests before cleanup**

Likely tests:

```python
def test_no_stale_posterior_kernel_modules():
    assert list((PACKAGE_ROOT / "models").rglob("posterior_kernels.py")) == []
```

Additional tests should be added only for concrete redundancy risks found in Phase 1.

**Step 2: Run the targeted tests and confirm they fail only when they should**

Run:

```bash
conda run -n cmass_lens python -m pytest Bayesian_inference/tests/test_component_kernel_boundary.py -q
```

Expected:

- Existing boundary tests pass.
- Any newly added test must protect a real intended invariant.

**Step 3: Apply cleanup in small patches**

Allowed cleanup examples:

- Remove dead imports and stale `__all__` entries.
- Rename comments/docstrings that still say “production kernels” where the file is now posterior-owned.
- Remove duplicated test helpers only when one local helper can express the same fixture contract without hiding data shape.
- Extract tiny local private helpers inside the same `posterior.py` only when they reduce repeated argument construction without changing Numba signatures or runtime behavior.

Disallowed cleanup examples:

- Moving CMASS/Sonnenfeld posterior loops into `numba_backend/`.
- Reintroducing `posterior_kernels.py`.
- Merging scientifically different CMASS and Sonnenfeld helpers just because their function shapes look similar.

**Step 4: Run targeted regression**

Run:

```bash
conda run -n cmass_lens python -m pytest \
  Bayesian_inference/tests/test_component_kernel_boundary.py \
  Bayesian_inference/tests/test_model_registry_config.py \
  Bayesian_inference/tests/test_numba_emcee_inference.py \
  Bayesian_inference/tests/test_sonnenfeld_runtime_model.py \
  -q
```

Expected:

- Pass.

---

## Phase 3: Build A Branch-Aware Main Comparison Harness

**Files:**

- Create: `Bayesian_inference/scripts/compare_log_prob_with_main.py`
- Test: `Bayesian_inference/tests/test_main_comparison_harness.py`
- Output directory: `Bayesian_inference/docs/reports/main_comparison/`

**Step 1: Write tests for the comparison result schema**

The first test should validate pure JSON/schema behavior, not run both branches.

Expected schema:

```json
{
  "case_name": "cmass_synthetic_devauc",
  "main_git_hash": "...",
  "candidate_git_hash": "...",
  "environment": {
    "python": "...",
    "numba_threads": 1,
    "omp_max_active_levels": "1"
  },
  "main": {
    "log_prob_value": 0.0,
    "normalization_value": 0.0,
    "steady_log_prob_median_seconds": 0.0
  },
  "candidate": {
    "log_prob_value": 0.0,
    "normalization_value": 0.0,
    "steady_log_prob_median_seconds": 0.0
  },
  "comparison": {
    "log_prob_abs_diff": 0.0,
    "normalization_abs_diff": 0.0,
    "steady_speed_ratio_candidate_over_main": 1.0
  }
}
```

Run:

```bash
conda run -n cmass_lens python -m pytest Bayesian_inference/tests/test_main_comparison_harness.py -q
```

Expected before implementation:

- Fail because the harness module does not exist.

**Step 2: Implement branch-specific import adapters**

The harness must support both known APIs:

- `main`: `cmass_lens_inference.model.build_compiled_model`, `cmass_lens_inference.model.log_prob`
- current candidate: `cmass_lens_inference.numba_backend.likelihood_engine.build_compiled_model`, `cmass_lens_inference.numba_backend.likelihood_engine.log_prob`

The harness should run each branch in a subprocess with isolated `PYTHONPATH`, rather than importing both branches into one Python interpreter.

**Step 3: Generate or point to branch-compatible data**

Preferred order:

1. real devauc data if all required HDF5 files exist;
2. medium synthetic dataset generated once and consumed by both branches;
3. tiny synthetic dataset for CI-style fast checks.

For synthetic comparison:

- `main` consumes raw legacy-style observation/cross-section files.
- candidate consumes canonical inference HDF5 generated from the same raw scientific content.
- theta vector and integration sizes are explicitly written into both branch-specific YAML configs.

**Step 4: Benchmark protocol**

For each branch:

- one cold call, recorded separately;
- one warmup call, not included in steady statistics;
- `repeats=20` steady calls for synthetic/medium;
- `repeats=5` for real-data case if runtime is large;
- fixed thread strategy, initially `parallel_strategy=off` or one Numba thread for deterministic comparison;
- optional second pass with `kernel_only` to measure production parallel strategy.

---

## Phase 4: Main Numerical Consistency Test

**Files:**

- Create or update: `Bayesian_inference/docs/reports/main_comparison/<timestamp>_summary.md`
- Create JSON outputs under: `Bayesian_inference/docs/reports/main_comparison/<timestamp>/`

**Step 1: Run synthetic branch comparison**

Run shape:

```bash
conda run -n cmass_lens python Bayesian_inference/scripts/compare_log_prob_with_main.py \
  --main-ref main \
  --candidate-ref HEAD \
  --case cmass_synthetic_devauc \
  --repeats 20 \
  --output-dir Bayesian_inference/docs/reports/main_comparison/<timestamp>
```

Expected:

- Writes machine-readable JSON.
- Prints absolute/relative differences for log-prob and normalization.

**Step 2: Apply acceptance thresholds**

Expected:

- `abs(log_prob_candidate - log_prob_main) <= 1e-8` or explained.
- `abs(norm_candidate - norm_main) <= 1e-10` or explained.
- If thresholds fail, stop cleanup work and diagnose the semantic difference first.

---

## Phase 5: Main Speed Comparison

**Files:**

- Same harness/report files as Phase 4.

**Step 1: Run steady-state benchmark**

Run:

```bash
conda run -n cmass_lens python Bayesian_inference/scripts/compare_log_prob_with_main.py \
  --main-ref main \
  --candidate-ref HEAD \
  --case cmass_synthetic_devauc \
  --repeats 50 \
  --output-dir Bayesian_inference/docs/reports/main_comparison/<timestamp>
```

Expected:

- `first_call_seconds` recorded but not used as speed acceptance.
- `steady_log_prob_median_seconds` is the main speed metric.

**Step 2: Run real-data or medium-data benchmark**

If real HDF5 files exist:

```bash
conda run -n cmass_lens python Bayesian_inference/scripts/compare_log_prob_with_main.py \
  --main-ref main \
  --candidate-ref HEAD \
  --case cmass_real_devauc_slit \
  --repeats 5 \
  --output-dir Bayesian_inference/docs/reports/main_comparison/<timestamp>
```

If real HDF5 files do not exist:

- Run `cmass_medium_synthetic_devauc`.
- Mark real-data benchmark as data-gated with missing path list.

**Step 3: Interpret speed**

Acceptance:

- Candidate steady median <= `1.10 * main steady median`.

If candidate is slower:

- Compare `likelihood_seconds` and `normalization_seconds`.
- Run a smaller profiling pass around `normalization_mc_numba` and `log_likelihood_lenses_numba`.
- Do not start broad cleanup until the slowdown is localized.

---

## Phase 6: Post-Cleanup Re-run And Report

**Files:**

- Update by append if used for this work:
  - `Bayesian_inference/docs/component_kernel_refactor_progress.md`
- Create final report:
  - `Bayesian_inference/docs/reports/main_comparison/<timestamp>_summary.md`

**Step 1: Re-run the same main comparison after redundancy cleanup**

Use the same command, same data case, same theta, same repeats, same thread settings.

Expected:

- Candidate post-cleanup numerical values match candidate pre-cleanup within `rtol=1e-12, atol=1e-10`.
- Candidate post-cleanup steady median does not regress by more than 5% relative to candidate pre-cleanup.

**Step 2: Run final regression**

Run:

```bash
git diff --check
find Bayesian_inference/src/cmass_lens_inference/models -name 'posterior_kernels.py' -print
rg -n "cmass|sonnenfeld|CMASS|Sonnenfeld|SLACS" \
  Bayesian_inference/src/cmass_lens_inference/components \
  Bayesian_inference/src/cmass_lens_inference/numba_backend \
  -g '*.py'
conda run -n cmass_lens python -m pytest Bayesian_inference/tests -q -rs
```

Expected:

- `git diff --check`: no output.
- `find`: no output.
- `rg`: no output.
- pytest: pass, with any real-data skips recorded.

**Step 3: Write final summary**

The final summary must include:

- What redundancy was removed.
- What redundancy was intentionally kept and why.
- main hash and candidate hash.
- numerical comparison table.
- speed comparison table.
- data-gated checks, if any.
- final acceptance status.
