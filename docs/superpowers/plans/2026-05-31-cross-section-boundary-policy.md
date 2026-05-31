# Cross-Section Boundary Policy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` or `superpowers:subagent-driven-development` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 canonical dataset 的 `lensing_cross_section.boundary_policy` 成为运行时 cross-section 边界语义的唯一合同，使 CMASS separable cross-section 在 `theta_E` 轴外按 `theta_E^2` 解析延拓，同时保持 Sonnenfeld finite-fibre 二维表在 `theta_E` 轴外置零。

**Architecture:** Cross-section 的数据来源由 `source` 表达，边界评价策略由 `boundary_policy` 表达；reader 暴露这两个字段，writer 为不同来源写入正确 policy，runtime 将二者解析成 Numba 可消费的整数 mode。CMASS inference 和 posterior diagnostics 统一走 policy-aware evaluator；Sonnenfeld 和其他真实二维表继续走现有 `interp_cross_section_theta_gamma` 语义，不做全局线性外推。

**Tech Stack:** Python 3.11 in `cmass_lens`, HDF5/h5py, NumPy, Numba, pytest, existing `statistical_sl` package layout, existing canonical HDF5 workspace data.

---

## Scope, Constraints, And Completion Gate

本计划只修复 cross-section 边界合同和运行时消费语义。它不重新设计 inference 参数、FP prior、mass definition、posterior diagnostic workflow，也不启动长时间 inference。

Execution constraints:

- 所有 Python / pytest / CLI 命令必须通过 `conda run -n cmass_lens --no-capture-output ...` 执行。
- 不得中断任何已经启动的长任务；本计划的验证默认只跑单元测试、导入 smoke、短小数值检查。
- 不得全局修改 `interp_cross_section_theta_gamma` 的 out-of-range 行为，除非只是补注释或测试。
- 不得让 Sonnenfeld finite-fibre tables 获得 CMASS 的 `theta_E^2` 解析延拓。
- 不得把修复写成 `if model_name == "cmass"` 的硬编码；运行时必须由 canonical cross-section `source + boundary_policy` 解析得到 evaluator mode。
- 当前 workspace 里的 CMASS canonical HDF5 需要做一次只改 attr、不改数值数组的 metadata migration；Sonnenfeld canonical HDF5 保持原 policy。

Definition of done:

- `CanonicalCrossSectionGrid` 暴露 `source` 和 `boundary_policy`。
- Writer 对 legacy CMASS separable input 写入 `source="separable_cs_over_theta_ein"` 和 `boundary_policy="theta_squared_extrapolate_clip_gamma"`。
- Writer 对 Sonnenfeld finite-fibre input 写入 `source="mufibre3_cs_grid"` 和 `boundary_policy="zero_outside_theta_clip_gamma"`。
- 当前 `workspace/data/canonical/inference_dataset_devauc_slit_m5_hunits_v1.hdf5` 的 `lensing_cross_section.attrs["boundary_policy"]` 被迁移为 `theta_squared_extrapolate_clip_gamma`，且 `theta_e_axis`、`gamma_axis`、`cross_section_grid` 的字节内容不变。
- 当前两个 Sonnenfeld canonical HDF5 仍为 `source="mufibre3_cs_grid"` 和 `boundary_policy="zero_outside_theta_clip_gamma"`。
- CMASS context 从 separable canonical grid 恢复 `cs_over_theta_grid`，并携带 Numba 可用的 `cross_section_mode_code`。
- CMASS inference normalization、FP-enabled population summary、observed-lens likelihood 都使用同一个 policy-aware selection helper。
- CMASS posterior diagnostics / PPC 里的 detectable 和 selected population 也使用同一个 policy-aware cross-section 语义。
- Sonnenfeld inference 和 Sonnenfeld posterior diagnostics 仍使用现有 finite-grid `zero_outside_theta_clip_gamma` 语义。
- 测试证明：CMASS 在 `theta_E > theta_axis.max()` 时 cross-section 非零并满足 `area = pi * (theta_E * cs_over_theta(gamma))**2`；generic finite 2D interpolation 在同样越界时仍返回 `0.0`。
- `git diff --check` 通过；focused pytest 通过。

Non-goals:

- 不重跑完整 CMASS inference。
- 不做 posterior corner / posterior diagnostics 科学结果对比。
- 不重建 canonical 数值数据。
- 不迁移历史 run directories。
- 不删除或重构旧的 data-preparation pipeline。

## Contract Vocabulary

新增或集中维护这些合同常量，避免字符串散落在 writer、reader、preprocessing、tests 中：

```python
SOURCE_SEPARABLE_CS_OVER_THETA_EIN = "separable_cs_over_theta_ein"
SOURCE_MUFIBRE3_CS_GRID = "mufibre3_cs_grid"

BOUNDARY_ZERO_OUTSIDE_THETA_CLIP_GAMMA = "zero_outside_theta_clip_gamma"
BOUNDARY_THETA_SQUARED_EXTRAPOLATE_CLIP_GAMMA = "theta_squared_extrapolate_clip_gamma"

CROSS_SECTION_MODE_GRID_ZERO_OUTSIDE = 0
CROSS_SECTION_MODE_SEPARABLE_THETA_SQUARED = 1
```

Policy resolver 的基本规则：

```text
source=mufibre3_cs_grid
boundary_policy=zero_outside_theta_clip_gamma
  -> CROSS_SECTION_MODE_GRID_ZERO_OUTSIDE

source=separable_cs_over_theta_ein
boundary_policy=theta_squared_extrapolate_clip_gamma
  -> CROSS_SECTION_MODE_SEPARABLE_THETA_SQUARED

source=separable_cs_over_theta_ein
boundary_policy=zero_outside_theta_clip_gamma
  -> CROSS_SECTION_MODE_GRID_ZERO_OUTSIDE
     只用于读取旧数据时保持诚实语义；当前 workspace CMASS canonical 文件必须迁移到新 policy。

source=mufibre3_cs_grid
boundary_policy=theta_squared_extrapolate_clip_gamma
  -> error
     真实二维 finite-fibre 表不能声明 CMASS separable 解析延拓。
```

## Task 1: Add Failing Contract Tests

**Files:**

- Create: `tests/test_cross_section_boundary_policy.py`
- Modify only if useful: `tests/test_cmass_devauc_fp_contract.py`

**Step 1: Test canonical metadata inventory**

- [ ] 写测试读取三个现有 canonical HDF5：
  - `workspace/data/canonical/inference_dataset_devauc_slit_m5_hunits_v1.hdf5`
  - `workspace/data/canonical/inference_dataset_sonnenfeld2024_slacs_m5_fixed_v1.hdf5`
  - `workspace/data/canonical/inference_dataset_sonnenfeld2024_slacs_m5_hunits_v1.hdf5`
- [ ] 期望 CMASS devauc 的 `source` 为 `separable_cs_over_theta_ein`。
- [ ] 期望 CMASS devauc 的 `boundary_policy` 为 `theta_squared_extrapolate_clip_gamma`。
- [ ] 期望两个 Sonnenfeld 文件的 `source` 为 `mufibre3_cs_grid`，`boundary_policy` 为 `zero_outside_theta_clip_gamma`。

Run:

```bash
conda run -n cmass_lens --no-capture-output python -m pytest tests/test_cross_section_boundary_policy.py -q
```

Expected before migration: FAIL because current CMASS devauc HDF5 still says `zero_outside_theta_clip_gamma`.

**Step 2: Test reader exposes `source`**

- [ ] 使用 `load_canonical_inference_dataset(...)` 读取 CMASS devauc canonical file。
- [ ] 断言 `dataset.cross_section.source == "separable_cs_over_theta_ein"`。
- [ ] 断言 `dataset.cross_section.boundary_policy == "theta_squared_extrapolate_clip_gamma"` after migration。

Expected before implementation: FAIL because `CanonicalCrossSectionGrid` has no `source` field.

**Step 3: Test resolver rejects impossible policy/source combinations**

- [ ] 调用新的 resolver：
  - `mufibre3_cs_grid + theta_squared_extrapolate_clip_gamma` 必须 raise `ValueError`。
  - `separable_cs_over_theta_ein + theta_squared_extrapolate_clip_gamma` 必须返回 separable mode。
  - `mufibre3_cs_grid + zero_outside_theta_clip_gamma` 必须返回 grid mode。

Expected before implementation: FAIL because resolver does not exist.

## Task 2: Centralize Cross-Section Policy Constants

**Files:**

- Create: `src/statistical_sl/core/cross_section_policy.py`
- Modify: `src/statistical_sl/core/__init__.py` if package exports are maintained there
- Test: `tests/test_cross_section_boundary_policy.py`

**Step 1: Implement constants and resolver**

- [ ] 新增 `CrossSectionPolicyError(ValueError)` 或直接使用 `ValueError`。
- [ ] 新增字符串常量和整数 mode 常量。
- [ ] 新增 `resolve_cross_section_mode(source: str, boundary_policy: str) -> int`。
- [ ] 新增 `is_separable_theta_squared_mode(mode_code: int) -> bool` only if it improves call-site readability.

Implementation requirements:

- 函数注释必须解释为什么 `source` 和 `boundary_policy` 不能合并成一个字段。
- 错误消息必须同时包含 source 和 boundary_policy，方便排查 HDF5 metadata。
- 不要导入 model-specific modules，避免 core 反向依赖模型。

**Step 2: Run resolver tests**

Run:

```bash
conda run -n cmass_lens --no-capture-output python -m pytest tests/test_cross_section_boundary_policy.py::test_cross_section_policy_resolver_rejects_invalid_combinations -q
```

Expected: PASS.

## Task 3: Update Canonical Reader And Writer Contracts

**Files:**

- Modify: `src/statistical_sl/inference/canonical_dataset.py`
- Modify: `src/statistical_sl/data_preparation/dataset_schema/writer.py`
- Test: `tests/test_cross_section_boundary_policy.py`
- Reference docs to update if wording is now stale: `docs/reports/2026-05-05-canonical-inference-dataset-schema.md`

**Step 1: Reader exposes source**

- [ ] Add `source: str` to `CanonicalCrossSectionGrid`.
- [ ] In `_load_cross_section`, read `group.attrs.get("source", "")`.
- [ ] Keep `boundary_policy` read path unchanged except import/use constants if helpful.

**Step 2: Writer chooses policy by source**

- [ ] Replace single `DEFAULT_BOUNDARY_POLICY` write with source-aware policy selection.
- [ ] CMASS legacy separable source writes `BOUNDARY_THETA_SQUARED_EXTRAPOLATE_CLIP_GAMMA`.
- [ ] Sonnenfeld `mufibre3_cs_grid` source writes `BOUNDARY_ZERO_OUTSIDE_THETA_CLIP_GAMMA`.
- [ ] Raise if `_read_cross_section_product` returns an unknown source.

**Step 3: Update schema report**

- [ ] Amend the old report language that says the unified 2D interface hides separable origin from runtime.
- [ ] New wording must say: canonical still stores a unified 2D grid for common consumers, but `source + boundary_policy` are runtime-visible because separable CMASS has a valid analytic extension outside the finite theta grid.

Run:

```bash
conda run -n cmass_lens --no-capture-output python -m pytest tests/test_cross_section_boundary_policy.py::test_canonical_reader_exposes_cross_section_source -q
```

Expected after metadata migration: PASS.

## Task 4: Migrate Current Workspace Canonical Metadata

**Files/Data:**

- Modify HDF5 attr only: `workspace/data/canonical/inference_dataset_devauc_slit_m5_hunits_v1.hdf5`
- Do not modify numeric datasets in that file.
- Do not modify the two Sonnenfeld canonical HDF5 files.

**Step 1: Record pre-migration numeric hashes**

Run:

```bash
conda run -n cmass_lens --no-capture-output python - <<'PY'
from pathlib import Path
import hashlib
import h5py

path = Path("workspace/data/canonical/inference_dataset_devauc_slit_m5_hunits_v1.hdf5")
with h5py.File(path, "r") as handle:
    group = handle["lensing_cross_section"]
    for name in ("theta_e_axis", "gamma_axis", "cross_section_grid"):
        payload = group[name][()].tobytes()
        print(name, hashlib.sha256(payload).hexdigest())
    print("source", group.attrs["source"])
    print("boundary_policy", group.attrs["boundary_policy"])
PY
```

Expected before migration: numeric hashes printed; `boundary_policy zero_outside_theta_clip_gamma`.

**Step 2: Update only the boundary attr**

Run:

```bash
conda run -n cmass_lens --no-capture-output python - <<'PY'
from pathlib import Path
import h5py

path = Path("workspace/data/canonical/inference_dataset_devauc_slit_m5_hunits_v1.hdf5")
with h5py.File(path, "r+") as handle:
    group = handle["lensing_cross_section"]
    source = group.attrs.get("source")
    if isinstance(source, bytes):
        source = source.decode("utf-8")
    if source != "separable_cs_over_theta_ein":
        raise RuntimeError(f"Refusing to migrate non-CMASS separable source: {source!r}")
    group.attrs["boundary_policy"] = "theta_squared_extrapolate_clip_gamma"
PY
```

**Step 3: Verify numeric hashes did not change**

- [ ] Re-run Step 1.
- [ ] Confirm all three numeric hashes match pre-migration values.
- [ ] Confirm only the attr changed.

**Step 4: Run metadata tests**

Run:

```bash
conda run -n cmass_lens --no-capture-output python -m pytest tests/test_cross_section_boundary_policy.py::test_workspace_canonical_cross_section_metadata_matches_policy_contract -q
```

Expected: PASS.

## Task 5: Add Policy-Aware Numba Cross-Section Evaluator

**Files:**

- Modify: `src/statistical_sl/numerics/numba/kernels/selection_likelihood.py`
- Do not behaviorally modify: `src/statistical_sl/numerics/numba/kernels/interpolation.py`
- Test: `tests/test_cross_section_boundary_policy.py`

**Step 1: Add separable evaluator**

- [ ] Import `interp1d_clip` from `.interpolation`.
- [ ] Add Numba helper:

```python
@nb.njit(cache=True, inline="always")
def separable_theta_squared_cross_section(
    theta_e: float,
    gamma: float,
    gamma_axis: np.ndarray,
    cs_over_theta_grid: np.ndarray,
) -> float:
    """Evaluate CMASS separable cross-section with analytic theta_E^2 scaling."""
```

Required behavior:

- `theta_e <= 0` returns `0.0`.
- non-finite `theta_e` or `gamma` returns `0.0`.
- gamma is interpolated/clipped with `interp1d_clip`.
- area is `math.pi * (theta_e * cs_over_theta) ** 2`.
- negative or non-finite `cs_over_theta` returns `0.0`.

**Step 2: Add selection helper**

- [ ] Add helper that combines separable area with `p_find`, analogous to existing `cross_section_find_weight`.
- [ ] Add one mode-aware helper if it avoids duplicate call-site branches:

```python
@nb.njit(cache=True, inline="always")
def policy_cross_section_find_weight(
    theta_e: float,
    gamma: float,
    theta_for_detection: float,
    theta0: float,
    loga: float,
    mode_code: int,
    theta_e_axis: np.ndarray,
    gamma_axis: np.ndarray,
    cross_section_grid: np.ndarray,
    cs_over_theta_grid: np.ndarray,
) -> float:
    ...
```

Required behavior:

- grid mode delegates to existing `cross_section_find_weight`.
- separable mode uses analytic theta-squared helper.
- unknown mode returns `0.0` or raises only outside Numba; prefer resolver to catch invalid modes before kernels run.

**Step 3: Tests**

- [ ] Assert `interp_cross_section_theta_gamma(theta_above_max, ...) == 0.0`.
- [ ] Assert `separable_theta_squared_cross_section(theta_above_max, gamma, ...)` equals expected analytic value.
- [ ] Assert `policy_cross_section_find_weight` multiplies by `p_find` for separable mode.

Run:

```bash
conda run -n cmass_lens --no-capture-output python -m pytest tests/test_cross_section_boundary_policy.py::test_cmass_separable_cross_section_extrapolates_as_theta_squared tests/test_cross_section_boundary_policy.py::test_generic_grid_cross_section_still_zeroes_theta_outside -q
```

Expected: PASS.

## Task 6: Wire CMASS Context Construction

**Files:**

- Modify: `src/statistical_sl/models/cmass/context.py`
- Modify: `src/statistical_sl/models/cmass/preprocessing.py`
- Modify: `src/statistical_sl/models/cmass/runtime.py`
- Inspect/possibly modify: `src/statistical_sl/models/cmass/assembly.py`
- Inspect/possibly modify: `src/statistical_sl/inference/compiled_context.py`
- Test: `tests/test_cross_section_boundary_policy.py`

**Step 1: Add context field**

- [ ] Add `cross_section_mode_code: int` to `CMASSModelContext`.
- [ ] Add matching `ContextStaticSpec` or `ContextScalarSpec` in `cmass/runtime.py`; choose the existing backend convention that best matches mode flags such as `use_sersic_index`, `fp_enabled`, or `gamma_mode_code`.

**Step 2: Recover `cs_over_theta_grid` from canonical grid**

- [ ] In `cmass/preprocessing.py`, add a helper that recovers `cs_over_theta(gamma)` only when source/policy resolve to separable theta-squared mode.
- [ ] Use the largest strictly positive `theta_e_axis` row for numerical stability.
- [ ] Formula:

```python
cs_over_theta = np.sqrt(np.maximum(cross_section_grid[row_index], 0.0) / np.pi) / theta_value
```

- [ ] Validate shape is `(gamma_axis.size,)`.
- [ ] Validate all recovered values are finite and non-negative.
- [ ] For grid-zero mode, keep `cs_over_theta_grid` as zeros unless a downstream compatibility path needs the original separable values.

**Step 3: Populate interpolation-grid factor**

- [ ] If `cs_over_theta_int` is still consumed by any live path, populate it by interpolating recovered `cs_over_theta_grid` onto `gamma_grid_int`.
- [ ] If it is not consumed after policy-aware evaluator wiring, keep it for backend compatibility but document why it remains.

**Step 4: Audit `inference/compiled_context.py`**

- [ ] Determine whether `build_compiled_context` is still reachable from current production PPC or legacy compatibility CLI.
- [ ] If reachable, add `cross_section_mode_code` there too and set it from legacy separable inputs.
- [ ] If not reachable for production inference, add a comment/test proving no current production path depends on it, but still keep dataclass construction valid after adding the required field.

**Step 5: Tests**

- [ ] Build CMASS context from current devauc config.
- [ ] Assert `context.cross_section_mode_code` is separable theta-squared mode.
- [ ] Assert `context.cs_over_theta_grid` matches the value recovered from canonical `cross_section_grid`.
- [ ] If `workspace/data/external/cs_grid_power.h5` is available, compare recovered `cs_over_theta_grid` against its compressed legacy grid.

Run:

```bash
conda run -n cmass_lens --no-capture-output python -m pytest tests/test_cross_section_boundary_policy.py::test_cmass_context_recovers_separable_cross_section_factor -q
```

Expected: PASS.

## Task 7: Wire CMASS Inference Likelihood

**Files:**

- Modify: `src/statistical_sl/models/cmass/posterior.py`
- Test: `tests/test_cross_section_boundary_policy.py`

**Step 1: Update Numba function signatures**

- [ ] `normalization_mc_numba` receives `cross_section_mode_code` and `cs_over_theta_grid`.
- [ ] `population_summary_mc_numba` receives `cross_section_mode_code` and `cs_over_theta_grid`.
- [ ] `log_likelihood_lenses_numba` receives `cross_section_mode_code` and `cs_over_theta_grid`.

**Step 2: Replace CMASS call sites**

- [ ] Replace each CMASS call to `cross_section_find_weight(...)` with `policy_cross_section_find_weight(...)`.
- [ ] Preserve detection probability argument `theta_for_detection=theta_e`.
- [ ] Do not change parameter unpacking, priors, FP prior logic, source-redshift density, or mass interpolation.

**Step 3: Update `log_prob` argument passing**

- [ ] Pass `context.cross_section_mode_code`.
- [ ] Pass `context.cs_over_theta_grid`.
- [ ] Ensure FP-enabled path and no-FP path both pass the same fields.

**Step 4: Tests**

- [ ] Add a small direct-kernel test if feasible: construct a synthetic theta/gamma grid and prove `normalization_mc_numba` no longer drops `theta_E > 5` solely because of the finite theta axis.
- [ ] If direct-kernel setup is too expensive, test the shared helper plus context wiring; do not create a brittle full likelihood fixture just to exercise one branch.

Run:

```bash
conda run -n cmass_lens --no-capture-output python -m pytest tests/test_cross_section_boundary_policy.py tests/test_smoke.py -q
```

Expected: PASS.

## Task 8: Wire CMASS Posterior Diagnostics / PPC

**Files:**

- Modify: `src/statistical_sl/posterior_predictive/adapters/cmass.py`
- Do not modify behavior unless tests prove necessary: `src/statistical_sl/posterior_predictive/adapters/sonnenfeld.py`
- Test: `tests/test_cross_section_boundary_policy.py`

**Step 1: Update chunk function signature**

- [ ] `_shared_parent_diagnostics_numba_chunk` receives `cross_section_mode_code` and `cs_over_theta_grid`.
- [ ] Caller passes `context.cross_section_mode_code` and `context.cs_over_theta_grid`.

**Step 2: Replace raw interpolation**

- [ ] Replace direct `interp_cross_section_theta_gamma(...)` call with policy-aware cross-section evaluation.
- [ ] Preserve current definitions of:
  - `detectable_weight = cross_section`
  - `selected_weight = detectable_weight * discovery_probability`
- [ ] Do not change binning, parent sampling, sigma-star calculation, or diagnostic artifact layout.

**Step 3: Tests**

- [ ] Add a focused diagnostics-helper test if feasible to prove CMASS PPC uses separable extrapolation for `theta_E > 5`.
- [ ] Keep Sonnenfeld diagnostics tests or smoke checks confirming generic grid behavior is unchanged.

Run:

```bash
conda run -n cmass_lens --no-capture-output python -m pytest tests/test_cross_section_boundary_policy.py tests/test_post_canonical_workflow_contract.py tests/test_posterior_predictive_config.py -q
```

Expected: PASS.

## Task 9: Protect Sonnenfeld And Generic Grid Semantics

**Files:**

- Inspect: `src/statistical_sl/models/sonnenfeld2024_slacs/posterior.py`
- Inspect: `src/statistical_sl/models/sonnenfeld2024_slacs_sigma_star_gamma/posterior.py`
- Inspect: `src/statistical_sl/posterior_predictive/adapters/sonnenfeld.py`
- Test: `tests/test_cross_section_boundary_policy.py`

**Step 1: Confirm no Sonnenfeld call site is switched to separable mode**

- [ ] Confirm Sonnenfeld context resolves to grid-zero mode.
- [ ] Confirm Sonnenfeld posterior code can continue using `cross_section_find_weight(...)` directly, or receives grid-zero mode if a shared helper is adopted.
- [ ] Confirm Sonnenfeld PPC direct calls to `interp_cross_section_theta_gamma(...)` remain valid for finite-fibre tables.

**Step 2: Add guard test**

- [ ] Use a small synthetic finite 2D grid.
- [ ] Assert `theta_E > max(theta_axis)` returns zero under grid-zero mode.
- [ ] Assert invalid combination `mufibre3_cs_grid + theta_squared_extrapolate_clip_gamma` fails before Numba kernels run.

Run:

```bash
conda run -n cmass_lens --no-capture-output python -m pytest tests/test_cross_section_boundary_policy.py::test_sonnenfeld_grid_policy_does_not_extrapolate -q
```

Expected: PASS.

## Task 10: Final Verification

**Files:**

- All files touched by previous tasks.

**Step 1: Focused tests**

Run:

```bash
conda run -n cmass_lens --no-capture-output python -m pytest \
  tests/test_cross_section_boundary_policy.py \
  tests/test_cmass_devauc_fp_contract.py \
  tests/test_smoke.py \
  tests/test_post_canonical_workflow_contract.py \
  tests/test_posterior_predictive_config.py \
  -q
```

Expected: PASS.

**Step 2: Import smoke**

Run:

```bash
conda run -n cmass_lens --no-capture-output python - <<'PY'
from statistical_sl.inference.canonical_dataset import load_canonical_inference_dataset
from statistical_sl.numerics.numba.kernels.selection_likelihood import policy_cross_section_find_weight
from statistical_sl.models.cmass.runtime import build_context_bundle

print(load_canonical_inference_dataset)
print(policy_cross_section_find_weight)
print(build_context_bundle)
PY
```

Expected: imports succeed.

**Step 3: Metadata verification**

Run:

```bash
conda run -n cmass_lens --no-capture-output python - <<'PY'
from pathlib import Path
import h5py

for path in sorted(Path("workspace/data/canonical").glob("*.hdf5")):
    with h5py.File(path, "r") as handle:
        group = handle["lensing_cross_section"]
        print(path.name, group.attrs["source"], group.attrs["boundary_policy"])
PY
```

Expected:

```text
inference_dataset_devauc_slit_m5_hunits_v1.hdf5 separable_cs_over_theta_ein theta_squared_extrapolate_clip_gamma
inference_dataset_sonnenfeld2024_slacs_m5_fixed_v1.hdf5 mufibre3_cs_grid zero_outside_theta_clip_gamma
inference_dataset_sonnenfeld2024_slacs_m5_hunits_v1.hdf5 mufibre3_cs_grid zero_outside_theta_clip_gamma
```

**Step 4: Static hygiene**

Run:

```bash
git diff --check
```

Expected: no output.

**Step 5: Optional numerical sanity check, not a full inference run**

- [ ] Evaluate the CMASS policy-aware cross-section at:
  - `theta_E = 4.5`
  - `theta_E = 5.5`
  - same representative `gamma`
- [ ] Confirm the ratio approximately follows `(5.5 / 4.5) ** 2` after accounting for the same interpolated `cs_over_theta(gamma)`.
- [ ] This is only a sanity check; posterior agreement with 4.29 requires a later explicit inference comparison task.

## Suggested Commit Boundaries

1. `test: add cross-section boundary policy contracts`
2. `feat: expose canonical cross-section policy metadata`
3. `feat: add policy-aware cross-section evaluator`
4. `fix: apply cmass theta-squared cross-section policy`
5. `fix: align cmass diagnostics cross-section policy`
6. `docs: update canonical cross-section boundary policy`

Do not commit the metadata migration together with unrelated code churn. If the HDF5 file is untracked in this checkout, record the migration command and verification output in the final response instead of trying to stage it.
