# 目标：Sonnenfeld 外部 Reference 数值对比

本文档定义本地 `sonnenfeld2024_slacs` 与外部 reference
`/Users/liurongfu/reference_codes/strong_lensing_tools/papers/slacs_selection`
之间的数值对比目标、边界、验收标准和实施计划。

当前任务不是再次修改科学模型，而是建立一套可复现、可审计、可逐层定位差异的 reference comparison workflow。最终产物应让后续开发者能够回答：

- 本地实现在哪些层面已经和 reference 数值一致；
- 哪些差异来自积分算法、随机数、grid/data artifact 或单位约定；
- 哪些差异是真正的模型语义偏差；
- 每次修改 Sonnenfeld posterior 后，应该跑哪些对比来防止 drift。

## Reference 与本地对象

### 外部 reference

主 reference 路径：

```text
/Users/liurongfu/reference_codes/strong_lensing_tools/papers/slacs_selection
```

关键脚本：

- `scripts/fit_full.py`：selection-corrected full inference 主参考路径。
- `scripts/fit_slonly.py`：no-selection 局部对照，可用于 per-lens likelihood 的局部项检查。
- `scripts/mz_distribution.py`：parent population `p(z_d, M_*)` 抽样语义。
- `scripts/fitpars.py`：FP prior 默认值与若干全局常数。
- `scripts/make_crosssect_grid.py`：finite-fibre cross-section 生成语义。
- `scripts/make_slacs_lensing_grids.py`、`scripts/make_slacs_jeans_grids.py`：per-lens lensing/dynamics grid 生成语义。

### 本地目标

第一阶段只对比本地 paper-native 变体：

```text
model.name = sonnenfeld2024_slacs
unit_convention = legacy_fixed_kpc
mass_definition_label = m5
```

`sonnenfeld2024_slacs_hunit` 是第二阶段目标。h-unit 对比必须等 paper-native 对比合同稳定后再做，避免同时引入单位坐标平移和 reference 语义差异。

## 最终目标

建立一个 reference comparison harness，使本地 Sonnenfeld 实现能和外部 reference 做分层数值对比，并稳定输出机器可读 JSON 与人工可读 Markdown report。

最终报告必须至少包含：

- 本地 git hash、dirty 状态摘要和被比较文件列表。
- reference 文件 manifest：关键脚本路径、mtime、size、SHA256。外部 reference 不一定是 git repo，不能只记录目录名。
- Python 环境、`PYTHONPATH`、`conda` 环境名、Numba 线程数。
- 输入数据 manifest：canonical dataset、SLACS table、cross-section grid、lensing grid、dynamics grid 的路径、schema、shape、unit metadata、SHA256。
- theta 向量、参数命名映射、随机种子、sample count、积分设置。
- 分层数值结果：primitive、grid、normalization、FP prior、per-lens likelihood、full posterior decomposition。
- 每一层的容差、是否通过、失败时的绝对/相对误差和首个失败项。

最终对比不应只是一个总 `log_prob` 数字。必须输出分解：

```text
log posterior
  = sum_lens log likelihood_i
  - N_lens * log(selection_normalization)
  + fp_prior_log_term
  + box_prior_or_rejection_term
```

如果某一层无法做 exact comparison，报告必须明确写出原因，例如：

- reference 是 stochastic importance sampling，本地是 deterministic quadrature；
- reference artifact 缺失，只能先做 schema/primitive 层；
- grid 来源不同，只能比较趋势或相对项；
- full chain HDF5 没有保存中间项，不能反推出分解。

## 非目标

- 不修改外部 reference repo。
- 不启动 reference 的长时间 `emcee` 采样作为默认验证。
- 不把 `full_inference.hdf5` 的 chain/logp 直接当作本地逐点 oracle。
- 不在第一阶段对比 h-unit variant。
- 不修改 CMASS 模型或 CMASS main-comparison harness 的科学语义。
- 不把 data preparation 和 runtime comparison 混成一层；grid 生成语义先单独验证。

## 核心判断

`fit_full.py` 不能直接 import 后调用，因为它在 top-level 准备数据并启动采样。比较 harness 必须采用以下之一：

1. 在本地实现一个只读 reference oracle adapter，逐行映射 reference 的可比较公式，并在注释中标明来自哪个 reference 文件和行段。
2. 在隔离 subprocess 中运行一个轻量 wrapper，只执行 reference 数据准备和单点 logp/term 计算，不启动 MCMC。

无论选哪种方式，都必须保持 reference repo 只读，并记录 reference manifest。

## 实施计划

### Phase 0：Reference Artifact Audit

目标：先确定有哪些 reference artifact 真实存在、可读、单位明确。

实现内容：

- 新增脚本入口：

```text
Bayesian_inference/scripts/compare_sonnenfeld_with_reference.py
```

- 支持 `--audit-only`，输出 artifact manifest。
- 检查以下路径或由参数显式传入：
  - `SLACS_table.cat`
  - `parent_sample.fits`
  - `full_inference.hdf5`
  - `slonly_inference.hdf5`
  - `fibre_crosssect_grid.hdf5`
  - `slacs_lensing_grids.hdf5`
  - `slacs_jeans_grids.hdf5`
  - 本地 canonical Sonnenfeld dataset
- 对每个 artifact 写出：存在性、shape/schema、units、SHA256、是否足以进入下一阶段。

验收：

- 缺失 artifact 不导致假通过；report 中标记为 `missing` 或 `data_gated`。
- audit-only 不 import 会启动 MCMC 的 reference 脚本。

### Phase 1：Result Schema 与 Subprocess 隔离

目标：先把对比结果格式和运行隔离做好。

实现内容：

- 借鉴 `compare_log_prob_with_main.py` 的模式，每个实现走独立 subprocess。
- 本地 subprocess 使用当前 worktree 的 `Bayesian_inference/src`。
- reference subprocess 使用：

```text
PYTHONPATH=/Users/liurongfu/reference_codes/strong_lensing_tools:/Users/liurongfu/tools
```

- 新增测试：

```text
Bayesian_inference/tests/test_sonnenfeld_reference_comparison_harness.py
```

测试只验证 schema、manifest、error classification，不依赖大型 HDF5。

验收：

- `conda run -n cmass_lens python -m pytest Bayesian_inference/tests/test_sonnenfeld_reference_comparison_harness.py -q` 通过。
- JSON schema 能表达 `passed`、`failed`、`skipped_data_gated`、`not_comparable`。

### Phase 2：参数与单位映射

目标：消除“同名不同义”和“同义不同名”的错误比较。

实现内容：

- 明确 reference 参数顺序：

```text
mu_m5, sigma_m5, beta_m5, xi_m5,
mu_gamma, sigma_gamma, beta_gamma, xi_gamma,
mu_zs, sigma_zs, t_find, la_find
```

- 明确本地参数顺序：

```text
mu5_0, beta5, xi5, sigma5,
mu_gamma_0, beta_gamma, xi_gamma, sigma_gamma,
mu_zs, sigma_zs, theta0, loga
```

- 写一个显式 mapper，不允许靠数组位置隐式转换。
- report 中同时输出 reference theta、本地 theta、参数名和单位。

验收：

- mapper 有单元测试。
- 如果参数名缺失、维度不等于 12、或 variant 不是 `sonnenfeld2024_slacs`，harness 直接拒绝。

### Phase 3：Primitive Oracle

目标：先比较不会被 MC 噪声污染的标量公式。

对比项：

- `pfind(theta_E_est, t_find, 10**la_find)`。
- source-redshift truncated Gaussian draw/mask 条件。
- parent density 分布结构：`dV/dz`、arctan completeness、Schechter 项。
- size relation `mu_r(M_*)` 与 `delta_r`。
- FP prior 默认常数与二维 FP fit：

```text
log10(sigma) = mu + beta * (M_* - mpiv_slacs) + xi * delta_r
```

- FP prior penalty 只对 scatter、`mu`、`beta` 加项。
- finite-fibre cross-section grid 的 out-of-bounds policy。

验收：

- 每个 primitive 有固定输入的 reference/local 对比。
- 容差默认 `rtol=1e-10, atol=1e-12`；若 reference 使用 spline/interp，容差在测试里写明原因。

### Phase 4：Data/Grid Oracle

目标：确认本地 canonical dataset 的数据产品和 reference grid 是同一物理量。

对比项：

- SLACS lens table 字段：`z_d`、`z_s`、`M_*`、`R_e`、`theta_E`、velocity dispersion。
- per-lens `m5_grid` / `dm5drein_grid`。
- per-lens `s2_grid`。
- global finite-fibre `mufibre3_cs_grid(theta_E, gamma)`。
- population sigma grid 的定义：`sigma^2 / 10**m5`。

验收：

- 如果本地 canonical dataset 与 reference grid 同源，要求 shape、axis、unit metadata 和若干固定点数值一致。
- 如果同源 artifact 缺失，report 标记 data-gated，不进入 full posterior exact comparison。

### Phase 5：Population Normalization Oracle

目标：比较 selection normalization，而不是只比较最终 posterior。

实现内容：

- 固定同一组 theta。
- 固定 parent sample 或固定 reference random stream。
- 输出 reference/local：
  - parent sample count；
  - good mask count；
  - mean selection weight；
  - `selection_normalization`；
  - `theta_E`、`theta_E_est`、cross-section、`pfind` 的 summary statistics。

验收：

- 如果本地和 reference 使用同一 parent sample，normalization 应进入严格容差。
- 如果本地使用 deterministic quadrature 而 reference 使用 MC，则报告必须同时给出 convergence sweep，例如 `npop = 1e3, 1e4, 1e5`，不能用单点差异下结论。

### Phase 6：FP Prior Oracle

目标：FP prior 必须可以独立对比。

实现内容：

- 用同一 parent population sample 中 `M_* > 11` 的子样本。
- 比较 `fpfit_mu`、`fpfit_beta`、`fpfit_xi`、`fpfit_scatter`。
- 比较 `fp_prior_log_term`。

验收：

- 当 parent sample 和 sigma grid 相同，FP diagnostics 与 prior penalty 进入严格容差。
- report 中不能只写 prior 总项，必须写四个 fit diagnostics。

### Phase 7：Per-Lens Likelihood Oracle

目标：逐个 lens 比较 likelihood term，定位是单个 lens/grid 问题还是 population normalization 问题。

实现内容：

- 对每个 lens 输出：
  - source-redshift term；
  - mass/size density term；
  - gamma density term；
  - lensing Jacobian；
  - cross-section；
  - discovery probability；
  - velocity-dispersion likelihood；
  - final `log_likelihood_i`。
- 如果 reference 使用 per-lens stochastic importance sample、本地使用 deterministic quadrature，先实现 reference-compatible local evaluator 或固定 reference sample 后再比较。

验收：

- 至少支持 `--lens-index N` 做单 lens 调试。
- 失败 report 指向具体 lens 和具体 term。

### Phase 8：Full Posterior Decomposition

目标：在前面各层通过后，才比较 full posterior。

实现内容：

- 从 `full_inference.hdf5` 或显式 YAML 读取 theta。
- 对每个 theta 输出：

```json
{
  "theta_id": "...",
  "local": {
    "sum_lens_log_likelihood": 0.0,
    "selection_normalization": 0.0,
    "normalization_term": 0.0,
    "fp_prior_log_term": 0.0,
    "total_log_prob": 0.0
  },
  "reference": {
    "sum_lens_log_likelihood": 0.0,
    "selection_normalization": 0.0,
    "normalization_term": 0.0,
    "fp_prior_log_term": 0.0,
    "total_log_prob": 0.0
  },
  "diff": {
    "total_abs": 0.0,
    "normalization_abs": 0.0,
    "fp_prior_abs": 0.0,
    "max_lens_abs": 0.0
  }
}
```

验收：

- 先比较 3 个 theta：reference posterior 中位附近、低概率点、边界附近但未 rejection 的点。
- 通过标准不是预设“必须完全相等”，而是按 Phase 3-7 已证明的可比层给出明确容差。
- 若 full posterior 不通过，报告必须指向前面某一层的失败，不允许只留下一个总差值。

## 推荐命令形态

Audit：

```bash
conda run -n cmass_lens python Bayesian_inference/scripts/compare_sonnenfeld_with_reference.py \
  --reference-root /Users/liurongfu/reference_codes/strong_lensing_tools/papers/slacs_selection \
  --candidate-root /Users/liurongfu/.codex/worktrees/b283/CMASS_lens_project \
  --model sonnenfeld2024_slacs \
  --audit-only \
  --output-dir Bayesian_inference/docs/reports/sonnenfeld_reference_comparison/<timestamp>
```

Primitive/grid/term comparison：

```bash
conda run -n cmass_lens python Bayesian_inference/scripts/compare_sonnenfeld_with_reference.py \
  --reference-root /Users/liurongfu/reference_codes/strong_lensing_tools/papers/slacs_selection \
  --candidate-root /Users/liurongfu/.codex/worktrees/b283/CMASS_lens_project \
  --model sonnenfeld2024_slacs \
  --candidate-config <path-to-sonnenfeld-yaml> \
  --theta-source reference-chain:median \
  --stages primitive,grid,normalization,fp,per-lens,posterior \
  --numba-threads 1 \
  --output-dir Bayesian_inference/docs/reports/sonnenfeld_reference_comparison/<timestamp>
```

## 文件计划

新增：

- `Bayesian_inference/scripts/compare_sonnenfeld_with_reference.py`
- `Bayesian_inference/tests/test_sonnenfeld_reference_comparison_harness.py`
- `Bayesian_inference/docs/reports/sonnenfeld_reference_comparison/<timestamp>/manifest.json`
- `Bayesian_inference/docs/reports/sonnenfeld_reference_comparison/<timestamp>/comparison.json`
- `Bayesian_inference/docs/reports/sonnenfeld_reference_comparison/<timestamp>/summary.md`

可选新增：

- `Bayesian_inference/src/cmass_lens_inference/reference_oracles/sonnenfeld_slacs.py`

只有当 formula adapter 需要被测试复用时，才新增 `reference_oracles` 包；否则先把 wrapper 保持在 `scripts/` 下，避免把外部 reference 语义混入 production package。

## 风险与处理

| 风险 | 处理 |
| --- | --- |
| reference 脚本 top-level 会启动 MCMC | 不直接 import `fit_full.py`；用只执行单点计算的 wrapper 或本地 oracle adapter |
| 缺少 reference grid HDF5 | audit 阶段标记 data-gated；不伪造 full posterior 结论 |
| MC 方差导致误判 | 固定 random stream；做 convergence sweep；优先比较分解项 |
| 本地 deterministic quadrature 与 reference importance sampling 不同 | 先比较积分目标，再比较收敛趋势；必要时实现 reference-compatible local evaluator |
| h-unit 平移混入第一阶段 | 第一阶段只做 `legacy_fixed_kpc/m5` |
| CMASS harness 被误改 | Sonnenfeld comparison 用新脚本和新测试，不改 CMASS main-comparison 科学逻辑 |

## 第一批可执行任务

1. 写 `test_sonnenfeld_reference_comparison_harness.py`，先锁定 JSON schema、manifest、status enum。
2. 写 `compare_sonnenfeld_with_reference.py --audit-only`，只做 artifact manifest。
3. 加入参数 mapper，并测试 reference theta 与 local theta 的 12D 显式转换。
4. 实现 primitive oracle：`pfind`、FP prior defaults、FP OLS、source-z mask。
5. 在有真实 grid artifact 后再推进 grid/normalization/per-lens/full posterior 阶段。

## 完成定义

本计划完成时，应满足：

- `conda run -n cmass_lens python -m pytest Bayesian_inference/tests/test_sonnenfeld_reference_comparison_harness.py -q` 通过。
- `--audit-only` 能生成完整 manifest，并正确标记缺失 artifact。
- 至少 primitive + FP prior comparison 可以在没有大型 HDF5 grid 的情况下运行。
- 有 grid artifact 时，能生成包含 normalization、FP、per-lens 和 full posterior decomposition 的 report。
- report 足以定位数值差异属于哪一层，而不是只报告 total log-prob mismatch。
