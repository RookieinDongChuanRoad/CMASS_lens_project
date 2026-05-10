# Sonnenfeld Reference Comparison 进度

本文档按追加方式记录 `sonnenfeld_reference_comparison_target.md` 的执行进度。目标文档是约束来源，不在执行过程中修改。

## 2026-05-10 启动

- 目标文档：`Bayesian_inference/docs/sonnenfeld_reference_comparison_target.md`。
- 新建本进度文档，后续每完成一个 phase 都在这里追加记录。
- 执行顺序锁定为目标文档中的 Phase 0 到 Phase 8。
- 第一批执行范围：Phase 0 `Reference Artifact Audit` 与 Phase 1 `Result Schema 与 Subprocess 隔离`。

## 2026-05-10 Phase 0 完成：Reference Artifact Audit

- 新增脚本：`Bayesian_inference/scripts/compare_sonnenfeld_with_reference.py`。
- 已实现 `--audit-only`，只做 artifact manifest，不 import `fit_full.py`，因此不会触发 reference 的 top-level MCMC。
- 真实 reference audit 已运行，输出目录：
  `Bayesian_inference/docs/reports/sonnenfeld_reference_comparison/20260510_phase0_audit/`。
- 真实 audit 状态为 `skipped_data_gated`，原因：
  - 缺少 `reference:fibre_crosssect_grid.hdf5`；
  - 缺少 `reference:slacs_lensing_grids.hdf5`；
  - 缺少 `reference:slacs_jeans_grids.hdf5`；
  - 未传入 `candidate:canonical_dataset`。
- 已确认存在的 reference artifacts：
  - `SLACS_table.cat`；
  - `parent_sample.fits`；
  - `full_inference.hdf5`；
  - `slonly_inference.hdf5`。
- 验证命令：
  `conda run -n cmass_lens python Bayesian_inference/scripts/compare_sonnenfeld_with_reference.py --reference-root /Users/liurongfu/reference_codes/strong_lensing_tools/papers/slacs_selection --candidate-root /Users/liurongfu/.codex/worktrees/b283/CMASS_lens_project --model sonnenfeld2024_slacs --audit-only --output-dir Bayesian_inference/docs/reports/sonnenfeld_reference_comparison/20260510_phase0_audit`。

## 2026-05-10 Phase 1 完成：Result Schema 与 Subprocess 隔离

- 新增测试：`Bayesian_inference/tests/test_sonnenfeld_reference_comparison_harness.py`。
- 测试覆盖：
  - manifest 缺失 artifact 时返回 `skipped_data_gated`，不假装通过；
  - present artifact 记录 `sha256`、`size_bytes` 与 HDF5 dataset schema；
  - CLI `--audit-only` 写出 `manifest.json` 与 `summary.md`。
- JSON status enum 已包含 `passed`、`failed`、`skipped_data_gated`、`not_comparable`。
- 本阶段尚未运行数值 comparison subprocess；已先建立 CLI/schema 隔离边界，后续 phase 在此基础上增加 reference/local evaluator。
- 验证命令：
  `conda run -n cmass_lens python -m pytest Bayesian_inference/tests/test_sonnenfeld_reference_comparison_harness.py -q`。
- 验证结果：`3 passed`。

## 2026-05-10 Phase 2 完成：参数与单位映射

- 在 `Bayesian_inference/scripts/compare_sonnenfeld_with_reference.py` 中加入显式参数顺序：
  - reference order：`mu_m5, sigma_m5, beta_m5, xi_m5, mu_gamma, sigma_gamma, beta_gamma, xi_gamma, mu_zs, sigma_zs, t_find, la_find`；
  - local order：`mu5_0, beta5, xi5, sigma5, mu_gamma_0, beta_gamma, xi_gamma, sigma_gamma, mu_zs, sigma_zs, theta0, loga`。
- 新增 `map_reference_theta_to_local(...)`，只允许通过参数名完成 reference 到本地的 12D 映射。
- 新增测试覆盖：
  - `sigma_m5 -> sigma5`、`beta_m5 -> beta5` 等非同位置映射；
  - 缺失参数或错误长度会在 comparison 前失败。
- 验证命令：
  `conda run -n cmass_lens python -m pytest Bayesian_inference/tests/test_sonnenfeld_reference_comparison_harness.py -q`。
- 验证结果：`5 passed`。

## 2026-05-10 Phase 3 完成：Primitive Oracle

- 在 `Bayesian_inference/scripts/compare_sonnenfeld_with_reference.py` 中新增 primitive stage：
  `--stages primitive`。
- Primitive stage 不依赖大型 HDF5 grid，也不 import reference 的 MCMC 脚本。
- 已比较并通过的 primitive 项：
  - `pfind(theta_E_est, t_find, 10**la_find)`；
  - source-redshift mask：`z_s > z_d + 0.05` 且 `0.05 < z_s < 2.0`；
  - Hyde-Bernardi quadratic size relation；
  - FP prior defaults；
  - 两预测量 FP OLS：`log10(sigma) = mu + beta * (M_* - mpiv_slacs) + xi * delta_r`。
- 真实 primitive report 输出目录：
  `Bayesian_inference/docs/reports/sonnenfeld_reference_comparison/20260510_phase3_primitive/`。
- 验证命令：
  `conda run -n cmass_lens python -m pytest Bayesian_inference/tests/test_sonnenfeld_reference_comparison_harness.py -q`。
- 验证结果：`7 passed`。
- 运行命令：
  `conda run -n cmass_lens python Bayesian_inference/scripts/compare_sonnenfeld_with_reference.py --reference-root /Users/liurongfu/reference_codes/strong_lensing_tools/papers/slacs_selection --candidate-root /Users/liurongfu/.codex/worktrees/b283/CMASS_lens_project --model sonnenfeld2024_slacs --stages primitive --output-dir Bayesian_inference/docs/reports/sonnenfeld_reference_comparison/20260510_phase3_primitive`。
- Primitive comparison 结果：`passed`。

## 2026-05-10 Phase 4 完成：Data/Grid Oracle Gate

- 新增 data/grid stage：`--stages grid`。
- 当前实现先执行 artifact gate：只有 reference grid 与本地 canonical dataset 都存在时，才允许进入后续数值 grid oracle。
- 真实 data/grid report 输出目录：
  `Bayesian_inference/docs/reports/sonnenfeld_reference_comparison/20260510_phase4_grid/`。
- 真实 data/grid 结果：`skipped_data_gated`。
- 阻塞原因：
  - 缺少 `reference:fibre_crosssect_grid.hdf5`；
  - 缺少 `reference:slacs_lensing_grids.hdf5`；
  - 缺少 `reference:slacs_jeans_grids.hdf5`；
  - 缺少 `candidate:canonical_dataset`。
- 已搜索路径：
  - `/Users/liurongfu/reference_codes/strong_lensing_tools`；
  - `/Users/liurongfu/.codex/worktrees/b283/CMASS_lens_project`；
  - `/Users/liurongfu/Work/CMASS_lens_project`。
- 未找到可直接用于 Phase 4 的 paper-native Sonnenfeld canonical dataset 或 reference grid HDF5。
- 验证命令：
  `conda run -n cmass_lens python -m pytest Bayesian_inference/tests/test_sonnenfeld_reference_comparison_harness.py -q`。
- 验证结果：`8 passed`。
- 运行命令：
  `conda run -n cmass_lens python Bayesian_inference/scripts/compare_sonnenfeld_with_reference.py --reference-root /Users/liurongfu/reference_codes/strong_lensing_tools/papers/slacs_selection --candidate-root /Users/liurongfu/.codex/worktrees/b283/CMASS_lens_project --model sonnenfeld2024_slacs --stages grid --output-dir Bayesian_inference/docs/reports/sonnenfeld_reference_comparison/20260510_phase4_grid`。
- 后续 Phase 5-8 依赖 Phase 4 的 grid/canonical data gate；在缺失工件补齐前，不能做 normalization、FP prior、per-lens likelihood 或 full posterior 的严格数值 comparison。
