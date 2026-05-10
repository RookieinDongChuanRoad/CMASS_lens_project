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

## 2026-05-10 数据/Grid 生成与 Phase 4 重新运行

- 稳定 staging 目录：
  `/Users/liurongfu/Work/CMASS_lens_project/outputs/_staging/sonnenfeld_reference_comparison_20260510_133000/`。
- 已补齐输入工件：
  - `reference_root/fibre_crosssect_grid.hdf5`：复用已存在的 paper-native finite-fibre grid；
  - `reference_root/slacs_jeans_grids.hdf5`：用 external reference `make_slacs_jeans_grids.py` 重新生成；
  - `reference_root/slacs_lensing_grids.hdf5`：`m5_grid` 与 `dm5drein_grid` 用 reference 公式生成，`mufibre*_cs_grid` 用 global fibre grid 插值生成并写入 surrogate metadata；
  - `candidate/inference_dataset_sonnenfeld2024_slacs_m5_fixed_v1.hdf5`：复用已存在的 paper-native canonical dataset。
- 重新运行 Phase 4 输出目录：
  `Bayesian_inference/docs/reports/sonnenfeld_reference_comparison/20260510_phase4_grid_generated_v2/`。
- Phase 4 结果：`failed`。
- 失败项：
  - `m5_grid`：最大绝对差约 `2.643e-05 dex`，说明 mass-grid 仍有小的生成路径/常数差异；
  - `dm5drein_grid`：最大绝对差约 `0.715`，不是容差问题，而是 reference 语义为 `dm5/dR_ein[kpc]`，本地 canonical 字段为 `dmass/dtheta_E[arcsec]`。
- 已通过项：
  - SLACS table 字段对 canonical `/lenses`；
  - `s2_grid`；
  - global `mufibre3_cs_grid(theta_E,gamma)`；
  - population sigma-unit schema。
- 重要结论：Phase 4 解除了缺失工件 gate，但暴露出真实 grid/Jacobian 语义差异；因此后续 full posterior 不能标成 strict reference pass。

## 2026-05-10 Phase 5 完成：Population Normalization Oracle

- 输出目录：
  `Bayesian_inference/docs/reports/sonnenfeld_reference_comparison/20260510_phase5_8_generated/`。
- Phase 5 状态：`not_comparable`。
- 本地 decomposition 已可运行：
  - parent sample count：`1000`；
  - selection normalization：`0.03946471741805178`；
  - `normalization_mc_numba` 与 `population_summary_mc_numba` 的 normalization 内部一致，绝对差 `0.0`。
- 不可严格对比原因：
  - supplied reference tree 缺少 `rein_grid.hdf5`、`sigma2_grid.hdf5`、`mz_inference.hdf5`；
  - external `fit_full.py` 的 normalization 依赖这些 artifacts 和 reference 随机流。

## 2026-05-10 Phase 6 完成：FP Prior Oracle

- 输出目录：
  `Bayesian_inference/docs/reports/sonnenfeld_reference_comparison/20260510_phase5_8_generated/`。
- Phase 6 状态：`not_comparable`。
- 本地 FP diagnostics 已输出：
  - `fpfit_mu = 2.3078135617031843`；
  - `fpfit_beta = 0.3293272060875592`；
  - `fpfit_xi = -0.2103267434044116`；
  - `fpfit_scatter = 0.055655674942292434`；
  - `fp_prior_log_term = -4.076783136083912`。
- 本地 FP diagnostics 全部 finite，说明 Sonnenfeld posterior 当前确实把 FP prior term 接入了 decomposition。
- 不可严格对比原因：
  - reference `full_inference.hdf5` 只保存 FP blob，不保存足够的 parent-population replay state；
  - exact replay 仍依赖缺失的 reference artifacts 和随机流。

## 2026-05-10 Phase 7 完成：Per-Lens Likelihood Oracle

- 输出目录：
  `Bayesian_inference/docs/reports/sonnenfeld_reference_comparison/20260510_phase5_8_generated/`。
- Phase 7 状态：`not_comparable`。
- 已支持 `--lens-index`，本次使用 `--lens-index 0`。
- 关键发现：
  - external reference per-lens likelihood 读取 `slacs_lensing_grids.hdf5/<lens>/mufibre3_cs_grid(gamma)`；
  - 当前本地 likelihood 使用 global `lensing_cross_section.cross_section_grid(theta_E,gamma)`；
  - 因此 per-lens cross-section term 不是同一条数值路径，不能做 strict per-lens likelihood replay。
- 后续若要 Phase 7 strict pass，需要本地 per-lens likelihood 支持 reference-compatible `cs_lens_splines`，或明确证明 global grid 在每个观测 `theta_E` 上等价。

## 2026-05-10 Phase 8 完成：Full Posterior Decomposition

- 输出目录：
  `Bayesian_inference/docs/reports/sonnenfeld_reference_comparison/20260510_phase5_8_generated/`。
- Phase 8 状态：`not_comparable`。
- 已比较 3 个 reference-chain theta：
  - `max_logp`；
  - `low_logp`；
  - `theta0_high`。
- 本地 decomposition 与 production `log_prob` 内部完全一致：
  - 3 个 theta 的 `local_parts_vs_direct_abs` 最大值为 `0.0`。
- local total 与 stored reference chain logp 的差异：
  - `max_logp`：约 `13.175`；
  - `low_logp`：约 `10.831`；
  - `theta0_high`：约 `7.671`。
- 这些差异不能直接解释为单一 posterior bug，因为 Phase 4/7 已发现 grid/Jacobian 与 per-lens cross-section 路径差异，且 reference chain 不保存 normalization/per-lens decomposition。

## 2026-05-10 运行异常记录

- 曾启动 external reference `make_slacs_lensing_grids.py` 的完整 per-lens finite-fibre 版本。
- 该任务运行约 30 分钟后仍在第一个重计算阶段，CPU 约 `100%`。
- 启动后发现最初 staging 目录从可见目录树消失，进程仍持有已 unlink 的 cwd/output 文件描述符；因此即使自然完成，也大概率无法从路径回收生成的 HDF5。
- 按用户运行控制红线，未私自 kill 该进程；已在对话中请求用户确认是否允许终止这个无效长任务。
