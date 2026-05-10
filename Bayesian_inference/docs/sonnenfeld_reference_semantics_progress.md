# Sonnenfeld Reference 语义修正进度

本文档只追加记录执行进度；不记录不可变目标。不可变目标见 `sonnenfeld_reference_semantics_final_target.md`。

## 2026-05-09 计划建立

- 已建立不可修改的最终目标文档：`Bayesian_inference/docs/sonnenfeld_reference_semantics_final_target.md`。
- 本进度文档开始记录后续迭代。
- 下一步：先写失败测试，锁定本地 Sonnenfeld posterior 不允许继续暴露 `proposal_density` / `parent_density / proposal_density` 语义。

## 2026-05-09 TDD red: parent sample 语义

- 新增测试：`test_sonnenfeld_context_uses_reference_parent_sample_not_runtime_proposal`。
- 红灯结果符合预期：当前 `SonnenfeldModelContext` 没有 `parent_sample_zd`，说明 runtime 还没有物理 parent sample 合同。
- 下一步：在 preprocessing 中生成 reference-faithful parent sample arrays，并把 posterior kernel 改为消费这些 arrays。

## 2026-05-09 TDD green: parent sample 语义

- 已在 preprocessing/context/runtime contract 中加入 `parent_sample_zd`、`parent_sample_mstar`、`parent_sample_log_re`、`parent_sample_delta_r`。
- posterior normalization 与 population summary kernel 已改为消费这些 parent samples，不再在 kernel 内构造 ad-hoc proposal 或做 `parent_density / proposal_density` 加权。
- 单测 `test_sonnenfeld_context_uses_reference_parent_sample_not_runtime_proposal` 已转绿。
- 下一步：跑现有 Sonnenfeld 测试，修正仍然绑定旧 proposal / 旧 parent-grid 语义的测试与实现。

## 2026-05-09 runtime 测试回归

- 已修正旧测试对 parent density 的期待：红移因子不再是 `z_d^2`，而是显式传入 reference `comovd(z)^2 dcomovd/dz` 体积因子。
- 已修正 h-unit parent mass bounds 的测试合同：边界来自 reference physical `[9.0, 12.5]` 后做 h-unit 坐标转换，不再来自 `mbar + offset`。
- `python -m pytest Bayesian_inference/tests/test_sonnenfeld_runtime_model.py -q` 已通过。
- 下一步：继续修正 source-redshift 截断、`good` mask、FP summary/default prior 语义。

## 2026-05-09 TDD red: source-z 与 FP 默认 prior

- 新增 `test_sonnenfeld_population_source_redshift_draw_is_nonnegative_truncated`，当前失败值为 `z_s=-0.7`，确认 population 抽样仍是普通 Gaussian。
- 新增 `test_sonnenfeld_population_source_mask_matches_reference_window`，当前失败原因是缺少 reference source-window helper。
- 新增 `test_sonnenfeld_fp_prior_defaults_match_reference_fitpars`，当前失败值为 `fp_fiducial_scatter=0.075`，reference 应为 `0.047`。
- 下一步：实现非负截断 source-z 抽样、`z_s > z_d + 0.05`/`0.05 < z_s < 2.0` mask，并把默认 FP prior 改成 reference `fitpars.py` 数值。

## 2026-05-09 TDD green: source-z、FP 默认 prior 与 sigma proxy

- population source redshift 已改为 reference 的 `z_s >= 0` 截断 Gaussian 抽样，source-window mask 已显式实现 `z_s > z_d + 0.05` 且 `0.05 < z_s < 2.0`。
- FP prior 默认值已改为 reference `fitpars.py` 数值：`fiducial_fpscat=0.047`、`err_fpscat=0.008`、`mu_v_prior=2.341871`、`err_mu_v=0.03`、`beta_v_prior=0.25774`、`err_beta_v=0.03`。
- 删除了 Sonnenfeld population sigma proxy 中的 `max(0.1, ...)` floor；reference 语义是直接使用 `sigma * (1 + 0.0625 N(0,1))`。
- `python -m pytest Bayesian_inference/tests/test_sonnenfeld_runtime_model.py Bayesian_inference/tests/test_model_registry_config.py Bayesian_inference/tests/test_config_profiles_io.py -q` 已通过。
- 下一步：做最终 targeted grep 与工作区差异检查，确认本次 semantic audit 范围内没有残留 proposal-era 热路径。

## 2026-05-09 全量验证

- 首次全量 `Bayesian_inference/tests` 暴露 2 个组件边界失败：generic FP reducer 文案含具体模型 token，Sonnenfeld posterior 边界测试仍期待旧的 generic size/density import。
- 已修正 generic FP reducer 文案，避免 `numba_backend` 携带具体生产模型语义。
- 已修正 Sonnenfeld posterior 边界测试：现在锁定 posterior 不再定义 `_parent_density_for_draw`、`_active_truncation_mass_threshold`、`_size_relation_mean` 这类旧热路径私有实现。
- targeted grep 在 Sonnenfeld posterior/backend 源码中未发现 `proposal_density`、`parent_density / proposal_density`、`max(0.1, ...)`、旧 source-z 普通 Gaussian 抽样或旧 parent-density 私有函数残留。
- `python -m pytest Bayesian_inference/tests -q` 已通过；结果为全量通过，另有 3 个既有 skipped。
