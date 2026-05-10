# Sonnenfeld 2024 SLACS Sigma-Star Gamma Model Progress

目标文档：`Bayesian_inference/docs/plans/2026-05-10-sonnenfeld2024-slacs-sigma-star-gamma-goal.md`

执行原则：

- 按目标文档中的 Task 顺序推进。
- 每个 Task 完成后，只以追加方式记录本文件。
- 所有测试和实现命令使用 `cmass_lens` 环境。
- 原始目标文档保持只读，不在实现过程中修改。

## 进度日志

### 2026-05-10 Task 0: 初始化执行环境和进度记录

状态：完成。

记录：

- 当前 worktree 起点为 detached HEAD `fd62cde`。
- 已创建并切换到实现分支 `codex/sonnenfeld-sigma-star-gamma`。
- 已确认目标文档处于只读状态，后续实现不修改该目标文档。
- 已创建本进度文档；后续 Task 完成后继续在本文件末尾追加记录。

### 2026-05-10 Task 1: 新增 registry/schema 红灯测试

状态：完成。

变更：

- 在 `Bayesian_inference/tests/test_model_registry_config.py` 中新增 `test_model_registry_exposes_sonnenfeld_sigma_star_gamma_variants`。
- 在 `Bayesian_inference/tests/test_component_specs.py` 中新增 `test_sonnenfeld_sigma_star_gamma_assembly_uses_component_sources`。

验证：

- 命令：`conda run -n cmass_lens python -m pytest Bayesian_inference/tests/test_model_registry_config.py Bayesian_inference/tests/test_component_specs.py -q`
- 结果：预期失败。
- 失败点 1：`get_model_definition("sonnenfeld2024_slacs_sigma_star_gamma")` 仍报 unsupported model。
- 失败点 2：`cmass_lens_inference.models.sonnenfeld2024_slacs_sigma_star_gamma` 模块尚不存在。

结论：

- RED 测试有效，失败原因正是新模型 registry 和并列模型包尚未实现。

### 2026-05-10 Task 2: 新增并列模型 assembly 包

状态：完成。

变更：

- 新增 `Bayesian_inference/src/cmass_lens_inference/models/sonnenfeld2024_slacs_sigma_star_gamma/__init__.py`。
- 新增 `Bayesian_inference/src/cmass_lens_inference/models/sonnenfeld2024_slacs_sigma_star_gamma/assembly.py`。
- 新模型包定义 paper-native 和 h-unit 两个 `ModelSpec` 入口：
  - `sonnenfeld2024_slacs_sigma_star_gamma`
  - `sonnenfeld2024_slacs_sigma_star_gamma_hunit`
- 新模型参数 schema 为 11D，并使用 `beta_sigma_star_gamma` 代替原 Sonnenfeld 的 `beta_gamma` / `xi_gamma`。
- gamma relation component 使用现有 `sigma_star_linear_gamma_component`。

验证：

- 命令：`conda run -n cmass_lens python -m pytest Bayesian_inference/tests/test_component_specs.py::test_sonnenfeld_sigma_star_gamma_assembly_uses_component_sources -q`
- 结果：通过。

说明：

- registry 尚未注册新模型；Task 1 中的 registry 红灯会在 Task 5 处理。

### 2026-05-10 Task 3: 新增 sigma-star-gamma posterior

状态：完成。

变更：

- 新增 `Bayesian_inference/src/cmass_lens_inference/models/sonnenfeld2024_slacs_sigma_star_gamma/posterior.py`。
- 以现有 Sonnenfeld posterior 为结构基础，但新文件独立归属新模型包。
- 将 theta unpack 从原 Sonnenfeld 12D 改为 11D。
- 将 normalization population draw 中的 gamma 均值改为：
  `mu_gamma_0 + beta_sigma_star_gamma * (mstar - log10(2 pi) - 2 log_re - 9)`。
- 将 per-lens likelihood 中的 gamma 均值改为基于 `mstar_grid` 和 `log_re_obs` 的 sigma-star 形式。
- 新 posterior 的 diagnostic kernel 名称改为 `sonnenfeld2024_slacs_sigma_star_gamma`。

验证：

- 命令：`conda run -n cmass_lens python -m py_compile Bayesian_inference/src/cmass_lens_inference/models/sonnenfeld2024_slacs_sigma_star_gamma/posterior.py`
- 结果：通过。

说明：

- 新 posterior 尚未接入 registry；完整 log-prob runtime 验证将在 Task 7 执行。

### 2026-05-10 Task 4: 新增 runtime wrapper

状态：完成。

变更：

- 新增 `Bayesian_inference/src/cmass_lens_inference/models/sonnenfeld2024_slacs_sigma_star_gamma/runtime.py`。
- wrapper 复用现有 `sonnenfeld2024_slacs.runtime` 的 canonical context builder 和 data spec。
- 未新增数据准备逻辑，保持新模型只改变 posterior gamma relation 的边界。

验证：

- 命令：`conda run -n cmass_lens python -c "from cmass_lens_inference.models.sonnenfeld2024_slacs_sigma_star_gamma import runtime; adapter = runtime.get_runtime_adapter(); print(adapter.data_spec.normalization_samples_field)"`
- 结果：输出 `base_normals`，说明 wrapper 可导入并返回有效 runtime adapter。

### 2026-05-10 Task 5: 注册并导出新模型

状态：完成。

变更：

- 修改 `Bayesian_inference/src/cmass_lens_inference/models/__init__.py`，导出并列模型包和 runtime wrapper。
- 修改 `Bayesian_inference/src/cmass_lens_inference/model_registry.py`，新增两个 concrete model name：
  - `sonnenfeld2024_slacs_sigma_star_gamma`
  - `sonnenfeld2024_slacs_sigma_star_gamma_hunit`
- 两个 registry 分支都绑定到新包的 `assembly`、`runtime` 和 `posterior.log_prob`。
- unsupported model 错误信息已加入两个新 model name。

验证：

- 命令：`conda run -n cmass_lens python -m pytest Bayesian_inference/tests/test_model_registry_config.py Bayesian_inference/tests/test_component_specs.py -q`
- 结果：通过。

结论：

- Task 1 的 registry/schema 红灯已转绿。

### 2026-05-10 Task 6: 新增配置和配置测试

状态：完成。

变更：

- 新增 `Bayesian_inference/configs/sonnenfeld2024_slacs_sigma_star_gamma.yaml`。
- 新增 `Bayesian_inference/configs/sonnenfeld2024_slacs_sigma_star_gamma_hunit.yaml`。
- 修改 `Bayesian_inference/tests/test_config_profiles_io.py`，新增真实 repo YAML 的加载合同测试。
- 两个新配置均显式设置 `fp_prior.enabled: true`。
- paper-native 配置使用 `legacy_fixed_kpc` / `m5` / fixed dataset。
- h-unit 配置使用 `h_units_v1` / `m5_hinvkpc` / hunit dataset。
- 两个配置均使用 11D 参数 schema，并排除 `beta_gamma` / `xi_gamma`。

验证：

- 命令：`conda run -n cmass_lens python -m pytest Bayesian_inference/tests/test_config_profiles_io.py -q`
- 结果：通过。

### 2026-05-10 Task 7: 新增 synthetic runtime 测试

状态：完成。

变更：

- 新增 `Bayesian_inference/tests/test_sonnenfeld_sigma_star_gamma_runtime_model.py`。
- 测试覆盖 paper-native 新模型的 synthetic canonical fixture `log_prob`。
- 测试覆盖 h-unit 新模型的 synthetic canonical fixture `log_prob`。
- 测试覆盖新模型短 emcee run 的 chain shape 为 `(2, 24, 11)`。
- 测试确认原 `sonnenfeld2024_slacs` 和 `sonnenfeld2024_slacs_hunit` 仍为 12D。

验证：

- 命令：`conda run -n cmass_lens python -m pytest Bayesian_inference/tests/test_sonnenfeld_runtime_model.py Bayesian_inference/tests/test_sonnenfeld_sigma_star_gamma_runtime_model.py -q`
- 结果：通过。

### 2026-05-10 Task 8: 最终聚焦验证

状态：完成。

验证：

- 命令：`conda run -n cmass_lens python -m pytest Bayesian_inference/tests/test_model_registry_config.py Bayesian_inference/tests/test_component_specs.py Bayesian_inference/tests/test_config_profiles_io.py Bayesian_inference/tests/test_sonnenfeld_runtime_model.py Bayesian_inference/tests/test_sonnenfeld_sigma_star_gamma_runtime_model.py -q`
- 结果：通过。

数据可用性检查：

- 命令：`ls -l /Users/liurongfu/Work/CMASS_lens_project/data/external/inference_dataset_sonnenfeld2024_slacs_m5_fixed_v1.hdf5 /Users/liurongfu/Work/CMASS_lens_project/data/external/inference_dataset_sonnenfeld2024_slacs_m5_hunits_v1.hdf5`
- 结果：两个正式 `data/external` HDF5 文件当前都不存在。

结论：

- 新模型的代码合同、registry、配置加载和 synthetic runtime 验证均已通过。
- 真实生产配置能通过 YAML/config schema 加载，但实际启动真实 inference 前仍需要把对应 canonical HDF5 放到正式 `data/external` 路径。

### 2026-05-10 Dataset check: Sonnenfeld h-units canonical input

状态：完成。

检查目的：

- 回答 `inference_dataset_sonnenfeld2024_slacs_m5_hunits_v1.hdf5` 在当前工作区内应如何准备。
- 区分“已有可直接运行的数据文件”和“需要从 raw/staging artifacts 重新生成的 h-units Sonnenfeld 数据文件”。

当前工作区事实：

- `data/external/inference_dataset_sonnenfeld2024_slacs_m5_fixed_v1.hdf5` 存在。
- `data/external/inference_dataset_devauc_slit_m5_hunits_v1.hdf5` 存在，但这是 CMASS/devauc h-units 数据，不是 Sonnenfeld/SLACS 数据。
- `data/external/inference_dataset_sonnenfeld2024_slacs_m5_hunits_v1.hdf5` 当前不存在。
- `data/raw/SLACS_table.cat` 存在，可作为 SLACS observation builder 的 raw catalog 输入。
- 当前 worktree 内没有 `fibre_crosssect_grid.hdf5`；可复用的现有 fibre cross-section grid 位于主仓库 staging：
  `/Users/liurongfu/Work/CMASS_lens_project/outputs/_staging/sonnenfeld_reference_comparison_20260510_133000/reference_root/fibre_crosssect_grid.hdf5`。

语义检查：

- fixed Sonnenfeld canonical 文件 metadata 为 `unit_convention=legacy_fixed_kpc`、`mass_definition_label=m5`、`lensing_cross_section.source=mufibre3_cs_grid`，并包含 `population_sigma_unit`。
- CMASS/devauc h-units canonical 文件 metadata 为 `unit_convention=h_units_v1`、`mass_definition_label=m5_hinvkpc`、`lensing_cross_section.source=separable_cs_over_theta_ein`，不应当作为 Sonnenfeld h-units 输入。
- 当前 `prepare_dataset.io.slacs_observations` 的 SLACS 专用 observation 和 population sigma builder 仍硬编码 `legacy_fixed_kpc` / `m5`；因此不能只调用 `--build-slacs-observation-hdf5` 和 `--build-slacs-population-sigma-hdf5` 得到目标 h-units canonical 文件。

准备路径结论：

- 不能把 fixed canonical 文件简单复制或改名为 `m5_hunits_v1`。
- 正确路径应当生成 h-units observation HDF5、h-units population sigma table，并用 Sonnenfeld finite-fibre cross-section grid 写出 canonical dataset。
- 当前不改代码也可以通过“fixed SLACS observation -> generic h-units processor -> direct h-units population sigma builder -> canonical writer”的路径准备。
- 更长期可维护的做法是把 SLACS 专用 builder 扩展为显式接收 `unit_convention` / `h_ref`，让 h-units 数据准备成为 CLI 的一等入口。

### 2026-05-10 Dataset build: `inference_dataset_sonnenfeld2024_slacs_m5_hunits_v1.hdf5`

状态：完成。

生成步骤：

- 从 `data/raw/SLACS_table.cat` 生成 fixed-kpc SLACS observation HDF5：
  `data/_staging/sonnenfeld2024_slacs_hunit/observations_SLACS_deV_with_mass_grids_fixed_m5.hdf5`。
- 用通用 `prepare_dataset` processor 转换为 h-units observation HDF5：
  `data/_staging/sonnenfeld2024_slacs_hunit/observations_SLACS_deV_with_mass_grids_fixed_m5.updated.hdf5`。
- 直接调用 `prepare_dataset.io.sigma_tables.build_sigma_unit_table(...)` 生成 h-units population sigma table：
  `data/_staging/sonnenfeld2024_slacs_hunit/slacs_population_sigma_unit_m5_hunits_v1.h5`。
- 用 canonical writer 和 Sonnenfeld finite-fibre cross-section grid 生成目标 canonical dataset：
  `data/external/inference_dataset_sonnenfeld2024_slacs_m5_hunits_v1.hdf5`。
- 同步一份到当前 hunit YAML 配置引用的正式路径：
  `/Users/liurongfu/Work/CMASS_lens_project/data/external/inference_dataset_sonnenfeld2024_slacs_m5_hunits_v1.hdf5`。

关键验证：

- h-units observation 转换结果：`groups=59 m5=59 dm5=59 s2=59 failures=0`。
- canonical metadata：
  - `unit_convention=h_units_v1`
  - `h_ref=0.7`
  - `profile_name=devauc`
  - `mass_definition_label=m5_hinvkpc`
  - `lensing_cross_section.source=mufibre3_cs_grid`
- canonical shape：
  - `n_lenses=59`
  - `lensing_mass_grids/log_enclosed_mass_grid.shape=(59, 17)`
  - `velocity_dispersion_grids/per_lens_s2/s2_grid.shape=(59, 17)`
  - `velocity_dispersion_grids/population_sigma_unit/s_unit_grid.shape=(17, 21, 21)`
- worktree copy 和正式配置路径 copy 的 SHA256 一致：
  `979c27575b4690570555764143703443767b7172b5775d4fdfadcdba078fc87c`。
- 使用降采样 integration 设置做 runtime smoke test：
  - `sonnenfeld2024_slacs_hunit`: 12D，有限 `log_prob=954.9682669947777`，`normalization=0.008779708559724448`。
  - `sonnenfeld2024_slacs_sigma_star_gamma_hunit`: 11D，有限 `log_prob=951.3961743677944`，`normalization=0.00797519244529383`。

结论：

- 目标 h-units Sonnenfeld/SLACS canonical dataset 已准备完成。
- 当前两个 hunit YAML 配置沿自身 `data.inference_dataset_path` 可找到该文件，并通过真实 runtime smoke test。

### 2026-05-10 Production fit: `sonnenfeld2024_slacs_sigma_star_gamma_hunit`

状态：完成。

启动配置：

- 配置文件：`Bayesian_inference/configs/sonnenfeld2024_slacs_sigma_star_gamma_hunit.yaml`。
- 模型名：`sonnenfeld2024_slacs_sigma_star_gamma_hunit`。
- unit convention：`h_units_v1`。
- 输入 canonical dataset：
  `/Users/liurongfu/Work/CMASS_lens_project/data/external/inference_dataset_sonnenfeld2024_slacs_m5_hunits_v1.hdf5`。
- 输出 run directory：
  `/Users/liurongfu/Work/CMASS_lens_project/outputs/devauc/20260510_192633_devauc_sonnenfeld2024-slacs-sigma-star-gamma-hunit`。
- sampling 设置：`n_walkers=24`、`n_steps=10000`、`burn_in=2000`、`n_dim=11`。
- integration 设置：`normalization_samples=100000`。
- 并行策略：`kernel_only`，`workers=12`。

关键结果：

- CLI 退出码为 0，`run_result.json` 记录 `status=completed`。
- `completed_steps=10000`，`checkpoint_step=10000`。
- `acceptance_fraction_mean=0.396075`。
- `chain.h5` 的 emcee backend iteration 为 `10000`。
- chain shape 为 `(10000, 24, 11)`。
- log-prob shape 为 `(10000, 24)`。
- `outputs/devauc/latest` 指向本次 run id：
  `20260510_192633_devauc_sonnenfeld2024-slacs-sigma-star-gamma-hunit`。

参数顺序：

- `mu5_0`
- `beta5`
- `xi5`
- `sigma5`
- `mu_gamma_0`
- `beta_sigma_star_gamma`
- `sigma_gamma`
- `mu_zs`
- `sigma_zs`
- `theta0`
- `loga`

结论：

- h-units 的 `sonnenfeld2024_slacs_sigma_star_gamma` 生产拟合已经完整落盘。
- 该 run 的核心 runtime contract 与目标语义一致：`model.name=sonnenfeld2024_slacs_sigma_star_gamma_hunit`、`unit_convention=h_units_v1`、`mass_definition_label=m5_hinvkpc`、`gamma_distribution=sigma_star_dependent`。

### 2026-05-10 Posterior diagnostics: `sonnenfeld2024_slacs_sigma_star_gamma_hunit`

状态：完成。

新增诊断能力：

- `Posterior_predictive_test` 的 Sonnenfeld predictive registry 原本只暴露
  `sonnenfeld2024_slacs` 和 `sonnenfeld2024_slacs_hunit`。
- 已将 `sonnenfeld2024_slacs_sigma_star_gamma` 和
  `sonnenfeld2024_slacs_sigma_star_gamma_hunit` 接入同一个 Sonnenfeld
  predictive adapter。
- adapter 内部按 model name 区分 theta 语义：
  - 原始 Sonnenfeld：12D，`gamma = mu_gamma_0 + beta_gamma * (M_* - pivot) + xi_gamma * delta_r + scatter`。
  - sigma-star-gamma：11D，`gamma = mu_gamma_0 + beta_sigma_star_gamma * (log Sigma_* - 9) + scatter`。
- 同时修复 Sonnenfeld predictive payload 的输出合同：`sigma_latent` 现在显式写出
  `"sigma"`，满足 generic PPC writer 对 `replicated_statistics.npz` 的要求。

生成的 run-root posterior artifacts：

- `posterior_corner.png`
- `posterior_corner_result.json`
- `mcmc_trace.png`
- `mcmc_diagnostics.json`

MCMC 诊断结果：

- burn-in：`2000`。
- post-burn posterior samples：`192000`。
- `acceptance_fraction_mean=0.396075`。
- `emcee` integrated autocorrelation time 计算成功，无错误。
- walker split-Rhat 最大值约 `1.033357257187879`。

生成的 PPC / trend diagnostics artifacts：

- `ppc/ppc_overview.png`
- `ppc/ppc_summary.json`
- `ppc/replicated_statistics.npz`
- `ppc/run_manifest.json`
- `ppc/fig8_like.png`
- `ppc/fig8_like_summary.json`
- `ppc/fig8_like_curves.npz`
- `ppc/gamma_vs_sigma_star.png`
- `ppc/gamma_vs_sigma_star_summary.json`
- `ppc/gamma_vs_sigma_star_curves.npz`
- `ppc/gamma_vs_logre_kpc.png`
- `ppc/gamma_vs_logre_kpc_summary.json`
- `ppc/gamma_vs_logre_kpc_curves.npz`
- `ppc/gamma_vs_delta_r.png`
- `ppc/gamma_vs_delta_r_summary.json`
- `ppc/gamma_vs_delta_r_curves.npz`

PPC / trend 诊断设置：

- posterior draws：`512`，从 post-burn chain 随机抽样。
- parent sample size：`10000`。
- mass bins：`19`，范围 `[10.15, 12.05]`。
- predictive backend：`numba_sonnenfeld_parent`。
- predictive schema：`sonnenfeld2024_slacs_ppt_diagnostics_v1`。
- model metadata：`model_name=sonnenfeld2024_slacs_sigma_star_gamma_hunit`、`gamma_mode=sigma_star_dependent`。
- `replicated_statistics.npz` 关键 shape：
  - `theta_sample_theta_ein.shape=(512, 23)`
  - `sigma_sample_sigma.shape=(512, 7)`
- `fig8_like_curves.npz` 的 `mass_bin_centers.shape=(19,)`。

验证命令摘要：

- `conda run -n cmass_lens --no-capture-output env PYTHONPATH=$PWD/Bayesian_inference/src:$PWD/Posterior_predictive_test/src python -m pytest Posterior_predictive_test/tests/test_predictive_registry.py -q`
- 结果：`12 passed`。
- 低成本 real-run smoke：`n_posterior_draws=4`、`parent_sample_size=64`，成功写出 staging diagnostics。
- 正式 diagnostics：`n_posterior_draws=512`、`parent_sample_size=10000`，成功写出 run-local `ppc/` 目录。

结论：

- h-units sigma-star-gamma Sonnenfeld run 已完成 posterior corner、MCMC chain diagnostic、PPC overview 和 Fig. 8-like/gamma trend diagnostics。
- 本次 posterior-predictive 诊断使用 512 个 posterior draws；不是 full 192000-draw post-burn chain。
