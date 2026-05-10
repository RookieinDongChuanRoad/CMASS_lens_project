# Posterior Predictive Test 模型自适应重构进度记录

> 本文件只记录执行进度，独立于最终目标文档
> `Posterior_predictive_test/model_adaptive_ppt_refactor_plan.md`。
> 最终目标文档建立后视为不可修改；本文件后续只允许追加新记录，不回改历史记录。

## 2026-05-09 初始化

### 本轮执行边界

- 工作区：`/Users/liurongfu/.codex/worktrees/b283/CMASS_lens_project`
- 运行环境：`cmass_lens`
- 最终目标文档：`Posterior_predictive_test/model_adaptive_ppt_refactor_plan.md`
- 进度文档：`Posterior_predictive_test/model_adaptive_ppt_refactor_progress.md`

### 本轮目标

1. 不修改最终目标文档。
2. 建立 append-only 进度记录。
3. 先做最小可交付切片：
   - 增加 PPT 侧 predictive contract / registry 的测试约束；
   - 让 CMASS diagnostics 通过 registry/contract 入口构建 canonical PPC context；
   - 保持现有 CMASS diagnostics 行为不变；
   - 对不支持 predictive diagnostics 的模型给出清晰错误。

### 风险控制

- 当前 worktree 已有多处未提交改动，本轮不回滚、不重排、不清理与本任务无关的文件。
- 本轮只触碰 `Posterior_predictive_test` 相关文件，除非实现确实需要连接上游 model interface。
- 测试命令必须通过 `conda run -n cmass_lens ...` 执行。

## 2026-05-09 TDD 红灯：predictive registry

### 新增测试

- `Posterior_predictive_test/tests/test_predictive_registry.py`

### 执行命令

```bash
conda run -n cmass_lens pytest -q Posterior_predictive_test/tests/test_predictive_registry.py
```

### 当前结果

- 状态：失败，符合预期。
- 失败原因：
  - `lensing_posterior_predictive.registry` 尚不存在；
  - `_build_ppc_context()` 仍未通过 `get_predictive_definition()` dispatch；
  - 当前 context builder 仍包含 `runtime_config.model.name != "cmass"` 的硬编码分支。

## 2026-05-09 最小实现：PPT-local predictive registry

### 修改文件

- `Posterior_predictive_test/src/lensing_posterior_predictive/interfaces.py`
- `Posterior_predictive_test/src/lensing_posterior_predictive/registry.py`
- `Posterior_predictive_test/src/lensing_posterior_predictive/adapters/__init__.py`
- `Posterior_predictive_test/src/lensing_posterior_predictive/adapters/cmass.py`
- `Posterior_predictive_test/src/lensing_posterior_predictive/predictive.py`

### 实现内容

- 新增 `PredictiveDefinition`，记录模型名、backend、支持的 diagnostics、
  外部输入需求、artifact schema 版本和 context builder。
- 新增 `get_predictive_definition(model_name)`：
  - `cmass` 返回当前 Numba shared-parent diagnostics 的薄定义；
  - 其他模型抛出 `UnsupportedPredictiveModelError`，避免误走 CMASS fallback。
- 将 canonical `_build_ppc_context()` 从 CMASS 硬编码分支改为通过 predictive registry
  构建上下文。
- 将 canonical CMASS observation record reconstruction 移入 `adapters/cmass.py`。

### 定点验证

```bash
conda run -n cmass_lens pytest -q Posterior_predictive_test/tests/test_predictive_registry.py
```

结果：`3 passed`。

## 2026-05-09 回归验证：PPT 测试集

### 执行命令

```bash
conda run -n cmass_lens pytest -q Posterior_predictive_test/tests
```

### 当前结果

- 状态：通过。
- 证据：命令退出码为 `0`，pytest 输出为全量通过点阵。

### 本轮剩余限制

- 本轮尚未移动 Numba shared-parent diagnostics 热路径；当前只完成 registry /
  context-building 的最小切片。
- `Posterior_predictive_test/src/lensing_posterior_predictive/predictive.py` 中仍有
  CMASS-only Numba kernel 和 CMASS posterior helper import。后续 Phase 1 的下一步
  应继续把该 kernel 迁出 generic workflow 文件。
- `--sigma-table` 仍是 CLI 层全局必填；model-declared external input 目前只记录在
  `PredictiveDefinition` 中，尚未驱动 CLI 参数解析。

## 2026-05-10 Phase 1 完成：CMASS predictive core 迁出 generic workflow

### 本阶段目标

- 按最终目标文档 Phase 1，将 CMASS-only posterior predictive 生成与 diagnostics
  热路径从通用 `predictive.py` 中迁出。
- 保留当前生产路径的 Numba backend，不改变 CMASS diagnostics 的数值合同。
- 让通用 workflow 只负责运行目录读取、posterior draw 选择、artifact 写入和 registry
  dispatch；模型相关的 posterior 参数解包、selection kernel 和 diagnostics reducer 由
  CMASS adapter 持有。

### 修改文件

- `Posterior_predictive_test/src/lensing_posterior_predictive/interfaces.py`
- `Posterior_predictive_test/src/lensing_posterior_predictive/adapters/cmass.py`
- `Posterior_predictive_test/src/lensing_posterior_predictive/predictive.py`
- `Posterior_predictive_test/tests/test_predictive_registry.py`
- `Posterior_predictive_test/tests/test_posterior_predictive.py`

### 实现内容

- `PredictiveDefinition` 增加 `run_diagnostics` hook，使 diagnostics 执行入口成为
  model-owned contract。
- 将 CMASS Numba shared-parent diagnostics kernel、posterior 参数解包依赖、sigma table
  Numba array bridge 和 population-bin reducer 移入 `adapters/cmass.py`。
- `run_posterior_diagnostics()` 改为通过 `get_predictive_definition(runtime_config.model.name)`
  获取 active model 的 diagnostics hook，并使用 hook 返回的 backend/schema metadata。
- `predictive.py` 不再直接 import `cmass_lens_inference.models.cmass.posterior` 或
  Numba primitive kernel。
- 旧的 reducer / sigma table 单元测试改为从 `adapters.cmass` 导入 CMASS 私有 helper，
  避免继续把 generic workflow 文件当作 CMASS backend 的承载层。

### 验证

```bash
conda run -n cmass_lens pytest -q Posterior_predictive_test/tests/test_predictive_registry.py
```

结果：`5 passed`。

```bash
conda run -n cmass_lens pytest -q Posterior_predictive_test/tests/test_posterior_predictive.py Posterior_predictive_test/tests/test_predictive_registry.py
```

结果：通过，pytest 输出为全量通过点阵。

```bash
conda run -n cmass_lens pytest -q Posterior_predictive_test/tests
```

结果：通过，pytest 输出为全量通过点阵。

### 剩余未做事项

- CMASS predictive core 目前位于 PPT 侧 `adapters/cmass.py`，尚未上移到
  `Bayesian_inference` 的 model-owned predictive module；这是 Phase 1 允许的过渡形态。
- Phase 2 仍需让 context construction 更明确地复用上游 model registry /
  `build_compiled_model()`，避免 PPT adapter 自己重新选择 CMASS preprocessing path。
- `predictive.py` 中 artifact schema、trend panel 和 CLI 外部输入仍带有 CMASS-era 假设，
  需要在后续 Phase 3 / Phase 4 中继续拆分。

## 2026-05-10 Phase 2 完成：canonical context 接入 inference model registry

### 本阶段目标

- PPT 在 canonical-only 路径下根据 `runtime_config.model.name` 解析 active inference
  model definition。
- CMASS predictive context 的数值对象通过上游
  `get_model_definition(...).build_compiled_model(...)` 构建。
- 保持 legacy raw observation / cross-section snapshot 路径不变，避免破坏旧 run 的 PPC
  兼容性。

### 修改文件

- `Posterior_predictive_test/src/lensing_posterior_predictive/adapters/cmass.py`
- `Posterior_predictive_test/tests/test_predictive_registry.py`

### 实现内容

- `adapters/cmass.py` 不再直接调用 `load_cmass_canonical_dataset()` 或
  `build_cmass_context_from_canonical_dataset()`。
- CMASS adapter 现在通过 `cmass_lens_inference.model_registry.get_model_definition()`
  取得 inference registry entry，并调用其 `build_compiled_model(runtime_config)`。
- PPT adapter 仍单独读取 canonical dataset 来重建 `ObservationRecord`，这是为了兼容现有
  artifact writer 对 observed summaries 的输入形状；该读取只使用通用
  `load_canonical_inference_dataset()` 和 active model 声明的 required capabilities。
- 新增架构测试，锁定 canonical context builder 复用 inference registry 的边界。

### 验证

```bash
conda run -n cmass_lens pytest -q Posterior_predictive_test/tests/test_predictive_registry.py
```

结果：`6 passed`。

```bash
conda run -n cmass_lens pytest -q Posterior_predictive_test/tests
```

结果：通过，pytest 输出为全量通过点阵。

### 剩余未做事项

- 当前 predictive hook 仍是 PPT-local registry bridge，而不是直接挂在
  `Bayesian_inference` 的 `ModelDefinition` dataclass 上；最终是否上移需要结合
  Sonnenfeld predictive hook 的实现形态再决定。
- `ObservationRecord` reconstruction 仍在 CMASS PPT adapter 内部；如果后续更多模型需要
  observed-summary artifact，应考虑把 observation payload 做成 model-owned output，而不是
  让 PPT 假设 `ObservationRecord`。
- Phase 3 仍需把 `--sigma-table`、manifest predictive metadata、panel/trend schema 从
  CMASS-era 全局假设改为 model-declared contract。

## 2026-05-10 Phase 3 完成：外部输入与 predictive artifact metadata 模型化

### 本阶段目标

- `--sigma-table` 不再是 CLI parser 层的无条件必填项。
- CMASS 继续通过 predictive definition 声明自己需要 `sigma_table` 外部输入。
- summary / manifest artifact 写入模型名、predictive backend、schema version、
  supported diagnostics 和 required external inputs。
- CMASS Fig. 8 panel order 由 CMASS predictive definition 提供，不再在
  `run_posterior_diagnostics()` 内硬编码。

### 修改文件

- `Posterior_predictive_test/src/lensing_posterior_predictive/interfaces.py`
- `Posterior_predictive_test/src/lensing_posterior_predictive/adapters/cmass.py`
- `Posterior_predictive_test/src/lensing_posterior_predictive/predictive.py`
- `Posterior_predictive_test/src/lensing_posterior_predictive/cli.py`
- `Posterior_predictive_test/tests/test_predictive_registry.py`
- `Posterior_predictive_test/tests/test_posterior_predictive.py`

### 实现内容

- `PredictiveDefinition` 增加：
  - `trend_category_names`
  - `build_trend_panel_order(mass_definition)`
- CMASS adapter 提供 `_build_cmass_trend_panel_order()`，返回当前 5-panel trend schema。
- CLI 的 `posterior-predictive`、`posterior-trends`、`posterior-diagnostics`
  子命令将 `--sigma-table` 改为 optional；实际是否必须由 active model 的
  `required_external_inputs` 决定。
- 新增 `_resolve_sigma_table_path_for_definition()`：
  - CMASS 缺少 sigma table 时抛出包含模型名和 `sigma_table` 的清晰错误；
  - 为后续不需要 sigma table 的模型保留空输入路径。
- `ppc_summary.json`、`fig8_like_summary.json`、`run_manifest.json` 和 result metadata
  写入 predictive contract metadata：
  - `model_name`
  - `predictive_backend`
  - `predictive_schema_version`
  - `supported_diagnostics`
  - `required_external_inputs`

### 验证

```bash
conda run -n cmass_lens pytest -q Posterior_predictive_test/tests/test_predictive_registry.py Posterior_predictive_test/tests/test_posterior_predictive.py::test_run_posterior_diagnostics_generates_shared_parent_ppc_and_trend_artifacts Posterior_predictive_test/tests/test_posterior_predictive.py::test_cli_surface_exposes_canonical_trend_defaults Posterior_predictive_test/tests/test_posterior_predictive.py::test_cmass_diagnostics_requires_declared_sigma_table_input
```

结果：`9 passed`。

```bash
conda run -n cmass_lens pytest -q Posterior_predictive_test/tests
```

结果：通过，pytest 输出为全量通过点阵。

### 剩余未做事项

- `predictive.py` 内部仍有若干 CMASS-era plotting helper 使用固定
  `TREND_CATEGORY_NAMES`；主 diagnostics path 已改用 predictive definition，但
  legacy redraw / helper 层尚未完全拆出。
- observed overlay policy 仍主要由 CMASS-era helper 决定；Phase 4 需要先补强
  canonical observation contract，再继续分离 raw legacy overlay 与 canonical overlay。
- CLI help 文案仍说 “Jeans sigma-unit interpolation table”，后续可以进一步按模型动态
  展示输入需求；当前 argparse 静态 help 只做到 optional。

## 2026-05-10 Phase 4 完成：canonical observation contract 读取链路

### 本阶段目标

- canonical-only PPC 优先使用 canonical metadata 中的 observation / aperture /
  seeing contract。
- 不再只能靠 canonical dataset 文件名中的 `_boss_` 推断 BOSS/slit。
- 旧 canonical 文件缺少该 metadata 时继续保留 filename fallback，避免破坏历史 run。

### 修改文件

- `prepare_dataset/prepare_dataset/dataset_schema/writer.py`
- `prepare_dataset/tests/test_canonical_dataset_writer.py`
- `Bayesian_inference/src/cmass_lens_inference/canonical_dataset.py`
- `Bayesian_inference/tests/test_canonical_dataset.py`
- `Posterior_predictive_test/src/lensing_posterior_predictive/predictive.py`
- `Posterior_predictive_test/tests/test_posterior_predictive.py`

### 实现内容

- canonical writer 增加 raw observation group 的 explicit aperture metadata 收集：
  - `observation_flavor`
  - `sigma_definition`
  - `aperture_shape`
  - `aperture_width_arcsec`
  - `aperture_height_arcsec`
  - `aperture_radius_arcsec`
  - `seeing_fwhm_arcsec`
- writer 对不完整或跨 lens 不一致的显式 observation contract fail fast；完全缺失时不写入，
  以兼容历史 raw products。
- `CanonicalMetadata` 增加对应 optional 字段，reader 从 `/metadata` attrs 暴露这些字段。
- PPT 增加 `_load_observation_contract_from_canonical_dataset_path()`：
  - 新 canonical 文件有 metadata 时直接使用；
  - 旧文件无 metadata 时返回 `None`，由既有 filename fallback 接管；
  - 显式 metadata 不完整时抛出清晰错误。
- `ObservationContract` 增加 `sigma_definition`，sigma table 校验现在也比较该字段。

### 验证

```bash
conda run -n cmass_lens pytest -q prepare_dataset/tests/test_canonical_dataset_writer.py::test_write_canonical_inference_dataset_records_observation_contract_metadata Bayesian_inference/tests/test_canonical_dataset.py::test_load_canonical_inference_dataset_exposes_observation_contract_metadata Posterior_predictive_test/tests/test_posterior_predictive.py::test_canonical_observation_contract_uses_metadata_before_filename
```

结果：`3 passed`。

```bash
conda run -n cmass_lens pytest -q prepare_dataset/tests/test_canonical_dataset_writer.py Bayesian_inference/tests/test_canonical_dataset.py Posterior_predictive_test/tests
```

结果：`77 passed`。

### 剩余未做事项

- 旧 canonical products 缺少 observation contract metadata 时仍会 fallback 到 filename；
  这是兼容路径，Phase 6 清理时应明确标为 legacy fallback。
- observed gamma / mass overlay 仍不是 canonical block；当前只解决 aperture/sigma-table
  合同，不解决长期 observed-summary schema。
- annotation / redraw 仍大量使用 raw observation path，后续 legacy 收拢时需要拆到
  `legacy.py` 或等价模块。

## 2026-05-10 Phase 5 完成：Sonnenfeld predictive registry hook

### 本阶段目标

- `sonnenfeld2024_slacs` 和 `sonnenfeld2024_slacs_hunit` 不再被 PPT registry 视为
  unsupported。
- Sonnenfeld predictive hook 不复用 CMASS posterior helper、CMASS gamma mode 或
  CMASS 外部 sigma table contract。
- 先实现可运行的最小 parent-population diagnostics payload，供后续独立精修
  Sonnenfeld artifact schema。

### 修改文件

- `Posterior_predictive_test/src/lensing_posterior_predictive/adapters/sonnenfeld.py`
- `Posterior_predictive_test/src/lensing_posterior_predictive/registry.py`
- `Posterior_predictive_test/src/lensing_posterior_predictive/predictive.py`
- `Posterior_predictive_test/src/lensing_posterior_predictive/types.py`
- `Posterior_predictive_test/tests/test_predictive_registry.py`

### 实现内容

- 新增 `adapters/sonnenfeld.py`：
  - context construction 通过 `get_model_definition(model_name).build_compiled_model(...)`；
  - canonical dataset load 使用 active model 的 required capabilities；
  - `required_external_inputs=()`，即 Sonnenfeld 不再要求 CMASS observed-aperture
    `--sigma-table`；
  - predictive backend 名为 `numba_sonnenfeld_parent`；
  - schema version 为 `sonnenfeld2024_slacs_ppt_diagnostics_v1`。
- Sonnenfeld diagnostics payload 使用 Sonnenfeld context 的 parent sample、population
  sigma-unit grid、theta_E kernel、velocity-proxy selection primitives 和 cross-section
  weight primitive，避免调用 CMASS-specific posterior helpers。
- `run_posterior_diagnostics()` 允许模型没有 `sigma_table` 外部输入：
  - CMASS 缺少时仍按 Phase 3 的 model-declared contract 报错；
  - Sonnenfeld 路径将 `sigma_table_path` / `sigma_table_leaf_path` 记录为 `null`。
- `PosteriorDiagnosticsResult`、`PosteriorTrendResult`、`PosteriorPredictiveResult`
  的 `sigma_table_path` 类型放宽为 `Path | None`。
- registry 测试改为：
  - `toy_hierarchical` 仍明确 unsupported；
  - 两个 Sonnenfeld model 均有独立 predictive definition。

### 验证

```bash
conda run -n cmass_lens pytest -q Posterior_predictive_test/tests/test_predictive_registry.py
```

结果：`8 passed`。

```bash
conda run -n cmass_lens pytest -q Posterior_predictive_test/tests
```

结果：通过，pytest 输出为全量通过点阵。

```bash
conda run -n cmass_lens pytest -q Bayesian_inference/tests/test_sonnenfeld_runtime_model.py Posterior_predictive_test/tests
```

结果：`84 passed`。

### 剩余未做事项

- Sonnenfeld 现在有独立 predictive hook，但 artifact writer 仍复用现有 shared
  `run_posterior_diagnostics()` 输出路径；这保证最小闭环，但还不是完全独立的
  Sonnenfeld plotting/artifact schema。
- 当前 Sonnenfeld panel order 已通过 predictive definition 声明，但 composite figure
  writer 仍保留 CMASS-era helper 结构；Phase 6 应把 legacy/raw/CMASS-specific plotting
  helper 收拢，防止新模型长期继承 CMASS Fig.8 语义。
- 还没有加入真实 Sonnenfeld posterior run 的端到端 PPT fixture；目前验证覆盖 registry、
  context construction 和上游 Sonnenfeld runtime tests。

## 2026-05-10 Phase 6 完成：legacy 收拢与架构文档

### 本阶段目标

- 将 pre-registry raw CMASS run snapshot 解析从 generic `predictive.py` 中移出。
- 明确 legacy raw path 只支持 CMASS，不能成为 Sonnenfeld 或其他新模型的 fallback。
- 增加文档说明当前 PPT model-aware registry 结构和新增模型入口。
- 增加架构测试，防止 generic predictive workflow 重新定义 legacy CMASS parser。

### 修改文件

- `Posterior_predictive_test/src/lensing_posterior_predictive/legacy.py`
- `Posterior_predictive_test/src/lensing_posterior_predictive/predictive.py`
- `Posterior_predictive_test/src/lensing_posterior_predictive/cli.py`
- `Posterior_predictive_test/tests/test_predictive_registry.py`
- `Posterior_predictive_test/README.md`

### 实现内容

- 新增 `legacy.py`：
  - 承载 pre-registry raw observation / cross-section config snapshot parser；
  - 明确只解析 CMASS legacy snapshots；
  - 如果 legacy snapshot 显式声明非 `cmass` model，立即报错。
- `predictive.py` 删除 legacy parser 定义，只在 production parser 失败且文件满足 legacy
  raw markers 时调用 `load_legacy_ppc_runtime_config()`。
- CLI description 从 CMASS-only 改为 model-aware；`--sigma-table` help 文案说明它是
  model-declared input，目前由 CMASS diagnostics 要求。
- 新增 `Posterior_predictive_test/README.md`，记录：
  - generic workflow / model adapter / legacy path 的职责边界；
  - CMASS 与 Sonnenfeld 的 predictive input 差异；
  - 新模型接入 `PredictiveDefinition` 的最小字段。
- 新增架构测试：
  - legacy raw parser 拒绝非 CMASS model；
  - generic `predictive.py` 不再定义 legacy parser 函数。

### 验证

```bash
conda run -n cmass_lens pytest -q Posterior_predictive_test/tests/test_predictive_registry.py Posterior_predictive_test/tests/test_posterior_predictive.py::test_cli_surface_exposes_canonical_trend_defaults
```

结果：`11 passed`。

```bash
conda run -n cmass_lens pytest -q Posterior_predictive_test/tests Bayesian_inference/tests/test_sonnenfeld_runtime_model.py Bayesian_inference/tests/test_canonical_dataset.py prepare_dataset/tests/test_canonical_dataset_writer.py
```

结果：`98 passed`。

### 最终剩余风险

- `predictive.py` 仍是较大的 orchestration + plotting + artifact writer 混合文件；
  legacy parser 已收拢，但 plotting / artifact 模块化还可以继续做工程整理。
- Sonnenfeld hook 已注册并有最小 diagnostics payload，但尚缺真实 Sonnenfeld posterior
  run 的端到端 PPT fixture；后续科学验证应使用真实或更接近真实的 canonical SLACS run。
- CMASS 历史 artifact schema 仍被保留以保证兼容；如未来要彻底移除 CMASS-era Fig.8
  默认，需要另起一次 artifact schema migration。
