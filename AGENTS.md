# AGENTS.md - CMASS_lens_project

本文件是给新开对话后的 agent 使用的仓库入口手册。它应帮助 agent
快速恢复项目结构、科学合同、验证命令和安全边界。遇到不确定之处时，
以 live code、workspace configs、tests 和 docs 为准，不要只凭历史记忆判断。

## 运行环境硬约束

- 所有 Python、pytest、CLI、脚本运行都必须基于 `cmass_lens` 环境。
- 推荐命令形式：

```bash
conda run -n cmass_lens python -m pytest -q
conda run -n cmass_lens statistical-sl --help
```

- 不要在未说明原因的情况下使用系统 Python 或其他 conda 环境。
- 不要未经用户明确允许中断、kill、cancel 已启动的长任务、数据处理、
  inference、diagnostics、plotting 或测试进程。若怀疑有损失风险，先说明
  当前状态、风险和可选动作，再等用户确认。

## 项目定位

本仓库是 `Statistical_SL` 强透镜统计推断项目。当前主包是：

```text
src/statistical_sl/
```

`workspace/` 是面向运行和产物的工作区，承载 configs、recipes、data 和
outputs。设计方向是未来可拆为 package + user workspace，但当前仍是单仓库
过渡态。

不要恢复旧 public identity：

- 不要新增 `prepare_dataset/` 作为生产源码根。
- 不要新增 `Bayesian_inference/` 或 `Posterior_predictive_test/` 作为生产源码根。
- 不要给当前生产路径加旧 `prepare_dataset` import shim、CLI shim 或 `sys.path`
  fallback。
- `legacy/` 只能作为历史参考或冷归档，不应作为当前 workflow 的 fallback
  implementation。

## 关键目录

```text
src/statistical_sl/core/                 # 跨 workflow 的稳定 schema / policy / artifact 合同
src/statistical_sl/data_preparation/     # 数据准备、canonical writer、direct pipeline、物理表构建
src/statistical_sl/inference/            # inference runtime、config、sampler、canonical reader
src/statistical_sl/models/               # CMASS / Sonnenfeld / toy models 与组件
src/statistical_sl/posterior_predictive/ # PPC、diagnostics、trends、predictive adapters
src/statistical_sl/pipeline/             # post-canonical pipeline recipes / dry-run / orchestration
workspace/configs/                       # data-prep / inference / posterior-predictive configs
workspace/recipes/                       # run-facing workflow recipes
workspace/data/                          # raw / external / canonical data products
workspace/outputs/                       # one run, one directory outputs
docs/contracts/                          # public data and handoff contracts
docs/data-preparation/                   # data-prep workflow docs
docs/superpowers/plans/                  # design plans and historical implementation plans
legacy/                                  # historical reference material only
```

目录级 `AGENTS.md` 可以覆盖更具体的规则。例如 `outputs/AGENTS.md` 面向
Bayesian diagnostics 产物分析；在 `outputs/` 下工作时必须同时遵守该文件。

## Data Preparation 现状

data preparation 现在有两条并列路线：

1. 兼容 prepared-observation-HDF5 路线
   - 入口：`statistical-sl prepare-dataset --build-canonical-inference-dataset ...`
   - 主要代码：`src/statistical_sl/data_preparation/dataset_schema/writer.py`
   - 输入：observation HDF5、cross-section HDF5、可选 sigma bundle / population
     sigma HDF5。
   - 用途：保留现有 HDF5 中间产物路线，不是 direct pipeline。

2. direct source-to-canonical 路线
   - 入口：

```bash
conda run -n cmass_lens statistical-sl prepare-dataset \
  --build-canonical-direct \
  --config workspace/configs/data_preparation/cmass/devauc_direct_hunits.yaml
```

   - 主要代码：`src/statistical_sl/data_preparation/direct_pipeline/`
   - 输入：source catalog、trusted velocity measurements、cross-section source。
   - 输出：validated canonical inference HDF5 和可选 audit JSON。
   - 文档：`docs/data-preparation/direct-canonical-pipeline.md`
   - velocity handoff 合同：`docs/contracts/velocity_measurements_v1.md`

direct pipeline 的核心语义：

- source catalog 提供 lens facts，不自动成为 trusted velocity source。
- `velocity_measurements_v1` 是推荐的 trusted sigma handoff。
- pPXF CSV 是 compatibility adapter，不是推荐的长期上游合同。
- CMASS catalog `sigma` / `sigma_err` 默认只进入 provenance，除非 config 显式
  选择 trusted `catalog_columns`。
- `num_sigma = 0` 是合法状态，用于保留缺少 trusted measurement 的 lens。
- accepted sigma rows 必须携带 aperture geometry 和 seeing。
- 不同 lens 可有不同 aperture；同一 lens 内 incompatible aperture 会被拒绝。
- direct writer 应从 catalog + trusted measurement 组装 `CanonicalDatasetPayload`，
  不能把旧 observation HDF5 当 direct pipeline 的核心数据模型。

## Inference / PPC / Diagnostics 语义

- `diagnostics` 在本项目中指 posterior-predictive 后验检查 workflow，包含
  PPC 和 trends，不要把它拆成无关 stage-owned 顶层输出目录。
- 输出布局默认是 `one run, one directory`，在 `workspace/outputs/.../<run_id>/`
  或等价 run-owned 目录下组织。
- 不要新建 `outputs/inference` / `outputs/posterior_predictive` 这种 stage-owned
  顶层拆分，除非用户明确要求重新设计并更新计划。
- post-canonical recipes 不应偷偷重新跑 data preparation。

## 科学合同红线

以下变化不能作为普通重构顺手完成，必须先单独说明设计、风险和验证：

- unit convention、`h_ref`、mass-definition label、mass radius 语义。
- sigma likelihood、aperture/seeing、`num_sigma`、missing policy。
- cross-section source meaning、boundary policy、theta/gamma extrapolation。
- selection normalization、PPC parent-sample semantics。
- model id、parameter schema、prior 常数、4.29 reference semantics。
- canonical HDF5 schema、capability strings、workspace data contract。

CMASS separable cross-section 的 boundary/extrapolation 问题尤其要谨慎：
generic 2D extrapolation 与 CMASS-specific analytic `theta_E^2` semantics 不等价。
修改前先读 `src/statistical_sl/core/cross_section_policy.py`、相关 tests 和
`docs/superpowers/plans/2026-05-31-cross-section-boundary-policy.md`。

## 任务前先读什么

按任务类型优先读这些文件：

- data-prep direct pipeline：
  - `docs/data-preparation/direct-canonical-pipeline.md`
  - `docs/contracts/velocity_measurements_v1.md`
  - `src/statistical_sl/data_preparation/direct_pipeline/runner.py`
  - `src/statistical_sl/data_preparation/direct_pipeline/config.py`
- repository / workspace 结构：
  - `docs/superpowers/plans/2026-05-24-repository-integration-structure.md`
- canonical schema：
  - `src/statistical_sl/core/canonical_schema.py`
  - `src/statistical_sl/inference/canonical_dataset.py`
- inference config / runtime：
  - `workspace/configs/inference/...`
  - `src/statistical_sl/inference/config.py`
  - `src/statistical_sl/models/*/runtime.py`
- posterior predictive / diagnostics：
  - `workspace/configs/posterior_predictive/...`
  - `src/statistical_sl/posterior_predictive/`
  - `workspace/recipes/...`
- historical comparison only：
  - `legacy/`
  - `docs/superpowers/plans/`

如果计划文档和 live code 冲突，先报告冲突并以 live code + tests 作为当前事实，
不要把旧计划当作已实现事实。

## 验证命令

常用 smoke / boundary：

```bash
conda run -n cmass_lens python -m pytest tests/test_smoke.py tests/test_dependency_boundaries.py -q
```

direct data-prep focused suite：

```bash
conda run -n cmass_lens python -m pytest \
  tests/data_preparation/test_direct_pipeline_*.py \
  tests/test_dependency_boundaries.py \
  tests/test_smoke.py \
  -q
```

全量测试：

```bash
conda run -n cmass_lens python -m pytest -q
```

静态边界和 whitespace：

```bash
rg -n 'prepare_dataset|Bayesian_inference|Posterior_predictive_test|python -m prepare_dataset|prepare_dataset/examples|canonical_pipeline|catalog_sources|sigma_sources|payload_builder' \
  src tests workspace pyproject.toml docs/contracts docs/data-preparation
git diff --check
git status --short --untracked-files=all
```

说明：

- 旧计划和历史 docs 可以提旧名称；当前生产代码、tests、workspace configs 和
  current workflow docs 不应把旧名称作为当前执行路径。
- real-data tests 依赖本地 `workspace/data`。若缺数据导致失败，要明确标为数据
  prerequisite，不要把它误判为代码 regression。

## Git / Worktree 纪律

- 大型重构默认使用隔离 worktree，除非用户明确要求在当前主工作区直接改。
- 开始前记录 branch、HEAD、worktree path 和 `git status --short --untracked-files=all`。
- 不要回滚、删除或覆盖用户已有的未提交改动。
- 如果主工作区脏，只处理任务相关文件；无关脏文件只报告，不清理。
- 合并前后至少跑与变更相关的 focused tests；能跑全量测试时优先跑全量测试。
- 合并、stage、commit、push、PR 都要清楚说明实际成功的命令和剩余风险。

## Agent 可做与不可做

Agent 适合做：

- 梳理 source contract、schema、config 和 tests。
- 草拟 source manifest、column mapping、YAML config、adapter 和 audit report。
- 添加 deterministic validation、focused tests 和 migration docs。
- 对比 live code 与历史计划，指出已实现、未实现和风险边界。

Agent 不应静默裁决：

- 某个 sigma source 是否科学可信。
- aperture/seeing 的物理含义。
- unit convention 和 h-unit / fixed-kpc 转换语义。
- missing policy 应该是 `fail` 还是 `num_sigma_zero`。
- cross-section grid 的物理 meaning 或 extrapolation policy。
- production inference / PPC 合同是否可以改变。

这些问题需要显式设计、用户确认和可验证证据。

## 代码与文档风格

- 默认中文沟通，技术名词、路径、命令和 schema 名保留英文原文。
- 修改代码时优先沿用现有模块边界和 helper API，不要引入无关抽象。
- 注释应解释科学合同、边界条件和非显然实现原因；不要给显而易见的语句写空泛注释。
- 文档应短而可执行，优先链接到 live code、configs 和 tests。
- 不要把计划文档正文复制进实现文档；计划是历史和设计证据，不是运行手册。

## 完成声明标准

不要只因为代码已提交或某个命令跑完就说“完全完成”。完成声明必须说明：

- 实际修改了哪些边界。
- 跑了哪些验证命令，结果是什么。
- 哪些验证没有跑，原因和风险是什么。
- 当前 `git status` 是否干净；若不干净，哪些是无关用户改动。
- 是否仍有 skipped real-data audit、legacy quarantine、workspace data prerequisite
  或生产配置实跑缺口。
