# Repository Integration And Workspace Separation Plan

> 状态：已完成。本文档记录当前已经确认的仓库整合与 workspace 分离设计，可作为后续实施计划和重构 worktree 的输入。若后续出现新的命名、边界、迁移顺序或兼容性问题，应先重新进入设计讨论，再修订本文档。

## 目标

把当前分散在 `Bayesian_inference/`、`Posterior_predictive_test/`、`prepare_dataset/` 三个目录中的代码，整理成一个更通用、可维护、可复用的强透镜分析代码包，同时保留研究工作区中真实运行所需的配置、数据、脚本和输出。

这次重构的目标不是简单移动目录，而是重新定义以下边界：

- 可复用库代码与用户工作区的边界。
- 数据准备、贝叶斯推断、posterior predictive workflow 三个阶段之间的边界。
- 通用强透镜工具与 `cmass` 这类具体样本 / 模型之间的边界。
- 数值后端、模型科学语义、运行 orchestration 之间的边界。
- 长期可拆成 package repo + workspace repo 的仓库形态。

所有命名、迁移和验证都必须继续以 `cmass_lens` conda 环境为标准运行环境。

## 已确认的设计方向

### 1. 公共项目名定为 `Statistical_SL`

`cmass` 是最初工作中的样本名称，不应继续作为通用代码包的身份。重构后：

- 公共 Python 包名不再包含 `cmass_lens`。
- CLI 名称不再包含 `cmass-lens-*`。
- `cmass` 继续作为 model id / config namespace / dataset label 存在。
- 当前已有的 `cmass`、`cmass_lens_only`、`sonnenfeld2024_slacs` 等模型，应收敛到统一的 model registry 结构下。

已确认命名：

- 项目 / distribution 语义名：`Statistical_SL`，含义是 statistical strong lensing。
- Python import namespace 建议采用小写风格：`statistical_sl`。
- 新 CLI 建议采用同一语义：`statistical-sl`。

说明：Python 技术上可以导入大写包名，但长期维护、打包发布、命令行输入和跨平台路径处理都更适合使用小写 import namespace。因此计划中用 `Statistical_SL` 指项目身份，用 `statistical_sl` 指源码包。

### 2. 先保持单仓库，但按未来可拆 workspace 设计

短期不直接拆成两个仓库。原因是当前项目仍有大量本地路径、HDF5 数据、运行产物、历史输出和本机环境约束，直接拆仓库会放大迁移风险。

目标形态是单仓库内清晰分层：

```text
CMASS_lens_project/
  pyproject.toml
  README.md

  src/
    statistical_sl/
      core/
      data_preparation/
      inference/
      posterior_predictive/
      models/
      numerics/
      cli.py

  workspace/
    configs/
    scripts/
    notebooks/
    reports/
    recipes/
    data/
    outputs/

  tests/
  docs/
  legacy/
```

长期如果拆仓库：

- `src/statistical_sl/`、`tests/`、库文档留在 package repo。
- `workspace/` 整体可迁移到 workspace repo。
- workspace repo 通过 editable install 或 released package 使用库代码。

### 3. `workspace/` 应该包含数据和输出

`workspace` 不应只放脚本和配置。真正的工作区应包含用户运行时会直接使用和产生的东西：

```text
workspace/
  configs/
  scripts/
  notebooks/
  reports/
  recipes/

  data/
    raw/
    external/
    canonical/
    derived/
    caches/

  outputs/
    ...
```

迁移决策：

- 新结构中 `workspace/data/` 和 `workspace/outputs/` 立刻成为 canonical 位置。
- 不设计根目录 `data/`、`outputs/` 的长期兼容路径。
- 整体重构正式实施前，必须先创建隔离 worktree；新 worktree 内直接按 `workspace/` 目标结构整理，避免在当前主工作区移动大量本地数据与输出。
- 当前主工作区中的根目录 `data/`、`outputs/` 只作为迁移前参考和必要数据源，不作为新代码默认路径。
- 库代码不能硬编码 workspace 位置；所有路径应来自 config 或 runtime context。

### 4. 保留“每次 run 一个目录”的输出结构

修正前一版设计：不要把输出主结构设计成 `outputs/inference/`、`outputs/posterior_predictive/` 并列。这样会打散一次完整科学运行的产物。

更合理的是保留现在的思路：一次 run 对应一个 run directory。该 run directory 内部再区分 inference、posterior predictive、配置快照、manifest 等内容。

建议目标结构：

```text
workspace/outputs/
  <analysis_family>/
    <run_id>/
      run_manifest.json
      config_snapshots/
        pipeline.yaml
        data_preparation.yaml
        inference.yaml
        posterior_predictive.yaml

      data_preparation/
        canonical_dataset_manifest.json
        validation_report.json

      inference/
        chain.h5
        run_result.json
        posterior_corner.png
        posterior_corner_result.json
        sampler_state/

      posterior_predictive/
        diagnostics/
          <diagnostic_run_id>/
            ppc_summary.json
            replicated_statistics.npz
            fig8_like.png
            fig8_like_summary.json
            fig8_like_curves.npz
            gamma_vs_sigma_star.png
            gamma_vs_sigma_star_summary.json
            gamma_vs_sigma_star_curves.npz
            gamma_vs_logre_kpc.png
            gamma_vs_logre_kpc_summary.json
            gamma_vs_logre_kpc_curves.npz
            gamma_vs_delta_r.png
            gamma_vs_delta_r_summary.json
            gamma_vs_delta_r_curves.npz
            figures/
            manifest.json
```

说明：

- `<analysis_family>` 可以暂时沿用 `devauc`、`sersic` 这样的 profile / analysis branch 名称，也可以后续改为更通用的 `cmass/devauc`、`sonnenfeld2024_slacs/paper_native`。
- `<run_id>` 仍然应包含时间戳和简短语义标签，方便人类审计。
- 如果某次只跑 inference，则 `posterior_predictive/` 可以不存在。
- 如果后续对同一个 inference run 重跑 posterior diagnostics，不应覆盖旧结果；应生成新的 `<diagnostic_run_id>`。
- 短期如果仍使用根目录 `outputs/`，内部结构也应遵循同一规则。

### 5. `diagnostics` 是 posterior predictive 下的完整后验检验 workflow

术语上应统一：

- `posterior_predictive` 是总类，表示基于 posterior samples 生成 replicated population、统计量、趋势图、检查图等后验预测 / 后验检验工作。
- `diagnostics` 是 posterior predictive 下面当前维护的完整后验检验 workflow，对应当前 `posterior-diagnostics`。它不是和 trend 平级的单一产物，而是一次联合运行。
- `diagnostics` 应同时生成 PPC summary / replicated statistics / overview figures / Fig. 8-like trend / `gamma_vs_*` trend artifacts。
- `posterior-trends` 和旧 `posterior-predictive` 更适合作为兼容入口或结果视图：它们可以调用完整 diagnostics workflow，再从 joint result 中返回旧 API 需要的子集。

因此输出结构中不应出现和 `posterior_predictive/` 并列的顶层 `diagnostics/`，也不应把 `diagnostics` 与 `trends` 设计成两个平级 workflow。更清楚的结构是：

```text
posterior_predictive/
  diagnostics/
    <diagnostic_run_id>/
      ppc_summary.json
      replicated_statistics.npz
      fig8_like.*
      gamma_vs_*.*
      manifest.json
```

后续需要进一步讨论：是否保留 CLI 子命令名 `posterior-diagnostics`，还是改成 `posterior-predictive diagnostics`。无论命令名如何，语义上它都应表示一次完整 posterior predictive diagnostics workflow，而不是只表示某个局部 diagnostic artifact。

## 目标库结构

```text
src/
  statistical_sl/
    __init__.py

    core/
      schemas/
      unit_conventions.py
      mass_definition.py
      manifests.py
      artifacts.py
      validation.py

    data_preparation/
      cli.py
      catalogs/
      measurement_sources/
      policies/
      builders/
      physics/
      dataset_schema/
      writers/
      validators/

    inference/
      cli.py
      config.py
      runtime/
      samplers/
      outputs/
      backends/
        numba_emcee/

    posterior_predictive/
      cli.py
      config.py
      interfaces.py
      registry.py
      adapters/
      workflows/
        diagnostics/
      plots/
      backends/
        numba/

    models/
      interfaces.py
      registry.py
      components/
        lensing/
        dynamics/
        selection/
        population/
      cmass/
      cmass_lens_only/
      sonnenfeld2024_slacs/
      sonnenfeld2024_slacs_sigma_star_gamma/
      toy_hierarchical/

    numerics/
      numba/
        kernels/
        runtime.py

    cli.py
```

`core/` 的边界必须刻意收窄。它只承载跨 data preparation、inference、posterior predictive 共享的稳定合同，例如 canonical schema、unit convention、mass definition、run manifest、artifact schema 和通用 validation。它不承载 workflow orchestration、model runtime assembly、backend glue、模型科学公式，也不承载 workspace 路径策略。

`models/components/` 是可复用科学组件层，不属于 `core/`。例如 lensing、dynamics、selection、population 这类可组合科学构件应在这里沉淀；具体模型再通过自己的 model spec、runtime adapter、likelihood hook 和 predictive hook 组合这些组件。

当前代码到目标结构的主要映射：

```text
Bayesian_inference/src/cmass_lens_inference/components/
  -> src/statistical_sl/models/components/

Bayesian_inference/src/cmass_lens_inference/models/<model>/
  -> src/statistical_sl/models/<model_id>/

Bayesian_inference/src/cmass_lens_inference/numba_backend/
  -> src/statistical_sl/inference/backends/numba_emcee/
   + src/statistical_sl/numerics/numba/kernels/
   + src/statistical_sl/models/<model_id>/*_numba.py

Posterior_predictive_test/src/lensing_posterior_predictive/adapters/
  -> src/statistical_sl/posterior_predictive/adapters/

Posterior_predictive_test/src/lensing_posterior_predictive/predictive.py
  -> src/statistical_sl/posterior_predictive/workflows/diagnostics/
   + src/statistical_sl/posterior_predictive/plots/
   + src/statistical_sl/posterior_predictive/adapters/

prepare_dataset/prepare_dataset/physics/
  -> src/statistical_sl/data_preparation/physics/

prepare_dataset/prepare_dataset/dataset_schema/
  -> src/statistical_sl/data_preparation/dataset_schema/
```

## 依赖方向

核心依赖方向：

```text
workspace configs/scripts
        |
        v
data_preparation
        |
        v
canonical dataset
        |
        v
inference
        |
        v
inference run artifacts
        |
        v
posterior_predictive
```

横向共享内容只能下沉到 `core/`、`models/` 或 `numerics/`：

```text
data_preparation      -> core
inference             -> core, models, numerics
posterior_predictive  -> core, models, numerics
models                -> core, numerics
```

禁止方向：

```text
posterior_predictive -> inference.backends.numba_emcee
data_preparation     -> inference
core                 -> data_preparation / inference / posterior_predictive
numerics             -> model-specific runtime config
```

## Numba 后端设计

当前问题：PPC 代码会深层 import inference 的 `numba_backend.kernels`。这说明共享数值原语放在了 inference 私有目录下，导致 posterior predictive 被迫依赖 inference 的内部实现。

目标拆分：

```text
numerics/
  numba/
    kernels/
      distributions.py
      interpolation.py
      lensing.py
      selection.py
      integration.py
      statistics.py
    runtime.py

inference/
  backends/
    numba_emcee/
      compiled_model.py
      likelihood_engine.py
      sampler.py

posterior_predictive/
  adapters/
    cmass.py
    sonnenfeld2024_slacs.py
  backends/
    numba/
      diagnostics.py
      reducers.py

models/
  cmass/
    spec.py
    likelihood_numba.py
    predictive_numba.py
  sonnenfeld2024_slacs/
    spec.py
    likelihood_numba.py
    predictive_numba.py
```

设计原则：

- `numerics.numba.kernels` 只放通用数值原语。
- `inference.backends.numba_emcee` 只负责推断后端 glue：compiled model、log probability、sampler、backend metadata。
- `posterior_predictive.backends.numba` 只负责 posterior diagnostics 后端 glue：diagnostics chunk runner、reducers、线程策略，以及一次 parent-population pass 中同时产出的 PPC 与 trend 数组。
- `posterior_predictive.adapters` 是模型接入 posterior predictive workflow 的边界；它可以调用模型 predictive hook 和 PPC backend，但不应被 backend 取代。
- 模型科学语义放在 `models/<model>/`，例如 likelihood 和 predictive hook。
- 可复用科学构件放在 `models/components/`，例如 lensing、dynamics、selection、population 组件。
- PPC 可以依赖 `models` 与 `numerics`，但不能依赖 inference backend。

关于 inference 中的现有 `numba_backend`：

- 不应直接删除。
- 第一阶段应迁移 / 重命名为 `inference.backends.numba_emcee`。
- 其中真正通用的 kernels 再逐步下沉到 `numerics.numba.kernels`。
- 模型相关 likelihood 逻辑迁到 `models/<model>/likelihood_numba.py`。

## 配置设计

一次完整 pipeline 不应该强迫用户手工维护三个完全独立、重复字段很多的 YAML。更合理的是“两层配置”，并把 pipeline recipe 作为面向日常运行的主入口：

1. pipeline recipe：日常运行主入口，声明 workspace root、步骤顺序、单步 config 路径、产物传递关系和 run 输出位置。
2. 单步 config：高级入口和复现入口，保证 data preparation、inference、posterior predictive 可以独立运行和独立验证。

建议结构：

```text
workspace/configs/
  data_preparation/
    cmass/devauc_hunits.yaml
    sonnenfeld2024_slacs/paper_native.yaml

  inference/
    cmass/devauc.yaml
    cmass/sersic.yaml
    sonnenfeld2024_slacs/paper_native.yaml

  posterior_predictive/
    cmass/devauc_diagnostics.yaml
    sonnenfeld2024_slacs/paper_native_diagnostics.yaml

  pipelines/
    cmass/devauc_full.yaml
    sonnenfeld2024_slacs/paper_native_full.yaml
```

pipeline recipe 示例：

```yaml
schema_version: statistical_sl_pipeline_v1
name: cmass_devauc_full
workspace_root: /Users/liurongfu/Work/CMASS_lens_project/workspace

steps:
  data_preparation:
    config: ../data_preparation/cmass/devauc_hunits.yaml
    output_dataset: data/canonical/inference_dataset_devauc_hunits.hdf5

  inference:
    config: ../inference/cmass/devauc.yaml
    dataset: ${steps.data_preparation.output_dataset}
    output_run_dir: outputs/devauc/${run_id}

  posterior_predictive:
    config: ../posterior_predictive/cmass/devauc_diagnostics.yaml
    run_dir: ${steps.inference.output_run_dir}
    output_dir: ${steps.inference.output_run_dir}/posterior_predictive/diagnostics/${diagnostic_run_id}
```

配置决策：

- pipeline recipe 是主要用户入口；单步 config 保留为高级复现入口。
- `workspace_root` 必须显式写在 pipeline recipe 中，使一次运行的 workspace 边界可审计、可复现。
- CLI 可以提供 `--workspace-root` 作为显式覆盖，用于临时迁移或本机路径调整；环境变量最多作为本地 convenience fallback，不作为持久配置合同。
- 现有 inference config 的字段名是否需要系统性重命名，不在本轮结构计划中提前决定；第一阶段默认保守迁移，不借 package 重构顺手改科学配置语义。

## 完成定义与验收标准

本计划完成的含义不是“目录已经移动完”，而是仓库已经形成可维护、可验证、可继续拆分的结构边界。完成时应同时满足结构、依赖、行为、配置、文档和验证六类条件。

后续执行者可以把本节作为 completion gate：只有当所有必须交付物都存在、阶段验收条件全部满足、最低验证命令通过、边界约束没有被违反时，才可以声明本轮整合完成。若存在任何偏离，最终交付必须明确写出偏离项、原因、风险和后续处理建议，不能把“部分迁移成功”描述为“计划完成”。

### 必须交付的结果

本轮计划最终必须交付以下结果：

- 一个隔离 worktree 中完成的重构分支，主工作区原有文件、数据和输出不被无意改写。
- 一个统一的 Python package surface：根目录 `pyproject.toml`、`src/statistical_sl/`、新公共 CLI `statistical-sl`。
- 三个 workflow 模块的清晰边界：`data_preparation`、`inference`、`posterior_predictive` 可以独立运行，也可以被 pipeline recipe 串联。
- 一组窄而稳定的共享合同：canonical schema、unit convention、mass definition、manifest / artifact names 等进入 `core/`，且 `core/` 不反向依赖 workflow。
- 一组明确的数值与模型边界：通用 kernels 位于 `numerics/`，inference backend glue 位于 `inference/backends/`，posterior predictive backend glue 位于 `posterior_predictive/backends/`，模型专属逻辑位于 `models/<model_id>/`。
- 一个 canonical workspace layout：`workspace/configs`、`workspace/recipes`、`workspace/scripts`、`workspace/notebooks`、`workspace/reports`、`workspace/data`、`workspace/outputs`。
- 一套可运行的代表性配置和 recipe，证明 data preparation、inference、posterior predictive 的输入输出合同在新结构中仍然闭合。
- 旧入口、旧脚本、旧测试和历史实现被删除、冻结或归档；新代码运行时不可通过 import、动态 import、`sys.path` hack、wrapper、配置路径或 CLI fallback 依赖它们。
- 如果保留 `legacy/`，它只能作为冷归档存在；当前 package、workspace、测试、配置、文档化 workflow 和外部用户不得以任何形式依赖 `legacy/` 下的文件。
- 文档更新到足以让后续维护者理解新结构、运行入口、workspace 约定、legacy 边界和常见验证命令。
- 验证记录清楚说明运行环境、关键命令、通过结果、跳过项和剩余风险。

### 总体完成定义

本轮仓库整合可以判定为完成，必须同时满足：

- `src/statistical_sl/` 成为唯一新的可复用源码包入口，公共 import namespace 不再使用 `cmass_lens_*`。
- `src/statistical_sl/` 的任何生产路径都不依赖旧三目录或 `legacy/`；这里的生产路径包括公共 import、CLI、workflow、model registry、backend、config loader、pipeline recipe 和默认测试可运行时到达的所有 Python 代码。
- data preparation、inference、posterior predictive 三个阶段仍可独立运行，也可通过 pipeline recipe 串联运行。
- `workspace/` 成为配置、脚本、notebook、报告、数据和输出的 canonical 工作区位置。
- 当前已确认的模型边界保留：`models/components/`、`posterior_predictive/adapters/`、`data_preparation/physics/` 不被合并进含混的大模块。
- `core/` 只包含共享合同，不包含 workflow、model runtime、backend glue、模型科学公式或 workspace 路径策略。
- PPC 不再 import inference 私有 backend，例如不允许 `posterior_predictive` 依赖 `inference.backends.numba_emcee`。
- 第一阶段不引入 `jax_numpyro` backend；如果未来恢复 JAX/NumPyro，必须另起设计提案。
- 旧入口已经删除、冻结或移出生产路径；不保留旧 CLI shim。
- `Bayesian_inference/`、`Posterior_predictive_test/`、`prepare_dataset/` 在最终完成状态下应被物理删除；若某些历史材料确需保留，只能以不可执行归档或迁移说明形式存在，不能作为 fallback implementation。
- 在 `cmass_lens` 环境下，核心测试、入口 smoke test、配置解析和至少一个轻量级 end-to-end dry run 能通过。

### 不算完成的情况

以下情况即使代码可以运行，也不能判定本计划完成：

- 只是把三个旧目录搬进 `src/`，但公共 namespace、CLI、配置、workspace、依赖边界仍沿用旧结构。
- 新包仍以 `cmass_lens_*` 作为公共身份，或者旧 `cmass-lens-*` CLI 被当作主入口继续维护。
- 新包通过 `cmass_lens_inference`、`lensing_posterior_predictive`、旧 `prepare_dataset`、`_legacy_paths.py`、`_legacy_imports.py` 或 `reexport_module(...)` 调用旧实现。
- `src/`、`workspace/`、默认测试、配置解析、CLI 或文档化当前 workflow 依赖 `legacy/` 下的任何文件。
- `posterior_predictive` 仍直接 import inference 私有 backend、runner、sampler 或 compiled model factory。
- `core/` 收纳了 workflow orchestration、模型公式、backend glue、I/O side effect 或 workspace 路径策略。
- 新结构依赖临时 `sys.path` hack 才能完成主要生产路径；短期兼容层可以存在，但必须有明确归档或删除边界。
- 没有形成 `workspace/` 下的数据、配置、recipe 和输出约定，导致后续仍需要猜测运行目录。
- 迁移后没有最小测试、静态依赖检查、CLI smoke test 或代表性 dry run 作为证据。
- 真实大数据、长链输出、缓存、临时 notebook 产物被混入版本管理。
- 为了迁移结构顺手改变科学语义、单位定义、采样目标、模型 id 或历史结果解释，却没有单独设计、记录和验收。

### 阶段验收标准

Phase 0 完成条件：

- 新建隔离 worktree，并在该 worktree 内继续执行后续迁移。
- worktree 分支名、路径、起始 commit、`cmass_lens` 环境可用性已经记录。
- 主工作区已有修改未被移动、覆盖、回滚或混入新 worktree。
- 本计划文档在新 worktree 中可读，并作为后续实施输入。

Phase 1 完成条件：

- 根目录存在统一 `pyproject.toml`，并能在 `cmass_lens` 环境中以 editable 方式安装或被测试命令正确发现。
- `src/statistical_sl/` 包结构存在，且 `python -c "import statistical_sl"` 成功。
- 新 CLI 入口以 `statistical-sl` 为公共命名；旧 `cmass-lens-*` 入口不作为兼容 shim 保留。
- 第一阶段只迁移 package/import surface，不改变科学公式、采样语义、数据 schema 或默认模型行为。
- 迁移后的 import 路径有最小 smoke test 覆盖。

Phase 2 完成条件：

- canonical dataset schema、unit convention、mass definition、run manifest、artifact schema 等共享合同已经迁入窄 `core/`。
- `core/` 不依赖 `data_preparation`、`inference`、`posterior_predictive` 或 workspace 目录。
- data preparation、inference、posterior predictive 都通过 `core` 中的稳定合同交换数据语义，而不是互相读取内部实现。
- 原 canonical dataset 的关键 metadata、capability blocks 和字段命名在迁移后保持可读、可验证。

Phase 3 完成条件：

- 通用 Numba kernels 已下沉到 `numerics.numba.kernels`。
- inference 专属 backend glue 已收敛到 `inference.backends.numba_emcee`。
- posterior predictive 专属 diagnostics backend glue 已收敛到 `posterior_predictive.backends.numba`。
- 模型专属 likelihood / predictive hook 位于 `models/<model_id>/`。
- 可复用科学组件位于 `models/components/`。
- 代码检查确认不存在 `posterior_predictive -> inference.backends.numba_emcee` 的依赖。

Phase 4 完成条件：

- `workspace/configs`、`workspace/scripts`、`workspace/notebooks`、`workspace/reports`、`workspace/recipes`、`workspace/data`、`workspace/outputs` 结构存在。
- pipeline recipe 是主要运行入口，且显式记录 `workspace_root`。
- 单步 config 仍可独立复现 data preparation、inference、posterior predictive。
- 新运行产物遵循“每次 run 一个目录”的结构，run 内部再区分 `data_preparation/`、`inference/`、`posterior_predictive/diagnostics/`。
- 大型数据和输出已经 inventory；不把真实大文件无意纳入版本管理。

Phase 5 完成条件：

- 旧 `prepare_intepolation_grids`、旧 `cmass_posterior_predictive`、`key_tests/` 等历史材料已经完成依赖清点；仍被当前系统需要的部分已经迁入新结构，不再需要的部分已经删除、冻结或作为冷归档处理。
- README、runbook 和必要迁移说明已经指向新结构和新 CLI。
- 不再维护旧 CLI shim；需要保留的便利入口只能调用新 workflow。
- `legacy/` 是冷归档，不是依赖边界；新测试、默认 CLI、新配置、workspace 脚本、notebook、文档化当前 workflow 和外部用户都不应依赖 `legacy/` 下的文件。

### 最低验证要求

正式声明重构完成前，至少需要在 `cmass_lens` 环境中通过以下类别的验证：

- import smoke test：确认 `statistical_sl`、三个阶段子包、model registry 可以导入。
- CLI smoke test：确认 `statistical-sl --help`、主要子命令 `--help` 可运行。
- config validation：至少一个 pipeline recipe 和每个阶段的代表性单步 config 可以解析。
- dependency boundary check：用静态搜索或测试确认禁止依赖没有出现；检查范围必须覆盖 `src/`、`tests/`、`workspace/`、`pyproject.toml`、`conftest.py`、README / runbook 中的当前 workflow。
- unit / integration tests：现有核心测试在迁移后通过，或有明确记录说明哪些 legacy 测试被移动、冻结或替换。
- artifact contract check：轻量运行或 saved-artifact replay 能产生 / 读取 `run_manifest.json`、`chain.h5`、`posterior_corner_result.json`、`ppc_summary.json` 等关键产物。
- git hygiene：最终 diff 不包含未说明的大文件、临时缓存、运行输出或用户原有无关修改。

推荐的最低命令形态如下，实际文件路径可以随迁移后的测试布局调整，但验证类别不能减少：

```bash
conda run -n cmass_lens python -m pip install -e . --no-deps
conda run -n cmass_lens python -m pytest --collect-only -q
conda run -n cmass_lens python -m pytest tests -q
rg -n "cmass_lens_inference|lensing_posterior_predictive|from prepare_dataset|import prepare_dataset|Bayesian_inference|Posterior_predictive_test|_legacy_paths|_legacy_imports|reexport_module|legacy/" src tests workspace pyproject.toml conftest.py
git diff --check
git status --short --untracked-files=all
```

旧目录测试只能作为迁移前后的对照材料使用，不能作为最终验收命令，因为最终状态不应要求旧三目录存在。README / runbook 需要人工检查当前 workflow 是否仍指导用户运行 `legacy/`、旧三目录或旧包名；历史说明可以保留，但不能被写成当前推荐路径。若某些 legacy 测试被移动、冻结或替换，最终验收不能简单删除这些测试结果；必须在文档或最终交付说明中列出对应测试、处理方式和替代验证。

### 验收流程

正式验收建议按以下顺序执行：

- 先检查 worktree 与 git 状态，确认所有变更都发生在隔离 worktree，且没有混入缓存、大文件、运行输出或主工作区无关修改。
- 再检查结构，确认 `src/statistical_sl/`、`workspace/`、`tests/`、`docs/`、`legacy/` 的职责符合本计划。
- 再做静态依赖检查，重点确认 `core` 没有反向依赖 workflow，`posterior_predictive` 没有依赖 inference 私有 backend，旧 `cmass_lens_*` 不再作为公共入口或实现依赖，`legacy/` 没有被当前系统依赖。
- 再运行 import、CLI、config、unit test、integration / dry-run 验证，所有 Python 验证必须通过 `cmass_lens` 环境执行。
- 最后检查关键产物合同，确认代表性 run 或 saved-artifact replay 能读写预期 manifest、chain、corner、posterior predictive summary。
- 最终交付说明必须包含：变更摘要、关键路径、验证命令与结果、未执行验证及原因、已知风险、后续建议。

## 边界约束与非目标

本计划的边界约束如下：

- 不在主工作区直接执行大规模目录移动；所有实施必须发生在隔离 worktree。
- 不改变科学模型语义、物理公式、采样目标分布、posterior interpretation 或数据单位定义；若验证发现必须改变，先停下并单独设计。
- 不借结构迁移顺手重命名 model id；`cmass_lens_only` 保持不改，`cmass` 仍可作为 model id / config namespace / dataset label。
- 不保留旧 CLI shim；兼容便利命令如果存在，必须调用新 workflow，并在文档中标为 convenience wrapper。
- 不设计根目录 `data/`、`outputs/` 的长期兼容路径；新 canonical 位置是 `workspace/data/` 与 `workspace/outputs/`。
- 不把本机真实大数据、长链输出、缓存目录、临时 notebook 输出纳入版本管理。
- 不把 `core/` 变成杂物层；任何带有 workflow、runtime、backend、model-specific 语义的内容都不能进入 `core/`。
- 不让 posterior predictive 依赖 inference 私有 backend；共享数值逻辑只能下沉到 `numerics` 或模型层。
- 不让当前系统依赖 `legacy/`；任何仍需被 import、执行、测试、配置引用或作为当前文档 workflow 运行的内容，都必须迁入 `src/statistical_sl/`、`workspace/` 或 `tests/` 的正式位置。
- 不把 `legacy/` 当作 fallback implementation、compatibility layer 或 regression suite 的默认来源；如果保留它，它只能用于人工查阅历史材料。
- 不在第一阶段恢复或新增 JAX/NumPyro backend。
- 不中断已经启动的长时间任务；如需停止，必须先说明风险并等待用户确认。

越界处理规则：

- 如果迁移过程中发现必须改变科学公式、单位定义、数据 schema、模型 id 或输出语义，立即停止对应改动，把问题记录为新的设计议题；不能把语义变化伪装成结构迁移。
- 如果发现旧目录中有无法安全分类的大文件、历史输出或 notebook 产物，先 inventory 并写清建议归宿；不要在没有确认的情况下移动进版本管理。
- 如果某个旧 CLI、旧脚本或旧测试仍被真实 workflow 依赖，优先把依赖关系记录清楚，再迁到新入口、新测试或新 workspace 位置；不能用归入 `legacy/` 的方式继续满足当前依赖。
- 如果为了兼容必须保留短期 wrapper，wrapper 必须只转发到新实现，并在文档中标明它不是长期公共 API。
- 如果最低验证无法完成，最终状态只能标记为部分完成或 blocked，并明确缺失验证对应的风险面。

非目标：

- 本计划不要求立刻拆成 package repo + workspace repo。
- 本计划不要求重新生成全部数据集或重跑全部历史 inference。
- 本计划不要求重写模型科学实现。
- 本计划不要求一次性清理全部历史 notebook、论文草稿、临时分析或旧输出。
- 本计划不要求发布 PyPI 包；只确定 published distribution name 应使用 `statistical-sl`。

## 迁移策略草案

### Phase 0: 决策冻结与隔离 worktree

- 顶层项目名已确认：`Statistical_SL`。
- Python import namespace 已建议：`statistical_sl`。
- 重构实施前必须新建隔离 worktree，不在当前主工作区直接大规模移动目录。
- 新 worktree 内直接使用 `workspace/data` 与 `workspace/outputs` 作为 canonical 路径。
- 新 CLI 不保留旧 `cmass-lens-*` 入口或其他旧 CLI shim。
- `posterior-trends` 可以保留为兼容 wrapper / convenience command，但不作为独立 workflow 重新实现。

### Phase 1: 建立新包骨架

- 在根目录建立统一 `pyproject.toml`。
- 创建 `src/statistical_sl/`。
- 先迁移 import surface，不改变科学逻辑。
- 保持测试继续在 `cmass_lens` 环境运行。

### Phase 2: 抽出共享合同

- 把 canonical dataset schema、unit convention、mass definition、manifest schema、run artifact schema 移到 `core/`。
- `core/` 只保留稳定合同；不迁入 inference runtime context、model-specific derived context、backend layout 或 workspace path policy。
- 保证 data preparation、inference、posterior predictive 都只通过稳定合同交互。

### Phase 3: 重整 Numba ownership

- 把通用 kernels 下沉到 `numerics.numba.kernels`。
- 把 inference 后端改名 / 收敛到 `inference.backends.numba_emcee`。
- 把 PPC Numba diagnostics 收敛到 `posterior_predictive.backends.numba`。
- 保留 `posterior_predictive.adapters` 作为模型 predictive logic 的接入层。
- 把模型专属 likelihood / predictive hooks 放入 `models/<model>/`。
- 把可复用科学组件收敛到 `models/components/`。

### Phase 4: 整理 workspace

- 迁移 configs、scripts、notebooks、reports、recipes。
- 在隔离 worktree 内直接建立 `workspace/data` 与 `workspace/outputs` 目标结构。
- 对大型未跟踪数据 / 输出，实施前先 inventory；只把运行必需的样例、manifest、配置和路径合同纳入版本管理，真实大文件按本地数据策略处理。
- 更新 README 和 runbook。

### Phase 5: legacy quarantine

- 清点旧的 `prepare_intepolation_grids`、旧 `cmass_posterior_predictive`、`key_tests/` 历史 comparison harness 等材料，区分“仍被当前系统需要迁移的实现”和“只供人工查阅的历史材料”。
- 仍被当前系统需要的实现、脚本或测试必须迁入新结构，不能留在 `legacy/` 中被外界调用。
- 只供人工查阅的历史材料才可以进入 `legacy/`；进入后即声明为冷归档，不再作为 import、执行、测试、配置或文档化 workflow 的依赖。
- 删除或冻结不再维护的入口。
- 不保留旧 CLI shim；新结构只维护新 CLI。

## 待讨论问题

当前无已识别的结构性待讨论问题。后续讨论中如果出现新的命名、边界、迁移顺序或兼容性问题，再追加到本节。

## 已确认决策

1. 项目语义名使用 `Statistical_SL`，含义是 statistical strong lensing。
2. 新源码包 namespace 建议使用 `statistical_sl`。
3. 整体重构正式实施前，新建隔离 worktree，并在新 worktree 内直接整理成 `workspace/` 目标结构。
4. `workspace/data` 与 `workspace/outputs` 立刻作为新结构 canonical 路径，不设计根目录 `data/`、`outputs/` 的长期兼容路线。
5. 不保留旧 CLI shim。
6. 完整 diagnostics workflow 的 CLI 可以改成 `posterior-predictive diagnostics`；旧 `posterior-trends` 可以保留为兼容 wrapper / convenience command。
7. `cmass_lens_only` 作为 model id 不需要改名。
8. `key_tests/` 不进入新 `tests/regression/`；若保留在 `legacy/`，只能作为冷归档，不能被当前测试、CLI、workspace、README 当前 workflow 或外部用户依赖。
9. 保留并正式化 `models/components/`，作为 lensing / dynamics / selection / population 等可复用科学组件层。
10. 保留并正式化 `posterior_predictive/adapters/`，作为模型接入 posterior predictive workflow 的边界。
11. 保留并正式化 `data_preparation/physics/`，作为数据准备阶段可复用科学计算层。
12. 第一阶段目标结构不包含 `inference/backends/jax_numpyro/`；JAX/NumPyro 若未来恢复，应作为独立后端设计提案进入。
13. pipeline recipe 是主要用户入口；单步 config 保留为高级复现入口。
14. `workspace_root` 显式写入 pipeline recipe；CLI 可显式覆盖，环境变量最多作为本地 convenience fallback。
15. published distribution name 使用 `statistical-sl`；项目展示名继续使用 `Statistical_SL`，Python import namespace 使用 `statistical_sl`，CLI 使用 `statistical-sl`。
16. `legacy/` 若保留，只能是人工查阅的冷归档；当前系统、默认测试、CLI、配置、workspace 脚本、README / runbook 当前 workflow 和外部用户都不得以任何形式依赖其中的文件。

## 当前结论摘要

- 接受“单仓库过渡，按未来 package + workspace 可拆分设计”。
- 项目语义名定为 `Statistical_SL`，Python 源码包建议为 `statistical_sl`。
- published distribution name 使用 `statistical-sl`，CLI 使用 `statistical-sl`。
- 接受去 `cmass_lens_*` 公共命名，`cmass` 退回模型 / 样本层。
- `cmass_lens_only` model id 保持不改。
- 接受 `workspace/` 包含数据、输出、配置、脚本、notebook、报告，并且 `workspace/data`、`workspace/outputs` 是新结构 canonical 路径。
- 正式实施前必须先建隔离 worktree。
- 不保留旧 CLI shim；但 `posterior-trends` 可保留为兼容 wrapper / convenience command。
- 修正输出结构：保留每次 run 一个目录，不采用 `outputs/inference` 与 `outputs/posterior_predictive` 并列的主结构。
- 修正术语：`diagnostics` 是 `posterior_predictive` 下当前维护的完整后验检验 workflow，包含 PPC summary 和 trend artifacts；`trends` 不是与 diagnostics 平级的 workflow。
- Numba 后端按 `numerics`、`inference backend`、`posterior_predictive backend`、`model hooks` 四层拆分。
- 保留当前代码已经形成的 `models/components`、`posterior_predictive/adapters`、`data_preparation/physics` 边界。
- `core` 收窄为共享合同层，不放 workflow、模型 runtime、backend glue 或 workspace 路径策略。
- 第一阶段目标结构删除 `jax_numpyro` backend。
- pipeline recipe 是主要用户入口，且显式记录 `workspace_root`。
- `key_tests/` 不进入新 `tests/regression/`；若保留在 `legacy/`，只能作为冷归档，不能被当前系统或外部用户依赖。
- `legacy/` 不作为依赖边界、兼容层、fallback implementation 或默认 regression suite；仍被当前系统需要的内容必须迁入新结构，不需要的内容删除、冻结或作为不可执行冷归档。
