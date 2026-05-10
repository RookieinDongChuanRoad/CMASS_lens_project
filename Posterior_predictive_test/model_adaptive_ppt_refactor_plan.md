# Posterior Predictive Test 模型自适应重构目标与计划

> 本文档中的 `PPT` 指当前仓库里的 `Posterior_predictive_test` 包，不是 PowerPoint。
> `PPC` 指 posterior predictive checks，是 PPT 包当前承载的主要诊断族之一。

## 1. 最终目标

当前 `Posterior_predictive_test` 已经把 PPC histogram 和 posterior trends 合并到
Numba shared-parent diagnostics 路径里，但它本质上仍然是一个 CMASS-specific
diagnostics 实现。最终目标是把它重构成：

**model-aware、registry-driven、Numba-backed 的 posterior predictive diagnostics
工作流。**

具体目标如下：

1. `Posterior_predictive_test` 不再在通用 orchestration 层直接 import
   `cmass_lens_inference.models.cmass.*`。
2. PPT 从已完成 inference run 的 `config_snapshot.yaml` 读取 `model.name`，
   并通过 `Bayesian_inference` 的模型注册机制解析 active model。
3. 不在 PPT 里重复实现一套模型科学逻辑；PPT 只负责：
   - 读取 run directory、posterior chain、config snapshot；
   - 选择 posterior draws；
   - 调用 active model 暴露的 predictive / diagnostic hook；
   - 写出 JSON、NPZ、PNG、manifest 等诊断产物；
   - 维护 CLI、artifact schema、兼容旧 run 的 reader。
4. 模型相关的 latent population draw、selection weight、lensing observable、
   sigma model、trend quantities 等科学逻辑应尽量复用 inference normalization
   已经使用的上游 Numba kernel 或共享 helper。
5. 新模型接入时，不应该修改 PPT 的主流程；最多新增该模型自己的薄
   predictive hook 和模型特定 artifact/plot schema。
6. 保持生产后端方向一致：PPT 的热路径继续以 Numba 为目标，不引入 JAX /
   NumPyro 回退路径。

## 2. 关键设计判断

### 2.1 不能直接把 normalization 当成 PPC

PPC 的生成模拟样本过程确实和 inference normalization 高度相似：二者都需要从
同一个 latent population 出发，经过同一个 selection function，并使用同一套
lensing / interpolation / sigma primitives。

但是二者的输出目标不同：

- normalization 目标是 scalar reduction，例如 selection normalization；
- PPC 目标是 materialized diagnostics，例如 replicated `theta_E`、`gamma`、
  mass、sigma、parent/detectable/selected trends，以及 observed-vs-replicated
  summaries。

因此，PPC 不应该直接调用 `normalization_mc_numba()` 作为黑盒。更合理的方向是
把每个模型 posterior kernel 内部已经存在的公共步骤抽成可复用 building blocks：

```text
draw_population_state(...)
compute_lensing_observables(...)
compute_selection_weight(...)
compute_sigma_model(...)
```

然后：

```text
normalization_mc_numba        = 共享模型核心 + scalar reduction
predictive_diagnostics_numba  = 共享模型核心 + materialized samples / bins / trends
```

### 2.2 adapter 不是目的，predictive contract 才是目的

“model adapter” 这个名字不是必须的。真正必须存在的是一个模型特定的
predictive contract。这个 contract 可以挂在 `Bayesian_inference` 的
`ModelDefinition` 上，也可以由 PPT 包内的 registry 间接绑定。

优先设计方向：

```python
ModelDefinition(
    ...,
    evaluate_log_prob=...,
    build_predictive_engine=...,  # optional; model can opt in
)
```

PPT 主流程只做：

```python
model_definition = get_model_definition(runtime_config.model.name)
compiled_model = model_definition.build_compiled_model(runtime_config)
predictive_engine = model_definition.build_predictive_engine(compiled_model)
payload = predictive_engine.run_diagnostics(posterior_draws, options)
```

如果某个模型没有实现 predictive hook，PPT 应该给出清晰错误，而不是退回到
CMASS 逻辑。

### 2.3 inference package 和 PPT package 的职责边界

`Bayesian_inference` 应该拥有：

- 模型 registry；
- model spec 和 parameter schema；
- model runtime context；
- model-owned Numba posterior kernels；
- 可复用的模型生成 / selection / observable building blocks；
- 可选的 predictive engine hook。

`Posterior_predictive_test` 应该拥有：

- CLI；
- run directory / chain / config snapshot 读取；
- posterior draw selection；
- diagnostics orchestration；
- artifact schema；
- plotting；
- legacy PPC run compatibility。

也就是说，科学模型如何生成模拟样本属于上游模型；诊断工作流如何运行和落盘属于
PPT。

## 3. 当前代码状态

### 3.1 canonical dataset

当前 canonical inference dataset 已经提供模型共享的数据容器，包括：

- `metadata`
- `lenses`
- `lensing_mass_grids`
- `lensing_cross_section`
- `velocity_dispersion_grids`

它通过 capability strings 表达模型需要的数据能力，例如：

- `lens_observations.v1`
- `lensing_mass_grids.v1`
- `lensing_cross_section.theta_gamma_grid.v1`
- `velocity_dispersion.per_lens_s2.v1`
- `velocity_dispersion.fp_within_re.v1`
- `velocity_dispersion.population_sigma_unit.v1`

这个结构已经足够支撑 inference runtime 区分 CMASS 和 Sonnenfeld，但还不是完整
PPT contract。缺口包括：

- observed aperture / sigma flavor 目前仍有路径名推断逻辑；
- observed gamma / mass overlay 不是统一 canonical block；
- 每个模型应该产出哪些 replicated observables 还没有 schema；
- 每个模型可画哪些 trend panels 还没有声明式 contract。

### 3.2 Bayesian inference model registry

`Bayesian_inference` 当前已经通过 `model.name` 选择具体模型。已知 concrete models：

- `cmass`
- `sonnenfeld2024_slacs`
- `sonnenfeld2024_slacs_hunit`
- `toy_hierarchical`

每个模型都通过 `ModelSpec` 声明：

- model name；
- unit convention；
- mass aperture；
- sampled parameter schema；
- required / optional canonical capabilities；
- backend kernel key。

每个模型再通过 `ModelRuntimeAdapter` 构造自己的 compiled context。这个结构已经
满足 inference 自动适配，但还没有暴露 predictive diagnostics hook。

### 3.3 Posterior_predictive_test

当前 PPT 的 `predictive.py` 已经包含 Numba shared-parent diagnostics，并同时生成：

- PPC replicated statistics；
- Fig. 8-like trend curves；
- observed overlay；
- `ppc_summary.json`；
- `fig8_like_summary.json`；
- `replicated_statistics.npz`；
- figures。

主要问题：

1. 通用 workflow 文件直接 import CMASS preprocessing 和 CMASS posterior helper。
2. canonical PPC 路径显式拒绝非 `cmass` model。
3. Numba diagnostics kernel 内部写死 CMASS parent population、CMASS gamma mode、
   CMASS trend axes。
4. `--sigma-table` 是全局必填，但不同模型对外部 sigma table 的需求可能不同。
5. Fig. 8-like panel order 和 observed overlays 是 CMASS-specific，不应成为所有
   模型的默认 contract。

## 4. 目标架构

建议目标结构如下：

```text
Bayesian_inference/
  src/cmass_lens_inference/
    model_interfaces.py
      ModelSpec
      ModelRuntimeAdapter
      ModelDefinition
      PredictiveEngine / PredictiveDefinition  # 新增或等价 contract

    models/
      cmass/
        posterior.py
        predictive.py          # CMASS predictive core / Numba diagnostics hook
      sonnenfeld2024_slacs/
        posterior.py
        predictive.py          # Sonnenfeld predictive core / Numba diagnostics hook
      toy_hierarchical/
        posterior.py
        predictive.py          # optional smoke diagnostics hook

Posterior_predictive_test/
  src/lensing_posterior_predictive/
    predictive.py              # public API compatibility layer
    orchestration.py           # generic run/chain/config orchestration
    artifacts.py               # JSON/NPZ/PNG writers
    plotting.py                # generic plotting utilities
    legacy.py                  # old raw CMASS snapshot compatibility
```

这里的 `predictive.py` 文件名只是建议。实际实现时可以选择更准确的命名，例如
`diagnostics.py` 或 `posterior_predictive.py`，但职责边界必须清楚。

## 5. predictive contract 草案

一个模型的 predictive hook 至少需要回答以下问题：

```python
class PredictiveDefinition:
    model_name: str
    backend: str
    supported_diagnostics: tuple[str, ...]
    required_external_inputs: tuple[str, ...]
    required_capabilities: tuple[str, ...]
    artifact_schema_version: str
```

运行时接口可以是：

```python
class PredictiveEngine:
    def run_diagnostics(
        self,
        posterior_draws: np.ndarray,
        *,
        parent_sample_size: int,
        random_seed: int,
        options: Mapping[str, Any],
    ) -> PredictiveDiagnosticsPayload:
        ...
```

`PredictiveDiagnosticsPayload` 应该携带：

- replicated observable arrays；
- observed summary inputs；
- trend draw arrays；
- trend axis specs；
- plot panel specs；
- manifest metadata；
- model-specific warnings。

这样 PPT 写 artifact 时不需要知道 CMASS 或 Sonnenfeld 的内部参数含义。

## 6. 分阶段实施计划

### Phase 1: 无科学变化地抽出 CMASS predictive core

目标：先把当前代码拆开，但不改变数值结果和输出 artifact。

工作内容：

1. 从 `Posterior_predictive_test/src/lensing_posterior_predictive/predictive.py`
   中抽出 CMASS-only Numba diagnostics kernel。
2. 将 CMASS parent draw、selection、trend binning、sigma simulation 放入
   `Bayesian_inference` 的 CMASS model-owned predictive module，或 PPT 的
   `adapters/cmass.py` 作为过渡。
3. 保留现有 public API 和 CLI 参数。
4. 测试证明 refactor 前后 `ppc_summary.json`、`fig8_like_summary.json`、
   `replicated_statistics.npz` 的核心字段和 shape 不变。

验收标准：

- 现有 PPT tests 全部通过；
- CMASS diagnostics 输出路径和文件名不变；
- 通用 orchestration 文件中不再出现 CMASS posterior helper import。

### Phase 2: 接入 model registry

目标：PPT 主流程根据 `runtime_config.model.name` 自动选择 predictive hook。

工作内容：

1. 给 `ModelDefinition` 增加 optional predictive hook，或建立等价 registry bridge。
2. PPT 加载 `config_snapshot.yaml` 后，通过 `get_model_definition()` 获取 active model。
3. 用 active model 的 `build_compiled_model()` 构建 context。
4. 如果模型没有 predictive hook，抛出模型名明确的错误。

验收标准：

- `cmass` 通过 registry 跑通；
- unsupported model 错误信息清楚；
- PPT orchestration 不再出现 `if runtime_config.model.name != "cmass"` 的硬编码失败分支。

### Phase 3: 外部输入和 artifact schema 模型化

目标：不同模型可以声明自己需要哪些外部输入和输出 schema。

工作内容：

1. 把 `--sigma-table` 从“所有 diagnostics 必填”改成 model-declared input。
2. CMASS 继续要求 observed-aperture sigma table。
3. Sonnenfeld 后续可选择使用 canonical `population_sigma_unit` 或声明自己的外部输入。
4. artifact manifest 写入：
   - `model_name`
   - `predictive_backend`
   - `predictive_schema_version`
   - `supported_diagnostics`
   - `required_external_inputs`
5. 把 panel order、trend quantity names、observed overlay policy 从 CMASS 常量改为 payload 字段。

验收标准：

- CMASS artifact 仍兼容当前 downstream reader；
- 新 schema 能表达非 CMASS 模型没有 CMASS Fig.8 panels 的情况；
- CLI 对缺失外部输入的错误由模型声明驱动。

### Phase 4: canonical observation contract 补强

目标：减少 PPT 对 raw observation 或 filename convention 的依赖。

工作内容：

1. 为 canonical dataset 增加或读取 explicit observation contract：
   - observation flavor；
   - aperture shape；
   - aperture radius / width / height；
   - seeing FWHM；
   - sigma definition。
2. 如果 observed gamma / mass overlay 需要长期支持，定义 optional observed-summary
   block 或 sidecar schema。
3. PPT 优先使用 canonical metadata；legacy raw HDF5 只作为兼容路径。

验收标准：

- canonical-only run 不再靠文件名推断 slit / boss；
- raw legacy overlay 逻辑与 canonical logic 分离；
- 缺失 optional observed overlay 时，PPT 能清楚降级，而不是静默画错。

### Phase 5: Sonnenfeld predictive hook

目标：让 `sonnenfeld2024_slacs` 和 `sonnenfeld2024_slacs_hunit` 拥有自己的 PPC
diagnostics。

工作内容：

1. 先定义 Sonnenfeld PPC 的科学问题，而不是直接套 CMASS Fig.8：
   - replicated `theta_E` 是否需要；
   - observed sigma likelihood 如何展示；
   - velocity-proxy `theta_E_est` 是否需要 diagnostic；
   - selected / detectable / parent 的趋势轴是什么；
   - hunit 和 paper-native 变体是否共享同一套 plot schema。
2. 从 Sonnenfeld posterior kernel 中抽出共享 population / selection helper。
3. 实现 Sonnenfeld Numba predictive diagnostics。
4. 加小规模 synthetic canonical fixture 测试。

验收标准：

- `sonnenfeld2024_slacs` 和 `sonnenfeld2024_slacs_hunit` 可以被 PPT registry 识别；
- capability 不足时错误指向 `population_sigma_unit` 等真实缺口；
- diagnostics 不复用 CMASS-only gamma mode 或 Fig.8 常量。

### Phase 6: 清理 legacy 和文档化

目标：把历史兼容路径收拢，避免长期污染新架构。

工作内容：

1. 将 pre-registry raw CMASS run snapshot 解析移入 `legacy.py`。
2. 标明 legacy path 只支持 CMASS。
3. 更新 CLI help、README / planning docs。
4. 增加架构测试，防止通用 PPT orchestration 再次 import concrete model。

验收标准：

- legacy tests 仍通过；
- 新模型路径不经过 legacy parser；
- 文档能说明如何为新模型添加 predictive hook。

## 7. 不做的事情

本次重构不应该做以下事情：

1. 不把 PPT CLI、plotting、artifact writer 重新塞回 `Bayesian_inference`。
2. 不引入 JAX / NumPyro 作为 diagnostics backend。
3. 不把 CMASS Fig.8 panel schema 强行定义为所有模型的通用 schema。
4. 不在 PPT 里重新实现 Sonnenfeld inference posterior。
5. 不让 model file 了解 CLI、output directory、PNG writer 等 workflow 细节。
6. 不把 canonical dataset 变成任意 posterior diagnostic 的大杂烩；只有稳定、
   可复用、数据准备阶段自然拥有的信息才进入 canonical schema。

## 8. 最小可交付版本

如果先做一个最小版本，范围应控制为：

1. 增加 predictive hook contract。
2. CMASS 通过 hook 跑通现有 diagnostics。
3. PPT 主流程通过 `model.name` dispatch。
4. 非 CMASS 模型给出明确 unsupported predictive diagnostics 错误。
5. 所有现有 CMASS PPT tests 通过。

这一步完成后，架构边界就已经正确；Sonnenfeld predictive diagnostics 可以作为后续
独立科学实现推进。

## 9. 风险点

1. **科学语义漂移**：如果抽取 shared helper 时没有保持 normalization 和 PPC 使用
   同一套 selection math，会产生难以发现的模型不一致。
2. **artifact 兼容性破坏**：当前 downstream 可能依赖 `ppc_summary.json`、
   `fig8_like_summary.json`、`replicated_statistics.npz` 的字段名。
3. **Sonnenfeld 误套 CMASS schema**：Sonnenfeld selection 和 parent population
   语义不同，必须先定义诊断问题再实现。
4. **canonical schema 过度膨胀**：observed overlay 不能因为 PPT 临时需要就随意塞入
   canonical；应区分 canonical input、derived diagnostic sidecar、legacy raw attrs。
5. **Numba compile / cache 成本**：把 kernels 拆细后需要关注首次编译和 cache 行为，
   避免测试和真实运行变慢。

## 10. 推荐下一步

下一步建议先做 Phase 1 和 Phase 2：

1. 写一个失败测试，断言 PPT generic orchestration 不直接 import CMASS model。
2. 写一个失败测试，断言 `cmass` diagnostics 通过 registry / predictive hook 运行。
3. 抽出 CMASS predictive core。
4. 保持现有 CMASS outputs 和 tests 不变。
5. 再决定 predictive hook 最终挂在 `ModelDefinition` 上，还是先用 PPT-local bridge。

这个顺序可以先修架构边界，同时避免一开始就把 Sonnenfeld PPC 的科学定义问题和
CMASS 代码拆分问题混在一起。
