# Numba Component Architecture Decisions

## 这份文档要解决什么问题

这份文档面向完全不了解本项目的人。它解释我们当前在代码结构设计上面临的核心决策：

```text
在 JAX/NumPyro 已经退出 production path 之后，
我们怎样设计一个仍然容易扩展新层次模型、
同时又能保持 Numba/emcee 高性能的推断代码结构？
```

这个问题不是简单的文件整理问题。它同时牵涉三件事：

1. 科学模型应该怎样表达，才能让未来的人看得懂。
2. 数值计算应该怎样组织，才能让 Numba 真正加速。
3. 新模型应该怎样接入，才能不反复修改 sampler、runner、output 等框架代码。

如果这个边界设计失败，代码会走向两个坏方向之一：

- 过度手写：每个模型都有一整套独立 kernel，短期性能好，但长期难维护。
- 过度抽象：试图用动态组件系统自动生成所有模型，表面优雅，但 Numba 很难高效编译，科学语义也难审计。

我们要找的是中间路线：**静态组件化 + 显式模型 adapter + 共享 Numba kernel 库**。

读完这份文档后，即使不看任何代码，读者也应该能够参与讨论这些问题：

1. 为什么我们需要整理模型架构，而不是只继续给每个模型手写 kernel。
2. 为什么 Numba backend 不能简单复制 JAX 时代的自动组装思路。
3. 哪些模型部件适合做成可复用组件，哪些部件必须保留为显式模型逻辑。
4. 新增一个未来层次模型时，理想上应该改哪些地方，不应该改哪些地方。
5. 怎样判断一次重构是在降低复杂度，还是只是在制造新的抽象负担。

## 项目背景

这个仓库中的 Bayesian inference 代码用于强引力透镜相关的群体模型推断。一个模型通常不只是一个简单公式，而是一整套结构：

- 参数向量，例如质量关系、大小关系、gamma 关系、source redshift 分布、选择函数参数。
- 数据合同，例如 canonical inference dataset 中必须有哪些 HDF5 block 和 capability。
- 预处理逻辑，例如构造 redshift grid、mass grid、velocity-dispersion grid、Monte Carlo random basis。
- likelihood 计算，例如逐 lens 积分、选择归一化、观测 velocity dispersion likelihood。
- sampler 和输出，例如 emcee walker、HDFBackend、metadata、diagnostic blob。

### 最小科学背景

强引力透镜样本不是从宇宙中随机、完整地抽出来的。一个星系系统能否进入样本，取决于它是否产生可观测的透镜信号，也取决于 survey selection、fiber aperture、velocity-dispersion proxy、Einstein radius 等观测和选择效应。

因此，推断一个群体模型时，我们通常要同时计算两类量：

1. **已观测 lens 的 likelihood**：给定某组群体参数，这些已观测 lens 的质量、半光半径、速度弥散、红移等观测值有多可能出现。
2. **选择归一化 normalization**：给定同一组群体参数，整个 parent population 中有多少对象会被选择进样本。

最终 posterior 的典型形状可以粗略理解为：

```text
posterior(theta)
  = observed_lens_likelihood(theta)
    - number_of_lenses * log(selection_normalization(theta))
    + optional_priors(theta)
```

这里的 `theta` 是模型的 sampled hyper-parameters，也就是 MCMC 正在探索的一组参数。

### 什么是层次模型

这里说的层次模型不是一个单独公式，而是一个“生成故事”：

```text
先从 parent population 生成 lens 的红移、恒星质量、大小、密度斜率等真实物理量；
再通过透镜几何和观测模型生成可观测量；
再通过选择函数判断哪些对象会进入样本；
最后比较进入样本的对象与真实观测数据是否一致。
```

这类模型的难点在于，不同论文或不同数据集可能共享某些组件，例如 power-law lensing 几何、Gaussian scatter、cross-section interpolation，但又可能在 selection correction、parent density、proposal distribution、观测 likelihood 上有本质差异。

所以我们希望复用组件，但不能把所有模型差异都藏进一个黑盒自动组装系统里。

### 关键术语

| 术语 | 在本文中的意思 |
| --- | --- |
| `theta` | MCMC 当前 proposal 的参数向量。为了 Numba 性能，它在 hot path 中是固定顺序的 flat array。 |
| likelihood | 给定 `theta`，观测数据出现的概率密度或 log 概率。 |
| normalization | selection-corrected inference 中的选择归一化项，通常需要 Monte Carlo 或数值积分。 |
| component | 一个可复用的科学/数值部件，例如 gamma relation、cross-section interpolation、velocity likelihood。 |
| adapter | 把具体模型的组件按固定顺序串起来的显式执行层。它表达该模型的 posterior 结构。 |
| backend | 真正执行计算和采样的技术栈。当前 production backend 是 Numba + emcee。 |
| hot path | 每次 MCMC proposal 都会反复执行、需要重点加速的代码路径。 |
| canonical dataset | production inference 的标准化 HDF5 输入。它把 raw observation、cross-section、sigma grid 等预处理结果整理成统一合同。 |

当前 production path 是：

```text
canonical inference dataset
  -> ModelSpec + ModelRuntimeAdapter
  -> model-specific Numba kernels
  -> backend-owned posterior reduction and diagnostics
  -> emcee sampler
  -> emcee HDFBackend chain.h5 outputs
```

JAX/NumPyro 已经不是 production dependency。现在的唯一 production backend 是 `numba/emcee`。

### 一次 log-prob 计算在概念上做什么

不看代码，也可以把一次 posterior 计算理解成下面的流程：

```text
输入:
  theta = 当前 MCMC proposal 参数
  context = 数据集、网格、预计算表、随机基等 parameter-independent 信息

步骤:
  1. 检查 theta 是否在允许范围内。
  2. 用 theta 描述 parent population。
  3. 对 parent population 做选择归一化积分。
  4. 对每个已观测 lens 做 likelihood 积分。
  5. 加上可选先验或诊断项。
  6. 返回一个 scalar log probability。

输出:
  log_prob(theta)
  diagnostics，例如 normalization value、timing、FP prior summary
```

架构整理的目标就是决定：上面这些步骤中，哪些应该是通用框架，哪些应该是可复用组件，哪些必须由具体模型显式写清楚。

## 当前已经确定的约束

为了避免讨论发散，先列出当前已经确定的事实和边界。

1. Production backend 已经回到 `numba/emcee`。
2. JAX/NumPyro 不再是 production run path。
3. Production 数据入口是 canonical inference dataset，不是散落的 raw observation/cross-section/sigma-table 路径。
4. 目标不是恢复旧 monolithic 代码，而是在新边界下让模型作者更容易扩展模型。
5. 新模型不应该要求修改 sampler、runner、output 等框架层。
6. Numba hot path 必须保持静态、类型明确、数组/标量输入明确。
7. 科学语义必须可审计，尤其是 selection normalization 和 proposal correction。

## 我们真正面临的架构选择

当前不是在选择“要不要组件化”。组件化方向是明确的。真正的问题是组件化到什么程度。

### 选择 A：每个模型完全手写一套 kernel

优点：

- 性能最容易控制。
- 每个模型的 hot path 最直接。
- 短期实现速度可能最快。

缺点：

- 新模型复制粘贴多。
- 共享公式容易出现多个版本。
- 长期很难知道两个模型的差异是科学差异还是实现漂移。

这个方向适合救急，但不适合作为长期架构。

### 选择 B：做一个动态组件 DSL，自动组装模型

优点：

- 表面上新增模型最方便。
- 配置文件可以很灵活。
- 看起来接近 JAX 时代“函数组合后整体编译”的体验。

缺点：

- Numba 不擅长动态对象图。
- hot path 可能被 Python dispatch 拖慢。
- 自动组装很难表达复杂 selection correction。
- 科学审计会变困难，因为 posterior 结构被隐藏在 DSL 解释器里。

这个方向对当前 production backend 风险过高。

### 选择 C：静态组件化 + 显式 adapter

优点：

- 共享数值组件可以复用和单独测试。
- 每个模型的 posterior 结构仍然显式可读。
- Numba hot path 仍然静态。
- 新模型不需要碰 sampler/output。

缺点：

- 新模型仍然需要一个 adapter。
- 不能实现“任意 YAML 组件自动变成高性能 kernel”的效果。
- 需要维护清楚的 component contract 和测试矩阵。

本文推荐选择 C。

## 当前已有的代码边界

本节列出当前代码中的边界，主要用于把上面的概念映射到仓库。不了解代码的读者可以把文件名当作附录信息；理解架构决策不依赖打开这些文件。

### 1. 模型声明层

模型声明层回答这个问题：

```text
这个模型是什么？
```

当前主要文件：

- `src/cmass_lens_inference/model_interfaces.py`
- `src/cmass_lens_inference/models/cmass.py`
- `src/cmass_lens_inference/models/sonnenfeld2024_slacs.py`
- `src/cmass_lens_inference/models/components/*/parameters.py`

这一层定义：

- 模型名称。
- 参数名称、公开 config 名称、边界。
- 单位约定，例如 `h_units_v1` 或 `legacy_fixed_kpc`。
- 质量 aperture，例如 `m5`。
- 需要的 canonical dataset capability。
- backend kernel key。

这一层应该让科学用户能看懂模型的参数面和数据要求，而不需要理解 emcee、HDF5 output、Numba compilation 或 runner。

长期目标中，参数名和参数顺序不应该在 `models/<model>/parameters.py` 与 component 声明里重复维护。
如果 `ComponentSpec` 已经声明了参数 block，那么 assembly 应该只负责选择组件并决定 block 顺序。
模型目录里如需单独文件，应该放固定科学常数，例如 `constants.py` 或 `paper_constants.py`，而不是再次声明同一批 sampled parameters。

### 2. 运行时上下文层

运行时上下文层回答这个问题：

```text
给定 canonical dataset 和 config，kernel 需要哪些 parameter-independent 数组？
```

当前主要文件：

- `src/cmass_lens_inference/models/cmass_runtime.py`
- `src/cmass_lens_inference/models/sonnenfeld2024_slacs_runtime.py`
- `src/cmass_lens_inference/models/components/cmass/preprocessing.py`
- `src/cmass_lens_inference/models/components/sonnenfeld2024_slacs/preprocessing.py`
- `src/cmass_lens_inference/models/components/*/context.py`

这一层负责：

- 读取 canonical inference dataset。
- 校验 required capabilities。
- 构造 cosmology grid、mass grid、sigma grid。
- 构造 Monte Carlo random basis。
- 做单位约定相关的 deterministic shift。
- 打包成 model-specific context。

重要原则：**context 只放 parameter-independent 数据。** 当前 proposal 的 sampled hyper-parameters 不应该塞进 context，而应该通过 flat `theta` 传给 log-prob kernel。

### 3. Numba 执行层

Numba 执行层回答这个问题：

```text
给定 theta 和 context，怎样快速算 posterior log probability？
```

当前主要文件：

- `src/cmass_lens_inference/numba_backend/primitives.py`
- `src/cmass_lens_inference/numba_backend/cmass_kernels.py`
- `src/cmass_lens_inference/numba_backend/sonnenfeld_kernels.py`
- `src/cmass_lens_inference/numba_backend/likelihood_engine.py`
- `src/cmass_lens_inference/numba_backend/model_adapter.py`

这一层负责：

- Numba-compiled primitive。
- 选择归一化。
- 逐 lens likelihood。
- posterior reduction。
- timing diagnostic blob。
- 根据 `backend_kernel` 分发到具体模型 kernel。

### 4. Sampler 和输出层

Sampler 和输出层回答这个问题：

```text
怎样运行链、保存链、记录 metadata？
```

当前主要文件：

- `src/cmass_lens_inference/runner.py`
- `src/cmass_lens_inference/emcee_sampler.py`
- `src/cmass_lens_inference/outputs.py`
- `src/cmass_lens_inference/posterior_corner.py`

这一层不应该包含模型科学公式。它只应该负责：

- config validation。
- walker 初始化。
- emcee HDFBackend。
- run summary。
- metadata。
- checkpoint。
- posterior corner 读取 production `chain.h5`。

## 当前结构的主要问题

当前结构已经比旧的 monolithic backend 清楚很多，但仍然有一个明显问题：

```text
模型声明层已经变薄了，
但 Numba kernel 层还没有真正组件化。
```

具体表现是：

1. `likelihood_engine.py` 同时承担 backend dispatch、box prior、diagnostics、posterior reduction、CMASS log-prob、Sonnenfeld log-prob。
2. `cmass_kernels.py` 和 `sonnenfeld_kernels.py` 都是大块模型级 kernel 文件。
3. 一些低层 primitive 已经共享，但中层科学组件还没有形成稳定边界。
4. 新增模型时，虽然不需要懂 sampler/output，但仍然很可能需要手写一大段 model-specific Numba kernel。

这就是我们现在面临的架构整理决策。

## 不能照搬 JAX 的原因

JAX 和 Numba 的适合用法不同。

JAX 更适合这样的模式：

```text
一组纯函数
  -> 用 Python 组合
  -> JAX trace
  -> JIT 编译较大的计算图
```

Numba 更适合这样的模式：

```text
静态函数
  -> 明确类型
  -> 明确数组和标量参数
  -> @njit 编译 hot loop
```

因此，在 Numba production backend 下，不现实的目标是：

```text
YAML 里任意选择组件
  -> Python 动态拼装任意模型
  -> Numba 自动生成高性能 posterior kernel
```

这个目标有几个问题：

- Numba 不擅长动态函数列表和动态对象图。
- Numba hot path 不适合字符串 dispatch、字典查找、动态 class 方法调用。
- 每种组件组合都可能产生不同函数签名，强行自动编译会导致复杂的 compilation 和 cache 行为。
- 层次模型的关键区别往往在积分测度、proposal correction、selection normalization，而不是单个公式替换。
- 过度 DSL 化会让科学审计更困难。

所以，我们不应该追求 JAX 式自动组装。我们应该追求 Numba 友好的静态组件化。

## 可行的目标

可行目标是：

```text
未来模型可以复用一批已经写好、已经测试、已经 Numba 加速的组件。

如果新模型只是重新组合已有科学部件，
它只需要很薄的 model adapter。

如果新模型引入新的科学语义，
它需要新增一个清楚命名、独立测试的组件，
然后通过薄 adapter 接入 production path。
```

换句话说：

- 组件可以复用。
- 组件可以加速。
- 模型可以更容易组装。
- 但最终 production model 仍然需要一个显式的静态 adapter。

## 建议采用的核心设计

### 决策 1：保留 `ModelSpec`，但让它描述组件选择

`ModelSpec` 现在已经承担模型声明职责。下一步可以让它进一步表达：

- 由 component 聚合出来的参数 block。
- capability requirements。
- unit/mass contract。
- component composition metadata。
- backend adapter key。

但不要让 `ModelSpec` 直接驱动 Numba 动态编译。

推荐语义：

```text
ModelSpec 是科学声明，
不是 Numba 编译器输入。
```

### 决策 2：新增组件声明层

建议新增一个轻量 `ComponentSpec` 概念，用于描述一个科学组件。

一个组件应该说明：

- `name`：组件名。
- `kind`：组件类别，例如 population、selection、likelihood、prior。
- `parameters`：它拥有或消费哪些 sampled parameters，包括参数名、公开名、边界和 block 内顺序。
- `required_context_fields`：它需要 context 中有哪些数组或标量。
- `required_capabilities`：它要求 canonical dataset 具备哪些 capability。
- `provided_quantities`：它输出哪些中间量。
- `numba_kernel`：对应的 Numba 加速实现。
- `reference_tests`：它必须通过哪些 reference 数值测试。

这个 `ComponentSpec` 应该先服务于：

- 文档。
- 参数 schema。
- capability audit。
- context field audit。
- 测试矩阵。

它不应该一开始就承担“自动生成 Numba kernel”的职责。

`ComponentSpec` 应该成为参数 block 的主要来源。assembly 可以重排或组合多个 block，但不应该再次手写同一批参数名。
如果某个参数无法自然归属到已有 component，优先新增一个小的 model-level component，而不是在模型目录里另建一个重复的 `parameters.py`。

### 决策 3：让 `assembly.py` 成为模型侧聚合点

模型目录中应该有一个清楚的聚合入口：

```text
models/<model>/assembly.py
```

它负责：

- 选择该模型使用哪些 components。
- 决定 component parameter blocks 的全局顺序。
- 聚合所有 component 的 required capabilities。
- 添加少量模型级 metadata、unit contract、mass aperture。
- 生成最终 `ModelSpec`。

它不应该：

- 重新手写每个 component 已经声明过的参数名。
- 把 capability requirement 分散到另一个模型级文件中维护。
- 包含 runtime preprocessing。
- 包含 Numba hot path。

这意味着 `models/<model>/capabilities.py` 通常不需要存在。
如果 capability 已经由 components 声明，assembly 直接聚合即可。
只有当模型有无法归属到任何 component 的全局数据要求时，才应在 assembly 中显式追加，并在注释中解释为什么它不是 component capability。

模型目录中的常数文件应该只放固定科学常数：

```text
models/<model>/constants.py
models/<model>/paper_constants.py
```

例如论文表格常数、固定 pivot、固定 aperture 默认值等。它不应该成为 sampled parameter schema 的第二来源。

### 决策 4：把 Numba kernel 分成三层

建议 Numba backend 形成三层。

第一层：低层 primitive。

```text
numba_backend/kernels/distributions.py
numba_backend/kernels/interpolation.py
numba_backend/kernels/lensing.py
numba_backend/kernels/math.py
```

这些函数足够底层，例如：

- normal pdf/cdf。
- truncated normal。
- skew normal transform。
- axis bracket。
- 1D/2D/4D interpolation。
- trapezoid integration。
- power-law Einstein radius。

第二层：中层科学组件。

```text
numba_backend/kernels/population.py
numba_backend/kernels/selection.py
numba_backend/kernels/observables.py
numba_backend/kernels/likelihood.py
numba_backend/kernels/priors.py
```

这些函数有明确科学语义，例如：

- parent mass density。
- size relation。
- gamma relation。
- source redshift density。
- discovery probability。
- finite-fibre cross-section selection。
- velocity-dispersion likelihood。
- stellar-mass quadrature。
- Fundamental Plane sufficient statistics。

第三层：模型生产绑定，放在 `models/` 侧。

```text
models/cmass/production.py
models/sonnenfeld2024_slacs/production.py
```

这一层负责把一个具体模型的 posterior 逻辑串起来：

```text
theta
  -> population draw / density
  -> selection normalization
  -> per-lens likelihood
  -> optional prior or diagnostic
  -> posterior total
```

这一层仍然是显式的，因为它表达了模型的积分结构和 selection correction 结构。
它应该归模型所有，而不是归 backend 所有，因为新增模型时，这一层最容易变化。
backend 只提供通用执行器和共享 kernel。

这里需要区分两个容易混淆的词：

- **model-owned adapter**：位于 `models/<model>/production.py`，描述这个模型怎样把组件串成 posterior。
- **backend-owned factory**：位于 `numba_backend/`，负责把模型侧给出的 spec、runtime context 和 kernel callable 包装成可被 sampler 调用的 compiled model。

前者随模型增加而变化；后者只应随 backend 机制变化。

### 决策 5：拆薄 `likelihood_engine.py`

`likelihood_engine.py` 应该变成真正的 backend engine，而不是模型实现文件。

它应该保留：

- host-side box prior rejection。
- adapter lookup。
- common diagnostic blob。
- common timing wrapper。
- unsupported kernel error。

它不应该继续直接实现：

- `_cmass_log_prob`
- `_sonnenfeld_log_prob`
- 大量 model-specific 参数传递

这些应该移动到模型侧的 production 绑定文件，例如：

```text
models/cmass/production.py
models/sonnenfeld2024_slacs/production.py
```

### 决策 6：保留 flat `theta`，不要在 hot path 使用动态对象

为了 Numba 性能，hot path 中参数应该继续是固定顺序的 flat array。

Python 层可以保持可读：

```text
mu5_0
beta5
xi5
sigma5
...
```

但进入 Numba kernel 后应该是：

```text
theta[0]
theta[1]
theta[2]
...
```

可以用 helper 做 unpack，但不要在 hot loop 里做字典或字符串查找。

### 决策 7：新模型不应该直接修改 runner/sampler/output

新增模型时，理想流程应该是：

1. 新增或复用 `ComponentSpec`。
2. 新增 `models/<model>/assembly.py`。
3. 新增 `models/<model>/runtime.py`。
4. 新增 `models/<model>/production.py`。
5. 如有模型特有固定常数，新增 `models/<model>/constants.py`。
6. 在 `model_registry.py` 注册一次。
7. 添加 component tests、adapter tests、short emcee smoke test。

不应该需要修改：

- `runner.py`
- `emcee_sampler.py`
- `outputs.py`
- `posterior_corner.py`

如果新增模型需要改这些文件，通常说明边界设计仍然不够好。

## 推荐目标目录结构

下面是建议的中期目标结构。它不是一次性大爆炸重构目标，而是用来指导后续逐步整理。

```text
src/cmass_lens_inference/
  model_interfaces.py
  model_registry.py

  components/
    interfaces.py
    registry.py

    distributions/
      gaussian.py
      truncated.py
      skewnorm.py

    lensing/
      powerlaw.py
      cross_section.py
      sigma_unit.py

    population/
      mass_function.py
      size_relation.py
      gamma_relation.py
      source_redshift.py

    selection/
      discovery.py
      velocity_proxy.py
      finite_fibre.py

    likelihood/
      stellar_mass_quadrature.py
      velocity_dispersion.py

    priors/
      fundamental_plane.py

  numba_backend/
    engine.py
    diagnostics.py
    compiled_model_factory.py

    kernels/
      distributions.py
      interpolation.py
      lensing.py
      population.py
      selection.py
      likelihood.py
      integration.py
      priors.py

  models/
    cmass/
      assembly.py
      runtime.py
      production.py
      constants.py
      context.py

    sonnenfeld2024_slacs/
      assembly.py
      runtime.py
      production.py
      paper_constants.py
      context.py
```

这套结构里有一个刻意的双层设计：

- `components/`：面向科学和架构阅读者，描述组件是什么。
- `numba_backend/kernels/`：面向性能实现，提供组件的 Numba 加速函数。

不要把这两者混成一层。科学组件声明可以更可读；Numba kernel 必须更静态、更底层。

## 推荐迁移路径

### Phase 1：先拆 engine 和 model adapter

目标：

```text
让 likelihood_engine.py 不再承载具体模型逻辑，
让新增模型时最先改动的位置留在 models/。
```

动作：

- 新增 `models/<model>/production.py`。
- 把 CMASS log-prob 组装移动到 `models/cmass/production.py`。
- 把 Sonnenfeld log-prob 组装移动到 `models/sonnenfeld2024_slacs/production.py`。
- `likelihood_engine.py` 只做 dispatch、box prior、diagnostics。

这一步风险最低，因为不改变数学，只改变文件边界。

### Phase 2：拆低层 primitive

目标：

```text
把 primitives.py 拆成按数值职责命名的 kernel 模块。
```

动作：

- `distributions.py`：normal、truncated normal、skew normal。
- `interpolation.py`：axis bracket、1D interpolation、cross-section interpolation。
- `lensing.py`：cosmology lookup、Einstein radius。
- `selection.py`：discovery probability。
- `integration.py`：trapezoid。

这一步必须要求零行为变化。

### Phase 3：提炼已被两个模型共享的中层组件

优先提炼真实共享组件，而不是凭想象预先抽象。

候选组件：

- power-law lensing geometry。
- `theta_E x gamma` cross-section selection。
- sigma-unit interpolation。
- velocity-dispersion likelihood。
- source-redshift density。
- truncated proposal correction。

每提炼一个组件，都要有对应 reference test。

### Phase 4：引入 `ComponentSpec`

目标：

```text
让模型 assembly 能清楚说明自己由哪些科学组件组成。
```

注意：这一步先不要自动生成 Numba kernel。

`ComponentSpec` 先用于：

- 参数检查。
- capability 检查。
- context field 检查。
- 文档输出。
- 测试覆盖映射。

这一步同时要明确两个 ownership 规则：

- sampled parameter 的名字、公开名、边界和 block 内顺序由 component 声明；assembly 只负责聚合和排序 component block。
- capability requirement 由 component 声明；assembly 负责取并集、处理模型级额外要求，并生成最终 `ModelSpec.required_capabilities`。

### Phase 5：改写 CMASS 和 Sonnenfeld assembly

目标：

```text
让现有两个 production 模型成为组件化结构的示范样例。
```

CMASS assembly 应该表达：

- parent population mass function。
- size relation。
- enclosed mass relation。
- gamma relation。
- source-redshift distribution。
- lensing cross-section selection。
- optional Fundamental Plane prior。
- per-lens likelihood。
- 从上述 components 聚合出的参数顺序和 capability 集合。

Sonnenfeld assembly 应该表达：

- Table-1 parent mass/redshift density。
- quadratic size relation。
- enclosed mass relation。
- gamma relation。
- source-redshift density。
- velocity-proxy selection。
- finite-fibre cross-section。
- per-lens likelihood。
- 从上述 components 聚合出的参数顺序和 capability 集合。

### Phase 6：用 toy model 验收扩展性

目标：

```text
验证新增模型是否真的不需要碰 runner/sampler/output。
```

toy model 不需要有最终科学意义。它只需要证明架构边界成立：

- 复用已有 component。
- 新增一个薄 adapter。
- 注册一次。
- 能跑 synthetic log-prob。
- 能跑 short emcee。

## 不建议做的事情

### 不建议做完整动态 DSL

不要在当前阶段设计类似：

```yaml
model:
  components:
    - mass_relation: ...
    - gamma_relation: ...
    - selection: ...
```

然后期望系统自动生成所有 Numba posterior kernel。

这个设计看起来灵活，但会引入过多动态行为，和 Numba 的优势方向冲突。

### 不建议一次性重写所有 kernel

现在 CMASS 和 Sonnenfeld production path 已经能跑。下一步应该是保守整理，而不是从头写一个新 backend。

每次拆分都应该满足：

- 数值行为不变。
- 测试先覆盖。
- benchmark 不明显退化。
- 文件职责更清楚。

### 不建议过早抽象只有一个模型使用的公式

如果一个公式目前只有一个模型使用，可以先留在 model adapter 或 model-specific kernel 中。

只有当第二个 production 模型真实需要它时，再提升为 shared component。

## 决策标准

后续每次讨论一个重构方向，可以用下面的问题判断是否值得做。

### 这个抽象是否真的降低新增模型成本？

如果它只是把代码从一个文件移动到另一个文件，但新增模型仍然要复制大段 kernel，那么价值有限。

### 这个抽象是否保持科学语义可审计？

如果一个科学用户无法从 assembly 和 component spec 看出模型的 selection correction、likelihood 和 normalization 是什么，这个抽象就是失败的。

### 这个抽象是否符合 Numba 的性能模型？

如果它要求 hot path 使用动态对象、字符串 dispatch、字典 lookup 或 Python callback，它大概率不适合 production kernel。

### 这个抽象是否保护 sampler/output 不被模型污染？

新增模型不应该让 `runner.py`、`emcee_sampler.py`、`outputs.py` 变复杂。

### 这个抽象是否能被单元测试锁住？

每个共享 component 都应该有独立数值测试。否则它被多个模型复用后，任何修改都会扩大风险。

## 外部读者可以怎样参与评审

如果读者不熟悉当前代码，也仍然可以从下面几个角度给出有效建议。

### 1. 评审科学边界

可以检查：

- 哪些部分是不同模型真正共享的科学结构。
- 哪些部分只是现在两个模型碰巧长得像，但科学语义不同。
- selection normalization 是否被当作一等公民处理，而不是被埋进某个 helper。
- proposal correction 是否在模型 adapter 中显式可见。
- 单位约定和质量 aperture 是否足够早地进入模型合同。

### 2. 评审组件粒度

可以检查：

- 一个 component 是否有清楚的输入、输出、参数和上下文要求。
- component 是否小到能单独测试，又大到有真实科学语义。
- 是否存在“为了漂亮分层而拆得过细”的模块。
- 是否存在“明明两个模型共享，却仍然复制实现”的逻辑。

### 3. 评审 Numba 可行性

可以检查：

- hot path 是否依赖动态 Python 对象。
- 是否需要在每次 likelihood 调用中做字符串 dispatch 或字典查找。
- 参数顺序是否固定。
- context 是否只包含 parameter-independent 数据。
- adapter 是否可以被 Numba 友好地调用或包裹。

### 4. 评审新增模型路径

可以用一个假想模型做思想实验：

```text
假设新增一个模型，它复用现有 power-law lensing 和 velocity likelihood，
但有新的 size relation 和新的 selection function。
```

然后检查这个模型应该改哪些地方：

- 应该新增 size-relation component。
- 应该新增 selection component。
- 应该新增一个 model assembly。
- 应该新增一个 model adapter。
- 应该注册模型。

同时检查它不应该改哪些地方：

- 不应该改 emcee sampler。
- 不应该改 output writer。
- 不应该改 posterior corner reader。
- 不应该改 canonical dataset reader 的通用部分，除非数据合同本身扩展了。

### 5. 评审验收是否足够硬

可以检查验收标准是否覆盖：

- 数值等价。
- 单组件 reference tests。
- synthetic model smoke tests。
- real-data regression。
- benchmark。
- failure mode，例如 missing capability、theta 越界、normalization 非法。

## 仍然开放的问题

下面这些问题不要求在第一轮重构中全部解决，但它们是架构讨论中应该持续追踪的风险。

1. `ComponentSpec` 是否只做文档和审计，还是未来允许生成一部分 adapter boilerplate？
2. 模型 adapter 应该是纯 Python 调用 Numba kernels，还是 adapter 本身也应该尽量 `@njit`？
3. context field audit 应该在 config load 时做，还是在 build compiled model 时做？
4. toy model 应该模拟真实层次模型的哪些复杂性，才足以验收扩展能力？
5. 共享组件的 reference tests 应该以 analytic fixture 为主，还是以冻结数值 fixture 为主？
6. 如果未来某个模型需要完全不同的 sampler，是否应该扩展 backend interface，还是明确开新 production backend？

## 最终推荐

最终推荐采用：

```text
ModelSpec 声明科学模型
ComponentSpec 描述可复用科学组件
RuntimeAdapter 构造 parameter-independent context
Numba kernels 实现低层和中层加速组件
Model adapter 显式组装具体 posterior
emcee/runner/output 保持框架层职责
```

这意味着未来新增模型有三种路径。

### 路径 A：完全复用已有组件

适合新模型只是重新组合已有 mass relation、selection、likelihood。

工作量：

- 新增 assembly。
- 新增 runtime。
- 新增薄 adapter。
- 注册模型。
- 添加测试。

### 路径 B：复用大部分组件，但新增一个科学组件

适合新模型只有一个局部公式不同。

工作量：

- 新增 component spec。
- 新增对应 Numba kernel。
- 新增 component reference test。
- 新增薄 adapter。

### 路径 C：引入新的积分结构或 selection correction

适合真正不同的层次模型。

工作量：

- 新增多个 component。
- 新增更厚的 model adapter。
- 增加 synthetic/reference fixture。
- 增加 performance benchmark。

这不是架构失败。层次模型的积分结构本来就是模型科学的一部分，应该显式存在，而不是被隐藏在自动组装系统里。

## 验收标准

这次架构整理如果成功，应该满足：

1. 新增模型不需要修改 `runner.py`、`emcee_sampler.py`、`outputs.py`。
2. `likelihood_engine.py` 不再包含 CMASS 或 Sonnenfeld 的大块模型逻辑。
3. 共享 Numba kernel 有明确模块边界和独立测试。
4. CMASS 和 Sonnenfeld 的现有 production tests 继续通过。
5. real-data CMASS equivalence 不退化。
6. synthetic Sonnenfeld log-prob 和 short emcee run 不退化。
7. steady-state log-prob benchmark 没有明显性能回退。
8. 新增 toy model 可以证明组件化接入路径真实可用。

## 一句话总结

我们现在要做的不是把 Numba 伪装成 JAX，而是建立一个 Numba 友好的模型组件体系：

```text
科学声明足够清楚，
组件边界足够稳定，
hot path 足够静态，
model adapter 足够显式，
framework 层足够干净。
```

这条路线牺牲了一部分“任意动态组装”的想象空间，但换来的是 production 性能、科学可审计性和长期可维护性。
