# Canonical Inference Dataset Schema Design

本文档记录当前 CMASS / Sonnenfeld 2024 SLACS 模型切换架构下的
canonical inference dataset 设计。它是一份架构设计记录，不是最终的
machine-readable HDF5 schema specification。后续实现数据准备模块、
canonical dataset reader 和 schema validator 时，应以本文档的边界为准，
再把字段类型、shape、attribute 名称和版本迁移规则细化成可测试代码。

## 设计目标

canonical dataset 的职责是为 Bayesian inference 提供稳定、单位一致、
shape 明确的数值输入。它只保存以下内容：

- lens-level 观测量；
- 预计算 lensing mass grid；
- 预计算 lensing cross-section grid；
- velocity-dispersion / sigma 相关预计算网格；
- unit convention、profile、mass definition、schema version 等 metadata。

canonical dataset 不负责定义科学概率模型。以下内容属于 model 或 runtime，
不进入核心数据 schema：

- foreground population，例如 CMASS 的 `P(z_d) * P(m*)` 或 Sonnenfeld 的
  Table 1 `P_g(z_d, m*)`；
- Sonnenfeld 2024 Table 1 参数；
- `P_g(z_d, m*)`、`P_s(z_s)`、`P_find(...)` 的数学形式；
- NumPyro/JAX inference 的 fixed random basis；
- `normalization_samples`、sampler seed、chain settings 等 runtime 控制量；
- `parent_sample.fits` 这类用于论文上游拟合的原始 parent sample。

这个边界的核心原因是：canonical dataset 应该描述“观测和预计算数值对象”，
model 文件描述“这些对象如何进入概率生成模型”。这样 CMASS 与 Sonnenfeld 的
数据读取路径可以共享，而模型差异保留在 `models/*.py` 的科学公式中。

## 顶层 Schema Blocks

建议 canonical HDF5 文件使用以下顶层 blocks：

```text
inference_dataset.hdf5
├── metadata
├── lenses
├── lensing_mass_grids
├── lensing_cross_section
└── velocity_dispersion_grids
```

每个 block 的职责如下。

## `metadata`

`metadata` 描述整个文件的数据契约。它用于启动 inference 前的快速校验，
不承载模型公式。

建议字段：

```text
schema_version
unit_convention
h_ref
profile_name
mass_definition_label
mass_radius_kpc
cosmology_h0
cosmology_omega_m
capabilities
```

`capabilities` 用于声明文件包含哪些可用数值能力，例如：

```text
lens_observations.v1
lensing_mass_grids.v1
lensing_cross_section.theta_gamma_grid.v1
velocity_dispersion.per_lens_s2.v1
velocity_dispersion.population_sigma_unit.v1
velocity_dispersion.fp_within_re.v1
```

这里的 capabilities 不是模型选择入口。模型只根据自己的需求检查 capability
是否存在；模型的参数、population distribution 和 selection 公式仍由 model
模块定义。

## `lenses`

`lenses` 保存每个 lens 的 canonical observation record。字段应当已经完成单位
转换和 raw-file alias normalization，inference runtime 不应再处理原始 HDF5
别名或 arcsec 到 physical radius 的转换。

建议字段：

```text
lens_id                  # [N_lens]
z_d                      # [N_lens]
z_s                      # [N_lens]
log_mstar_obs            # [N_lens]
log_mstar_err            # [N_lens]
log_re_obs               # [N_lens]
n_obs                    # [N_lens]
theta_e_obs              # [N_lens]
num_sigma                # [N_lens]
sigma_obs                # [N_lens, N_sigma_max]
sigma_err                # [N_lens, N_sigma_max]
```

语义要求：

- `log_mstar_obs` 必须使用 `metadata.unit_convention` 对应的质量坐标。
- `log_re_obs` 必须是同一 unit convention 下的物理 size 坐标。
- `num_sigma = 0` 表示该 lens 没有 velocity-dispersion likelihood。
- `sigma_obs` / `sigma_err` 的第二维可固定为当前项目需要的 `N_sigma_max = 2`。
  当 `num_sigma < N_sigma_max` 时，未使用槽位不得参与 likelihood。

这部分与当前 `ObservationRecord` 高度对应，也是 CMASS 与 Sonnenfeld 共享程度
最高的输入层。

## `lensing_mass_grids`

`lensing_mass_grids` 保存每个 lens 沿 `gamma` 轴预计算出的 lensing mass 轨迹。
CMASS 和 Sonnenfeld 都可以沿 `m5(gamma)` 或相同语义的 enclosed-mass 轨迹做
per-lens likelihood integral。

建议字段：

```text
gamma_grid                 # [N_lens, N_gamma] 或 [N_gamma]
log_enclosed_mass_grid      # [N_lens, N_gamma]
dmass_dthetaein_grid        # [N_lens, N_gamma]
s2_grid                     # [N_lens, N_gamma]
has_s2                      # [N_lens]
```

`s2_grid` 的约定需要特别严格：

- `s2_grid` 可以统一保存为 `[N_lens, N_gamma]`。
- `num_sigma = 0` 时，该 lens 不存在 velocity-dispersion likelihood；
  inference 必须忽略该 lens 的 `s2_grid` 数值。
- `num_sigma > 0` 且 `has_s2 = true` 时，使用该 lens 的 `s2_grid` 计算
  `sigma_model`。
- `num_sigma > 0` 且 `has_s2 = false` 时，dataset validation 或 inference
  startup 必须失败，因为观测 sigma 存在但缺少对应模型网格。
- 对 `num_sigma = 0` 的 lens，`s2_grid` 填充值没有物理意义；实现不得依赖其
  数值。JAX runtime 可以在加载时用 0 填充，配合 `num_sigma` / `has_s2`
  mask 避免无效值进入 likelihood。

`dmass_dthetaein_grid` 的具体命名后续可以在 machine-readable schema 中收紧。
当前设计只要求它提供从 observed Einstein-radius constraint 到 mass-grid
积分所需的 Jacobian。

## `lensing_cross_section`

`lensing_cross_section` 保存 lensing/source-plane cross-section 预计算结果。
这个 block 不叫 `selection_cross_section`，因为完整 selection 还包括模型中的
`P_find(...)` 或其他发现概率；数据里保存的只是 lensing cross-section 数值网格。

统一使用二维 `theta_E x gamma` 网格：

```text
theta_e_axis              # [N_theta]
gamma_axis                # [N_gamma_cs]
cross_section_grid         # [N_theta, N_gamma_cs]
boundary_policy
```

推荐插值语义：

```text
g = interp2d(theta_E, gamma, cross_section_grid)
```

设计决策：

- CMASS 当前的一维 separable cross-section 不在 inference runtime 中保留特殊形态。
  数据准备阶段应将它转换为二维网格：

  ```text
  cross_section(theta_E, gamma)
    = pi * (cs_over_theta_ein(gamma) * theta_E)^2
  ```

- Sonnenfeld 可以直接写入 finite-fibre / seeing-convolved / flux-thresholded 的
  `g(theta_E, gamma)` 网格。
- inference backend 只消费统一二维接口，不关心 cross-section 是否来自 separable
  近似。
- `boundary_policy` 应在后续 validator 中变成明确枚举。首选策略是
  `zero_outside_theta_clip_gamma`：`theta_E` 超出可用范围时 cross-section 置零，
  `gamma` 轴可按预计算网格规则 clip 或 zero，具体实现需与生成网格时的物理约定
  保持一致。

统一二维接口的收益是：CMASS 和 Sonnenfeld 的 likelihood / normalization 都可以
共享同一个 cross-section reader 和 JAX interpolation helper。

## `velocity_dispersion_grids`

velocity-dispersion 相关网格分为 per-lens likelihood 用途和 population-level
normalization / optional prior 用途。建议在同一个顶层 block 下分子组：

```text
velocity_dispersion_grids
├── per_lens_s2
├── population_sigma_unit
└── fp_within_re
```

### `per_lens_s2`

`per_lens_s2` 可以直接引用或镜像 `lensing_mass_grids.s2_grid`，用于已观测 lens 的
velocity-dispersion likelihood：

```text
sigma_model(gamma)
  = sqrt(s2_grid(gamma) * 10**log_enclosed_mass_grid(gamma))
```

这里必须遵守 `num_sigma` / `has_s2` mask 约定。

### `population_sigma_unit`

`population_sigma_unit` 用于 latent population draw 的 velocity-dispersion proxy。
它不是 lens-level 观测数据，而是为了在 normalization Monte Carlo 中从
`(gamma, z_d, R_e, n)` 等 latent variables 预测 sigma。

建议字段：

```text
gamma_axis
z_d_axis
log_re_axis
n_axis optional
sigma_unit_grid
aperture_metadata
```

Sonnenfeld 如果使用 velocity-dispersion proxy 构造 `theta_E_est`，可以消费这个
block。CMASS 当前如果不需要该 proxy，可以不要求这个 capability。

### `fp_within_re`

`fp_within_re` 支持 CMASS 当前 optional FP prior。它可以与
`population_sigma_unit` 共享类似结构，但 aperture / sigma definition 不同：

```text
gamma_axis
z_d_axis optional
log_re_axis
n_axis optional
sigma_unit_grid
sigma_definition = within_re
aperture_metadata
```

这个 block 是 optional capability；模型或 prior 没有启用时不应成为 inference
必需输入。

## CMASS 与 Sonnenfeld 的数据需求对照

CMASS 默认模型预计需要：

```text
metadata
lenses
lensing_mass_grids
lensing_cross_section
velocity_dispersion_grids.per_lens_s2      # 当 num_sigma > 0 时需要
velocity_dispersion_grids.fp_within_re     # 仅 FP prior 启用时需要
```

CMASS 不需要 canonical dataset 提供 foreground population。当前模型中的
`z_d` Gaussian、stellar-mass skew-normal、size relation 和 gamma relation 都属于
`models/cmass.py` 的科学公式或模型常数。

Sonnenfeld 2024 SLACS 模型预计需要：

```text
metadata
lenses
lensing_mass_grids
lensing_cross_section
velocity_dispersion_grids.per_lens_s2          # lens-level sigma likelihood
velocity_dispersion_grids.population_sigma_unit # theta_E_est proxy / normalization
```

Sonnenfeld 不需要 runtime 读取 `parent_sample.fits`，也不需要 canonical dataset
保存 `P_g(z_d, m*)` 的 fitted auxiliary table。若实现的是论文模型本身，
Sonnenfeld Table 1 参数应作为 `models/sonnenfeld2024_slacs.py` 的模型常数，
并由该模型按公式计算 foreground population。

## 模型与 Runtime 的边界

模型文件负责：

- sampled parameter schema；
- foreground population 公式；
- source redshift distribution；
- mass / gamma / size relation；
- `P_find(...)` 公式；
- per-lens likelihood integrand；
- selection normalization integrand；
- optional prior / diagnostics。

canonical dataset reader 负责：

- 验证 schema version 和 capabilities；
- 验证 unit convention、`h_ref`、profile、mass definition metadata；
- 加载 lens observations 和预计算网格；
- 将旧 CMASS separable cross-section 转换后的二维 grid 提供给 runtime；
- 对 `num_sigma` / `has_s2` 的不一致状态 fail fast。

runtime 负责：

- 根据 config 生成 fixed random basis；
- 将 canonical dataset 转成模型需要的 source context；
- 根据 `DataSpec` 构造 JAX context；
- 调用通用 JAX / NumPyro backend。

## 后续实现建议

首个实现阶段不应同时重写科学模型。建议顺序为：

1. 为当前 CMASS 输入写 canonical dataset writer / converter。
2. 为 canonical dataset 写 reader 和 validator。
3. 将现有 CMASS inference runtime 改为读取 canonical dataset，并验证 log-prob
   与当前路径等价。
4. 将旧 CMASS 一维 cross-section 在数据准备阶段转换成二维
   `lensing_cross_section`。
5. 在同一 schema 上实现 Sonnenfeld 2024 所需的二维 cross-section 和
   `population_sigma_unit` grid。
6. 最后实现 `models/sonnenfeld2024_slacs.py` 的真实数值模型。

这个顺序避免把数据 schema 重构、CMASS 等价性验证和 Sonnenfeld 科学模型实现
混在同一轮改动中。
