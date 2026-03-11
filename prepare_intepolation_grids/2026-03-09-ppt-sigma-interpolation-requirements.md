# Posterior Predictive Test 的 Sigma 插值表需求

## 背景
`Posterior_predictive_test` 线程已经锁定 posterior predictive sample 的生成逻辑：

- 对每个 posterior hyper-parameter draw，先从模型定义的总体分布生成候选 lenses
- 用 strong-lens selection 权重
  `w_sel ∝ P_find(theta_E) * pi * (cs(gamma) * theta_E)^2`
  定义 selected distribution
- 从该 selected distribution 中独立抽两次：
  - 一次抽 23 个 replicated lenses，用于 `theta_E` 的 PPT
  - 一次再抽 7 个 replicated lenses，用于 `sigma` 的 PPT

这里需要 `prepare_intepolation_grids` 线程支持的，不再是逐星系 `s2_grid`，而是一张适用于任意 replicated lens 的大插值表。

## 目标
请评估并尽量实现一张可供 PPT 线程直接消费的 Jeans 插值表，使其能够对任意 replicated lens 计算：

`sigma_model = sqrt(S_unit * 10**m5)`

其中：

`S_unit = sigma^2 / 10**m5`

也就是说，插值表应提供“单位质量归一化的 Jeans 响应”，而不是最终的 `sigma` 本身。

## 为什么不直接插值 sigma
- `sigma^2` 对质量归一化是线性的。
- replicated lens 已经保留了该 lens 的 `m5`。
- 因此只要插值出 `S_unit`，PPT 线程就能在后处理阶段恢复物理 `sigma`。

这样做的好处是：
- 减少插值维度
- 把 Jeans 核心依赖的变量与 population-level 抽样变量解耦
- 避免把 `theta_E` / `z_s` / `m5` 这类不属于 Jeans 本体的量硬塞进插值表

## 推荐插值维度
建议分两张表，而不是混成一张：

### 1. devauc
推荐表：

`S_unit_devauc(gamma, z_d, log10(R_e/kpc))`

### 2. sersic
推荐表：

`S_unit_sersic(gamma, z_d, log10(R_e/kpc), n)`

## 明确不应作为插值轴的量
以下量不建议进入 Jeans 插值轴：

- `m5 / M5`
- `theta_E`
- `z_s`
- `mstar`

原因：
- `m5 / M5` 可在线性恢复阶段处理
- `theta_E` 只通过 mass normalization 间接影响 `sigma`
- `z_s` 不属于 Jeans 核心本体
- `mstar` 影响的是 replicated lens 的总体生成层，不是 Jeans 解本身

## 物理口径
请沿用当前 `prepare_intepolation_grids` 的生产口径：

- `aperture_width = 1.6 arcsec`
- `aperture_height = 0.9 arcsec`
- `seeing = 0.9 arcsec`

不要退回历史 `0.8 arcsec` slit 宽度。

## 与现有代码的关系
请优先评估以下现有实现是否可以直接复用或轻量扩展：

- `/Users/liurongfu/Work/CMASS_lens_project/prepare_intepolation_grids/interpolation_grids/physics/jeans.py`
- `/Users/liurongfu/Work/CMASS_lens_project/prepare_intepolation_grids/interpolation_grids/models.py`
- `/Users/liurongfu/Work/CMASS_lens_project/prepare_intepolation_grids/interpolation_grids/io/hdf5.py`
- `/Users/liurongfu/Work/CMASS_lens_project/prepare_intepolation_grids/tests/test_jeans_regression.py`

当前实现偏向“逐真实 galaxy 生成 `s2_grid`”。
这次需求是“对任意 replicated lens 提供可查询的单位质量 Jeans 响应”。

如果现有 `compute_s2_grid()` 的职责不适合直接扩展，请新增更稳定、更清晰的插值表构建入口，而不要把原函数改成职责混杂的万能接口。

## 推荐网格覆盖范围
基于当前真实样本范围，PPT 线程建议至少覆盖：

- `gamma`: `[1.2, 2.8]`
- `z_d`: `[0.43, 0.82]`
- `devauc log10(R_e/kpc)`: `[0.45, 1.20]`
- `sersic log10(R_e/kpc)`: `[0.50, 1.40]`
- `sersic n`: `[2.5, 10.5]`

如果你认为需要更宽的缓冲范围，请明确说明理由。

## 建议输出格式
PPT 线程当前实现优先消费 `.npz`，建议字段名如下：

### devauc
- `profile_name`
- `gamma_axis`
- `zd_axis`
- `log_re_kpc_axis`
- `s_unit_grid`

### sersic
- `profile_name`
- `gamma_axis`
- `zd_axis`
- `log_re_kpc_axis`
- `n_axis`
- `s_unit_grid`

如果实现线程更倾向于 HDF5，也可以提出，但请保证字段名和轴定义足够明确，避免 PPT 线程再做推测性兼容。

## PPT 线程会如何消费这张表
PPT 线程在生成 replicated lens 后，会保留这些 latent 字段：

- `theta_E`
- `gamma`
- `z_d`
- `z_s`
- `profile`
- `m5`
- `R_e_kpc`
- `n`（仅 `sersic`）

随后：

1. 根据 `profile` 选择对应表
2. 在表上查询 `S_unit`
3. 计算 `sigma_model = sqrt(S_unit * 10**m5)`
4. 叠加观测噪声：
   `sigma_rep ~ Normal(sigma_model, 0.0625 * sigma_model)`

## 验收标准
如果实现该需求，至少应满足：

- 能分别产出 `devauc` 和 `sersic` 的插值表
- 表轴定义与单位清晰
- 在网格点上可与直接 Jeans 计算结果比对
- 插值结果在表范围内稳定、有限、无明显非物理跳变
- 注释清楚说明：
  - 这张表解决什么问题
  - 为什么选这些维度
  - 为什么 `m5` 不在插值轴中
  - 与现有逐星系 `s2_grid` 生产逻辑是什么关系

## 开放问题
请实现线程重点判断：

1. 现有 `prepare_intepolation_grids` 架构里，最合理的表生成入口应放在哪个模块？
2. 是否需要新增独立测试来证明“大表插值”与“逐点 Jeans 直算”一致？
3. 是否需要把 `devauc` 与 `sersic` 的表生成逻辑明确拆成两个入口，以避免 profile 分支在一个函数里过度膨胀？

## 非目标
以下内容不属于这份需求：

- 修改 `theta_E` replicated-sample 逻辑
- 在本线程实现 posterior predictive CLI
- 改回旧的历史 aperture 口径
- 继续以逐星系 `s2_grid` 作为 PPT 的直接输入
