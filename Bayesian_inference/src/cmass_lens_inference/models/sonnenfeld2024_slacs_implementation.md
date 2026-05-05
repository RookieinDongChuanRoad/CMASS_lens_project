# Sonnenfeld 2024 SLACS Model Implementation Notes

本文档记录如何在当前 `model_registry + JAX backend + NumPyro sampler`
架构下实现 `sonnenfeld2024_slacs.py`。它是实现说明，不是当前模型的
运行时代码。

核心边界：

- `sonnenfeld2024_slacs.py` 只负责科学模型：参数 schema、latent draw、
  per-lens likelihood、selection hook、summary / extra prior hook。
- `jax_backend` 统一负责 `jit`、`vmap`、Monte Carlo normalization、
  diagnostics reduction，以及 sampler-facing `log_prob_value`。
- grid generation belongs to data preparation。`fibre_crosssect_grid.hdf5`
  等预计算网格应在 `prepare_intepolation_grids` 侧生成；inference 只读取
  和插值预计算结果。

## Posterior Structure

Sonnenfeld 2024 SLACS debiased model 仍使用选择修正的层次后验：

```text
log p(eta | d)
  = sum_i log L_i(eta)
    - N_lens * log Z_norm(eta)
    + log p_extra(eta)
```

其中 `eta` 是 population-level hyper-parameters，`L_i` 是单个 lens 的
selection-weighted likelihood，`Z_norm` 是 selected sample 的归一化项。

归一化项应写成：

```text
Z_norm(eta)
  = integral d psi_g d psi_s
      P_g(psi_g | eta)
      P_s(psi_s | psi_g, eta)
      P_sel(psi_g, psi_s | eta)
```

首版实现应继续使用固定 random basis 做 Monte Carlo 积分。每个参数步重算
`Z_norm(eta)`，但 random basis 在一次 run 内固定，避免同一 `eta` 下的数值噪声。
`sonnenfeld2024_slacs.py` 不应自己写完整 JIT / VMAP 框架；它只提供 backend
可调用的模型 hook。

## Model Hooks

`sonnenfeld2024_slacs.py` 应实现与 CMASS 当前模型同级的 registry entry，但
模型内部应进一步拆成 backend hook，而不是复制 CMASS 的整套 `log_prob`。

### Parameter Schema

默认 sampled parameter order：

```text
mu5_0
beta5
xi5
sigma5
mu_gamma_0
beta_gamma
xi_gamma
sigma_gamma
mu_zs
sigma_zs
theta0
loga
```

其中：

- `mu5_0, beta5, xi5, sigma5` 描述 `m5 | m*, R_e`。
- `mu_gamma_0, beta_gamma, xi_gamma, sigma_gamma` 描述
  `gamma | m*, R_e`。
- `mu_zs, sigma_zs` 描述 effective source-redshift distribution。
- `theta0, loga` 描述 `P_find(theta_E_est)`，其中 `a = 10**loga`。

质量定义首版应与 Sonnenfeld 2024 对齐，使用 fixed `M_2D(<5 kpc)` 的
`m5`。若后续要支持 `h_units_v1` 或 `m5_hinvkpc`，应作为新的
`model.components.mass_definition` 变体加入，并显式写清与 Sonnenfeld
fixed-5-kpc 结果的转换。

### Foreground Population

foreground population 不能沿用当前 CMASS 的独立
`P(z_d) * S(m*)` 近似。Sonnenfeld 2024 的 parent population 使用 Table 1
对应的联合分布：

```text
P_g(z_d, m*) proportional to
  dV/dz_d
  * f_t(z_d, m*)
  * 10**((m* - mbar) * (alpha + 1))
  * exp(-10**(m* - mbar))
```

`f_t(z_d, m*)` 是随 redshift 变化的低质量截断 / completeness 项。
实现时应等价迁移 Sonnenfeld repo 中 `mz_distribution.py` 的
`msdist(z, ms)` / `draw_mz()` 逻辑，或者在数据准备阶段预生成可 JAX 消费的
inverse-CDF / tabulated sampler。

`draw_parent(params, normals, context)` 应从联合 `P_g(z_d, m*)` 抽样，而不是
分别从一维 `z_d` Gaussian 和 skew-normal `m*` 抽样。

### Structural Relations

latent foreground variables：

```text
psi_g = (z_d, m*, R_e, m5, gamma)
```

若 profile 变体需要 Sersic index，则扩展为：

```text
psi_g = (z_d, m*, n, R_e, m5, gamma)
```

size relation：

```text
mu_R(m*, n)
  = mu_R0 + beta_R * (m* - mstar_pivot)
    + optional nu_R * (log10(n) - log10(4))
```

residual：

```text
Delta_R = log10(R_e / kpc) - mu_R(m*, n)
```

mass and slope population：

```text
mu5 = mu5_0 + beta5 * (m* - mstar_pivot) + xi5 * Delta_R
mu_gamma = mu_gamma_0 + beta_gamma * (m* - mstar_pivot) + xi_gamma * Delta_R

m5 ~ Normal(mu5, sigma5)
gamma ~ Normal(mu_gamma, sigma_gamma)
```

`mstar_pivot`、size-relation constants 和 unit shifts 应从 compiled context
读取，不应在 hot JAX kernel 中硬编码。

### Source Distribution

source 部分需要保留 Sonnenfeld 的 effective source-redshift distribution：

```text
P_s(z_s | eta) = TruncatedNormal(mu_zs, sigma_zs; z_s > 0)
```

实现时要注意：`z_s` 分布可以是 effective source term，但 foreground 的
`z_d, m*` 必须是 Table 1 的联合 parent distribution。不要把这两件事混成
当前 CMASS 的独立一维近似。

### Per-lens Likelihood

单 lens likelihood 沿每个观测 lens 的 `m5(gamma)` 轨迹积分。输入 observation
grid 至少需要：

```text
gamma_grid
mass_grid        # m5(gamma)
dmass_dthetaein_grid
s2_grid          # optional, sigma^2 / 10**m5
```

积分形式：

```text
L_i(eta)
  = integral d gamma d m*
      W_i(gamma, m* | eta)
```

其中被积项包含：

```text
P_g(z_d_i, m*)
P(log R_e_i | m*, n_i)
P(m5_i(gamma) | m*, R_e_i, eta)
P(gamma | m*, R_e_i, eta)
P_s(z_s_i | eta)
P_find(theta_E_est_i(gamma, m5_i, ...))
g_i(theta_E_i, gamma)
abs(dm5 / dtheta_E)
P_sigma
```

`P_sigma` 规则：

- `num_sigma = 0`：不乘 velocity-dispersion likelihood。
- `num_sigma = 1`：乘一个 Gaussian likelihood。
- `num_sigma = 2`：乘两个独立 Gaussian likelihood，共用同一
  `sigma_model`，使用各自观测误差。

速度弥散模型：

```text
sigma_model = sqrt(s2(gamma) * 10**m5(gamma))
```

其中 `s2 = sigma^2 / 10**m5`，必须与 mass definition 和 aperture convention
匹配。

### Selection

selection 必须严格使用 Sonnenfeld 的 velocity-dispersion proxy，而不是直接
把真实 lensing `theta_E` 传给 `P_find`。

normalization draw 中：

```text
sigma_proxy = sigma_model * (1 + sigma_noise)
theta_E_est = 4 * pi * (sigma_proxy / c)**2 * D_ds / D_s
P_find = 1 / (1 + exp(-10**loga * (theta_E_est - theta0)))
```

`sigma_noise` 应来自 fixed random basis 中专门的标准正态列，并使用与
Sonnenfeld reference implementation 一致的 fractional scatter / measurement
proxy 约定。

单 lens likelihood 中也应使用同一 proxy 逻辑构造 `theta_E_est_i`，而不是
退回 `P_find(theta_E_i)`。如果观测数据或 reference grids 已经预计算了对应
proxy 所需量，应优先读取预计算量；否则由 `s2_grid` 和 `m5(gamma)` 即时计算。

### Cross-section

Sonnenfeld cross-section uses `mufibre3_cs_grid(theta_E, gamma)`。

不要使用当前 CMASS 的 separable approximation：

```text
pi * (cs(gamma) * theta_E)**2
```

Sonnenfeld 的 `g` 是 finite-fibre, seeing-convolved, flux-thresholded
source-plane cross-section。inference 侧应做二维插值：

```text
g(theta_E, gamma)
  = interp2d(tein_grid, gamma_grid, mufibre3_cs_grid, theta_E, gamma)
```

边界行为应与 reference implementation 对齐：`theta_E` 超出 `tein_grid`
大端时 cross-section 置零；其余轴向 clipping / zeroing 规则应在 grid reader
中集中定义并测试。

## Cross-section Grid Generation

`fibre_crosssect_grid.hdf5` 的生成不是 `sonnenfeld2024_slacs.py` 的职责。
它属于数据准备流程，建议迁入 `prepare_intepolation_grids`，与 lensing grids、
Jeans / sigma-unit grids 一起作为 inference 输入产物管理。

推荐新增数据准备模块时对齐 Sonnenfeld repo 的 `make_crosssect_grid.py`。
其计算目标是 power-law lens 在 `(theta_E, gamma)` 网格上的 fibre-selected
lensing cross-section。

### Inputs and Grid Axes

grid axes：

```text
tein_grid  = linspace(0, 5, 51)
gamma_grid = linspace(gamma_min, gamma_max, 81)
```

survey / observation constants：

```text
fibre_arcsec = 1.5
seeing_arcsec = 1.5
muB_min = 1.0
```

这些常数应由数据准备 config 管理，并写入 HDF5 attrs，避免 inference 端
隐式依赖硬编码值。

### Physical Computation

对每个 `(theta_E, gamma)`：

1. 定义 axisymmetric power-law deflection：

   ```text
   alpha(x) = theta_E * sign(x) * (abs(x) / theta_E)**(2 - gamma)
   ```

2. 定义 convergence 和 magnification：

   ```text
   kappa(x) = (3 - gamma) / 2 * (abs(x) / theta_E)**(1 - gamma)
   mu_r(x)^-1 = 1 + alpha(x) / x - 2 * kappa(x)
   mu_t(x)^-1 = 1 - alpha(x) / x
   ```

3. 找 radial caustic，并在 source-plane coordinate `beta` 上建立积分网格。

4. 对每个 `beta` 解 lens equation，得到 bright / faint image positions
   `x_A` 和 `x_B`。

5. 对每个 image 计算 magnification，并用 Gaussian PSF 做 fibre aperture 内
   flux convolution。PSF width：

   ```text
   psf_sigma = seeing_arcsec / 2.35
   ```

6. 定义通过条件：

   ```text
   mufibre2: mutot_seeing > 2 and muB > muB_min
   mufibre3: mutot_seeing > 3 and muB > muB_min
   ```

7. 在 source plane 积分：

   ```text
   cross_section = integral_good 2 * pi * beta d beta
   ```

### HDF5 Output Contract

`fibre_crosssect_grid.hdf5` 至少应包含：

```text
tein_grid
gamma_grid
mufibre2_cs_grid
mufibre3_cs_grid
ycaust_grid
```

attrs 至少应包含：

```text
muB_min
fibre_arcsec
seeing_arcsec
source_flux_reference
spectroscopic_flux_threshold_reference
generator_name
generator_version
```

inference 只消费这些 datasets / attrs，不在 sampling 过程中生成或修改 grid。

## Implementation Dependencies and TODO

- 扩展 data config，使 Sonnenfeld 模型可以声明独立的
  `fibre_cross_section_path`。当前 `cross_section_path` 面向 CMASS 一维
  `cs_over_theta_ein(gamma)`，不应混用。
- 在 `prepare_intepolation_grids` 中实现或迁移 `make_crosssect_grid.py`
  等价逻辑，并提供 HDF5 schema validation。
- 将 JAX backend 改成 hook-driven engine：backend 负责 JIT / VMAP /
  normalization loop / diagnostics，模型只提供 draw、integrand、selection、
  summary hook。
- 实现 Sonnenfeld 专属 grid reader，把 `mufibre3_cs_grid(theta_E, gamma)`
  转成 JAX-friendly arrays 和边界规则。
- 实现 Table-1 `P(z_d, m*)` 联合 parent distribution 的 JAX-friendly
  density 和 deterministic sampler。
- 实现 velocity-dispersion proxy 的 `theta_E_est`，并用 reference code 的
  representative parameter point 做数值对照。
- 保留 `sonnenfeld2024_slacs.py` 的 explicit failure，直到上述 hook、grid
  reader 和 tests 都到位。

## Tests Required for the Future Implementation

- CMASS current model 在 hook-driven backend 下与拆分前 `log_prob` 等价。
- `fibre_crosssect_grid.hdf5` reader 能读取 `mufibre3_cs_grid`，并在
  `(theta_E, gamma)` 上给出与 SciPy reference interpolation 一致的值。
- `theta_E_est` velocity-dispersion proxy 与 Sonnenfeld reference
  implementation 在同一输入下数值一致。
- Sonnenfeld `selection_weight` 使用 `mufibre3_cs_grid(theta_E, gamma)`，
  不调用 CMASS `pi * (cs(gamma) * theta_E)**2` 路径。
- `sonnenfeld2024_slacs.py` consumes precomputed grids only；测试中应禁止
  sampling path 调用任何 grid generator。
