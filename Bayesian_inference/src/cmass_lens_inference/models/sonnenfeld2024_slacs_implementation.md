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

论文 Equation 38 把 parent stellar-mass function 的
`mbar, alpha, m_t^(0..5), sigma_t` 也列入完整模型参数集合。但第 4.7 节随后
说明：这些参数先由 parent sample 的 stellar-mass measurements 拟合，Table 1
给出 best-fit 值；由于 parent sample 很大，正式 SLACS lens inference 中固定
这些 Table 1 参数，只拟合上面列出的 lens-population、source-redshift 和
finding-probability 参数。因此当前 `ParameterSchema` 不应采样 Table 1
foreground-population constants，除非未来显式实现 parent-sample 联合拟合。

质量定义首版应与 Sonnenfeld 2024 对齐，使用 fixed `M_2D(<5 kpc)` 的
`m5`。若后续要支持 `h_units_v1` 或 `m5_hinvkpc`，应作为新的具体模型名
暴露，例如 `sonnenfeld2024_slacs_hunit`，并显式写清与 Sonnenfeld
fixed-5-kpc 结果的转换。不要把 paper-native fixed `m5` 和 hunit backend
坐标混在同一个 `sonnenfeld2024_slacs` 名称下。

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

其中 Table 1 固定值为：

```text
mbar = 11.06
alpha = -1.207
m_t(z) = sum_{k=0}^5 m_t^(k) z^k
(m_t^0, ..., m_t^5)
  = (9.388, 7.855, 48.34, -312.5, 535.7, -274.2)
sigma_t = 0.0007
```

`f_t(z_d, m*)` 是随 redshift 变化的低质量截断 / completeness 项，论文
Equation 27 写成：

```text
f_t(z, m*) = (1/pi) * arctan((m* - m_t(z)) / sigma_t) + 1/2
```

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

注意：Sonnenfeld 2024 本文并没有在 population model 中引入 Sersic-index
条件项。半光半径使用 `r_e = log10(R_e/kpc)`，并采用 Hyde & Bernardi (2009)
的 early-type quadratic mass-size relation。不要把当前 CMASS/HSC Sersic
profile 的 `n` 依赖搬进 paper-native Sonnenfeld 模型。

size relation：

```text
R(r_e | m*) = Normal(mu_R(m*), sigma_R^2)

mu_R(m*) = mu_R,0 + mu_R,1 * m* + mu_R,2 * m*^2

(mu_R,0, mu_R,1, mu_R,2) = (7.55, -1.84, 0.11)
sigma_R = 0.112
```

residual：

```text
Delta_R = r_e - mu_R(m*)
        = log10(R_e / kpc) - mu_R(m*)
```

mass and slope population：

```text
mu5
  = mu5_0
    + beta5 * (m* - 11.3)
    + xi5 * Delta_R

mu_gamma
  = mu_gamma_0
    + beta_gamma * (m* - 11.3)
    + xi_gamma * Delta_R

m5    ~ Normal(mu5, sigma5^2)
gamma ~ Normal(mu_gamma, sigma_gamma^2)
```

paper-native `sonnenfeld2024_slacs` 使用 physical stellar-mass coordinate
`m* = log10(M*/Msun)` 和 fixed `m5 = log10(M_2D(<5 kpc)/Msun)`。显式
`sonnenfeld2024_slacs_hunit` 变体若消费 hunit canonical dataset，所有
stellar-mass location constants (`11.3`, `mbar`, `m_t(z)`, 以及 mass-size
relation 中的 `m*` 坐标) 必须先被一致地变换到 active coordinate。unit
shifts 应在 preprocessing/context 层完成，不应在 hot JAX kernel 中硬编码。

### Fundamental Plane Prior

Sonnenfeld 2024 的 FP prior 不能复用当前 CMASS 默认参考值。论文第 4.6 节
使用的 relation 是：

```text
log10(sigma_ap)
  ~ Normal(
      mu_FP,0
      + beta_FP * (m* - 11.3)
      + xi_FP * Delta_R,
      sigma_FP^2
    )
```

其中 `Delta_R = r_e - mu_R(m*)`，所以这是包含 size residual 的二维
fundamental-plane relation，不是当前 CMASS common helper 中的一维
`sigma-logM*` relation。

论文 Equation 37 给出的参考 prior 是：

```text
P(mu_FP,0) = Normal(2.342, 0.030^2)
P(beta_FP) = Normal(0.258, 0.030^2)
P(sigma_FP) = Normal(0.047, 0.008^2)
```

这些数值与 CMASS 当前 `FPPriorConfig` 默认值不同：

```text
CMASS mu_v_prior        = 2.34548
CMASS beta_v_prior      = 0.176
CMASS fiducial_scatter  = 0.075
```

因此 Sonnenfeld FP prior 如果启用，应新增 Sonnenfeld 专属 component /
config defaults，而不是调用 `components.common.fp_prior` 的 CMASS-oriented
默认参考值。当前 runnable v1 暂时返回 neutral `extra_prior`，这是为了避免
把 CMASS FP prior 错误套到 Sonnenfeld 模型上；这不是 paper-level FP
constraint 的实现。

实现细节也要忠于论文：给定一组 population 参数 `eta` 时，论文不是把
`mu_FP,0`、`beta_FP`、`xi_FP`、`sigma_FP` 作为 lens-level likelihood 的独立
采样参数直接加入，而是先在大规模 mock population 上拟合 Equation 36，再用
Equation 37 对拟合出的 FP summary 加权。`xi_FP` 出现在 Equation 36 的 fit
中，但 Equation 37 只对 `mu_FP,0`、`beta_FP` 和 `sigma_FP` 给出显式 Gaussian
prior。

### Source Distribution

source 部分需要保留 Sonnenfeld 的 effective source-redshift distribution：

```text
P_s^eff(z_s | eta)
  = Normal(mu_zs, sigma_zs^2)
```

论文 Equation 33 写的是普通 Gaussian effective distribution，并没有写成
`z_s > 0` truncated normal。实现时可以为了数值安全在 proposal 或 support
上做截断，但这必须标成 backend 近似，不能写成 paper model 本身。

同时要注意：`P_s^eff` 不是 background sources 的真实 redshift distribution，
也不是已经 lens-selected 的 source redshift distribution。它是经过 source
light-dependent detectability factor `l(psi_s^l)` 加权并边缘化后的 effective
source-redshift term。foreground 的 `z_d, m*` 仍必须来自 Table 1 的联合
parent distribution。不要把这两件事混成当前 CMASS 的独立一维近似。

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
P(r_e_i | m*)
P(m5_i(gamma) | m*, Delta_R_i, eta)
P(gamma | m*, Delta_R_i, eta)
P_s^eff(z_s_i | eta)
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
sigma_ap_obs_proxy = sigma_ap_model * (1 + 0.0625 * epsilon)
theta_E_est
  = 4 * pi * (sigma_ap_obs_proxy / c)^2 * D_ds / D_s
P_find = 1 / (1 + exp(-10**loga * (theta_E_est - theta0)))
```

其中 `epsilon` 应来自 fixed random basis 中专门的标准正态列。论文第 4.5 节
采用所有 `sigma_ap` measurements 统一 6.25% fractional uncertainty，这是
SLACS lenses 的 median relative uncertainty。

单 lens likelihood 中也应使用同一 proxy 逻辑构造 `theta_E_est_i`，而不是
退回 `P_find(theta_E_i)`。需要特别小心：论文 likelihood 同时包含
`P(sigma_ap_obs | sigma_ap_model)` 和 `P_find(theta_E_est)`。因此单 lens
integral 中的 `P_find` 应与 observed-velocity-dispersion proxy 的噪声模型
保持一致；不能简单把 noiseless `sigma_model` 当作 Bolton et al. 选择时用到
的 observed `sigma_ap_obs`。如果 reference implementation 对这一项做了
条件化或额外 MC 平均，应按 reference implementation 对齐。

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

6. 定义通过条件。论文第 4.4 节的 reference source 条件是：
   emission-line flux in the fibre after PSF convolution 至少被放大 3 倍，
   并且 photometric data 中产生至少两个 magnification 大于 1 的 image。
   如果数据准备代码同时保存 `mufibre2` / `mufibre3` 这类派生网格，必须在
   schema 中说明它们对应的 detection threshold，而不是在 inference 侧猜。

   ```text
   spectroscopic condition: fibre_line_flux_magnification >= 3
   photometric condition: at least two images with magnification > 1
   ```

7. 在 source plane 积分：

   ```text
   cross_section = integral_good 2 * pi * beta d beta
   ```

### HDF5 Output Contract

若沿用现有命名，`fibre_crosssect_grid.hdf5` 至少应包含：

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

- inference 生产入口已经收口到 canonical dataset；不要重新引入
  `fibre_cross_section_path` 或 legacy `cross_section_path` 作为 production
  config。Sonnenfeld 需要的 finite-fibre cross-section 应作为 canonical
  capability / canonical HDF5 block 输入。
- 在 `prepare_intepolation_grids` 中实现或迁移 `make_crosssect_grid.py`
  等价逻辑，并提供 HDF5 schema validation。
- JAX backend 已经是 hook-driven engine：backend 负责 JIT / VMAP /
  normalization loop / diagnostics，模型只提供 draw、integrand、selection、
  summary hook。后续工作应集中在 paper-faithful Sonnenfeld hooks 和
  canonical data preparation，而不是重新拆 backend。
- 实现 Sonnenfeld 专属 grid reader，把 `mufibre3_cs_grid(theta_E, gamma)`
  转成 JAX-friendly arrays 和边界规则。
- Table-1 `P(z_d, m*)` 的 arctan completeness density 已进入 runtime hooks；
  后续仍需用 Sonnenfeld reference implementation 的 representative parameter
  point 对 normalization proposal / density ratio 做数值对照。
- velocity-dispersion proxy 的 `theta_E_est` 已进入 selection hooks，并使用
  论文第 4.5 节的 6.25% fractional uncertainty；后续仍需与 reference code
  对照单 lens conditional selection 项。
- 实现 Sonnenfeld 专属 FP prior summary：在 mock population 上拟合
  Equation 36，并用 Equation 37 的 `(2.342, 0.258, 0.047)` 参考值加权。
  不要复用 CMASS `FPPriorConfig` 默认值或一维 `components.common.fp_prior`
  作为 paper-level Sonnenfeld FP constraint。
- `sonnenfeld2024_slacs.py` 现在已经是 registry assembly layer。当前 runnable
  v1 只能作为工程路径 smoke test；在完成上述 reference-alignment 之前，不应
  把它标成 paper-level reproduction。

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
