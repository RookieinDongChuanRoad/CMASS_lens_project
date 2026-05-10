# CMASS 主线方法报告：Sonnenfeld-style 重写版

本文把当前 CMASS strong-lens 主线分析链重写成更接近 Sonnenfeld (2024) section 3/4 的叙事结构。核心调整是：先在 `Methods` 中说明问题本身、选择效应修正和为了可计算而做的压缩；再在 `The Model` 中逐块介绍质量模型、前景星系分布、源红移分布、截面、发现概率、动力学项、FP prior、gamma population modes 与推断流程。

本文的事实基础不是 `2026-03-17` 的旧 rerun，而是 `2026-04-20` 到 `2026-04-22` 这一轮主线产物。当前数值结果采用 BIC 优选组：

- `devauc`: `outputs/devauc/archived/20260421_170915_devauc_m10_sigma_star_fp_prior_slit_good_drop2sigma_within_re_20260421`
- `sersic`: `outputs/sersic/archived/20260421_163640_sersic_m10_independent_fp_prior_slit_good_drop2sigma_within_re_20260421`

正文仍然保持方法论导向。工程路径只作为可追溯锚点出现，不把 pipeline 日志改写成论文主体。

## 1. Overview

这套代码实现的是一个选择效应修正的强透镜层级贝叶斯模型。对每个 lens，观测到的 lens redshift、source redshift、Einstein radius、stellar mass、effective radius，以及可用时的 aperture velocity dispersion，先被压缩成单镜头 likelihood 可以使用的物理轨迹；随后，群体层模型描述 foreground lenses、background sources、strong-lensing cross-section 和 lens-finding probability；最后，后验样本被前向生成 parent、detectable 和 selected populations，用 posterior predictive checks 与 Fig. 8-like trends 检查模型是否能重现观测 catalog 的整体统计特征。

这一流程可以概括为

$$
\log p(\eta \mid D)
=
\sum_{i=1}^{N_{\mathrm{lens}}}\log \mathcal L_i(\eta)
- N_{\mathrm{lens}}\log Z_{\mathrm{norm}}(\eta)
+ \log p(\eta),
$$

其中 $\eta$ 是群体层超参数，$\mathcal L_i$ 是第 $i$ 个 lens 的选择效应加权单镜头 likelihood，$Z_{\mathrm{norm}}$ 是 selected-lens population 的归一化项。

当前主线需要区分三层事实：

- 代码能力：支持 `dependent`、`independent`、`sigma_star_dependent` 三种 `gamma_model.mode`，也支持 `slit` 与 `boss` 两种 observed-aperture contract。
- 生产 pipeline：`2026-04-20` 起切到 `m10 + slit rebuilt observations + fp_prior + /within_re/m10`；`2026-04-21` 在 `good_drop2sigma_within_re` 口径上跑通 `devauc/sersic × independent/sigma_star_dependent` 四组 inference 与后处理；`2026-04-22` 用 BIC 汇总 8 条 run。
- 本文结果：正文数值只引用 4/21 BIC 优选组，不把 `2026-03-17` rerun 当作当前主结果，也不把失效的 `outputs/*/latest` 当作证据锚点。

## 2. Data Products And Current Evidence

进入当前主模型的第 $i$ 个 lens 数据向量可写成

$$
D_i =
\left\{
z_{d,i},
z_{s,i},
\theta_{E,i},
\log M_{\ast,i}^{\mathrm{obs}},
\log R_{e,i}^{\mathrm{obs}},
n_i^{\mathrm{obs}},
\sigma_{\mathrm{ap},i}^{\mathrm{obs}}
\right\}.
$$

这些观测量在统计角色上不对称。$z_d$、$z_s$ 和 $\theta_E$ 决定 lensing geometry 与 enclosed-mass track；$\log M_\ast$、$\log R_e$ 和 profile branch 决定 foreground population relation；$\sigma_{\mathrm{ap}}$ 只在 `num_sigma > 0` 时进入动力学 likelihood。

当前 `good_drop2sigma_within_re` 口径有几个必须明确的样本事实：

- raw HDF5 仍保留全部 23 个 lens group，所有 lens 都贡献 lensing 和 selection 项。
- `good` 文件没有删除 lens，而是把 `140929-011410` 与 `220506+014703` 两颗最低 `logM*` 的 sigma lens 改写成 `num_sigma = 0`。
- 当前 BIC 优选结果对应 11 个 lens-level sigma objects 和 14 个原始 sigma measurement points。
- `good` raw 文件 root attrs 中显式记录了 `derivation_note` 与 `excluded_sigma_lens_ids`。

主线还支持 BOSS raw observation products：

- `data/raw/observations_deV_with_BOSS_mass_grids.hdf5`
- `data/raw/observations_with_BOSS_mass_grids_all.hdf5`

BOSS branch 是代码支持的 observation contract，不是本文当前主结果的证据来源。本文结果全部来自 slit rebuilt / good HDF5 分支。

## 3. Methods

### 3.1 The inference problem

理想情况下，强透镜样本应该被看作 foreground galaxy population、background source population 与 survey selection process 的联合结果。若 foreground galaxy 的 latent parameters 记为 $\psi_g$，background source 的 latent parameters 记为 $\psi_s$，则 selected strong-lens population 可以抽象写成

$$
P_{\mathrm{SL}}(\psi_g,\psi_s)
\propto
P_g(\psi_g)\,
P_s(\psi_s)\,
P_{\mathrm{sel}}(\psi_g,\psi_s).
$$

其中 $P_g$ 是 foreground parent population，$P_s$ 是 background source population，$P_{\mathrm{sel}}$ 是一个 galaxy-source pair 被发现并进入样本的概率。对只关心 foreground lenses 的分析，source variables 可以被边缘化，得到一个有效的 lensing-bias factor：

$$
P_{\mathrm{SL}}(\psi_g)
\propto
P_g(\psi_g)\,
b_{\mathrm{SL}}(\psi_g),
$$

$$
b_{\mathrm{SL}}(\psi_g)
\equiv
\int d\psi_s\,
P_s(\psi_s)\,
P_{\mathrm{sel}}(\psi_g,\psi_s).
$$

真正的困难在于 $P_{\mathrm{sel}}$。它包含强透镜几何、source brightness distribution、spectroscopic detectability、photometric follow-up、human prioritization 和最终样本定义。当前 CMASS 主线没有试图从第一性原理完整重建这些过程，而是构造一个可计算的有效 selected-population density，并把其中无法直接建模的部分压缩为少数经验参数。

对第 $i$ 个 lens，完整单镜头 likelihood 可抽象为

$$
\mathcal L_i(\eta)
=
\int d\psi_{g,i}\,d\psi_{s,i}\,
P(D_i\mid \psi_{g,i},\psi_{s,i})\,
P_{\mathrm{SL}}(\psi_{g,i},\psi_{s,i}\mid \eta).
$$

当前代码把这个表达式化简为围绕 $\gamma$ 与 latent stellar mass 的低维积分，并用 Monte Carlo normalization 修正 selected population 的体积因子。整体后验仍保持 selection-corrected hierarchical posterior 的形式：

$$
p(\eta\mid D)
\propto
p(\eta)
\prod_{i=1}^{N_{\mathrm{lens}}}
\mathcal L_i(\eta)
Z_{\mathrm{norm}}(\eta)^{-N_{\mathrm{lens}}}.
$$

### 3.2 A practical solution in this codebase

当前主线的实际解法分三步。

第一步，把 source surface-brightness 信息压缩成 Einstein radius。也就是说，代码不直接拟合每个 source 的成像像素、形态和 flux，而是把强透镜几何约束写成

$$
\theta_E^{\mathrm{obs}}
\quad\Longrightarrow\quad
m_R(\gamma).
$$

对固定 aperture $R$，二维投影质量定义为

$$
m_R \equiv \log_{10}M_{2\mathrm D}(<R).
$$

当前主结果使用 $R=10\,\mathrm{kpc}$，即 `m10`。`m5` 与 `m10` 是同一 power-law profile 在不同 aperture 下的两种质量表示，而不是两套模型：

$$
m_{10}
=
m_5 + (3-\gamma)\log_{10}2.
$$

第二步，把动力学计算压缩成 sigma-unit response。Jeans 计算预先写成

$$
S_{\mathrm{unit}}(\gamma)
\equiv
\frac{\sigma^2}{10^{m_R}},
$$

因此给定 Einstein-radius track 上的质量 $m_R(\gamma)$，模型速度弥散为

$$
\sigma_{\mathrm{model}}(\gamma)
=
\sqrt{
S_{\mathrm{unit}}(\gamma)\,
10^{m_R(\gamma)}
}.
$$

第三步，把 selected-lens population 的归一化用 Monte Carlo 估计。也就是说，代码不只拟合已经观测到的 23 个 lenses，还显式惩罚那些会让 selected population 总量变得不合理的超参数区域。

### 3.3 Data compression and observed quantities

当前 raw HDF5 同时承载三类统计对象。

第一类是主模型数据。23 个 lens group 都进入 lensing likelihood 和 selection normalization。`devauc` 分支固定使用 $n_i=4$；`sersic` 分支使用观测到的 Sersic index。

第二类是动力学子样本。当前 `good` raw 文件有 11 个 lens-level sigma objects。若同一 lens 有多个 dispersion measurements，PPC 观测侧会先做 inverse-variance aggregation，再形成一个 lens-level $\sigma$ 值进入 summary statistics。

第三类是 Fig. 8-like 观测点。annotation workflow 直接读取 raw HDF5 中的 single-lens flat-prior attrs，因此它保留 14 个逐测量 sigma points，而不是 PPC 使用的 11 个聚合值。

这一点是当前文档最容易写错的地方：`ppc_summary.json` 仍把 `sample_sizes.sigma` 写成历史常量 `7`，但当前 `good_drop2sigma` 结果中真正参与 `observed.sigma.*` 计算的是 11 个聚合后的 lens-level sigma 值。

### 3.4 Effective selected-lens distribution

经过上述压缩后，代码使用一个有效 selected-lens density 来组织 likelihood 与 normalization：

$$
P_{\mathrm{SL}}^{\mathrm{eff}}
(\psi_g,z_s\mid\eta)
\propto
P_g(\psi_g\mid\eta)\,
P_s^{\mathrm{eff}}(z_s\mid\eta)\,
g(\theta_E,\gamma)\,
P_{\mathrm{find}}(\theta_E\mid\eta).
$$

其中：

- $P_g$ 描述 foreground lens population。
- $P_s^{\mathrm{eff}}$ 是已吸收 source-light selection 的有效 source-redshift distribution。
- $g(\theta_E,\gamma)$ 是 cross-section lookup。
- $P_{\mathrm{find}}$ 是 lens-finding efficiency。

归一化项为

$$
Z_{\mathrm{norm}}(\eta)
=
\int d\psi_g\,dz_s\,
P_g(\psi_g\mid\eta)\,
P_s^{\mathrm{eff}}(z_s\mid\eta)\,
g(\theta_E,\gamma)\,
P_{\mathrm{find}}(\theta_E\mid\eta).
$$

这就是 posterior 中

$$
-N_{\mathrm{lens}}\log Z_{\mathrm{norm}}(\eta)
$$

的来源。它把“模型会生成多少可进入样本的 lenses”纳入推断，而不是只对已经观测到的 lenses 做条件拟合。

### 3.5 Posterior predictive summaries

PPC 和 Fig. 8-like trends 都不是新的模型。它们是把 posterior samples 前向生成到模拟 catalog 后做的约简。

对 PPC，代码从每个 posterior draw 生成 replicated selected catalog，并计算 summary statistics：

$$
T(D^{\mathrm{rep},(s)})
\in
\{
\mathrm{median}, \mathrm{std}, p_{10}, p_{90}
\}.
$$

然后比较

$$
\left\{
T(D^{\mathrm{rep},(s)})
\right\}_s
\quad\mathrm{and}\quad
T(D_{\mathrm{obs}}).
$$

对 Fig. 8-like trends，代码在 stellar-mass bins $B$ 上分别计算 parent、detectable 和 selected populations 的 band：

$$
p\!\left(
\bar y(B)
\mid
\mathrm{population\ class},D
\right),
$$

其中 $y\in\{m_{10},\gamma,\sigma_{\mathrm{ap}}\}$。这张图上的模型带来自 hierarchical posterior predictive distribution；观测点来自 single-lens flat-prior summaries。两者不能互相替代。

## 4. The Model

### 4.1 Power-law mass profile and individual-lens parameters

当前主线假设每个 lens 的总质量分布可用 spherical power-law profile 的 enclosed mass 与 density slope 描述。单个 foreground lens 的 latent parameters 可概括为

$$
\psi_g
=
\{
z_d,
m_\ast,
r_e,
m_{10},
\gamma,
n
\},
$$

其中 $m_\ast=\log_{10}M_\ast$，$r_e=\log_{10}R_e$，$m_{10}=\log_{10}M_{2\mathrm D}(<10\,\mathrm{kpc})$。

Einstein radius 被视为高精度几何约束。给定 $z_d$、$z_s$、$\theta_E$ 和 $\gamma$，代码可直接得到

$$
m_{10,i}(\gamma)
=
\log_{10}\!\left[
\pi\Sigma_c(z_{d,i},z_{s,i})\,
r_{\mathrm{Ein},i}^{\gamma-1}
(10\,\mathrm{kpc})^{3-\gamma}
\right].
$$

因此，单镜头积分不在自由的 $(m_{10},\gamma)$ 二维空间里做，而是在 Einstein-radius track 上沿 $\gamma$ 积分。

### 4.2 Foreground lens population

Foreground population 由 stellar mass、size、enclosed mass 和 density slope 的条件分布组成。主线可写成

$$
P_g(\psi_g\mid\eta)
=
P(m_\ast)\,
P(r_e\mid m_\ast,n)\,
P(m_{10},\gamma\mid m_\ast,r_e,n).
$$

质量关系写成 Gaussian conditional law：

$$
P(m_{10}\mid m_\ast,\Delta_R)
=
\mathcal N(
\mu_{10}(m_\ast,\Delta_R),
\sigma_{10}^2
),
$$

$$
\mu_{10}
=
\mu_{10,0}
+ \beta_{10}(m_\ast-11.3)
+ \xi_{10}\Delta_R.
$$

这里

$$
\Delta_R
=
r_e-\mu_R(m_\ast,n)
$$

是相对 profile-specific size relation 的 residual。`devauc` 与 `sersic` 两条 profile 分支共享同一层级结构，但 profile-specific structural constants 和 sigma-unit table 不同。

### 4.3 Source-redshift distribution

代码不完整建模 source light population，而是使用一个有效 source-redshift distribution：

$$
P_s^{\mathrm{eff}}(z_s\mid\eta)
=
\mathcal N(z_s\mid\mu_{z_s},\sigma_{z_s}^2).
$$

这个分布不是背景源整体红移分布，也不是已经观测到的 lensed-source redshift histogram。它是把 source-light selection、detectability 与可透镜化权重吸收后，用于 selected-lens likelihood 的有效分布。

### 4.4 Lensing cross-section and lens-finding probability

强透镜截面项在当前代码中用 lookup 表表示：

$$
g(\theta_E,\gamma)
\equiv
\mathrm{cs\_over\_theta\_ein}(\theta_E,\gamma).
$$

它负责描述给定几何和 density slope 时，一个 lens 对可探测 source 的强透镜权重。

Lens-finding efficiency 用 sigmoid 表示：

$$
P_{\mathrm{find}}(\theta_E)
=
\frac{1}{
1+\exp[-a(\theta_E-\theta_0)]
}.
$$

当前 posterior summary 中的 `theta0` 和 `loga` 就是这部分选择函数的自由参数。`loga` 存的是 slope parameter 的对数表示。

### 4.5 Velocity-dispersion likelihood and FP prior

当第 $i$ 个 lens 有速度弥散观测时，动力学 likelihood 采用

$$
P(\sigma_{\mathrm{ap},i}^{\mathrm{obs}}\mid\gamma)
=
\mathcal N(
\sigma_{\mathrm{ap},i}^{\mathrm{obs}}
\mid
\sigma_{\mathrm{model},i}(\gamma),
\delta\sigma_i^2
).
$$

若同一 lens 有多个 $\sigma_{\mathrm{ap}}$ measurements，单镜头 likelihood 会逐项累加；PPC 观测侧则先聚合成 lens-level 值后再计算 summary statistics。

FP prior 是当前 4/20-4/21 pipeline 的重要变化。主线使用 `/within_re/m10` synthetic sigma 来约束 parent population 的 fundamental-plane-like velocity-dispersion relation；但 likelihood、PPC 和 posterior trends 的 observed-aperture sigma 仍来自 `/slit/m10`。因此，`within_re` 是 FP prior 的 sigma definition，不是对 slit observation flavor 的整体替换。

### 4.6 Gamma population modes

当前源码支持三种 $\gamma$ population mode。

`independent` 模式为

$$
\mu_\gamma
=
\mu_{\gamma,0}.
$$

`dependent` 模式为

$$
\mu_\gamma
=
\mu_{\gamma,0}
+ \beta_\gamma(m_\ast-11.3)
+ \xi_\gamma\Delta_R.
$$

`sigma_star_dependent` 模式为

$$
\mu_\gamma
=
\mu_{\gamma,0}
+ \beta_{\Sigma_\ast,\gamma}
(\log\Sigma_\ast-\log\Sigma_{\ast,0}).
$$

当前 BIC 优选结果是 profile-dependent 的：`devauc` 选择 `sigma_star_dependent`，`sersic` 选择 `independent`。因此本文不能把 `devauc` 与 `sersic` 解释成同一 gamma 参数化下的纯 profile 对照。

### 4.7 Inference procedure and production pipeline

当前生产 run 使用：

- `m10` mass definition；
- `24` walkers；
- `10000` steps；
- `2000` warmup；
- `100000` normalization samples；
- `fp_sigma_definition = within_re`；
- `fp_sigma_table_leaf_path = /within_re/m10`。

`2026-04-20/21` 的实际生产顺序是：

1. `prepare_intepolation_grids` 维护 raw HDF5 与 sigma bundle。
2. `cmass_lens_inference cli run` 分别跑 `devauc` / `sersic` inference。
3. `cmass_lens_inference cli posterior-corner-latest` 生成 corner plot。
4. `lensing_posterior_predictive posterior-predictive-monitor` 生成 PPC。
5. `lensing_posterior_predictive posterior-trends` 生成 Fig. 8-like trends。
6. `lensing_posterior_predictive annotate-fig8-observations` 回填观测点。
7. `Bayesian_inference/scripts/compute_bic_after_20260420.py` 汇总 8 条 run 的 BIC。

`2026-04-21` orchestrator 先跑 `devauc/sersic` 的 `independent` pair，再跑 `devauc/sersic` 的 `sigma_star_dependent` pair，并对每一对分别执行 corner、PPC、posterior trends 与 Fig. 8 annotation。

## 5. Current Local Results

### 5.1 Run matrix and BIC selection

`2026-04-22` BIC 汇总使用 8 条 run，样本量为 `n=23`，burn-in 为 `2000`。

**devauc**

| Run | gamma mode | obs | `k` | max log-like | BIC | `delta_BIC` |
| --- | --- | --- | --- | --- | --- | --- |
| `20260421_170915_devauc_m10_sigma_star_fp_prior_slit_good_drop2sigma_within_re_20260421` | `sigma_star_dependent` | `observations_deV_with_mass_grids_good.hdf5` | 11 | -34.775211 | 104.040859 | 0.000000 |
| `20260421_162512_devauc_m10_independent_fp_prior_slit_good_drop2sigma_within_re_20260421` | `independent` | `observations_deV_with_mass_grids_good.hdf5` | 10 | -43.352651 | 118.060244 | 14.019385 |
| `20260420_125501_devauc_m10_sigma_star_fp_prior_slit_rebuilt_obs_within_re_20260420` | `sigma_star_dependent` | `observations_deV_with_mass_grids.hdf5` | 11 | -44.082115 | 122.654666 | 18.613806 |
| `20260421_144356_devauc_m10_independent_fp_prior_slit_rebuilt_obs_within_re_test_20260421` | `independent` | `observations_deV_with_mass_grids.hdf5` | 10 | -53.413512 | 138.181967 | 34.141108 |

**sersic**

| Run | gamma mode | obs | `k` | max log-like | BIC | `delta_BIC` |
| --- | --- | --- | --- | --- | --- | --- |
| `20260421_163640_sersic_m10_independent_fp_prior_slit_good_drop2sigma_within_re_20260421` | `independent` | `observations_with_mass_grids_all_good.hdf5` | 10 | -52.086028 | 135.526997 | 0.000000 |
| `20260421_172028_sersic_m10_sigma_star_fp_prior_slit_good_drop2sigma_within_re_20260421` | `sigma_star_dependent` | `observations_with_mass_grids_all_good.hdf5` | 11 | -51.279443 | 137.049321 | 1.522324 |
| `20260421_145549_sersic_m10_independent_fp_prior_slit_rebuilt_obs_within_re_test_20260421` | `independent` | `observations_with_mass_grids_all.hdf5` | 10 | -62.326496 | 156.007935 | 20.480937 |
| `20260420_130706_sersic_m10_sigma_star_fp_prior_slit_rebuilt_obs_within_re_20260420` | `sigma_star_dependent` | `observations_with_mass_grids_all.hdf5` | 11 | -61.558556 | 157.607549 | 22.080552 |

结论是不对称的：`devauc` 明显偏好 `sigma_star_dependent`，而 `sersic` 只弱偏好 `independent`。这也是本文主结果采用两种不同 gamma schema 的原因。

### 5.2 Posterior summaries

去掉前 `2000` steps 后，两条选中 run 各自提供 `192000` 个 posterior samples。

**`devauc / sigma_star_dependent / 11 parameters`**

| Parameter | Posterior summary |
| --- | --- |
| `mu10_0` | 11.7286 (-0.0170/+0.0158) |
| `beta10` | 0.6997 (-0.0817/+0.0678) |
| `xi10` | -0.1685 (-0.1906/+0.1608) |
| `sigma10` | 0.0589 (-0.0196/+0.0171) |
| `mu_gamma_0` | 1.8812 (-0.0265/+0.0313) |
| `beta_sigma_star_gamma` | 0.5725 (-0.1261/+0.0970) |
| `sigma_gamma` | 0.0791 (-0.0241/+0.0246) |
| `mu_zs` | 1.2654 (-0.1856/+0.2365) |
| `sigma_zs` | 0.8281 (-0.1329/+0.1513) |
| `theta0` | 0.5468 (-0.3675/+0.7344) |
| `loga` | 1.0726 (-1.7461/+1.2960) |

**`sersic / independent / 10 parameters`**

| Parameter | Posterior summary |
| --- | --- |
| `mu10_0` | 11.6660 (-0.0195/+0.0198) |
| `beta10` | 0.3995 (-0.0345/+0.0325) |
| `xi10` | -0.6891 (-0.1080/+0.1326) |
| `sigma10` | 0.0379 (-0.0186/+0.0251) |
| `mu_gamma_0` | 2.0251 (-0.0264/+0.0264) |
| `sigma_gamma` | 0.1397 (-0.0170/+0.0156) |
| `mu_zs` | 1.2989 (-0.2011/+0.2446) |
| `sigma_zs` | 0.8279 (-0.1336/+0.1499) |
| `theta0` | 0.6098 (-0.4001/+0.8129) |
| `loga` | 0.8199 (-1.4669/+1.3961) |

### 5.3 Posterior predictive checks

当前 `good` raw HDF5 上，PPC 的观测 $\sigma$ 样本是 11 个 lens-level aggregation，不是 `ppc_summary.json` 中遗留字段 `sample_sizes.sigma = 7` 所暗示的 7 个对象。

| Statistic | Observed | `devauc` replicated mean | `sersic` replicated mean |
| --- | --- | --- | --- |
| `theta.median` | 1.2480 | 1.4346 | 1.3537 |
| `theta.std` | 0.3263 | 1.7517 | 0.5248 |
| `sigma.median` | 239.0 | 246.9596 | 264.0310 |
| `sigma.std` | 29.1868 | 44.2979 | 55.0334 |

两条 selected run 都能大致重现 $\theta_E$ 与 $\sigma$ 的中心位置，但 replicated catalog 的宽度仍偏大。`devauc` 的 $\theta_E$ 宽度偏差更明显；两个分支的 $\sigma$ 宽度都偏大，其中 `sersic` 更宽。

### 5.4 Fig. 8-like trends near $\log M_\ast \approx 11.3$

| Profile | Class | `m10` | `gamma` | `sigma_ap` (km/s) |
| --- | --- | --- | --- | --- |
| `devauc` | `parent` | 11.6607 | 1.9669 | 221.2968 |
| `devauc` | `detectable` | 11.6846 | 1.9912 | 232.4576 |
| `devauc` | `selected` | 11.6868 | 1.9933 | 233.5387 |
| `sersic` | `parent` | 11.6276 | 2.0267 | 225.5544 |
| `sersic` | `detectable` | 11.6745 | 2.0348 | 239.3056 |
| `sersic` | `selected` | 11.6807 | 2.0365 | 241.1194 |

在当前 BIC 优选口径下，两条 run 都表现出从 `parent` 到 `selected` 的 `m10` 上升、`sigma_ap` 上升，以及温和的 $\gamma$ 上升。这个结论不能回套到旧的 `2026-03-17` dependent-gamma rerun。

## 6. Interpretation Boundaries

第一，single-lens flat-prior summary 与 hierarchical population posterior 必须分开。Fig. 8-like 观测点来自前者；群体参数、PPC 和 trend bands 来自后者。

第二，`good` raw HDF5 保留全部 23 个 lens group，只关闭两颗低 `logM*` lens 的 sigma 参与资格。它不是一个删除 lens 后的 catalog。

第三，当前存在三个 sigma 口径：raw 文件中 11 个 lens-level sigma objects，Fig. 8 overlay 中 14 个逐测量 sigma points，PPC 中 11 个 lens-level aggregated sigma values。

第四，`ppc_summary.json` 的 `sample_sizes.sigma = 7` 是遗留常量字段，不代表当前真实样本量。

第五，`within_re` 只用于 FP prior 的 sigma definition 与 sigma bundle leaf；likelihood、PPC 和 posterior trends 仍使用 `/slit/m10` observed-aperture sigma。

第六，`outputs/devauc/latest` 与 `outputs/sersic/latest` 当前都是坏链。正文必须使用显式 run dir；尤其是 `sersic/latest` 指向的 sigma-star basename 不等于当前 BIC 优选的 `sersic` run。

第七，BOSS observation branch 是代码支持能力，不是本文主结果来源。

第八，当前 `devauc` 与 `sersic` 主结果混合了 profile 选择与 gamma-mode 选择，因此不能被解释成只改变 light-profile assumption 的纯 profile 对照。

## 7. Code And Data Anchors

本文对应的主要代码与数据锚点如下。

| Object | Anchor |
| --- | --- |
| raw HDF5 and sigma bundle preparation | `prepare_intepolation_grids` |
| hierarchical inference | `Bayesian_inference` |
| posterior predictive checks and trends | `Posterior_predictive_test` |
| BIC summary | `outputs/_staging/20260422_bic_after_20260420/bic_report.md` |
| selected devauc run | `outputs/devauc/archived/20260421_170915_devauc_m10_sigma_star_fp_prior_slit_good_drop2sigma_within_re_20260421` |
| selected sersic run | `outputs/sersic/archived/20260421_163640_sersic_m10_independent_fp_prior_slit_good_drop2sigma_within_re_20260421` |
| devauc good raw HDF5 | `data/raw/observations_deV_with_mass_grids_good.hdf5` |
| sersic good raw HDF5 | `data/raw/observations_with_mass_grids_all_good.hdf5` |

如果这份文档被继续推进成论文方法段，最应该保留的组织原则是：先把选择效应修正的问题写清楚，再介绍代码选择的近似和压缩，最后才逐块展开模型组件。这样读者看到参数表和 PPC 结果时，已经知道每个数值对应的是哪个统计对象，而不是某个后处理脚本的产物。
