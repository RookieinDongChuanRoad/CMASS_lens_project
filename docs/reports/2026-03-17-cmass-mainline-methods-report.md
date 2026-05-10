# CMASS 主线代码方法论报告

本文基于当前仓库主线研究代码与现有运行产物，对整条 CMASS strong-lens 分析链做一次方法论导向的重述。重点不是工程实现拆解，而是把代码里真正被显式建模的贝叶斯对象写成统一的论文记号，并说明这些公式如何串成一条从观测量、到选择效应修正的层级后验、再到 posterior predictive checks 与 Fig. 8-like 趋势图的完整分析链。本文只覆盖 `prepare_intepolation_grids`、`Bayesian_inference`、`Posterior_predictive_test` 以及必要的数据语义说明，不讨论仓库中用于比较或回溯的辅助工作区。

## Overview

这套代码实现的是一个选择效应修正的强透镜层级贝叶斯模型。对每个 lens，观测到的 $(z_d, z_s, \theta_E, \log M_\ast, \log R_e)$ 与可用时的 aperture velocity dispersion 先通过幂律质量模型和 Jeans 响应被转成沿 $\gamma$ 的质量轨迹 $m_R(\gamma)$ 与动力学预测 $\sigma_{\mathrm{model}}(\gamma)$；随后，群体层模型对 stellar-mass function、size relation、enclosed mass、density slope、source-redshift 分布和 lens-finding efficiency 做联合建模，并通过

$$
\log p(\eta \mid D)
=
\sum_{i=1}^{N_{\mathrm{lens}}} \log \mathcal{L}_i(\eta)
- N_{\mathrm{lens}} \log Z_{\mathrm{norm}}(\eta)
+ \log p(\eta)
$$

推断超参数 $\eta$；最后，再从 $p(\eta \mid D)$ 前向生成 parent population、detectable population 和 selected population，用 replicated catalogs 与质量分箱趋势带检查模型是否能重现观测样本的整体统计特征。

当前主线的“代码能力”和“本地数值证据”需要明确区分。源码能力层面，这套框架现在同时支持 `dependent`、`independent`、`sigma_star_dependent` 三种 `gamma_model.mode`，也同时支持 `slit` 与 `boss` 两种 observed-aperture contract，以及供 FP prior 使用的独立 `within_re` sigma definition。生产流水线层面，`2026-04-20` 起的主线 run 已切到 `m10 + slit rebuilt observations + fp_prior + /within_re/m10`；`2026-04-21` 又在 `good_drop2sigma_within_re` 口径上完整跑通了 `devauc/sersic × independent/sigma_star_dependent` 四组 inference、posterior corner、PPC、posterior trends 与 Fig. 8 观测点回填；`2026-04-22` 再用 BIC 汇总这 8 条 run 做模型选择。本文的数值结果段因此不再沿用 `2026-03-17` 的 rerun，而改为引用这套 `2026-04-20/21` pipeline 的显式 run 目录；BOSS branch 仍作为当前主线支持的 observation contract 说明，但不是本文当前主结果的证据来源。

## Data And Derived Quantities

### Observed quantities entering the model

对第 $i$ 个 lens，进入主模型的观测量可写成

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

其中：

- 全部 23 个 lens group 都保留在 raw HDF5 中，并都贡献透镜和选择项；
- 动力学 likelihood 是否使用某个 lens，取决于该 group 的 `num_sigma`，而不是一个写死常数；
- `2026-04-20` 的 rebuilt-observation slit 文件仍保留 `13` 个 lens-level sigma objects，对应 `16` 个原始 sigma measurements；
- `2026-04-21` 的 `good` 文件没有删掉任何 lens，而是把 `140929-011410` 与 `220506+014703` 两颗最低-`log M_\ast` 的 sigma lenses 改写为 `num_sigma = 0`；因此当前 BIC 优选结果对应的是 `11` 个 lens-level sigma objects 与 `14` 个原始 sigma measurement points；
- `devauc` 分支取 $n_i^{\mathrm{obs}} = 4$；
- `sersic` 分支使用观测到的 $n_i^{\mathrm{obs}}$。

这组观测量在统计上的角色并不对称。$z_d$ 与 $z_s$ 决定透镜几何，$\theta_E$ 决定质量轨迹 $m_R(\gamma)$，$\log M_\ast^{\mathrm{obs}}$ 与 $\log R_e^{\mathrm{obs}}$ 约束群体层关系中的 latent structural state，而 $\sigma_{\mathrm{ap}}^{\mathrm{obs}}$ 只在有动力学数据时参与单镜头似然。

对当前 `good` 口径，还需要额外记住一件事：root attrs 已显式写入 `derivation_note` 与 `excluded_sigma_lens_ids`，因此“drop2sigma”在这里表示重写动力学子样本口径，而不是删掉 lens group 或改动透镜几何样本本身。

### BOSS observation products as a new upstream data branch

当前主线除了历史 slit-like raw observation 文件之外，还支持两份新的 BOSS raw HDF5：

- `data/raw/observations_deV_with_BOSS_mass_grids.hdf5`
- `data/raw/observations_with_BOSS_mass_grids_all.hdf5`

它们不是从旧 raw HDF5 克隆出来的副本，而是从 BOSS summary table 重新构建的独立观测产品族。这一点在方法上是重要的，因为它意味着主线现在显式支持两种不同的 observation contract：历史 slit 分支使用矩形 aperture，而 BOSS 分支使用圆形 aperture。对 BOSS raw 文件，主线约定

$$
\mathrm{aperture\_shape} = \mathrm{circular},
\qquad
\mathrm{aperture\_radius\_arcsec} = 1.0,
\qquad
\mathrm{seeing\_fwhm\_arcsec} = 1.5.
$$

因此，这两份 BOSS 文件虽然继续写入与透镜几何、质量定义和动力学相关的同类 attrs，但其观测几何与原 slit 分支并不相同。主线接口上，这条分支由 `python -m interpolation_grids --build-boss-observation-hdf5` 生成。

BOSS raw 文件还显式写入

$$
\log \Sigma_\ast
=
\log M_\ast - \log_{10}(2\pi R_{e,\mathrm{kpc}}^2),
$$

即代码 attrs 中的 `log10_Sigma_star`。这个量本身不是新的 likelihood 项，但它为后续 `sigma_star_dependent` gamma-mode 提供了可直接解释的物理量基准。本文把它视为新增上游数据语义，而不是当前结果段已经切换到 BOSS sample 的证据。

### Lensing geometry and enclosed-mass definition

代码统一使用

$$
m_R \equiv \log_{10} M_{2\mathrm{D}}(<R)
$$

来表示固定 aperture $R$ 内的二维投影质量。当前支持 $R = 5\,\mathrm{kpc}$ 和 $R = 10\,\mathrm{kpc}$，分别对应 $m_5$ 与 $m_{10}$。它们不是两套无关模型，而是同一幂律质量分布在不同 aperture 下的两种表示。对幂律密度斜率 $\gamma$，二者满足解析关系

$$
m_{10} = m_5 + (3-\gamma)\log_{10} 2.
$$

临界面密度和 Einstein 半径的关系由

$$
\Sigma_c(z_d, z_s)
=
\frac{c^2}{4 \pi G}\frac{D_s}{D_d D_{ds}}
$$

和

$$
10^{m_R}
=
\pi \Sigma_c \, r_{\mathrm{Ein}}^{\gamma-1} R^{3-\gamma}
$$

给出，因此

$$
r_{\mathrm{Ein}}
=
\left[
\frac{10^{m_R}}{\pi \Sigma_c R^{3-\gamma}}
\right]^{1/(\gamma-1)},
\qquad
\theta_{\mathrm{Ein}}
=
\frac{r_{\mathrm{Ein}}}{D_d}\,206265.
$$

在单镜头分析里，观测到的 $\theta_{E,i}$ 被当作高精度几何约束，因此 $(m_R,\gamma)$ 并不是自由的二维参数平面，而被压缩到一条

$$
m_{R,i}(\gamma \mid z_{d,i}, z_{s,i}, \theta_{E,i})
$$

轨迹上。把 $r_{\mathrm{Ein},i} = D_{d,i}\theta_{E,i}/206265$ 代回上式，可显式写成

$$
m_{R,i}(\gamma)
=
\log_{10}\!\left[
\pi \Sigma_c(z_{d,i},z_{s,i})\,
r_{\mathrm{Ein},i}^{\gamma-1}
R^{3-\gamma}
\right].
$$

也就是说，后续单镜头 likelihood 的积分变量是 $\gamma$ 和 latent stellar mass $m_\ast$，而不是 $(m_R,\gamma)$ 的自由二维搜索。

### Dynamics and sigma-unit response

动力学部分使用 Jeans 响应表中 tabulate 的单位质量量

$$
S_{\mathrm{unit}}
\equiv
\frac{\sigma^2}{10^{m_R}}.
$$

因此，给定某个 lens 在 $\gamma$ 处对应的轨迹质量 $m_{R,i}(\gamma)$，模型速度弥散写成

$$
\sigma_{\mathrm{model},i}(\gamma)
=
\sqrt{
S_{\mathrm{unit},i}(\gamma)\,
10^{m_{R,i}(\gamma)}
}.
$$

这一步把 Jeans 响应与质量定义严格耦合在一起，因此 $m_5/m_{10}$ 的变化会一致地传播到 dynamics 项和 posterior predictive 阶段的 $\sigma_{\mathrm{ap}}$ 预测。

对同一物理 lens，在两种包络半径定义之间，$S_{\mathrm{unit}}$ 也有精确解析变换：

$$
S_{\mathrm{unit}}(R_{\mathrm{to}})
=
S_{\mathrm{unit}}(R_{\mathrm{from}})
\left(
\frac{R_{\mathrm{from}}}{R_{\mathrm{to}}}
\right)^{3-\gamma}.
$$

特别地，

$$
S_{\mathrm{unit}}^{(10)}
=
S_{\mathrm{unit}}^{(5)}\,2^{-(3-\gamma)}.
$$

### Single-lens flat-prior summaries used by Fig. 8 points

Fig. 8 叠加的观测点不是 hierarchical posterior 的 per-lens 投影，而是 single-lens flat-prior posterior 的摘要。该摘要先用 Einstein radius 把二维 $(m_5,\gamma)$ 空间压成

$$
m_5^{\mathrm{grid}}(\gamma),
$$

然后用 Jeans 响应得到

$$
\sigma_{\mathrm{model}}(\gamma)
=
\sqrt{
10^{m_5^{\mathrm{grid}}(\gamma)}
s_2(\gamma)
}.
$$

对应的一维 log-posterior 为

$$
\log p(\gamma \mid \mathrm{data})
\propto
\sum_{j=1}^{N_\sigma}
\left[
-\frac{1}{2}
\left(
\frac{
\sigma_{\mathrm{model}}(\gamma)-\sigma_{\mathrm{obs},j}
}{
\sigma_{\mathrm{err},j}
}
\right)^2
- \log \sigma_{\mathrm{err},j}
\right]
- \log \left|
\frac{d m_5}{d \theta_{\mathrm{Ein}}}
\right|.
$$

其中 $N_\sigma \in \{1,2\}$。因此，Fig. 8 中每个 lens 的 $\gamma$ 与 $m_5$ 误差条来自同一个一维 posterior：先沿 $\gamma$ 轴取

$$
q_{16}^{(\gamma)},\quad q_{50}^{(\gamma)},\quad q_{84}^{(\gamma)},
$$

再把同一 posterior 投影到 $m_5^{\mathrm{grid}}(\gamma)$，并进一步通过

$$
m_{10}^{\mathrm{grid}}(\gamma)
=
m_5^{\mathrm{grid}}(\gamma) + (3-\gamma)\log_{10} 2
$$

得到 $m_{10}$ 摘要。这不是群体层 hierarchical posterior 的 per-lens 拆分。

## Hierarchical Model

### Population hyper-parameters and fixed profile-specific structure

主线推断器有两个 profile 分支：`devauc` 与 `sersic`。它们共享同一统计骨架和同一后验结构，只在结构关系中关于 $n$ 的处理不同。统一地，把推断超参数记为

$$
\eta =
\left(
\mu_{m_R,0},
\beta_{m_R},
\xi_{m_R},
\sigma_{m_R},
\mu_{\gamma,0},
\beta_{\gamma},
\xi_{\gamma},
\sigma_{\gamma},
\mu_{z_s},
\sigma_{z_s},
\theta_0,
\log a
\right).
$$

在 `dependent` 参数化下，这组参数对 $R=10\,\mathrm{kpc}$ 可写成

$$
(
\mu_{10,0},
\beta_{10},
\xi_{10},
\sigma_{10},
\mu_{\gamma,0},
\beta_{\gamma},
\xi_{\gamma},
\sigma_{\gamma},
\mu_{z_s},
\sigma_{z_s},
\theta_0,
\log a
).
$$

与此同时，profile-dependent 的固定结构常数来自外部文献抽取，而不在 MCMC 中采样。例如 size relation 的中心项和散度由 profile 决定；对 `devauc`，$n=4$ 固定；对 `sersic`，$n$ 进入结构均值关系。

### Stellar-mass function, size relation, and conditional mass-slope law

群体模型以 latent stellar mass

$$
m_\ast \equiv \log_{10}(M_\ast/M_\odot)
$$

为起点。deflector 的 stellar-mass function 采用 skew-normal：

$$
S(m_\ast)
=
\frac{2}{s_\ast}\,
\phi\!\left(\frac{m_\ast-\mu_\ast}{s_\ast}\right)
\Phi\!\left[
\alpha_\ast
\frac{m_\ast-\mu_\ast}{s_\ast}
\right],
$$

其中 $(\mu_\ast, s_\ast, \alpha_\ast)$ 由 profile 固定，$\phi$ 与 $\Phi$ 分别是标准正态 PDF 与 CDF。

结构关系先定义 mean size relation

$$
\mu_r(m_\ast,n)
=
\mu_{r,0} + \beta_r(m_\ast-11.4)
$$

对于 `sersic` 分支，再加上

$$
\nu_r \left[\log_{10} n - \log_{10} 4\right].
$$

于是 size residual 为

$$
\Delta_R
=
\log_{10} R_e - \mu_r(m_\ast,n).
$$

在此基础上，代码显式建模 enclosed mass 与 density slope 的条件均值：

$$
\mu_{m_R}(m_\ast, R_e, n; \eta)
=
\mu_{m_R,0}
+ \beta_{m_R}(m_\ast-11.4)
+ \xi_{m_R}\Delta_R,
$$

$$
\mu_{\gamma}(m_\ast, R_e, n; \eta)
=
\mu_{\gamma,0}
+ \beta_{\gamma}(m_\ast-11.4)
+ \xi_{\gamma}\Delta_R.
$$

这对应 `dependent` gamma 模式的写法。与此同时，主线源码现在把 $\mu_\gamma$ 的参数化显式推广成三种 mode-aware 写法：

$$
\mu_\gamma
=
\begin{cases}
\mu_{\gamma,0}
+ \beta_{\gamma}(m_\ast-11.4)
+ \xi_{\gamma}\Delta_R,
& \texttt{dependent}, \\
\mu_{\gamma,0},
& \texttt{independent}, \\
\mu_{\gamma,0}
+ \beta_{\Sigma_\ast,\gamma}\,(\log \Sigma_\ast - 9),
& \texttt{sigma\_star\_dependent}.
\end{cases}
$$

其中

$$
\log \Sigma_\ast - 9
=
m_\ast - \log_{10}(2\pi) - 2\log_{10}R_{e,\mathrm{kpc}} - 9.
$$

这三种模式对应三套 sampled parameter schema：

- `dependent`: 12 维
- `independent`: 10 维
- `sigma_star_dependent`: 11 维

`2026-04-21` 的 full4 slit-good pipeline 明确比较了 `independent` 与 `sigma_star_dependent` 两种模式在 `devauc` 和 `sersic` 上的表现，`2026-04-22` 的 BIC 汇总最终选择 `devauc + sigma_star_dependent` 与 `sersic + independent` 作为本文当前主结果锚点。

需要强调的是，`sigma_star_dependent` 改变的只是 $\mu_\gamma$ 的群体均值参数化；单镜头 likelihood 的积分结构、selection normalization 的定义，以及总体后验主公式都保持不变。它现在不再只是源码里“可支持”的能力，而是已经进入 `2026-04-20/21` 生产 run 的实际比较矩阵。

因此，群体层条件分布写成

$$
m_R \mid m_\ast, R_e, n, \eta
\sim
\mathcal N\!\left(
\mu_{m_R}(m_\ast, R_e, n; \eta),
\sigma_{m_R}^2
\right),
$$

$$
\gamma \mid m_\ast, R_e, n, \eta
\sim
\mathcal N\!\left(
\mu_{\gamma}(m_\ast, R_e, n; \eta),
\sigma_{\gamma}^2
\right).
$$

对 generative population 而言，代码还要求 $\gamma$ 落在物理支持区间内，因此该 law 在前向生成与归一化 Monte Carlo 中实现为

$$
\gamma \sim \mathrm{TruncNormal}_{[1.2,\,2.8]}
\left(
\mu_{\gamma},
\sigma_\gamma^2
\right).
$$

### Source-redshift law and lens-finding efficiency

source redshift 使用非负截断高斯：

$$
p_s^{\mathrm{eff}}(z_s \mid \eta)
=
\frac{
\phi\!\left(
\frac{z_s-\mu_{z_s}}{\sigma_{z_s}}
\right)
}{
\sigma_{z_s}
\left[
1-\Phi\!\left(
\frac{-\mu_{z_s}}{\sigma_{z_s}}
\right)
\right]
}
\mathbf{1}(z_s \ge 0).
$$

deflector redshift 在 generative law 中固定为

$$
p(z_d) = \mathcal N(0.558, 0.085^2).
$$

发现概率采用 sigmoid：

$$
P_{\mathrm{find}}(\theta_E \mid \eta)
=
\frac{1}{
1+\exp\!\left[
-a(\theta_E-\theta_0)
\right]
},
\qquad
a = 10^{\log a}.
$$

强透镜截面项写成

$$
g(\theta_E,\gamma)
=
\pi
\left[
c_s(\gamma)\,\theta_E
\right]^2,
$$

其中 $c_s(\gamma)$ 来自 cross-section lookup table 中的 `cs_over_theta_ein(\gamma)`。

### Single-lens likelihood

对第 $i$ 个 lens，代码的核心对象是单镜头积分

$$
\mathcal L_i(\eta)
=
\int d\gamma \int dm_\ast \;
W_i(\gamma,m_\ast \mid \eta),
$$

其中

$$
\begin{aligned}
W_i(\gamma,m_\ast \mid \eta)
=\;&
p(z_{d,i})\,
p(\log M_{\ast,i}^{\mathrm{obs}} \mid m_\ast)\,
S(m_\ast)\,
p(\log R_{e,i}^{\mathrm{obs}} \mid m_\ast, n_i)\,
\\
&\times
p\!\left(
m_{R,i}(\gamma)
\mid
m_\ast, R_{e,i}, n_i, \eta
\right)\,
p\!\left(
\gamma
\mid
m_\ast, R_{e,i}, n_i, \eta
\right)\,
\\
&\times
p_s^{\mathrm{eff}}(z_{s,i}\mid\eta)\,
P_{\mathrm{find}}\!\left(\theta_{E,i}\mid\eta\right)\,
g(\theta_{E,i},\gamma)\,
J_i(\gamma)\,
p_{\sigma,i}(\gamma),
\end{aligned}
$$

且 Jacobian 项为

$$
J_i(\gamma)
=
\left|
\frac{dm_{R,i}}{d\theta_{\mathrm{Ein}}}
\right|.
$$

关键点在于：$m_R$ 不是独立积分变量。观测到的 Einstein radius 先把 $(m_R,\gamma)$ 约束到 $m_{R,i}(\gamma)$ 上，因此 dynamics 项是沿同一条轨迹定义的：

$$
\sigma_{\mathrm{model},i}(\gamma)
=
\sqrt{
S_{\mathrm{unit},i}(\gamma)\,
10^{m_{R,i}(\gamma)}
}.
$$

若第 $i$ 个 lens 有 $N_{\sigma,i}$ 个速度弥散观测，则

$$
p_{\sigma,i}(\gamma)
=
\prod_{k=1}^{N_{\sigma,i}}
\mathcal N\!\left(
\sigma_{\mathrm{ap},ik}^{\mathrm{obs}}
\mid
\sigma_{\mathrm{model},i}(\gamma),
\delta \sigma_{ik}^2
\right),
$$

而对没有动力学数据的 lens，直接取

$$
p_{\sigma,i}(\gamma)=1.
$$

因此，动力学与透镜信息的耦合并不是两个互相独立的 likelihood 简单相加，而是：$\theta_E$ 先决定质量轨迹 $m_R(\gamma)$，Jeans 响应再把这条轨迹转成 $\sigma_{\mathrm{model}}(\gamma)$，最后由速度弥散观测沿同一条轨迹重新加权。

### Selection normalization and the full posterior

选择修正不是单镜头项里的一个额外权重，而是完整后验的一部分。定义 latent population state

$$
x = (z_d, z_s, m_\ast, n, R_e, m_R, \gamma),
$$

则样本归一化项是

$$
Z_{\mathrm{norm}}(\eta)
=
\int
dx\;
p_{\mathrm{pop}}(x \mid \eta)\,
P_{\mathrm{find}}(\theta_E(x)\mid\eta)\,
g(\theta_E(x),\gamma)\,
\mathbf{1}(z_d > 0, z_s > z_d, \theta_E > 0).
$$

代码中它不是解析积分，而是用固定随机基底上的 Monte Carlo 估计。记 $x_j(\eta)$ 为由同一组标准正态基底变换得到的第 $j$ 个群体样本，则实现上对应

$$
\widehat Z_{\mathrm{norm}}(\eta)
=
\frac{1}{N_{\mathrm{MC}}}
\sum_{j=1}^{N_{\mathrm{MC}}}
\frac{1}{
1-\Phi\!\left(
\frac{-\mu_{z_s}}{\sigma_{z_s}}
\right)
}
P_{\mathrm{find}}\!\left(\theta_{E,j}\mid\eta\right)
g(\theta_{E,j},\gamma_j)\,
\mathbf{1}(z_{d,j} > 0, z_{s,j} > z_{d,j}, \theta_{E,j} > 0),
$$

其中前面的截断因子来自 $z_s \ge 0$ 的 source-redshift law。

超参数先验在主线源码里实现为盒先验，可写成

$$
\log p(\eta)
=
\begin{cases}
0, & \eta \in \mathcal{B}, \\
-\infty, & \eta \notin \mathcal{B},
\end{cases}
$$

其中 $\mathcal{B}$ 由当前 `gamma_mode` 对应的参数边界共同定义。也就是说，posterior 的真正数值入口不是“soft prior + likelihood”的近似，而是盒先验裁剪后的严格 support。

于是完整后验就是

$$
\log p(\eta \mid D)
=
\sum_{i=1}^{N_{\mathrm{lens}}}
\log \mathcal L_i(\eta)
- N_{\mathrm{lens}} \log Z_{\mathrm{norm}}(\eta)
+ \log p(\eta).
$$

这正是当前代码真正实现的统计结构。若只在单镜头项中乘选择权重、却不减去 $N_{\mathrm{lens}}\log Z_{\mathrm{norm}}$，就无法把已观测样本与 parent population 区分开来。

## Inference Strategy

群体层后验由 `emcee` 采样，目标分布就是上式的 $p(\eta \mid D)$。`devauc` 与 `sersic` 两条分支共享同一后验构造逻辑；它们的差异只体现在 profile-specific 的 structural law 和 sigma-unit table 上，而不是后验形式本身。

当前 `2026-04-20/21` 生产 run 在采样层共享同一组主参数：`24` 个 walkers、`10000` 个 steps、`2000` 个 warmup、`m10` 质量定义、`100000` 个 normalization samples。本文结果段统一使用 post-burn-in 扁平链，因此每个参数 $\eta_k$ 的摘要都写成

$$
q_{16}(\eta_k),\quad q_{50}(\eta_k),\quad q_{84}(\eta_k),
$$

其中 $q_p$ 表示去掉前 `2000` steps 后的 posterior samples 的第 $p$ 百分位数。

对当前被选中的 `2026-04-21` run，还需要额外强调三件事。第一，`config_snapshot.yaml` 已显式记录 `gamma_model.mode`。第二，`metadata.json` 也显式记录了 `fp_sigma_definition = within_re` 与 `fp_sigma_table_leaf_path = /within_re/m10`。第三，这并不意味着 likelihood 或 PPC 也切到 `within_re`；那部分仍使用 observation-flavor 对应的 observed-aperture sigma leaf。

### Current production sequence

`2026-04-20/21` 的主线 pipeline 不是抽象顺序，而是已经落在真实脚本与真实 run 目录上的生产编排。当前工作站上的现行顺序是：

1. `prepare_intepolation_grids` 维护 raw observation HDF5 与 sigma bundle。
2. `cmass_lens_inference cli run` 分别跑 `devauc` / `sersic` inference。
3. `cmass_lens_inference cli posterior-corner-latest` 生成 posterior corner。
4. `lensing_posterior_predictive posterior-predictive-monitor` 生成 PPC。
5. `lensing_posterior_predictive posterior-trends` 生成 Fig. 8-like 与其它 trend 图。
6. `lensing_posterior_predictive annotate-fig8-observations` 把 raw HDF5 中的观测点回填到已有 `fig8_like.png`。
7. `Bayesian_inference/scripts/compute_bic_after_20260420.py` 汇总 8 条 run 的 BIC 比较。

其中 `2026-04-21` 的 orchestrator 不是“只跑一对 profile”。`outputs/_staging/20260421_full4_slit_good_drop2sigma_within_re/run_full_pipeline.sh` 的真实顺序是：先跑 `devauc/sersic` 两条 `independent` inference，并对这两条 run 完整执行 corner、PPC、posterior trends 与 Fig. 8 回填；随后再跑 `devauc/sersic` 两条 `sigma_star_dependent` inference，并重复相同的后处理链。也就是说，当前主线的“结果产物”本身就已经是按 mode-aware bundle workflow 组织出来的。

这条生产链里，`within_re` 的角色也必须写清。FP prior 固定消费 `/within_re/m10` synthetic sigma，而 likelihood、PPC 和 posterior trends 里的 observed-aperture velocity dispersion 预测仍然读取 `/slit/m10`。因此 `within_re` 是一套独立 sigma definition，不是把 `slit` observation flavor 整体替换掉。

## Posterior Predictive Methodology

### Posterior-to-population forward model

`Posterior_predictive_test` 不是另一个推断器，而是从已经完成的后验出发做前向检验。它先从后验中抽样

$$
\eta^{(s)} \sim p(\eta \mid D),
$$

再在每个 posterior draw 下生成一批 latent parent-population galaxies：

$$
x_j^{(s)} \sim p_{\mathrm{pop}}(x \mid \eta^{(s)}).
$$

这里的 posterior draw 不是被“扁平地”解释成一套固定参数名，而是按与 inference 完全一致的 mode-aware schema 解包。因此，在前向生成时，$\mu_\gamma$ 也复用与上文相同的三种写法：

$$
\mu_\gamma^{(s)}
=
\begin{cases}
\mu_{\gamma,0}^{(s)}
+ \beta_{\gamma}^{(s)}(m_\ast-11.4)
+ \xi_{\gamma}^{(s)}\Delta_R,
& \texttt{dependent}, \\
\mu_{\gamma,0}^{(s)},
& \texttt{independent}, \\
\mu_{\gamma,0}^{(s)}
+ \beta_{\Sigma_\ast,\gamma}^{(s)}(\log \Sigma_\ast - 9),
& \texttt{sigma\_star\_dependent}.
\end{cases}
$$

也就是说，PPC 与 trend workflow 并没有引入另一套 $\gamma$ population law；它只是把与 inference 完全一致的 mode-aware 均值关系带进 posterior predictive forward model。

对每个 $x_j^{(s)}$，代码会计算

$$
\theta_{E,j}^{(s)},
\qquad
w_{\mathrm{det},j}^{(s)} = g(\theta_{E,j}^{(s)}, \gamma_j^{(s)}),
\qquad
w_{\mathrm{sel},j}^{(s)} = w_{\mathrm{det},j}^{(s)} \,
P_{\mathrm{find}}(\theta_{E,j}^{(s)} \mid \eta^{(s)}).
$$

因此三层人群可以写成：

- `parent`: 由 $p_{\mathrm{pop}}(x \mid \eta^{(s)})$ 直接生成的总体；
- `detectable`: 用 $w_{\mathrm{det}}$ 加权的几何可探测子总体；
- `selected`: 再乘上 $P_{\mathrm{find}}$ 后的 survey-like 样本。

### Histogram PPC

Histogram PPC 的目标不是重跑推断，而是比较 summary statistics 的 replicated distribution。对每个 posterior draw，代码从 `selected` population 中抽样得到两个 replicated catalogs：

$$
D_{\theta}^{\mathrm{rep},(s)} \sim \mathrm{Sample}_{23}(w_{\mathrm{sel}}),
\qquad
D_{\sigma}^{\mathrm{rep},(s)} \sim \mathrm{Sample}_{7}(w_{\mathrm{sel}}).
$$

其中 $\theta_E$ catalog 直接使用选中的 latent $\theta_E$，而 $\sigma$ catalog 则先根据 sigma-unit 表计算

$$
\sigma_{\mathrm{model}}^{(s)}
=
\sqrt{
S_{\mathrm{unit}}^{(s)}\,
10^{m_R^{(s)}}
},
$$

再加入 PPC 中使用的 measurement-like noise：

$$
\sigma_{\mathrm{rep}}^{(s)}
\sim
\mathcal N\!\left(
\sigma_{\mathrm{model}}^{(s)},
\left[f_\sigma \sigma_{\mathrm{model}}^{(s)}\right]^2
\right),
\qquad
f_\sigma = 0.0625.
$$

对每个 replicated catalog，代码都只比较四个统计量

$$
T \in \{\mathrm{median}, \mathrm{std}, p_{10}, p_{90}\}.
$$

真正进入 PPC 图和 `ppc_summary.json` 的对象是

$$
T\!\left(D_{\mathrm{obs}}\right)
\quad \text{vs.} \quad
\left\{
T\!\left(D^{\mathrm{rep},(s)}\right)
\right\}_{s=1}^{N_{\mathrm{draw}}}.
$$

其中 $\theta_E$ 的观测统计直接使用全部 23 个 lens 的 Einstein radius，而 $\sigma$ 的观测统计在当前 `good` raw 文件里先压缩成 11 个 lens-level 值；若某个 lens 有两个 velocity-dispersion measurements，则先做 inverse-variance weighted mean，再进入 $T(D_{\mathrm{obs}})$。需要显式指出的是：当前 `ppc_summary.json` 仍把 `sample_sizes.sigma` 写成历史常量 `7`，但 `2026-04-21 good_drop2sigma` 结果里真正参与观测侧 summary-statistic 计算的是这 11 个聚合后的 lens-level sigma 值；与此同时，Fig. 8 观测点回填仍保留 14 个原始 sigma measurement points。PPC 检验的因此不是逐个 lens 的 residual，而是整份 catalog 的样本级统计摘要是否像观测 catalog。

### Fig. 8-like trend bands

Fig. 8-like 趋势图采用另一种 posterior predictive 约简方式。对每个 posterior draw $s$ 和每个 stellar-mass bin $B$，代码把某个 quantity $y$ 压缩成三类 population-specific 曲线：

$$
\bar y_{\mathrm{parent}}^{(s)}(B)
=
\frac{1}{N_B}
\sum_{j \in B} y_j^{(s)},
$$

$$
\bar y_{\mathrm{detectable}}^{(s)}(B)
=
\frac{
\sum_{j \in B} w_{\mathrm{det},j}^{(s)} y_j^{(s)}
}{
\sum_{j \in B} w_{\mathrm{det},j}^{(s)}
},
$$

$$
\bar y_{\mathrm{selected}}^{(s)}(B)
=
\frac{
\sum_{j \in B} w_{\mathrm{sel},j}^{(s)} y_j^{(s)}
}{
\sum_{j \in B} w_{\mathrm{sel},j}^{(s)}
}.
$$

这里

$$
y \in \{m_{10}, \gamma, \sigma_{\mathrm{ap}}\}.
$$

最终图上的带并不是单次 simulation 曲线，而是每个 mass bin 上这些 draw-wise curves 的 percentiles：

$$
p_{16}(B), \quad p_{50}(B), \quad p_{84}(B).
$$

也就是说，`fig8_like_summary.json` 存储的是

$$
p\!\left(
\bar y(B)
\mid
\text{population class}, D
\right)
$$

的 percentile bands，而不是某条“最佳拟合直线”。

观测点叠加则直接读取 raw HDF5 中已写好的 single-lens flat-prior attrs。于是同一张图上的模型带和点误差条来自两个不同统计层级：

- 模型带来自 hierarchical posterior 的 posterior predictive distribution；
- 点误差条来自 per-lens flat-prior summaries。

## Current Local Results

本文结果段不再把 `2026-03-17` rerun 当作“当前主线结果”。当前证据锚点改为 `2026-04-20` 到 `2026-04-22` 这一轮真实生产 pipeline：`2026-04-20` 完成 `rebuilt_obs_within_re` 过渡 run，`2026-04-21` 完成 `good_drop2sigma_within_re` 的四组完整比较，`2026-04-22` 再用 `outputs/_staging/20260422_bic_after_20260420/bic_report.md` 做统一 BIC 筛选。由于 `outputs/devauc/latest` 与 `outputs/sersic/latest` 目前都指向已归档目录下的 basename，symlink 已经失效，所以下文统一引用显式 run 目录，而不是使用 `latest`。

这意味着，下面三组结果分别对应上文的三个数学对象，但数值证据改为：

- `devauc` 采用 `outputs/devauc/archived/20260421_170915_devauc_m10_sigma_star_fp_prior_slit_good_drop2sigma_within_re_20260421`；
- `sersic` 采用 `outputs/sersic/archived/20260421_163640_sersic_m10_independent_fp_prior_slit_good_drop2sigma_within_re_20260421`；
- 它们分别对应 BIC 汇总后的当前最优 `devauc` 与 `sersic` 分支，而不是沿用同一种 gamma 参数化的历史配对结果。

### 4/20-4/21 run matrix and model selection

本轮用于比较的 8 条 run 分成三组：

- `rebuilt_obs_within_re` 两条：`2026-04-20` 的 `devauc/sersic` `sigma_star_dependent` 过渡 run；
- `rebuilt_obs_within_re_test` 两条：`2026-04-21` 的 `devauc/sersic` `independent` 对照 run；
- `good_drop2sigma_within_re` 四条：`2026-04-21` 在 `good` raw HDF5 上同时跑 `independent` 与 `sigma_star_dependent` 的完整比较。

`2026-04-22` 的 BIC 汇总结果如下。这里的 `n=23` 对应 lens group 数，`k` 是各自的超参数维度；`independent` 为 `k=10`，`sigma_star_dependent` 为 `k=11`。

**devauc**

| Run | gamma mode | observation file | `k` | max log-like | BIC | `ΔBIC` |
| --- | --- | --- | --- | --- | --- | --- |
| `20260421_170915_devauc_m10_sigma_star_fp_prior_slit_good_drop2sigma_within_re_20260421` | `sigma_star_dependent` | `observations_deV_with_mass_grids_good.hdf5` | 11 | -34.775211 | 104.040859 | 0.000000 |
| `20260421_162512_devauc_m10_independent_fp_prior_slit_good_drop2sigma_within_re_20260421` | `independent` | `observations_deV_with_mass_grids_good.hdf5` | 10 | -43.352651 | 118.060244 | 14.019385 |
| `20260420_125501_devauc_m10_sigma_star_fp_prior_slit_rebuilt_obs_within_re_20260420` | `sigma_star_dependent` | `observations_deV_with_mass_grids.hdf5` | 11 | -44.082115 | 122.654666 | 18.613806 |
| `20260421_144356_devauc_m10_independent_fp_prior_slit_rebuilt_obs_within_re_test_20260421` | `independent` | `observations_deV_with_mass_grids.hdf5` | 10 | -53.413512 | 138.181967 | 34.141108 |

**sersic**

| Run | gamma mode | observation file | `k` | max log-like | BIC | `ΔBIC` |
| --- | --- | --- | --- | --- | --- | --- |
| `20260421_163640_sersic_m10_independent_fp_prior_slit_good_drop2sigma_within_re_20260421` | `independent` | `observations_with_mass_grids_all_good.hdf5` | 10 | -52.086028 | 135.526997 | 0.000000 |
| `20260421_172028_sersic_m10_sigma_star_fp_prior_slit_good_drop2sigma_within_re_20260421` | `sigma_star_dependent` | `observations_with_mass_grids_all_good.hdf5` | 11 | -51.279443 | 137.049321 | 1.522324 |
| `20260421_145549_sersic_m10_independent_fp_prior_slit_rebuilt_obs_within_re_test_20260421` | `independent` | `observations_with_mass_grids_all.hdf5` | 10 | -62.326496 | 156.007935 | 20.480937 |
| `20260420_130706_sersic_m10_sigma_star_fp_prior_slit_rebuilt_obs_within_re_20260420` | `sigma_star_dependent` | `observations_with_mass_grids_all.hdf5` | 11 | -61.558556 | 157.607549 | 22.080552 |

当前主线因此表现出一个明确但不对称的选择结果：

- `devauc` 明显偏好 `sigma_star_dependent`，相对同一 `good` 文件上的 `independent` 分支有 `ΔBIC = 14.02` 的优势；
- `sersic` 仅弱偏好 `independent`，相对 `sigma_star_dependent` 的优势只有 `ΔBIC = 1.52`；
- `2026-04-21` orchestrator 更新 `latest` 时把两条 `latest` 都指到了 sigma-star 那一对 basename，所以 `devauc/latest` 恰好与 BIC 最优一致，但 `sersic/latest` 并不等于当前 BIC 最优，而且两个 symlink 现在都已经是坏链。

### Posterior summaries

去掉前 `2000` steps 后，两条选中 run 各自提供 `192000` 个 posterior samples。由于当前被选中的两支 schema 不同，结果段不再强行使用一张共享 12 参数表，而是严格按各自 `parameter_order` 分开列出。

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

这组结果需要按“当前最优 profile 分支各自配套的 gamma mode”来解释，而不是继续把 `devauc` 与 `sersic` 当成同一参数化下的纯 profile 对照。当前可比的是：在 BIC 优选设定下，`devauc` 倾向较低的中心 `mu_gamma_0` 与更窄的 `sigma_gamma`，同时其 $\gamma$ 均值显式依赖 $\Sigma_\ast$；`sersic` 则保留 `independent` 模式，只用一个更宽的 `sigma_gamma` 来吸收群体散度。

### Posterior predictive checks

这一节对应的是 replicated summary-statistic 分布

$$
\left\{ T\!\left(D^{\mathrm{rep},(s)}\right) \right\}_{s}
$$

与观测统计量 $T(D_{\mathrm{obs}})$ 的比较。

当前 `good` raw HDF5 上，PPC 里的观测侧 $\sigma$ 不是历史文档里写的 “7 个 sigma lenses”，而是先把同一 lens 的多次 dispersion measurement 做 inverse-variance aggregation，再形成 `11` 个 lens-level $\sigma$ 值进入统计量。`ppc_summary.json` 里的 `sample_sizes.sigma = 7` 仍然是遗留常量字段，但 `observed.sigma.*` 数值本身已经对应这 `11` 个聚合值；它们与 raw `good` 文件直接重算出来的 median/std/p10/p90 一致。

四个最直接的 PPC 统计量如下：

| Statistic | Observed | `devauc` replicated mean | `sersic` replicated mean |
| --- | --- | --- | --- |
| `theta.median` | 1.2480 | 1.4346 | 1.3537 |
| `theta.std` | 0.3263 | 1.7517 | 0.5248 |
| `sigma.median` | 239.0 | 246.9596 | 264.0310 |
| `sigma.std` | 29.1868 | 44.2979 | 55.0334 |

第一，$\theta_E$ 与 $\sigma$ 的中位数仍然能被两条选中 run 大致重现，说明当前 posterior predictive catalog 在中心位置上没有明显失真。

第二，分布宽度仍然偏大，但偏差结构与旧的 `2026-03-17` 结果并不一样。`devauc` 的 $\theta_E$ 标准差明显偏宽，而 `sersic` 的 $\theta_E$ 宽度已经接近观测；相反，两个分支对 $\sigma$ 的 replicated 宽度都偏大，其中 `sersic` 更宽。这说明“模型整体过宽”仍然成立，但不应再照搬旧结果里那种对两条分支都同样失配的描述。

### Fig. 8-like trends near $\log M_\ast \approx 11.3$

这一节对应的是

$$
p\!\left(
\bar y(B)
\mid
\text{population class}, D
\right)
$$

的带状摘要，其中 $y \in \{m_{10}, \gamma, \sigma_{\mathrm{ap}}\}$。

在当前两条 BIC 优选 run 里，$\log M_\ast \approx 11.3$ 附近的中位带如下：

| Profile | Class | `m10` | `gamma` | `sigma_ap` (km/s) |
| --- | --- | --- | --- | --- |
| `devauc` | `parent` | 11.6607 | 1.9669 | 221.2968 |
| `devauc` | `detectable` | 11.6846 | 1.9912 | 232.4576 |
| `devauc` | `selected` | 11.6868 | 1.9933 | 233.5387 |
| `sersic` | `parent` | 11.6276 | 2.0267 | 225.5544 |
| `sersic` | `detectable` | 11.6745 | 2.0348 | 239.3056 |
| `sersic` | `selected` | 11.6807 | 2.0365 | 241.1194 |

这里的图像关系比旧版 `2026-03-17` 结果更简单：两条选中 run 都表现出从 `parent` 到 `selected` 的 `m10` 上升、`sigma_ap` 上升，以及温和的 $\gamma$ 上升。也就是说，在当前 4/21 BIC 优选口径下，selection 对 $\gamma$ 的影响不再呈现旧文档里那种 “`devauc` 推高、`sersic` 略压低” 的分裂图景。

同时还要记住，Fig. 8-like overlay 中观测点的口径与 PPC 不同：当前 `good` 文件里，annotation workflow 对应的是 `11` 个 `gamma` 点、`11` 个 mass 点和 `14` 个逐测量 `sigma` 点，因此这张图展示的是 raw flat-prior attrs 与 posterior predictive bands 的并置，而不是 23 个 lens 的统一 posterior list。

## Interpretation Boundaries

第一，single-lens flat-prior summary 与 hierarchical population posterior 仍然必须严格区分。前者服务于 Fig. 8-like 观测点 annotation，后者才是整套群体模型真正推断的对象；把两者混成“逐 lens 的 hierarchical posterior”会直接改写统计解释。

第二，`good` raw HDF5 不是删掉了 2 个 lens group，而是保留全部 `23` 个 lens，只把最低 `logM*` 的两颗 sigma lens 改写成 `num_sigma = 0`。因此当前主线的样本变化是“关闭两颗 lens 的 sigma 观测参与资格”，不是“从 catalog 中删除对象”。

第三，当前 `good` 文件上的三个 sigma 口径必须分开写：raw 文件里有 `11` 个 lens-level sigma objects；同一批 raw attrs 里保留 `14` 个逐测量 sigma points 供 Fig. 8 overlay 使用；PPC 则先做 lens-level inverse-variance aggregation，再用聚合后的 `11` 个值计算 `observed.sigma.*`。

第四，`ppc_summary.json` 中的 `sample_sizes.sigma = 7` 是遗留常量字段，不代表当前 `good` 文件的真实样本量。当前真实观测 sigma 样本量必须以 raw `good` HDF5 重算出的 `11` 个 lens-level 值为准。

第五，$m_{10}$ 不是独立于 $m_5$ 的新物理模型。它只是同一 power-law profile 在不同 aperture radius 下的 enclosed-mass 定义，因此对 `m10` 结果的解释始终应回到“质量定义变化”，而不是“模型家族变化”。

第六，当前 `2026-04-20/21` run 的 snapshot 已经显式记录 `gamma_model`、`fp_sigma_definition` 和 `fp_sigma_table_leaf_path`，所以不再需要沿用旧报告里那种“从 parameter order 反推 gamma mode”的 caveat。相应地，当前 `devauc` 与 `sersic` 的主结果本身就混合了 profile 选择与 gamma-mode 选择。

第七，`within_re` 只用于 FP prior 所引用的 sigma 定义与 sigma bundle path；likelihood、PPC 和 posterior-trends 里的观测 aperture sigma 仍然来自 `/slit/m10`。把这两层 aperture contract 合并成“现在都切到 within_re”会误写实际流程。

第八，`outputs/devauc/latest` 与 `outputs/sersic/latest` 当前都是坏链，因此正文不能把 `latest` 当作可靠证据锚点。更具体地说，4/21 orchestrator 把 `latest` 指到了 sigma-star 那一对 basename；这让 `devauc/latest` 恰好与 BIC 最优一致，但 `sersic/latest` 与当前 BIC 最优并不一致。

第九，BOSS raw observation branch 仍然只是当前代码支持的 observation contract 之一，不是本文主结果的证据来源。本文当前结果全部来自 slit rebuilt / good HDF5 分支。

第十，本文仍然只讨论主线代码、主线产物与本地已有 run 的方法含义，不延伸到历史 notebook、旧版对照流程或其它工程性重构的数值一致性审计。

## Notation And Code Mapping

下表只做最小必要的符号映射，用来把正文中的论文记号与代码语义对象对齐，而不是恢复工程流水账。

| Symbol | Meaning in this report | Code / data anchor |
| --- | --- | --- |
| $m_R$ | 固定 aperture $R$ 内的投影质量对数 | `m5` / `m10` mass definition |
| $m_5, m_{10}$ | $R=5,10\,\mathrm{kpc}$ 两种 enclosed-mass 表示 | raw HDF5 `mass_definitions/{m5,m10}` |
| $\theta_E$ | Einstein radius | raw observation attrs / `theta_ein_arcsec` |
| $S_{\mathrm{unit}}$ | 单位质量速度弥散响应 | upstream Jeans / PPC sigma table |
| $\mu_r(m_\ast,n)$ | mean size relation | profile-specific structural constants |
| $\Delta_R$ | size residual relative to $\mu_r$ | compiled context and PPC generator |
| $\mu_{m_R}$ | 条件质量均值 | `mu5_0` or `mu10_0`, `beta*`, `xi*` |
| $\mu_\gamma$ | 条件密度斜率均值 | mode-aware gamma population law |
| $\Sigma_\ast$ | 物理 stellar surface density | raw BOSS attrs `log10_Sigma_star` |
| $\beta_{\Sigma_\ast,\gamma}$ | `sigma_star_dependent` 模式中的 gamma-mean slope | `beta_sigma_star_gamma` |
| $P_{\mathrm{find}}$ | 发现概率 sigmoid | `theta0`, `loga` |
| $g(\theta_E,\gamma)$ | 强透镜截面项 | `cs_over_theta_ein` lookup |
| $\mathcal L_i$ | 第 $i$ 个 lens 的单镜头似然 | likelihood kernel |
| $Z_{\mathrm{norm}}$ | 选择归一化项 | normalization MC kernel |
| $D^{\mathrm{rep}}$ | posterior predictive replicated catalog | PPC workflow |
| `parent/detectable/selected` | 三层后验预测人群 | trend / PPC reducers |
| `boss` observation branch | 圆形 aperture 的 BOSS raw observation contract | BOSS raw HDF5 / `--build-boss-observation-hdf5` |

## Implementation Anchors

`prepare_intepolation_grids` 是这条方法链的物理前处理层。它把 Einstein-radius 约束转成 $m_5/m_{10}$ 轨迹，把 Jeans 动力学核转成 unit-mass 响应，并把这些量写回 raw HDF5 或汇总成 PPC 可调用的 sigma-unit 表。方法上，这一层的意义不是“准备缓存”，而是把后续推断真正需要的物理对象提前标准化。

`Bayesian_inference` 是群体层后验的核心实现层。这里定义了 profile-specific structural priors、单镜头 likelihood、样本归一化、参数模式切换以及 `emcee` 采样接口。方法上，这一层对应本文的 Hierarchical Model 与 Inference Strategy。

`Posterior_predictive_test` 是后验预测与趋势图层。它不重新定义模型，而是复用同一套 latent population law，把 posterior 变成 explicit replicated populations、summary-statistic PPC 和 Fig. 8-like mass-binned curves。方法上，这一层对应本文的 Posterior Predictive Methodology。
