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

推断超参数 $\eta$；最后，再从 $p(\eta \mid D)$ 前向生成 parent population、detectable population 和 selected population，用 replicated catalogs 与质量分箱趋势带检查模型是否能重现观测样本的整体统计特征。当前主线在这个统一框架上又增加了两项能力：一是 `gamma_model.mode = sigma_star_dependent`，二是从 BOSS summary table 独立重建的 BOSS raw observation HDF5 产品；但本文的数值结果段仍只引用 `2026-03-17` 的 active `m10` dependent-gamma rerun，不把这些新增能力误写成已有新 posterior 结果。

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

- 全部 23 个 lens 都贡献透镜和选择项；
- 只有 7 个 `num_sigma > 0` 的 lens 进入动力学 likelihood；
- `devauc` 分支取 $n_i^{\mathrm{obs}} = 4$；
- `sersic` 分支使用观测到的 $n_i^{\mathrm{obs}}$。

这组观测量在统计上的角色并不对称。$z_d$ 与 $z_s$ 决定透镜几何，$\theta_E$ 决定质量轨迹 $m_R(\gamma)$，$\log M_\ast^{\mathrm{obs}}$ 与 $\log R_e^{\mathrm{obs}}$ 约束群体层关系中的 latent structural state，而 $\sigma_{\mathrm{ap}}^{\mathrm{obs}}$ 只在有动力学数据时参与单镜头似然。

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

在当前活跃的 `2026-03-17` 本地结果里，这组参数对 $R=10\,\mathrm{kpc}$ 写成

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

这对应当前活跃 `2026-03-17` rerun 使用的 `dependent` gamma 模式。与此同时，主线源码现在把 $\mu_\gamma$ 的参数化显式推广成三种 mode-aware 写法：

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

需要强调的是，`sigma_star_dependent` 改变的只是 $\mu_\gamma$ 的群体均值参数化；单镜头 likelihood 的积分结构、selection normalization 的定义，以及总体后验主公式都保持不变。它是当前主线已支持的模型能力，但不是本文当前本地主结果的证据来源。

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

当前活跃的两个本地结果都是 `2026-03-17` 的 `m10` rerun，分别位于：

- `outputs/devauc/20260317_144144_devauc_m10_rerun_20260317`
- `outputs/sersic/20260317_144956_sersic_m10_rerun_20260317`

这两个 run 使用相同的采样框架：`24` 个 walkers、`10000` 个 steps、`2000` 个 warmup。本文结果段统一使用 post-burn-in 扁平链，因此每个参数 $\eta_k$ 的摘要都写成

$$
q_{16}(\eta_k),\quad q_{50}(\eta_k),\quad q_{84}(\eta_k),
$$

其中 $q_p$ 表示去掉前 `2000` steps 后的 posterior samples 的第 $p$ 百分位数。

需要额外说明的是，当前 `2026-03-17` 的 `config_snapshot.yaml` 没有显式记录 `gamma_model` 字段。这是 snapshot 代际问题，而不是统计模型本身的歧义。就当前活跃结果而言，posterior parameter order 明确包含 `beta_gamma` 和 `xi_gamma`，因此这些 run 可以被解释为 dependent gamma 模式。

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

其中 $\theta_E$ 的观测统计直接使用全部 lens 的 Einstein radius，而 $\sigma$ 的观测统计先压缩成 7 个 lens-level 值；若某个 lens 有两个 velocity-dispersion measurements，则先做 inverse-variance weighted mean，再进入 $T(D_{\mathrm{obs}})$。因此它检验的不是逐个 lens 的 residual，而是整份 catalog 的样本级统计摘要是否像观测 catalog。

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

本文结果段只使用 `2026-03-17` 的两个活跃 `m10` rerun，以及对应的 `chain.h5`、`ppc_summary.json` 和 `fig8_like_summary.json`。这两条 run 仍按 dependent gamma 模式解释，不把 `sigma_star_dependent` 或 BOSS observation 分支混入当前数值证据。因此，下面三组结果分别对应上文的三个数学对象：

- 超参数表对应 $p(\eta \mid D)$ 的边缘 posterior 摘要；
- PPC 对应 $T(D^{\mathrm{rep}})$ 与 $T(D_{\mathrm{obs}})$ 的比较；
- Fig. 8-like 趋势对应不同 population class 条件下的 posterior predictive bands。

### Posterior summaries

去掉前 `2000` steps 后，两个 run 各自提供 `192000` 个 posterior samples。12 个超参数的 $q_{50}^{+q_{84}-q_{50}}_{-q_{50}+q_{16}}$ 摘要如下。

| Parameter | devauc | sersic |
| --- | --- | --- |
| `mu10_0` | 11.6389 (-0.0552/+0.0357) | 11.5855 (-0.0489/+0.0366) |
| `beta10` | 0.6747 (-0.1328/+0.1316) | 0.4335 (-0.0998/+0.1107) |
| `xi10` | -0.8154 (-0.3086/+0.2694) | -0.9341 (-0.1862/+0.1749) |
| `sigma10` | 0.0440 (-0.0245/+0.0358) | 0.0398 (-0.0201/+0.0267) |
| `mu_gamma_0` | 1.9965 (-0.1343/+0.1222) | 2.1957 (-0.1180/+0.0811) |
| `beta_gamma` | -0.9653 (-0.3559/+0.4402) | -0.6200 (-0.2478/+0.3670) |
| `xi_gamma` | -1.1961 (-1.2245/+1.4312) | 0.5903 (-0.7786/+0.5067) |
| `sigma_gamma` | 0.1136 (-0.0566/+0.0676) | 0.0901 (-0.0542/+0.0877) |
| `mu_zs` | 1.2615 (-0.1817/+0.2485) | 1.2630 (-0.1801/+0.2437) |
| `sigma_zs` | 0.8245 (-0.1346/+0.1441) | 0.8231 (-0.1324/+0.1581) |
| `theta0` | 0.6657 (-0.3831/+0.4258) | 0.6757 (-0.3894/+0.4905) |
| `loga` | 0.9856 (-1.3811/+1.2346) | 0.9228 (-1.3739/+1.2233) |

从这组后验可以看出，两条分支对 source-redshift 与 selection 参数给出了相当接近的结果，但对 `mu_gamma_0` 与 `xi_gamma` 的偏好不同：`sersic` 分支更偏向较高的中心 $\gamma$，同时对结构残差的响应方向与 `devauc` 并不相同。这说明在当前数据与模型设定下，光分布假设会显著影响对 density-slope population relation 的解释。

### Posterior predictive checks

这一节对应的是 replicated summary-statistic 分布

$$
\left\{
T\!\left(D^{\mathrm{rep},(s)}\right)
\right\}_{s}
$$

与观测统计量 $T(D_{\mathrm{obs}})$ 的比较。

第一，$\theta_E$ 与 $\sigma$ 的中位数基本可以被模型重现。以 `devauc` 为例，$\theta_E$ 的观测中位数是 `1.2480`，replicated mean of medians 为 `1.3025`；$\sigma$ 的观测中位数是 `263.0 km/s`，replicated mean of medians 为 `265.1450 km/s`。`sersic` 分支也给出类似结果：$\theta_E$ 中位数 `1.2888`，$\sigma$ 中位数 `267.6540 km/s`，都与观测样本接近。

第二，replicated catalog 的分布宽度明显偏大。最显眼的是标准差统计：观测 $\theta_E$ 标准差只有 `0.3263`，而 `devauc` 的 replicated mean of standard deviations 达到 `5.9761`，`sersic` 也达到 `4.4578`；观测 $\sigma$ 标准差为 `16.5386 km/s`，而 `devauc` 与 `sersic` 的 replicated mean 分别为 `70.4261` 与 `51.9504 km/s`。与此一致，`p90` 也普遍偏高。这表明当前层级模型虽然能把 catalog 的中心位置调到合理范围，但仍倾向于生成过宽的样本分布。

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

在 $\log M_\ast = 11.3$ 附近，三层人群的差异已经很清楚。

对于 `devauc` 分支：

- $m_{10}$ 的中位趋势从 `parent` 的 `11.5751` 上升到 `detectable` 的 `11.6295`，再到 `selected` 的 `11.6434`；
- $\gamma$ 也从 `parent` 的 `2.1004` 上升到 `selected` 的 `2.1769`；
- $\sigma_{\mathrm{ap}}$ 从 `239.44 km/s` 上升到 `275.98 km/s`。

这说明在 `devauc` 分支下，selection 倾向于挑选更高 enclosed mass、更陡 density slope、也更高 aperture velocity dispersion 的对象。

对于 `sersic` 分支：

- $m_{10}$ 仍然从 `11.5435` 上升到 `11.6292`；
- $\sigma_{\mathrm{ap}}$ 从 `264.46 km/s` 上升到 `279.20 km/s`；
- 但 $\gamma$ 的行为不同：`parent` 中位数为 `2.2557`，而 `selected` 为 `2.2167`，即 selection 后并没有进一步推高 $\gamma$，反而略低于 parent 中位值。

因此，当前代码给出的信息不是“selection 总会把 $\gamma$ 推高”，而是：selection 对 $m_{10}$ 与 $\sigma_{\mathrm{ap}}$ 的推高是稳定的，但对 $\gamma$ 的影响依赖于 profile 分支与群体关系参数化。

## Interpretation Boundaries

第一，single-lens flat-prior summary 与 hierarchical population posterior 必须严格区分。前者是为了给每个 lens 生成 Fig. 8 风格的 $m_5/m_{10}$ 与 $\gamma$ 点误差条；后者才是整套群体层模型真正推断的对象。把 single-lens 摘要误写成 hierarchical posterior 的 per-lens 结果，会直接改变统计解释。

第二，当前 raw 文件中的这些 flat-prior 摘要 attrs 只存在于有速度弥散观测的 7 个 lens 上，而不是全部 23 个 lens。这意味着 Fig. 8 观测点本身对应的是“可做动力学摘要的子样本”，不是整个 lens sample 的逐对象后验列表。

第三，$m_{10}$ 不是独立于 $m_5$ 的新物理模型。它只是同一 power-law profile 在不同 aperture radius 下的 enclosed-mass 定义，因此对 $m_{10}$ 结果的解读应始终回到“质量定义变化”，而不是“模型家族变化”。

第四，当前活跃 run 的 snapshot 没显式记录 `gamma_model` 字段，这是历史配置快照遗留问题。本文把它们解释为 dependent gamma 模式，是依据 posterior parameter order 与当前源码契约做出的有根据判断。

第五，trend 图中的 $\sigma_{\mathrm{ap}}$ 观测点和 PPC 中用于计算 $T(D_{\mathrm{obs}})$ 的 7-lens sigma 样本不是同一个对象。前者在 overlay 里保留 raw HDF5 的逐测量点，`num_sigma=2` 的 lens 会画出两个点；后者则先把同一 lens 的多次 dispersion measurement 聚合成一个 lens-level 值，再进入 replicated-statistic 比较。

第六，`sigma_star_dependent` 是新增支持的 gamma population mode，不等于本文已经给出了该模式下的新 posterior 结果。本文当前结果段仍然只对应 `2026-03-17` 的 dependent-gamma rerun。

第七，BOSS raw observation HDF5 是新增上游数据分支，不等于本文当前结果已经切换到 BOSS sample。正文对 BOSS 的讨论只涉及主线现在支持什么 observation contract，以及这些 attrs 如何进入方法定义。

第八，本文不展开最近其它工程演化，例如 `fig8 observation annotation workflow`、`sigma bundle / bundle-aware PPT loading` 和 `Avoid recomputing sigma tables for m10 mass radii`。这些变更可能影响工作流接口或数据搬运，但不属于本次方法论修订的主角。

第九，本文只讨论当前主线代码在本地已有 run 上表现如何，不延伸到与历史 notebook 或其它对照流程的数值一致性比较。

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
