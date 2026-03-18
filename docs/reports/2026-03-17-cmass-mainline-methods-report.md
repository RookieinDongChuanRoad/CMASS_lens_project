# CMASS 主线代码方法论报告

本文基于当前仓库主线研究代码与现有运行产物，对整条 CMASS strong-lens 分析链做一次方法论导向的重述。重点不是工程实现拆解，而是回答四个问题：这套代码在统计上建模了什么对象，为什么要这样分层，动力学与透镜信息如何耦合，posterior predictive checks 与 Fig. 8-like 趋势图究竟在检验什么。本文只覆盖 `prepare_intepolation_grids`、`Bayesian_inference`、`Posterior_predictive_test` 以及必要的数据语义说明，不讨论仓库中用于比较或回溯的辅助工作区。

## Overview

这套代码实现的是一条从“观测量”到“选择效应修正的群体后验”再到“后验预测诊断”的完整链条。上游首先把每个 lens 的 Einstein radius 约束转换成沿密度斜率 $\gamma$ 的 enclosed-mass 轨迹，并用 Jeans 核生成与 aperture velocity dispersion 对应的单位质量动力学响应；中游在群体层上同时建模 stellar-mass function、结构关系、enclosed mass、density slope、source-redshift 分布和发现概率，用层级贝叶斯方法推断超参数；下游再从该后验前向生成 parent population、detectable population 和 selected population，用 replicated catalogs 与质量分箱趋势图检查模型是否能重现观测样本的整体统计特征。

## Data And Derived Quantities

主线分析直接消费的观测量是 lens redshift $z_d$、source redshift $z_s$、Einstein radius $\theta_E$、stellar mass、effective radius，以及可用时的 aperture velocity dispersion。当前两份主 raw 文件 `data/raw/observations_deV_with_mass_grids.hdf5` 与 `data/raw/observations_with_mass_grids_all.hdf5` 都包含 23 个 lens group；其中 16 个 lens 没有速度弥散约束，4 个 lens 有 1 个速度弥散点，3 个 lens 有 2 个速度弥散点。因此，整套层级模型中的 lensing 与 selection 项作用于全部 23 个 lens，而动力学 likelihood 只作用于 7 个 `num_sigma > 0` 的 lens。

这套代码在物理上使用的是 projected enclosed mass 定义，而不是把“质量”当作单一无歧义的标量。当前显式支持两种 aperture 定义：$m_5 = \log_{10} M_{2D}(<5\,\mathrm{kpc})$ 与 $m_{10} = \log_{10} M_{2D}(<10\,\mathrm{kpc})$。它们不是两套互不相关的模型，而是同一 power-law mass profile 在不同 aperture radius 下的两种表示，代码中通过解析关系在二者之间转换，因此 $m_{10}$ 更适合被理解为“同一统计骨架下的另一种质量定义”，而不是新的物理假设。

Einstein radius 在方法上不是一个再与其他量并列拟合的弱观测，而是一个强约束：在给定 $(z_d, z_s, \theta_E)$ 的条件下，power-law 质量模型把二维 $(m_R, \gamma)$ 平面压缩为一条 $m_R(\gamma)$ 轨迹。后续单镜头 likelihood 并不是在独立的 $(m_R, \gamma)$ 二维网格上自由搜索，而是沿这条轨迹积分。这个设计直接决定了本项目如何把透镜信息与动力学信息耦合起来。

动力学部分不直接拟合一个自由的 $\sigma_\mathrm{model}$ 函数，而是通过 Jeans 响应表来完成。代码里实际使用的量是

$$
S_\mathrm{unit} = \frac{\sigma^2}{10^{m_R}}
$$

也就是说，Jeans 核先给出单位质量归一化下的速度弥散响应，随后再与 $10^{m_R}$ 相乘恢复物理 $\sigma_\mathrm{model}$。这让动力学项与质量定义 $m_5/m_{10}$ 保持严格一致，也使 posterior predictive 阶段可以对任意 replicated lens 做一致的 sigma 预测。

最后需要单独强调 Fig. 8 观测点的统计地位。项目自带的说明文件 `data/slacs_flatprior_m5_gamma_measurement_notes.md` 明确指出，这些每个 lens 的 $m_5/m_{10}$ 与 $\gamma$ 误差条来自 single-lens flat-prior inference：Einstein radius 先把 $(m_R, \gamma)$ 压成 $m_R(\gamma)$，再沿该轨迹用速度弥散 likelihood 得到一维 posterior，并提取中位数与置信区间。因此，Fig. 8 观测点不是群体层 hierarchical posterior 的 per-lens 投影，而是另一类统计对象。当前 raw HDF5 中已有这些摘要 attrs，供后续趋势图直接读取。

## Hierarchical Model

主线推断器有两个 profile 分支：`devauc` 与 `sersic`。它们共享同一统计骨架和同一后验结构，只在 tracer/结构先验上不同。`devauc` 分支固定 $n = 4$，适合 de Vaucouleurs 光度分布；`sersic` 分支则允许 Sersic index 进入结构关系，并在 likelihood 中使用观测到的 $n$。因此，这两个分支的差异不是“一个做选择效应修正、另一个不做”，而是“在相同群体模型框架下，用不同的光分布描述去约束相同的一组群体超参数逻辑”。

在群体层，这套代码显式建模以下量：

- deflector 的 stellar-mass function；
- 尺度关系，即 size 相对于 stellar mass 和 profile 参数的偏离；
- enclosed mass 的群体均值与散布；
- density slope $\gamma$ 的群体均值与散布；
- source redshift 的有效分布；
- 发现概率 $P_\mathrm{find}(\theta_E)$；
- 强透镜截面项。

其中，enclosed mass 与 $\gamma$ 不是彼此独立的自由场，而是条件于星族和结构量的群体分布。当前活跃 `2026-03-17` rerun 的参数向量包含

`mu10_0, beta10, xi10, sigma10, mu_gamma_0, beta_gamma, xi_gamma, sigma_gamma, mu_zs, sigma_zs, theta0, loga`

这说明本地当前主结果采用的是“dependent gamma”参数化：质量与密度斜率都可以随 stellar mass 和结构残差而系统变化，而不是完全独立的 $\gamma$ 模式。

单镜头 likelihood 的方法论核心在于它如何把 lensing、stellar structure、source distribution 与 dynamical constraint 乘在一起。给定一个 lens，代码并不把 $m_R$ 作为独立积分维度，而是先利用 Einstein radius 轨迹得到 $m_R(\gamma)$，然后在 $\gamma$ 与 latent stellar mass $m_\ast$ 上做二维积分。被积项包含以下因素：

- 观测 stellar mass 对 latent $m_\ast$ 的约束；
- stellar-mass function；
- 尺度关系 $P(\log r_e \mid m_\ast, n)$；
- 条件质量分布 $P(m_R(\gamma) \mid m_\ast, r_e, \eta)$；
- 条件密度斜率分布 $P(\gamma \mid m_\ast, r_e, \eta)$；
- source redshift 项；
- 发现概率 $P_\mathrm{find}$；
- lensing cross-section；
- 由 $\theta_E \rightarrow m_R(\gamma)$ 变换带来的 Jacobian；
- 若该 lens 有速度弥散观测，则再乘上 $P_\sigma$。

这意味着动力学与透镜信息的耦合并不是“两个独立 likelihood 相加”，而是：Einstein radius 决定质量轨迹，Jeans 响应把这条质量轨迹转成 $\sigma_\mathrm{model}(\gamma)$，然后速度弥散 likelihood 沿同一条轨迹重新加权。也正因为如此，只有 7 个有速度弥散数据的 lens 会直接影响 dynamical term，但全部 23 个 lens 都仍然会通过选择修正后的 lensing likelihood 约束群体参数。

选择效应修正不是一个额外的后处理权重，而是后验的核心组成。代码里对样本归一化的处理方式是：用同一个群体模型前向生成潜在 parent population，并计算该模型下“被 sample 进 lens catalog”的期望权重；这个量就是归一化项 $Z_\mathrm{norm}$。如果只在单镜头项中乘 selection weight，而不在样本归一化中除掉对应的期望，就无法真正恢复 parent population 的参数分布。

因此，后验的基本结构可以写成

$$
\log P(\eta \mid \mathrm{data})
= \sum_i \log L_i(\eta) - N_\mathrm{lens}\,\log Z_\mathrm{norm}(\eta) + \log \mathrm{prior}(\eta)
$$

其中 $L_i$ 是单镜头积分，$Z_\mathrm{norm}$ 是同一模型下的选择期望值。代码中的数值实现正是围绕这两个量组织起来的。

## Inference Strategy

群体层后验由 `emcee` 采样。当前活跃的两个本地结果都是 `m10` rerun，分别位于：

- `outputs/devauc/20260317_144144_devauc_m10_rerun_20260317`
- `outputs/sersic/20260317_144956_sersic_m10_rerun_20260317`

这两个 run 在采样层面使用相同的基本策略：`24` 个 walkers，`10000` 个 steps，`2000` 个 warmup。后验摘要可以从各自 `chain.h5` 中直接读取。对于本文的结果段，我统一采用 post-burn-in 扁平链，即去掉前 `2000` steps 后的全部样本，共 `192000` 个 posterior samples。

在数值上，推断器把后验计算拆成两个互补部分：一部分是全样本 likelihood，另一部分是样本归一化。前者对每个 lens 累积积分结果，后者则用 Monte Carlo 估计群体模型在选择函数下的总体期望。实现层面虽然使用预编译数组上下文和 `numba` kernel 来提高速度，但从方法论上看，这只是为了让“选择修正的层级后验”在数值上可跑；它并不改变模型的统计结构。

需要额外指出的是，当前 `2026-03-17` 的 `config_snapshot.yaml` 没有显式记录 `gamma_model` 字段。这是 snapshot 代际问题，而不是统计模型本身的歧义。就当前活跃结果而言，后验参数顺序明确包含 `beta_gamma` 和 `xi_gamma`，因此可以反向判断这些 run 使用的是 dependent gamma 模式。

## Posterior Predictive Methodology

`Posterior_predictive_test` 不是另一个推断器，而是从已经完成的 posterior 出发做前向检验。它复用与主模型一致的群体生成逻辑，先从后验链里抽样超参数，再在每个 posterior draw 下生成一批潜在 parent galaxies，给它们赋予 $m_\ast$、$r_e$、$n$、$m_R$、$\gamma$、$z_d$、$z_s$ 等隐变量，并据此计算 Einstein radius、lensing cross-section 与 discovery probability。

这样做之后，代码里自然出现三层人群：

- `parent`: 由群体模型直接生成的潜在总体；
- `detectable`: 在几何截面意义下能够形成并暴露为 strong lens 的子样本；
- `selected`: 再经过 survey-like 发现概率筛选后的最终样本。

这三层人群的区分，是理解后续诊断图的关键。

Histogram PPC 做的不是“重新拟合一次”，而是在每个 posterior draw 下显式抽出 replicated lens catalogs。当前主线工作流固定抽取：

- `23` 个 Einstein-radius lenses；
- `7` 个 sigma lenses。

然后分别对 replicated $\theta_E$ 与 replicated $\sigma$ 计算 `median / std / p10 / p90`，再看真实观测统计量在这些 replicated 分布中的位置。因此，这里的 PPC 不是逐 lens residual plot，而是样本级 summary-statistic 检验：它回答的是“这套层级模型生成出来的 catalog，其整体分布是否像观测 catalog”。

Fig. 8-like 趋势图对应的是另一种诊断思路。它不再抽一个固定的 23-lens catalog，而是对每个 posterior draw 先生成一整批 parent population，再按 stellar mass 分箱，对每个 mass bin 分别计算 `parent`、`detectable`、`selected` 三类人群在 $m_{10}$、$\gamma$、$\sigma_\mathrm{ap}$ 上的后验带。最终图上的 `p16/p50/p84` 曲线不是单次 simulation 的噪声实现，而是这些 mass-binned curves 在 posterior draws 上的统计摘要。

观测点叠加则来自 raw HDF5 中已经写好的 single-lens 摘要 attrs。换句话说，趋势图里的模型带是 hierarchical posterior 的前向预测，而点和误差条是 single-lens flat-prior 量测摘要。这两个对象被故意画在同一张图上，但它们不是同一统计层次的量。

## Current Local Results

本文结果段只使用 `2026-03-17` 的两个活跃 `m10` rerun，以及对应的 `ppc_summary.json` 和 `fig8_like_summary.json`。

### Posterior summaries

去掉前 `2000` steps 后，两个 run 各自提供 `192000` 个 posterior samples。12 个超参数的 `50/-16/+84%` 摘要如下。

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

当前 PPC 使用的是 `tail_capped_full_chain` 模式，即直接使用 post-burn-in 链尾部的 `192000` 个 posterior draws，而不是少量随机抽样。它给出的核心图景可以概括为两点。

第一，$\theta_E$ 与 $\sigma$ 的中位数基本可以被模型重现。以 `devauc` 为例，$\theta_E$ 的观测中位数是 `1.2480`，replicated mean of medians 为 `1.3025`；$\sigma$ 的观测中位数是 `263.0 km/s`，replicated mean of medians 为 `265.1450 km/s`。`sersic` 分支也给出类似结果：$\theta_E$ 中位数 `1.2888`，$\sigma$ 中位数 `267.6540 km/s`，都与观测样本接近。

第二，replicated catalog 的分布宽度明显偏大。最显眼的是标准差统计：观测 $\theta_E$ 标准差只有 `0.3263`，而 `devauc` 的 replicated mean of standard deviations 达到 `5.9761`，`sersic` 也达到 `4.4578`；观测 $\sigma$ 标准差为 `16.5386 km/s`，而 `devauc` 与 `sersic` 的 replicated mean 分别为 `70.4261` 与 `51.9504 km/s`。与此一致，`p90` 也普遍偏高。这表明当前层级模型虽然能把 catalog 的中心位置调到合理范围，但仍倾向于生成过宽的样本分布。

### Fig. 8-like trends near $\log M_\ast \approx 11.3$

在 $\log M_\ast = 11.3$ 附近，三层人群的差异已经很清楚。

对于 `devauc` 分支：

- $m_{10}$ 的中位趋势从 `parent` 的 `11.5751` 上升到 `detectable` 的 `11.6295`，再到 `selected` 的 `11.6434`；
- $\gamma$ 也从 `parent` 的 `2.1004` 上升到 `selected` 的 `2.1769`；
- $\sigma_\mathrm{ap}$ 从 `239.44 km/s` 上升到 `275.98 km/s`。

这说明在 `devauc` 分支下，selection 倾向于挑选更高 enclosed mass、更陡 density slope、也更高 aperture velocity dispersion 的对象。

对于 `sersic` 分支：

- $m_{10}$ 仍然从 `11.5435` 上升到 `11.6292`；
- $\sigma_\mathrm{ap}$ 从 `264.46 km/s` 上升到 `279.20 km/s`；
- 但 $\gamma$ 的行为不同：`parent` 中位数为 `2.2557`，而 `selected` 为 `2.2167`，即 selection 后并没有进一步推高 $\gamma$，反而略低于 parent 中位值。

因此，当前代码给出的信息不是“selection 总会把 $\gamma$ 推高”，而是：selection 对 $m_{10}$ 与 $\sigma_\mathrm{ap}$ 的推高是稳定的，但对 $\gamma$ 的影响依赖于 profile 分支与群体关系参数化。

## Interpretation Boundaries

第一，single-lens flat-prior summary 与 hierarchical population posterior 必须严格区分。前者是为了给每个 lens 生成 $m_5/m_{10}$ 与 $\gamma$ 的观测点和误差条；后者才是整套群体层模型真正推断的对象。把 Fig. 8 观测点误写成 hierarchical posterior 的 per-lens 结果，会直接改变统计解释。

第二，当前 raw 文件中这些 flat-prior 摘要 attrs 只存在于有速度弥散观测的 7 个 lens 上，而不是全部 23 个 lens。这意味着 Fig. 8 观测点本身对应的是“可做动力学摘要的子样本”，不是整个 lens sample 的逐对象后验列表。

第三，$m_{10}$ 不是一套独立于 $m_5$ 的新物理模型。它只是同一 power-law profile 在不同 aperture radius 下的 enclosed-mass 定义，因此对 $m_{10}$ 结果的解读应始终回到“质量定义变化”而不是“模型家族变化”。

第四，当前活跃 run 的 snapshot 没显式记录 `gamma_model` 字段，这是历史配置快照遗留问题。本文把它们解释为 dependent gamma 模式，是依据 posterior parameter order 与当前源码契约做出的有根据推断。

第五，本文只讨论当前主线代码在本地已有 run 上表现如何，不延伸到与历史 notebook 或其它对照流程的数值一致性比较。

## Implementation Anchors

`prepare_intepolation_grids` 是这条方法链的物理前处理层。它把 Einstein-radius 约束转成 $m_5/m_{10}$ 轨迹，把 Jeans 动力学核转成 unit-mass 响应，并把这些量写回 raw HDF5 或汇总成 PPC 可调用的大型 sigma-unit 表。方法上，这一层的意义不是“准备缓存”，而是把后续推断真正需要的物理对象提前标准化。

`Bayesian_inference` 是群体层后验的核心实现层。这里定义了 profile-specific structural priors、单镜头 likelihood、样本归一化、参数模式切换以及 `emcee` 采样接口。方法上，这一层对应本文的 Hierarchical Model 与 Inference Strategy。

`Posterior_predictive_test` 是后验预测与趋势图层。它不重新定义模型，而是复用同一套 latent population law，把 posterior 变成 explicit replicated populations、summary-statistic PPC 和 Fig. 8-like mass-binned curves。方法上，这一层对应本文的 Posterior Predictive Methodology。
