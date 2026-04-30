---
title: TDCOSMO XIX 速度弥散方法报告
tags:
  - cmass
  - velocity-dispersion
  - tdcosmo
  - paper-note
  - obsidian
source:
  - https://arxiv.org/abs/2502.16034
  - https://doi.org/10.1051/0004-6361/202554229
related:
  - "[[2026-03-17-cmass-mainline-methods-report]]"
created: 2026-03-30
---

# TDCOSMO XIX 速度弥散方法报告

## 摘要
这篇文章 [[TDCOSMO XIX. Measuring stellar velocity dispersion with sub-percent accuracy for cosmography]] 讨论的核心不是“换一个更复杂的 kinematic fitter”，而是如何把速度弥散测量中的残余系统误差压到适合 time-delay cosmography 的水平，并显式估计这些系统误差在样本内的协方差。作者的主张是：

1. 对高质量光谱，formal statistical error 通常不是瓶颈，主要瓶颈是模板库选择、连续谱处理和波段选择带来的 residual systematics。
2. 模板库选择不应该被当成一次性的 pipeline 决策，而应该被当成 nuisance choice；最终的速度弥散、系统误差和 covariance 都应对这些 choice 做边缘化。
3. 对高 S/N 的现代数据，经过 clean template libraries + BIC weighting 后，单个光谱的残余系统误差可以降到 sub-percent；对较低 S/N 的 SDSS/DR12 级数据，随机误差仍然较大，但模板引起的系统误差仍可控制在约 1--2% 水平。
4. 这篇文章对我们最大的启示，不是“立即替换你现有主层次模型”，而是：在主线模型里使用的 `sigma` 不应只是一串来自单一模板库、单一配置的点估计和对角误差，而应该升级为“按模板库 marginalize 过的 `sigma` summary + 样本级 covariance 结构”。

## 这篇文章在讨论什么

### 研究问题
作者关心的是强透镜时间延迟宇宙学里的速度弥散精度要求。文中一开始就强调，相对误差传播近似满足

$$
\delta H_0 / H_0 \approx 2\, \delta\sigma / \sigma,
$$

因此如果目标是 few-percent 的 $H_0$，则单个速度弥散测量不能只满足“看起来误差不大”，而是要在 accuracy 和 covariance 上都足够受控。

### 用了什么数据和方法
他们用四类 massive elliptical galaxy 光谱来系统检查速度弥散测量：

- MUSE
- KCWI
- JWST/NIRSpec
- SDSS

拟合代码统一使用 `pPXF`。作者不是只比较一个模板库，而是系统比较多种 empirical / semi-empirical / SSP 模板，并重点讨论以下因素：

- 模板库本身是否 clean
- 波段选择
- additive / multiplicative polynomial 阶数
- 模板库分辨率与数据分辨率的相对关系
- 这些选择如何在样本层面引入 shared covariance

### 他们的主要发现
文章摘要和结果段给出的主结论可以概括为：

- 对高质量数据，模板库选择是主要的残余系统误差源。
- clean 过的 stellar libraries 比“原始全库”重要得多；不清理模板，偏差可以到几个百分点，个别情况甚至 10--15%。
- SSP 和 semi-empirical alpha-enhanced 模板不适合 1% 级速度弥散工作。
- 在 MUSE/KCWI/JWST 这类高质量数据上，经过建议流程后，模板相关的系统误差和样本内 covariance 都能压到 sub-percent。
- 在 SDSS 级数据上，随机误差仍然比较大，但模板相关系统误差与 covariance 仍可压到约 1.5% 左右。

## 文章的方法论骨架

### 1. 模板库不是固定选择，而是 nuisance parameter
文章的最重要方法学观点是：不要先“选一个最喜欢的模板库”再报一个 `sigma`。正确做法是对多个 clean stellar libraries 都测一次，然后把模板库作为离散 nuisance choice 边缘化。

他们比较的主要 empirical stellar libraries 是：

- Indo-US
- MILES
- XSL

作者明确不推荐把 SSP 或 semi-empirical alpha-enhanced 模板当成主力高精度 kinematic 模板，因为那样会带来几 percent 级别的偏差。

### 2. 用 BIC 在样本层面给模板库赋权
他们不是按单个 galaxy 选模板库，而是对一个样本的总 BIC 赋权。BIC 定义为

$$
\mathrm{BIC} = k\ln n - 2\ln\hat L,
$$

对于一个样本，作者把所有 galaxy 的 likelihood 和像素数合并成样本级 BIC，再比较不同模板库的相对优劣。

作者特别强调：BIC 差值本身也有 sampling noise，因此并不是直接用 $\exp(-\Delta \mathrm{BIC}/2)$，而是先对 $\Delta\mathrm{BIC}$ 的不确定度做 bootstrap，再把“真实的”证据比做积分边缘化。用论文记号，模板库 $k$ 的权重是

$$
w_k = \int f(x)\, g(x\mid \Delta\mathrm{BIC}_k,\sigma_{\Delta\mathrm{BIC},k})\,dx,
$$

其中 $f(x)$ 对应标准 evidence-ratio 近似，$g$ 是由样本 bootstrap 得到的高斯近似。

这一步的重点不是“BIC 本身”，而是：

- 模板库权重由数据决定
- 权重是在样本层面估计的
- 权重自身的不确定度也被纳入了

### 3. 把系统误差和 covariance 从模板库分散性中显式导出
在得到了每个模板库的测量 $\sigma_k$ 和权重 $w_k$ 后，论文不是只给出一个 weighted mean，而是同时给出随机误差、系统误差和 covariance。

加权平均速度弥散是

$$
\bar\sigma = \frac{\sum_k w_k\sigma_k}{\sum_k w_k}.
$$

统计误差由各模板库拟合自身给出的 formal error 组合：

$$
\delta\bar\sigma = \sqrt{\frac{\sum_k w_k (\delta\sigma_k)^2}{\sum_k w_k}}.
$$

模板选择引入的系统误差由模板间分散性给出。因为库的数量有限，作者使用了 Bessel-corrected estimator：

$$
(\Delta_B\bar\sigma)^2 = \frac{\sum_k w_k (\sigma_k-\bar\sigma)^2}{\sum_k w_k - \sum_k w_k^2 / \sum_k w_k}.
$$

更重要的是，对样本内多个 galaxy，他们进一步估计模板选择引起的 covariance：

$$
C_{B,ij}\bar\sigma =
\frac{\sum_k w_k (\sigma_{k,i}-\bar\sigma_i)(\sigma_{k,j}-\bar\sigma_j)}{\sum_k w_k - \sum_k w_k^2 / \sum_k w_k}
+ \delta_{ij}\,\delta\bar\sigma_i^2.
$$

这一步对我们最关键，因为它明确指出：模板库引起的误差不是独立的；同一个库如果在一个样本上倾向性地推高 `sigma`，它通常会对整个样本的多个对象一起推高。

### 4. 推荐的 recipe
文章给出的操作 recipe 可以简化成下面几步：

1. 只在足够高的 S/N 上做 high-accuracy sigma inference。
2. 使用 clean stellar libraries，而不是全量原始模板库。
3. 先测试 polynomial orders，确认结果在一个稳定平台上。
4. 用所有可用的 clean libraries 都测一次。
5. 在样本层面计算 BIC，并据此给模板库赋权。
6. 用这些权重推导加权平均 sigma、系统误差和 covariance。
7. 如果模板相关系统误差已经很小，但还想进一步追到更低的系统误差，再对 polynomial orders 和小范围波段变化做边缘化。

## 文章里和我们最相关的数值结果

### 高 S/N 现代数据
对高质量数据，文章给出的 BIC-weighted 结果是：

- KCWI: 平均统计误差约 `0.86%`，Bessel-corrected 模板系统误差约 `0.67%`，样本内 off-diagonal covariance 振幅约 `0.47%`
- MUSE: 平均统计误差约 `2.04%`，Bessel-corrected 模板系统误差约 `0.78%`，off-diagonal covariance 振幅约 `0.45%`

同时，作者发现：

- KCWI 和 MUSE 更偏好 `Indo-US`
- 不是所有数据集都偏好同一个模板库

### SDSS 级数据
对 SDSS 样本，BIC-weighted 结果大致是：

- 平均统计误差约 `3.24%`
- Bessel-corrected 模板系统误差约 `1.46%`
- 样本内 off-diagonal covariance 振幅约 `1.27%`
- 模板权重几乎全落在 `XSL`

这里最重要的启示是：

- 在 SDSS/BOSS 质量的数据上，随机误差依然较大
- 但模板系统误差不是可以忽略的 0，而是约 1--2% 的量级
- 更关键的是，这个系统误差是 correlated 的，而不是每个 galaxy 独立抽样的噪声

## 对我们当前速度弥散使用的启示

## 1. 现在的问题不只是“BOSS 误差条太大”
结合 [[2026-03-17-cmass-mainline-methods-report]]，你当前主线层次模型里的动力学观测层，本质上还是把每个 lens 的 `sigma_obs` 当成一个已经压缩好的观测量输入 likelihood。这个做法对工程是方便的，但如果上游 `sigma` 产品来自：

- 单一模板库
- 单一波段选择
- 单一 polynomial setting
- 未显式传播模板选择带来的 shared covariance

那么主线模型就天然少了一层非常关键的系统误差控制。

也就是说，当前问题不是“formal sigma_err 会不会低估”，而是：

- 你现在喂给主线模型的 `sigma` 是否已经对模板库 choice 做过 marginalization
- 若没有，样本内共享系统误差就没有被表达出来

## 2. BOSS 和 7-lens 不应该被看成同一种 upstream sigma 产品
这篇文章直接告诉我们：模板偏好会随数据集变化。

文中结果是：

- KCWI/MUSE 更偏好 Indo-US
- SDSS 更偏好 XSL

这对你当前 BOSS 和 7-lens 的比较非常关键。它意味着：

- 即使 aperture 已经统一到同一个半径
- 即使你对两边都跑了 pPXF

只要两边的光谱质量、波段覆盖和模板偏好不同，你就不能天然把它们当作“同一种统计产品”的重复测量。

更具体地说：

- 你的 7-lens 高质量 slit/X-shooter 数据，更像文章中的高 S/N 现代数据场景
- BOSS 更接近文章中的 SDSS 场景
- 这两类数据很可能不偏好同一个模板库，也不会共享同样大小的系统误差或 covariance 结构

因此，如果未来要把 BOSS 和 7-lens 同时纳入主层次模型，正确做法不是先把两边强行校成一个“统一 sigma catalog”，而是：

- 保持两类观测通道各自的 aperture / spectral-contract
- 为每个 channel 单独生成 marginalize 过模板库的 sigma products
- 如有需要，再在主模型里加入 instrument-channel nuisance

## 3. 对我们当前 BOSS sigma 的直接提醒
你这批 BOSS sigma 目前是从 DR12 summary/product 读进来的，不是按这篇文章的 recipe 重新测出来的。结合这篇 paper，可以更清楚地看出风险：

- BOSS/SDSS 质量数据的 random error 本来就偏大
- 就算如此，模板系统误差和 covariance 仍然不是 0
- 而 catalog-level sigma 一般不会把“clean libraries + multi-library weighting + sample covariance”这一整层显式交给你

所以对主线 cosmography 来说，当前的 BOSS sigma 更适合被视为：

- 一个 useful but incomplete 的动力学观测输入
- 而不是已经达到 TDCOSMO XIX 那种“系统误差处理完成”的终态产品

## 4. 对你当前 7-lens X-shooter 测量的直接提醒
这篇 paper 也不支持“只选一个最优模板库 + 一套最优参数”就宣布 sigma 已经足够可靠。相反，它说明：

- 如果目标是普通 galaxy work，单一配置可能已经够用
- 但如果目标是 cosmography 级别，必须把模板库选择造成的 spread 和 covariance 显式估出来

因此对你当前 7-lens 的 high-S/N sigma，我认为这篇 paper 最直接的启示是：

- 你现在的 `ppxf_results_optimal.csv` 可以继续当作工作流中的一个中间产品
- 但最终进主线 inference 的 sigma，不应只来自这一份 optimal result
- 理想上应该升级为：对多个 clean libraries、稳定 polynomial range 和受控波段都测量一次，然后再压缩成一个 weighted sigma summary

## 5. 这篇文章对“BOSS + 7-lens 怎么同时利用”的具体启示
如果把这篇 paper 的思想翻译成你现在的工程问题，我认为最可行的路线是：

### 路线 A: 先升级 sigma product，再进主线层次模型
先不要改主层次模型本体，而是先把 BOSS 和 7-lens 的 sigma 观测产品升级成更接近 TDCOSMO XIX recipe 的版本。

对每个 channel 分别做：

1. 准备 clean template libraries
2. 在稳定的 polynomial orders 和波段上，对每个 spectrum 用多套库都测一次
3. 在样本层面计算 BIC weights
4. 导出每个对象的：
   - `sigma_bar`
   - `delta_sigma_stat`
   - `delta_sigma_sys`
5. 对整个样本导出模板相关 covariance matrix

之后再把这些 summary products 喂给主线模型。

### 路线 B: 在主线 likelihood 中显式加入样本级 sigma covariance
一旦上游拿到了 covariance matrix，主线里的 sigma likelihood 就不应再只用独立高斯项，而应按 channel 使用多元高斯：

$$
\log p(\boldsymbol\sigma^{\rm obs}\mid \boldsymbol\sigma^{\rm model})
\propto
-\frac{1}{2}(\boldsymbol\sigma^{\rm obs}-\boldsymbol\sigma^{\rm model})^T
C^{-1}
(\boldsymbol\sigma^{\rm obs}-\boldsymbol\sigma^{\rm model}).
$$

这里的 $C$ 至少应该包括：

- formal statistical errors
- 模板库选择带来的 correlated systematics

### 路线 C: 对 BOSS 和 7-lens 使用 channel-specific sigma products
由于 BOSS/SDSS 与高 S/N slit data 的模板偏好可能不同，合理做法是：

- 为 BOSS 23 个对象生成一套 `sigma_BOSS_summary + C_BOSS`
- 为 7-lens slit/X-shooter 生成一套 `sigma_slit_summary + C_slit`
- 在主线模型里把它们视为同一 23 个 lens 上的两类观测通道，而不是先压成一个统一 catalog

## 对我们下一步工作的具体建议

### 优先级 1: 不要再把单模板库 sigma 当成最终产品
对 7-lens 和 BOSS，优先把“多模板库、多设置测量 -> 加权 sigma summary + covariance”这个步骤补上。单一 `optimal` 结果仍可保留，但不应再作为最终 cosmography-level sigma product。

### 优先级 2: 先做 instrument-specific sigma remeasurement
具体建议：

- 7-lens: 按高 S/N slit 光谱特性，先测 Indo-US / XSL / MILES clean libraries 的一致性
- BOSS: 对 DR12/BOSS 光谱重新做多模板库测量，而不是继续直接依赖 catalog sigma
- 两边各自先在自己的样本层面算 BIC 权重，不要强行共用一套权重

### 优先级 3: 在主线 inference 中显式支持 sigma covariance
当上游 sigma product 准备好后，把主线动力学 likelihood 从“独立高斯点”升级为“允许 channel-specific covariance 的高斯向量 likelihood”。

### 优先级 4: overlap 的 7 个 lens 用来检验 channel offset，而不是直接混为同类测量
这 7 个重叠对象最重要的作用不是“手工校正 BOSS 到 X-shooter”，而是：

- 检查 BOSS 和 7-lens channel 是否存在 residual offset
- 检查二者的 scatter 是否能被各自的 stat+sys budget 解释
- 决定主线模型中是否需要加入额外的 instrument nuisance

## 对当前项目的直接结论
这篇文章给你的最核心启示可以压缩成一句话：

**如果目标是强透镜宇宙学级别的速度弥散使用，真正需要进入主层次模型的不是单个 `sigma` 点估计，而是对模板库 choice 做过边缘化、并显式携带样本级 covariance 的 `sigma` 观测产品。**

对你现在的项目，这意味着：

1. BOSS 和 7-lens 应该继续保留为两种不同 observation channel。
2. 两边都需要多模板库 remeasurement，而不是只依赖单一 pipeline 输出或 catalog sigma。
3. 在主线层次模型里，最值得新增的不是另起一个外部 `sigma-M*-Re` 经验模型，而是把 sigma likelihood 升级到可以吃 channel-specific covariance 的形式。

## 参考链接
- arXiv: [TDCOSMO XIX. Measuring stellar velocity dispersion with sub-percent accuracy for cosmography](https://arxiv.org/abs/2502.16034)
- A&A DOI: [10.1051/0004-6361/202554229](https://doi.org/10.1051/0004-6361/202554229)
- 相关本地方法报告：[[2026-03-17-cmass-mainline-methods-report]]
