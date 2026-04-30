# 建议文件名：`h_dependence_audit_updated.md`

> 审计范围说明  
> 1. 本文只审计 $h$ 依赖；默认把其他宇宙学参数固定，只抽取纯 $h$ 缩放。  
> 2. 上传的是 Markdown 源文稿，不是排版后的 PDF；因此“出现位置”列统一用“章节 + 源文件行号”，不使用页码。  
> 3. 本版更新纳入用户补充信息：上游 $\log M_\ast^{\rm obs}$ 确实来自 photometry / luminosity / SED / SPS 链。因此本文把 stellar mass 的纯 $h$ 缩放锁定为 $M_\ast\propto h^{-2}$，不再记作待定指数。  
> 4. 下文中若写 $Q \propto h^n$，默认表示在固定 $(\Omega_M,\Omega_\Lambda,\dots)$ 形状参数时，仅由 $H_0$ 改变引入的缩放。

---

## 1. 核心结论摘要

这份方法报告描述的是一条 CMASS strong-lens 的层级贝叶斯分析链：单镜头层面用 $(z_d,z_s,\theta_E,\log M_\ast,\log R_e,\sigma_{\rm ap})$ 约束幂律质量分布与 Jeans 动力学，群体层面再联合建模 stellar-mass function、size relation、enclosed mass、density slope、source-redshift 分布和选择函数。  

本版审计把用户补充的上游信息纳入判断：$\log M_\ast^{\rm obs}$ 来自 photometry / luminosity / SED / SPS 链，因此在固定其余宇宙学参数、只抽取 $H_0$ 缩放时，
$$
D_L\propto h^{-1},\qquad L\propto D_L^2\propto h^{-2},\qquad M_\ast=(M_\ast/L)L\propto h^{-2},
$$
其中 SED/SPS 主要给出 $M_\ast/L$，其纯距离缩放为 $h^0$。  

所以，stellar mass 的 log 重标不再是待定项，而应写成
$$
m_\ast(h)=m_{\ast,\rm ref}-2\log_{10}\!\left(\frac{h}{h_{\rm ref}}\right).
$$
这会同步影响 $\log M_\ast$ 质量分箱、stellar-mass function 的横轴 convention、以及所有以 $m_\ast$ 作为条件变量的群体层关系。  

文稿里另一个可以直接从公式推出的结论仍然是：当前主线真正建模的透镜质量不是“Einstein radius 内质量”，而是固定物理半径 $R=5,10\,{\rm kpc}$ 内的二维投影质量 $M_{2\rm D}(<R)$。在该定义下，
$$
M_{2\rm D}(<R_{\rm fixed\ kpc})\propto h^{2-\gamma},
$$
而不是常见 Einstein mass 或 dynamical mass 场景里的统一 $h^{-1}$。  

因此，你的核心怀疑在本版审计中被确认：**stellar mass 与文稿当前主 lensing-constrained mass 的 $h$ 依赖不同。** 前者是 $h^{-2}$，后者是 $h^{2-\gamma}$，并且后者的指数还随 $\gamma$ 改变。  

一个重要的正面结果是：若 $R_e$ 是由角半径经 $D_A$ 转成的 kpc 量，则 $R_e\propto h^{-1}$，因此文稿中的
$$
\Sigma_\ast=\frac{M_\ast}{2\pi R_e^2}
$$
在纯 $h$ 缩放下是 $h^0$。这意味着 `sigma_star_dependent` 模式使用的 $\log\Sigma_\ast-9$ 是本项目里较接近 $h$-robust 的回归自变量，但前提是上游 $M_\ast$ 与 $R_e$ 确实使用同一个 $h_{\rm ref}$。  

最危险的比较不再是“stellar mass 指数未知”，而是**两个已知但不同的 $h$ 指数被放在同一图、同一回归或同一质量比中却没有显式标注**。尤其是 $m_{10}$ vs $\log M_\ast$、$\mu_{m_R}(m_\ast,R_e,n)$、Fig. 8-like mass-binned trends、single-lens flat-prior mass points、以及任何 stellar fraction / dark-matter fraction 解释。  

如果要改变 $h$ convention，$M_\ast$ 可以整体平移 $-2\log_{10}(h/h_{\rm ref})$，但 fixed-kpc $m_R$ 不能全样本统一平移；它必须逐 lens、逐 posterior draw 使用
$$
m_R(h)=m_{R,\rm ref}+(2-\gamma)\log_{10}\!\left(\frac{h}{h_{\rm ref}}\right).
$$
所以当前项目最优先的下一步，是把 $h_{\rm ref}$、$M_\ast$ 的 $-2$ 指数、$R_e$ 的 $-1$ 指数、以及 $m_5/m_{10}$ 的 $(2-\gamma)$ 指数写进 raw HDF5 attrs、summary JSON、图轴和表头。  

### A. 项目理解摘要（8–12 句话）

1. 这份文档对应的是一个对 CMASS strong-lens 样本做选择效应修正的层级贝叶斯分析。  
2. 单镜头输入量包括 deflector/source redshift、Einstein radius、observed stellar mass、observed effective radius、Sérsic index，以及可用时的 aperture velocity dispersion。  
3. 文稿把 $(m_R,\gamma)$ 的二维自由度先用 Einstein radius 压成一条 $m_R(\gamma)$ 轨迹，再用 Jeans 响应把这条轨迹映射成 $\sigma_{\rm model}(\gamma)$。  
4. 群体层则对 stellar-mass function、size relation、enclosed mass、density slope、source-redshift 分布和选择函数联合建模。  
5. 完整后验的关键结构是 $\sum_i \log \mathcal L_i - N_{\rm lens}\log Z_{\rm norm} + \log p(\eta)$，所以选择修正不是附加权重，而是后验本体的一部分。  
6. 这份文稿当前的主质量定义是 $m_{10}$（或 $m_5$），即固定物理 aperture 内的二维投影质量，而不是“Einstein radius 内质量”作为主参数。  
7. posterior predictive 部分再从后验抽样生成 parent / detectable / selected 三层 population，并用 replicated catalogs 与 Fig. 8-like 趋势带检验模型。  
8. 文稿中最核心的观测量是 $z_d,z_s,\theta_E,\sigma_{\rm ap}^{\rm obs}$；最核心的推导量是 $R_e$、$m_\ast$、$\Sigma_c$、$m_R$、$\Sigma_\ast$、$\sigma_{\rm model}$。  
9. 本版已确认 $M_\ast\propto h^{-2}$，而 fixed-kpc lensing mass 满足 $M_R\propto h^{2-\gamma}$，所以二者的比较需要显式 convention，而不是默认同一个质量单位即可。  
10. 尤其是当前文稿同时出现了 $m_{10}$、$\log M_\ast$、$\log \Sigma_\ast$、$\mu_{m_R}$ 和 Fig. 8-like stellar-mass binning，因此“单位一致但 $h$ 指数不一致”的风险很集中地落在 mass–mass 与 mass–size 关系上。  

### B. 观测量 vs 推导量分类

> 说明：文稿把 $\log M_\ast^{\rm obs}$、$\log R_e^{\rm obs}$、$n^{\rm obs}$ 放进了 $D_i$ 这个“observed quantities entering the model”的集合里；但从“观测量 vs 推导量”的审计语义上看，它们并不是 detector-level direct observables，而是上游产品。

#### B1. 直接观测量

| 量 | 符号 | 为什么属于这一类 | 是否可能在后续引入 $h$ 依赖 |
|---|---|---|---|
| deflector redshift | $z_d$ | 直接观测到的红移；本身无物理尺度单位 | 自身 $h^0$；一旦进入 $D_d$ 会引入 $h^{-1}$ |
| source redshift | $z_s$ | 同上 | 自身 $h^0$；进入 $D_s,D_{ds}$ 后引入 $h^{-1}$ |
| Einstein radius（角量） | $\theta_E$ | 角度是直接几何观测量 | 自身 $h^0$；转成 $r_{\rm Ein}$ 或质量后才引入 $h$ |
| aperture velocity dispersion | $\sigma_{\rm ap}^{\rm obs}$ | 直接由谱线宽度测得的速度量 | 通常 $h^0$；与 mass 结合时只是要求其他量的 $h$ 依赖相互抵消 |
| aperture / seeing 几何（arcsec） | aperture radius, seeing FWHM | 文稿给的是角口径定义 | 自身 $h^0$；若用于建立 physical aperture，则会引入 $h^{-1}$ |

#### B2. 推导量

| 量 | 符号 | 为什么属于这一类 | 是否可能引入 $h$ 依赖 |
|---|---|---|---|
| observed stellar mass product | $\log M_\ast^{\rm obs}$ | 虽然在模型里作为“observed input”，但物理上是上游 photometry / luminosity / SED / SPS 产物 | 是；本版已确定为 $M_\ast\propto h^{-2}$ |
| latent stellar mass | $m_\ast$ | 层级模型中的潜变量，不是直接观测量 | 是；继承 $\log M_\ast^{\rm obs}$ 的 $h^{-2}$ convention |
| observed effective radius（若已转成 kpc） | $\log R_e^{\rm obs}$ | 物理半径来自角尺度乘距离 | 是；通常 $R_e \propto h^{-1}$ |
| Sérsic index | $n^{\rm obs}$ | 来自成像拟合，不是 detector-level primitive observable | 一般 $h^0$（无量纲） |
| angular-diameter / luminosity distances | $D_A,D_L$ | 都是 redshift 经 cosmology 变换得到的量 | 是；固定 $\Omega$ 时 $h^{-1}$ |
| physical Einstein radius | $r_{\rm Ein}$ | 由 $\theta_E$ 和 $D_d$ 推得 | 是；$h^{-1}$ |
| critical surface density | $\Sigma_c$ | 由距离组合得到 | 是；$h^{+1}$ |
| fixed-kpc lensing mass | $M_{2\rm D}(<R),m_R,m_5,m_{10}$ | 由 $\Sigma_c$、$r_{\rm Ein}$、$\gamma$、固定物理半径共同得到 | 是；$h^{2-\gamma}$ |
| unit-mass Jeans response | $S_{\rm unit}$ | 由动力学表格预计算出来的派生量 | 是；其具体 rescaling 需与 mass definition 和 dynamics grid 一起审计 |
| model velocity dispersion | $\sigma_{\rm model}$ | 由 $S_{\rm unit}$ 和 $m_R$ 共同推得 | 目标上应为 $h^0$；但要靠整个 grid / mass convention 一致实现 |
| stellar surface density | $\Sigma_\ast$ | 由 $M_\ast$ 和 $R_e$ 构成 | 在本项目的标准链路中纯 $h$ 缩放为 $h^0$，因为 $M_\ast\propto h^{-2}$ 且 $R_e^2\propto h^{-2}$ |
| size residual | $\Delta_R$ | 由 $R_e$ 与外部 size relation 差值得到 | 可能；取决于 size relation 常数的 $h$ convention |
| conditional means | $\mu_r,\mu_{m_R},\mu_\gamma$ | 都是建立在前述推导量上的统计关系 | 可能；取决于输入量 convention 是否统一 |

---

## 2. 完整的 $h$ 依赖审计表

> 读法说明  
> - “$h$ 依赖”列写的是量纲级 rescaling。对 log 量，我同时在说明里指出相应的加法偏移。  
> - 本版已把 stellar-mass 指数锁定为 $-2$。若某行仍写“信息不足”，通常指缺少 fiducial $h_{\rm ref}$、外部校准常数的原始 convention、或代码表格生成细节，而不是 $M_\ast$ 的幂次未知。  
> - 对 fixed-kpc lensing mass，指数是 $2-\gamma$，所以不是一个全样本统一常数。

### 2.1 距离与尺度转换

| 名称 | 符号 | 出现位置 | 当前定义 | 当前单位/表示 | 来源链路 | $h$ 依赖 | 依赖来源 | 文档是否写清楚 | 混用/误导风险 | 下一步处理 | 信息充足性 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| deflector/source redshift | $z_d,z_s$ | §Data And Derived Quantities → Observed quantities，源文件行 23–49 | 直接进入 $D_i$ | 无量纲 | 直接观测 | $h^0$ | 红移本身不含 $H_0$ | 否（但问题不大） | 低；主要风险在后续转距离 | 保持为原始观测量；不要把它和距离混写 | 充足 |
| 角直径距离 | $D_d,D_s,D_{ds}$ | §Lensing geometry，行 96–123 | 通过 $\Sigma_c$ 隐式出现 | 文稿未写；应为长度 | $z_d,z_s \rightarrow D(z)$ | $h^{-1}$ | 固定其余宇宙学参数时，所有距离 $\propto H_0^{-1}$ | 否 | 中；文稿未写 fiducial cosmology | 在方法部分显式声明“以下所有距离均按 $h=h_{\rm ref}$ 计算” | 对纯 $h$ 缩放足够；对全 cosmology 不足 |
| 光度距离 | $D_L$ | **文稿未显式出现** | 无 | 无 | 用户补充确认：stellar mass 上游经过 luminosity / SED / SPS，因此此步存在于 $\log M_\ast^{\rm obs}$ 的上游 | $h^{-1}$ | 与角直径距离同理 | 文稿未写；本版由用户补充确认 | 高；这是 stellar-mass 链 $h^{-2}$ 的主要来源 | 在文稿中补出 $D_L\rightarrow L\rightarrow M_\ast$ 的 convention block | 指数充足；缺文稿内显式 $h_{\rm ref}$ |
| Einstein radius（角量） | $\theta_E$ | $D_i$ 行 27–38；§Lensing geometry 行 114–142；PPC 行 713–759 | 角 Einstein radius | arcsec（从 `theta_ein_arcsec` 可推断） | 直接观测 / raw attrs | $h^0$ | 角量本身不变 | 部分清楚（角量但未谈 $h$） | 低 | 在图表轴上保留角量有助于降低混淆 | 充足 |
| Einstein radius（物理半径） | $r_{\rm Ein}$ | §Lensing geometry，行 114–142 | $r_{\rm Ein}=D_d\theta_E/206265$ | 物理长度 | $z_d,\theta_E \rightarrow D_d \rightarrow r_{\rm Ein}$ | $h^{-1}$ | 来自 $D_d$ | 否 | 中；后续进入 lensing mass | 在代码/文稿显式写 $r_{\rm Ein}(h)=r_{\rm Ein,ref}(h/h_{\rm ref})^{-1}$ | 充足 |
| observed effective radius（物理） | $\log R_e^{\rm obs}$, $R_e$ | $D_i$ 行 27–38；$\Delta_R$ 与 $\Sigma_\ast$ 行 327–377 | 文稿中被当作已知 physical size；BOSS 分支明确写 $R_{e,\rm kpc}$ | 看语义应为 kpc 的对数 | 上游角尺度 × 距离；但文稿未展示 | $R_e \propto h^{-1}$；$\log R_e$ 加 $-\log_{10}(h/h_{\rm ref})$ | 角尺度转物理尺度 | **没有写清楚** | **高**；它一方面进入 $\Delta_R$，一方面进入 $\Sigma_\ast$ 和 dynamics table | 先核对 raw HDF5 存的是 arcsec 还是 kpc，以及转换用的 cosmology | **部分不足；缺上游半径定义** |
| 固定 aperture 半径 | $R=5,10\,{\rm kpc}$ | §Lensing geometry，行 84–94；Notation 表行 984–985 | fixed physical aperture | kpc | 模型定义，不是观测量 | 在当前文稿定义下视为 $h^0$（固定 physical kpc） | **正是“固定 physical kpc”这一选择，导致 $M_R$ 不是简单 $h^{-1}$** | 未写 | **高**；很多人会习惯性把它误读成“带 $h^{-1}$ 的长度” | 在文稿开头明确说明：这里的 $R$ 是 fixed physical kpc，不是 $h^{-1}$ kpc | 充足 |
| BOSS aperture geometry | aperture radius, seeing FWHM | §BOSS branch，行 60–80 | `aperture_radius_arcsec=1.0`, `seeing_fwhm_arcsec=1.5` | arcsec | upstream observation contract | $h^0$ | 角量本身不变 | 清楚其角单位，但没写 $h$ | 低到中；只在转 physical aperture 时有风险 | 保留 arcsec 口径，并在任何转 kpc 的地方单独标注 | 充足 |

### 2.2 光度、恒星质量与结构量

| 名称 | 符号 | 出现位置 | 当前定义 | 当前单位/表示 | 来源链路 | $h$ 依赖 | 依赖来源 | 文档是否写清楚 | 混用/误导风险 | 下一步处理 | 信息充足性 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| observed stellar-mass product | $\log M_\ast^{\rm obs}$ | $D_i$ 行 27–38；single-lens likelihood 行 487–515 | 进入 likelihood 的 observed input；用户补充其上游来自 photometry / luminosity / SED / SPS 链 | 对数质量；应明确写为 $\log_{10}[M_\ast(h_{\rm ref})/M_\odot]$ | flux/photometry $\rightarrow$ luminosity/absolute magnitude $\rightarrow$ SED/SPS 给出 $M_\ast/L$ $\rightarrow M_\ast$ | $h^{-2}$；log 形式平移 $-2\log_{10}(h/h_{\rm ref})$ | 来自 $D_L\propto h^{-1}$ 与 $L\propto D_L^2$；SED/SPS 的 $M_\ast/L$ 不再引入距离因子 | **文稿本身未写清楚；本版依据用户补充已确定指数** | **高**；不是指数未知，而是若图表不标 $h_{\rm ref}$ 会误导外部比较 | 在 raw attrs、方法小节、表头中写入 $M_\ast\propto h^{-2}$ 和 $h_{\rm ref}$ | 指数已充足；仍缺文稿内显式 $h_{\rm ref}$ 元数据 |
| latent stellar mass | $m_\ast\equiv \log_{10}(M_\ast/M_\odot)$ | §Stellar-mass function，行 292–311 | 群体模型的 latent mass | 对数太阳质量 | 由 $\log M_\ast^{\rm obs}$ 通过 measurement model 约束 | $h^{-2}$；$m_\ast(h)=m_{\ast,{\rm ref}}-2\log_{10}(h/h_{\rm ref})$ | 继承上游 stellar-mass convention | 否 | 高；它同时作为 x 轴、binning variable 和条件变量 | 所有使用 $m_\ast-11.4$ 的地方都要说明 11.4 是在哪个 $h_{\rm ref}$ convention 下的 pivot | 指数已充足；pivot convention 需补 |
| stellar-mass function anchor | $S(m_\ast)$, $(\mu_\ast,s_\ast,\alpha_\ast)$ | 行 300–311 | skew-normal stellar-mass function | $m_\ast$ 空间 PDF | 外部 profile-fixed 常数 + latent $m_\ast$ | 横轴随 $m_\ast$ 按 $-2\log_{10}(h/h_{\rm ref})$ 平移；$\mu_\ast$ 必须同样平移 | 外部 stellar-mass-function 标定的 mass convention | **没有写清楚** | **高**；外部固定先验若与当前 raw mass convention 不同，会整体偏移 | 审计外部 literature 常数原始 $h$ 约定，并统一到当前 $h_{\rm ref}$ | **部分不足；缺外部常数来源与单位说明** |
| mean size relation | $\mu_r(m_\ast,n)$ | 行 313–325 | $\mu_{r,0}+\beta_r(m_\ast-11.4)$（sersic 再加 $\nu_r[\log n-\log4]$） | $\log R_e$ | 由 fixed structural constants + $m_\ast,n$ 给出 | 非单一固定幂次；取决于 $\mu_{r,0},\beta_r,\nu_r$ 使用的 $h$ convention | 同时依赖 $m_\ast$ 与 $R_e$ 的 convention | **没有写清楚** | **极高**；$\Delta_R$ 是否真是“残差”取决于先验常数与输入是否同 convention | 追溯 size-relation 文献常数，并显式写明已否转换 | **不足；缺 structural constants 的原始 convention** |
| size residual | $\Delta_R$ | 行 327–333；$\mu_{m_R},\mu_\gamma$ 行 337–377 | $\log_{10}R_e-\mu_r(m_\ast,n)$ | 对数残差 | $R_e$ 与 size relation 的差 | 不存在单一固定幂次；若输入和 size relation 已统一 convention，则可视为近似 $h^0$ | 残差是否保 $h^0$ 取决于两端是否同 convention | **没有写清楚** | **极高**；`dependent` 模式直接用它驱动 $\mu_{m_R}$ 与 $\mu_\gamma$ | 先在代码和文稿中确认 $\Delta_R$ 是否在数值上对 $h$ 不变 | **不足；缺 external-calibration audit** |
| stellar surface density | $\Sigma_\ast$ | BOSS 行 72–80；sigma-star mode 行 371–377；Notation 行 992 | $\log\Sigma_\ast=\log M_\ast-\log_{10}(2\pi R_{e,\rm kpc}^2)$ | 物理 surface density 的对数 | $M_\ast$ 和 $R_e$ 组合 | $h^0$，前提是 $M_\ast\propto h^{-2}$ 且 $R_e\propto h^{-1}$ | $M_\ast$ 与 $R_e^2$ 的纯 $h$ 因子相消 | **公式清楚；文稿未点明其 $h$-cancellation** | 中；它是 `sigma_star_dependent` 模式中相对稳健的自变量，但仍需同一 $h_{\rm ref}$ | 明确写出“本项目中 $\Sigma_\ast$ 在纯 $h$ 缩放下为 $h^0$” | 指数充足；仍需确认 $R_e$ kpc 转换的 fiducial cosmology |
| gamma population mean（dependent） | $\mu_\gamma=\mu_{\gamma,0}+\beta_\gamma(m_\ast-11.4)+\xi_\gamma\Delta_R$ | 行 345–369；PPC 行 672–688 | $\gamma$ 的群体均值 | 无量纲 | 由 $m_\ast,\Delta_R$ 驱动 | $\gamma$ 本身是 $h^0$，但这套回归的自变量 convention-sensitive | 依赖 $m_\ast,\Delta_R$ 是否已统一 | **没有写清楚** | 高；即便 $\gamma$ 是无量纲，回归参数也会吸收 convention 差异 | 统一 $m_\ast,R_e,\Delta_R$ 后再解释 $\beta_\gamma,\xi_\gamma$ | **部分不足** |
| gamma population mean（sigma-star dependent） | $\mu_\gamma=\mu_{\gamma,0}+\beta_{\Sigma_\ast,\gamma}(\log\Sigma_\ast-9)$ | 行 355–377；PPC 行 674–688 | $\gamma$ 的群体均值 | 无量纲 | 由 $\Sigma_\ast$ 驱动 | 在本版确认的标准链路下，回归自变量 $\log\Sigma_\ast-9$ 对纯 $h$ 缩放为 $h^0$ | $M_\ast\propto h^{-2}$ 与 $R_e^2\propto h^{-2}$ 相消 | **文稿没有点明；本版应补入** | 中；这是该 mode 相比直接用 $m_\ast$ 更少受 $h$ convention 影响的优点 | 在 mode 描述处明确说明该 robust 性的条件：$M_\ast$ 与 $R_e$ 必须同 $h_{\rm ref}$ | 指数充足 |
| luminosity / absolute magnitude / distance modulus / $M_\ast/L$ | $L, M_{\rm abs}, DM, M_\ast/L$ | 文稿未显式出现；用户补充为 $\log M_\ast^{\rm obs}$ 的上游链 | photometry/flux 经 $D_L$、distance modulus、SED/SPS 转成 stellar mass | $L$、mag、dimensionless $M_\ast/L$ | flux $\rightarrow D_L\rightarrow L\rightarrow M_\ast/L\rightarrow M_\ast$ | $D_L\propto h^{-1}$, $L\propto h^{-2}$, $M_\ast/L\propto h^0$, $M_\ast\propto h^{-2}$；$M_{\rm abs}$ 随 $+5\log_{10}(h/h_{\rm ref})$ 平移 | 距离模数和 luminosity conversion | **文稿完全未写；本版依据用户补充确定** | **高**；这是 stellar-mass $h^{-2}$ 的源头 | 在方法报告中补一个“stellar mass upstream convention”小节 | 指数充足；仍缺具体 SPS/IMF/$h_{\rm ref}$ 元数据 |

### 2.3 透镜与总质量链

| 名称 | 符号 | 出现位置 | 当前定义 | 当前单位/表示 | 来源链路 | $h$ 依赖 | 依赖来源 | 文档是否写清楚 | 混用/误导风险 | 下一步处理 | 信息充足性 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| critical surface density | $\Sigma_c$ | §Lensing geometry，行 96–102 | $\frac{c^2}{4\pi G}\frac{D_s}{D_dD_{ds}}$ | 质量 / 面积 | $z_d,z_s\to D_d,D_s,D_{ds}\to \Sigma_c$ | $h^{+1}$ | 距离比 $D_s/(D_dD_{ds})\propto h$ | **没有写 $h$** | 中 | 在方法里直接给出 $\Sigma_c(h)$ 重标式 | 充足 |
| 固定物理半径内的投影质量 | $M_{2\rm D}(<R)$, $m_R$ | §Lensing geometry，行 84–142；Notation 行 984–985 | $m_R\equiv\log_{10}M_{2\rm D}(<R)$, $10^{m_R}=\pi\Sigma_c r_{\rm Ein}^{\gamma-1}R^{3-\gamma}$ | 对数质量；文稿未标明 $M_\odot$ 还是 $h$-scaled mass | $z_d,z_s,\theta_E,\gamma,R \to \Sigma_c,r_{\rm Ein} \to M_R$ | $\boxed{M_R\propto h^{2-\gamma}}$；等价地 $m_R$ 平移 $(2-\gamma)\log_{10}(h/h_{\rm ref})$ | $\Sigma_c\propto h$, $r_{\rm Ein}^{\gamma-1}\propto h^{-(\gamma-1)}$, 而 $R$ 是 fixed physical kpc | **没有写清楚** | **极高**；这不是统一常数偏移，不能用“所有质量都除以 $h$”处理 | 在文稿和代码里把这个式子单独写成一个 rescaling helper，并逐 lens / 逐 draw 应用 | **充足** |
| $m_5,m_{10}$ 两种 mass definition | $m_5,m_{10}$ | 行 90–94，237–241，943–950，984–985 | $m_{10}=m_5+(3-\gamma)\log_{10}2$ | 对数质量 | 同一 power-law 在不同 fixed-kpc aperture 下的表示 | 两者都按 $h^{2-\gamma}$ 缩放；它们的差 $(3-\gamma)\log_{10}2$ 为 $h^0$ | 两者共享同一个 fixed-kpc 定义 | **没有写清楚** | 高；结果表里的 `m10` 很容易被读成“普通质量对数” | 在所有表头写全：`$\log_{10}[M_{2D}(<10\,{\rm kpc})/M_\odot]$` 并注明 $h$ convention | 充足 |
| Einstein radius 内质量（隐含量） | $M_{\rm Ein}\equiv M_{2\rm D}(<r_{\rm Ein})$ | 文稿未命名；由行 106–123 的公式隐含 | 若令 $R=r_{\rm Ein}$，则 $M_{\rm Ein}=\pi\Sigma_c r_{\rm Ein}^2$ | 质量 | $\Sigma_c$ 与 $r_{\rm Ein}$ 组合 | $h^{-1}$ | $\Sigma_c\propto h$, $r_{\rm Ein}^2\propto h^{-2}$ | 否 | 中；容易和当前主质量定义 $m_{10}$ 混淆 | 在文稿里显式区分“Einstein mass”和“fixed-kpc mass” | **足以推导，但文稿未单独呈现** |
| 单镜头质量轨迹 | $m_{R,i}(\gamma)$ | 行 126–144；single-lens likelihood 行 497–501 | 由 $\theta_{E,i}$ 压缩得到的一维轨迹 | $\log$ mass as function of $\gamma$ | $z_d,z_s,\theta_E,\gamma,R\to m_R(\gamma)$ | $h^{2-\gamma}$；且指数随 $\gamma$ 变化 | 继承 fixed-kpc mass 的公式 | 否 | **极高**；single-lens flat-prior mass 点不能靠单一常数重标 | 若需要改 $h$，必须用整条 $m_R(\gamma)$ 轨迹，而不是只改中位数 | 充足 |
| Jacobian | $J_i(\gamma)=\lvert\frac{dm_{R,i}}{d\theta_{\rm Ein}}\rvert$ | 行 517–525；flat-prior 行 224–226 | 轨迹压缩的 Jacobian | 角度逆量 | 来自 $m_R(\gamma,\theta_E)$ 对 $\theta_E$ 的导数 | $h^0$ | $dm_R/d\theta_E=(\gamma-1)/(\ln10\,\theta_E)$，距离项在导数里消掉 | 文稿未点明，但可从公式推出 | 低 | 无需重点改动，但可在注释里写清“不带 $h$” | 充足 |
| unit-mass Jeans response | $S_{\rm unit}$, $s_2(\gamma)$ | 行 148–186；flat-prior 行 196–205；PPC 行 721–728 | $S_{\rm unit}\equiv \sigma^2/10^{m_R}$ | $({\rm km\,s^{-1}})^2/{\rm mass}$ | Jeans table + 质量定义 | **从定义上**应随 $M_R^{-1}$ 变化；若仅抽取 mass-definition 缩放，则近似 $h^{\gamma-2}$ | 为保持 $\sigma_{\rm model}$ 与观测速度量一致，`S_unit` 必须与 $m_R$ convention 配套 | **没有写清楚** | 高；若只改 `m10` 不改 sigma table，动力学会失配 | 把 dynamics grid 的 cosmology / physical-aperture convention 也纳入审计；必要时重建表格 | **部分不足；精确实现还依赖 grid 构造** |
| model velocity dispersion | $\sigma_{\rm model}$ | 行 156–165，529–536，721–738 | $\sqrt{S_{\rm unit}10^{m_R}}$ | km/s | $S_{\rm unit}$ 与 $m_R$ 组合 | 目标量是 $h^0$，但前提是 `S_unit` 与 $m_R$ 使用同一 convention | 速度量是可观测物理量；任何 $h$ 依赖都应在中间层相消 | 文稿未明确指出其 $h$-free | 中；若只部分重标某一侧会人为引入假 $h$ 依赖 | 任何改 $h$ 的操作都要一起检查 $m_R$ 与 sigma table | **部分不足；取决于 grid 是否一起重建** |
| observed/model dispersion likelihood term | $p_{\sigma,i}$ | 行 538–556 | 高斯 likelihood on $\sigma_{\rm ap}^{\rm obs}$ | km/s likelihood | $\sigma_{\rm ap}^{\rm obs},\sigma_{\rm model}$ | 观测量 $h^0$；模型端目标也是 $h^0$ | 动力学量本身不依赖 $h$，但其计算需要一致的 mass convention | 没有专门写 | 中；风险在上游 mass convention，不在 likelihood 形式本身 | 不必改 likelihood 形式，但要改输入 convention 说明 | 充足 |

### 2.4 密度、分数、比值与全局统计对象

| 名称 | 符号 | 出现位置 | 当前定义 | 当前单位/表示 | 来源链路 | $h$ 依赖 | 依赖来源 | 文档是否写清楚 | 混用/误导风险 | 下一步处理 | 信息充足性 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| stellar surface density | $\Sigma_\ast$ | 行 72–80，373–377，992–993 | $M_\ast/(2\pi R_e^2)$ 的对数 | 物理 surface density | $M_\ast,R_e$ 组合 | $h^0$，若 $M_\ast\propto h^{-2}$ 且 $R_e\propto h^{-1}$ | 两个 $h^{-2}$ 因子相除 | 公式清楚；文稿未写 $h$-cancellation | 中 | 把 $\Sigma_\ast$ 标为本项目优先输出的 $h$-robust summary | 指数充足；仍需确认 $R_e$ 元数据 |
| projected mass density（隐含） | $\Sigma(R)$ | **文稿未作为独立量给出**；由 power-law normalization 隐含 | 未显式定义 | mass/area | 由 $\Sigma_c,r_{\rm Ein},\gamma$ 的 power-law 归一化隐含给出 | 若在 fixed physical $R$ 上评估，归一化通常继承 $h^{2-\gamma}$ | 与 fixed-kpc mass 同源 | 否 | 中；若后续从 $m_R$ 反推 $\Sigma$，会继承同样问题 | 如后续要用 $\Sigma(R)$，必须写明是在角坐标还是物理坐标上定义 | **不足；缺显式定义** |
| convergence | $\kappa$ | **文稿未显式出现** | 无 | 无量纲 | 若将 $\Sigma$ 与 $\Sigma_c$ 相除得到 | 不能仅凭文稿唯一指定；在角坐标上常可视为 $h^0$，在 fixed-kpc 坐标上不应直接这么说 | 取决于径向坐标 convention | 否 | 中；最容易被“$\kappa$ 无量纲所以一定 $h$-free”误导 | 若要在后续文稿里引入 $\kappa$，必须同时写清坐标变量是 $\theta$ 还是 $R$ | **不足** |
| mass ratio / stellar fraction（隐含比较量） | $M_R/M_\ast$, $f_\ast=M_\ast/M_R$ | 文稿未显式定义；但 $\mu_{m_R}(m_\ast,\cdot)$ 与 Fig. 8-like $m_{10}$ vs $\log M_\ast$ 比较隐含依赖它们，行 337–377、763–796、927–950 | 无显式定义 | 质量比 / 分数 | 由 fixed-kpc lensing mass 与 stellar mass 组合 | $M_R/M_\ast \propto h^{4-\gamma}$；$f_\ast=M_\ast/M_R\propto h^{\gamma-4}$ | $M_R\propto h^{2-\gamma}$ 与 $M_\ast\propto h^{-2}$ 不同 | **没有写清楚** | **极高**；这是最可能“数值看起来合理、物理比较却不一致”的位置 | 在比较前先把两类质量写成同一 $h_{\rm ref}$ 下的显式表达；不要把 ratio 当作 $h^0$ | 指数充足；实际数值仍需 $\gamma$ posterior |
| gamma-mode regressor using $\Sigma_\ast$ | $\log\Sigma_\ast-9$ | 行 355–377，674–688 | `sigma_star_dependent` 的 $\gamma$ 回归自变量 | 对数 surface density 偏移量 | 由 $M_\ast$ 与 $R_e$ 构成 | 在标准上游链路下为 $h^0$ | $\Sigma_\ast$ 的纯 $h$ 抵消 | 文稿未点明这一点 | 中；这是该参数化的一个 convention 优势 | 把“$\Sigma_\ast$ 是纯 $h$-robust，但不是免除 fiducial cosmology 元数据”的条件写清楚 | 指数充足 |
| full posterior normalization | $Z_{\rm norm}(\eta)$ | 行 560–624 | 对 latent population 的 selection normalization | 无明确单位 | $x=(z_d,z_s,m_\ast,n,R_e,m_R,\gamma)\to \theta_E(x)\to P_{\rm find},g$ | 不是单一 $h^n$；会通过 $m_\ast,R_e,m_R,\theta_E(x)$ 整体继承 convention | 全局 forward model 建在 h-sensitive latent variables 上 | **没有写清楚** | **高**；不能只后处理图表而不管 normalization | 如果要改 $h$，应从 raw products / interpolation grids / inference 一路一致重建，而不是只改结果表 | **部分不足** |

### 2.5 图、表、caption、符号映射与结果呈现

| 名称 | 符号 | 出现位置 | 当前定义 | 当前单位/表示 | 来源链路 | $h$ 依赖 | 依赖来源 | 文档是否写清楚 | 混用/误导风险 | 下一步处理 | 信息充足性 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Fig. 8-like 质量轴/质量 bin | $\log M_\ast$（mass bin） | 文稿描述行 761–819；结果表行 927–950 | 以 stellar-mass bin $B$ 压缩曲线；例子写“$\log M_\ast\approx 11.3$” | 对数质量，但无 explicit $h$ 标注 | 来自 latent/observed stellar mass | x 轴整体平移 $-2\log_{10}(h/h_{\rm ref})$ | stellar-mass convention | **没有写清楚** | **极高**；x 轴按 $h^{-2}$ 平移，而 y 轴 `m10` 按 $h^{2-\gamma}$ 平移 | 在 figure/table 标题或 caption 明写 $M_\ast$ 的 $h^{-2}$ convention 与 $h_{\rm ref}$ | 指数充足；缺图件最终标签 |
| Fig. 8-like `m10` bands/points | $m_{10}$ | 行 792–816；结果表 943–950；flat-prior summary 行 229–243 | $y\in\{m_{10},\gamma,\sigma_{\rm ap}\}$ | `m10` 数值但无单位注释 | 来自 posterior predictive / flat-prior attrs | $h^{2-\gamma}$ | fixed physical aperture mass | **没有写清楚** | **极高**；不同 lens / draw 的 $\gamma$ 不同，不能统一平移 | 在图例或表头明确写 fixed-kpc 定义和 rescaling rule | 充足 |
| posterior summary parameter | `mu10_0` | 结果表行 869–898 | $\mu_{m_R}$ 在 reference point 的质量均值参数 | 数值表，没有单位注释 | 来自 $\mu_{m_R}(m_\ast,R_e,n)$ | 至少继承 `m10` 的 convention；严格说不是单一固定幂次，因为模型还依赖 $m_\ast,\Delta_R$ convention | $m_R$ 与回归自变量都可能改写 | **没有写清楚** | **高**；读者会把它当作可直接外比的“质量归一化” | 结果表中给全名和 convention；必要时附显式 $h$-rescaled 版本 | **部分不足** |
| PPC 的 $\theta$ / $\sigma$ 统计量 | `theta.median`, `theta.std`, `sigma.median`, `sigma.std` | PPC 行 912–925 | 对 replicated catalogs 的 summary statistics | $\theta$ 为 arcsec，$\sigma$ 为 km/s | 来自 selected population 的 observable-like catalog | 表面量纲上 $h^0$ | 角量和速度量自身不含 $h$ | 单位部分算清楚；$h$ 未特别说明 | 低到中；若只看这张表容易以为模型对 cosmology 完全不敏感 | 在正文补一句：observable-level PPC 不直接暴露 $h$，但其 latent generation 仍依赖 convention | 充足 |
| actual figure axes / captions | — | **上传文稿未包含真实图轴标签与 caption 文本**；只有文字描述和表格 | 无法直接审计 | 无 | 无 | 无法直接判定 | 缺真实图文件或 caption 文本 | 否 | 中到高；最终排版时最容易漏掉 convention | 在投稿前逐张检查 figure PNG/PDF 和 caption | **不足；缺图件本身** |
| 符号映射表中的 mass notation | $m_R,m_5,m_{10},\Sigma_\ast$ | Notation 表，行 982–1000 | 只给 meaning，不给 $h$ 约定 | 语义性标签 | 整体汇总 | 继承各自主链 | 同前 | **没有写清楚** | 中；notation table 本应是最佳声明位置，却没有写 convention | 把 $h$ scaling 或 fiducial cosmology 补进 notation/units 小节 | 充足到可修改 |

---

## 3. 重点问题深挖：stellar mass 与 lensing-constrained mass 的 $h$ 依赖差异

### 3.1 Stellar mass

#### 3.1.1 文中的 stellar mass 是如何得到的？

文稿内部只把 $\log M_\ast^{\rm obs}$ 作为单镜头输入 $D_i$，并把
$$
m_\ast\equiv \log_{10}(M_\ast/M_\odot)
$$
作为群体层 latent variable；它随后进入 $S(m_\ast)$、$\mu_r(m_\ast,n)$、$\mu_{m_R}(m_\ast,R_e,n)$、$\mu_\gamma(m_\ast,R_e,n)$ 以及 Fig. 8-like 的 stellar-mass binning。  

用户补充说明后，上游链路已经可以确定：$\log M_\ast^{\rm obs}$ 来自 photometry / luminosity / SED / SPS。也就是说，它不是原始观测量，而是由观测 flux/photometry、红移距离、absolute magnitude 或 luminosity、以及 SPS 模型给出的 $M_\ast/L$ 共同得到的 derived quantity。

#### 3.1.2 它是否经过 luminosity、absolute magnitude、distance modulus、SED fitting 等步骤？

是。按照用户补充的语境，应把 stellar mass 链写成
$$
f_\nu,\;m_{\rm app},\;z
\;\rightarrow\;
D_L(z;h),\;DM(z;h),\;M_{\rm abs}(h),\;L(h)
\;\rightarrow\;
(M_\ast/L)_{\rm SPS}
\;\rightarrow\;
M_\ast(h).
$$
这里 SED/SPS 的主要作用是根据颜色、SED 形状、IMF、SFH、metallicity、dust 等假设给出 $M_\ast/L$。纯 $h$ 依赖主要不来自 SPS 模型本身，而来自把观测 flux 转成 luminosity 的距离因子。

#### 3.1.3 $h$ 依赖具体来自哪一步？

在固定 $(\Omega_M,\Omega_\Lambda,\dots)$ 形状参数、只改变 $H_0$ 的情况下：
$$
D_L\propto h^{-1}.
$$
由 flux–luminosity relation，
$$
L=4\pi D_L^2 f
$$
可得
$$
L\propto h^{-2}.
$$
如果 SPS/SED fitting 给出的 $M_\ast/L$ 不再含额外距离因子，则
$$
M_\ast=(M_\ast/L)L\propto h^{-2}.
$$
等价地，absolute magnitude 会随 $h$ 改变为
$$
M_{\rm abs}(h)=M_{{\rm abs,ref}}+5\log_{10}\!\left(\frac{h}{h_{\rm ref}}\right),
$$
因此 luminosity 和 stellar mass 在 log 空间平移
$$
\log_{10}L(h)=\log_{10}L_{\rm ref}-2\log_{10}\!\left(\frac{h}{h_{\rm ref}}\right),
$$
$$
m_\ast(h)=m_{\ast,\rm ref}-2\log_{10}\!\left(\frac{h}{h_{\rm ref}}\right).
$$

#### 3.1.4 当前文稿是否已经把这个 $h$ 依赖表达清楚？

没有。  
文稿把 $\log M_\ast^{\rm obs}$ 放进 $D_i$，但没有在正文、notation table、结果表或 Fig. 8-like 描述里写出
$$
M_\ast\propto h^{-2}.
$$
因此，本版审计不再把 stellar mass 指数列为未知；但仍然认为文稿表达不充分，因为它没有声明：

1. $\log M_\ast^{\rm obs}$ 的 fiducial $h_{\rm ref}$；
2. stellar mass 是否已经写成 $M_\odot$、$h^{-2}M_\odot$、或“assuming $h=h_{\rm ref}$”；
3. 外部 stellar-mass function 与 size relation 常数是否已经换算到同一 convention；
4. Fig. 8-like mass bins 中 “$\log M_\ast\approx 11.3$” 的 $h$ 语义。

#### 3.1.5 若后续要统一 convention，stellar mass 最适合用什么形式表达？

建议保留两个层级的表达：

1. **数值工作版**  
   $$
   m_{\ast,\rm ref}\equiv
   \log_{10}\!\left[\frac{M_\ast(h_{\rm ref})}{M_\odot}\right].
   $$
   这用于代码、表格和图轴的实际数值。

2. **显式重标版**  
   $$
   m_\ast(h)=m_{\ast,\rm ref}-2\log_{10}\!\left(\frac{h}{h_{\rm ref}}\right).
   $$
   这用于跨文献比较、改变 fiducial cosmology、以及解释 mass ratio / stellar fraction。

对于本项目，最推荐的写法是：
$$
\log_{10}\!\left[M_\ast(h_{\rm ref})/M_\odot\right],
\qquad M_\ast\propto h^{-2}.
$$
不要只在单位里写一个模糊的 $M_\odot$，也不要只写 $h^{-2}M_\odot$ 而不说明当前数值是否已经代入了 $h_{\rm ref}$。

#### 3.1.6 与 $\Sigma_\ast$ 的直接后果

文稿定义
$$
\log\Sigma_\ast
=
\log M_\ast-\log_{10}(2\pi R_{e,\rm kpc}^2).
$$
若 $R_e$ 是由角半径经 $D_A$ 转成 kpc，则 $R_e\propto h^{-1}$。结合 $M_\ast\propto h^{-2}$，得到
$$
\Sigma_\ast\propto \frac{h^{-2}}{(h^{-1})^2}=h^0.
$$
因此，在本项目的标准链路下，$\log\Sigma_\ast-9$ 是一个纯 $h$ 缩放下不变的回归自变量。这使 `sigma_star_dependent` 模式在 $h$-convention 上比直接使用 $m_\ast$ 更稳健。  
但这不是自动成立的“魔法”：必须保证 $M_\ast$ 与 $R_e$ 使用同一个 fiducial distance scale，且 $\Sigma_\ast$ 的数值由二者同步生成。


### 3.2 Lensing-constrained mass

#### 3.2.1 文中的透镜质量是通过哪些观测约束或模型得到的？

文稿里主质量量不是 halo mass，也不是一个自由的二维 $(m_R,\gamma)$ 面，而是由观测到的 $\theta_E$ 先把 $(m_R,\gamma)$ 压成 $m_{R,i}(\gamma)$ 的一维轨迹；再由动力学项沿这条轨迹用 $\sigma_{\rm ap}^{\rm obs}$ 重新加权。  

因此，lensing-constrained mass 的链路是：
$$
(z_d,z_s,\theta_E,\gamma,R)\rightarrow D_d,D_s,D_{ds}\rightarrow \Sigma_c,r_{\rm Ein}\rightarrow M_{2\rm D}(<R),
$$
并在有动力学数据时通过
$$
M_{2\rm D}(<R),\gamma \rightarrow S_{\rm unit} \rightarrow \sigma_{\rm model}
$$
再进一步受 $\sigma_{\rm ap}^{\rm obs}$ 约束。

#### 3.2.2 推导链路里是否隐含角尺度到物理尺度的转换？

是，而且这是 $h$ 依赖的核心来源之一。  
文稿显式写出
$$
r_{\rm Ein}=D_d\theta_E/206265,
$$
因此从 $\theta_E$ 到物理半径的转换是直接写在主链里的。  

另外，主质量定义用的是 $R=5,10\,{\rm kpc}$ 这种 fixed physical aperture，而不是 fixed angular aperture；这使得 $h$ 依赖不再是一个简单常数。

#### 3.2.3 是否通过 Einstein radius、projected radius、critical surface density 或类似量进入质量表达？

是。文稿显式写出：
$$
10^{m_R}=\pi \Sigma_c\, r_{\rm Ein}^{\gamma-1} R^{3-\gamma},
\qquad
\Sigma_c=\frac{c^2}{4\pi G}\frac{D_s}{D_dD_{ds}}.
$$
所以 $\Sigma_c$、$r_{\rm Ein}$ 和固定物理半径 $R$ 三者都直接进入质量定义。  
这使得当前的 lensing-constrained mass 不是“只靠 $\Sigma_c$”也不是“只靠角 Einstein radius”，而是一个同时混合了距离、角度和固定物理尺度的量。

#### 3.2.4 $h$ 依赖具体来自哪一步？

从文稿公式可以直接推出：

- $D_d,D_s,D_{ds}\propto h^{-1}$
- $\Sigma_c\propto h^{+1}$
- $r_{\rm Ein}\propto h^{-1}$
- $R=5,10\,{\rm kpc}$ 在当前定义下是 fixed physical length

因此
$$
M_{2\rm D}(<R_{\rm fixed\ kpc})
\propto
\Sigma_c\,r_{\rm Ein}^{\gamma-1}R^{3-\gamma}
\propto
h^{+1}\,h^{-(\gamma-1)}\,h^0
=
h^{2-\gamma}.
$$

这一步是**可以仅凭文稿内部严格推出**的。  
也正因为如此，fixed-kpc lensing mass 与常见的 $M(<r_{\rm Ein})\propto h^{-1}$ 不是同一件事。

#### 3.2.5 它与 stellar mass 的 $h$ 依赖是否相同？

不相同。现在两条链路的指数都可以明确写出：

$$
M_\ast \propto h^{-2},
\qquad
M_R \equiv M_{2\rm D}(<R_{\rm fixed\ kpc})\propto h^{2-\gamma}.
$$

只有在非常特殊且非本项目设定的情况下，即 $\gamma=4$，二者才会有相同的 $h^{-2}$ 指数；而文稿中的 $\gamma$ 物理区间是约 $1.2$–$2.8$，实际强透镜样本也通常接近 $\gamma\simeq 2$。因此在本项目相关范围内，二者的 $h$ 依赖不同是确定结论。  

这还意味着，若构造比值：
$$
\frac{M_R}{M_\ast}\propto h^{4-\gamma},
\qquad
\frac{M_\ast}{M_R}\propto h^{\gamma-4}.
$$
对近似 isothermal 的 $\gamma\simeq2$，$M_R/M_\ast$ 近似按 $h^2$ 缩放，而 stellar fraction $M_\ast/M_R$ 近似按 $h^{-2}$ 缩放。

### 3.3 两者比较：当前文稿里能否直接比？

#### 3.3.1 文中 stellar mass 与 lensing mass 当前是否可以直接比较？

在同一个 pipeline、同一个隐藏的 fiducial cosmology 下，代码当然可以同时使用二者；但从审计和对外呈现的角度，当前文稿还不能被视作“已经提供了无歧义、可复现、可外比的质量比较”。  

原因不是 stellar mass 指数未知，而是文稿没有显式声明：
$$
M_\ast\propto h^{-2},
\qquad
M_R\propto h^{2-\gamma},
\qquad
h_{\rm ref}=\ ? .
$$
只要这三项没有出现在方法、表头或 caption 中，读者就无法判断 `m10`、$\log M_\ast$、$\Sigma_\ast$、mass bin、mass ratio 是否在同一 convention 下解释。

#### 3.3.2 若不能，问题出在哪里？

主要有四个层级的问题：

1. **stellar mass 的指数已确定，但文稿没有写。**  
   $\log M_\ast^{\rm obs}$ 应明确标注为 luminosity-based/SPS mass，满足 $M_\ast\propto h^{-2}$。

2. **lensing mass 是 $\gamma$-dependent scaling。**  
   fixed-kpc $M_R$ 按 $h^{2-\gamma}$ 缩放，因此同一个简单常数偏移不足以统一全样本。

3. **比较链上还混入了 size relation 与 $\Sigma_\ast$。**  
   $\Sigma_\ast$ 在标准链路下是 $h^0$，但 $\Delta_R$、$\mu_r$、外部 stellar-mass function 常数仍要求与同一个 $h_{\rm ref}$ 对齐。

4. **结果呈现没有显式写 convention。**  
   `mu10_0`、Fig. 8-like 质量轴、flat-prior mass points、notation table 都没有写明 $h$ 或 fiducial cosmology。

#### 3.3.3 哪些结论、图、表或叙述最可能因此产生歧义？

最危险的有五类：

1. **$\mu_{m_R}(m_\ast,R_e,n)$ 的物理解读。**  
   横轴 stellar mass 按 $h^{-2}$ 平移，纵轴 fixed-kpc lensing mass 按 $h^{2-\gamma}$ 平移；二者不是同一个 $h$ 方向的平移。

2. **Fig. 8-like $m_{10}$ vs $\log M_\ast$ 的带和点。**  
   x 轴整体按 $-2\log_{10}(h/h_{\rm ref})$ 平移，y 轴按 $(2-\gamma)\log_{10}(h/h_{\rm ref})$ 平移，且 y 轴平移还依赖 posterior draw 的 $\gamma$。

3. **posterior summary 里的 `mu10_0`。**  
   它很容易被读成一个可直接拿去跟别的文献比的“质量归一化”，但 fixed-kpc mass 的 convention 与外部 $h^{-1}M_\odot$ halo mass convention 并不相同。

4. **任何 stellar fraction / dark-matter fraction 叙述。**  
   $$
   f_\ast=M_\ast/M_R\propto h^{\gamma-4}
   $$
   明显不是 $h^0$。

5. **对 gamma-mode 的物理解读。**  
   `sigma_star_dependent` 的一个优点是 $\Sigma_\ast$ 在当前标准链路下为 $h^0$。如果文稿不说明这一点，读者可能无法区分“物理自变量更合适”与“convention 更稳健”这两层含义。

#### 3.3.4 若我要继续推进这个项目，比较这两类质量前最该先统一什么？

最先要统一的是**整个项目的 mass/size convention 元数据**：

1. $\log M_\ast^{\rm obs}$ 的 fiducial $h_{\rm ref}$ 与固定指数 $-2$；  
2. $\log R_e^{\rm obs}$ 是否是 kpc，以及它用哪个 cosmology 转出来；  
3. $m_5/m_{10}$ 的显式 rescaling 公式  
   $$
   m_R(h)=m_{R,\rm ref} + (2-\gamma)\log_{10}(h/h_{\rm ref});
   $$
4. $\Sigma_\ast$ 的 derived convention：  
   $$
   \Sigma_\ast(h)=\Sigma_{\ast,\rm ref}
   $$
   仅在 $M_\ast$ 与 $R_e$ 同步使用同一 $h_{\rm ref}$ 时成立；
5. 外部 structural constants（stellar-mass function、size relation）是否已经转换到同一 convention。  

只有这五点统一了，后面的 mass–mass、mass–size、$\Sigma_\ast$-based 回归解释才稳。

### 3.4 一个容易被忽略但很关键的点：`m10` 不能靠“全体统一平移”修正

因为
$$
m_R(h)=m_{R,\rm ref} + (2-\gamma)\log_{10}(h/h_{\rm ref}),
$$
这里的平移量取决于 $\gamma$。  

这带来两个直接后果：

1. **per-lens `m10`** 的 $h$ 修正需要知道该 lens 的 $\gamma$ posterior，而不是只拿一个中位 `m10` 数值做整体平移。  
2. **群体层 `mu10_0` / trend band** 的 $h$ 修正也不是一个全体常数，因为不同 posterior draw 的 $\gamma$ 分布不同。  

与之相对，stellar mass 的重标是全局常数平移：
$$
m_\ast(h)=m_{\ast,\rm ref}-2\log_{10}(h/h_{\rm ref}).
$$
所以，对于这份文稿里的质量比较，真正的问题不是“两个质量都不能重标”，而是**一个可以整体重标，另一个必须带着 $\gamma$ 逐点重标**。


## 4. 下一步建议清单

### 4.1 立即要检查的地方

#### A. 哪几个公式的单位链路必须重新核对？

1. **主质量定义式**
   $$
   10^{m_R}=\pi \Sigma_c r_{\rm Ein}^{\gamma-1}R^{3-\gamma}.
   $$
   这是整份文稿里最关键的 $h$ 来源，必须明确写出 $R$ 是 fixed physical kpc，并补上 $M_R\propto h^{2-\gamma}$。

2. **$\Sigma_\ast$ 定义式**
   $$
   \log\Sigma_\ast=\log M_\ast-\log_{10}(2\pi R_{e,\rm kpc}^2).
   $$
   这条公式本身对；在本版确认的 $M_\ast\propto h^{-2}$、$R_e\propto h^{-1}$ 链路下它是纯 $h$-free，但文稿必须显式写出这个抵消。

3. **$\Delta_R$ 的定义**
   $$
   \Delta_R=\log_{10}R_e-\mu_r(m_\ast,n).
   $$
   这里必须核对 size relation 固定常数的原始 $h$ convention；否则 $\Delta_R$ 可能不是“纯残差”。

4. **$\sigma_{\rm model}=\sqrt{S_{\rm unit}10^{m_R}}$**  
   这条式子提醒你：只修质量表头、不重建 dynamics grid，是不够的。

#### B. 哪几个图轴或表头必须确认？

1. 所有出现 `m10`, `m5`, `mu10_0`, `beta10`, `sigma10` 的表头；  
2. Fig. 8-like 的 x 轴 `$\log M_\ast$` 与 y 轴 `m10`；  
3. 任何 future draft 里若出现 `stellar fraction`, `mass ratio`, `dark matter fraction` 的图表。  

这些地方都必须写明是“在 $h_{\rm ref}$ 下的数值”还是“显式保留 $h$ scaling”。

#### C. 哪几个 derived quantity 最容易把 $h$ 依赖写错？

1. `m10`：因为很多人会下意识写成 $h^{-1}$，但这里其实是 $h^{2-\gamma}$；  
2. `\Sigma_\ast`：因为很多人会下意识写成 $h^0$，但那其实隐含了 $M_\ast\propto h^{-2}$；  
3. `\Delta_R`：因为它看起来像残差，但前提是 size relation 常数和输入 $R_e$ 已在同一 convention；  
4. `mu10_0`：因为它像是一个简单“质量零点”，其实它嵌在一个有 convention 风险的条件回归里。

### 4.2 建议建立的统一 convention

#### A. 是否应该对所有量显式写出 $h$ scaling？

我建议分层处理，而不是“所有量都写”或“所有量都不写”：

- **直接观测量**（$z,\theta_E,\sigma_{\rm ap}$）通常不必强行写 $h$ scaling；  
- **所有 physical length / mass / surface-density / luminosity 派生量** 应显式写；  
- **任何作为横轴、条件变量、表头或比较对象的 derived quantity** 必须写。  

也就是说，不是“所有量都显式写”，而是**所有进入物理解读的 derived quantity 都显式写**。

#### B. 是否只对质量和长度写出 $h$ scaling？

至少要覆盖以下四类：

1. physical lengths：$R_e, r_{\rm Ein}, R_{\rm ap}, b$；  
2. masses：$M_\ast, M_{2D}(<R), M_{\rm Ein}$；  
3. densities：$\Sigma_c, \Sigma_\ast$，以及今后若引入的 $\Sigma(R)$；  
4. mass or density ratios：$M_R/M_\ast$, $f_\ast$, dark-matter fraction。  

如果只写“质量和长度”，会漏掉 $\Sigma_\ast$ 这种你这份项目里非常关键的中间量。

#### C. 是否对某些量保留角量或观测量，而不急于转成物理量？

是，值得这样做。  
尤其是：

- $\theta_E$ 优先保持 arcsec；  
- 观测 aperture geometry 优先保持 arcsec；  
- 若某个比较只需要 geometry，不必过早转成 kpc。  

这样做能减少不必要的 $h$ 传播。  
但对于本项目当前核心质量定义 $m_{10}$，你已经选择了 fixed physical kpc，所以这部分不能靠“全部留在角量”回避。

#### D. 图、表、正文里采用什么统一写法最不容易混淆？

我建议采用“双层写法”：

1. **正文首次出现时给出显式定义 + scaling**  
   例如  
   $$
   m_{10,\rm ref}\equiv \log_{10}[M_{2D}(<10\,{\rm kpc};h_{\rm ref})/M_\odot],
   \qquad
   m_{10}(h)=m_{10,\rm ref}+(2-\gamma)\log_{10}(h/h_{\rm ref}).
   $$

2. **图表轴/表头给出 reference convention**  
   例如  
   $\log_{10}[M_{2D}(<10\,{\rm kpc})/M_\odot]$ at $h=0.7$`  
   或  
   $\log_{10}(M_\ast/M_\odot)$ assuming $h=0.7$`。

这样最不容易混淆，也最利于后续跨文献比较。

### 4.3 对分析流程的建议

#### A. 哪些中间变量应该重新命名或重新定义？

1. `m10` / `m5` 最好在注释和表头里扩成  
   `logM2D_lt_10kpc_ref` / `logM2D_lt_5kpc_ref` 这一类更难误读的名字；  
2. `logMstar` 最好再加一个元数据字段标明其 $h$ convention，例如 `stellar_mass_h_exp`;  
3. `Re` 最好同时保留 `Re_arcsec` 与 `Re_kpc_ref` 两个版本。  

这些改动不是为了美观，而是为了让 downstream 代码和表格生成阶段不再“默认你自己记得 convention”。

#### B. 哪些结果最好额外输出一个“显式含 $h$”版本？

至少建议额外输出：

1. per-lens 或 per-draw 的 $\gamma$；  
2. per-lens 或 per-draw 的 $m_{10,\rm ref}$；  
3. 一个函数式元数据  
   $$
   m_{10}(h)=m_{10,\rm ref}+(2-\gamma)\log_{10}(h/h_{\rm ref});
   $$
4. stellar mass 的对应式  
   $$
   m_\ast(h)=m_{\ast,\rm ref}-2\log_{10}(h/h_{\rm ref});
   $$
5. 由于上游质量已确认是 luminosity-based/SED/SPS 链，直接输出  
   $$
   \log\Sigma_\ast(h)=\log\Sigma_{\ast,\rm ref}
   $$
   作为项目里的一个 $h$-robust summary。  

这些量最好在 notebook、CSV / JSON summary、以及表格生成脚本里一起出，而不是只在论文里口头说明。

#### C. 哪些地方应该在代码、notebook、表格生成阶段就统一 convention？

1. raw HDF5 写入层：把 $H_0$、$h$、physical-vs-angular unit、mass exponent 元数据写进 attrs；  
2. interpolation grids：注明 sigma tables 是按哪个 physical aperture / size convention 建的；  
3. inference config snapshot：显式记录 fiducial cosmology；  
4. posterior-summary 导出：所有 mass / size / density 列自动带上 convention 注记；  
5. figure builder：轴标签从元数据自动生成，而不是手写字符串。  

把 convention 固定在最上游，比事后在论文里“补一句说明”可靠得多。

#### D. 哪些比较在 convention 未统一前不应直接解释？

在下列比较统一前，不建议给出物理解读：

1. $m_{10}$ 与 $\log M_\ast$ 的斜率、归一化和散度；  
2. 任何 stellar fraction / dark-matter fraction 结论；  
3. `dependent` vs `sigma_star_dependent` 的“谁更物理”解释；  
4. 与外部文献的 mass normalization 直接比较。  

这些地方最容易出现“数值看起来没问题，但实际在比不同 convention 下的量”。

### 4.4 风险提示

#### A. 如果现在忽略这些 $h$ 依赖，最可能出错的结论是什么？

1. **你可能会把 fixed-kpc lensing mass 误当成简单 $h^{-1}$ 量。**  
   这会直接错写 `m10` 的重标方式。

2. **你可能会把 $M_\ast$ 与 $M_R$ 的比值解释成真实的 stellar fraction。**  
   如果二者 $h$ 指数不同，这个分数就带着隐藏的 convention 偏差。

3. **你可能会误解 gamma-mode 的优劣。**  
   比如 `sigma_star_dependent` 看起来更稳定，可能 partly 因为 $\Sigma_\ast$ 更接近 $h$-robust，而不是因为它纯粹“更物理”。

4. **你可能会在外部比较时得到“数值差一个看似不大的常数”，却不知道那其实是 convention 差。**

#### B. 哪些地方最容易出现“数值看起来合理，但物理比较并不一致”的问题？

1. `mu10_0` 这种 summary parameter；  
2. Fig. 8-like 的质量–质量趋势；  
3. flat-prior mass points 与 hierarchical bands 的对比；  
4. $\Delta_R$ 参与的 conditional law；  
5. 任何后处理生成的 table / CSV，如果列名只写 `m10`、`logMstar` 而不写 convention。  

最危险的一点是：**这些结果很可能在你当前单一 pipeline 内部“看起来完全平滑且合理”，但一旦换一个 $h$ convention、或拿去和外部工作对比，歧义就会暴露出来。**

### 4.5 按优先级整理的行动清单

#### 必须马上做

1. **把 $\log M_\ast^{\rm obs}$ 的 $M_\ast\propto h^{-2}$ 与 fiducial $h_{\rm ref}$ 写进元数据。**  
   原因：指数已经由上游 photometry/luminosity/SED/SPS 链确定；现在最危险的是文稿和输出文件没有显式记录。

2. **确认 $\log R_e^{\rm obs}$ / `R_{e,\rm kpc}` 的单位链，并写明 $R_e\propto h^{-1}$。**  
   原因：$R_e$ 同时进入 $\Delta_R$、$\Sigma_\ast$、dynamics grid，是多条链路的共同支点。

3. **把 $m_R \propto h^{2-\gamma}$ 明写进项目记录。**  
   原因：这是当前最容易被误写、且会直接影响 lensing-mass 比较的核心结论。

4. **检查外部 structure priors 的 $h$ convention，尤其是 stellar-mass function 与 size relation 的 pivot。**  
   原因：即便 raw 数据一致，固定的 $\mu_r$ / $S(m_\ast)$ 常数若没转换，也会把模型整体推歪。

#### 建议尽快做

1. **在 raw / summary 输出里同时保存 reference-value 和显式 rescaling 公式。**  
   原因：这能把“你脑子里记得的 convention”变成机器可追踪的元数据。

2. **让 figure/table 标签自动带 convention。**  
   原因：最终最容易出错的往往不是推导，而是呈现。

3. **把 per-lens/per-draw $\gamma$ 与 `m10` 一起导出。**  
   原因：fixed-kpc lensing mass 的 $h$ 修正依赖 $\gamma$，没有它就不能精确重标。

4. **单独做一个 notebook 验证 $\Sigma_\ast$ 是否在当前 pipeline 里真是 $h$-free。**  
   原因：这会直接影响你对 `sigma_star_dependent` 的解释。

#### 后续可优化

1. **考虑是否增加一个更易于外比的质量定义。**  
   例如额外输出 $M(<r_{\rm Ein})$ 或 $M(<10\,h^{-1}{\rm kpc})$ 作为辅助量。  
   原因：它们在 cross-paper comparison 上可能更直观，但不应替代当前主模型定义。

2. **给代码加自动化 unit / convention tests。**  
   原因：一旦 raw products 或 figure scripts 改动，自动测试比人工记忆可靠。

3. **在最终论文里加一个短小但明确的 “cosmology and $h$-scaling conventions” 小节。**  
   原因：这能把当前这份内部审计，转化成对读者也可见的可复现说明。

---

## 5. 不确定项与信息缺口

本版已经把 stellar mass 的纯 $h$ 指数从待定项改为确定值：
$$
M_\ast\propto h^{-2}.
$$
因此，下面的不确定项不再包括“stellar mass 是否为 $h^{-2}$”本身，而主要集中在 fiducial convention、外部校准和代码元数据上。

1. **$\log M_\ast^{\rm obs}$ 的 fiducial $h_{\rm ref}$ 与写法**  
   缺失内容：文稿没有说明 raw HDF5 中的 $\log M_\ast^{\rm obs}$ 是 `assuming h=0.7`、`h^{-2}M_\odot`，还是已经转换到某个 $H_0$。  
   影响：指数已知，但外部比较和图轴解释仍会含糊。

2. **SED/SPS 细节不是 $h$ 指数缺口，但仍是质量系统误差缺口**  
   缺失内容：IMF、SPS model、dust law、SFH library、aperture/total-mass correction。  
   影响：这些通常不改变纯 $h^{-2}$ 缩放，但会改变 $M_\ast$ 零点和系统误差；不应与 $h$ convention 混为一谈。

3. **$\log R_e^{\rm obs}$ 的原始单位与转换过程**  
   缺失内容：raw 文件里到底存 arcsec 还是 kpc；若存 kpc，用的是哪个 cosmology。  
   影响：$\Sigma_\ast$ 的 $h^0$ 抵消、$\Delta_R$、dynamics grid 都依赖这一步与 $M_\ast$ 使用同一个 distance convention。

4. **外部 structural constants 的原始 convention**  
   缺失内容：stellar-mass function 与 size relation 文献常数是否已经换算到当前 $M_\ast\propto h^{-2}$、$R_e\propto h^{-1}$ convention。  
   影响：$\mu_r$、$\Delta_R$、$S(m_\ast)$ 可能隐含整体偏移。

5. **dynamics sigma table 的 cosmology / physical-aperture 元数据**  
   缺失内容：`S_unit` 表格生成时是否使用了随 $h$ 变化的 physical apertures / sizes，以及这些表是否会在改 cosmology 时重建。  
   影响：无法把 `S_unit` 的精确 $h$ 依赖写成完全封闭的单一幂次；但 $\sigma_{\rm model}$ 作为速度预测目标应为 $h^0$。

6. **真实 figure 轴标签与 caption 文本**  
   缺失内容：上传文稿没有包含最终图件本身。  
   影响：我能审计 figure 的文字描述和 summary table，但不能替你确认最终排版图里是否已经写了 convention。

7. **raw HDF5 attrs 是否已经写入 fiducial cosmology 元数据**  
   缺失内容：文稿没有展示相关 attrs。  
   影响：无法判断当前 pipeline 的自洽性是“显式记录”还是“默会约定”。

---

## 6. 其他工作如何处理 $h$-independent / $h$-explicit measurements

### 6.1 调研结论概览

调研后可以把外部实践分成四类。它们的共同点不是“让所有物理量都没有 $h$”，而是**把 $h$ 的位置显式化**，避免读者把不同 measurement channel 的质量当成同一 convention。

| 外部做法 | 典型领域 | 核心策略 | 对本项目的启发 |
|---|---|---|---|
| 显式 $h$-scaled units | galaxy evolution、weak lensing、SHMR | stellar mass 写 $h^{-2}M_\odot$，halo/lensing/dynamical mass 常写 $h^{-1}M_\odot$，距离写 $h^{-1}{\rm Mpc}$ | 你的 $M_\ast$ 应写 $h^{-2}$，但 $m_{10}$ 不能机械写 $h^{-1}$，因为它是 fixed-kpc power-law mass |
| 固定 fiducial cosmology | 观测 catalog、SED mass catalog、强/弱透镜分析 | 直接采用 $h_{\rm ref}$ 或 $H_0$，所有 derived quantities 都在该 cosmology 下给数值 | 可用于代码内部，但必须在 attrs / tables / captions 明确记录 |
| 使用 $h$-robust derived variables | surface density、colors、mass-to-light ratio、部分 dimensionless ratios | 构造时让距离因子抵消，例如 $\Sigma_\ast=M_\ast/(2\pi R_e^2)$ 在本项目中为 $h^0$ | `sigma_star_dependent` 的 $\log\Sigma_\ast$ 是较好的 convention-stable 自变量 |
| 近年 LSS 的真正 $h$-independent 参数化 | full-shape clustering、cosmological inference | 不再使用 $h^{-1}{\rm Mpc}$ 尽量隐藏 $h$，而改用 physical densities $\omega_i=\Omega_i h^2$、固定 Mpc scale 的 $\sigma_{12}$ 等 | 对本项目的类比是：不要把 $h$ 藏进“单位”，而应输出 $h_{\rm ref}$+重标公式 |

### 6.2 距离、luminosity 与 stellar mass：外部通常如何处理？

标准 cosmological distance 文献把 $H_0=100h\,{\rm km\,s^{-1}\,Mpc^{-1}}$ 作为距离尺度的来源；角直径距离用于把角尺度转成 physical transverse size，luminosity distance 由 flux–luminosity relation 定义。因此，只要固定其他 cosmological shape parameters，$D_A$ 与 $D_L$ 都带 $h^{-1}$ 的整体尺度。  

实际观测 catalog 中，stellar mass 往往不是直接观测量，而是 photometry/SED/SPS 产物。例如 SDSS MPA-JHU 的 galaxy properties 页面说明其 stellar masses 使用 Bayesian 方法和模型网格，并基于 ugriz photometry 输出 fiber 和 total stellar mass；KiDS+GAMA 的 SHMR 工作也说明 GAMA stellar masses 来自 Taylor et al. 的 SPS fits to SDSS ugriz photometry，并有 aperture/fluxscale correction。  

这些实践与本项目的更新结论一致：当 $M_\ast/L$ 来自 SED/SPS，而 luminosity 来自 $D_L^2$ 时，stellar mass 的纯 $h$ 缩放就是
$$
M_\ast\propto h^{-2}.
$$

### 6.3 Weak-lensing / SHMR 文献：显式把 stellar mass 与 halo mass 放在不同 $h$ 单位

KiDS+GAMA 的 stellar-to-halo mass relation 是最贴近本项目的外部例子之一。该工作把 galaxy stellar mass bins 写成 $\log_{10}(M_\ast/h^{-2}M_\odot)$，而 halo mass 使用 $h^{-1}M_\odot$ 一类 convention；文中还明确指出他们处理了不同 catalog 之间的 $h$ 差异。更重要的是，他们在比较 $M_h/M_\ast$ 时把 ratio 的 $h$ dependence 也显式写出来，例如用 $[h]$ 标注 dark-matter-to-stellar-mass ratio。  

这对本项目的直接启发是：**即使两个量都叫 mass，也不应默认同一个 $h$ exponent。** 外部 SHMR 工作通常已经承认 $M_\ast$ 与 $M_h$ 的 $h$ convention 不同；你的 fixed-kpc lensing mass 又比常规 halo mass 更特殊，因为它不是简单 $h^{-1}$，而是 $h^{2-\gamma}$。

### 6.4 Croton / Baldry 式“little $h$”实用规则：不要把 $h$ 当单位

Croton 的 little-$h$ 文章强调，同一个“mass”如果来自不同 measurement method，可以有不同的 $h$ 依赖：由 luminosity scaled 得到的 mass 常是 $h^{-2}$，由 dynamics 这类 velocity$^2$$\times$distance 得到的 mass 常是 $h^{-1}$。Baldry 的 practical guide 也把这几类常见缩放列成表：distance $\sim h^{-1}$、luminosity $\sim h^{-2}$、scaled-luminosity mass $\sim h^{-2}$、velocity-squared-times-distance mass $\sim h^{-1}$。  

对本项目来说，这条规则要稍微再推进一步：你的 lensing-constrained $m_{10}$ 既不是纯 luminosity mass，也不是简单 velocity$^2$$\times$distance mass，而是 fixed physical aperture 下的 power-law projected mass。因此它应按文稿自己的公式处理为 $h^{2-\gamma}$。

### 6.5 LSS / full-shape clustering 的较新做法：从 $h^{-1}{\rm Mpc}$ 转向真正物理参数

Sánchez 对 $h^{-1}{\rm Mpc}$ units 的批评提供了另一个方向：传统上用 $h^{-1}{\rm Mpc}$ 似乎能“去掉 $h$”，但实际上可能遮蔽 $h$ 对 power spectrum amplitude 和 reference scale 的影响。该文建议用 fixed Mpc scale 的 $\sigma_{12}$ 替代传统 $\sigma_8$，并在后续 full-shape clustering 工作中使用 physical densities $\omega_i=\Omega_i h^2$ 这类 $h$-independent 参数。  

这对本项目的启发不是要照搬 $\sigma_{12}$，而是原则相同：**不要用单位符号把 convention 问题藏起来。** 对你的项目，更干净的写法是：

$$
\text{reported value at }h_{\rm ref}
\quad + \quad
\text{explicit rescaling law}.
$$

具体到核心量：

$$
m_\ast(h)=m_{\ast,\rm ref}-2\log_{10}(h/h_{\rm ref}),
$$
$$
R_e(h)=R_{e,\rm ref}(h/h_{\rm ref})^{-1},
$$
$$
\Sigma_\ast(h)=\Sigma_{\ast,\rm ref},
$$
$$
m_R(h)=m_{R,\rm ref}+(2-\gamma)\log_{10}(h/h_{\rm ref}).
$$

### 6.6 推荐加入你项目文稿的一段 convention 声明草案

下面这段不是润色正文，而是建议你之后在方法报告或论文中加入的 convention block：

> Unless otherwise stated, all physical quantities are reported at a fiducial $h_{\rm ref}$. Stellar masses are luminosity-based SED/SPS estimates and scale as $M_\ast\propto h^{-2}$. Physical sizes scale as $R_e\propto h^{-1}$. Therefore the stellar surface density $\Sigma_\ast=M_\ast/(2\pi R_e^2)$ is invariant under a pure $h$ rescaling. The lensing-constrained mass used in this work is the projected mass enclosed within a fixed physical aperture $R=5$ or $10\,{\rm kpc}$, not the Einstein mass; for the power-law model it scales as $M_{2{\rm D}}(<R)\propto h^{2-\gamma}$. Consequently, comparisons between $M_\ast$ and $M_{2{\rm D}}(<R)$, or ratios constructed from them, are always interpreted at the same $h_{\rm ref}$ or rescaled using the above laws.

### 6.7 调研来源简表

| 来源 | 相关做法 | 本项目可借鉴点 |
|---|---|---|
| Hogg 1999, *Distance measures in cosmology* | 明确定义 $H_0=100h$、$D_A$、$D_L$、flux–luminosity relation 与 distance modulus | 用它支撑 $D_A,D_L\propto h^{-1}$，从而推出 $L,M_\ast\propto h^{-2}$ |
| SDSS MPA-JHU galaxy properties documentation | stellar mass 由 Bayesian/model-grid 方法和 ugriz photometry 得到，并输出 fiber/total mass | 说明 stellar mass catalog 是上游 derived product，不是直接观测量 |
| van Uitert et al. 2016, KiDS+GAMA SHMR | stellar mass bins 用 $h^{-2}M_\odot$，halo mass 用 $h^{-1}M_\odot$，并处理 catalog 间 $h$ 差异 | 质量比较必须显式区分 stellar mass 与 lensing/halo mass 的 $h$ exponent |
| Croton 2013, *Damn You, Little h!* | 强调 $h$ 不是单位，不同 measurement channel 的同名物理量可以有不同 $h$ scaling | 不能默认所有 mass 同一 $h$ 依赖；必须按 derivation chain 标注 |
| Baldry practical $h$-dependence guide | 汇总 distance、luminosity、scaled-luminosity mass、dynamical mass 的常见 $h$ 缩放 | 可作为代码/表格标签的 sanity-check checklist |
| Sánchez 2020, *Arguments against using $h^{-1}{\rm Mpc}$ units* | 批评把 $h$ 藏在 $h^{-1}{\rm Mpc}$ 中会遮蔽物理依赖，建议 fixed-Mpc / physical-density 参数化 | 本项目应采用“fiducial value + explicit rescaling law”，而不是只在单位里塞 $h$ |

---

## 最终一句话结论

**在用户补充上游 stellar-mass 链路后，stellar mass 的 $h$ 依赖已经可以确定为 $M_\ast\propto h^{-2}$；文稿内部公式则确定 fixed-kpc lensing mass 为 $M_R\propto h^{2-\gamma}$。**  
因此，stellar mass 与 lensing-constrained mass 的 $h$ 依赖确实不同；下一步应把 $h_{\rm ref}$、$M_\ast$ 的 $-2$ 指数、$R_e$ 的 $-1$ 指数、$\Sigma_\ast$ 的 $h^0$ 抵消、以及 $m_5/m_{10}$ 的 $(2-\gamma)$ 指数写进元数据、图轴、表头和方法 convention block。
