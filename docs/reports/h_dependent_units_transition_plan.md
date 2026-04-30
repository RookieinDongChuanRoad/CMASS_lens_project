# 建议文件名：`h_dependent_units_transition_plan.md`

# 将 CMASS strong-lens 主线改为 h-dependent units 的推进计划

## 0. 核心判断

你的项目与 Sonnenfeld (2024) 的 SLACS debias 工作在统计结构上非常接近：二者都把强透镜样本理解为从 parent population 经 lensing detectability 与 lens-finding probability 选择出来的 biased sample；二者都用幂律总质量分布，以 projected enclosed mass 与 logarithmic density slope $\gamma$ 表征 galaxy mass structure；二者都把 Einstein radius 的约束压缩成一条 $m_R(\gamma)$ 质量轨迹，再用 stellar kinematics / Jeans response 沿同一条轨迹重新加权；二者也都把 parent、detectable、selected population 的 posterior predictive trends 当作最终解释对象。

这次 h-convention 重构的目标不是改变这套统计骨架，而是把所有带纯 $h$ 缩放的物理量改写成 **h-dependent units 下的 h-free numerical coefficients**。也就是说，对任意量 $Q$，若

$$
Q = \tilde Q\,h^n U,
$$

则代码、表格和图轴优先保存或展示 $\tilde Q$，并在单位里写出 $h^n U$。这样 $\tilde Q$ 在只改变 $h$、固定其他宇宙学参数时不再变化。

最重要的模型层决定是：下一版不要再把主质量定义为 fixed physical aperture 的

$$
M_{2\mathrm D}(<5\,\mathrm{kpc})\quad\text{or}\quad M_{2\mathrm D}(<10\,\mathrm{kpc}),
$$

而应改为

$$
M_{2\mathrm D}(<5\,h^{-1}\mathrm{kpc})\quad\text{or}\quad M_{2\mathrm D}(<10\,h^{-1}\mathrm{kpc}),
$$

并把对应质量写成 $h^{-1}M_\odot$ 单位。这样 lensing-constrained projected mass 的数值可以像 KiDS+GAMA SHMR 那样成为 h-free coefficient，而不是继续保留当前 fixed-kpc 定义下的 $h^{2-\gamma}$ 依赖。

---

## 1. 从 Sonnenfeld (2024) 反推对本项目的更新理解

### 1.1 Sonnenfeld (2024) 的方法骨架

Sonnenfeld (2024) 的核心问题是：SLACS lenses 不是 parent early-type galaxy population 的无偏抽样；强透镜截面、spectroscopic detectability、HST follow-up prioritisation 和人工 lens finding 都会改变被观测 lens sample 的 $(M_\ast,R_e,\sigma_{\rm ap},M_5,\gamma)$ 分布。因此，他没有只拟合已发现 lenses，而是显式写出

$$
P_{\rm SL}(\psi_g,\psi_s)\propto P_g(\psi_g)P_s(\psi_s)P_{\rm sel}(\psi_g,\psi_s),
$$

再把 selection 拆成 detectable part 与 finding part，并用 empirical $P_{\rm find}$ 与 cross-section factor $g$ 来近似不可完全 forward-model 的 survey process。

在 mass model 上，他把每个 lens 的总质量分布写成 spherical power-law，并用

$$
M_5\equiv M_{2\mathrm D}(<5\,\mathrm{kpc}),\qquad \gamma
$$

作为两个径向自由度。选择 $5\,\mathrm{kpc}$ 的动机是 lensing directly constrains projected mass，且 5 kpc 接近 SLACS median physical Einstein radius $4.2\,\mathrm{kpc}$，所以外推较小。数据向量包含 lens redshift、source redshift、stellar mass、half-light radius、stellar velocity dispersion 和 Einstein radius；Einstein radius uncertainty 被近似忽略后，$(m_5,\gamma)$ 积分可以经变量变换压到 $m_5^{\rm obs}(\gamma)$ 轨迹上，再与 Jeans-predicted $\sigma_{\rm ap}$ 一起进入 likelihood。

Sonnenfeld 的结果展示也与本项目很接近：他报告 $\mu_{5,0}$、$\beta_5$、$\xi_5$、$\mu_\gamma$、$\beta_\gamma$、$\xi_\gamma$ 等 population parameters，并分别画 parent population、detectable lenses 和 selected SLACS lenses 的 posterior predicted trends。

### 1.2 CMASS 主线与 Sonnenfeld (2024) 的对应关系

你当前 CMASS 主线报告中的对象可以直接映射到 Sonnenfeld 的框架：

| Sonnenfeld (2024) | CMASS 主线当前写法 | 解释 |
|---|---|---|
| $M_5=M_{2\rm D}(<5\,\mathrm{kpc})$ | $m_R$, especially $m_5$ / $m_{10}$ | projected enclosed mass definition |
| $\gamma$ | $\gamma$ | power-law logarithmic density slope |
| $m_5^{\rm obs}(\gamma)$ | $m_{R,i}(\gamma)$ | Einstein radius collapses $(m_R,\gamma)$ to a 1D track |
| $P_{\rm find}$ | sigmoid $P_{\rm find}(\theta_E)$ | empirical lens-finding efficiency |
| $g(\theta_E,\gamma)$ | cross-section lookup $g(\theta_E,\gamma)$ | geometric / cross-section term |
| parent / detectable / selected | parent / detectable / selected PPC catalogs | debiased population interpretation |
| $P(d_i\mid\eta)$ with selection normalization | $\sum_i\log\mathcal L_i-N_{\rm lens}\log Z_{\rm norm}$ | selection-corrected hierarchical posterior |

两者的关键差异是：Sonnenfeld 文稿把 $H_0=70\,\mathrm{km\,s^{-1}\,Mpc^{-1}}$ 固定，并把 $M_5/M_\odot$、$M_\ast/M_\odot$、$R_e/\mathrm{kpc}$ 直接作为 fixed-cosmology 数值使用；你的下一步目标则是把这些数值拆成 h-free coefficient + h-dependent unit，避免后续与 stellar mass catalog、weak-lensing halo mass 或外部 SHMR 文献比较时发生 convention 混用。

### 1.3 这对本项目 h 处理的启发

Sonnenfeld 的统计形式可以借鉴，但他的 fixed $5\,\mathrm{kpc}$ mass definition 不应原样继承到下一版 h-free convention。原因是你之前审计已经推出，在当前 fixed physical aperture 定义下，

$$
10^{m_R}=\pi\Sigma_c\,r_{\rm Ein}^{\gamma-1}R^{3-\gamma},
$$

其中

$$
\Sigma_c\propto h,\qquad r_{\rm Ein}\propto h^{-1},\qquad R=\mathrm{fixed\ kpc}\propto h^0,
$$

所以

$$
M_{2\rm D}(<R_{\rm fixed\ kpc})\propto h^{2-\gamma}.
$$

这导致 $m_5$ 或 $m_{10}$ 的 h exponent 随 $\gamma$ 变化，无法像 KiDS+GAMA 的 halo mass 那样写成统一的 $h^{-1}M_\odot$ 数值。把 aperture 改成 $\tilde R\,h^{-1}\mathrm{kpc}$ 后，

$$
R\propto h^{-1},
$$

于是

$$
M_{2\rm D}(<\tilde R h^{-1}\mathrm{kpc})
\propto h\,h^{-(\gamma-1)}h^{-(3-\gamma)}=h^{-1}.
$$

这就是下一版应采用 $h^{-1}M_\odot$ lensing mass unit 的核心理由。

---

## 2. 统一 h-dependent units 的总原则

### 2.1 统一定义

后续所有 h-dependent units 都建议采用同一个抽象定义：

$$
Q_{\rm phys}=\tilde Q\,h^nU,
$$

其中：

- $Q_{\rm phys}$：物理量本身；
- $U$：不含 $h$ 的基本单位，如 $M_\odot$、$\mathrm{kpc}$、$L_\odot$；
- $n$：该 measurement channel 的纯 $h$ exponent；
- $\tilde Q$：代码和图表中优先使用的 h-free 数值。

如果当前 raw HDF5 或旧结果是在 fiducial $h_{\rm ref}$ 下保存的 physical-unit 数值

$$
q_{\rm old}=\log_{10}(Q_{\rm phys}(h_{\rm ref})/U),
$$

则 h-free coefficient 的 log 数值是

$$
\tilde q = q_{\rm old}-n\log_{10}h_{\rm ref}.
$$

例如：

$$
\tilde m_\ast = m_{\ast,\rm old}+2\log_{10}h_{\rm ref}
$$

因为 $M_\ast\propto h^{-2}$，而

$$
\tilde m_{R_h}=m_{R,\rm phys}(h_{\rm ref})+\log_{10}h_{\rm ref}
$$

如果这个 $m_{R,\rm phys}$ 已经是同一 $\tilde R h^{-1}\mathrm{kpc}$ aperture 下的 physical mass。

---

## 3. 哪些物理量需要改用 h-dependent units

### 3.1 建议作为下一版标准的单位表

| 类别 | 量 | 建议保存/展示的 h-free 数值 | 建议单位或轴标签 | 纯 $h$ exponent | 说明 |
|---|---|---|---|---:|---|
| 直接观测量 | redshift $z_d,z_s$ | 原值 | dimensionless | 0 | 不加 h 单位 |
| 直接观测量 | angle $\theta_E$, aperture arcsec, seeing arcsec | 原值 | arcsec | 0 | 尽量保留角量以减少 convention 风险 |
| 直接观测量 | velocity dispersion $\sigma_{\rm ap}$ | 原值 | $\mathrm{km\,s^{-1}}$ | 0 | 速度本身不需要 h unit |
| 距离 | $D_A,D_L,D_d,D_s,D_{ds}$ | $hD$ | $h^{-1}\mathrm{Mpc}$ 或 $h^{-1}\mathrm{kpc}$ | -1 | 只要由 redshift + cosmology 得到，固定其他参数时 $D\propto h^{-1}$ |
| 长度 | $r_{\rm Ein}$, $R_e$, projected radius, impact parameter | $R/(h^{-1}\mathrm{kpc})$ | $h^{-1}\mathrm{kpc}$ | -1 | $R_e$ 的 size relation 必须同步改单位 |
| 主 aperture | $R$ | $\tilde R=5,10$ | $h^{-1}\mathrm{kpc}$ | -1 | 主质量定义从 $5\,\mathrm{kpc}$ 改为 $5\,h^{-1}\mathrm{kpc}$ |
| Luminosity | $L$ | $L/(h^{-2}L_\odot)$ | $h^{-2}L_\odot$ | -2 | flux $\rightarrow D_L^2\rightarrow L$ |
| Stellar mass | $M_\ast$ | $M_\ast/(h^{-2}M_\odot)$ | $h^{-2}M_\odot$ | -2 | photometry / luminosity / SED / SPS 链 |
| Stellar mass log | $m_\ast^{(h2)}$ | $\log_{10}[M_\ast/(h^{-2}M_\odot)]$ | dex | 0 after unit choice | 旧 $m_\ast$ pivot 需要转换 |
| Lensing projected mass | $M_{\tilde R h}$ | $M_{2D}(<\tilde R h^{-1}\mathrm{kpc})/(h^{-1}M_\odot)$ | $h^{-1}M_\odot$ | -1 | 这是新 $m_R$ 的核心定义 |
| Einstein mass | $M_{\rm Ein}$ | $M_{\rm Ein}/(h^{-1}M_\odot)$ | $h^{-1}M_\odot$ | -1 | 若后续报告 Einstein-aperture mass，它自然属于此类 |
| Halo / dynamical mass | $M_h,M_{\rm dyn}$ | $M/(h^{-1}M_\odot)$ | $h^{-1}M_\odot$ | -1 | 与 weak lensing / dynamics 常用 convention 对齐 |
| Critical surface density | $\Sigma_c$ | $\Sigma_c/(hM_\odot\mathrm{kpc}^{-2})$ | $hM_\odot\mathrm{kpc}^{-2}$ | +1 | $\Sigma_c\propto D_s/(D_dD_{ds})\propto h$ |
| Total projected density | $\Sigma_{\rm tot}$ | $\Sigma_{\rm tot}/(hM_\odot\mathrm{kpc}^{-2})$ | $hM_\odot\mathrm{kpc}^{-2}$ | +1 | total/lensing mass $h^{-1}$ over area $h^{-2}$ |
| 3D total density | $\rho_{\rm tot}$ | $\rho/(h^2M_\odot\mathrm{kpc}^{-3})$ | $h^2M_\odot\mathrm{kpc}^{-3}$ | +2 | total mass $h^{-1}$ over volume $h^{-3}$ |
| Stellar surface density | $\Sigma_\ast=M_\ast/(2\pi R_e^2)$ | $\Sigma_\ast/(M_\odot\mathrm{kpc}^{-2})$ | $M_\odot\mathrm{kpc}^{-2}$ | 0 | $h^{-2}/h^{-2}$ 抵消；这是 `sigma_star_dependent` 的优势 |
| Stellar 3D density, if used | $\rho_\ast$ | $\rho_\ast/(hM_\odot\mathrm{kpc}^{-3})$ | $hM_\odot\mathrm{kpc}^{-3}$ | +1 | stellar mass $h^{-2}$ over volume $h^{-3}$ |
| Convergence | $\kappa$ | 原值 | dimensionless | 0 for lensing total $\Sigma/\Sigma_c$ | 若是 stellar convergence $\kappa_\ast=\Sigma_\ast/\Sigma_c$，则不是 $h^0$，见下行 |
| Stellar convergence / stellar fraction proxy | $\kappa_\ast=\Sigma_\ast/\Sigma_c$ | $h\kappa_\ast$ | dimensionless with explicit $h^{-1}$ scaling | -1 | 因 $\Sigma_\ast\propto h^0$, $\Sigma_c\propto h$ |
| Stellar-to-lensing mass fraction | $f_\ast=M_\ast/M_{\tilde Rh}$ | $hf_\ast$ | dimensionless, but report as $h^{-1}$ scaling | -1 | 维度无量纲但不是 h-free |
| Lensing-to-stellar mass ratio | $M_{\tilde Rh}/M_\ast$ | $(M_{\tilde Rh}/M_\ast)/h$ | dimensionless with explicit $h$ scaling | +1 | 与 KiDS+GAMA 报告 $M_h/M_\ast\,[h]$ 的逻辑一致 |

### 3.2 最容易误判的三类量

#### 3.2.1 `m_star` 与 `m_R` 都叫 mass，但单位指数不同

下一版应明确：

$$
m_\ast^{(h2)}\equiv \log_{10}\left(\frac{M_\ast}{h^{-2}M_\odot}\right),
$$

而

$$
m_{\tilde Rh}\equiv \log_{10}\left(\frac{M_{2\rm D}(<\tilde Rh^{-1}\mathrm{kpc})}{h^{-1}M_\odot}\right).
$$

因此 stellar mass 与 lensing mass 的 ratio 仍然有 $h$ dependence。使用 h-dependent units 只是让每个 measurement channel 的单独数值 h-free，不会让不同 channel 的 dimensionless ratio 自动 h-free。

#### 3.2.2 `Sigma_star` 可以保持 h-free，但 `Sigma_c` 不可以

用新变量写

$$
\log\Sigma_\ast
= m_\ast^{(h2)}-\log_{10}(2\pi)-2r_e^{(h)},
$$

其中

$$
r_e^{(h)}\equiv \log_{10}\left(\frac{R_e}{h^{-1}\mathrm{kpc}}\right).
$$

这个组合完全抵消 $h$，所以 `sigma_star_dependent` 中的 $\log\Sigma_\ast-9$ 可以保持现有数值定义。相反，$\Sigma_c$ 和 total projected density 必须写成 $hM_\odot\mathrm{kpc}^{-2}$。

#### 3.2.3 fixed-kpc `m5` 与 h-aperture `m5h` 不是同一个物理量

Sonnenfeld 的 $M_5$ 是 $M_{2\rm D}(<5\,\mathrm{kpc})$。下一版建议的 $M_{5h}$ 是 $M_{2\rm D}(<5h^{-1}\,\mathrm{kpc})$。若 $h_{\rm ref}=0.7$，后者在 fiducial cosmology 下对应 $7.14\,\mathrm{kpc}$，不是 Sonnenfeld 的 $5\,\mathrm{kpc}$。因此：

- 与 KiDS/GAMA/halo-mass convention 对齐时，使用 $m_{5h}$ 或 $m_{10h}$。
- 与 Sonnenfeld (2024) 的 $\mu_{5,0}$ 数值直接比较时，必须临时转换回 fixed $5\,\mathrm{kpc}$ convention。

---

## 4. 新的 $m_R$ 定义

### 4.1 推荐符号

建议在下一版文稿和代码中不要继续把 `m5` / `m10` 当作无歧义名字使用，而是引入清楚的新名字：

$$
m_{\tilde R h}\equiv
\log_{10}\left[
\frac{M_{2\mathrm D}(<\tilde R\,h^{-1}\mathrm{kpc})}{h^{-1}M_\odot}
\right],
\qquad \tilde R\in\{5,10\}.
$$

代码建议命名：

| 旧名 | 新名建议 | 含义 |
|---|---|---|
| `m5` | `m5_hinvkpc` or `m5h` | $\log_{10}[M_{2D}(<5h^{-1}\mathrm{kpc})/(h^{-1}M_\odot)]$ |
| `m10` | `m10_hinvkpc` or `m10h` | $\log_{10}[M_{2D}(<10h^{-1}\mathrm{kpc})/(h^{-1}M_\odot)]$ |
| `mu5_0` | `mu5h_0` | population mean in new mass unit |
| `mu10_0` | `mu10h_0` | population mean in new mass unit |
| `mass_definitions/m5` | `mass_definitions/m5_hinvkpc` | HDF5 branch should encode aperture and unit |
| `mass_definitions/m10` | `mass_definitions/m10_hinvkpc` | same |

### 4.2 h-free lensing formula

定义 h-free 距离和密度系数：

$$
\tilde D_A\equiv \frac{D_A}{h^{-1}\mathrm{Mpc}},
\qquad
\tilde r_{\rm Ein}\equiv \frac{r_{\rm Ein}}{h^{-1}\mathrm{kpc}},
\qquad
\tilde\Sigma_c\equiv \frac{\Sigma_c}{hM_\odot\mathrm{kpc}^{-2}}.
$$

则

$$
\tilde r_{\rm Ein}=\tilde D_d\,\theta_E/206265
$$

并且

$$
10^{m_{\tilde Rh}}
=
\pi\tilde\Sigma_c\,\tilde r_{\rm Ein}^{\gamma-1}\,\tilde R^{3-\gamma}.
$$

反向从 $(m_{\tilde Rh},\gamma)$ 预测 $\theta_E$ 时，使用

$$
\tilde r_{\rm Ein}
=
\left[
\frac{10^{m_{\tilde Rh}}}{\pi\tilde\Sigma_c\tilde R^{3-\gamma}}
\right]^{1/(\gamma-1)},
\qquad
\theta_E=\frac{\tilde r_{\rm Ein}}{\tilde D_d}206265.
$$

这个表达没有显式 $h$，适合放进 `m_R(gamma)` grid、single-lens likelihood、selection normalization 和 PPC generator。

### 4.3 与当前 fixed-kpc 结果的解析转换

如果旧结果是在 $h_{\rm ref}$ 下用 fixed $\tilde R\,\mathrm{kpc}$ aperture 得到的

$$
m_{\tilde R,\rm old}^{\rm fixed}
=
\log_{10}\left[\frac{M_{2D}(<\tilde R\,\mathrm{kpc};h_{\rm ref})}{M_\odot}\right],
$$

则对应新定义的 h-free coefficient 是

$$
\boxed{
 m_{\tilde Rh}
 =m_{\tilde R,\rm old}^{\rm fixed}-(2-\gamma)\log_{10}h_{\rm ref}
}
$$

前提是：旧 aperture 的数值系数也是 $\tilde R$，例如 old `m5` 是 $5\,\mathrm{kpc}$，new `m5h` 是 $5h^{-1}\,\mathrm{kpc}$。这个公式必须逐 lens、逐 posterior draw、逐 $\gamma$ 应用；不要只对 median mass 做一次常数平移。

两个 h-aperture mass definitions 之间仍然满足

$$
m_{10h}=m_{5h}+(3-\gamma)\log_{10}2.
$$

### 4.4 与 Sonnenfeld fixed-5-kpc 结果比较时的反向转换

若你要把下一版的 $m_{5h}$ 与 Sonnenfeld (2024) 的 $\mu_{5,0}$ 对齐，必须回到 fixed $5\,\mathrm{kpc}$ at $h_{\rm ref}=0.7$ 的 convention：

$$
m_{5}^{\rm fixed}(h_{\rm ref})
=
m_{5h}+(2-\gamma)\log_{10}h_{\rm ref}.
$$

当 $\gamma\approx2$ 时这个修正很小，但它不是严格为零；并且对 $\gamma$ posterior 有散度时，应该 draw-wise 转换后再汇总。

---

## 5. 对模型各模块的具体改动

### 5.1 Raw observation / HDF5 层

必须新增或修改以下 attrs：

| attr / field | 建议内容 |
|---|---|
| `h_ref` | 当前 fiducial $h$，例如若 $H_0=70$ 则 `0.7`；先从代码确认，不要假定 |
| `cosmology_convention` | 固定 $\Omega_M,\Omega_\Lambda$ 时抽取 little-h scaling |
| `stellar_mass_unit` | `h^-2 Msun` |
| `stellar_mass_log_definition` | `log10(M_star / (h^-2 Msun))` |
| `size_unit` | `h^-1 kpc` |
| `size_log_definition` | `log10(Re / (h^-1 kpc))` |
| `mass_aperture_unit` | `h^-1 kpc` |
| `mass_unit` | `h^-1 Msun` |
| `mass_log_definition` | `log10(M_2D(<R_hinvkpc) / (h^-1 Msun))` |
| `mass_definitions` | new keys `m5_hinvkpc`, `m10_hinvkpc`; old `m5`, `m10` kept only if explicitly labeled `fixed_kpc_legacy` |
| `log10_Sigma_star_unit` | `Msun kpc^-2`, no h factor |
| `unit_version` | e.g. `h_units_v1` |

转换公式：

$$
m_\ast^{(h2)}=m_{\ast,\rm old}+2\log_{10}h_{\rm ref},
$$

$$
r_e^{(h)}=r_{e,\rm old}+\log_{10}h_{\rm ref},
$$

$$
\log\Sigma_\ast^{\rm new}
=m_\ast^{(h2)}-\log_{10}(2\pi)-2r_e^{(h)}
=\log\Sigma_\ast^{\rm old}.
$$

因此 `log10_Sigma_star` 可以作为第一组 sanity check：如果上游 $M_\ast$ 和 $R_e$ 的旧 h convention 一致，那么转成 h-dependent units 后 $\log\Sigma_\ast$ 应保持不变。

### 5.2 Mass grid / interpolation grid 层

当前主线的关键公式是：

$$
m_{R,i}(\gamma)
=\log_{10}\left[
\pi\Sigma_c(z_{d,i},z_{s,i})r_{{\rm Ein},i}^{\gamma-1}R^{3-\gamma}
\right].
$$

下一版应换成 h-free 版本：

$$
m_{\tilde Rh,i}(\gamma)
=\log_{10}\left[
\pi\tilde\Sigma_c(z_{d,i},z_{s,i})\tilde r_{{\rm Ein},i}^{\gamma-1}\tilde R^{3-\gamma}
\right].
$$

实现建议：

1. 在 `prepare_intepolation_grids` 中新增 mass definition object，而不是仅改 label。
2. 将 `aperture_kpc` 改为 `aperture_hinv_kpc` 或 `aperture_coefficient_hinv_kpc`。
3. 将所有 distance-derived physical lengths 统一保存为 coefficient：`D_A_hinv_Mpc`, `r_ein_hinv_kpc`, `Re_hinv_kpc`。
4. `m5_hinvkpc` 与 `m10_hinvkpc` 都应从 h-free formula 直接生成。
5. 保留一个 legacy conversion function，用于旧结果对照：

```text
mR_hinv_from_fixed_kpc(mR_fixed, gamma, h_ref)
    = mR_fixed - (2 - gamma) * log10(h_ref)
```

6. 对所有 old `m10` result，不要把这条转换当作最终科学结果；它只能用于 migration test。正式 run 应从 raw products / grids 重建。

### 5.3 Dynamics / Jeans response 层

当前代码使用

$$
\sigma_{\rm model}(\gamma)=\sqrt{S_{\rm unit}(\gamma)10^{m_R(\gamma)}}.
$$

如果新 $m_R$ 是 h-free coefficient $m_{\tilde Rh}$，则 $S_{\rm unit}$ 的含义必须同步变成

$$
S_{\rm unit}^{(h)}\equiv \frac{\sigma^2}{10^{m_{\tilde Rh}}}.
$$

若只做 analytic migration test，旧 fixed-kpc response 与新 response 的近似关系是

$$
S_{\rm unit}^{(h)}(\gamma)
=
S_{\rm unit}^{\rm fixed}(\gamma)h_{\rm ref}^{2-\gamma}.
$$

这是因为

$$
10^{m_{\tilde Rh}}=10^{m_R^{\rm fixed}}h_{\rm ref}^{-(2-\gamma)}.
$$

但是正式推荐仍是 **重建 Jeans / interpolation tables**，原因有三点：

1. 新 mass definition 的 aperture 物理半径不同；
2. dynamics table 可能还隐含 light profile、aperture geometry、seeing、$R_e$ 单位等 convention；
3. 只改 mass grid 而不改 `S_unit` 会导致 $\sigma_{\rm model}$ 人为偏移。

验证标准：同一个 lens、同一个 $\gamma$、同一套物理 profile 下，旧 convention 和新 convention 预测出的 observable $\theta_E$ 与 $\sigma_{\rm ap}$ 应相同；若不同，说明 mass coefficient 与 dynamics response 没有同步转换。

### 5.4 Hierarchical population law 层

下一版建议把群体变量整体改写为：

$$
m_\ast^{(h2)}=\log_{10}\left(\frac{M_\ast}{h^{-2}M_\odot}\right),
$$

$$
r_e^{(h)}=\log_{10}\left(\frac{R_e}{h^{-1}\mathrm{kpc}}\right),
$$

$$
m_{\tilde Rh}=\log_{10}\left(\frac{M_{2D}(<\tilde Rh^{-1}\mathrm{kpc})}{h^{-1}M_\odot}\right).
$$

因此 size residual 应写成

$$
\Delta_R^{(h)}=r_e^{(h)}-\mu_r^{(h)}(m_\ast^{(h2)},n).
$$

质量均值关系改成

$$
\mu_{m_{\tilde Rh}}=
\mu_{\tilde Rh,0}
+\beta_{\tilde Rh}\left(m_\ast^{(h2)}-m_{\ast,\rm piv}^{(h2)}\right)
+\xi_{\tilde Rh}\Delta_R^{(h)}.
$$

重点：旧文稿中写死的 stellar-mass pivot，例如 $11.4$，不能自动沿用。应先判断它代表什么：

- 如果旧 $11.4$ 是在 $M_\odot$ physical unit at $h_{\rm ref}$ 下选的 pivot，则新 pivot 应为
  $$
  11.4+2\log_{10}h_{\rm ref}.
  $$
- 如果你想在新 convention 下重新选一个可读 pivot，则应使用新 $m_\ast^{(h2)}$ 分布的 median 或 rounded median。
- 无论采用哪种，都必须写进 HDF5 attrs、config 和文稿。

`independent` 和 `dependent` gamma modes 只需要替换变量定义；`sigma_star_dependent` 有一个优势：

$$
\log\Sigma_\ast-9
=m_\ast^{(h2)}-\log_{10}(2\pi)-2r_e^{(h)}-9
$$

是 h-free 的，因此它不需要额外 h offset。但它仍然依赖 $M_\ast$ 与 $R_e$ 是否来自同一 $h_{\rm ref}$。

### 5.5 Selection function 与 normalization 层

$P_{\rm find}$ 和 $g$ 的输入主要是 angular Einstein radius $\theta_E$ 与 $\gamma$，所以它们的数学形式可以保持不变：

$$
P_{\rm find}(\theta_E)=\frac{1}{1+\exp[-a(\theta_E-\theta_0)]},
$$

$$
g(\theta_E,\gamma)=\pi[c_s(\gamma)\theta_E]^2.
$$

需要改的是 latent variables 到 $\theta_E$ 的 forward map：

$$
(m_{\tilde Rh},\gamma,z_d,z_s)\rightarrow \tilde r_{\rm Ein}\rightarrow \theta_E.
$$

因此 `Z_norm` 结构不变，但 Monte Carlo latent population 的 variables 和 units 必须重建。不能只在最终表格里把 mass label 改成 $h^{-1}M_\odot$，否则 normalization 仍然是在旧 fixed-kpc mass prior 上做的。

### 5.6 Figures / tables / captions 层

建议所有 mass/size axis 统一改写为下面这种形式：

| 旧轴标签 | 新轴标签建议 |
|---|---|
| $\log M_\ast/M_\odot$ | $\log_{10}[M_\ast/(h^{-2}M_\odot)]$ |
| $\log R_e/\mathrm{kpc}$ | $\log_{10}[R_e/(h^{-1}\mathrm{kpc})]$ |
| $m_5$ | $\log_{10}[M_{2D}(<5h^{-1}\mathrm{kpc})/(h^{-1}M_\odot)]$ |
| $m_{10}$ | $\log_{10}[M_{2D}(<10h^{-1}\mathrm{kpc})/(h^{-1}M_\odot)]$ |
| $\log\Sigma_\ast$ | $\log_{10}[\Sigma_\ast/(M_\odot\mathrm{kpc}^{-2})]$ |
| $M_\ast/M_{10}$ | either $hM_\ast/M_{10h}$ or explicitly $M_\ast/M_{10h}\propto h^{-1}$ |

Caption 必须额外说明：

> Unless otherwise stated, stellar masses are quoted in $h^{-2}M_\odot$, lengths in $h^{-1}\mathrm{kpc}$, and lensing enclosed masses in $h^{-1}M_\odot$ within apertures of $5$ or $10\,h^{-1}\mathrm{kpc}$. Stellar surface density $\Sigma_\ast=M_\ast/(2\pi R_e^2)$ is quoted in $M_\odot\mathrm{kpc}^{-2}$, with no explicit little-h factor.

---

## 6. 分阶段推进计划

### Phase 0 — 冻结 convention 决策

**目标**：先决定并写死一套 unit contract，避免边做边改。

必须完成：

1. 确认当前 pipeline 的 $h_{\rm ref}$ 和 cosmology attrs；若当前确实对应 $H_0=70$，则 $h_{\rm ref}=0.7$。
2. 确认是否所有上游 $M_\ast$、$R_e$、mass grid、Jeans grid 使用同一个 $h_{\rm ref}$。
3. 决定新变量名：推荐 `m5_hinvkpc`, `m10_hinvkpc`, `mstar_h2`, `re_hinvkpc`。
4. 写一个 `UNIT_CONVENTION.md` 或 config block，作为代码和文稿的共同源头。

验收标准：任何人只看 HDF5 attrs 和 config，就能判断每个 log quantity 的 denominator 是什么。

### Phase 1 — 建立 analytic migration layer

**目标**：不急着重跑全 pipeline，先用解析转换验证所有 h scalings。

需要实现的 helper：

```text
logMstar_h2_from_physical(logMstar_old, h_ref)
    = logMstar_old + 2 * log10(h_ref)

logRe_hinv_from_physical(logRe_old, h_ref)
    = logRe_old + log10(h_ref)

logSigmaStar_from_h_units(logMstar_h2, logRe_hinv)
    = logMstar_h2 - log10(2*pi) - 2*logRe_hinv

mR_hinv_from_fixed_kpc(mR_fixed, gamma, h_ref)
    = mR_fixed - (2 - gamma) * log10(h_ref)

Sunit_hinv_from_fixed_kpc(Sunit_fixed, gamma, h_ref)
    = Sunit_fixed * h_ref**(2 - gamma)

m_fixed_kpc_from_mR_hinv(mR_hinv, gamma, h_ref)
    = mR_hinv + (2 - gamma) * log10(h_ref)
```

最关键的 sanity checks：

1. `logSigmaStar_new - logSigmaStar_old` 应接近 0。
2. 对任一 lens 和任一 $\gamma$，用 old fixed-kpc formula 得到的 $\theta_E$ 与用 new h-unit formula 反推的 $\theta_E$ 应一致。
3. 若同时转换 `S_unit`，$\sigma_{\rm model}$ 应一致。
4. `m10_hinv - m5_hinv` 应仍等于 $(3-\gamma)\log_{10}2$。

### Phase 2 — 重建 raw observation branches

**目标**：在 HDF5 层建立新 convention 分支，不污染旧结果。

建议建立新 raw 文件族，例如：

```text
data/raw/observations_*_hunits_v1.hdf5
```

每个 lens group 应同时保存：

| field | 说明 |
|---|---|
| `log10_Mstar_h2` | $\log_{10}[M_\ast/(h^{-2}M_\odot)]$ |
| `log10_Re_hinv_kpc` | $\log_{10}[R_e/(h^{-1}\mathrm{kpc})]$ |
| `log10_Sigma_star` | 不变，但要注明 unit |
| `theta_ein_arcsec` | 原始角量，不改 |
| `sigma_ap_km_s` | 原始速度量，不改 |
| `mass_definitions/m5_hinvkpc` | 新 mass grid |
| `mass_definitions/m10_hinvkpc` | 新 mass grid |
| `legacy/m5_fixed_kpc` | 可选，仅用于 regression test |
| `legacy/m10_fixed_kpc` | 可选，仅用于 regression test |

注意：如果 BOSS branch 与 slit branch 都要保留，二者的 aperture/seeing observation contract 仍要分开；h unit 重构不应把 slit、boss、within_re 这些 aperture contracts 混成一个概念。

### Phase 3 — 重建 interpolation grids 与 Jeans response

**目标**：让 mass grid、Einstein-radius map 和 dynamics response 全部服务于同一个 $m_{\tilde Rh}$ definition。

需要改动：

1. `prepare_intepolation_grids` 中新增 `mass_definition = m5_hinvkpc / m10_hinvkpc`。
2. 所有 `aperture_kpc = 5/10` 改成 `aperture_hinv_kpc = 5/10`。
3. 计算 $\Sigma_c$ 时输出 `Sigma_c_h_Msun_kpc2` coefficient。
4. 计算 $r_{\rm Ein}$ 时输出 `r_ein_hinv_kpc` coefficient。
5. 重建 `S_unit` 表；若短期无法重建，用 Phase 1 的 analytic rescaling 做 temporary branch，并在 attrs 中标明 `S_unit_source = analytic_rescaled_legacy`。
6. 为每张 grid 写入 unit metadata：mass denominator、length denominator、h_ref、是否 legacy。

验收标准：任一 grid 文件内部不再出现含混的 `kpc` aperture；必须明确是 `fixed_kpc` 还是 `hinv_kpc`。

### Phase 4 — 修改 Bayesian inference 层

**目标**：保持 Sonnenfeld-like posterior 结构不变，只替换 latent variables 和 unit definitions。

需要改动：

1. 将 likelihood 中的 $m_R$ 变量替换为 $m_{\tilde Rh}$。
2. 将 observed stellar mass likelihood 从 old $m_\ast$ 改为 $m_\ast^{(h2)}$。
3. 将 observed size likelihood / size relation 从 old $r_e$ 改为 $r_e^{(h)}$。
4. 重设 stellar-mass pivot：不要直接沿用 old `11.4`，除非确认它已经是 $h^{-2}M_\odot$ convention。
5. 更新 parameter names：`mu10_0` $\rightarrow$ `mu10h_0`，`beta10` $\rightarrow$ `beta10h`，等等。
6. 更新 prior ranges。因为 $m_\ast$ 与 $m_R$ 数值都会平移，旧盒先验边界不一定仍然覆盖合理空间。
7. `Z_norm` 的 Monte Carlo basis 可以保留，但 latent transform 必须用新 units。
8. `P_find`、$g$、$\theta_0$、`loga` 的形式不需要因为 h units 改动而改变；它们用角量。

验收标准：在 synthetic data 或单 lens regression test 中，新旧 convention 只改变变量数值和标签，不改变可观测 $\theta_E$、$\sigma_{\rm ap}$ 的预测。

### Phase 5 — 修改 PPC、posterior trends 和 Fig. 8-like overlay

**目标**：结果展示完全使用新 unit labels，同时保留必要的 legacy comparison。

需要改动：

1. PPC replicated catalog 的 mass fields 改为 `m5h`, `m10h`。
2. `parent/detectable/selected` trend reducers 的 x-axis 用 $m_\ast^{(h2)}$。
3. y-axis mass trends 用 $m_{5h}$ 或 $m_{10h}$。
4. single-lens flat-prior summary 必须重新生成；不要只把旧 median mass 平移，因为 $\gamma$ posterior 与 mass projection 有耦合。
5. Fig. 8-like overlay 中旧观测点如果保留，应标为 legacy fixed-kpc points；正式图中优先使用新 `mRh` points。
6. 增加一个 conversion appendix/table，说明如何把新结果转换回 Sonnenfeld-style fixed $5\,\mathrm{kpc}$ at $h=0.7$。

验收标准：任何图中只要出现 $M_\ast$、$R_e$、$m_R$、$\Sigma_c$、$\Sigma_\ast$、mass ratio，就必须从轴标签或 caption 读出 h convention。

### Phase 6 — 回归测试与科学 sanity checks

建议建立以下自动测试：

| 测试 | 公式 / 标准 | 目的 |
|---|---|---|
| stellar mass conversion | $m_\ast^{(h2)}=m_{\ast,old}+2\log h_{\rm ref}$ | 防止 stellar mass 没转 |
| size conversion | $r_e^{(h)}=r_{e,old}+\log h_{\rm ref}$ | 防止 size relation 混用 |
| $\Sigma_\ast$ invariance | $\Delta\log\Sigma_\ast\approx0$ | 检查 stellar mass 与 size convention 是否一致 |
| mass conversion | $m_{Rh}=m_R^{fixed}-(2-\gamma)\log h_{\rm ref}$ | 检查 lensing mass conversion |
| aperture ratio | $m_{10h}-m_{5h}=(3-\gamma)\log2$ | 检查 power-law aperture relation |
| $\theta_E$ invariance | old/new predicted $\theta_E$ agree | 检查 lensing geometry |
| $\sigma$ invariance | old/new predicted $\sigma_{\rm model}$ agree after `S_unit` conversion | 检查 dynamics response |
| selection invariance | old/new $P_{\rm find}(\theta_E)$, $g(\theta_E,\gamma)$ agree for same physical profile | 检查 selection layer |
| ratio scaling | $M_\ast/M_{Rh}\propto h^{-1}$ | 防止把 fraction 当成 h-free |

### Phase 7 — 正式 rerun 与结果解释

推荐顺序：

1. 先跑一个 minimal model / short chain，验证 log posterior finite、$Z_{\rm norm}$ 稳定、PPC 不崩。
2. 再复刻当前主结果矩阵中最重要的 branch，例如当前 BIC 最优的 profile + gamma-mode 组合。
3. 对比旧结果时分三类：
   - invariant observables：$\theta_E$、$\sigma_{\rm ap}$、$\gamma$；
   - reparameterized quantities：$m_\ast$、$R_e$、$m_R$；
   - changed-aperture scientific quantities：$M(<5\,\mathrm{kpc})$ vs $M(<5h^{-1}\,\mathrm{kpc})$。
4. 最后再跑完整 `devauc/sersic × independent/sigma_star_dependent` matrix，重新计算 BIC/PPC/trends。

---

## 7. 与 KiDS+GAMA-style convention 的对齐方式

KiDS+GAMA SHMR 文献中的实用做法是：stellar mass 用 $h^{-2}M_\odot$，weak-lensing / halo mass 用 $h^{-1}M_\odot$，因此 stellar-to-halo mass ratio 不是纯 h-free dimensionless number，而会带一个显式 $h^{-1}$ 或 $h$ 因子。

你的项目若采用本计划，应该形成完全平行的 convention：

$$
M_\ast = \tilde M_\ast h^{-2}M_\odot,
$$

$$
M_{2D}(<\tilde Rh^{-1}\mathrm{kpc})=\tilde M_{\tilde Rh}h^{-1}M_\odot,
$$

$$
\frac{M_\ast}{M_{\tilde Rh}}
=
\frac{\tilde M_\ast}{\tilde M_{\tilde Rh}}h^{-1}.
$$

所以在图和表中不要写一个裸的 “stellar fraction” 而不说明 $h$。更稳妥的写法是：

$$
h\frac{M_\ast}{M_{\tilde Rh}}
$$

作为 h-free coefficient，或直接在表中写

$$
\frac{M_\ast}{M_{\tilde Rh}} = X\,h^{-1}.
$$

---

## 8. 必须马上做、建议尽快做、后续可优化

### 8.1 必须马上做

1. **确认当前 $h_{\rm ref}$ 与所有上游产品的 unit metadata。** 这是所有转换公式的输入；没有它就无法安全迁移。
2. **决定并冻结新变量名。** 推荐 `mstar_h2`, `re_hinvkpc`, `m5_hinvkpc`, `m10_hinvkpc`，避免 `m5` 同时指 fixed-kpc 和 h-aperture。
3. **修改 $m_R$ 定义。** 先在文档和 config 中写清：
   $$
   m_{\tilde Rh}=\log_{10}\left[M_{2D}(<\tilde Rh^{-1}\mathrm{kpc})/(h^{-1}M_\odot)\right].
   $$
4. **建立 analytic migration tests。** 特别是 $\Sigma_\ast$ invariance、$m_{Rh}$ draw-wise conversion、$\theta_E$ invariance 和 $\sigma_{\rm model}$ invariance。
5. **不要在 convention 未统一前解释 mass ratio、stellar fraction、dark matter fraction。** 这些量在新 convention 下仍然有显式 h scaling。

### 8.2 建议尽快做

1. **重建 raw HDF5 h-units 分支。** 不要直接覆盖旧文件；旧 fixed-kpc branch 应保留为 legacy comparison。
2. **重建 interpolation / Jeans grids。** analytic rescaling 可用于 debug，但正式结果最好来自新 mass definition 下的一致 grid。
3. **重设 stellar-mass 和 size pivots。** 旧 $11.4$ 和 size-relation coefficients 需要确认原始 convention 后再转。
4. **更新 PPC 和 Fig. 8-like scripts。** 所有 axis label、table header、caption 必须与新 unit contract 一致。
5. **增加 Sonnenfeld comparison conversion。** 需要一键把 $m_{5h}$ 转回 $M(<5\,\mathrm{kpc})$ at $h=0.7$，否则与 Sonnenfeld 的 $\mu_{5,0}$ 比较会混淆。

### 8.3 后续可优化

1. **在代码中引入 unit-aware dataclass。** 例如每个 quantity 附带 `unit`, `h_power`, `aperture_type`, `h_ref`, `is_log`。
2. **把所有 figure/table 生成脚本接入统一 label registry。** 防止同一个量在不同图中写法不一致。
3. **输出双 convention appendix。** 主文使用 h-dependent units，附录给固定 $h=0.7$ physical-unit 数值，方便与 Sonnenfeld/SLACS 文献对比。
4. **把 ratio 类量单独建表。** 对 $M_\ast/M_{Rh}$、$M_{Rh}/M_\ast$、$\kappa_\ast$ 都显式写 $h$ scaling，避免读者误以为无量纲量必然 h-free。

---

## 9. 当前最大风险清单

1. **只改图轴标签，不改 mass grid / Jeans grid / normalization。** 这会制造最危险的“表面单位正确、模型内部仍是旧 convention”的错误。
2. **把 fixed $5\,\mathrm{kpc}$ 和 $5h^{-1}\mathrm{kpc}$ 当作同一个 aperture。** 二者在 $h=0.7$ 下相差 43%，不是小的 notation change。
3. **把 stellar mass 和 lensing mass 都叫 mass 后直接相除。** 新 convention 下 $M_\ast\propto h^{-2}$，$M_{Rh}\propto h^{-1}$，ratio 仍带 $h^{-1}$。
4. **沿用旧 stellar-mass pivot。** 旧 $11.4$ 若来自 physical $M_\odot$ at $h_{\rm ref}$，转到 $h^{-2}M_\odot$ 后数值应平移。
5. **忽略外部 size relation 的 h convention。** $\Delta_R$ 是回归变量；如果 $R_e$ 转了但 $\mu_r$ 的 literature coefficients 没转，$\xi_R$ 相关结论会偏。
6. **把 $\Sigma_\ast$ 的 h-free 性质误推广到所有 density。** $\Sigma_\ast$ 是 $h^0$，但 $\Sigma_c$ 和 total $\Sigma$ 是 $h^1$。

---

## 10. 推荐的最小可执行版本（MVP）

若你想尽快推进，而不是一开始就重构全 pipeline，可以先做一个 MVP：

1. 从当前 raw HDF5 读旧 $m_\ast$、$R_e$、$m_R(\gamma)$、`S_unit`。
2. 用解析公式生成 temporary h-unit branch：
   $$
   m_\ast^{(h2)},\quad r_e^{(h)},\quad m_{Rh},\quad S_{\rm unit}^{(h)}.
   $$
3. 在一个小 lens subset 上验证 $\theta_E$ 和 $\sigma_{\rm model}$ 不变。
4. 用这组 temporary branch 跑一条 short chain，检查 posterior 和 PPC 是否稳定。
5. 确认没有 hidden convention bug 后，再重建正式 grids 和 full run。

MVP 的输出不要作为最终科学结论，只作为 convention migration 的 integration test。

---

## 11. 需要补充确认的信息

1. 当前 CMASS pipeline 实际使用的 $H_0$、$h_{\rm ref}$、$\Omega_M$、$\Omega_\Lambda$ attrs 是否在所有 raw/interpolation/output 文件中一致。
2. $R_e$ 上游到底是从哪个 angular size、哪套 cosmology 转成 kpc；外部 size relation coefficients 的 h convention 是什么。
3. `S_unit` grids 的生成是否使用 physical kpc、arcsec aperture、或者 mixed aperture contract；这决定能否安全使用 analytic rescaling，还是必须全量重建。
4. 当前 stellar-mass function / parent population 的 mass bins 是否来自同一 $M_\ast$ catalog convention。
5. Fig. 8-like overlay 的 single-lens flat-prior attrs 是否保存了足够完整的 $m_R(\gamma)$ track；若只保存 quantiles，需要重新从 grid 生成。
6. 是否要在主文中主要报告 $m_{5h}$ 还是 $m_{10h}$。两者都可以，但文稿主线最好只设一个 primary mass definition，另一个放附录或 robustness。

---

## 12. 一句话执行路线

先冻结 `h_units_v1` 的 unit contract；然后在 raw HDF5 中新增 `mstar_h2`、`re_hinvkpc`、`m5_hinvkpc`、`m10_hinvkpc`；接着重建或解析迁移 mass grids 与 `S_unit`；再让 inference、selection normalization、PPC 和 Fig. 8-like scripts 全部读新变量；最后用 $\theta_E$、$\sigma_{\rm model}$、$\Sigma_\ast$ invariance tests 证明这次改动是 convention migration，而不是悄悄改了物理模型。
