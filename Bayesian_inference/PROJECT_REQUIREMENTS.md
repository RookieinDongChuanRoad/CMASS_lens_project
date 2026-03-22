# CMASS Lens 项目需求总规范（可直接用于实现）

本文件给出本项目的完整技术要求，目标是让开发者无需回看参考文献即可实现同等模型。
内容包括: 从文献抽取的结构和参数、本项目相对文献的改动、数据格式、似然分支、归一化积分、采样设置、性能约束和输出规范。

## 1. 项目目标

构建一个选择效应修正的强透镜层次贝叶斯推断框架，比较两种光度分布模型对超参数后验的影响:

- `devauc` 分支（de Vaucouleurs）
- `sersic` 分支（Sersic）

两分支共享同一套超参数定义、先验范围、采样器与输出格式，仅在结构参数先验和质量函数固定参数上不同。

## 2. 参考文献与“抽取后可直接用”的内容

参考文献原文文件位置（本机）:

- `/Users/liurongfu/Zotero/storage/98E4LV5X/Sonnenfeld - 2024 - The SLACS strong lens sample, debiased.pdf`
- `/Users/liurongfu/Zotero/storage/23KA6L67/Sonnenfeld 等 - 2019 - Hyper Suprime-Cam view of the CMASS galaxy sample. Halo mass as a function of stellar mass, size, an.pdf`

## 2.1 Sonnenfeld (2024) 抽取内容（主框架）

保留的核心结构:

- 选择修正的层次模型框架（单镜头似然与样本归一化相除的形式）。
- `P_find`（透镜被找到的效率函数）逻辑和函数形式。
- `P_s^{eff}(z_s)` 作为 source redshift 的有效分布项（高斯 + 非负截断）。
- 在给定 `theta_E, z_d, z_s` 下使用 `m_R(gamma)` 轨迹，并在积分中使用雅可比项。

本项目对 2024 框架的明确修改:

- Fundamental Plane 先验改为可选项，默认关闭；开启时复用当前 CMASS parent-population normalization Monte Carlo，不回退到 legacy `mz_distribution.draw_mz(...)` 抽样框架。
- 开启 Fundamental Plane 先验时，必须提供与当前 profile 和质量定义匹配的 sigma-unit table（`jeans_*_{m5,m10}_grid.h5`）。
- 不引入 source brightness 项。
- 不在群体分布里显式建模 redshift 的复杂联合分布；与 redshift 相关的观测近似为精确值（单镜头似然里直接用观测 `z_d,z_s`）。
- 样本归一化使用 Monte Carlo（而不是高维解析/全网格积分）。

## 2.2 Sonnenfeld et al. (2019) 抽取内容（结构参数先验）

### de Vaucouleurs 分支（参考 2019 Sec.3.1 + Table 1）

固定使用:

- `n = 4`
- 质量函数（skew-normal）参数:
- `mu_star = 11.252`
- `sigma_star = 0.202`
- `alpha_star = 10**0.17 = 1.4791083881682073`
- 尺度关系参数:
- `muR0 = 0.774`
- `betaR = 0.977`
- `sigmaR = 0.112`

### Sersic 分支（参考 2019 Sec.3.2 + Table 2）

固定使用:

- 质量函数（skew-normal）参数:
- `mu_star = 11.249`
- `sigma_star = 0.285`
- `alpha_star = 10**0.43 = 2.691534803926914`
- `log n | m*` 关系参数:
- `mu_n0 = 0.704`
- `beta_n = 0.464`
- `sigma_n = 0.163`
- `log r_e | m*, n` 关系参数:
- `muR0 = 0.817`
- `betaR = 1.184`
- `sigmaR = 0.133`
- `nuR = 0.383`

## 3. 全局物理与宇宙学设定

- `H0 = 70.0 km/s/Mpc`
- `Omega_m = 0.3`
- 距离插值表（内部实现常量，不作为用户运行时配置项暴露）:
- `z_table_max = 5.0`
- `z_table_size = 8001`

Einstein 半径计算（线性幂律质量模型，与你给定代码一致）:

- 质量定义记作 `m_R = log10 M_2D(<R)`。
- 当前实现仅支持 `R = 5 kpc` 和 `R = 10 kpc`。

- `Sigma_c = c^2/(4*pi*G) * Ds/(Dl*Dls)`
- `r_ein = (10**m_R / (pi*Sigma_c*R**(3-gamma)))**(1/(gamma-1))`
- `theta_ein = r_ein/Dl * 206265 (arcsec)`
- 当 `z_d >= z_s` 时，`theta_ein = 0`

## 4. 模型变量与超参数

群体超参数共 12 个（两分支共用）:

- 内部实现使用一组通用 mass hyper-parameters。
- 外部公共命名随质量定义切换:
- `R=5 kpc`: `mu5_0, beta5, xi5, sigma5`
- `R=10 kpc`: `mu10_0, beta10, xi10, sigma10`

1. `mu5_0`
2. `beta5`
3. `xi5`
4. `sigma5`
5. `mu_gamma_0`
6. `beta_gamma`
7. `xi_gamma`
8. `sigma_gamma`
9. `mu_zs`
10. `sigma_zs`
11. `theta0`
12. `loga`

先验范围:

- `mu5_0 in [9, 12]`
- `beta5 in [-3, 3]`
- `xi5 in [-3, 3]`
- `sigma5 in [1e-2, 0.2]`
- `mu_gamma_0 in [1.5, 2.5]`
- `beta_gamma in [-3, 3]`
- `xi_gamma in [-3, 3]`
- `sigma_gamma in [0, 0.5]`
- `mu_zs in [1, 3]`
- `sigma_zs in [0, 2]`
- `theta0 in [0, 3]`
- `loga in [-1, 3]`

严格正值约束:

- `sigma5 > 0`
- `sigma_gamma > 0`
- `sigma_zs > 0`

## 5. 固定分布与函数形式

## 5.1 恒星质量函数 `S(m*)`（skew-normal）

定义 `m* = log10(M*/Msun)`，采用:

`S(m*) = 2/scale * phi((m*-loc)/scale) * Phi(alpha*(m*-loc)/scale)`

其中:

- `phi` 是标准高斯 PDF。
- `Phi(x)=0.5*(1+erf(x/sqrt(2)))`。
- 与你提供实现完全一致（`skewnormpdf`）。
- 参数 `loc, scale, alpha` 按 profile 固定，非推断参数。

## 5.2 source redshift 分布

- `P_s^{eff}(z_s)` 形式保持高斯并做 `z_s >= 0` 截断。
- 截断区间: `[0, +inf)`。
- 在单镜头似然中使用观测 `z_s` 直接代入该分布。

## 5.3 deflector redshift 分布（只在归一化中积分）

- `P(z_d) = N(mu_d=0.558, sigma_d=0.085)`

## 5.4 发现概率 `P_find`

采用 sigmoid:

- `P_find = 1 / (1 + exp(-a*(theta_E - theta0)))`
- `a = 10**loga`

## 5.5 截面因子 `g`

- 从 `cs_grid_power.h5` 的 `compressed_grids` 读取 `gamma_grid` 与 `cs_over_theta_ein`。
- `cs_over_theta_ein` 视作 `gamma` 的一维函数，插值采用边界截断（clip）。
- `g = pi * (cs_over_theta_ein(gamma) * theta_ein)^2`

## 6. 结构关系（devauc 与 sersic）

## 6.1 `mu_R(m*, n)` 的定义

- 基础项: `muR0 + betaR*(m* - 11.4)`
- sersic 分支额外项: `nuR*(log10(n)-log10(4))`
- devauc 分支: `n=4`，因此无额外 `n` 依赖项

你确认的核心约束:

- 仅在 `mu_R(m*, n)` 中引入 `n` 依赖。
- `P(m_R,gamma | m*,r_e)` 不再额外显式加 `n` 项。

## 6.2 `n` 的使用策略

- `devauc`: 固定 `n=4`。
- `sersic`:
- 单镜头似然使用每个镜头观测 `n_ser`。
- 样本归一化中从 `log n | m*` 正态关系采样 `n`。

## 6.3 `m_R` 与 `gamma` 的条件高斯模型（两分支共用）

定义:

- `mu_R,struct = muR0 + betaR*(m* - 11.4)`（sersic 再加 `nuR*(log10(n)-log10(4))`）
- `Delta_R = log10(r_e[kpc]) - mu_R,struct`

则:

- `mu_mass(m*,r_e,n) = mu_mass_0 + beta_mass*(m* - 11.4) + xi_mass*Delta_R`
- `mu_gamma(m*,r_e,n) = mu_gamma_0 + beta_gamma*(m* - 11.4) + xi_gamma*Delta_R`
- `m_R | m*,r_e,n ~ N(mu_mass, sigma_mass^2)`
- `gamma | m*,r_e,n ~ N(mu_gamma, sigma_gamma^2)`

说明:

- 当 `R=5 kpc` 时，公共参数名对应 `mu5_0/beta5/xi5/sigma5`。
- 当 `R=10 kpc` 时，公共参数名对应 `mu10_0/beta10/xi10/sigma10`。
- 两个定义之间满足精确关系 `m10 = m5 + (3-gamma) log10(2)`。

## 7. 单镜头似然定义（核心可执行版）

每个 lens 的被积变量为 `gamma` 与 `m*`（二维积分），其中 `m_R` 不再独立积分，而是通过观测条件约束为 `m_R(gamma)`。

输入数据给出每个 lens 的 17 点网格:

- `gamma_grid_17`
- `mass_grid_17`
- `dmass_dthetaein_grid_17`
- 可选 `s2_grid_17`

实现要求:

- 在积分时将 `gamma` 从 17 点提升到 200 点细网格（线性插值，clip）。
- 使用雅可比 `|dm_R/dthetaein|`（绝对值）。
- **不要**因为 `mass_grid` 单调递减而重新排序；这是可物理出现的结果。

单镜头被积项包含:

- deflector redshift 项 `P(z_{d,i})`
- 观测恒星质量项 `P(logm_obs | m*)`
- 恒星质量函数 `S(m*)`
- 尺度关系项 `P(log r_e | m*, n)`
- `P(m_R | m*, r_e, eta)`
- `P(gamma | m*, r_e, eta)`
- `P_s^{eff}(z_s | eta)`
- `P_find(theta_E | eta)`
- 截面项 `g(theta_E,gamma)`
- 雅可比 `|dm_R/dthetaein|`
- 若有速度弥散观测，再乘速度弥散项

可直接实现的单镜头积分形式:

- `L_i(eta) = integral_{gamma} integral_{m*} W_i(gamma,m*|eta) dm* dgamma`
- `W_i = P(logm_obs|m*) * S(m*) * P(logr_e|m*,n_i) * P(m_R(gamma)|m*,r_e,eta) * P(gamma|m*,r_e,eta) * P_s^{eff}(z_{s,i}|eta) * P_find(theta_E(gamma)|eta) * g(gamma) * |dm_R/dthetaein| * P_sigma`
- `W_i = P(z_{d,i}) * P(logm_obs|m*) * S(m*) * P(logr_e|m*,n_i) * P(m_R(gamma)|m*,r_e,eta) * P(gamma|m*,r_e,eta) * P_s^{eff}(z_{s,i}|eta) * P_find(theta_E(gamma)|eta) * g(gamma) * |dm_R/dthetaein| * P_sigma`
- 其中 `P_sigma` 按 `num_sigma` 分支处理（见第 8 节）。

## 8. 速度弥散 likelihood 分支

数据按 `num_sigma` 分三类:

- `num_sigma = 0`: 不乘速度弥散项
- `num_sigma = 1`: 乘一个高斯项
- `num_sigma = 2`: 乘两个独立高斯项（同一 `sigma_model`，不同观测误差）

`s2` 的定义与你确认一致:

- `s2 = sigma^2 / 10**m_R`
- 因此 `sigma_model = sqrt(s2 * 10**m_R)`
- 先对 `s2_grid(gamma)` 插值，再乘 `10**m_R(gamma)` 开方

速度弥散项采用线性空间高斯:

- `N(sigma_obs | sigma_model, sigma_err^2)`

## 9. 样本归一化（selection normalization）

归一化是高维积分，策略固定为 MC 积分，每个参数步都重算:

- `N_norm = 1e5`
- 每一步 `eta` 更新时重算一次
- 使用固定 base random normals，避免同一 `eta` 的数值不一致

MC 抽样变量:

- `z_d ~ N(0.558,0.085)`
- `z_s ~ N(mu_zs,sigma_zs)` 且 `z_s > 0`
- `m* ~ skew-normal(mu_star,sigma_star,alpha_star)`
- `n`（仅 sersic 归一化时）从 `log n|m*` 抽样
- `gamma ~ TruncNormal(mu_gamma, sigma_gamma; [1.2, 2.8])`（对应 `truncnorm_rvs(1.2, 2.8)` 约束）
- `r_e, m5` 由对应条件正态抽样

归一化下限规则:

- 若 `Z_norm <= 1e-10`，该 `eta` 直接拒绝（`log_prob = -inf`）

总后验（忽略常数项）:

- `log_prob(eta) = sum_i log L_i(eta) - N_lens * log Z(eta) + log Prior(eta)`
- `Prior(eta)` 为第 4 节给定 box prior（并叠加严格正值约束）。

## 10. 数值离散化与插值

- `gamma` 单镜头积分点数: `200`
- `m*` 单镜头积分点数: `200`
- `m*` 积分范围: `logm_obs +- 5*logm_err`
- 积分方法: 梯形权重
- 插值: 一维线性插值 + 边界 clip

## 11. 数据文件与结构规范

## 11.1 观测文件（按 lens group 组织）

必须可读取的 attrs/datasets:

- attrs:
- `zd, zs`
- `logmchab, logmchab_err`
- `nser`
- `re_arcsec, rein_arcsec`
- `num_sigma`
- 可选 `sigma, sigma_err`
- 可选 `aperture_width`
- datasets:
- `gamma_grid`
- `mass_definitions/m5/mass_grid`
- `mass_definitions/m5/dmass_dthetaein_grid`
- `mass_definitions/m10/mass_grid`
- `mass_definitions/m10/dmass_dthetaein_grid`
- 可选 `mass_definitions/<label>/s2_grid`
- 过渡期兼容旧 root-level `m5_grid/dm5_dthetaein_grid/s2_grid`

数据现状:

- `observations_deV_with_mass_grids.hdf5` 用于 `devauc`
- `observations_with_mass_grids_all.hdf5` 用于 `sersic`
- `devauc` 分支读取字段约定:
- stellar mass 读 `logmchab_deV`（若缺失则回退 `logmchab`）
- effective radius 读 `reff_deV`（单位 `arcsec`，再统一转换为 `log10(kpc)` 进入模型）

## 11.2 截面文件 `cs_grid_power.h5`

仅使用:

- group: `compressed_grids`
- dataset: `gamma_grid`（或 `gamma_grids`）
- dataset: `cs_over_theta_ein`（或 `cs_over_theta_ein_grid`）

## 12. 推断与运行配置

采样器固定:

- `emcee`
- `n_walkers = 24`
- `n_steps = 10000`
- `warmup = 2000`（可在实现中配置，但当前标准为 2000）
- 需要保存 checkpoint
- 最低输出要求: 保留完整 chain

初始化策略:

- 使用高斯抖动初始化（围绕给定中心）
- 当前中心值:
- 质量定义必须显式写入配置:
- `mass_definition.enclosed_radius_kpc = 5` 或 `10`
- `mu5_0 = 11.32`
- `beta5 = 0.59`
- `xi5 = -0.11`
- `sigma5 = 0.06`
- `mu_gamma_0 = 1.99`
- `beta_gamma = 0.1`
- `xi_gamma = -0.67`
- `sigma_gamma = 0.149`
- `mu_zs = 1.8`
- `sigma_zs = 0.215`
- `loga = 1.0`
- `theta0 = 0.93`

## 13. 性能要求（强约束）

必须优先优化运行速度:

- 热点函数使用 `numba.njit`。
- 使用并行（线程优先，避免过度多进程竞争）。
- 插值选择与 `numba` 兼容的实现（1D 线性插值）。
- 运行时可见进度条（`tqdm` / emcee progress）。
- 文件锁策略要避免 HDF5 多进程/多读冲突（必要时禁用 file locking）。

## 14. 输出与文件管理要求

每次 run 生成独立目录:

- `runs/<run_id>/config_snapshot.yaml`
- `runs/<run_id>/metadata.json`
- `runs/<run_id>/chain.h5`
- `runs/<run_id>/checkpoints/latest_coords.npy`
- `runs/<run_id>/checkpoints/latest_log_prob.npy`
- `runs/<run_id>/checkpoints/latest_step.txt`
- `runs/<run_id>/run_result.json`

`devauc` 与 `sersic` 输出结构保持一致，方便后续并行对比。

## 15. 不再保留为“开放问题”的项（已定）

- 归一化积分采用 MC，不使用全网格高维积分。
- `N_norm = 1e5`。
- 每步都重算归一化。
- `gamma` 和 `m*` 积分均用 200 点。
- 使用 `|dm_R/dthetaein|`。
- 不引入 source brightness。
- Fundamental Plane 先验保留为 optional prior，默认关闭；若显式开启，则通过同一轮 population MC 额外拟合
  `log10(sigma_model) = a + b (m* - 11.3) + c Delta_r`，并只对 `a`、`b`、`scatter` 加全局先验惩罚。

## 16. 术语对照

- `d` = deflector（前景透镜）
- `s` = source（背景源）
- `eta` = 12 维群体超参数向量
- `m_R` = 幂律模型质量归一参数（通过 `theta_E` 约束成 `m_R(gamma)`，当前支持 `m5` 与 `m10`）
