# Sonnenfeld Reference 语义修正最终目标

本文档记录本次 Sonnenfeld/SLACS 科学模型语义修正的最终目标。文档建立后不可修改；后续执行进度只记录在 `sonnenfeld_reference_semantics_progress.md`。

## 背景

当前工作区的 Sonnenfeld 本地实现已经接入 numba/emcee production backend，但在 normalization 与 FP prior 修正过程中暴露出一个不可接受的问题：本地 posterior 使用了 ad-hoc truncated-normal proposal 来抽取 `(z_d, M_*)`，再用 `parent_density / proposal_density` 做 importance weighting。

这个 proposal 可以被解释为数值积分技巧，但它不是 Sonnenfeld reference implementation 的科学模型语义。用户已明确要求：本地 Sonnenfeld 模型必须忠于 reference 中原定的 `p(z_d, M_*)` parent distribution，不允许用 ad-hoc proposal 替换或污染科学公式。

## 最终目标

本地 `sonnenfeld2024_slacs` 的科学语义必须与外部 reference `/Users/liurongfu/reference_codes/strong_lensing_tools/papers/slacs_selection/scripts/fit_full.py` 对齐：

1. Parent population
   - `(z_d, M_*)` 必须来自 reference 的 parent distribution `p(z_d, M_*)`。
   - 分布结构必须包含 `dV/dz = comovd(z)^2 dcomovdz(z)`、arctan completeness、Schechter 项。
   - posterior hot path 不得再出现 `proposal_density` 或 `parent_density / proposal_density` 作为 Sonnenfeld 科学模型的一部分。

2. Parent sample boundary
   - reference 的 `mz_distribution.draw_mz(npop)` 语义应映射为本地 preprocessing/context 的确定性 parent sample 或等价 CDF/grid 产物。
   - runtime posterior 只消费已经按物理 parent distribution 准备好的 sample/grid。
   - 若为了 reproducibility 使用固定随机数或 quasi-random basis，也必须先映射到 `p(z_d, M_*)`，不能在 posterior 中临时构造 ad-hoc proposal。

3. Source-redshift and normalization mask
   - normalization 的 source-redshift draw 必须使用 reference 的 non-negative truncated Gaussian 语义。
   - good mask 必须包括 reference 条件：`z_s > z_d + 0.05`、`z_min < z_s < z_s_max`、mass/gamma/size/dynamics/lensing grid 边界。
   - grid 外推不能通过 clipping 伪造有效物理值；Sonnenfeld reference 语义下 grid 外应进入 rejection 或 zero contribution。

4. Fundamental Plane prior
   - FP fit 必须作用在 parent population sample 中满足 `M_* > 11` 的子样本上，而不是 selected lens sample，也不是 proposal sample。
   - FP 关系为 `log10(sigma) = mu + beta * (M_* - mpiv_slacs) + xi * delta_r`。
   - prior 只作用于 scatter、`mu`、`beta`；`xi` 仅作为 diagnostic。
   - Sonnenfeld 默认 FP prior 常数必须来自 reference `fitpars.py`：`fiducial_fpscat=0.047`、`err_fpscat=0.008`、`mu_v_prior≈2.341871`、`err_mu_v=0.03`、`beta_v_prior≈0.257740`、`err_beta_v=0.03`。不能用 CMASS 或其他模型默认值代替 Sonnenfeld reference 常数。

5. Per-lens likelihood
   - Per-lens likelihood 必须使用与 reference 一致的 parent `M_*` density、size relation、`m5` relation、gamma relation、source redshift density、Jacobian、selection cross-section 和 discovery probability。
   - 如果本地实现保留 deterministic quadrature 以替代 reference 的 stochastic importance sample，必须证明积分目标一致，且测试覆盖公式项。

6. Cross-section and velocity-dispersion grids
   - Sonnenfeld 使用 finite-fibre `mufibre3_cs_grid(theta_E, gamma)`，不能套用 CMASS separable/ratio cross-section 公式。
   - Population sigma grid 与 per-lens `s2` grid 必须按 reference 的有效定义和 grid 边界使用。

## 执行计划

1. 建立测试护栏
   - 添加测试证明 Sonnenfeld posterior 不再依赖 `proposal_density`。
   - 添加 parent density scalar/grid 测试，对齐 reference `mz_distribution.msdist` 结构。
   - 添加 source-z truncation 与 good-mask 测试。
   - 添加 FP prior 常数与二维 FP fit 测试。

2. 移除 proposal 语义
   - 从 Sonnenfeld posterior 中删除 ad-hoc `(z_d, M_*)` proposal 生成与 `parent/proposal` 权重。
   - 在 preprocessing/context 中准备 reference-faithful parent sample 或等价 CDF/grid。
   - posterior kernels 只消费 parent sample arrays。

3. 修正 parent population 公式
   - 用 cosmology table 或 preprocessing helper 计算 `comovd(z)^2 dcomovdz(z)`。
   - 统一 per-lens `parent_mstar_density_grid` 与 normalization 使用的 parent distribution。

4. 修正 normalization 条件
   - source-z 改为 non-negative truncated Gaussian。
   - 补齐 reference good mask 与 grid-bound checks。
   - 对 Sonnenfeld grid 外值禁止 clipping 成有效贡献。

5. 修正 FP prior
   - 引入 Sonnenfeld reference defaults，避免全局影响 CMASS。
   - FP summary 从 parent sample 直接累积，去掉 proposal weight。

6. 验证
   - 运行 Sonnenfeld runtime tests。
   - 运行新增 reference semantic tests。
   - 运行受影响 shared kernel tests。
   - 执行 `git diff --check`。

## 非目标

- 不在本次任务中重写 data preparation pipeline。
- 不在本次任务中修改 CMASS 科学模型。
- 不在本次任务中处理 Posterior_predictive_test 中已有脏改动。
- 不把 full `fit_full.py` chain/logp 直接当逐点 oracle；完整数值比较应在语义修正之后分层建立。
