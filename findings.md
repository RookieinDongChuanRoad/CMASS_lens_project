# Sonnenfeld reference 阅读发现

## 已知上下文

- 当前 worktree：`/Users/liurongfu/.codex/worktrees/b283/CMASS_lens_project`
- 外部 reference：`/Users/liurongfu/reference_codes/strong_lensing_tools/papers/slacs_selection`
- 本地生产路径：canonical dataset -> ModelSpec/RuntimeAdapter -> model-owned Numba posterior -> emcee。
- 需要先区分 `sonnenfeld2024_slacs` 与 `sonnenfeld2024_slacs_hunit`；reference 语义优先对应 paper-native fixed-kpc/m5。

## 初步文件盘点

- 主 reference 推断脚本：`scripts/fit_full.py`
- no-selection 对照：`scripts/fit_slonly.py`
- 参数常数：`scripts/fitpars.py`、`scripts/parent_sample_pars.py`
- SLACS 表读取：`scripts/read_slacs.py`
- cross-section grid：`scripts/make_crosssect_grid.py`
- SLACS lensing/dynamics grid：`scripts/make_slacs_lensing_grids.py`、`scripts/make_slacs_jeans_grids.py`
- 已落盘 inference 工件：`full_inference.hdf5`、`slonly_inference.hdf5`、`nofpprior_inefrence.hdf5`

## `fit_full.py` 的关键结构

- `fit_full.py` 是 selection-corrected 主路径：读取 `fibre_crosssect_grid.hdf5` 的 `mufibre3_cs_grid`，读取 `slacs_lensing_grids.hdf5` / `slacs_jeans_grids.hdf5` 的 per-lens grid。
- 参数顺序是 `mu_m5, sigma_m5, beta_m5, xi_m5, mu_gamma, sigma_gamma, beta_gamma, xi_gamma, mu_zs, sigma_zs, t_find, la_find`。
- parent normalization 用 `npop=10000` 个 parent sample draw，选择权重是 `cs_popsamp * pfind(tein_est_popsamp, t_find, 10**la_find)`。
- per-lens likelihood 用 `nis=1000` 个 importance sample；`gamma` 在每个 lens、每个 theta 下从 truncated normal 重抽，不是固定 gamma quadrature。
- full posterior 额外加 fundamental-plane prior：`fpfit_scat`、`mu_v_prior`、`beta_v_prior` 三项。
- 因此 reference full `logpfunc` 不是当前本地 deterministic Numba posterior 的逐点同构 oracle；需要先拆 primitive/grid/term 级比较，再决定是否构建一个 reference-compatible adapter。

## `fit_slonly.py` 的关键结构

- no-selection 对照没有 parent normalization、cross-section、source-redshift和 lens finding probability。
- 它的参数空间也不同：额外拟合 `mu_ms, sigma_ms`，不等同于当前本地 Sonnenfeld full model。
- 适合作为 per-lens lensing/dynamics grid 与 `m5/gamma/sigma` 项的局部参考，不适合作为完整 Sonnenfeld full posterior oracle。

## 待确认

- 已落盘 HDF5 是否包含 chain、lnprob、hyperparameter dataset，可直接抽 theta 做 oracle。
- `make_slacs_lensing_grids.py` 和当前 canonical writer 的 per-lens `m5_grid/dm5drein_grid/mufibre3_cs_grid` 是否同源同单位。
- 本地 current implementation 是否已经保留/禁用 FP prior；若当前没有 FP prior，则 full posterior 比较必须单独关掉或拆项。

## grid 与依赖状态

- reference 根目录只包含最终 `full_inference.hdf5` / `slonly_inference.hdf5` / PP HDF5 / `parent_sample.fits` / `SLACS_table.cat`；`scripts/` 下没有现成 `fibre_crosssect_grid.hdf5`、`slacs_lensing_grids.hdf5`、`slacs_jeans_grids.hdf5`、`rein_grid.hdf5`、`sigma2_grid.hdf5`。
- 当前 `cmass_lens` 环境能导入 `h5py`、`emcee`、`astropy`、`scipy`、`ndinterp`。
- reference 脚本还需要把 `/Users/liurongfu/reference_codes/strong_lensing_tools` 和 `/Users/liurongfu/tools` 加入 `PYTHONPATH`，之后 `sl_cosmology`、`sl_profiles`、`spherical_jeans` 都能导入。
- 本地 `/Users/liurongfu/Work/CMASS_lens_project/data/external` 当前没有可直接读取的 Sonnenfeld canonical HDF5；`b283` worktree 下也没有现成 canonical Sonnenfeld inference dataset。

## grid 生成语义

- reference `make_crosssect_grid.py` 生成全局 finite-fibre `mufibre3_cs_grid(theta_E, gamma)`，阈值是 total seeing-convolved fibre flux `>3` 且 `muB > 1`。
- reference `make_slacs_lensing_grids.py` 对每个 SLACS lens 和 gamma 生成 `m5_grid`、`dm5drein_grid`、以及该 lens fixed `theta_E` 下的 `mufibre3_cs_grid(gamma)`。
- reference `make_slacs_jeans_grids.py` 生成 per-lens unit-M5 的 `s2_grid(gamma)`，之后 likelihood 用 `sqrt(10**m5 * s2_grid)` 得到 velocity dispersion。
- 本地 `prepare_dataset` 已经迁入 Sonnenfeld finite-fibre cross-section 生成，并且 canonical writer 会直接保留 `mufibre3_cs_grid`，不是 CMASS ratio-based 重算。

## 本地 comparison 复用点

- `Bayesian_inference/scripts/compare_log_prob_with_main.py` 已提供一个可复用模式：每个被比较实现运行在独立 subprocess 中，用独立 `PYTHONPATH`，输出 JSON + Markdown report。
- 这个 harness 当前只支持 CMASS synthetic main-vs-candidate，不支持外部 Sonnenfeld reference。
- `Bayesian_inference/tests/test_sonnenfeld_runtime_model.py` 当前只覆盖 synthetic canonical smoke、unit convention、参数 schema、finite log-prob；它明确不是 paper/reference validation。

## 已识别的不等价点

- 参数命名/顺序不同：reference chain 是 `mu_m5, sigma_m5, beta_m5, xi_m5, ... t_find, la_find`；本地 theta 是 `mu5_0, beta5, xi5, sigma5, ... theta0, loga`。
- reference full posterior 有 FP prior；本地 Sonnenfeld posterior 当前 timing blob 中 `fp_prior_log_term=0`，没有把 FP prior 加进 Sonnenfeld posterior。
- reference per-lens likelihood 用 `gamma` importance sampling；本地用 deterministic `gamma_grid_int` trapezoid。
- reference normalization 对 `z_s` 用非负 truncated normal draw；本地 `_draw_population_state` 当前是 ordinary Gaussian draw 后用 `zs > zd` 过滤。
- reference parent `dV/dz` 用 `comovd(z)^2*dcomovdz(z)`；本地 `_parent_density_grid` 目前是 `z_d^2` proxy。
- reference per-lens selection 用 observed `slacs_tein_est` 的 `pfind`，本地 per-lens path若有观测 sigma也会用第一条 observed sigma 估计 `theta_est`；这个点有机会做局部等价测试。

## 推荐比较策略

- 不要一上来比较 `full_inference.hdf5/logp` 与本地 `log_prob`；这会把 FP prior、MC 近似、parent density、source-z proposal 和 grid 差异全部混在一起。
- 先做 deterministic 数据准备层 oracle，再做 scalar primitive oracle，再做 fixed-grid per-lens term oracle，最后才做 full posterior trend-level sanity check。

## 2026-05-09: proposal / parent 分布专项审计发现

- Reference `fit_full.py` 的 parent sample 不是 proposal importance sample，而是通过 `mz_distribution.draw_mz(npop)` 直接从 `p(z_d, M_*)` 抽样；FP fit 也直接使用这个 parent sample 中 `M_* > 11` 的子样本。
- Reference 的 `p(z_d, M_*)` 明确包含 `dV/dz = comovd(z)^2 * dcomovdz(z)`、arctan completeness、Schechter 项；本地 Sonnenfeld 当前仍使用 `z_d^2` proxy。
- 本地 `_draw_population_state` 当前把 `z_d, M_*` 从 ad-hoc truncated-normal proposal 抽出来，再用 `parent_density / proposal_density` 修正 normalization 和 FP summary。这在数值积分上可以解释为 importance sampling，但不是 reference 的科学结构；proposal 变量已经泄漏进模型实现，应该移出 posterior 科学公式层。
- Reference normalization 的 source-redshift draw 是 non-negative truncated Gaussian，并且 good mask 要求 `z_s > z_d + 0.05`、`z_s < z_s_max`、`z_s > z_min`、质量/grid/半径边界均在 reference grid 内；本地 normalization 目前只有 `z_s > z_d` 等较弱条件，且 sigma grid interpolation 使用 clipping。
- Reference FP prior 常数来自 `fitpars.py`：`fiducial_fpscat=0.047`、`err_fpscat=0.008`、`mu_v_prior=2.341871...`、`err_mu_v=0.03`、`beta_v_prior=0.257740...`、`err_beta_v=0.03`。本地 `FPPriorConfig` 默认值仍是另一套更紧的数值，需要模型特异化或修正默认来源。
