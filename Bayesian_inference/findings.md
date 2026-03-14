# Findings & Decisions

## Requirements
- 项目目标是构建带选择效应修正的强透镜层次贝叶斯推断框架，用于比较 `devauc` 与 `sersic` 两种光度分布模型对超参数后验的影响。
- 两个分支共享同一套 12 维超参数、box prior、`emcee` 采样配置、输出目录结构和运行结果格式。
- 两个分支仅在结构参数先验和固定质量函数参数上不同；`devauc` 固定 `n=4`，`sersic` 在单镜头 likelihood 用观测 `n_ser`，在归一化 MC 中采样 `n`。
- 主框架承接 Sonnenfeld (2024) 的选择修正层次模型，但删除 Fundamental Plane 先验、不引入 source brightness、不显式建模复杂 redshift 联合分布，并将样本归一化改为 Monte Carlo。
- 单镜头 likelihood 的积分变量固定为 `gamma` 与 `m*`；`m5` 不单独积分，而是通过观测轨迹约束为 `m5(gamma)`。
- 单镜头积分必须使用 17 点观测网格插值得到 200 点 `gamma` 细网格，并显式乘入雅可比 `|dm5/dthetaein|`。
- `num_sigma` 决定速度弥散 likelihood 分支数量：`0` 无该项，`1` 一个高斯，`2` 两个独立高斯。
- 归一化积分固定使用 MC，每个超参数步都重算一次，`N_norm = 1e5`，且同一步内要使用固定 base random normals 保持数值一致性。
- 当 `Z_norm <= 1e-10` 时，本步参数直接拒绝，`log_prob = -inf`。
- 性能约束是强制性的：热点函数需 `numba.njit`，插值实现需与 `numba` 兼容，并提供可见进度条。
- 输出必须按 `runs/<run_id>/...` 生成独立目录，并包含配置快照、metadata、完整 chain、checkpoint 和最终结果文件。

## Research Findings
- 本项目目录当前仅包含需求文档 [PROJECT_REQUIREMENTS.md](/Users/liurongfu/Work/CMASS_lens_project/Bayesian_inference/PROJECT_REQUIREMENTS.md)，尚未开始实现代码搭建。
- `conda env list` 已确认本机存在 `cmass_lens` 环境，路径为 `/opt/homebrew/anaconda3/envs/cmass_lens`。
- `planning-with-files` 要求的三份持久化文件应写在项目根目录，而不是 skill 安装目录。
- 项目需求已经把若干“开放问题”强行收束为固定决定，例如：MC 归一化、每步重算、200 点积分、使用 `|dm5/dthetaein|`、不引入 source brightness、删除 FP 先验。
- 当前工作目录不是 Git 仓库，因此不能使用 `git worktree`；实现阶段将直接在当前目录进行，Git 相关元数据只能尽力采集。
- 真实数据文件位于 `/Users/liurongfu/Work/CMASS_lens_project/data` 下，关键文件均已存在：`data/raw/observations_deV_with_m5_grids.hdf5`、`data/raw/observations_with_m5_grids_all.hdf5`、`data/external/cs_grid_power.h5`。
- `cs_grid_power.h5` 的真实字段名是 `compressed_grids/gamma_grids` 和 `compressed_grids/cs_over_theta_ein_grid`，因此读取层必须实现别名兼容。
- 两个 observation HDF5 的 group attrs 真实字段包含 `zd`、`zs`、`logmchab`、`logmchab_err`、`nser`、`num_sigma`、`re_arcsec`、`rein_arcsec`；`devauc` 文件额外有 `logmchab_deV`、`reff_deV`。
- 速度弥散相关数据在 23 个 lens 中有 7 个：`num_sigma` 分布为 `{0:16, 1:4, 2:3}`，相关 attrs 为 `sigma`、`sigma_err`，数据集为 `s2_grid`。
- `cmass_lens` 环境当前缺少 `emcee`，其余核心依赖检查尚未暴露阻塞。
- 当前已实现的源码骨架包含 `config.py`、`profiles.py`、`io.py`、`cosmology.py`、`distributions.py`、`interpolation.py`、`likelihood.py`、`normalization.py`、`sampler.py`、`outputs.py`、`runner.py` 和 `cli.py`，并且统一挂在 `src/cmass_lens_inference/` 下。
- 默认配置文件已建立在 `configs/devauc.yaml` 与 `configs/sersic.yaml`，都把输出根目录固定到 `/Users/liurongfu/Work/CMASS_lens_project/outputs`。
- 输出层已经实现 `profile/run_id` 目录树、`latest` 指针、`config_snapshot.yaml`、`metadata.json`、`chain.h5`、checkpoint 文件、`run_result.json` 以及 `logs/run.log`。
- 当前测试策略使用合成 HDF5 数据做最小闭环验证，因此能够快速验证结构正确性，但还没有做真实 23 lens、10000 步配置的长程性能验证。
- 并行实现现已落地到运行时配置、parallel policy、`emcee` walker 进程池、阶段耗时 blob 和 `run.log` 摘要输出。
- 当前实现的 `kernel_only` 主要通过单进程线程池并行 lens likelihood 求和；`process_pool` 通过 `spawn` 进程池并行 walker，并在 worker 初始化时钳制 BLAS/OpenMP 线程数，避免过度竞争。
- 真实数据 smoke 已验证 `process_pool` 模式可以运行，输出单个 `tqdm` 进度条，并在终端与 `run.log` 中写出 `step ... | lp ... | lens ... | norm ... | workers ... | strategy ...` 摘要行。
- `cmass_lens` 环境当前有 `emcee` 与 `tqdm`，但仍缺 `numba`；当前代码对 `numba` 线程设置做了软回退，因此不会阻塞运行，但也意味着这版内核优化主要依赖进程/线程调度而非 `numba` JIT。
- `numba` 现已安装到 `cmass_lens`，版本为 `0.64.0`。
- likelihood 与 normalization 的热点计算现已替换为真实 `njit` 内核，并通过测试验证公共路径会触发编译。
- 在同一份真实 `devauc` 2-step smoke 配置上，接入 numba 后吞吐从此前约 `1.2 it/s` 提升到约 `5.9 it/s`，阶段摘要仍然正常输出。
- 接入 numba 后，真实 smoke run 中出现多条 `OMP: Info #276: omp_set_nested routine deprecated` 提示；这不影响运行成功，但值得在后续性能/日志清理阶段单独处理。
- 原先写出的 `chain.h5` 只有顶层 `chain` / `log_prob` 数据集，不符合 `emcee.backends.HDFBackend` 期望的 `mcmc/` 布局，因此 notebook 中直接 `HDFBackend(...).get_chain()` 会抛出 `AttributeError: store == True`。
- 现已将 `chain.h5` 写出逻辑改为双格式：
  - 保留项目历史顶层 `chain` / `log_prob`
  - 额外写出 `emcee` 原生的 `mcmc/chain`、`mcmc/log_prob`、`mcmc/accepted` 与必要 attrs
- 修复后，下游 notebook 可以直接用 `emcee.backends.HDFBackend(path).get_chain(flat=True)` 读取。
- 当前实现比 reference 目录慢的主要结构原因已经确认：
  - normalization 仍在每次 `log_prob` 中调用 SciPy `ppf`
  - likelihood 仍由 Python 外层逐 lens 调用单镜头 kernel
  - 多个与参数无关的概率项尚未完全前移到上下文构建阶段
  - sampler 层仍保留 thread-pool lens 聚合和更厚的工程封装
- reference 目录的速度优势来自三件事同时成立：
  - 用单个 `base_normals` 连续矩阵驱动 normalization sampling
  - 用 `ModelContext.build()` 预计算 `p_zd_fixed`、`mstar_base`、`delta_r_grid` 等数组
  - 用 monolithic all-lens likelihood kernel 一次性完成全样本积分求和
- 本轮性能重构已锁定新的文件边界：
  - `src/cmass_lens_inference/kernels/primitives.py`：公共 numba 原语
  - `src/cmass_lens_inference/kernels/normalization.py`：normalization kernel
  - `src/cmass_lens_inference/kernels/likelihood.py`：all-lens likelihood kernel
  - `src/cmass_lens_inference/compiled_context.py`：连续数组编译上下文
  - `src/cmass_lens_inference/model.py`：生产级 `log_prob` 单入口
- 第二轮性能重构现已落地：
  - sampler 主路径已改为调用 `model.log_prob()`
  - 生产路径已退出 Python lens 循环和 `ThreadPoolExecutor`
  - `RandomBasis` 已改为单个 `base_normals` 连续矩阵
  - `resolve_parallelism(auto)` 现默认解析为 `kernel_only`
- benchmark 脚本已实现于 `scripts/benchmark_log_prob.py`，输出统一写到 `/Users/liurongfu/Work/CMASS_lens_project/outputs/benchmarks`。
- 关键修正点：
  - normalization kernel 必须使用原始 cross-section lookup table，而不是 likelihood 用的 200 点插值表；此前这里混用了不同长度的 `xp/fp`，会拉低数值一致性和 benchmark 结果。
- 当前 benchmark 结果：
  - `devauc`，`kernel_only`，5 次中位数：当前实现 `0.00522s`，reference `0.00540s`，速度比约 `0.966x`
  - `sersic`，`kernel_only`，5 次中位数：当前实现 `0.00502s`，reference `0.00602s`，速度比约 `0.834x`
  - 结论：当前实现已经达到并在两类 profile 上超过 reference 的同口径单次 `log_prob` 速度
- OpenMP `omp_set_nested` 提示的根因已经进一步缩小：
  - 不是 kernel 失效，也不是并行没有生效
  - 最小复现表明 `numba.set_num_threads()` 本身就可能触发这条提示
  - 若在 `import numba` 之前设置 `OMP_MAX_ACTIVE_LEVELS=1` 和 `KMP_WARNINGS=0`，提示会消失
  - 通过函数 import 方式调用 benchmark 时不出现提示，但直接运行脚本文件时会出现，说明问题出在脚本启动阶段的环境变量设置时机，而不是数值主链路
- OpenMP 提示清理现已通过启动钩子落地：
  - 根目录 `sitecustomize.py` 现会在任何项目导入前执行 `OMP_MAX_ACTIVE_LEVELS=1` 与 `KMP_WARNINGS=0` 的 `setdefault`
  - `scripts/sitecustomize.py` 已新增，用于覆盖 `python scripts/foo.py` 一类脚本直跑入口
  - 仓库根目录下运行 `python -c "import numba; numba.set_num_threads(12); print('ok')"` 已不再输出该 warning
  - 直接运行 `scripts/benchmark_log_prob.py` 也已不再输出该 warning
- OpenMP 清理后的 fresh benchmark 结果：
  - `devauc`，`kernel_only`，5 次中位数：当前实现 `0.00489s`，reference `0.00455s`，速度比约 `1.076x`
  - `sersic`，`kernel_only`，5 次中位数：当前实现 `0.00489s`，reference `0.00462s`，速度比约 `1.057x`
  - 结论：warning 已清除，但这轮 fresh benchmark 显示轻微性能回退；当前更接近“功能和吞吐基本稳定，但未能证明完全零回退”
- 当前 `chain.h5` 的生产方式仍然不是纯 `emcee.backends.HDFBackend`：
  - 采样主循环没有直接把 backend 传给 `EnsembleSampler`
  - 运行结束后仍由 `save_chain_artifacts()` 手工写 HDF5
  - 为兼容 notebook，`sampler.py` 额外维护了顶层 `chain` / `log_prob` 和人工构造的 `mcmc/` group
- 用户已明确要求撤销这层双格式兼容补丁，并改成“纯 emcee backend 输出”：
  - `chain.h5` 必须由 `emcee.backends.HDFBackend` 在采样过程中直接写出
  - `resume` 必须优先复用 backend 中的最后状态，而不是把 checkpoint 视为唯一真相
  - 顶层 `chain` / `log_prob` 不再保留
- 当前这轮 TDD 红灯已经锁定三项必须落地的改造：
  - `metadata.json` / `run_result.json` 需要新增 `chain_storage = "emcee_hdf_backend"`
  - `resume` 不能再覆盖已有 backend 内容，完成后 `backend.iteration` 必须从旧值继续递增
  - `log_prob` blob 不能再是 Python `dict`，否则标准 `HDFBackend` 无法稳定持久化
- 这轮改造已经完成，当前 `chain.h5` 的真实产出方式为：
  - `run` 时先创建 `emcee.backends.HDFBackend`
  - 新 run 显式 `backend.reset(n_walkers, ndim)`
  - `EnsembleSampler(..., backend=backend, blobs_dtype=LOG_PROB_BLOB_DTYPE)` 在采样过程中直接写盘
  - 运行结束后不再手工写 `chain` / `log_prob` 顶层数据集
- 当前 `model.log_prob()` 仍对外返回结构化 timing blob，但 sampler 层会把它展开成 `emcee` 兼容的 `(log_prob, field1, field2, ...)` 形式，并显式提供 `blobs_dtype`。
- 当前 `resume` 的 source-of-truth 已切换为 backend：
  - 优先读取 `chain.h5` 的最后一个 sample 和 `iteration`
  - checkpoint 仍保留，但只作为冗余恢复或兼容回退
  - 对不带 blobs 的旧 backend，当前实现会降级为“仅取最后坐标重新开始首步计算 blobs”
- 当前 `metadata.json` 与 `run_result.json` 都会写入 `chain_storage = "emcee_hdf_backend"`。
- 真实 smoke 验证结果：
  - 运行日期：2026-03-08
  - profile：`devauc`
  - 配置：2 steps、warmup 0、normalization_samples 256、`auto -> kernel_only`
  - 输出目录：`/Users/liurongfu/Work/CMASS_lens_project/outputs/devauc/20260308_211944_devauc_emcee_backend_smoke`
  - `chain.h5` 可被 `emcee.backends.HDFBackend` 直接读取，`iteration = 2`，shape 为 `(2, 24, 12)`
  - HDF5 顶层只剩 `mcmc`，不再存在顶层 `chain` / `log_prob`
- 2026-03-11 对最近一次 `devauc` 生产 run (`20260308_215643_devauc_devauc_prod_20260308`) 的只读诊断表明：
  - 异常链是 `walker 12`（0-based）
  - 该链 `mu5_0` 中位数约 `10.478`，末值固定在 `10.490740863626126`
  - 其余 23 条链 `mu5_0` 整体中位数约 `11.343`
  - `walker 12` 在 `13337` 步中仅发生 `196` 次位置变化，末尾连续 `375` 步完全不动
  - 该链 `log_prob` 为正且量级约 `53-55`，其余链在负值区间
- 由于下游后处理直接使用 `emcee.backends.HDFBackend(...).get_chain(flat=True)` 展平全部 walker，这条坏链会真实污染后验：
  - 原始 `n_posterior_samples = 272088`
  - 仅删除 `walker 12` 后变为 `260751`
  - 多个参数中位数发生可见偏移，其中 `mu5_0` 回到 `11.346` 左右
- 已对该 run 做一次性原地清洗：
  - `chain.h5` 从 `(13337, 24, 12)` 重写为 `(13337, 23, 12)`
  - `mcmc/log_prob`、`mcmc/blobs`、`mcmc/accepted` 同步删除第 12 条链
  - `mcmc.attrs['nwalkers']` 已改为 `23`
  - `latest_coords.npy` / `latest_log_prob.npy` 已改为 23 条 walker
  - `config_snapshot.yaml`、`metadata.json`、`run_result.json` 均新增 `maintenance` 区块并写明 `resume_supported: false`
  - `posterior_corner.png` 与 `posterior_corner_result.json` 已基于清洗后样本重生成
- 清洗后的 fresh 验证结果：
  - `emcee.backends.HDFBackend` 可正常读取清洗后的 `chain.h5`
  - `chain_shape = (13337, 23, 12)`
  - `log_prob_shape = (13337, 23)`
  - `accepted_shape = (23,)`
  - 过滤后 `max_logp = -11.326317929098293`，不再出现 `53+`
  - 过滤后 `mu5_0` 的 burn-in 后中位数为 `11.346249034783483`

## Technical Decisions
| Decision | Rationale |
|----------|-----------|
| 先做需求抽取与阶段规划，再进入实现 | 该项目数值链条长、约束多，先把不可变条件固定下来能减少后续返工 |
| 未来实现应采用“公共核心 + profile 配置”模式，而不是复制两套推断代码 | 两分支大部分逻辑共享，复制代码会提高维护成本并放大偏差风险 |
| 读取层需要预留字段别名兼容能力 | 规范已经指出 `cs_grid_power.h5` 与 observation HDF5 可能存在备选 dataset 名 |
| 归一化 MC 的随机基底应集中管理 | 需求明确要求同一步 `eta` 内数值一致，否则采样器会被数值噪声污染 |
| metadata 中的 Git 信息允许为空值或 `null` | 当前目录缺少 Git 元数据，不能让输出逻辑因 Git 不存在而失败 |
| CLI 本地可执行性优先采用 `sitecustomize.py` 解决，而不是强制 editable install | 这样既保留 `src/` 布局，也避免每次测试和本地调用都额外安装 |
| 第一版先保证“结构正确、接口稳定、最小可运行”，暂不承诺真实大样本配置的性能达标 | 当前用户先要求实现结构方案，真实长跑性能需要单独基准测试和优化 |
| 并行元数据必须同时写入 `metadata.json` 与 `run_result.json` | 这样命令行输出、自动化消费和离线排查都能看到一致的 resolved parallel settings |
| numba 热点优先落在单镜头二维积分和 normalization 选择项 reduction | 这两段是当前最明确的数值热点，改造收益高且不需要改变外部接口 |
| 第二轮性能重构改为对齐 reference 的“连续数组 + monolithic kernels”结构 | 仅靠单镜头 kernel 和局部 numba 化已经不足以追平 reference |
| 公共数值原语统一收敛到 `kernels/primitives.py` | 这样 likelihood 和 normalization 能共享同一套近似、采样与几何计算 |
| sampler/runner 保持对外接口稳定，但生产热点路径只允许经过 `model.log_prob()` | 避免热点逻辑继续散落在 wrapper、thread pool 和对象循环中 |
| `auto` 解析正式固定为 `kernel_only` | benchmark 已显示单进程 monolithic kernels 足以追平并超过 reference；默认不再优先 walker 进程池 |
| benchmark 采用“同机、同环境、同 profile、预热后多次取中位数”的口径 | 这样能把启动、I/O 和一次性编译噪声剥掉，只比较真实热点吞吐 |
| OpenMP 提示清理应优先修改 `sitecustomize.py` 一类启动钩子 | 这样可以在 `numba` 导入前完成环境设置，且不必改动 kernel、sampler 或线程预算逻辑 |

## Issues Encountered
| Issue | Resolution |
|-------|------------|
| `planning-with-files` 的路径在最初尝试中使用了错误目录 | 改为读取 `/Users/liurongfu/.codex/skills/planning-with-files/SKILL.md` |
| `session-catchup.py` 运行后无额外输出 | 视为当前项目没有可恢复的历史 planning 上下文，继续初始化新会话 |
| 当前目录不是 Git 仓库 | 实现时不依赖 Git 操作；只在 metadata 中尽力记录，失败则回落为空 |
| `cmass_lens` 环境缺失 `emcee` | 实现前先安装依赖，否则无法完成采样器和集成测试 |
| `emcee` 在退化 resume checkpoint 上会拒绝初始状态 | 在恢复路径加入退化检测与抖动回退，避免 resume 直接失败 |
| `cmass_lens` 环境缺失 `tqdm` | 安装依赖后继续；否则新进度显示实现无法导入 |
| `cmass_lens` 环境最初缺失 `numba` | 先安装依赖，再把热点真正改成 `njit` 内核 |

## Resources
- 需求规范: [PROJECT_REQUIREMENTS.md](/Users/liurongfu/Work/CMASS_lens_project/Bayesian_inference/PROJECT_REQUIREMENTS.md)
- `planning-with-files` 技能: `/Users/liurongfu/.codex/skills/planning-with-files/SKILL.md`
- `using-superpowers` 技能: `/Users/liurongfu/.codex/superpowers/skills/using-superpowers/SKILL.md`
- 参考文献 1: `/Users/liurongfu/Zotero/storage/98E4LV5X/Sonnenfeld - 2024 - The SLACS strong lens sample, debiased.pdf`
- 参考文献 2: `/Users/liurongfu/Zotero/storage/23KA6L67/Sonnenfeld 等 - 2019 - Hyper Suprime-Cam view of the CMASS galaxy sample. Halo mass as a function of stellar mass, size, an.pdf`
- conda 环境: `/opt/homebrew/anaconda3/envs/cmass_lens`

## Visual/Browser Findings
- 本轮没有读取图片或网页。
- 已将需求文档中的关键模型结构、固定参数、数值离散化要求和输出规范转写为文本记录，避免后续上下文丢失。
