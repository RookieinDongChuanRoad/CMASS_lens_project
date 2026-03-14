# Task Plan: CMASS Lens Bayesian Inference

## Goal
基于 [PROJECT_REQUIREMENTS.md](/Users/liurongfu/Work/CMASS_lens_project/Bayesian_inference/PROJECT_REQUIREMENTS.md) 搭建一个可长期维护的强透镜层次贝叶斯推断项目执行蓝图，明确 `devauc` 与 `sersic` 两个分支共享/分化的建模要求、运行环境前提和后续实现阶段。

## Current Phase
Phase 5

## Phases

### Phase 1: Requirements & Discovery
- [x] 通读项目需求并提炼不可变约束
- [x] 确认运行环境前提（`conda` 环境 `cmass_lens`）
- [x] 将关键发现记录到 `findings.md`
- **Status:** complete

### Phase 2: Planning & Architecture
- [x] 明确项目目录结构、模块边界和数据流
- [x] 拆分共享组件与 `devauc` / `sersic` 专属组件
- [x] 明确数值热点、缓存策略和 `numba` 兼容边界
- **Status:** complete

### Phase 3: Implementation
- [x] 实现数据读取、插值、分布函数与宇宙学工具
- [x] 实现单镜头 likelihood 与 MC 归一化
- [x] 实现 `emcee` 采样、checkpoint 与结果落盘
- **Status:** complete

### Phase 4: Testing & Verification
- [x] 校验输入 HDF5 结构兼容性
- [x] 校验数值积分、归一化、边界条件与拒绝逻辑
- [x] 记录运行性能与最小可复现实验结果
- **Status:** complete

### Phase 5: Delivery
- [x] 完成性能重构后的全量复核
- [x] 汇总风险、假设与后续建议
- [x] 向用户交付可执行结果与验证结论
- **Status:** in_progress

### Phase 6: Performance Refactor
- [x] 重构 `numba` 原语文件边界，集中到 `kernels/primitives.py`
- [x] 引入 compiled context，将参数无关项前移为连续数组
- [x] 用 monolithic likelihood/normalization kernels 替换生产主路径
- [x] 新增 benchmark 并确认速度不低于 reference 目录
- **Status:** complete

## Key Questions
1. 代码将采用怎样的模块划分，才能同时保证 `devauc` / `sersic` 分支共享主干逻辑、又避免 profile-specific 参数污染公共接口？
2. `cs_grid_power.h5` 与两类 observation HDF5 的真实字段名是否与规范完全一致，还是需要在读取层做稳健兼容？
3. 单镜头积分与归一化 MC 的热点函数如何组织，才能在 `numba` 约束下兼顾可读性与运行速度？
4. checkpoint、元数据快照与 run 目录命名策略如何设计，才能支持长时间采样恢复和结果比对？

## Decisions Made
| Decision | Rationale |
|----------|-----------|
| 使用 `planning-with-files` 在项目根目录创建 `task_plan.md`、`findings.md`、`progress.md` | 这是用户显式要求，也是复杂任务下的持久化上下文机制 |
| 当前先进入“需求理解 + 执行规划”阶段，而不是直接写实现代码 | 项目规范已经足够细，但实现前仍需先锁定架构边界、数据接口和验证路径 |
| 将 `cmass_lens` 视为默认运行环境前提写入计划文件 | 用户明确指定且本机 `conda env list` 已确认该环境存在 |
| 将 profile 差异收敛到结构先验参数和 `n` 使用策略，而不是复制两套完整推断管线 | 需求明确要求两分支共享超参数定义、先验范围、采样器与输出格式 |
| 输出根目录固定写入 `/Users/liurongfu/Work/CMASS_lens_project/outputs` | 用户已明确要求输出不放仓库内，并按 profile 分桶 |
| 当前在原目录直接实现，不使用 git worktree | 当前目录不是 Git 仓库，`using-git-worktrees` 前提不成立 |
| 使用 `sitecustomize.py` 为本地 `python -m cmass_lens_inference.cli` 提供 `src/` 路径引导 | 当前目录未安装为 editable package，CLI 烟测和本地开发都需要稳定发现包 |
| `resume` 遇到退化 checkpoint walker 时回退到抖动初始化 | 这能避免 `emcee` 因病态初值直接报错，同时保持恢复流程可执行 |
| 并行默认策略采用 `auto`，在当前 14 线程机器上解析为“预留 2 核后最多使用 12 核” | 这是用户明确要求的资源利用策略，同时保留基本桌面可用性 |
| `tqdm` 只保留单个总采样进度条，阶段耗时用摘要行输出到终端和 `run.log` | 用户明确要求不要刷屏，且需要能看出卡在 likelihood 还是 normalization |
| 本轮性能优化按“公共原语 / 编译上下文 / kernels / 外层编排”四层重构 | 这样可以把 numba 热点集中管理，并对齐 reference 目录的高性能结构 |
| `normal_ppf`、`skewnorm_sample`、`truncnorm_sample` 统一收敛到 `src/cmass_lens_inference/kernels/primitives.py` | 避免同类数值近似在多个文件中重复实现，降低回归风险 |
| 重构完成后 `auto` 默认策略应优先偏向 `kernel_only` | monolithic numba kernels 更可能在单进程内吃满 12 线程，避免 walker 进程复制开销 |
| benchmark 以 `scripts/benchmark_log_prob.py` 为统一入口，并把结果写入 `/Users/liurongfu/Work/CMASS_lens_project/outputs/benchmarks` | 这样可以长期复用同一套对比方法，而不污染源码目录 |
| OpenMP `omp_set_nested` 提示清理优先采用启动钩子修复，而不是修改 kernel 或并行策略 | 根因已定位为环境变量设置时机过晚，最安全的修复是把抑噪变量前移到 Python 启动阶段 |
| `chain.h5` 统一改为纯 `emcee.backends.HDFBackend` 输出 | 下游 notebook 需要直接 `HDFBackend(...).get_chain()`；继续手工写 HDF5 只会制造格式漂移和恢复路径歧义 |
| 最近一次 `devauc` 生产 run 的坏链清洗采用“原地事务式重写” | 用户明确要求直接剔除异常 walker，且该 run 已作为分析结果使用，重跑成本高于定点清洗 |
| 异常链固定判定为 `walker 12`（0-based）并整条删除 | 其 `mu5_0` 长期停在 `10.47-10.49`，与其余链 `11.3` 左右明显分离，且 `log_prob` 量级异常 |
| 清洗后的 `devauc` run 视为封存结果，不再承诺 `resume` | 原地删链会改变 walker 维度；保留分析自洽性优先于继续采样语义 |

## Errors Encountered
| Error | Attempt | Resolution |
|-------|---------|------------|
| 初次读取 `planning-with-files` 时路径写成 `/Users/liurongfu/.codex/superpowers/skills/planning-with-files/SKILL.md` | 1 | 改按技能清单给出的真实路径 `/Users/liurongfu/.codex/skills/planning-with-files/SKILL.md` 读取 |
| `git status` 返回 `fatal: not a git repository` | 1 | 记录当前目录无 Git 元数据，后续跳过 worktree / Git metadata 的强依赖，相关输出字段允许为空 |
| `cmass_lens` 环境缺少 `emcee` | 1 | 在实现前补装依赖，再继续采样器开发与测试 |
| `resume` 使用完全一致的 walker 坐标触发 `emcee` condition number 报错 | 1 | 恢复路径检测退化 checkpoint，必要时改用围绕初始化中心的抖动坐标 |
| 最近一次 `devauc` run 中 1 条 walker 与其余链明显脱离 | 1 | 先做只读诊断锁定 `walker 12`，再按事务式清洗方案原地删除并重建派生结果 |
| CLI 子进程无法发现 `src/` 包 | 1 | 添加 `sitecustomize.py` 自动把 `src/` 插入 `sys.path`，保证 `python -m cmass_lens_inference.cli` 可运行 |
| 并行实现首次测试时 `cmass_lens` 环境缺少 `tqdm` | 1 | 在 conda 环境中安装 `tqdm` 后继续 |
| `emcee` 在启用 backend 时拒绝 Python `dict` blob | 1 | 将 timing blob 改成结构化 dtype，并在 sampler 层显式展开返回值与 `blobs_dtype` |

## Notes
- 后续每进入新阶段前先回读本文件，避免实现时偏离需求。
- 若真实数据字段与规范不一致，优先在读取层做兼容，不要污染统计模型核心逻辑。
- 性能优化属于硬约束，不是实现完成后的“锦上添花”。
- 性能重构必须先写失败测试，再替换生产主路径；不允许直接“边改边猜”。
- 当前 benchmark 结论支持 `auto -> kernel_only`：单次 `log_prob` 在 `devauc`/`sersic` 上都已达到或优于 reference 的同口径速度水平。
- OpenMP 提示清理必须保持统计逻辑、线程预算逻辑和 benchmark 口径不变，只处理启动环境时机。
