# 任务计划：SLACS 后验预测检验与独立化迁移

## 目标
先前已完成 `devauc` / `sersic` 的 posterior predictive test、monitor、notebook comparison 和 Fig. 8 类趋势图实现；当前目标是在不改变科学合同的前提下，将这些 PPT 家族代码完整迁移出 `Bayesian_inference`，收拢到 `Posterior_predictive_test` 自己的独立包中，并把 `Bayesian_inference` 恢复为只包含 inference engine 与 `run/resume` 的纯净包。

## 当前阶段
Phase 12

## 阶段计划

### Phase 1: 需求与调研
- [x] 明确用户意图与已批准方案
- [x] 识别运行环境、输出目录、后验链结构和现有模型入口
- [x] 将关键发现记录到 findings.md
- **Status:** complete

### Phase 2: 测试设计与接口敲定
- [x] 为 PPC 核心逻辑、插值表消费和 CLI 入口设计失败测试
- [x] 明确新增模块、数据结构和输出文件格式
- [x] 记录关键接口决策及其理由
- **Status:** complete

### Phase 3: 实施与产出
- [x] 实现 posterior predictive 核心模块
- [x] 扩展 CLI 与结果落盘逻辑
- [x] 编写 sigma 插值表需求文档
- **Status:** complete

### Phase 4: 测试与验证
- [x] 运行新增定点测试并修复失败
- [x] 运行相关回归测试并记录结果
- [x] 在可行前提下做一次真实输入验证
- **Status:** complete

### Phase 5: 交付
- [x] 复核输出文件与关键约束
- [x] 更新 progress.md / findings.md 中的最终状态
- [x] 向用户交付变更与验证证据
- **Status:** complete

### Phase 6: 外部插值表监控接入
- [x] 为 HDF5 sigma 表读取补充兼容实现
- [x] 实现固定文件名 + `mtime` 门槛 + schema double check 的监控入口
- [x] 用测试锁定监控 API / CLI 合同
- [x] 对真实 `data/external` 目录做一次就绪检查
- **Status:** complete

### Phase 7: PPC replicate 规模提升
- [x] 将 PPC CLI / API 默认 `n_replicates` 提升到 `100000`
- [x] 保持完整 latent 落盘契约不变
- [x] 完成真实双 profile `1e5` 重跑并验证输出
- **Status:** complete

### Phase 8: Notebook 对比实现
- [x] 基于 `full_0103.h5` 设计 notebook baseline vs pipeline-matched 的 apples-to-apples 对比合同
- [x] 新增独立比较模块、结果脚本与测试
- [x] 用真实 notebook 资产完成一次小样本 sanity check
- [x] 发起并完成 `96000` posterior samples 的全量对比
- **Status:** complete

### Phase 9: Fig. 8 类趋势图
- [x] 设计并实现独立于 histogram PPC 的 posterior trend evaluator
- [x] 新增 `posterior-trends` CLI 与 `PosteriorTrendResult`
- [x] 以 average-size 模板输出 `m5` / `gamma` / `sigma_ap` 三面板图
- [x] 用真实 `devauc/latest` 与 `sersic/latest` 跑通并落盘
- **Status:** complete

### Phase 10: PPC 默认值回退
- [x] 将 PPC 默认 `n_replicates` 从 `100000` 回退到 `1000`
- [x] 回退默认值相关测试合同
- [x] 保留既有 `1e5` 真实结果文件，仅在 planning 中补状态说明
- **Status:** complete

### Phase 11: Canonical PPC 重构
- [x] 将 canonical posterior-draw 口径切到“尾部 `192000` draws”
- [x] 将 canonical `candidate_pool_size` 切到 `100000`
- [x] 将 PPC 主统计量从 `mean` 切到 `median`
- [x] 为 PPC posterior-draw 主循环加入多进程并行
- [x] 用真实 `devauc/latest` 和 `sersic/latest` 重跑 canonical 结果并覆盖落盘
- **Status:** complete

### Phase 12: Posterior_predictive_test 独立化迁移
- [x] 在 `Posterior_predictive_test/` 下建立独立包结构、CLI 和本地结果 dataclass
- [x] 将 PPT / trend / notebook comparison 实现与测试迁出 `Bayesian_inference`
- [x] 清理 `Bayesian_inference` 中的 PPT 源码、CLI 子命令、公开导出与测试
- [x] 完成双包测试、editable 安装与新 CLI 验证
- **Status:** complete

### Phase 13: devauc chain 覆盖后的产物刷新
- [x] 确认 `devauc/latest/chain.h5` 在执行前已静止
- [x] 用独立包 CLI 重跑 devauc 的 `posterior-predictive`
- [x] 用独立包 CLI 重跑 devauc 的 `posterior-trends`
- [x] 核对 7 个目标工件时间戳与关键 JSON 合同
- **Status:** complete

### Phase 14: PPC 默认输出根目录统一
- [x] 将 PPC / monitor / trends 的 CLI `--output-dir` 改为可选默认值
- [x] 将三个公开 API 的 `output_root_dir` 默认值统一到项目根 `output/`
- [x] 保留显式 `--output-dir` 覆盖默认行为
- [x] 补充并通过默认输出路径相关测试与回归
- **Status:** complete

## 关键问题
1. 新的 PPC 逻辑应放在 `Bayesian_inference` 包内哪些模块，才能最大化复用现有 inference 代码？
2. 在 sigma 插值表尚未产出的情况下，CLI 应如何设计输入契约，既能完整实现又不把假设写死？
3. replicated sample 与 normalization kernel 共享哪些分布和 selection 权重，哪些职责必须拆分？
4. notebook `full_0103.h5` 的结果差异究竟来自哪个实现分叉：总体生成、参数顺序、sigma 插值器，还是后处理统计口径？
5. 如何在不破坏现有科学合同的前提下，把 PPT 家族工作流从 inference engine 中完整剥离？
6. 当 `latest` 指向不变但底层 `chain.h5` 被覆盖时，如何刷新单 profile 的正式产物而不误伤另一个 profile？
7. 如何让 PPC 家族命令在不破坏显式参数覆盖的前提下，统一默认输出到项目根 `output/`？

## 已做决策
| 决策 | 理由 |
|------|------|
| PPT 家族代码整体迁移到 `Posterior_predictive_test/src/lensing_posterior_predictive/` | 用户要求保持 `Bayesian_inference` “纯粹”，不再承载研究型 PPT 工作流 |
| `theta_E` 与 `sigma` 的 replicated sample 分两次独立抽样 | 这是用户确认的统计口径，且与真实 23-lens / 7-lens 对照最一致 |
| sigma 侧消费“大插值表”而非逐星系 `s2_grid` | 更接近 `prepare_intepolation_grids` 的职责，也更适合对任意 replicated lens 评估 |
| planning 文件放在 `Posterior_predictive_test/` | 这是当前任务工作区，满足 `planning-with-files` 约束 |
| PPC 候选池大小独立于 normalization 的 `1e5` | replicated sample 需要显式 materialize lens 集合；若每次都展开 `1e5` 候选会显著拖慢真实运行 |
| 监控入口只盯固定文件名 `jeans_deV_grid.h5` / `jeans_sers_grid.h5` | 用户明确外部线程可能覆盖原文件名，而不是新建新文件 |
| HDF5 读取同时兼容 PPC-native `*_axis` schema 和外部 `*_grid` schema | 另一线程可能写新 schema，但当前真实外部文件和过渡版本仍使用 legacy 布局 |
| 监控触发时间默认锁定为 `2026-03-09T15:27:07+08:00` | 这是用户确认的“只有当前时刻之后覆盖的新表才允许触发”的门槛 |
| PPC 默认 `n_replicates` 曾提升到 `100000`，现已回退到 `1000` | `1e5` 被保留为一次历史性真实运行；当前源码默认行为回到较轻量的原始口径 |
| notebook apples-to-apples 对比单独放进 `notebook_comparison.py` 与独立脚本 | 这是历史工作流调试，不应污染生产 PPC API 和默认统计口径 |
| notebook 对比里的 sigma 查询必须直接使用 `RegularGridInterpolator((z, logRe, logn, gamma), s2_grid)` | 用户明确要求本地 pipeline 在这次比较中遵守 notebook 原生插值表契约，而不是走 `SigmaUnitTable` 适配层 |
| notebook baseline 不再计算 `P_SL_norm_mc` | 该项在 notebook 抽样时只是同一个 posterior draw 上的标量，权重归一化后完全抵消，省略不会改变 selected-lens 分布 |
| Fig. 8 类趋势图与 histogram PPC 分开实现 | 两者统计对象不同；趋势图是条件均值曲线，不应复用 23-lens / 7-lens replicated-sample 接口 |
| Fig. 8 顶部和中部的 parent 曲线直接使用 `mu5(m*)` 与 `mu_gamma(m*)` | 这是用户批准的“average size”口径，可以去掉不必要的 Monte Carlo 噪声 |
| canonical PPC 默认后验 draw 口径固定为尾部 `min(post-burnin flatten chain, 192000)` | `devauc` 需要与 `sersic` 对齐到 `192000`，同时保留“用真实后验链而非轻量子样本”的正式口径 |
| canonical PPC 默认候选池 cap 提升到 `100000` | 这次用户真正要放大的参数是 `candidate_pool_size`，不是 `n_replicates` |
| canonical PPC 主统计量改为 `median/std/p10/p90` | 用户明确要求把位置统计从 `mean` 切到 `median` |
| canonical PPC 默认执行策略为 posterior-draw chunk 的 `process_pool` | `192000 x 100000` 的真实作业必须并行，否则运行时间不可接受 |
| 新 CLI 根命令为 `lensing-posterior-predictive` | 迁移后 `cmass-lens-inference` 只保留 `run` / `resume`，不做兼容转发 |
| `Posterior_predictive_test` 通过依赖 `cmass_lens_inference` 复用底层能力 | 可以复用 inference engine，而不再把 PPT 实现放回 engine 包内 |
| 顶层 comparison 脚本不再做 `sys.path` 注入 | 新包已经可 editable 安装，脚本应通过正式包入口运行 |
| 本地双包安装顺序固定为先装 `Bayesian_inference`，再对 `Posterior_predictive_test` 执行 `pip install -e . --no-deps` | `cmass-lens-inference` 是本地包而非 PyPI 依赖，直接解析依赖名会失败 |
| 当单个 profile 的 `latest` run ID 不变但 `chain.h5` 被覆盖时，只重跑该 profile 对应的 `posterior-predictive` 与 `posterior-trends` | 这样可以刷新过期工件，同时避免无谓重跑另一个 profile |
| PPC / monitor / trends 的默认输出根目录统一为 `/Users/liurongfu/Work/CMASS_lens_project/output`，但显式 `--output-dir` 仍优先 | 满足“默认统一落盘 + 手动覆盖”两种使用场景，且不破坏现有脚本 |

## 遇到的问题
| 错误 | 尝试次数 | 处理方式 |
|------|----------|----------|
| `CMASS_lens_project` 不是 git 仓库 | 1 | 放弃 worktree / git 流程，直接在现有目录实施并加强本地验证 |
| HDF5 loader 初版只接受 `*_axis` schema，导致固定文件名 legacy 表监控失败 | 1 | 扩展 `SigmaUnitTable.from_path()`，增加 `*_grid` schema 转换与 `logn` 轴查询适配 |
| Fig. 8 趋势图需要固定 stellar-mass 并冻结在 average-size 模板，不能直接复用 PPC 候选池生成函数 | 1 | 新增独立 trend evaluator，在固定 `log M*` 和 representative `R_e/n` 条件下重新采样 `z_d/z_s/m5/gamma` |
| 新包 editable 安装初次失败：`pip` 试图从索引解析 `cmass-lens-inference` | 1 | 先将 `Bayesian_inference` editable 安装进环境，再对 `Posterior_predictive_test` 执行 `pip install -e . --no-deps` |

## 备注
- 随进度更新阶段状态：pending → in_progress → complete
- 重大决策前重读本计划
- 记录所有错误，避免重复犯错
