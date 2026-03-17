# 进展日志

## 会话日期：2026-03-09

### 阶段 1：需求与现状梳理
- **状态：** 已完成
- **开始时间：** 2026-03-09 13:36 CST
- 已执行动作：
  - 审阅 `outputs` 下最新 `devauc` / `sersic` run 结构与配置快照
  - 审阅 posterior predictive 计划，确认 `theta_E` 与 `sigma` 的抽样口径
  - 解释 normalization kernel 的 `1e5` 在代码中的入口与用途
  - 审阅 `prepare_intepolation_grids` 中 Jeans 与 `s2_grid` 逻辑，明确大插值表需求
- 新建/修改文件：
  - `task_plan.md`（新建）
  - `findings.md`（新建）
  - `progress.md`（新建）

### 阶段 2：测试设计与接口敲定
- **状态：** 已完成
- 已执行动作：
  - 审阅 `Bayesian_inference` 的 CLI、测试夹具和结果数据结构
  - 确定 PPC 逻辑将扩展现有 CLI，并采用测试先行
- 新建/修改文件：
  - `task_plan.md`（已更新）
  - `findings.md`（已更新）

### 阶段 3：实现与验证
- **状态：** 已完成
- **开始时间：** 2026-03-09 14:00 CST
- 已执行动作：
  - 新增 `cmass_lens_inference.posterior_predictive` 模块，实现后验链读取、候选总体生成、23-lens / 7-lens 独立抽样、sigma 插值表消费、结果落盘和概览图绘制
  - 扩展 `cmass_lens_inference.cli`，新增 `posterior-predictive` 子命令
  - 在 `types.py` 中新增 `PosteriorPredictiveResult`
  - 在 `pyproject.toml` 中补充 `matplotlib`
  - 在 `prepare_intepolation_grids` 下新增 sigma 插值表需求文档
  - 按 TDD 新增 `tests/test_posterior_predictive.py`，先看红灯，再补实现至转绿
- 新建/修改文件：
  - `/Users/liurongfu/Work/CMASS_lens_project/Bayesian_inference/src/cmass_lens_inference/posterior_predictive.py`（新建）
  - `/Users/liurongfu/Work/CMASS_lens_project/Bayesian_inference/src/cmass_lens_inference/cli.py`（修改）
  - `/Users/liurongfu/Work/CMASS_lens_project/Bayesian_inference/src/cmass_lens_inference/types.py`（修改）
  - `/Users/liurongfu/Work/CMASS_lens_project/Bayesian_inference/src/cmass_lens_inference/__init__.py`（修改）
  - `/Users/liurongfu/Work/CMASS_lens_project/Bayesian_inference/tests/test_posterior_predictive.py`（新建）
  - `/Users/liurongfu/Work/CMASS_lens_project/Bayesian_inference/pyproject.toml`（修改）
  - `/Users/liurongfu/Work/CMASS_lens_project/prepare_intepolation_grids/2026-03-09-ppt-sigma-interpolation-requirements.md`（新建）

## 测试结果
| 测试项 | 输入 | 预期结果 | 实际结果 | 状态 |
|--------|------|----------|----------|------|
| 计划阶段环境检查 | `git -C /Users/liurongfu/Work/CMASS_lens_project rev-parse --is-inside-work-tree` | 若是仓库则输出 `true` | 当前目录不是 git 仓库 | ✓ |
| PPC 定点测试 | `conda run -n cmass_lens pytest -q tests/test_posterior_predictive.py` | 新增 PPC API 与 CLI 测试全部通过 | `3 passed` | ✓ |
| PPC + runner 回归测试 | `conda run -n cmass_lens pytest -q tests/test_runner_cli.py tests/test_posterior_predictive.py` | 既有 run/resume/CLI 合同不被破坏，PPC 新测试继续通过 | `8 passed` | ✓ |
| HDF5 / 监控增强后的 PPC 测试 | `conda run -n cmass_lens pytest -q tests/test_posterior_predictive.py` | HDF5 loader、监控 API、监控 CLI 合同全部通过 | `8 passed` | ✓ |
| HDF5 / 监控增强后的回归测试 | `conda run -n cmass_lens pytest -q tests/test_runner_cli.py tests/test_posterior_predictive.py` | 原有 run/resume/CLI 行为保持通过，新增监控行为通过 | `13 passed` | ✓ |
| 真实外部目录就绪检查 | `python3` 读取 `data/external` 两个固定文件的 `mtime` | 仅当 `mtime > 2026-03-09T15:27:07+08:00` 才允许触发 | 两文件仍是 `2026-02-25T11:05:36+08:00`，不触发 | ✓ |
| 真实外部表触发 + 双 profile PPT | `conda run -n cmass_lens python -m cmass_lens_inference.cli posterior-predictive-monitor --external-dir /Users/liurongfu/Work/CMASS_lens_project/data/external --output-dir /Users/liurongfu/Work/CMASS_lens_project/Posterior_predictive_test/results --not-before 2026-03-09T15:27:07+08:00 --timeout-seconds 5` | 若外部线程已覆盖新表，则通过门槛检查并完成 `devauc` / `sersic` 两条真实 PPT | `status=completed`；两张表 `mtime` 分别为 `2026-03-09T20:09:56+08:00`、`2026-03-09T22:02:27+08:00`；双 profile 结果目录均生成 | ✓ |
| 默认 `1e5` replicate 的真实双 profile 重跑 | `conda run -n cmass_lens python -m cmass_lens_inference.cli posterior-predictive-monitor --external-dir /Users/liurongfu/Work/CMASS_lens_project/data/external --output-dir /Users/liurongfu/Work/CMASS_lens_project/Posterior_predictive_test/results --not-before 2026-03-09T15:27:07+08:00 --timeout-seconds 5` | 不显式传 `--n-replicates` 时，应使用新默认值 `100000` 并完成双 profile 真实输出 | `status=completed`；`devauc` / `sersic` 的 manifest 和 summary 都写入 `n_replicates=100000`；每个 `replicated_statistics.npz` 约 `171.67 MB` | ✓ |
| 默认值回退后的回归测试 | `conda run -n cmass_lens pytest -q tests/test_runner_cli.py tests/test_posterior_predictive.py` | 默认 `n_replicates` 回退到 `1000` 后，现有 runner / PPC 合同仍全部通过 | `22 passed` | ✓ |
| notebook 对比专用测试 | `conda run -n cmass_lens pytest -q tests/test_notebook_comparison.py` | notebook-native sigma loader、参数顺序映射、对比工件输出合同全部通过 | `4 passed` | ✓ |
| notebook 对比相关回归测试 | `conda run -n cmass_lens pytest -q tests/test_runner_cli.py tests/test_posterior_predictive.py tests/test_notebook_comparison.py` | 新比较模块不破坏现有 runner / PPC 合同 | `21 passed` | ✓ |
| notebook 真实资产小样本 sanity run | `conda run -n cmass_lens python /Users/liurongfu/Work/CMASS_lens_project/Posterior_predictive_test/compare_full0103_notebook_vs_pipeline.py --max-samples 8 --output-dir /Users/liurongfu/Work/CMASS_lens_project/Posterior_predictive_test/results/notebook_vs_pipeline/full_0103_sanity` | 应成功加载真实 `Population_model.py`、真实 `full_0103.h5` 与 notebook HDF5 sigma 表，并产出比较结果目录 | `status=completed`；`posterior_sample_count=8`；结果目录写入四个工件 | ✓ |
| notebook 全量 `96000` posterior samples 对比 | `conda run -n cmass_lens python /Users/liurongfu/Work/CMASS_lens_project/Posterior_predictive_test/compare_full0103_notebook_vs_pipeline.py --output-dir /Users/liurongfu/Work/CMASS_lens_project/Posterior_predictive_test/results/notebook_vs_pipeline/full_0103` | 应完成 notebook corrected baseline vs pipeline-matched 的正式全量比较，并写出四个工件 | `status=completed`；`posterior_sample_count=96000`；结果目录体积约 `18 MB`；合同检查通过 | ✓ |
| canonical PPC 新合同定点测试 | `conda run -n cmass_lens pytest -q /Users/liurongfu/Work/CMASS_lens_project/Bayesian_inference/tests/test_posterior_predictive.py -k "defaults_to_tail_capped_full_chain_mode or public_defaults_use_1000_replicates or select_posterior_draws_defaults_to_tail_capped_chain or resolve_candidate_pool_size_uses_new_100000_cap or parallel_execution_matches_single_process_results or monitor_defaults_to_tail_capped_common_draw_count or generates_expected_artifacts_for_sersic"` | 新增的 `192000` 尾部后验、`candidate_pool_size=100000`、`median`、并行执行合同应先红后绿 | RED：`7 failed`；GREEN：补实现后同一批测试 `8 passed` | ✓ |
| canonical PPC 全量回归测试 | `conda run -n cmass_lens pytest -q /Users/liurongfu/Work/CMASS_lens_project/Bayesian_inference/tests/test_runner_cli.py /Users/liurongfu/Work/CMASS_lens_project/Bayesian_inference/tests/test_posterior_predictive.py` | runner / PPC 现有合同与新 canonical PPC 合同应同时通过 | 首次因 `_load_posterior_draws()` 私有 helper 返回值变化失败 `1` 条旧测试；同步测试后 `28 passed` | ✓ |
| canonical PPC 真实双 profile 重跑 | `conda run -n cmass_lens python -m cmass_lens_inference.cli posterior-predictive-monitor --external-dir /Users/liurongfu/Work/CMASS_lens_project/data/external --output-dir /Users/liurongfu/Work/CMASS_lens_project/Posterior_predictive_test/results --devauc-run-dir /Users/liurongfu/Work/CMASS_lens_project/outputs/devauc/latest --sersic-run-dir /Users/liurongfu/Work/CMASS_lens_project/outputs/sersic/latest --not-before 2026-03-09T15:27:07+08:00 --timeout-seconds 5` | 应覆盖当前 canonical 结果目录，并把合同切到 `192000` 尾部后验、`candidate_pool_size=100000`、`median` 和 `process_pool` | `status=completed`；`devauc` 于 `2026-03-10 21:29:42` 落盘，`sersic` 于 `2026-03-10 21:39:11` 落盘；两边 manifest 都写入 `n_posterior_draws_used=192000`, `candidate_pool_size=100000`, `posterior_draw_mode=tail_capped_full_chain`, `worker_processes=12` | ✓ |

## 错误日志
| 时间戳 | 错误 | 尝试次数 | 处理方式 |
|--------|------|----------|----------|
| 2026-03-09 13:55 CST | `fatal: not a git repository` | 1 | 停止依赖 git/worktree，转为目录内直接实施并强化本地计划文件 |
| 2026-03-09 16:12 CST | HDF5 监控测试失败：source 只接受 `*_axis` schema | 1 | 重新读取源码，补上 `*_grid` legacy schema 转换和 `logn` 轴查询适配 |
| 2026-03-09 16:20 CST | 真实旧表因 `s2_grid` 中少量负值被过严阈值拒绝 | 1 | 将负值规则改为“占比不超过 5%，最小值不低于 `-1e-4`”，再裁到非负 |
| 2026-03-10 00:xx CST | 将真实 PPC 默认 replicate 数量提升到 `1e5` 后，运行时间显著增长 | 1 | 保持当前算法与完整 latent 落盘不变，等待真实双 profile 作业完成并记录文件体积 |
| 2026-03-10 16:xx CST | 用户要求撤销 `1e5` 默认值改动，但保留已产出的 `1e5` 历史结果 | 1 | 只回退源码和测试默认值到 `1000`，不覆盖既有结果文件，并在 planning 中补充状态说明 |
| 2026-03-10 14:xx CST | notebook 对比测试首次失败：缺少 `cmass_lens_inference.notebook_comparison` 模块 | 1 | 新增独立比较模块、结果 dataclass 和真实脚本，并用测试锁定 notebook 原生插值与参数映射合同 |
| 2026-03-10 15:04 CST | 发起 notebook 全量对比时结果目录长时间保持 `0B` | 1 | 持续监控进程 CPU 时间与子进程状态，确认脚本采用“全部计算完成后再统一落盘”的行为，不属于假死 |
| 2026-03-10 21:11 CST | `using-git-worktrees` 失败：`Bayesian_inference` 仓库 `No commits yet`，只能创建空 orphan worktree | 1 | 记录该限制并退回当前源码树实施，再用本地回归和真实重跑兜底 |
| 2026-03-10 21:18 CST | PPC 全量回归首次失败：一个旧测试仍按 `_load_posterior_draws()` 的旧返回值形状索引 | 1 | 同步测试到新的 `(posterior_draws, posterior_draw_mode)` 私有 helper 合同，再次回归转绿 |

## 会话日期：2026-03-10

### 阶段 9：Fig. 8 类趋势图
- **状态：** 已完成
- **开始时间：** 2026-03-10 14:45 CST
- 已执行动作：
  - 基于已批准方案，为 `m5`、`gamma`、`sigma_ap` 新增独立 trend evaluator，而不是复用 histogram PPC
  - 新增 average-size helper，明确 `devauc` 固定 `n=4`、`sersic` 先求平均 `n` 再映射到平均 `R_e`
  - 新增 `posterior-trends` CLI 和 `PosteriorTrendResult`
  - 落盘 `fig8_like.png`、`fig8_like_summary.json`、`fig8_like_curves.npz`
  - 用真实 `devauc/latest` + `jeans_deV_grid.h5` 与 `sersic/latest` + `jeans_sers_grid.h5` 各跑一遍正式结果
- 新建/修改文件：
  - `/Users/liurongfu/Work/CMASS_lens_project/Bayesian_inference/src/cmass_lens_inference/posterior_predictive.py`（修改）
  - `/Users/liurongfu/Work/CMASS_lens_project/Bayesian_inference/src/cmass_lens_inference/cli.py`（修改）
  - `/Users/liurongfu/Work/CMASS_lens_project/Bayesian_inference/src/cmass_lens_inference/types.py`（修改）
  - `/Users/liurongfu/Work/CMASS_lens_project/Bayesian_inference/src/cmass_lens_inference/__init__.py`（修改）
  - `/Users/liurongfu/Work/CMASS_lens_project/Bayesian_inference/tests/test_posterior_predictive.py`（修改）

### 阶段 11：Canonical PPC 重构
- **状态：** 已完成
- **开始时间：** 2026-03-10 21:09 CST
- 已执行动作：
  - 将 PPC 默认 posterior draw 逻辑改成尾部 `min(post-burnin flatten chain, 192000)`，并让 monitor 默认对齐双 profile 的共同尾部长度
  - 将默认 `candidate_pool_size` cap 从 `4096` 提升到 `100000`
  - 将 PPC 主统计量从 `mean` 切到 `median`
  - 为 PPC posterior-draw 主循环加入多进程 `process_pool` 并行，并用“每个 posterior draw 一个确定性 seed”锁定串行/并行一致性
  - 完成真实 `devauc/latest` 与 `sersic/latest` 的 canonical 重跑并覆盖结果目录
- 新建/修改文件：
  - `/Users/liurongfu/Work/CMASS_lens_project/Bayesian_inference/src/cmass_lens_inference/posterior_predictive.py`（修改）
  - `/Users/liurongfu/Work/CMASS_lens_project/Bayesian_inference/src/cmass_lens_inference/cli.py`（修改）
  - `/Users/liurongfu/Work/CMASS_lens_project/Bayesian_inference/tests/test_posterior_predictive.py`（修改）

## 5 个重启检查问题
| 问题 | 回答 |
|------|------|
| 我现在在哪一步？ | Phase 11：canonical PPC 重构与真实双 profile 重跑已完成 |
| 我接下来要去哪？ | 向用户交付新的 canonical 合同、真实运行耗时和关键统计结果 |
| 我的目标是什么？ | 让 canonical PPC 的真实产物与最新合同一致：`192000` 尾部后验、`candidate_pool_size=100000`、`median`、多进程并行 |
| 我学到了什么？ | 这次真正需要放大的不是 replicate 数，而是 candidate pool；同时 `devauc` / `sersic` 的 posterior draw 对齐必须通过尾部裁切明确写进合同 |
| 我已经做了什么？ | 已完成测试先行、源码实现、并行化、完整回归，以及真实双 profile canonical 重跑 |

## 会话日期：2026-03-11

### 阶段 12：Posterior_predictive_test 独立化迁移
- **状态：** 已完成
- **开始时间：** 2026-03-11 13:3x CST
- 已执行动作：
  - 在 `Posterior_predictive_test/` 下新增 `pyproject.toml`、`src/cmass_posterior_predictive/` 与迁移后的 `tests/`
  - 将 `posterior_predictive`、`posterior_trends`、`notebook_comparison`、结果 dataclass 与 CLI 全部迁入新包
  - 将 `Bayesian_inference` 的 CLI 收缩到 `run` / `resume`，删除 PPT 源码和对应测试
  - 去掉 `compare_full0103_notebook_vs_pipeline.py` 中的 `sys.path` 注入
  - 将 `Bayesian_inference` 与 `Posterior_predictive_test` 都以 editable 方式安装到 `cmass_lens` 环境
  - 用新包入口真实跑通一次 `posterior-trends`
  - 启动过一次 `posterior-predictive-monitor` 真实验证，确认新 CLI 能通过外部表检查并进入多进程计算；为避免长时间占用机器，验证后手动停止该长任务
- 新建/修改文件：
  - `/Users/liurongfu/Work/CMASS_lens_project/Posterior_predictive_test/pyproject.toml`（新建）
  - `/Users/liurongfu/Work/CMASS_lens_project/Posterior_predictive_test/src/cmass_posterior_predictive/*.py`（新建）
  - `/Users/liurongfu/Work/CMASS_lens_project/Posterior_predictive_test/tests/*.py`（新建/迁移）
  - `/Users/liurongfu/Work/CMASS_lens_project/Posterior_predictive_test/compare_full0103_notebook_vs_pipeline.py`（修改）
  - `/Users/liurongfu/Work/CMASS_lens_project/Bayesian_inference/src/cmass_lens_inference/cli.py`（修改）
  - `/Users/liurongfu/Work/CMASS_lens_project/Bayesian_inference/src/cmass_lens_inference/__init__.py`（修改）
  - `/Users/liurongfu/Work/CMASS_lens_project/Bayesian_inference/src/cmass_lens_inference/types.py`（修改）
  - `/Users/liurongfu/Work/CMASS_lens_project/Bayesian_inference/src/cmass_lens_inference/posterior_predictive.py`（删除）
  - `/Users/liurongfu/Work/CMASS_lens_project/Bayesian_inference/src/cmass_lens_inference/notebook_comparison.py`（删除）
  - `/Users/liurongfu/Work/CMASS_lens_project/Bayesian_inference/tests/test_posterior_predictive.py`（删除）
  - `/Users/liurongfu/Work/CMASS_lens_project/Bayesian_inference/tests/test_notebook_comparison.py`（删除）
- 关键验证：
  - `cd /Users/liurongfu/Work/CMASS_lens_project/Posterior_predictive_test && conda run -n cmass_lens pytest -q tests` -> `32 passed`
  - `cd /Users/liurongfu/Work/CMASS_lens_project/Bayesian_inference && conda run -n cmass_lens pytest -q tests/test_runner_cli.py` -> `6 passed`
  - `conda run -n cmass_lens cmass-lens-inference --help` -> 仅暴露 `run` / `resume`
  - `conda run -n cmass_lens cmass-posterior-predictive --help` -> 暴露 `posterior-predictive`、`posterior-predictive-monitor`、`posterior-trends`、`notebook-comparison`
  - `conda run -n cmass_lens cmass-posterior-predictive posterior-trends --run-dir /Users/liurongfu/Work/CMASS_lens_project/outputs/sersic/latest --sigma-table /Users/liurongfu/Work/CMASS_lens_project/data/external/jeans_sers_grid.h5 --output-dir /Users/liurongfu/Work/CMASS_lens_project/Posterior_predictive_test/results` -> `status=completed`

### 阶段 13：devauc chain 覆盖后的产物刷新
- **状态：** 已完成
- **开始时间：** 2026-03-11 14:31 CST
- 已执行动作：
  - 对 `/Users/liurongfu/Work/CMASS_lens_project/outputs/devauc/latest/chain.h5` 做两次 `mtime` 采样，确认 chain 已静止
  - 运行 `cmass-posterior-predictive posterior-predictive`，覆盖 devauc 的 histogram PPC 工件
  - 运行 `cmass-posterior-predictive posterior-trends`，覆盖 devauc 的 Fig. 8 类趋势图工件
  - 再次检查 `chain.h5` 的 `mtime`，确认执行期间没有发生第二次覆盖
  - 读取 `run_manifest.json`、`ppc_summary.json`、`fig8_like_summary.json`，核对合同字段
  - 直接查看 `ppc_overview.png` 与 `fig8_like.png`，确认图片已更新到正确目录
- 关键命令：
  - `conda run -n cmass_lens cmass-posterior-predictive posterior-predictive --run-dir /Users/liurongfu/Work/CMASS_lens_project/outputs/devauc/latest --sigma-table /Users/liurongfu/Work/CMASS_lens_project/data/external/jeans_deV_grid.h5 --output-dir /Users/liurongfu/Work/CMASS_lens_project/Posterior_predictive_test/results --candidate-pool-size 100000 --worker-processes 12`
  - `conda run -n cmass_lens cmass-posterior-predictive posterior-trends --run-dir /Users/liurongfu/Work/CMASS_lens_project/outputs/devauc/latest --sigma-table /Users/liurongfu/Work/CMASS_lens_project/data/external/jeans_deV_grid.h5 --output-dir /Users/liurongfu/Work/CMASS_lens_project/Posterior_predictive_test/results`
- 关键验证：
  - `chain.h5` 刷新前后 `mtime` 一致：`1773210178.3643317`
  - PPC 返回 `status=completed`，`n_replicates=192000`
  - trends 返回 `status=completed`，`n_posterior_draws=256`, `n_mass_bins=19`
  - 7 个目标工件都晚于本次执行开始时间 `2026-03-11 14:31:36 CST`

### 阶段 14：PPC 默认输出目录统一到项目根 output
- **状态：** 已完成
- **开始时间：** 2026-03-16
- 已执行动作：
  - 在 `cmass_posterior_predictive.predictive` 中新增 `DEFAULT_PPC_OUTPUT_ROOT_DIR`
  - 将 `run_posterior_predictive`、`run_posterior_trends`、`wait_for_external_sigma_tables_and_run` 的 `output_root_dir` 改为可选默认参数
  - 将 CLI 里 `posterior-predictive` / `posterior-trends` / `posterior-predictive-monitor` 的 `--output-dir` 从必填改为默认值
  - 在 CLI 帮助文案中显示默认输出路径
  - 更新测试以覆盖“省略 `--output-dir` 走默认目录”和“显式参数仍覆盖默认值”
- 关键验证：
  - `conda run -n cmass_lens pytest -q tests/test_cli_surface.py tests/test_posterior_predictive.py -k "public_defaults_use_1000_replicates or canonical_trend_defaults or standalone_cli_exposes_only_ppt_family_commands"` -> `3 passed`
  - `conda run -n cmass_lens pytest -q tests/test_cli_surface.py tests/test_posterior_predictive.py` -> `38 passed`
