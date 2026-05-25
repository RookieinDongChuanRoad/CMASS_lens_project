# 发现与决策

## 需求
- 在 `key_tests` 下建立独立测试工作区，不修改原始 `Bayesian_inference` 根目录下已有 planning 文件。
- 打通两套 pipeline：
- 当前实现来自 `Bayesian_inference/src/cmass_lens_inference`
- 参考实现来自 `Desktop/Spectrum_reduction`，以最小复制方式放入 `key_tests/reference_pipeline`
- 两套 pipeline 统一使用 `CMASS_lens_project/data` 下的数据文件。
- 所有输出统一放在 `key_tests/output` 下。
- 需要同时覆盖 `sersic` 与 `devauc` 两个 profile。
- 需要执行 smoke runs 与 compare runs。
- 需要生成对比报告与叠加 corner 图 notebook，并输出 PNG。

## 调研发现
- `key_tests` 初始为空目录，适合作为完全隔离的工作区。
- 当前实现可通过 `cmass_lens_inference.cli` 或 `runner.run_inference()` 启动，链存储使用 `emcee.backends.HDFBackend`。
- 当前实现支持 `parallel_strategy`、`num_threads`、`reserve_cores` 等并行控制项。
- 参考实现依赖 `Population_model.py`、`Population_deV_model.py`、`Foreground_*`、`constants*.py`，以及 `/Users/liurongfu/tools/numba_friendly` 与 `/Users/liurongfu/tools/dangular_grid.h5`。
- `cmass_lens` 环境已经可以导入当前实现依赖、参考实现依赖以及 `corner`。
- `inference.ipynb` 的 corner 画法核心是叠加 `corner.corner(..., fig=fig, color=...)`，适合复用为对比 notebook。
- 当前实现的 `runner.run_inference()` 已成功在 `key_tests/output/current/sersic/smoke/...` 下生成真实 run 目录与 `chain.h5`。
- 参考实现 driver 已成功在 `key_tests/output/reference/sersic/smoke/reference_chain.h5` 生成 HDF backend。
- 当前实现参数顺序是 `... theta0, loga`，参考实现旧脚本实际期望 `... loga, theta0`。
- 全量编排 `run_comparison.py` 已成功完成 8 个 run，但旧 compare 结果只有 `80` 步，现已被判定为过时链路验证结果。
- notebook 执行前，Jupyter 被用户级 `python3` kernelspec 中损坏的 JSON 阻断；修复后已可正常执行 notebook。
- `compare_corner.ipynb` 已执行完成，并在 `output/figures/` 生成 `sersic_compare_corner.png` 与 `devauc_compare_corner.png`。
- `workspace_support.py` 是 compare runtime 的单一真源；把 `MODE_SETTINGS["compare"]` 改成 `10000/2000/500/2000` 后，当前实现 YAML 和参考实现 driver 会同步继承新配置。
- 旧 compare 结果的安全处理方式是按原相对路径移动到 `output/archive/compare_80step/`；这样既保留证据链，又能保证 canonical compare 路径只代表长跑结果。
- 用户给出的运行经验可作为 reference 长跑止损规则：单次 `reference` `10000` 步通常在 `1-2` 小时附近；若超过 `5` 小时，则先查看已跑步数，若 `< 5000` 步可直接停止。
- `compare_corner.ipynb` 原先只把 corner 图保存到 `output/figures/`，因为 `plot_overlay()` 返回的是文件路径而不是 `Figure`；要在 notebook 内直接展示图片，最小改动是继续 `savefig()`，但把 `Figure` 返回给单元格输出。

## 技术决策
| 决策 | 理由 |
|------|------|
| 使用 `workspace_support.py` 统一封装路径、配置与标签规则 | 避免这些约定分散在多个脚本中，方便测试 |
| 为当前实现生成专用 YAML 配置，而不是直接改原始配置 | 保持原仓库配置不变，同时精确控制输出路径和步数 |
| 为参考实现编写参数化 driver，而不是直接复用原 `run_mcmc.py` | 原脚本硬编码 backend 路径和函数选择，不适合隔离工作区 |
| 参考实现通过 `reference_pipeline/data` 中的数据映射读取 CMASS 数据 | 这样无需篡改 `Population_*` 中的相对路径假设 |
| 用 summary JSON 作为 run 产物的统一索引层 | notebook 和报告都需要稳定地读取不同实现生成的链文件 |
| notebook 中必须先把 reference chain 重排为 current 参数顺序再叠加作图 | 否则 corner 图最后两维会被错误对齐 |
| 通过 `jupyter nbconvert --execute --ExecutePreprocessor.kernel_name=cmass_lens` 执行 notebook | 这样能稳定使用目标 conda 环境并生成 PNG 产物 |
| compare 长跑 burn-in 统一用 `discard=2000` | notebook、report、manifest 需要基于同一剔除规则读取长跑链 |
| 旧 compare 的 report / manifest / PNG 也必须一起归档 | 它们描述的是 80 步结果，不能继续留在默认位置误导后续比较 |

## 遇到的问题
| 问题 | 处理方式 |
|------|----------|
| 参考实现依赖 `numba_friendly` 这类工作区外资源 | 通过 driver 明确插入 `/Users/liurongfu/tools` 到 `sys.path` |
| 参考实现参数顺序与当前实现不一致 | 通过支撑模块集中定义 `CURRENT_PARAMETER_ORDER`、`REFERENCE_PARAMETER_ORDER` 和重排函数 |
| 用户级 `python3` kernelspec JSON 损坏 | 修复 `kernel.json` 语法并安装 `cmass_lens` 专用 kernelspec |
| 旧 compare 产物已经占用 canonical 路径 | 在运行新 compare 之前增加自动归档步骤，把旧目录、report、manifest、PNG 移到 `output/archive/compare_80step/` |

## 资源
- 当前实现根目录：`/Users/liurongfu/Work/CMASS_lens_project/Bayesian_inference`
- 参考实现根目录：`/Users/liurongfu/Desktop/Spectrum_reduction`
- 数据根目录：`/Users/liurongfu/Work/CMASS_lens_project/data`
- 工具根目录：`/Users/liurongfu/tools`
- notebook 参考：`/Users/liurongfu/Desktop/Spectrum_reduction/inference.ipynb`
- 当前 smoke summary：`/Users/liurongfu/Work/CMASS_lens_project/key_tests/output/current/sersic/smoke/current_run_summary.json`
- 参考 smoke summary：`/Users/liurongfu/Work/CMASS_lens_project/key_tests/output/reference/sersic/smoke/reference_run_summary.json`
- 全量运行清单：`/Users/liurongfu/Work/CMASS_lens_project/key_tests/reports/run_manifest.json`
- 对比报告：`/Users/liurongfu/Work/CMASS_lens_project/key_tests/reports/pipeline_comparison.md`
- 对比 notebook：`/Users/liurongfu/Work/CMASS_lens_project/key_tests/notebooks/compare_corner.ipynb`
- PNG 图像：`/Users/liurongfu/Work/CMASS_lens_project/key_tests/output/figures/`
- compare 归档根目录：`/Users/liurongfu/Work/CMASS_lens_project/key_tests/output/archive/compare_80step`

## 可视化/浏览器发现
- `inference.ipynb` 使用 `corner` 直接从 HDF backend 读取链并叠加不同结果，适合作为本轮 notebook 的视觉基线。
