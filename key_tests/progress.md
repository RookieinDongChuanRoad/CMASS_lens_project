# 进展日志

## 会话日期：2026-03-08

### 阶段 6：compare 长跑重生成
- **状态：** 进行中
- **开始时间：** 2026-03-08 22:10 CST
- 已执行动作：
  - 读取现有 `task_plan.md`、`findings.md`、`progress.md`，确认旧 compare 结果仍是 `80` 步。
  - 为 `compare=10000` 和旧结果归档补充失败测试。
  - 首轮测试按预期在导入阶段失败，缺失 `archive_existing_compare_artifacts()`。
  - 实现 `workspace_support.py` 中的长跑 compare 配置与归档逻辑。
  - 更新 `run_comparison.py`，在重跑前自动归档旧 compare 目录、report、manifest、PNG。
  - 更新 `reporting.py`，明确 `smoke` 只做链路检查，`compare` 是 `10000` 步正式长跑。
  - 更新 `notebooks/compare_corner.ipynb`，把 notebook 文案和摘要改成正式长跑 compare 语义，并标注旧 `80` 步结果已归档。
  - 重新运行 `pytest`，8 项测试全部通过。
  - 启动 `run_comparison.py` 长跑重生成；已确认旧 compare 报告、manifest、PNG 先归档到 `output/archive/compare_80step/`。
  - 运行中观测到 `reference / sersic / compare` 已进入 `spawn + 12 workers` 多进程阶段；backend 可读到实时步数。
  - 记录用户提供的 reference 止损规则：单次 `10000` 步通常 `1-2` 小时，若超过 `5` 小时且步数 `< 5000` 可停止。
  - 新增 `tests/test_compare_notebook.py`，先把“notebook 内直接显示图，同时保留 PNG 导出”写成失败测试。
  - 按预期看到失败：`plot_overlay()` 仍返回 `output_path`，不会在 notebook 单元格内渲染图像。
  - 更新 `notebooks/compare_corner.ipynb`，保留 `fig.savefig(...)`，并改为返回 `Figure` 供单元格直接显示。
  - notebook 结构测试重新通过。
- 新建/修改文件：
  - `tests/test_workspace_support.py`（更新）
  - `tests/test_compare_notebook.py`（新建）
  - `workspace_support.py`（更新）
  - `run_comparison.py`（更新）
  - `reporting.py`（更新）
  - `notebooks/compare_corner.ipynb`（更新）
  - `task_plan.md`（更新）
  - `findings.md`（更新）
  - `progress.md`（更新）

### 阶段 1：工作区与约束落盘
- **状态：** 进行中
- **开始时间：** 2026-03-08 21:45 CST
- 已执行动作：
  - 读取 `planning-with-files`、`test-driven-development`、`jupyter-notebook`、`verification-before-completion` 技能说明。
  - 确认 `key_tests` 当前为空目录，适合作为隔离工作区。
  - 核对当前实现真实入口为 `cmass_lens_inference.runner` / `cli`，链存储为 `emcee` HDF backend。
  - 核对参考实现最小必需依赖和 `cmass_lens` 环境可导入性。
  - 梳理 `inference.ipynb` 中 corner 图叠加方式，确定 notebook 输出方向。
- 新建/修改文件：
  - `task_plan.md`（新建）
  - `findings.md`（新建）
  - `progress.md`（新建）

### 阶段 2：测试先行的支撑层
- **状态：** 已完成
- 已执行动作：
  - 新增 `tests/test_workspace_support.py`，先为配置生成、参考实现入口映射和参数顺序转换写测试。
  - 首轮测试按预期失败，暴露 `workspace_support.py` 中缺失的常量和转换接口。
  - 实现 `workspace_support.py`，集中管理路径、模式、profile、参数标签和顺序转换。
  - 追加参数顺序测试，确认 reference 侧最后两维使用 `loga, theta0`。
- 新建/修改文件：
  - `tests/test_workspace_support.py`（新建并更新）
  - `workspace_support.py`（新建并更新）

### 阶段 3：双链路打通
- **状态：** 已完成
- 已执行动作：
  - 创建 `current_pipeline/`、`reference_pipeline/`、`configs/`、`output/`、`notebooks/`、`reports/` 目录。
  - 实现 `setup_workspace.py`，可复制参考脚本、刷新数据链接、生成当前实现专用 YAML 配置。
  - 实现 `current_pipeline/run_current_pipeline.py`，成功跑通 `sersic smoke`。
  - 实现 `reference_pipeline/run_reference_pipeline.py`，成功跑通 `sersic smoke`。
  - 实现 `reporting.py` 与 `run_comparison.py` 的基础编排逻辑。
- 新建/修改文件：
  - `setup_workspace.py`（新建）
  - `current_pipeline/run_current_pipeline.py`（新建）
  - `reference_pipeline/run_reference_pipeline.py`（新建）
  - `reporting.py`（新建）
  - `run_comparison.py`（新建）

### 阶段 4：运行与产物生成
- **状态：** 已完成
- 已执行动作：
  - 运行 `run_comparison.py`，完成 `current/reference x sersic/devauc x smoke/compare` 共 8 个 run。
  - 自动生成 `reports/pipeline_comparison.md` 与 `reports/run_manifest.json`。
  - 使用 `jupyter-notebook` 脚手架生成 `notebooks/compare_corner.ipynb`。
  - 为 notebook 填入 compare summary 读取、reference 参数重排、corner 叠加作图和 PNG 保存逻辑。
  - 安装 `nbformat`、`nbclient`、`nbconvert` 以执行 notebook。
  - 修复损坏的用户级 `python3` kernelspec，并安装 `cmass_lens` 专用 kernelspec。
  - 执行 notebook，生成 `output/figures/sersic_compare_corner.png` 与 `output/figures/devauc_compare_corner.png`。
- 新建/修改文件：
  - `reports/pipeline_comparison.md`（新建）
  - `reports/run_manifest.json`（新建）
  - `notebooks/compare_corner.ipynb`（新建并执行）
  - `output/figures/sersic_compare_corner.png`（新建）
  - `output/figures/devauc_compare_corner.png`（新建）

### 阶段 5：验证与交付
- **状态：** 已完成
- 已执行动作：
  - 验证 notebook、报告、manifest 与两张 PNG 全部存在。
  - 复核报告中的 run summary、raw steps、discard、flat samples 与 chain 路径。
  - 将 kernelspec 修复、参数顺序差异与实际 compare 步数写回 planning 文件。
- 新建/修改文件：
  - `task_plan.md`（更新）
  - `findings.md`（更新）
  - `progress.md`（更新）

## 测试结果
| 测试项 | 输入 | 预期结果 | 实际结果 | 状态 |
|--------|------|----------|----------|------|
| 当前依赖导入 | `conda run -n cmass_lens python -c "import emcee,h5py,numba,yaml,scipy,corner"` | 成功导入 | 成功导入 | ✓ |
| 参考依赖导入 | `conda run -n cmass_lens python -c "... import Population_model ..."` | 成功导入 | 成功导入 | ✓ |
| 支撑层单测 | `conda run -n cmass_lens pytest -q tests/test_workspace_support.py` | 测试通过 | 6 项通过 | ✓ |
| 长跑配置与归档单测 | `conda run -n cmass_lens pytest -q tests/test_workspace_support.py` | 测试通过 | 8 项通过 | ✓ |
| 工作区准备 | `conda run -n cmass_lens python setup_workspace.py` | 成功生成配置、复制文件、创建数据映射 | 成功 | ✓ |
| 当前 `sersic smoke` | `conda run -n cmass_lens python current_pipeline/run_current_pipeline.py --profile sersic --mode smoke` | 生成 run 目录、chain、summary | 成功 | ✓ |
| 参考 `sersic smoke` | `conda run -n cmass_lens python reference_pipeline/run_reference_pipeline.py --profile sersic --mode smoke` | 生成 HDF backend 与 summary | 成功 | ✓ |
| 全量编排 | `conda run -n cmass_lens python run_comparison.py` | 完成 8 个 run 并生成报告 | 成功 | ✓ |
| notebook 执行 | `conda run -n cmass_lens jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=cmass_lens .../compare_corner.ipynb` | 执行 notebook 并生成 PNG | 成功 | ✓ |
| 产物存在性检查 | 脚本检查 notebook/report/manifest/两张 PNG | 全部存在 | 全部存在 | ✓ |

## 错误日志
| 时间戳 | 错误 | 尝试次数 | 处理方式 |
|--------|------|----------|----------|
| 2026-03-08 21:55 CST | `ImportError: cannot import name 'CURRENT_INITIAL_CENTER'` | 1 | 实现 `workspace_support.py` 中的常量与配置函数 |
| 2026-03-08 21:58 CST | `ImportError: cannot import name 'CURRENT_PARAMETER_ORDER'` | 1 | 在支撑模块中补充 current/reference 参数顺序与转换函数 |
| 2026-03-08 22:06 CST | `json.decoder.JSONDecodeError` during `jupyter nbconvert` kernel discovery | 1 | 修复 `~/.local/share/jupyter/kernels/python3/kernel.json` 并安装 `cmass_lens` kernelspec |
| 2026-03-08 22:12 CST | `ImportError: cannot import name 'archive_existing_compare_artifacts'` | 1 | 在 `workspace_support.py` 中新增 compare 归档接口，并重新运行测试 |
|        |      | 1        |          |

## 5 个重启检查问题
| 问题 | 回答 |
|------|------|
| 我现在在哪一步？ | Phase 6 |
| 我接下来要去哪？ | 先重新验证 smoke，再运行 4 个 `10000` 步 compare，并重生成报告、notebook 与 PNG |
| 我的目标是什么？ | 在 `key_tests` 中把正式 compare 升级为长跑版本，同时归档旧 80 步结果 |
| 我学到了什么？ | compare 的 runtime、归档和下游报告读取都可以通过 `workspace_support.py` 与 `run_comparison.py` 集中控制 |
| 我已经做了什么？ | 已完成工作区、旧版 8 个 run、长跑配置与归档实现、并通过新的支撑层测试 |
