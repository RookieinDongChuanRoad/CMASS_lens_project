# 任务计划：双 pipeline 长跑关键测试工作区

## 目标
在 `key_tests` 中维护隔离测试工作区，统一使用 `CMASS_lens_project/data` 数据，并把 4 个正式 `compare` run 升级为 `10000` 步长跑版本；旧的 `80` 步 compare 结果保留但归档，不再占用 canonical compare 路径。

## 当前阶段
Phase 6

## 阶段计划

### Phase 1: 工作区与约束落盘
- [x] 创建 `task_plan.md`、`findings.md`、`progress.md`
- [x] 建立 `key_tests` 固定目录结构
- [x] 明确当前实现与参考实现的最小可运行入口
- **Status:** complete

### Phase 2: 测试先行的支撑层
- [x] 为工作区配置生成与参考实现入口映射编写失败测试
- [x] 实现工作区支撑模块
- [x] 记录关键路径和运行假设
- **Status:** complete

### Phase 3: 双链路打通
- [x] 接通当前 pipeline wrapper
- [x] 复制参考实现最小必需文件并接通 driver
- [x] 将两套实现都改为输出到 `key_tests/output`
- **Status:** complete

### Phase 4: 首轮运行与产物生成
- [x] 完成当前/参考两套 pipeline 的 smoke runs
- [x] 完成 `sersic` / `devauc` 四组旧 compare runs（`80` 步，现已过时）
- [x] 生成 notebook 与 PNG corner 图
- **Status:** complete

### Phase 5: 首轮验证与交付
- [x] 验证所有运行产物可读
- [x] 生成结构化对比报告
- [x] 汇总差异、风险和实际运行配置
- **Status:** complete

### Phase 6: compare 长跑重生成
- [x] 先用测试锁定 `compare = 10000/2000/500`
- [x] 实现旧 compare 产物自动归档规则
- [ ] 重新验证 smoke 链路未被破坏
- [ ] 完成 4 个 `compare` 长跑 run
- [ ] 重新生成报告、manifest、notebook 与 PNG
- [ ] 复核所有 canonical compare 路径只包含 `10000` 步结果
- **Status:** in_progress

## 关键问题
1. 如何在不改动参考实现核心数学逻辑的前提下，让它稳定读取 `CMASS_lens_project/data` 并把输出写入 `key_tests/output`？
2. 当前实现的输出目录结构如何包装，才能既复用原 runner 又保持 `key_tests` 下的结果可比和可追踪？
3. notebook 如何稳定读取两套不同格式的链文件，并在同一张 corner 图中做清晰标注？

## 已做决策
| 决策 | 理由 |
|------|------|
| 本轮 planning 文件直接写在 `key_tests` 根目录 | 用户明确要求不要污染原仓库中的 planning 文件 |
| 参考实现采用“最小复制 + 独立运行” | 这样既能隔离输出，又能保留参考实现原始逻辑 |
| 当前实现不复制源码，只通过 wrapper 调用原仓库 | 减少重复代码和后续维护成本 |
| 先 smoke，再 compare 短跑 | 先验证链路，再做可比输出，风险更低 |
| 参考实现起始参数向量必须按旧顺序重排为 `... loga, theta0` | 旧 `Population_*` 与当前实现的最后两维顺序不同，必须显式转换 |
| 当前与参考运行都额外写 summary JSON | 这样 notebook 与报告可直接消费统一元数据，而不用猜目录结构 |
| 4 个正式 `compare` run 固定为 `10000` 步，统一 `discard=2000` | 用户明确要求 compare 不能再是几十步 |
| `smoke` 继续保持 4 步 | 仅用于链路打通，不参与科学比较 |
| 旧 `80` 步 compare 结果必须先归档再重跑 | 避免新旧结果混在 canonical compare 路径中 |
| 修复用户级 `python3` kernelspec 后再执行 notebook | 否则 `nbconvert` 会在 kernel 发现阶段因损坏 JSON 失败 |

## 遇到的问题
| 错误 | 尝试次数 | 处理方式 |
|------|----------|----------|
| 为参考实现复用当前初值时，发现参数顺序与当前实现不一致 | 1 | 在 `workspace_support.py` 中显式定义 current/reference 参数顺序并提供转换函数 |
| `nbconvert` 初次执行因损坏的用户级 `python3` kernelspec 失败 | 1 | 修复 `~/.local/share/jupyter/kernels/python3/kernel.json`，并额外安装 `cmass_lens` kernelspec |
| compare 旧结果只有 80 步，不满足新的正式对比约束 | 1 | 将 compare runtime 提升到 `10000/2000/500`，并在重跑前归档旧结果 |
|      | 1        |          |

## 备注
- 所有新增代码和产物都放在 `key_tests` 下。
- 每完成一个阶段都更新状态并补充 `findings.md` 与 `progress.md`。
- 真正宣称完成前，必须重新运行关键验证命令。
