# Task Plan: 插值网格重构与数据落地

## Goal
将本项目整理为一个可长期维护的 Python 模块 + CLI 工具，在 `cmass_lens` 标准环境下稳定重算并覆盖目标 HDF5 文件中的 `m5_grid`、`dm5_dthetaein_grid` 与符合条件的 `s2_grid`，同时保留足够清晰的测试、文档和运行记录，便于后续继续维护。

## Current Phase
Phase 5

## Phases
### Phase 1: 需求澄清与环境摸底
- [x] 阅读 `project.md`，明确物理目标、输入输出和覆盖规则
- [x] 阅读外部参考实现与 `spherical_jeans` 依赖
- [x] 检查真实 HDF5 文件结构、字段名、样本数量和 `s2_grid` 覆盖范围
- [x] 确认标准运行环境为 `cmass_lens`
- **Status:** complete

### Phase 2: 方案设计与工程结构
- [x] 明确交付形态为“模块 + CLI”，不是一次性脚本
- [x] 明确 HDF5 默认写回策略为“先安全输出，再替换”
- [x] 将计算逻辑拆分为配置、物理计算、HDF5 I/O、CLI 四层
- [x] 明确参考脚本兼容测试与生产策略测试要分离
- **Status:** complete

### Phase 3: 核心实现
- [x] 实现 `m5_grid` 与 `dm5_dthetaein_grid` 的本地计算模块
- [x] 实现 `s2_grid` 的自由 Sersic / deV 两条分支
- [x] 实现 HDF5 安全读写、按 group 过滤、按需更新 `s2_grid`
- [x] 实现 `cmass_lens` 环境检查入口与 README / `environment.yml`
- **Status:** complete

### Phase 4: 测试、验证与数据更新
- [x] 建立 `m5` 参考脚本一致性测试
- [x] 建立真正的自由 Sersic `s2_grid` 参考脚本对照测试
- [x] 建立 `s2_grid` 生产策略测试，锁定 `aperture_width=1.6 arcsec` 与 `seeing=0.9 arcsec`
- [x] 在 `cmass_lens` 下通过完整测试套件
- [x] 先生成安全输出文件，再原位覆盖 `data/raw/` 中两份目标文件
- **Status:** complete

### Phase 5: 交付、记录与后续可继续项
- [x] 将关键发现、决策、验证证据写入 planning files
- [x] 记录已知风险和外部依赖约束
- [x] 标记当前已完成状态与后续可继续动作
- **Status:** complete

## Key Questions
1. 标准环境是否明确唯一？答案：是，唯一标准环境为 `cmass_lens`。
2. `dm5` 导数字段名应该用哪个？答案：以真实文件为准，使用 `dm5_dthetaein_grid`。
3. `s2_grid` 的旧脚本兼容与新业务 aperture 策略是否应混为一谈？答案：不应混用，必须拆为“参考脚本兼容”与“生产策略”两类测试。
4. `seeing` 是否随 aperture 切换而变化？答案：不变化，固定 `0.9 arcsec`。

## Decisions Made
| Decision | Rationale |
|----------|-----------|
| 使用 `planning-with-files` 在项目根目录持久化状态 | 避免上下文丢失，便于后续接手 |
| 交付形态为模块 + CLI | 更适合长期维护、重复运行与测试 |
| `cmass_lens` 为唯一标准环境 | 降低运行歧义，保证验证口径一致 |
| `spherical_jeans` 保持为外部依赖 | 避免在本项目中维护一份副本，但要求环境必须能导入 |
| HDF5 默认采用安全写回流程 | 先生成校验过的新文件，再替换正式文件，降低数据损坏风险 |
| `m5` 与 `dm5` 优先保持参考实现兼容 | 当前验收重点是物理定义正确和历史兼容，而不是性能 |
| 真实导数字段名使用 `dm5_dthetaein_grid` | 真实文件已经采用该字段名，优先兼容现有数据 |
| `s2_grid` 参考测试与生产策略测试分离 | 旧脚本基线是 `0.8 arcsec`，而新业务设定是 `1.6 arcsec`，两者不能混成一个断言 |
| 生产 `s2_grid` 统一采用 `aperture_width=1.6 arcsec` | 这是用户明确要求的新业务设定 |
| 生产 `s2_grid` 保持 `seeing=0.9 arcsec` | 避免 aperture 变更时再引入第二个变量 |

## Errors Encountered
| Error | Attempt | Resolution |
|-------|---------|------------|
| 当前目录不是 git 仓库，`git status` 无法作为状态来源 | 1 | 改为直接读取文件系统与 HDF5 内容 |
| 某些 HDF5 检查命令在会话包装层下长时间无输出 | 1 | 改用更小范围、边跑边返回的检查方式 |
| 初始实现缺少 `group_names` 过滤能力 | 1 | 先补失败测试，再实现按 group 处理 |
| `cmass_lens` 环境缺少 `pytest`，导致环境契约不完整 | 1 | 在 `cmass_lens` 中安装 `pytest` 并重新验证 |
| 清理中间 `.updated.hdf5` 副本时曾被命令策略拦截 | 1 | 先保留副本并确认正式目标文件与之完全一致；后续如需清理可单独执行 |

## Notes
- 当前实现、测试、文档、环境契约和目标数据文件更新都已经完成。
- `data/raw/` 中两份正式目标文件已经切换到新的 `1.6 arcsec` 生产 aperture 规则。
- 由于使用了新的 aperture 规则，当前正式 `s2_grid` 结果不再与旧历史文件中的 `0.8 arcsec` 基线一致，这属于预期业务变化，不是回归失败。
- 仍有一个来自外部依赖 `spherical_jeans` 的 `VisibleDeprecationWarning`；这是上游兼容性提醒，不影响本地测试通过。
