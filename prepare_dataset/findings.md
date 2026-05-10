# Findings & Decisions

## Requirements
- 将 `project.md` 中描述的插值网格准备流程落地为可维护的本地项目，而不是继续依赖散落的参考脚本。
- 对每个 galaxy 更新 `mass_definitions/{m5,m10}/mass_grid` 与 `mass_definitions/{m5,m10}/dmass_dthetaein_grid`。
- 仅对 `num_sigma > 0` 的 galaxy 更新 `mass_definitions/{m5,m10}/s2_grid`。
- 输出文件保持原有 HDF5 分组结构；root-level 只保留 `gamma_grid`，不再保留 root-level `m5_grid`、`dm5_dthetaein_grid` 或 `s2_grid`。
- 标准环境必须统一为 `cmass_lens`。
- 代码需要具备较高可读性、充足注释和可继续交接能力。
- 测试必须包含参考脚本一致性验证，而不是只做自洽验证。

## Research Findings
- 当前项目目录不是 git 仓库，不能依赖 git 元数据管理进度。
- 物理计算的关键外部依赖位于 `/Users/liurongfu/tools/spherical_jeans`。
- 主要参考脚本有两个：
  - `/Users/liurongfu/Desktop/Spectrum_reduction/make_m5_grids.py`
  - `/Users/liurongfu/Desktop/Spectrum_reduction/make_jeans_grid.py`
- 原始数据目录是 `/Users/liurongfu/Work/CMASS_lens_project/data/raw/`。
- 当前目标文件有两份：
  - `observations_with_m5_grids_all.hdf5`
  - `observations_deV_with_m5_grids.hdf5`
- 两份文件各有 23 个 group，其中 7 个 group 带有 `s2_grid`。
- 真实 HDF5 中导数字段名是 `dm5_dthetaein_grid`，不是 `project.md` 里的 `dm5_dtheta_ein_grid`。
- 自由 Sersic 分支依赖 `re_arcsec` 与 `nser`。
- deV 分支依赖 `reff_deV`。
- `spherical_jeans.sigma_model.sigma2` 返回的是 `sigma^2 / G`，所以需要显式单位换算回 `km^2/s^2`。
- 在旧历史路径里，带 `s2_grid` 的样本 `aperture_width` 全都是 `0.8 arcsec`。
- 真正的自由 Sersic `make_jeans_grid.py` 参考路径使用：
  - 数据中的 `aperture_width`
  - 固定 `aperture_height = 0.9 arcsec`
  - 固定 `seeing = 0.9 arcsec`
- 用户随后要求正式生产逻辑改成：
  - `aperture_width = 1.6 arcsec`
  - `aperture_height = 0.9 arcsec`
  - `seeing = 0.9 arcsec`
- 生产 aperture 从 `0.8` 切到 `1.6 arcsec` 后，样例 galaxy 的 `s2_grid` 变化量约为 8%，属于明显物理设定变更，而非数值噪声。
- `cmass_lens` 最初缺少 `pytest`，后来已补齐，因此现在标准环境既能运行也能完成验证。
- 在 `cmass_lens` 中运行时会出现来自 `spherical_jeans` 的 `VisibleDeprecationWarning`，这是外部依赖行为，不是本地实现错误。

## Technical Decisions
| Decision | Rationale |
|----------|-----------|
| 将代码组织为配置 / 物理计算 / HDF5 I/O / CLI 四层 | 降低耦合，便于测试和维护 |
| 用本地模块封装旧脚本逻辑，而不是继续复制脚本改写 | 让行为可测试、可复用、可文档化 |
| `m5_grid` 与 `dm5_dthetaein_grid` 先严格兼容参考脚本 | 当前优先级是正确性和历史一致性 |
| 新增 `python -m prepare_dataset.env_check` | 让环境验证成为明确的项目入口 |
| 用 `environment.yml` 明确声明 `cmass_lens` 所需依赖 | 防止标准环境只存在于口头约定 |
| 将自由 Sersic 的真正脚本对照测试单独实现 | 现有 raw HDF5 结果只能算历史产物，不足以证明与脚本本身一致 |
| 保留 deV 分支的独立业务逻辑测试 | 该分支并没有同等直接的旧脚本基线可引用 |
| 将生产 aperture 规则切换为固定 `1.6 arcsec` | 这是用户明确要求的新物理设定 |
| 将 `seeing` 固定为 `0.9 arcsec`，不随 aperture 改动 | 避免两个变量同时漂移，便于解释结果变化 |
| 数据文件更新坚持“先安全输出，再原位替换” | 防止大文件批处理过程中写坏正式数据 |

## Issues Encountered
| Issue | Resolution |
|-------|------------|
| 规格文本与真实 HDF5 字段名不一致 | 以真实文件 schema 为准，并在记录中显式写明 |
| 历史 `s2_grid` 与新业务 aperture 目标冲突 | 把“参考脚本兼容测试”和“生产策略测试”拆开 |
| 目标环境最初只能运行代码，不能运行测试 | 在 `cmass_lens` 中补装 `pytest` 并重新做整体验证 |
| 旧的 planning files 已混合“历史口径”和“新口径” | 重写为统一、结构化、可恢复的版本 |

## Resources
- 项目说明：`/Users/liurongfu/Work/CMASS_lens_project/prepare_dataset/project.md`
- 计划文件：`/Users/liurongfu/Work/CMASS_lens_project/prepare_dataset/task_plan.md`
- 发现记录：`/Users/liurongfu/Work/CMASS_lens_project/prepare_dataset/findings.md`
- 进度日志：`/Users/liurongfu/Work/CMASS_lens_project/prepare_dataset/progress.md`
- 本地包入口：`/Users/liurongfu/Work/CMASS_lens_project/prepare_dataset/prepare_dataset/cli.py`
- HDF5 处理逻辑：`/Users/liurongfu/Work/CMASS_lens_project/prepare_dataset/prepare_dataset/io/hdf5.py`
- `s2_grid` 生产逻辑：`/Users/liurongfu/Work/CMASS_lens_project/prepare_dataset/prepare_dataset/physics/jeans.py`
- `m5` 相关逻辑：`/Users/liurongfu/Work/CMASS_lens_project/prepare_dataset/prepare_dataset/physics/m5.py`
- 环境检查入口：`/Users/liurongfu/Work/CMASS_lens_project/prepare_dataset/prepare_dataset/env_check.py`
- 标准环境声明：`/Users/liurongfu/Work/CMASS_lens_project/prepare_dataset/environment.yml`
- 运行文档：`/Users/liurongfu/Work/CMASS_lens_project/prepare_dataset/README.md`
- Jeans 依赖：`/Users/liurongfu/tools/spherical_jeans/`
- 历史 Jeans 参考脚本：`/Users/liurongfu/Desktop/Spectrum_reduction/make_jeans_grid.py`
- 历史 `m5` 参考脚本：`/Users/liurongfu/Desktop/Spectrum_reduction/make_m5_grids.py`

## Visual/Browser Findings
- 本任务未依赖浏览器、图片或 PDF 视觉检查。
