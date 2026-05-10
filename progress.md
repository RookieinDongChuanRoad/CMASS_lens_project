# Sonnenfeld reference 数值比较进度

## 2026-05-10 reference comparison 目标文档

- 新增 `Bayesian_inference/docs/sonnenfeld_reference_comparison_target.md`。
- 文档把“语义修正目标”和“外部 reference 数值对比目标”分开：前者约束本地模型应当是什么，后者约束怎样分层证明本地实现与 reference 可比。
- 计划采用 audit -> schema/subprocess -> 参数映射 -> primitive -> data/grid -> normalization -> FP prior -> per-lens -> full posterior decomposition 的顺序。
- 明确第一阶段只比较 `sonnenfeld2024_slacs` 的 paper-native `legacy_fixed_kpc/m5` 变体，不把 h-unit 平移混入首轮 reference comparison。

## 2026-05-09 proposal / parent 分布专项审计

- 只读对照了外部 reference `scripts/fit_full.py`、`scripts/mz_distribution.py`、`scripts/fitpars.py` 与本地 `sonnenfeld2024_slacs/posterior.py`、`preprocessing.py`。
- 关键结论：本地当前 proposal importance sampling 不应继续作为 Sonnenfeld 科学模型的长期实现；修正计划应把 parent sample 或 parent CDF 生成前移到 preprocessing/data contract，posterior hot path 只消费物理 parent sample。
- 还发现同类公式偏差：parent redshift volume factor、source-z truncation/good mask、grid out-of-bounds handling、FP prior 默认常数。

## 2026-05-09

- 启动 reference 代码阅读与规划。
- 已确认外部 reference 目录存在 `scripts/`、`SLACS_table.cat`、posterior predictive outputs 和 inference HDF5。
- 已细读 `fit_full.py`、`fit_slonly.py`、`fitpars.py`、`parent_sample_pars.py`、`make_crosssect_grid.py`、`make_slacs_lensing_grids.py`、`make_slacs_jeans_grids.py`、`read_slacs.py` 的关键路径。
- 已检查 reference HDF5 schema；chain 文件只有 posterior samples/logp/fpfit，不包含中间 grids。
- 已验证 `cmass_lens` 环境加 `PYTHONPATH=/Users/liurongfu/reference_codes/strong_lensing_tools:/Users/liurongfu/tools` 后可导入 reference 所需核心模块。
- 已确认本地已有 CMASS main comparison harness，可复用隔离 subprocess/report 模式，但不能直接比较 Sonnenfeld reference。
- 已形成分层数值比较计划：data-prep/grid oracle -> scalar primitive oracle -> per-lens likelihood term oracle -> normalization oracle -> full posterior trend sanity check。
