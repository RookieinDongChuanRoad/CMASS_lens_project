# Sonnenfeld reference 数值比较规划

目标：阅读 `/Users/liurongfu/reference_codes/strong_lensing_tools/papers/slacs_selection`，明确当前本地 Sonnenfeld 实现应如何与外部 reference 做数值比较。

### Phase 1: reference 入口与工件盘点
**Status:** complete

- 识别主推断脚本、对照脚本、grid 生成脚本和已落盘 HDF5 工件。
- 明确哪些 reference 文件能作为数值 oracle，哪些只是生成数据准备产物。

### Phase 2: 数学合同映射
**Status:** complete

- 对齐参数命名、单位约定、latent variables、selection correction、per-lens likelihood、normalization。
- 找出当前本地实现与 reference 的已知差异，避免把非等价量拿来直接比较。

### Phase 3: 最小数值比较方案
**Status:** complete

- 规划从 primitive/grid 级到 full log-prob 级的分层 comparison harness。
- 明确输入数据、随机数、容差、输出报告和不应修改的边界。

## Errors Encountered

| Error | Attempt | Resolution |
|-------|---------|------------|
| 无 | - | - |
