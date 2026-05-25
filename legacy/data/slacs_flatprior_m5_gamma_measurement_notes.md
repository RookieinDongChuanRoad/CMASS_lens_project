# SLACS Fig. 8 中 `m5` / `gamma` 观测值与误差条的计算说明

一句话结论：Fig. 8 里每个 lens 的 `m5` 和 `gamma` 误差条，来自同一个 **per-lens flat-prior posterior**；它们不是两个彼此独立的单独推断结果。

## 这份文档要解决什么问题

这份说明文档面向另一个需要接手实现或复核这部分逻辑的 agent。它的目标不是复述整篇论文，而是把下面这个问题讲成可以直接复现的算法：

> Sonnenfeld (2024) Fig. 8 里画出来的 `gamma` 观测值与误差条，以及对应的 `m5` 观测值与误差条，到底是怎么从每个 SLACS lens 的观测量和预计算网格里得到的？

如果只记一个结论，应记住：

- `gamma` 和 `m5` 都不是从 population-level hierarchical posterior 直接读出来的。
- 它们来自每个 lens 独立做的一次 flat-prior inference。
- 这次 inference 的核心思路是：把精确测得的 Einstein radius 当作约束，把二维 `(m5, gamma)` 空间压缩到一条 `m5(gamma)` 曲线上，然后只沿这条曲线用 `sigma_obs` 做一维 posterior。

## 来源

### 论文来源

参考论文：

- Sonnenfeld, 2024, *The SLACS strong lens sample, debiased*
- 本地 PDF 路径：
  [Sonnenfeld - 2024 - The SLACS strong lens sample, debiased.pdf](/Users/liurongfu/Zotero/storage/98E4LV5X/Sonnenfeld%20-%202024%20-%20The%20SLACS%20strong%20lens%20sample,%20debiased.pdf)

和本问题直接相关的论文表述有两处：

- 正文中说明：对每个 lens，用观测到的 Einstein radius 和 aperture velocity dispersion，在 `(m5, gamma)` 空间建立约束，并且假设 `m5` 与 `gamma` 上是 flat prior。
- Fig. 8 图注说明：图上的 error bars 是对 SLACS lenses “assuming a flat prior on both `m5` and `gamma`” 得到的测量值。

### 代码来源

这部分“观测值与误差条”的实现，不在 Fig. 8 的画图脚本本身，而在外部仓库对单 lens flat-prior grid 的准备和读取逻辑里：

- `strong_lensing_tools/papers/slacs_selection/scripts/get_slacs_flatprior_grids.py`
- `strong_lensing_tools/papers/slacs_selection/scripts/read_slacs.py`

这两个脚本分别负责：

- `get_slacs_flatprior_grids.py`
  - 读取 lensing grid 和 Jeans grid
  - 对每个 lens 构造一维 `logp_grid`
  - 把 `m5_grid`、`logp_grid` 和 `sigma_grid` 写入 `slacs_flatprior_grids.hdf5`
- `read_slacs.py`
  - 读取 `slacs_flatprior_grids.hdf5`
  - 把 `logp_grid` 归一化成 posterior
  - 提取 `gamma` 和 `m5` 的中位数与 68% credible interval

## 输入数据与中间量

对每个 lens，这套方法依赖三类输入。

### 1. Lensing grid

来自 `slacs_lensing_grids.hdf5`，关键量有：

- `m5_grid(gamma)`
- `dm5drein_grid(gamma)`

这里的含义是：

- 固定该 lens 的几何信息和观测 Einstein radius 后，对每个 `gamma`，都能求出一个与该 Einstein radius 一致的 `m5`
- 于是原本二维的 `(m5, gamma)` 空间，被压缩为一条一维曲线 `m5 = m5(gamma)`
- `dm5drein_grid` 是变量替换时出现的 Jacobian 因子，需要进入 posterior

### 2. Jeans grid

来自 `slacs_jeans_grids.hdf5`，关键量是：

- `s2_grid(gamma)`

它表示在该 lens 已知的光度学和动力学设定下，给定 `gamma` 时的单位质量归一化动力学响应。代码里后续会将它与 `10**m5` 相乘，从而恢复该 lens 对应的物理 `sigma_model^2`。

### 3. 观测量

来自 `SLACS_table.cat`，这里真正进入单-lens flat-prior posterior 的观测量是：

- `sigma_obs`
- `sigma_err`

Einstein radius 也被用到了，但它不是作为一个带误差的 likelihood 项再参与拟合；在这套实现里，它被视为精确约束，用来先把 `(m5, gamma)` 压缩成 `m5(gamma)`。

## 算法流程

下面的流程就是外部脚本真正做的事。

### Step 1. 用 Einstein radius 约束得到 `m5_grid(gamma)`

对于每个 lens，lensing grid 已经预先存好了这样一条曲线：

`m5_grid = m5(gamma | observed Einstein radius)`

这一步的物理意义是：

- Einstein radius 被视为高精度测量
- 对每一个候选 `gamma`，都能反解出一个与观测 Einstein radius 一致的 `m5`

因此，从这一步开始，不再是在完整二维 `(m5, gamma)` 平面上算 posterior，而是沿着这条一维曲线算 posterior。

### Step 2. 用 Jeans grid 计算沿曲线的 `sigma_model(gamma)`

`get_slacs_flatprior_grids.py` 里对应的代码逻辑是：

```python
s2_model = 10.**m5_grid * s2_spline(gamma_grid)
sigma_model = s2_model**0.5
```

它对应的公式是：

`sigma_model(gamma) = sqrt(10**m5_grid(gamma) * s2_grid(gamma))`

这一步的含义是：

- `s2_grid(gamma)` 提供单位质量归一化的动力学响应
- `10**m5_grid(gamma)` 提供质量归一化
- 两者相乘后得到 `sigma_model^2`
- 再开方得到 `sigma_model`

所以，`sigma_model` 不是自由拟合出来的，而是由当前 `gamma` 所对应的 `m5(gamma)` 和 Jeans 响应共同决定的。

### Step 3. 构造一维 posterior

脚本中真正写入 `logp_grid` 的表达式是：

```python
logp_grid = -0.5 * (sigma_model - sigma_obs)**2 / sigma_err**2 \
            - np.log(sigma_err) \
            - np.log(dm5drein_grid)
```

可写成：

`logp_grid = -0.5 * (sigma_model - sigma_obs)^2 / sigma_err^2 - log(sigma_err) - log(dm5drein_grid)`

各项含义如下：

- 第一项：`sigma_model` 和观测 `sigma_obs` 的 Gaussian likelihood
- 第二项：Gaussian likelihood 的归一化项，代码里保留了 `-log(sigma_err)`
- 第三项：Jacobian 项，不是额外先验，而是从原始变量变换到当前一维参数化时需要补上的项

### Step 4. 归一化得到 posterior `p_grid`

`read_slacs.py` 中的做法是：

```python
logp_grid -= logp_grid.max()
p_grid = np.exp(logp_grid)
p_grid /= p_grid.sum()
p_cumsum = p_grid.cumsum()
```

这样做的原因是：

- 先减去 `max(logp_grid)`，防止指数化时数值下溢或上溢
- 再归一化成离散 posterior 概率
- 最后取 cumulative sum，方便提取中位数和 16%-84% credible interval

### Step 5. 提取 `gamma` 的观测值和误差

`gamma` 的“观测值”不是某个单独的外部 catalog 值，而是这个 posterior 的摘要统计量。

实现方式是：

- 中位数：找到 `p_cumsum` 最接近 `0.5` 的网格点
- 下误差和上误差：取 `16%` 到 `84%` credible interval

代码逻辑相当于：

```python
range_here = (p_cumsum >= 0.16) & (p_cumsum <= 0.84)
med_ind = abs(p_cumsum - 0.5).argmin()

gamma_med = gamma_grid[med_ind]
gamma_uperr = gamma_grid[range_here][-1] - gamma_grid[med_ind]
gamma_dwerr = gamma_grid[med_ind] - gamma_grid[range_here][0]
```

因此，Fig. 8 里的 `gamma` 观测点和误差条，实际上是：

- 点：单 lens posterior 的中位数
- 误差条：同一 posterior 的 16%-84% 区间

### Step 6. 提取 `m5` 的观测值和误差

`m5` 的处理方式和 `gamma` 共享同一个 posterior，但不是重新做一次新的推断。

代码逻辑是：

```python
m5_med = m5_grid[med_ind]
m5_dw = m5_grid[range_here][0]
m5_up = m5_grid[range_here][-1]
```

然后根据 `m5_grid` 是否随 `gamma` 单调增加，决定上下误差的方向：

```python
if m5_up > m5_dw:
    m5_uperr = m5_up - m5_med
    m5_dwerr = m5_med - m5_dw
else:
    m5_uperr = m5_dw - m5_med
    m5_dwerr = m5_med - m5_up
```

这一步的物理和统计意义是：

- `m5` 不是独立于 `gamma` 再做一次 posterior
- 而是把同一个一维 posterior 从 `gamma` 轴投影到了 `m5_grid(gamma)` 这条曲线上
- 因为 `m5_grid(gamma)` 可能不单调，所以上下误差不能简单按索引顺序理解，必须按数值大小判断

## 为什么这不是两个独立后验

这是最容易被误解的地方。

在这套实现里，并不存在：

- 一个单独的 `p(gamma | data)`
- 再加一个彼此独立的 `p(m5 | data)`

实际存在的是：

- Einstein radius 先把二维 `(m5, gamma)` 约束到一条曲线 `m5(gamma)`
- 然后只沿着这条曲线，用 `sigma_obs` 构造一个一维 posterior

因此：

- `gamma` 的误差条来自这条一维 posterior
- `m5` 的误差条也来自同一条一维 posterior
- 两者天然强相关

换句话说，Fig. 8 里的 `m5` 和 `gamma` 误差条，是同一 inference 问题在两种变量表示下的摘要，而不是两次互相独立的测量。

## 为什么 `m5` 误差要特殊处理单调性

`read_slacs.py` 里专门写了一个分支来判断：

- 如果 `m5_up > m5_dw`，按通常方式计算上下误差
- 否则交换上下方向

这样做不是代码风格问题，而是统计定义上的必要处理。

原因是：

- credible interval 是先在 `gamma` 轴上按 posterior 取到的
- 然后才映射到 `m5_grid(gamma)`
- 如果 `m5_grid(gamma)` 在那一段不是单调递增，那么“较大 `gamma` 端”不一定对应“较大 `m5` 端”

所以，`m5` 的上下误差必须按数值大小来定义，而不能按网格索引的左右顺序来定义。

## 为什么这和 population model 的后验不是一回事

这套 `m5` / `gamma` 观测值与误差条的生成方式，是 **single-lens flat-prior inference**，它和论文里或本地项目里做的 population-level hierarchical inference 不是同一种统计对象。

差别在于：

- single-lens flat-prior inference
  - 每个 lens 单独做
  - 只使用该 lens 的观测 Einstein radius 和 `sigma_obs`
  - 对 `m5` 和 `gamma` 使用 flat prior
- population-level hierarchical inference
  - 同时拟合整个样本
  - 使用 population hyper-parameters
  - 把 selection effects 和总体分布一起建模

因此，Fig. 8 上这些散点和误差条，应该理解为：

- “单 lens 的 flat-prior measurement summary”

而不是：

- “层级模型下的 per-lens posterior summary”

## 给另一个 agent 的最短复现指南

如果另一个 agent 要复现 Fig. 8 中的 `m5` / `gamma` 观测点和误差条，最少需要按下面的顺序做。

### 优先参考的脚本

先看这两个文件：

- `strong_lensing_tools/papers/slacs_selection/scripts/get_slacs_flatprior_grids.py`
- `strong_lensing_tools/papers/slacs_selection/scripts/read_slacs.py`

### 需要的数据文件

- `slacs_lensing_grids.hdf5`
- `slacs_jeans_grids.hdf5`
- `SLACS_table.cat`
- 输出中间文件：`slacs_flatprior_grids.hdf5`

### 每个 lens 的最小复现步骤

1. 从 lensing grid 读取 `m5_grid(gamma)` 和 `dm5drein_grid(gamma)`。
2. 从 Jeans grid 读取 `s2_grid(gamma)`。
3. 用观测 `sigma_obs` 和 `sigma_err` 构造
   `logp_grid = -0.5 * (sigma_model - sigma_obs)^2 / sigma_err^2 - log(sigma_err) - log(dm5drein_grid)`。
4. 把 `logp_grid` 归一化为离散 posterior。
5. 取 `gamma` 的中位数和 16%-84% 区间。
6. 用同一个 posterior 在 `m5_grid(gamma)` 上读出 `m5` 的中位数和 16%-84% 区间。
7. 对 `m5` 的上下误差，按数值大小而不是索引顺序定义。

### 最终应输出的量

对每个 lens，最终需要的摘要量是：

- `gamma_med`
- `gamma_dwerr`
- `gamma_uperr`
- `m5_med`
- `m5_dwerr`
- `m5_uperr`

这些量就是 Fig. 8 中对应散点与误差条的直接来源。

## TL;DR for another agent

- Fig. 8 里的 `m5` 和 `gamma` 误差条，来自每个 lens 独立做的 flat-prior inference，不是 hierarchical model 的 per-lens posterior。
- Einstein radius 在这里被当作精确约束，先把 `(m5, gamma)` 压到一条 `m5(gamma)` 曲线上。
- 然后只沿这条曲线，用 `sigma_obs` 和 `sigma_err` 建立一维 posterior。
- `gamma` 的点和误差条，是这个 posterior 的中位数和 16%-84% 区间。
- `m5` 的点和误差条，不是单独再拟合，而是把同一个 posterior 投影到 `m5_grid(gamma)` 上得到。
- `logp_grid` 里的 `-log(dm5drein_grid)` 是 Jacobian 项，不是额外先验。
- `m5_grid(gamma)` 可能不单调，所以 `m5` 的上下误差必须按数值大小判断，不能直接按网格顺序取。
