# spherical_jeans 中 h-dependent units 的处理说明

## 1. 核心结论

当前 CMASS 迁移实现没有把 `h^-1 Msun` 和 `h^-1 kpc` 直接传入 `spherical_jeans`。

当前实现采用的是：

```text
spherical_jeans 内部:
  length unit = physical kpc
  mass unit   = Msun-normalized unit mass

spherical_jeans 输出后:
  将 legacy fixed-kpc 的 S_unit 解析转换到 h_units_v1
```

这条路径和“原生 h-unit Jeans 输入”不是同一种实现方式。二者都可以成立，但不能混用。

当前实现的关键约束是：

```text
不要声称 spherical_jeans 内部已经在使用 h^-1 Msun 或 h^-1 kpc。
它内部仍然是 physical kpc + Msun-normalized unit response。
```

最终写入 h-units sigma table 的量才解释为：

```text
S_unit_h = sigma^2 / 10**mRh
```

其中：

```text
mRh = log10[M_2D(<R h^-1 kpc) / (h^-1 Msun)]
```

## 2. spherical_jeans 的输入和输出

本项目调用的核心函数是：

```python
sigma_model.sigma2(
    (radial_grid, enclosed_mass_grid),
    aperture_kpc,
    tracer_parameters,
    tracer_profile,
    seeing=seeing_kpc,
)
```

它的输入含义如下。

| 输入 | 当前传入内容 | 单位解释 |
|---|---|---|
| `radial_grid` | Jeans 积分半径网格 | physical kpc |
| `enclosed_mass_grid` | 3D enclosed mass profile `M_3D(<r)` | 无量纲 unit mass，后续按 Msun 解释 |
| `aperture_kpc` | slit/BOSS/within-Re aperture | physical kpc |
| `seeing_kpc` | seeing FWHM | physical kpc |
| `tracer_parameters` | deV 的 `Re` 或 Sersic 的 `(Re, n)` | `Re` 为 physical kpc |
| `tracer_profile` | `deVaucouleurs` 或 `sersic` | 无单位函数，依赖输入半径单位自洽 |

`spherical_jeans` 返回的是：

```text
sigma^2 / G
```

维度相当于：

```text
mass_unit / length_unit
```

当前代码随后乘上：

```python
(G * M_sun / kpc).to("km2 / s2").value
```

因此当前解释为：

```text
sigma^2 in km^2 s^-2 per Msun-normalized unit mass
```

相关实现位置：

```text
prepare_intepolation_grids/interpolation_grids/physics/jeans.py
```

关键代码逻辑：

```python
normalization = 1.0 / powerlaw.M2d(5.0, gamma)
enclosed_mass_grid = normalization * powerlaw.M3d(radial_grid, gamma)
sigma2_over_g = sigma_model.sigma2(...)
S_unit = sigma2_over_g * (G * M_sun / kpc)
```

## 3. 我们传给 spherical_jeans 的质量到底是什么

当前传入的不是某个真实 lens 的 `10**m5`，也不是 `10**m5_hinvkpc`。

当前传入的是一个单位归一化的 power-law enclosed mass profile：

```text
M_3D_unit(<r) = M_3D_powerlaw(<r) / M_2D_powerlaw(<5 kpc)
```

因此它满足：

```text
M_2D_unit(<5 physical kpc) = 1
```

这里的 `1` 是 unit mass。进入物理单位换算时，我们把这个 unit mass 解释为 `1 Msun`，所以得到：

```text
S_fixed_5 = sigma^2 / 10**m5
```

其中：

```text
m5 = log10[M_2D(<5 physical kpc) / Msun]
```

对 `m10` 或其他 fixed-kpc aperture，不重新调用另一套质量单位，而是利用 power-law 关系：

```text
M_2D(<R) proportional to R^(3 - gamma)
```

得到：

```text
S_fixed_R = S_fixed_5 * (5 / R)^(3 - gamma)
```

## 4. h_units_v1 下为什么不能说 solver 输入质量是 h^-1 Msun

`h_units_v1` 的 mass definition 是：

```text
mRh = log10[M_2D(<R h^-1 kpc) / (h^-1 Msun)]
```

这意味着：

```text
aperture physical radius = R / h_ref kpc
mass denominator         = 1 / h_ref Msun
```

但是当前 `spherical_jeans` 内部没有直接采用这两个单位。它仍然使用：

```text
aperture radius for base normalization = R physical kpc
mass denominator                       = Msun
```

然后在输出层做解析转换。

所以更准确的说法是：

```text
最终 HDF5 中的 S_unit_h 是 h^-1 Msun convention 下的响应；
但 spherical_jeans 本身没有直接收到 h^-1 Msun 质量单位。
```

## 5. h_units_v1 的解析转换公式

从 legacy fixed-kpc 响应出发：

```text
S_fixed_R = sigma^2 per 1 Msun inside R physical kpc
```

h-units 中，`mRh = 0` 代表：

```text
M_2D(<R h^-1 kpc) = 1 h^-1 Msun = 1 / h_ref Msun
```

而：

```text
R h^-1 kpc = R / h_ref physical kpc
```

对 power-law projected mass：

```text
M_2D(<r) proportional to r^(3 - gamma)
```

如果一个 profile 在 `R / h_ref` physical kpc 内有 `1 / h_ref Msun`，那么它在 `R` physical kpc 内的质量是：

```text
M_2D(<R physical kpc)
  = (1 / h_ref Msun) * (R / (R / h_ref))^(3 - gamma)
  = h_ref^(2 - gamma) Msun
```

因此：

```text
S_h_R = S_fixed_R * h_ref^(2 - gamma)
```

代码中的对应 helper 是：

```text
Sunit_hinv_from_fixed_kpc(Sunit_fixed, gamma, h_ref)
```

这就是当前 h-units sigma table 的核心转换。

## 6. 长度单位为什么仍然可以是 physical kpc

如果我们直接在 `spherical_jeans` 内部采用 h-unit，那么确实必须统一：

```text
mass unit   = h^-1 Msun
length unit = h^-1 kpc
```

这样：

```text
mass_unit / length_unit
  = (h^-1 Msun) / (h^-1 kpc)
  = Msun / kpc
```

最终乘 `G * M_sun / kpc` 仍然成立。

但当前实现不是这条路径。当前实现内部统一是：

```text
mass unit   = Msun
length unit = physical kpc
```

这同样是自洽的。h 因子没有在 solver 内部混入，而是在 solver 输出后整体转换。

真正错误的是混用，例如：

```text
错误例 1:
  mass unit = h^-1 Msun
  length unit = physical kpc

错误例 2:
  mass unit = Msun
  length unit = h^-1 kpc
```

这两种都会让 `sigma^2` 多出或少掉一个 h 因子。

当前代码避免了这种混用。

## 7. 当前实现的数据流

当前 h-units sigma table 构建的数据流是：

```text
1. 选择 h_units_v1

2. sigma table 的 Re 轴以 h-unit 形式存储:
     log10[Re / (h^-1 kpc)]

3. 调用 spherical_jeans 前转回 physical kpc:
     log10[Re / kpc] = log10[Re / (h^-1 kpc)] - log10(h_ref)

4. spherical_jeans 内部使用:
     physical kpc + Msun-normalized unit mass

5. 得到 legacy fixed-kpc S_unit

6. 按 power-law 公式转成 h_units_v1:
     S_h = S_fixed * h_ref^(2 - gamma)

7. HDF5 写入:
     unit_convention = h_units_v1
     h_ref = 0.7
     mass_unit = h^-1 Msun
     mass_aperture_unit = h^-1 kpc
     mass_definition_label = m5_hinvkpc 或 m10_hinvkpc
```

## 8. 与 raw HDF5 per-lens s2_grid 的关系

raw HDF5 的 per-lens `s2_grid` 路径也使用同一个 Jeans wrapper。

对真实 lens：

```text
Re_arcsec, seeing_arcsec, aperture_arcsec
```

先通过 cosmology 转成：

```text
physical kpc
```

再传给 `spherical_jeans`。

当输出文件选择 `h_units_v1` 时：

```text
s2_grid 写入 mass_definitions/m5_hinvkpc 或 mass_definitions/m10_hinvkpc
```

并且 `s2_grid` 已经是 h-units mass definition 下的 `S_unit`。

## 9. 什么时候当前转换不再安全

当前解析转换依赖一个核心假设：

```text
lensing total mass profile 是 power-law，因此 M_2D(<R) proportional to R^(3 - gamma)
```

如果未来换成 NFW、composite baryon+dark matter、broken power-law、Sersic mass-follow-light 等非纯 power-law profile，那么：

```text
S_h = S_fixed * h_ref^(2 - gamma)
```

不再一般成立。

那时应该改成原生 h-unit Jeans normalization，或者重新推导对应 mass profile 的 aperture conversion。

原生 h-unit Jeans 路径应该明确设定：

```text
M_2D(<R h^-1 kpc) = 1 h^-1 Msun
length unit       = h^-1 kpc
```

并保证传入 solver 的所有长度都在同一个 length unit 中。

## 10. 必要 sanity checks

为了确认没有错误的 h 因子，至少需要保留以下测试或检查：

```text
1. S_unit_h = S_unit_fixed * h_ref^(2 - gamma)

2. mRh = mFixed - (2 - gamma) * log10(h_ref)

3. 对同一物理 profile:
     sigma^2_fixed = S_fixed * 10**mFixed
     sigma^2_h     = S_h     * 10**mRh
   二者应一致。

4. m10h - m5h = (3 - gamma) * log10(2)

5. h-units sigma table 的 HDF5 attrs 必须包含:
     unit_convention = h_units_v1
     h_ref = 0.7
     mass_unit = h^-1 Msun
     mass_aperture_unit = h^-1 kpc
```

这些检查的目标不是验证 `spherical_jeans` 是否理解 h-units，而是验证：

```text
physical-kpc/Msun solver 输出
```

已经被正确迁移到：

```text
h^-1 kpc / h^-1 Msun convention
```

## 11. 推荐表述

在代码注释、论文方法和 run metadata 中，推荐使用下面这类表述：

```text
The Jeans solver is evaluated in physical kpc with a unit projected mass
normalization. For h_units_v1 products, the resulting fixed-kpc response is
analytically transformed to the h-dependent aperture and mass convention using
the power-law scaling S_h = S_fixed h_ref^(2-gamma).
```

中文表述：

```text
Jeans 数值积分本身在 physical kpc 和 Msun unit response 下完成。
h_units_v1 不是直接传入 solver 的单位系统，而是对 solver 输出的
fixed-kpc S_unit 做 power-law 解析迁移后得到的 HDF5 convention。
```

## 12. 代码逐段对照

本节把上面的物理解释直接对应到当前代码。后续如果有人修改 Jeans 或 sigma-table builder，应优先检查这些位置。

### 12.1 物理单位换算入口

文件：

```text
prepare_intepolation_grids/interpolation_grids/physics/jeans.py
```

代码：

```python
COSMOLOGY = FlatLambdaCDM(H0=70, Om0=0.3)
SIGMA2_TO_KM2_PER_S2 = (G * M_sun / kpc).to("km2 / s2").value
```

含义：

```text
spherical_jeans 返回 sigma^2 / G。
本项目在这里明确把 unit response 解释为 Msun/kpc，再转成 km^2/s^2。
```

这行代码是判断 solver 内部单位的关键证据。如果 solver 真正改成原生 h-unit 输入，仍然可以乘这个因子，但前提必须是：

```text
mass unit / length unit = (h^-1 Msun) / (h^-1 kpc) = Msun / kpc
```

当前实现不是原生 h-unit 输入，而是 physical-kpc/Msun 输入。

### 12.2 aperture 和 seeing 始终转成 physical kpc

文件：

```text
prepare_intepolation_grids/interpolation_grids/physics/jeans.py
```

函数：

```python
def _build_aperture_and_seeing_kpc(...):
    physical_kpc_per_arcsec = kpc_per_arcsec(zd)
    ...
    aperture_kpc = radius_or_slit_size_arcsec * physical_kpc_per_arcsec
    seeing_kpc = seeing_fwhm_arcsec * physical_kpc_per_arcsec
```

含义：

```text
slit aperture、BOSS aperture、seeing 全部从 arcsec 转成 physical kpc。
这里没有把 aperture 转成 h^-1 kpc。
```

`within_re` 分支也一样：

```python
if normalized_sigma_definition == WITHIN_RE_SIGMA_DEFINITION:
    return float(re_kpc), None
```

这里传入的 `re_kpc` 已经是 physical kpc。

### 12.3 tracer Re 和 radial grid 也是 physical kpc

文件：

```text
prepare_intepolation_grids/interpolation_grids/physics/jeans.py
```

函数：

```python
def _resolve_tracer_setup(profile_name, re_kpc, n_value=None):
    if normalized_profile == "devauc":
        tracer_parameters = re_kpc
        radial_anchor_kpc = re_kpc
    elif normalized_profile == "sersic":
        tracer_parameters = (re_kpc, n_value)
        radial_anchor_kpc = re_kpc

    radial_grid = np.logspace(
        np.log10(radial_anchor_kpc) - 3.0,
        np.log10(radial_anchor_kpc) + 3.0,
        DEFAULT_RADIAL_GRID_SIZE,
    )
```

含义：

```text
deV 的 Re、Sersic 的 Re、radial_grid 全部继承 re_kpc。
而 re_kpc 在调用前已经被明确处理成 physical kpc。
```

### 12.4 传给 spherical_jeans 的质量 profile

文件：

```text
prepare_intepolation_grids/interpolation_grids/physics/jeans.py
```

函数：

```python
def _compute_sigma_unit_values_for_prepared_inputs(...):
    for index, gamma in enumerate(np.asarray(gamma_grid, dtype=float)):
        normalization = 1.0 / powerlaw.M2d(5.0, gamma)
        enclosed_mass_grid = normalization * powerlaw.M3d(radial_grid, gamma)
        sigma2_over_g = sigma_model.sigma2(
            (radial_grid, enclosed_mass_grid),
            aperture_kpc,
            tracer_parameters,
            tracer_profile,
            **sigma2_kwargs,
        )
        output[index] = sigma2_over_g * SIGMA2_TO_KM2_PER_S2
```

含义：

```text
normalization = 1 / M2d(5.0, gamma)
```

这里的 `5.0` 是 `5 physical kpc`，因为 `radial_grid`、`aperture_kpc`、`Re` 都在 physical kpc 系统内。

因此：

```text
M_2D_unit(<5 physical kpc) = 1
```

传入 `spherical_jeans` 的 mass array 是：

```text
enclosed_mass_grid = M_3D_unit(<r)
```

它不是：

```text
M_2D(<5 h^-1 kpc) = 1 h^-1 Msun
```

也不是：

```text
M_2D(<10 h^-1 kpc) = 1 h^-1 Msun
```

这是当前实现最重要的代码事实。

### 12.5 fixed-kpc m5 到 mR 的半径转换

文件：

```text
prepare_intepolation_grids/interpolation_grids/physics/jeans.py
```

函数：

```python
def _sigma_unit_mass_scale_factor(gamma_grid, mass_radius_kpc):
    if float(mass_radius_kpc) == 5.0:
        return np.ones_like(gamma_array, dtype=float)
    return np.power(5.0 / float(mass_radius_kpc), 3.0 - gamma_array)
```

调用位置：

```python
legacy_values = base_values * _sigma_unit_mass_scale_factor(
    np.asarray(gamma_grid, dtype=float),
    mass_radius_kpc,
)
```

含义：

```text
base_values 是 per m5 fixed-kpc response。
mass_radius_kpc = 10 时，用 power-law 关系转成 per m10 fixed-kpc response。
```

公式：

```text
S_fixed_R = S_fixed_5 * (5 / R)^(3 - gamma)
```

这一步仍然是 fixed physical kpc convention。

### 12.6 fixed-kpc S_unit 到 h-units S_unit 的转换

文件：

```text
prepare_intepolation_grids/interpolation_grids/physics/jeans.py
```

函数：

```python
def compute_sigma_unit_grid(..., unit_convention, h_ref):
    ...
    legacy_values = base_values * _sigma_unit_mass_scale_factor(...)
    if normalized_convention == H_UNITS_V1:
        return Sunit_hinv_from_fixed_kpc(
            legacy_values,
            np.asarray(gamma_grid, dtype=float),
            h_ref=h_ref,
        )
```

被调用的 helper 在：

```text
prepare_intepolation_grids/interpolation_grids/unit_conventions.py
```

代码：

```python
def Sunit_hinv_from_fixed_kpc(sigma_unit_fixed_kpc, gamma, *, h_ref):
    return sigma_unit_fixed_kpc * h_ref**(2 - gamma)
```

含义：

```text
这一步才把 fixed-kpc/Msun response 转成 h_units_v1 response。
```

它对应公式：

```text
S_h_R = S_fixed_R * h_ref^(2 - gamma)
```

所以 h-units 不是通过改变 `spherical_jeans` 的输入实现的，而是通过输出层转换实现的。

### 12.7 sigma table 的 Re 轴如何处理 h-units

文件：

```text
prepare_intepolation_grids/interpolation_grids/io/sigma_tables.py
```

代码：

```python
def _default_log_re_axis_for_unit_convention(base_axis, unit_convention, h_ref):
    if unit_convention == H_UNITS_V1:
        return axis + np.log10(float(h_ref))
    return axis
```

含义：

```text
h-units table 对外存储:
  log10[Re / (h^-1 kpc)]

对于同一个 physical Re:
  log10[Re / (h^-1 kpc)] = log10[Re / kpc] + log10(h_ref)
```

但是调用 Jeans 前：

```python
def _physical_log_re_axis_for_jeans(stored_log_re_axis, unit_convention, h_ref):
    if unit_convention == H_UNITS_V1:
        return axis - np.log10(float(h_ref))
    return axis
```

含义：

```text
表的坐标可以是 h-units；
solver 的输入仍然被转回 physical kpc。
```

这就是避免 h-unit 长度直接进入 solver 的代码层防线。

### 12.8 build_sigma_unit_table 的完整数据流

文件：

```text
prepare_intepolation_grids/interpolation_grids/io/sigma_tables.py
```

deV 分支核心逻辑：

```python
log_re_axis = _default_log_re_axis_for_unit_convention(...)
values = _build_devauc_values(
    gamma_axis=gamma_axis,
    zd_axis=zd_axis,
    log_re_axis=_physical_log_re_axis_for_jeans(log_re_axis, ...),
    workers=workers,
    aperture_policy=resolved_aperture_policy,
)
scaled_values = values * np.power(5.0 / mass_radius_kpc, 3.0 - gamma_axis)[:, None, None]
values = _convert_sigma_values_for_unit_convention(scaled_values, gamma_axis, ...)
```

Sersic 分支同理，只是多一个 `n_axis` 维度：

```python
values = _build_sersic_values(
    gamma_axis=gamma_axis,
    zd_axis=zd_axis,
    log_re_axis=_physical_log_re_axis_for_jeans(log_re_axis, ...),
    n_axis=n_axis,
    workers=workers,
    aperture_policy=resolved_aperture_policy,
)
scaled_values = values * np.power(5.0 / mass_radius_kpc, 3.0 - gamma_axis)[:, None, None, None]
values = _convert_sigma_values_for_unit_convention(scaled_values, gamma_axis, ...)
```

按顺序解释：

```text
1. 生成对外存储的 Re 轴。
   h_units_v1 下存 log10[Re / (h^-1 kpc)]。

2. 调用 Jeans builder 前，把 Re 轴转回 physical log10[Re/kpc]。

3. Jeans builder 返回 fixed-kpc/Msun convention 下的 base S_unit。

4. 用 (5/R)^(3-gamma) 得到 fixed-kpc mR response。

5. 如果 unit_convention == h_units_v1，用 h_ref^(2-gamma) 转成 h-units response。

6. 写入 SigmaUnitTable，metadata 标记为 h_units_v1。
```

### 12.9 raw HDF5 per-lens s2_grid 的代码路径

文件：

```text
prepare_intepolation_grids/interpolation_grids/physics/jeans.py
```

函数：

```python
def compute_s2_grid(galaxy, gamma_grid, mass_radius_kpc, unit_convention, h_ref):
    physical_kpc_per_arcsec = kpc_per_arcsec(galaxy.zd)

    if uses_devaucouleurs_branch(galaxy.source_filename):
        return compute_sigma_unit_grid(
            profile_name="devauc",
            re_kpc=galaxy.reff_dev_arcsec * physical_kpc_per_arcsec,
            unit_convention=unit_convention,
            h_ref=h_ref,
        )
    else:
        return compute_sigma_unit_grid(
            profile_name="sersic",
            re_kpc=galaxy.re_arcsec * physical_kpc_per_arcsec,
            n_value=galaxy.nser,
            unit_convention=unit_convention,
            h_ref=h_ref,
        )
```

含义：

```text
真实 lens 的 Re 也是先从 arcsec 转成 physical kpc。
即使输出 raw HDF5 是 h_units_v1，solver 输入仍是 physical kpc。
unit_convention 只影响 compute_sigma_unit_grid 的输出转换。
```

### 12.10 unit_conventions.py 中的公式来源

文件：

```text
prepare_intepolation_grids/interpolation_grids/unit_conventions.py
```

与 Jeans 直接相关的是：

```python
def mR_hinv_from_fixed_kpc(log_mass_fixed_kpc, gamma, *, h_ref):
    return log_mass_fixed_kpc - (2 - gamma) * log10(h_ref)
```

和：

```python
def Sunit_hinv_from_fixed_kpc(sigma_unit_fixed_kpc, gamma, *, h_ref):
    return sigma_unit_fixed_kpc * h_ref**(2 - gamma)
```

二者是一对互逆意义的迁移：

```text
mRh 相对 mFixed 增加的方向:
  mRh = mFixed - (2 - gamma) log10(h_ref)

S_unit 相对 mass normalization 反向缩放:
  S_h = S_fixed * h_ref^(2 - gamma)
```

因此：

```text
S_h * 10**mRh = S_fixed * 10**mFixed
```

这就是 h-units 转换不改变同一物理 profile 的预测 `sigma^2` 的原因。

### 12.11 如果要改成原生 h-unit solver，代码需要动哪里

如果未来决定不再走“legacy solver 输出后解析转换”，而是让 `spherical_jeans` 原生使用 h-units，需要至少改动以下位置：

1. `jeans.py::_compute_sigma_unit_values_for_prepared_inputs`

   当前：

   ```python
   normalization = 1.0 / powerlaw.M2d(5.0, gamma)
   ```

   原生 h-unit 路径不能继续硬编码 `5.0 physical kpc`。它应按目标 aperture 使用：

   ```text
   R_hunit = 5 或 10，单位为 h^-1 kpc
   M_2D(<R_hunit h^-1 kpc) = 1 h^-1 Msun
   ```

2. `jeans.py::_build_aperture_and_seeing_kpc`

   当前把 aperture 和 seeing 转成 physical kpc。原生 h-unit 路径要么全部转成 `h^-1 kpc`，要么保留 physical kpc 但质量也必须保持 Msun，不能混用。

3. `sigma_tables.py::_physical_log_re_axis_for_jeans`

   当前 h-units table 坐标在调用 solver 前转回 physical kpc。原生 h-unit solver 不应再做这个反变换，而应保证所有 solver 长度统一在 `h^-1 kpc`。

4. `unit_conventions.py::Sunit_hinv_from_fixed_kpc`

   如果 solver 已经原生输出 h-units response，就不能再乘一次 `h_ref^(2-gamma)`，否则会重复引入 h 因子。

简言之，当前代码是：

```text
physical solver + h-unit output conversion
```

原生 h-unit solver 应该是：

```text
h-unit solver + no extra h-unit output conversion
```

两套路径只能选一套，不能叠加。
