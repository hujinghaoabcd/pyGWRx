# Multiscale Geographically Weighted Regression（`MGWR`）

> **pyGWRx 模型编号 02｜类别：多尺度局部回归**
> 本文同时说明原始方法与当前 pyGWRx 实现。凡实现与论文求解器不同之处均会明确指出。

正式来源：[Fotheringham, Yang & Kang (2017), *Multiscale Geographically Weighted Regression (MGWR)*](https://doi.org/10.1080/24694452.2017.1352480)。


## 1. 模型要解决的核心问题

空间统计中最危险的假设之一，是默认一组回归关系在整个研究区完全相同。若某个变量在城市中心、郊区、沿海和山区产生不同影响，一个全局系数只能给出平均效应，局部差异会被平均掉。地理加权方法的共同思想是：在每个目标位置建立一个局部窗口，让距离更近、时间更近、属性更相似或属于同一机制的观测获得更高权重，再估计该位置的局部统计量或局部模型。

需要强调：局部模型不是自动的因果模型。它首先是一种用于描述、探索和预测空间异质性的工具。带宽、核函数、局部共线性、异常值、残差空间相关和多重检验均会改变解释，必须与诊断一起使用。


## 2. 一句话思想

GWR 用一把尺子测量所有关系，而 MGWR 允许每个系数拥有自己的空间作用尺度。一个变量可以是几乎全局的，另一个变量只在很小邻域内变化。

## 3. 数学模型

MGWR 的加性形式为

$$
y_i=\sum_{k=0}^{p}\beta_k(s_i)x_{ik}+\varepsilon_i,
$$

但第 $k$ 个系数通过自己的权重矩阵 $W_{ik}(h_k)$ 估计。反向拟合中，对第 $k$ 项构造部分残差

$$
r_i^{(-k)}=y_i-\sum_{\ell\ne k}x_{i\ell}\hat\beta_\ell(s_i),
$$

再用带宽 $h_k$ 的单变量局部回归更新

$$
\hat\beta_k(s_i)
=\frac{\sum_jw_{ij}(h_k)x_{jk}r_j^{(-k)}}
{\sum_jw_{ij}(h_k)x_{jk}^2}.
$$

迭代直至系数面、残差平方和或带宽稳定。每个 $h_k$ 可解释为对应过程的空间尺度。

## 4. 算法流程

1. 先用单带宽 GWR 或指定带宽初始化。
2. 逐变量构造部分残差。
3. 为每个变量独立搜索 AICc/CV 最优带宽。
4. 用该变量的带宽更新局部系数面。
5. 循环反向拟合，直至收敛或带宽连续多轮不变。
6. 基于各项平滑矩阵构造精确或近似推断与 ENP。

## 5. pyGWRx 当前实现

```python
from pygwrx import MGWR

model = MGWR(kernel='bisquare', bandwidths=None, bandwidth_method='aicc', adaptive=True, bandwidth_range=None, bandwidth_ranges=None, init_bandwidth=None, optimization_method='golden_section', search_tol=1e-6, search_max_iter=200, max_iter=200, tol=1e-5, rss_score=False, bws_same_times=5, fit_intercept=True, distance_metric='euclidean', sigma2_v1=True, verbose=False)
```

pyGWRx 的 `MGWR` 采用 Gaussian 加性 MGWR，支持每个设计列独立带宽、自动或手动范围、反向拟合、精确平滑矩阵与协方差诊断和校准位置结果表；当前不支持独立新位置预测。截距若启用也具有自己的带宽。

### 5.1 输入语义

- `X`：自变量矩阵；启用 `fit_intercept=True` 时由模型统一添加截距。
- `y`：响应或类别，具体形状和分布要求由模型决定。
- `coords`：校准位置坐标；经纬度数据在使用欧氏距离前应投影，或选择模型支持的相应距离语义。
- 带宽：固定模式表示距离阈值，自适应模式通常表示近邻数量；两者不能混为同一单位。

### 5.2 结果与诊断

应优先检查：拟合值与残差、局部系数、带宽或尺度、有效参数个数、AICc/CV、标准误与显著性、局部共线性、影响度、残差空间结构。模型专用结果请结合下方图件和 `pygwrx.diagnostics` 使用。

## 6. 适用场景

当理论上不同驱动因素作用于不同空间尺度，或 GWR 的统一带宽导致某些系数过度平滑、另一些系数过度波动时使用。

## 7. 关键局限与误用风险

计算成本高于 GWR；带宽之间可能相互影响；局部系数仍可能共线；极大的带宽更接近全局效应，但不等于严格的全局固定系数；解释时必须把系数大小和带宽尺度同时考虑。

## 8. 推荐可视化

![09 mgwr bandwidths](../../assets/figures/core/09_mgwr_bandwidths.png)

![11 gwr mgwr comparison](../../assets/figures/core/11_gwr_mgwr_comparison.png)

## 9. 最小工作流

```python
# 以下是接口结构示意；不同模型的 fit 参数可能包括 times、attributes 或阶段列表。
model = MGWR(...)
model.fit(X, y, coords)

# 常见结果
# model.fitted_values_
# model.residuals_
# model.local_parameters_ / model.coef_
# model.diagnostics_
```

推荐工作顺序：全局模型 → 带宽/尺度选择 → 局部拟合 → 推断校正 → 局部共线性和影响诊断 → 空间分块验证 → 图件与结论。

## 10. 主要参考资料

- [Fotheringham, Yang & Kang (2017), *Multiscale Geographically Weighted Regression (MGWR)*](https://doi.org/10.1080/24694452.2017.1352480)
- [Comber et al. (2022), *A route map for the informed application of GWR*](https://doi.org/10.1111/gean.12316)

---

**版本说明：** 本文依据当前 pyGWRx 0.1.2 Alpha 源码与算法知识库整理。它描述的是当前已验证实现，而不是对任意同名软件的泛化说明。


## 11. 当前能力边界

- **输入：** X, y, coordinates
- **主要操作：** fit, score, calibration-location results
- **新位置能力：** Independent-target prediction is intentionally unavailable in the current validated API.
- **安装分组：** `base`
- **英文模型指南：** [打开](../../models/mgwr.md)
- **API：** [打开](../../api/models/mgwr.md)

## 12. 完整可运行示例

```python
# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Fit MGWR with fixed variable-specific bandwidths."""

from pygwrx import MGWR
from _common import print_model_result, spatial_regression

X, y, coords = spatial_regression(n=48, p=2)
model = MGWR(bandwidths=[24, 26, 28], adaptive=True, max_iter=8, tol=0.5).fit(
    X, y, coords, compute_inference=True
)
print_model_result(model)
try:
    model.predict(X.iloc[:2], coords.iloc[:2])
except NotImplementedError as exc:
    print("Expected MGWR prediction limitation:", exc)
```

该脚本是项目 API—示例覆盖检查所使用的正式示例，可通过 `python examples/run_all.py` 批量运行。
