# Standard Geographically Weighted Regression（`GWR`）

> **pyGWRx 模型编号 01｜类别：基础局部回归**
> 本文同时说明原始方法与当前 pyGWRx 实现。凡实现与论文求解器不同之处均会明确指出。

经典来源：[Brunsdon, Fotheringham & Charlton (1996), *Geographically Weighted Regression: A Method for Exploring Spatial Nonstationarity*](https://doi.org/10.1111/j.1538-4632.1996.tb00936.x)；系统专著：[Fotheringham, Brunsdon & Charlton (2002), *Geographically Weighted Regression: The Analysis of Spatially Varying Relationships*](https://www.wiley.com/en-us/Geographically+Weighted+Regression%3A+The+Analysis+of+Spatially+Varying+Relationships-p-9780471496168)。


## 1. 模型要解决的核心问题

空间统计中最危险的假设之一，是默认一组回归关系在整个研究区完全相同。若某个变量在城市中心、郊区、沿海和山区产生不同影响，一个全局系数只能给出平均效应，局部差异会被平均掉。地理加权方法的共同思想是：在每个目标位置建立一个局部窗口，让距离更近、时间更近、属性更相似或属于同一机制的观测获得更高权重，再估计该位置的局部统计量或局部模型。

需要强调：局部模型不是自动的因果模型。它首先是一种用于描述、探索和预测空间异质性的工具。带宽、核函数、局部共线性、异常值、残差空间相关和多重检验均会改变解释，必须与诊断一起使用。


## 2. 一句话思想

把一套全局回归拆成在每个空间位置标定的一组局部加权最小二乘回归。它回答的不是“全区域平均关系是多少”，而是“关系在每个位置附近是什么样”。

## 3. 数学模型

设观测位置为 $s_i=(u_i,v_i)$，设计矩阵为 $X$，响应为 $y$。位置 $s_i$ 的局部系数为

$$
\hat{\boldsymbol\beta}(s_i)
=\left(X^\top W_iX\right)^{-1}X^\top W_i y,
$$

其中 $W_i=\operatorname{diag}(w_{i1},\ldots,w_{in})$。常见核函数为

$$
\text{Gaussian: }w_{ij}=\exp\!\left[-\frac12(d_{ij}/h_i)^2\right],
$$

$$
\text{bisquare: }w_{ij}=\left[1-(d_{ij}/h_i)^2\right]^2\mathbf 1(d_{ij}<h_i),
$$

$$
\text{exponential: }w_{ij}=\exp(-d_{ij}/h_i).
$$

固定带宽中 $h_i=h$ 是距离；自适应带宽中 $h_i$ 是位置 $i$ 到第 $k$ 个近邻的距离。

## 4. 算法流程

1. 验证坐标、响应和自变量并决定是否添加截距。
2. 计算空间距离矩阵。
3. 通过 CV 或 AICc 选择固定距离带宽或自适应近邻数。
4. 在每个位置形成局部权重并求解 WLS。
5. 构造帽子矩阵，计算 ENP、AIC/AICc/BIC、局部标准误、t 值、Local R²、影响度和 Cook’s D。
6. 预测时在新位置重新形成权重并标定局部系数。

## 5. pyGWRx 当前实现

```python
from pygwrx import GWR

model = GWR(kernel='gaussian', bandwidth='cv', bandwidth_method='cv', adaptive=False, bandwidth_range=None, optimization_method='golden_section', fit_intercept=True, distance_metric='euclidean', sigma2_v1=True, verbose=False)
```

pyGWRx 的 `GWR` 是其他回归模型的统一基线。它支持 Gaussian、bisquare、exponential 或可调用核；固定/自适应带宽；CV/AICc；欧氏距离等；拟合后保存系数、预测、残差、帽子矩阵、推断统计量和局部诊断。`predict_result()` 返回新位置的系数与预测，而不是简单插值训练系数。

### 5.1 输入语义

- `X`：自变量矩阵；启用 `fit_intercept=True` 时由模型统一添加截距。
- `y`：响应或类别，具体形状和分布要求由模型决定。
- `coords`：校准位置坐标；经纬度数据在使用欧氏距离前应投影，或选择模型支持的相应距离语义。
- 带宽：固定模式表示距离阈值，自适应模式通常表示近邻数量；两者不能混为同一单位。

### 5.2 结果与诊断

应优先检查：拟合值与残差、局部系数、带宽或尺度、有效参数个数、AICc/CV、标准误与显著性、局部共线性、影响度、残差空间结构。模型专用结果请结合下方图件和 `pygwrx.diagnostics` 使用。

## 6. 适用场景

适合连续响应、样本位置明确、关系可能平滑地随空间变化、且研究目标需要局部解释或局部预测的场景。建议先拟合 OLS，再检查非平稳性、残差和带宽。

## 7. 关键局限与误用风险

GWR 假定所有系数共享同一空间尺度；局部样本过少会造成不稳定；局部共线性可能放大系数；大量逐位置检验存在多重比较问题；残差空间相关意味着遗漏结构；它不应仅凭较高的样本内 $R^2$ 被判为优越。

## 8. 推荐可视化

![01 coefficient](../../assets/figures/core/01_coefficient.png)

![02 coefficient significant](../../assets/figures/core/02_coefficient_significant.png)

![04 local r2](../../assets/figures/core/04_local_r2.png)

![05 standardized residual](../../assets/figures/core/05_standardized_residual.png)

![10 kernel weights](../../assets/figures/core/10_kernel_weights.png)

![12 diagnostic panel](../../assets/figures/core/12_diagnostic_panel.png)

## 9. 最小工作流

```python
# 以下是接口结构示意；不同模型的 fit 参数可能包括 times、attributes 或阶段列表。
model = GWR(...)
model.fit(X, y, coords)

# 常见结果
# model.fitted_values_
# model.residuals_
# model.local_parameters_ / model.coef_
# model.diagnostics_
```

推荐工作顺序：全局模型 → 带宽/尺度选择 → 局部拟合 → 推断校正 → 局部共线性和影响诊断 → 空间分块验证 → 图件与结论。

## 10. 主要参考资料

- [Brunsdon, Fotheringham & Charlton (1996), *Geographically Weighted Regression: A Method for Exploring Spatial Nonstationarity*](https://doi.org/10.1111/j.1538-4632.1996.tb00936.x)
- [Fotheringham, Brunsdon & Charlton (2002), *Geographically Weighted Regression: The Analysis of Spatially Varying Relationships*](https://www.wiley.com/en-us/Geographically+Weighted+Regression%3A+The+Analysis+of+Spatially+Varying+Relationships-p-9780471496168)
- [Comber et al. (2022), *A route map for the informed application of GWR*](https://doi.org/10.1111/gean.12316)

---

**版本说明：** 本文依据当前 pyGWRx 0.1.2 Alpha 源码与算法知识库整理。它描述的是当前已验证实现，而不是对任意同名软件的泛化说明。


## 11. 当前能力边界

- **输入：** X, y, coordinates
- **主要操作：** fit, score, predict, predict_result
- **新位置能力：** Validated local re-calibration at new coordinates.
- **安装分组：** `base`
- **英文模型指南：** [打开](../../models/gwr.md)
- **API：** [打开](../../api/models/gwr.md)

## 12. 完整可运行示例

```python
# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Fit, inspect, predict, and export a standard GWR model."""

from pygwrx import GWR, GWRPredictionResult
from _common import print_model_result, spatial_regression

X, y, coords = spatial_regression()
model = GWR(kernel="bisquare", bandwidth=24, adaptive=True).fit(X, y, coords)
print_model_result(model)
print("score=", model.score(X, y, coords))
result = model.predict_result(X.iloc[:3], coords.iloc[:3])
assert isinstance(result, GWRPredictionResult)
print(result.to_frame())
```

该脚本是项目 API—示例覆盖检查所使用的正式示例，可通过 `python examples/run_all.py` 批量运行。
