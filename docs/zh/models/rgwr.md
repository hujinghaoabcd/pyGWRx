# Robust Geographically Weighted Regression（`RGWR`）

> **pyGWRx 模型编号 03｜类别：异常值稳健局部回归**
> 本文同时说明原始方法与当前 pyGWRx 实现。凡实现与论文求解器不同之处均会明确指出。

方法背景：[Harris, Fotheringham & Juggins (2010), *Robust Geographically Weighted Regression*](https://doi.org/10.1080/00045600903550378)。


## 1. 模型要解决的核心问题

空间统计中最危险的假设之一，是默认一组回归关系在整个研究区完全相同。若某个变量在城市中心、郊区、沿海和山区产生不同影响，一个全局系数只能给出平均效应，局部差异会被平均掉。地理加权方法的共同思想是：在每个目标位置建立一个局部窗口，让距离更近、时间更近、属性更相似或属于同一机制的观测获得更高权重，再估计该位置的局部统计量或局部模型。

需要强调：局部模型不是自动的因果模型。它首先是一种用于描述、探索和预测空间异质性的工具。带宽、核函数、局部共线性、异常值、残差空间相关和多重检验均会改变解释，必须与诊断一起使用。


## 2. 一句话思想

普通 GWR 可能让单个异常观测污染其周围许多局部回归。RGWR 在地理权重之外再乘一个残差稳健权重，使异常值的影响逐轮下降。

## 3. 数学模型

第 $t$ 轮的总权重可写为

$$
\tilde w_{ij}^{(t)}=w_{ij}^{G}\,r_j^{(t)},
$$

其中 $w_{ij}^{G}$ 是地理核权重，$r_j^{(t)}\in[0,1]$ 是由标准化或学生化残差决定的稳健权重。经典分段函数为

$$
r_j=\begin{cases}
1,&|e_j|\le c_1,\\
\left[1-\left(\frac{|e_j|-c_1}{c_2-c_1}\right)^2\right]^2,
&c_1<|e_j|<c_2,\\
0,&|e_j|\ge c_2.
\end{cases}
$$

于是每轮仍然是局部 WLS，只是异常观测在所有目标位置中的贡献被抑制。

## 4. 算法流程

1. 拟合初始 GWR。
2. 计算标准化/学生化残差。
3. 按阈值生成观测级稳健权重。
4. 将稳健权重与每个位置的地理权重相乘并重新拟合。
5. 直到系数、权重或残差稳定；或者采用过滤模式直接剔除极端异常点。

## 5. pyGWRx 当前实现

```python
from pygwrx import RGWR

model = RGWR(kernel='gaussian', bandwidth='cv', bandwidth_method='cv', adaptive=False, method='automatic', max_iter=20, tol=1e-5, cut1=2.0, cut2=3.0, cut_filter=3.0, ...)
```

pyGWRx 提供 `automatic` 迭代稳健模式和过滤模式；保留每轮收敛历史、最终稳健权重和异常标记，并复用 GWR 的带宽、核与推断框架。

### 5.1 输入语义

- `X`：自变量矩阵；启用 `fit_intercept=True` 时由模型统一添加截距。
- `y`：响应或类别，具体形状和分布要求由模型决定。
- `coords`：校准位置坐标；经纬度数据在使用欧氏距离前应投影，或选择模型支持的相应距离语义。
- 带宽：固定模式表示距离阈值，自适应模式通常表示近邻数量；两者不能混为同一单位。

### 5.2 结果与诊断

应优先检查：拟合值与残差、局部系数、带宽或尺度、有效参数个数、AICc/CV、标准误与显著性、局部共线性、影响度、残差空间结构。模型专用结果请结合下方图件和 `pygwrx.diagnostics` 使用。

## 6. 适用场景

数据存在局部离群点、测量错误或重尾残差，但仍希望保留空间变化解释时使用。

## 7. 关键局限与误用风险

稳健权重不能修复遗漏变量、错误函数形式或空间自相关；真实的极端空间过程也可能被误当作异常值；阈值选择会影响结果；需同时报告普通 GWR 与 RGWR。

## 8. 推荐可视化

![01 rgwr weights](../../assets/figures/specialized/01_rgwr_weights.png)

![02 rgwr convergence](../../assets/figures/specialized/02_rgwr_convergence.png)

## 9. 最小工作流

```python
# 以下是接口结构示意；不同模型的 fit 参数可能包括 times、attributes 或阶段列表。
model = RGWR(...)
model.fit(X, y, coords)

# 常见结果
# model.fitted_values_
# model.residuals_
# model.local_parameters_ / model.coef_
# model.diagnostics_
```

推荐工作顺序：全局模型 → 带宽/尺度选择 → 局部拟合 → 推断校正 → 局部共线性和影响诊断 → 空间分块验证 → 图件与结论。

## 10. 主要参考资料

- [Harris, Fotheringham & Juggins (2010), *Robust Geographically Weighted Regression*](https://doi.org/10.1080/00045600903550378)

---

**版本说明：** 本文依据当前 pyGWRx 0.1.2 Alpha 源码与算法知识库整理。它描述的是当前已验证实现，而不是对任意同名软件的泛化说明。


## 11. 当前能力边界

- **输入：** X, y, coordinates
- **主要操作：** fit, score, predict, predict_result
- **新位置能力：** Validated local prediction using the fitted robust calibration state.
- **安装分组：** `base`
- **英文模型指南：** [打开](../../models/rgwr.md)
- **API：** [打开](../../api/models/rgwr.md)

## 12. 完整可运行示例

```python
# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Fit robust GWR in automatic down-weighting mode."""

import numpy as np
from pygwrx import RGWR
from _common import print_model_result, spatial_regression

X, y, coords = spatial_regression()
y = y.copy()
y[[2, 20]] += np.array([5.0, -4.0])
model = RGWR(bandwidth=24, adaptive=True, max_iter=8).fit(X, y, coords)
print_model_result(model)
print("robust_weights=", model.robust_weights_[:8])
print("predictions=", model.predict(X.iloc[:3], coords.iloc[:3]))
```

该脚本是项目 API—示例覆盖检查所使用的正式示例，可通过 `python examples/run_all.py` 批量运行。
