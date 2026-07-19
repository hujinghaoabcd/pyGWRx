# Scalable Geographically Weighted Regression（`ScalableGWR`）

> **pyGWRx 模型编号 11｜类别：大样本快速 GWR**
> 本文同时说明原始方法与当前 pyGWRx 实现。凡实现与论文求解器不同之处均会明确指出。

正式来源：[Murakami et al. (2020), *Scalable GWR: A Linear-Time Algorithm for Large-Scale GWR with Polynomial Kernels*](https://doi.org/10.1080/24694452.2020.1774350)。


## 1. 模型要解决的核心问题

空间统计中最危险的假设之一，是默认一组回归关系在整个研究区完全相同。若某个变量在城市中心、郊区、沿海和山区产生不同影响，一个全局系数只能给出平均效应，局部差异会被平均掉。地理加权方法的共同思想是：在每个目标位置建立一个局部窗口，让距离更近、时间更近、属性更相似或属于同一机制的观测获得更高权重，再估计该位置的局部统计量或局部模型。

需要强调：局部模型不是自动的因果模型。它首先是一种用于描述、探索和预测空间异质性的工具。带宽、核函数、局部共线性、异常值、残差空间相关和多重检验均会改变解释，必须与诊断一起使用。


## 2. 一句话思想

经典 GWR 的距离矩阵和反复带宽搜索通常是 $O(n^2)$ 内存/时间瓶颈。ScaGWR 先把每个位置附近的矩阵交叉乘积压缩为有限个多项式核基，再对少量全局核混合参数优化。

## 3. 数学模型

令 $Q$ 个近邻距离经尺度归一化为 $r_{ij}$。多项式核基可写成

$$
\phi_q(r_{ij})=r_{ij}^{q},\qquad q=0,\ldots,P,
$$

局部交叉乘积预压缩为

$$
A_{i,q}=\sum_{j\in\mathcal N_Q(i)}\phi_q(r_{ij})x_jx_j^\top,
\qquad
b_{i,q}=\sum_{j\in\mathcal N_Q(i)}\phi_q(r_{ij})x_jy_j.
$$

优化得到核混合系数 $c_q$ 后，

$$
A_i(c)=\sum_qc_qA_{i,q},\qquad b_i(c)=\sum_qc_qb_{i,q},
$$

局部系数只需解 $A_i(c)\beta_i=b_i(c)$。固定 $Q,P$ 时，预压缩和求解随 $n$ 近似线性增长。

## 4. 算法流程

1. 为每个点查询固定数量近邻。
2. 计算多项式核基下的局部矩阵/向量压缩量。
3. 用 CV/AICc 优化尺度、惩罚和核混合参数。
4. 从压缩量快速组装每个位置的正规方程。
5. 预测新位置时仅查询近邻并复用全局核参数。

## 5. pyGWRx 当前实现

```python
from pygwrx import ScalableGWR

model = ScalableGWR(bandwidth=100, kernel='gaussian', polynomial=4, criterion='cv', optimize_bandwidth=True, scale=None, penalty=None, sample_size=None, ...)
```

pyGWRx 实现正式 ScaGWR 的多项式核、Q 近邻预压缩、全局 OLS 收缩和 CV/AICc 优化，不建立完整 $n\times n$ 距离矩阵。

### 5.1 输入语义

- `X`：自变量矩阵；启用 `fit_intercept=True` 时由模型统一添加截距。
- `y`：响应或类别，具体形状和分布要求由模型决定。
- `coords`：校准位置坐标；经纬度数据在使用欧氏距离前应投影，或选择模型支持的相应距离语义。
- `bandwidth`：ScaGWR 中固定表示近邻数量 $Q$，不是固定距离，也不提供 `adaptive` 开关。

### 5.2 结果与诊断

应优先检查：拟合值与残差、局部系数、带宽或尺度、有效参数个数、AICc/CV、标准误与显著性、局部共线性、影响度、残差空间结构。模型专用结果请结合下方图件和 `pygwrx.diagnostics` 使用。

## 6. 适用场景

样本达到数万、数十万甚至更大，标准 GWR 距离矩阵不可承受，但仍需局部线性解释时使用。

## 7. 关键局限与误用风险

它是对核权重结构的受限近似，不与任意经典 GWR 完全相同；近邻数和多项式阶数影响精度；坐标近邻查询仍受维度和数据密度影响。

## 8. 推荐可视化

![10 scalable kernel](../../assets/figures/specialized/10_scalable_kernel.png)

## 9. 最小工作流

```python
# 以下是接口结构示意；不同模型的 fit 参数可能包括 times、attributes 或阶段列表。
model = ScalableGWR(...)
model.fit(X, y, coords)

# 常见结果
# model.fitted_values_
# model.residuals_
# model.local_parameters_ / model.coef_
# model.diagnostics_
```

推荐工作顺序：全局模型 → 带宽/尺度选择 → 局部拟合 → 推断校正 → 局部共线性和影响诊断 → 空间分块验证 → 图件与结论。

## 10. 主要参考资料

- [Murakami et al. (2020), *Scalable GWR: A Linear-Time Algorithm for Large-Scale GWR with Polynomial Kernels*](https://doi.org/10.1080/24694452.2020.1774350)

---

**版本说明：** 本文依据当前 pyGWRx 0.1.2 Alpha 源码与算法知识库整理。它描述的是当前已验证实现，而不是对任意同名软件的泛化说明。


## 11. 当前能力边界

- **输入：** X, y, coordinates
- **主要操作：** fit, predict, predict_result
- **新位置能力：** Validated using the fitted scalable kernel approximation.
- **安装分组：** `base`
- **英文模型指南：** [打开](../../models/scalable-gwr.md)
- **API：** [打开](../../api/models/scalable-gwr.md)

## 12. 完整可运行示例

```python
# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Fit scalable GWR with a fixed multiscale-kernel approximation."""

from pygwrx import ScalableGWR
from _common import print_model_result, spatial_regression

X, y, coords = spatial_regression(n=54, p=2)
model = ScalableGWR(
    bandwidth=24, optimize_bandwidth=False, polynomial=4, random_state=0
).fit(X, y, coords)
print_model_result(model)
print("predictions=", model.predict(X.iloc[:3], coords.iloc[:3]))
```

该脚本是项目 API—示例覆盖检查所使用的正式示例，可通过 `python examples/run_all.py` 批量运行。
