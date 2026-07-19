# Mixed Geographically Weighted Regression（`MixedGWR`）

> **pyGWRx 模型编号 16｜类别：全局—局部半参数回归**
> 本文同时说明原始方法与当前 pyGWRx 实现。凡实现与论文求解器不同之处均会明确指出。

正式来源：[Mei et al. (2004), *A Note on the Mixed Geographically Weighted Regression Model*](https://doi.org/10.1111/j.1085-9489.2004.00331.x)。


## 1. 模型要解决的核心问题

空间统计中最危险的假设之一，是默认一组回归关系在整个研究区完全相同。若某个变量在城市中心、郊区、沿海和山区产生不同影响，一个全局系数只能给出平均效应，局部差异会被平均掉。地理加权方法的共同思想是：在每个目标位置建立一个局部窗口，让距离更近、时间更近、属性更相似或属于同一机制的观测获得更高权重，再估计该位置的局部统计量或局部模型。

需要强调：局部模型不是自动的因果模型。它首先是一种用于描述、探索和预测空间异质性的工具。带宽、核函数、局部共线性、异常值、残差空间相关和多重检验均会改变解释，必须与诊断一起使用。


## 2. 一句话思想

不是所有系数都必须变化。Mixed GWR 让理论上稳定的变量保持一个全局系数，只让需要局部变化的变量形成系数面，从而减少过拟合并改善解释。

## 3. 数学模型

把设计矩阵分成全局部分 $X_g$ 与局部部分 $X_l$：

$$
y=X_g\gamma+\sum_k x_{lk}\beta_k(s)+\varepsilon.
$$

给定局部平滑矩阵 $S_l$，部分回归可以写为

$$
\hat\gamma=
\left[X_g^\top(I-S_l)^\top(I-S_l)X_g\right]^{-1}
X_g^\top(I-S_l)^\top(I-S_l)y,
$$

随后对 $y-X_g\hat\gamma$ 拟合局部 GWR，得到 $\beta(s)$。

## 4. 算法流程

1. 根据理论或检验划分 global/local 变量。
2. 对局部部分构造 GWR 平滑。
3. 通过部分回归估计全局系数。
4. 在扣除全局效应后的响应上估计局部系数。
5. 构造联合预测和条件推断；可用 AICc 或空间变异检验辅助变量分组。

## 5. pyGWRx 当前实现

```python
from pygwrx import MixedGWR

model = MixedGWR(kernel='bisquare', bandwidth='aicc', adaptive=True, local_vars=None, global_vars=None, intercept_fixed=True, ridge=0.0, ...)
```

pyGWRx 采用与 GWmodel 思路一致的部分回归/投影实现，而非含糊的交替近似；支持变量索引或名称、自动后向选择、固定截距、ridge 和空间变异检验。

### 5.1 输入语义

- `X`：自变量矩阵；启用 `fit_intercept=True` 时由模型统一添加截距。
- `y`：响应或类别，具体形状和分布要求由模型决定。
- `coords`：校准位置坐标；经纬度数据在使用欧氏距离前应投影，或选择模型支持的相应距离语义。
- 带宽：固定模式表示距离阈值，自适应模式通常表示近邻数量；两者不能混为同一单位。

### 5.2 结果与诊断

应优先检查：拟合值与残差、局部系数、带宽或尺度、有效参数个数、AICc/CV、标准误与显著性、局部共线性、影响度、残差空间结构。模型专用结果请结合下方图件和 `pygwrx.diagnostics` 使用。

## 6. 适用场景

有明确理论表明某些变量全域稳定、另一些变量空间变化，或 MGWR 显示部分变量带宽接近全域时使用。

## 7. 关键局限与误用风险

global/local 划分错误会产生偏差；自动选择仍涉及多重模型比较；全局系数不代表因果稳定；局部残差相关仍需检查。

## 8. 推荐可视化

![07 mixed coefficients](../../assets/figures/specialized/07_mixed_coefficients.png)

## 9. 最小工作流

```python
# 以下是接口结构示意；不同模型的 fit 参数可能包括 times、attributes 或阶段列表。
model = MixedGWR(...)
model.fit(X, y, coords)

# 常见结果
# model.fitted_values_
# model.residuals_
# model.local_parameters_ / model.coef_
# model.diagnostics_
```

推荐工作顺序：全局模型 → 带宽/尺度选择 → 局部拟合 → 推断校正 → 局部共线性和影响诊断 → 空间分块验证 → 图件与结论。

## 10. 主要参考资料

- [Mei et al. (2004), *A Note on the Mixed Geographically Weighted Regression Model*](https://doi.org/10.1111/j.1085-9489.2004.00331.x)
- [Comber et al. (2022), *A route map for the informed application of GWR*](https://doi.org/10.1111/gean.12316)

---

**版本说明：** 本文依据当前 pyGWRx 0.1.2 Alpha 源码与算法知识库整理。它描述的是当前已验证实现，而不是对任意同名软件的泛化说明。


## 11. 当前能力边界

- **输入：** X, y, coordinates, and global/local variable assignments
- **主要操作：** fit, score, predict
- **新位置能力：** Validated using global coefficients and re-estimated local components.
- **安装分组：** `base`
- **英文模型指南：** [打开](../../models/mixed-gwr.md)
- **API：** [打开](../../api/models/mixed-gwr.md)

## 12. 完整可运行示例

```python
# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Fit a semiparametric Mixed GWR with global and local predictors."""

from pygwrx import MixedGWR
from _common import mixed_regression, print_model_result

X, y, coords = mixed_regression()
model = MixedGWR(
    bandwidth=28,
    adaptive=True,
    global_vars=["global_x"],
    local_vars=["local_x"],
    intercept_fixed=True,
).fit(X, y, coords, compute_enp=False)
print_model_result(model)
print("global_coefficients=", model.coef_global_)
print("predictions=", model.predict(X.iloc[:3], coords.iloc[:3]))
```

该脚本是项目 API—示例覆盖检查所使用的正式示例，可通过 `python examples/run_all.py` 批量运行。
