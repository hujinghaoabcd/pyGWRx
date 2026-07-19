# Locally Compensated Ridge GWR（`LCRGWR`）

> **pyGWRx 模型编号 04｜类别：局部共线性补偿**
> 本文同时说明原始方法与当前 pyGWRx 实现。凡实现与论文求解器不同之处均会明确指出。

诊断与局部岭思想：[Wheeler (2007), *Diagnostic Tools and a Remedial Method for Collinearity in GWR*](https://doi.org/10.1068/a38325)。


## 1. 模型要解决的核心问题

空间统计中最危险的假设之一，是默认一组回归关系在整个研究区完全相同。若某个变量在城市中心、郊区、沿海和山区产生不同影响，一个全局系数只能给出平均效应，局部差异会被平均掉。地理加权方法的共同思想是：在每个目标位置建立一个局部窗口，让距离更近、时间更近、属性更相似或属于同一机制的观测获得更高权重，再估计该位置的局部统计量或局部模型。

需要强调：局部模型不是自动的因果模型。它首先是一种用于描述、探索和预测空间异质性的工具。带宽、核函数、局部共线性、异常值、残差空间相关和多重检验均会改变解释，必须与诊断一起使用。


## 2. 一句话思想

全局自变量不共线，并不保证每个局部窗口都不共线。LCR-GWR 在条件数过高的位置自动加入局部岭惩罚，而在条件良好的位置保持普通 GWR。

## 3. 数学模型

对位置 $i$ 的局部加权设计矩阵，令标准化后的交叉乘积矩阵特征值为 $d_{i,\max}$ 和 $d_{i,\min}$。希望补偿后条件数不超过阈值 $\kappa^*$：

$$
\frac{d_{i,\max}+\lambda_i}{d_{i,\min}+\lambda_i}=\kappa^*.
$$

解得

$$
\lambda_i=
\max\left\{0,
\frac{d_{i,\max}-\kappa^*d_{i,\min}}{\kappa^*-1}
\right\}.
$$

局部估计变为

$$
\hat\beta_i=(X^\top W_iX+\lambda_i P)^{-1}X^\top W_i y,
$$

其中 $P$ 通常不惩罚截距。

## 4. 算法流程

1. 拟合或构造每个位置的加权设计矩阵。
2. 计算局部相关、VIF、条件数和方差分解比例。
3. 若条件数超过阈值，求最小的局部 $\lambda_i$。
4. 用局部岭正规方程重新估计。
5. 比较补偿前后系数、方差和预测。

## 5. pyGWRx 当前实现

```python
from pygwrx import LCRGWR

model = LCRGWR(kernel='bisquare', bandwidth='cv', bandwidth_method='cv', adaptive=False, lambda_ridge=0.0, lambda_adjust=True, cn_thresh=30.0, ...)
```

pyGWRx 的 `LCRGWR` 支持固定岭参数或按局部条件数自动调整；保存 `local_cn_`、`local_lambdas_`、补偿后的系数与诊断。

### 5.1 输入语义

- `X`：自变量矩阵；启用 `fit_intercept=True` 时由模型统一添加截距。
- `y`：响应或类别，具体形状和分布要求由模型决定。
- `coords`：校准位置坐标；经纬度数据在使用欧氏距离前应投影，或选择模型支持的相应距离语义。
- 带宽：固定模式表示距离阈值，自适应模式通常表示近邻数量；两者不能混为同一单位。

### 5.2 结果与诊断

应优先检查：拟合值与残差、局部系数、带宽或尺度、有效参数个数、AICc/CV、标准误与显著性、局部共线性、影响度、残差空间结构。模型专用结果请结合下方图件和 `pygwrx.diagnostics` 使用。

## 6. 适用场景

解释性 GWR 中出现系数符号异常、局部标准误巨大、变量在某些区域高度相关时使用。

## 7. 关键局限与误用风险

岭补偿会引入偏差；条件数阈值不是自然常数；如果变量本身不可识别，惩罚只能稳定而不能创造信息；需配合局部 VIF/VDP 地图。

## 8. 推荐可视化

![07 gwr condition number](../../assets/figures/core/07_gwr_condition_number.png)

![08 lcr lambda](../../assets/figures/core/08_lcr_lambda.png)

## 9. 最小工作流

```python
# 以下是接口结构示意；不同模型的 fit 参数可能包括 times、attributes 或阶段列表。
model = LCRGWR(...)
model.fit(X, y, coords)

# 常见结果
# model.fitted_values_
# model.residuals_
# model.local_parameters_ / model.coef_
# model.diagnostics_
```

推荐工作顺序：全局模型 → 带宽/尺度选择 → 局部拟合 → 推断校正 → 局部共线性和影响诊断 → 空间分块验证 → 图件与结论。

## 10. 主要参考资料

- [Wheeler (2007), *Diagnostic Tools and a Remedial Method for Collinearity in GWR*](https://doi.org/10.1068/a38325)
- [Gollini et al. (2015), *GWmodel: an R Package for Exploring Spatial Heterogeneity*](https://doi.org/10.18637/jss.v063.i17)

---

**版本说明：** 本文依据当前 pyGWRx 0.1.2 Alpha 源码与算法知识库整理。它描述的是当前已验证实现，而不是对任意同名软件的泛化说明。


## 11. 当前能力边界

- **输入：** X, y, coordinates
- **主要操作：** fit, score, predict, predict_result
- **新位置能力：** Validated local prediction with fitted or locally adjusted ridge terms.
- **安装分组：** `base`
- **英文模型指南：** [打开](../../models/lcr-gwr.md)
- **API：** [打开](../../api/models/lcr-gwr.md)

## 12. 完整可运行示例

```python
# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Fit locally compensated ridge GWR for collinear predictors."""

from pygwrx import LCRGWR
from _common import collinear_regression, print_model_result

X, y, coords = collinear_regression()
model = LCRGWR(bandwidth=28, adaptive=True, cn_thresh=15.0, lambda_adjust=True).fit(
    X, y, coords
)
print_model_result(model)
print("local_condition_numbers=", model.local_condition_numbers_[:5])
print("local_lambdas=", model.local_lambdas_[:5])
```

该脚本是项目 API—示例覆盖检查所使用的正式示例，可通过 `python examples/run_all.py` 批量运行。
