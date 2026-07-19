# Geographically Weighted Principal Component Analysis（`GWPCA`）

> **pyGWRx 模型编号 14｜类别：局部多变量降维**
> 本文同时说明原始方法与当前 pyGWRx 实现。凡实现与论文求解器不同之处均会明确指出。

正式来源：[Harris, Brunsdon & Charlton (2011), *Geographically weighted principal components analysis*](https://doi.org/10.1080/13658816.2011.554838)。


## 1. 模型要解决的核心问题

空间统计中最危险的假设之一，是默认一组回归关系在整个研究区完全相同。若某个变量在城市中心、郊区、沿海和山区产生不同影响，一个全局系数只能给出平均效应，局部差异会被平均掉。地理加权方法的共同思想是：在每个目标位置建立一个局部窗口，让距离更近、时间更近、属性更相似或属于同一机制的观测获得更高权重，再估计该位置的局部统计量或局部模型。

需要强调：局部模型不是自动的因果模型。它首先是一种用于描述、探索和预测空间异质性的工具。带宽、核函数、局部共线性、异常值、残差空间相关和多重检验均会改变解释，必须与诊断一起使用。


## 2. 一句话思想

全局 PCA 假定同一套协方差结构适用于所有位置。GWPCA 在每个位置计算局部协方差和局部主成分，揭示多变量结构、主导变量和解释方差的空间变化。

## 3. 数学模型

位置 $i$ 的局部加权中心为 $\bar x_i$，局部加权数据为

$$
X_i^w=W_i^{1/2}(X-\mathbf1\bar x_i^\top).
$$

对其做 SVD：

$$
X_i^w=U_iD_iV_i^\top.
$$

$V_i$ 的列是局部载荷，$D_i^2$ 给出局部成分方差。第 $q$ 个累计解释率为

$$
PV_{i,q}=\frac{\sum_{k=1}^{q}D_{i,k}^2}{\sum_kD_{i,k}^2}.
$$

## 4. 算法流程

1. 决定是否对变量全局标准化。
2. 选择带宽，通常通过留一重构误差。
3. 在每个位置计算加权中心/尺度。
4. 执行局部 SVD并规范载荷符号。
5. 输出解释方差、载荷、得分和 winning variable。

## 5. pyGWRx 当前实现

```python
from pygwrx import GWPCA

model = GWPCA(n_components=2, kernel='bisquare', bandwidth='cv', adaptive=True, scaling=True, compute_scores=False, verbose=False)
```


### 5.1 输入语义

- `X`：自变量矩阵；启用 `fit_intercept=True` 时由模型统一添加截距。
- `y`：响应或类别，具体形状和分布要求由模型决定。
- `coords`：校准位置坐标；经纬度数据在使用欧氏距离前应投影，或选择模型支持的相应距离语义。
- 带宽：固定模式表示距离阈值，自适应模式通常表示近邻数量；两者不能混为同一单位。

### 5.2 结果与诊断

应优先检查：拟合值与残差、局部系数、带宽或尺度、有效参数个数、AICc/CV、标准误与显著性、局部共线性、影响度、残差空间结构。模型专用结果请结合下方图件和 `pygwrx.diagnostics` 使用。

## 6. 适用场景

变量相关结构、主导综合因子和降维方向可能随空间变化时使用。

## 7. 关键局限与误用风险

局部载荷存在符号不确定和成分交换；解释需结合解释方差；局部样本不足会不稳定；PCA 是无监督的，不保证对响应有预测价值。

## 8. 推荐可视化

![13 gwpca variance](../../assets/figures/specialized/13_gwpca_variance.png)

![14 gwpca loading](../../assets/figures/specialized/14_gwpca_loading.png)

## 9. 最小工作流

```python
# 以下是接口结构示意；不同模型的 fit 参数可能包括 times、attributes 或阶段列表。
model = GWPCA(...)
model.fit(X, y, coords)

# 常见结果
# model.fitted_values_
# model.residuals_
# model.local_parameters_ / model.coef_
# model.diagnostics_
```

推荐工作顺序：全局模型 → 带宽/尺度选择 → 局部拟合 → 推断校正 → 局部共线性和影响诊断 → 空间分块验证 → 图件与结论。

## 10. 主要参考资料

- [Harris, Brunsdon & Charlton (2011), *Geographically weighted principal components analysis*](https://doi.org/10.1080/13658816.2011.554838)

---

**版本说明：** 本文依据当前 pyGWRx 0.1.2 Alpha 源码与算法知识库整理。它描述的是当前已验证实现，而不是对任意同名软件的泛化说明。


## 11. 当前能力边界

- **输入：** Multivariate X and coordinates
- **主要操作：** fit, transform, select_bandwidth
- **新位置能力：** Not a response predictor; `transform()` returns local component scores.
- **安装分组：** `ml`
- **英文模型指南：** [打开](../../models/gwpca.md)
- **API：** [打开](../../api/models/gwpca.md)

## 12. 完整可运行示例

```python
# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Fit GWPCA, inspect local loadings, and transform observations."""

from pygwrx import GWPCA
from _common import print_model_result, spatial_regression

X, _, coords = spatial_regression(n=48, p=3)
model = GWPCA(n_components=2, bandwidth=24, adaptive=True).fit(
    X, coords, compute_cv=True
)
print_model_result(model)
print("scores_shape=", model.transform(X, coords).shape)
print("explained_variance_first_location=", model.local_pv_[0])
```

该脚本是项目 API—示例覆盖检查所使用的正式示例，可通过 `python examples/run_all.py` 批量运行。
