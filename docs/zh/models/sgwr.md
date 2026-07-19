# Similarity and Geographically Weighted Regression（`SGWR`）

> **pyGWRx 模型编号 10｜类别：地理—属性双邻近回归**
> 本文同时说明原始方法与当前 pyGWRx 实现。凡实现与论文求解器不同之处均会明确指出。

正式来源：[Lessani & Li (2024), *SGWR: similarity and geographically weighted regression*](https://doi.org/10.1080/13658816.2024.2342319)。扩展背景：[Yu et al. (2025), *Similarity and geographically weighted regression considering spatial scales of feature space*](https://doi.org/10.1016/j.spasta.2025.100897)。


## 1. 模型要解决的核心问题

空间统计中最危险的假设之一，是默认一组回归关系在整个研究区完全相同。若某个变量在城市中心、郊区、沿海和山区产生不同影响，一个全局系数只能给出平均效应，局部差异会被平均掉。地理加权方法的共同思想是：在每个目标位置建立一个局部窗口，让距离更近、时间更近、属性更相似或属于同一机制的观测获得更高权重，再估计该位置的局部统计量或局部模型。

需要强调：局部模型不是自动的因果模型。它首先是一种用于描述、探索和预测空间异质性的工具。带宽、核函数、局部共线性、异常值、残差空间相关和多重检验均会改变解释，必须与诊断一起使用。


## 2. 一句话思想

传统 GWR 只相信“地图上近”；SGWR 加入“属性上像”。两个相距很远但社会经济结构相似的地区，也可以互相提供局部信息。

## 3. 数学模型

标准化相似属性为 $z_{ik}$，属性差异定义为

$$
d_{ij}^{A}=\frac1m\sum_{k=1}^{m}|z_{ik}-z_{jk}|,
$$

相似性权重为

$$
w_{ij}^{A}=\exp[-(d_{ij}^{A})^2].
$$

地理权重 $W_i^{G}$ 与相似性权重 $W_i^{A}$ 进行凸组合：

$$
W_i^{SG}=\alpha W_i^{G}+(1-\alpha)W_i^{A},
\qquad 0\le\alpha\le1.
$$

$\alpha=1$ 时精确退化为 GWR；$\alpha=0$ 时完全由属性相似性决定邻域。

## 4. 算法流程

1. 选择用于衡量相似性的变量。
2. 对相似性变量按训练数据标准化。
3. 计算地理核与稠密属性相似性核。
4. 通过 AICc 搜索空间带宽与 $\alpha$。
5. 组合两类权重并执行局部 WLS。
6. 预测时对新点与训练样本重新计算两类权重。

## 5. pyGWRx 当前实现

```python
from pygwrx import SGWR

model = SGWR(bandwidth='aicc', adaptive=True, kernel='bisquare', alpha='aicc', similarity_vars=None, standardize_similarity=True, ...)
```

pyGWRx 实现论文式平均绝对标准化差异与凸组合；支持相似变量名称/索引、AICc 选择 $\alpha$、保存权重分量和直接局部重标定预测。

### 5.1 输入语义

- `X`：自变量矩阵；启用 `fit_intercept=True` 时由模型统一添加截距。
- `y`：响应或类别，具体形状和分布要求由模型决定。
- `coords`：校准位置坐标；经纬度数据在使用欧氏距离前应投影，或选择模型支持的相应距离语义。
- 带宽：固定模式表示距离阈值，自适应模式通常表示近邻数量；两者不能混为同一单位。

### 5.2 结果与诊断

应优先检查：拟合值与残差、局部系数、带宽或尺度、有效参数个数、AICc/CV、标准误与显著性、局部共线性、影响度、残差空间结构。模型专用结果请结合下方图件和 `pygwrx.diagnostics` 使用。

## 6. 适用场景

网络化、全球化或功能联系明显的过程，地理距离不足以定义“相关邻居”时使用。

## 7. 关键局限与误用风险

相似变量若包含响应泄漏会产生偏差；高维相似性会受距离集中影响；稠密相似权重可增加内存；属性相似不等于机制相同；必须报告 $\alpha$ 与选用变量。

## 8. 推荐可视化

![23 sgwr weights](../../assets/figures/specialized/23_sgwr_weights.png)

![24 sgwr profiles](../../assets/figures/specialized/24_sgwr_profiles.png)

## 9. 最小工作流

```python
# 以下是接口结构示意；不同模型的 fit 参数可能包括 times、attributes 或阶段列表。
model = SGWR(...)
model.fit(X, y, coords)

# 常见结果
# model.fitted_values_
# model.residuals_
# model.local_parameters_ / model.coef_
# model.diagnostics_
```

推荐工作顺序：全局模型 → 带宽/尺度选择 → 局部拟合 → 推断校正 → 局部共线性和影响诊断 → 空间分块验证 → 图件与结论。

## 10. 主要参考资料

- [Lessani & Li (2024), *SGWR: similarity and geographically weighted regression*](https://doi.org/10.1080/13658816.2024.2342319)
- [Yu et al. (2025), *Similarity and geographically weighted regression considering spatial scales of feature space*](https://doi.org/10.1016/j.spasta.2025.100897)

---

**版本说明：** 本文依据当前 pyGWRx 0.1.2 Alpha 源码与算法知识库整理。它描述的是当前已验证实现，而不是对任意同名软件的泛化说明。


## 11. 当前能力边界

- **输入：** X, y, coordinates, and similarity-variable specification
- **主要操作：** fit, predict, predict_result
- **新位置能力：** Validated by recomputing geographic and attribute-similarity weights for targets.
- **安装分组：** `base`
- **英文模型指南：** [打开](../../models/sgwr.md)
- **API：** [打开](../../api/models/sgwr.md)

## 12. 完整可运行示例

```python
# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Fit similarity and geographically weighted regression."""

from pygwrx import SGWR
from _common import print_model_result, spatial_regression

X, y, coords = spatial_regression(n=48, p=3)
model = SGWR(
    bandwidth=24,
    adaptive=True,
    alpha=0.45,
    similarity_vars=["x1", "x2"],
    store_weights=True,
).fit(X, y, coords)
print_model_result(model)
print("combined_weights_shape=", model.combined_weights_.shape)
print("predictions=", model.predict(X.iloc[:3], coords.iloc[:3]))
```

该脚本是项目 API—示例覆盖检查所使用的正式示例，可通过 `python examples/run_all.py` 批量运行。
