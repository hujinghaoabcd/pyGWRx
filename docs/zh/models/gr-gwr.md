# Geo-Regime Geographically Weighted Regression（`GRGWR`）

> **pyGWRx 模型编号 19｜类别：原创：分区内平滑、分区间突变**
> 本文同时说明原始方法与当前 pyGWRx 实现。凡实现与论文求解器不同之处均会明确指出。

pyGWRx 原创研究模型。它连接了 GWR、空间约束区域化和 ICM/Potts 型边界惩罚，但具体算法链条与条件诊断由本项目定义。


## 1. 模型要解决的核心问题

空间统计中最危险的假设之一，是默认一组回归关系在整个研究区完全相同。若某个变量在城市中心、郊区、沿海和山区产生不同影响，一个全局系数只能给出平均效应，局部差异会被平均掉。地理加权方法的共同思想是：在每个目标位置建立一个局部窗口，让距离更近、时间更近、属性更相似或属于同一机制的观测获得更高权重，再估计该位置的局部统计量或局部模型。

需要强调：局部模型不是自动的因果模型。它首先是一种用于描述、探索和预测空间异质性的工具。带宽、核函数、局部共线性、异常值、残差空间相关和多重检验均会改变解释，必须与诊断一起使用。


## 2. 一句话思想

标准 GWR 像一支软刷子，会把边界两侧不同的关系涂成渐变。GR-GWR 允许空间被划分为若干机制区：区内仍用 GWR 平滑，跨区边界则允许系数突然跳变。

## 3. 数学模型

机制标签为 $z_i\in\{1,\ldots,K\}$，目标为

$$
L(z)=\sum_i\left[y_i-x_i^\top\beta^{(z_i)}(s_i)\right]^2
+\lambda B(z),
$$

其中唯一无向边界数

$$
B(z)=\sum_{(i,j)\in E}\mathbf1(z_i\ne z_j).
$$

初始聚类特征为

$$
f_i=\left[
\sqrt{1-\gamma}\,\widetilde{\beta}^{slope}_i;
\sqrt{\gamma}\,\widetilde{s}_i
\right],
$$

$\gamma=0$ 为纯局部关系，$\gamma=1$ 为纯空间。ICM 更新节点 $i$ 到候选区 $r$ 的代价为

$$
C_i(r)=\left[y_i-x_i^\top\hat\beta_{i,r}^{(-i)}\right]^2
+\lambda\sum_{j\in\mathcal N(i)}\mathbf1(z_j\ne r),
$$

其中 $\hat\beta_{i,r}^{(-i)}$ 用当前属于 $r$ 的样本在位置 $i$ 重新做留一局部 WLS。

## 4. 算法流程

1. 对全体样本拟合标准 GWR，获得初始局部斜率。
2. 排除截距，对斜率与归一化坐标构造聚类特征。
3. 在统一对称 kNN 图及其 MST 连通约束下做 Ward 初始化。
4. 在每个机制区内部拟合 GWR。
5. 按确定性顺序 ICM，逐点比较当前区和相邻候选区的留一局部代价。
6. 禁止使源机制区断裂或小于最小样本量的移动。
7. 每轮完整重拟合并用总目标守卫，只接受不增更新。
8. 构造区内支持的帽子矩阵，报告条件 ENP/AICc。
9. 预测新点时先分配机制，再用该机制训练样本在新位置重新做局部 WLS。

## 5. pyGWRx 当前实现

```python
from pygwrx import GRGWR

model = GRGWR(n_regimes=3, bandwidth=20, kernel='bisquare', lambda_boundary=1.0, max_iter=10, tol=1e-4, spatial_constraint_weight=0.5, fit_intercept=True, n_neighbors=8, min_regime_size=None, enforce_connectivity=True, random_state=42, ...)
```

当前实现修复了旧原型的空标签、不连续标签、方向性边计数、同步更新和最近系数预测等问题；使用统一无向图、MST、连通约束 Ward、顺序 ICM、目标守卫和直接预测重标定。

### 5.1 输入语义

- `X`：自变量矩阵；启用 `fit_intercept=True` 时由模型统一添加截距。
- `y`：响应或类别，具体形状和分布要求由模型决定。
- `coords`：校准位置坐标；经纬度数据在使用欧氏距离前应投影，或选择模型支持的相应距离语义。
- 带宽：固定模式表示距离阈值，自适应模式通常表示近邻数量；两者不能混为同一单位。

### 5.2 结果与诊断

应优先检查：拟合值与残差、局部系数、带宽或尺度、有效参数个数、AICc/CV、标准误与显著性、局部共线性、影响度、残差空间结构。模型专用结果请结合下方图件和 `pygwrx.diagnostics` 使用。

## 6. 适用场景

行政边界、地质单元、政策区、市场区或土地利用分区导致关系在边界突变，而区内仍存在平滑变化时使用。

## 7. 关键局限与误用风险

机制数、图结构、$\lambda$、$\gamma$ 和初始 GWR 带宽会影响分区；ICM 只能获得局部最优；发现标签本身有离散模型选择复杂度，因此 AICc/ENP 仅条件于最终标签，不能当作完整无偏复杂度；真实机制需要外部证据验证。

## 8. 推荐可视化

![31 grgwr regimes](../../assets/figures/specialized/31_grgwr_regimes.png)

![32 grgwr convergence](../../assets/figures/specialized/32_grgwr_convergence.png)

![33 grgwr sizes](../../assets/figures/specialized/33_grgwr_sizes.png)

![34 grgwr coefficient](../../assets/figures/specialized/34_grgwr_coefficient.png)

## 9. 最小工作流

```python
# 以下是接口结构示意；不同模型的 fit 参数可能包括 times、attributes 或阶段列表。
model = GRGWR(...)
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
- [Mei et al. (2004), *A Note on the Mixed Geographically Weighted Regression Model*](https://doi.org/10.1111/j.1085-9489.2004.00331.x)
- [Comber et al. (2022), *A route map for the informed application of GWR*](https://doi.org/10.1111/gean.12316)

---

**版本说明：** 本文依据当前 pyGWRx 0.1.2 Alpha 源码与算法知识库整理。它描述的是当前已验证实现，而不是对任意同名软件的泛化说明。


## 11. 当前能力边界

- **输入：** X, y, coordinates, regime count, and connectivity settings
- **主要操作：** fit, predict, predict_result
- **新位置能力：** Validated using learned regime structure and target assignment logic.
- **安装分组：** `ml`
- **英文模型指南：** [打开](../../models/gr-gwr.md)
- **API：** [打开](../../api/models/gr-gwr.md)

## 12. 完整可运行示例

```python
# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Fit geo-regime GWR and inspect connected spatial regimes."""

from pygwrx import GRGWR, GRGWRPredictionResult
from _common import print_model_result, regime_regression

X, y, coords, truth = regime_regression(n=56)
model = GRGWR(n_regimes=2, bandwidth=18, max_iter=2, random_state=0).fit(X, y, coords)
print_model_result(model)
print("regime_sizes=", model.regime_sizes_)
print(
    "truth_agreement_or_label_swap=",
    max((model.regimes_ == truth).mean(), (model.regimes_ != truth).mean()),
)
result = model.predict_result(X.iloc[:3], coords.iloc[:3])
assert isinstance(result, GRGWRPredictionResult)
print(result.to_frame())
```

该脚本是项目 API—示例覆盖检查所使用的正式示例，可通过 `python examples/run_all.py` 批量运行。
