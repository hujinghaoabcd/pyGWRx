# Latent-Geometry Geographically Weighted Regression（`LGGWR`）

> **pyGWRx 模型编号 18｜类别：原创：可学习邻近几何**
> 本文同时说明原始方法与当前 pyGWRx 实现。凡实现与论文求解器不同之处均会明确指出。

pyGWRx 原创研究模型。它受 GWR、SGWR 和可学习权重方法启发，但其“线性潜在几何 + 解析 LOO 梯度 + 可分离安全退化”的组合是本项目定义。


## 1. 模型要解决的核心问题

空间统计中最危险的假设之一，是默认一组回归关系在整个研究区完全相同。若某个变量在城市中心、郊区、沿海和山区产生不同影响，一个全局系数只能给出平均效应，局部差异会被平均掉。地理加权方法的共同思想是：在每个目标位置建立一个局部窗口，让距离更近、时间更近、属性更相似或属于同一机制的观测获得更高权重，再估计该位置的局部统计量或局部模型。

需要强调：局部模型不是自动的因果模型。它首先是一种用于描述、探索和预测空间异质性的工具。带宽、核函数、局部共线性、异常值、残差空间相关和多重检验均会改变解释，必须与诊断一起使用。


## 2. 一句话思想

模型不直接接受地图给出的距离，而是学习一把新的尺子：坐标、社会经济、环境与结构属性经过线性映射后形成潜在空间，真正决定局部关系的邻近性在这个空间中计算。

## 3. 数学模型

令

$$
u_i=[s_i,a_i]\in\mathbb R^{2+q},\qquad z_i=A u_i\in\mathbb R^k.
$$

潜在距离和权重为

$$
d_{ij}=\|A(u_i-u_j)\|_2,\qquad
w_{ij}=K(d_{ij}/h).
$$

通过留一局部回归学习 $A$：

$$
L(A)=\frac1n\sum_i\left(y_i-x_i^\top\beta_i^{(-i)}(A)\right)^2,
$$

其中 $w_{ii}=0$。解析梯度为

$$
\frac{\partial L}{\partial A}
=-\frac{2}{n}\sum_i\sum_{j\ne i}
 r_i s_{ij}e_{ij}\frac{K'(d_{ij})}{d_{ij}}
 (z_i-z_j)(u_i-u_j)^\top.
$$

可分离形式为

$$
w_{ij}=K(d_{ij}^{geo}/h_g)\,
K(\|B(a_i-a_j)\|/h_a),
$$

当 $h_a\to\infty$ 时精确退化为标准 GWR。旋转不变的解释对象为

$$
M=A^\top A,
$$

因为 $d_{ij}^2=(u_i-u_j)^\top M(u_i-u_j)$。

## 4. 算法流程

1. 分别标准化坐标和属性几何输入。
2. 以坐标、PCA 或随机方式初始化潜在映射。
3. 在固定工作带宽下执行 LOO 局部回归。
4. 用解析梯度和 Adam 更新映射，进行梯度裁剪和尺度投影。
5. 多次重启保留最低 LOO 损失解。
6. 按 AICc 重选带宽，并可在几何与带宽间交替。
7. 用标准自权重局部回归产生最终系数、帽子矩阵和诊断。
8. 输出潜在坐标、$M=A^\top A$ 和变量贡献。

## 5. pyGWRx 当前实现

```python
from pygwrx import LGGWR

model = LGGWR(latent_dim=2, bandwidth=None, adaptive=False, kernel='gaussian', geometry='joint', learning_rate=0.05, max_iter=100, lambda_reg=0.0, grad_clip=10.0, select_bandwidth=True, fit_intercept=True, standardize_geometry=True, initialization='coordinate', n_restarts=1, scale_constraint='frobenius', bandwidth_updates=1, ...)
```

当前实现修正了旧原型中的随机占位梯度，使用解析梯度并通过有限差分测试；默认固定 Frobenius 范数解决 $A$ 与带宽的尺度不可识别；支持 joint/separable、确定性重启、DataFrame 列名安全预测和最终状态记录。

### 5.1 输入语义

- `X`：自变量矩阵；启用 `fit_intercept=True` 时由模型统一添加截距。
- `y`：响应或类别，具体形状和分布要求由模型决定。
- `coords`：校准位置坐标；经纬度数据在使用欧氏距离前应投影，或选择模型支持的相应距离语义。
- 带宽：固定模式表示距离阈值，自适应模式通常表示近邻数量；两者不能混为同一单位。

### 5.2 结果与诊断

应优先检查：拟合值与残差、局部系数、带宽或尺度、有效参数个数、AICc/CV、标准误与显著性、局部共线性、影响度、残差空间结构。模型专用结果请结合下方图件和 `pygwrx.diagnostics` 使用。

## 6. 适用场景

当空间非平稳更可能由上下文相似、功能联系或结构属性驱动，而非单纯地图距离时使用；也可作为“GWR 的距离定义是否正确”的研究工具。

## 7. 关键局限与误用风险

是原创研究模型而非已广泛复现的标准；线性映射不能表达复杂非线性几何；$A$ 本身受旋转影响，应解释 $A^\top A$；$O(n^2)$ 训练限制大样本；在纯地理或真实留出数据上不保证优于 GWR。

## 8. 推荐可视化

![27 lggwr latent](../../assets/figures/specialized/27_lggwr_latent.png)

![28 lggwr metric](../../assets/figures/specialized/28_lggwr_metric.png)

![29 lggwr training](../../assets/figures/specialized/29_lggwr_training.png)

![30 lggwr neighbours](../../assets/figures/specialized/30_lggwr_neighbours.png)

## 9. 最小工作流

```python
# 以下是接口结构示意；不同模型的 fit 参数可能包括 times、attributes 或阶段列表。
model = LGGWR(...)
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
- [Lessani & Li (2024), *SGWR: similarity and geographically weighted regression*](https://doi.org/10.1080/13658816.2024.2342319)
- [Hagenauer & Helbich (2022), *A geographically weighted artificial neural network*](https://doi.org/10.1080/13658816.2021.1871618)

---

**版本说明：** 本文依据当前 pyGWRx 0.1.2 Alpha 源码与算法知识库整理。它描述的是当前已验证实现，而不是对任意同名软件的泛化说明。


## 11. 当前能力边界

- **输入：** X, y, coordinates, and contextual attributes
- **主要操作：** fit, predict, predict_result
- **新位置能力：** Validated using the learned geometry transform and target attributes.
- **安装分组：** `base`
- **英文模型指南：** [打开](../../models/lg-gwr.md)
- **API：** [打开](../../api/models/lg-gwr.md)

## 12. 完整可运行示例

```python
# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Fit latent-geometry GWR with auxiliary contextual attributes."""

from pygwrx import LGGWR, LGGWRPredictionResult
from _common import latent_regression, print_model_result

X, y, coords, attributes = latent_regression()
model = LGGWR(
    latent_dim=2, bandwidth=2.5, select_bandwidth=False, max_iter=8, random_state=0
).fit(X, y, coords, attributes)
print_model_result(model)
print("latent_coordinates_shape=", model.latent_coords_.shape)
result = model.predict_result(X.iloc[:3], coords.iloc[:3], attributes.iloc[:3])
assert isinstance(result, LGGWRPredictionResult)
print(result.to_frame())
```

该脚本是项目 API—示例覆盖检查所使用的正式示例，可通过 `python examples/run_all.py` 批量运行。
