# Geographically and Temporally Weighted Regression（`GTWR`）

> **pyGWRx 模型编号 06｜类别：单尺度时空局部回归**
> 本文同时说明原始方法与当前 pyGWRx 实现。凡实现与论文求解器不同之处均会明确指出。

正式来源：[Huang, Wu & Barry (2010), *Geographically and temporally weighted regression for modeling spatio-temporal variation in house prices*](https://doi.org/10.1080/13658810802672469)。


## 1. 模型要解决的核心问题

空间统计中最危险的假设之一，是默认一组回归关系在整个研究区完全相同。若某个变量在城市中心、郊区、沿海和山区产生不同影响，一个全局系数只能给出平均效应，局部差异会被平均掉。地理加权方法的共同思想是：在每个目标位置建立一个局部窗口，让距离更近、时间更近、属性更相似或属于同一机制的观测获得更高权重，再估计该位置的局部统计量或局部模型。

需要强调：局部模型不是自动的因果模型。它首先是一种用于描述、探索和预测空间异质性的工具。带宽、核函数、局部共线性、异常值、残差空间相关和多重检验均会改变解释，必须与诊断一起使用。


## 2. 一句话思想

空间上接近但年代不同的观测不一定相似；时间接近但相距遥远的观测也不一定相似。GTWR 在一个综合时空距离中同时表达两者。

## 3. 数学模型

pyGWRx 默认支持与 GWmodel 思路一致的时空距离组合：

$$
d_{ij}^{ST}=\lambda d_{ij}^{S}+(1-\lambda)d_{ij}^{T}
+2\sqrt{\lambda(1-\lambda)d_{ij}^{S}d_{ij}^{T}}\cos(\xi),
$$

其中 $\lambda\in[0,1]$ 控制空间与时间相对贡献，$\xi$ 控制交叉项方向。另一常见形式是

$$
d_{ij}^{ST}=\sqrt{(d_{ij}^{S})^2+\tau(d_{ij}^{T})^2}.
$$

随后将 $d_{ij}^{ST}$ 代入 GWR 核函数并执行局部 WLS。若 `causal=True`，未来观测的权重置零。

## 4. 算法流程

1. 统一时间单位并计算空间、时间距离。
2. 选择时空距离组合和空间—时间比例参数。
3. 联合搜索比例参数与核带宽。
4. 形成时空权重并标定局部系数。
5. 按时间切片检查系数、残差和轨迹；预测时可启用因果过滤。

## 5. pyGWRx 当前实现

```python
from pygwrx import GTWR

model = GTWR(kernel='bisquare', bandwidth='cv', bandwidth_method='cv', adaptive=False, lambda_st=0.05, ksi=0.0, distance_combination='gwmodel', tau=1.0, causal=False, time_unit='auto', ...)
```

pyGWRx 的 `GTWR` 支持 datetime/数值时间、自动时间单位、`lambda_st` 搜索、`ksi`、`tau`、GWmodel 或欧氏组合、固定/自适应带宽和 causal 模式。

### 5.1 输入语义

- `X`：自变量矩阵；启用 `fit_intercept=True` 时由模型统一添加截距。
- `y`：响应或类别，具体形状和分布要求由模型决定。
- `coords`：校准位置坐标；经纬度数据在使用欧氏距离前应投影，或选择模型支持的相应距离语义。
- 带宽：固定模式表示距离阈值，自适应模式通常表示近邻数量；两者不能混为同一单位。

### 5.2 结果与诊断

应优先检查：拟合值与残差、局部系数、带宽或尺度、有效参数个数、AICc/CV、标准误与显著性、局部共线性、影响度、残差空间结构。模型专用结果请结合下方图件和 `pygwrx.diagnostics` 使用。

## 6. 适用场景

重复横截面、房价、环境监测、交通和社会过程同时具有空间与时间非平稳性时使用。

## 7. 关键局限与误用风险

空间与时间单位的缩放决定结果；单一综合带宽仍假定所有系数共享尺度；非因果模式会在预测历史时使用未来信息；时间密度不均会改变邻域。

## 8. 推荐可视化

![18 gtwr slices](../../assets/figures/specialized/18_gtwr_slices.png)

![19 gtwr trajectory](../../assets/figures/specialized/19_gtwr_trajectory.png)

![20 gtwr residuals](../../assets/figures/specialized/20_gtwr_residuals.png)

## 9. 最小工作流

```python
# 以下是接口结构示意；不同模型的 fit 参数可能包括 times、attributes 或阶段列表。
model = GTWR(...)
model.fit(X, y, coords)

# 常见结果
# model.fitted_values_
# model.residuals_
# model.local_parameters_ / model.coef_
# model.diagnostics_
```

推荐工作顺序：全局模型 → 带宽/尺度选择 → 局部拟合 → 推断校正 → 局部共线性和影响诊断 → 空间分块验证 → 图件与结论。

## 10. 主要参考资料

- [Huang, Wu & Barry (2010), *Geographically and temporally weighted regression for modeling spatio-temporal variation in house prices*](https://doi.org/10.1080/13658810802672469)

---

**版本说明：** 本文依据当前 pyGWRx 0.1.2 Alpha 源码与算法知识库整理。它描述的是当前已验证实现，而不是对任意同名软件的泛化说明。


## 11. 当前能力边界

- **输入：** X, y, coordinates, and row-wise times
- **主要操作：** fit, score, predict, predict_result
- **新位置能力：** Validated at new space-time targets; causal filtering is available when configured.
- **安装分组：** `base`
- **英文模型指南：** [打开](../../models/gtwr.md)
- **API：** [打开](../../api/models/gtwr.md)

## 12. 完整可运行示例

```python
# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Fit and predict with geographically and temporally weighted regression."""

from pygwrx import GTWR, GTWRPredictionResult
from _common import print_model_result, temporal_regression

X, y, coords, times = temporal_regression()
model = GTWR(kernel="bisquare", bandwidth=24, adaptive=True, lambda_st=0.3).fit(
    X, y, coords, times
)
print_model_result(model)
print("score=", model.score(X, y, coords, times=times))
result = model.predict_result(X.iloc[:3], coords.iloc[:3], times[:3])
assert isinstance(result, GTWRPredictionResult)
print(result.to_frame())
```

该脚本是项目 API—示例覆盖检查所使用的正式示例，可通过 `python examples/run_all.py` 批量运行。
