# Geographically Weighted Discriminant Analysis（`GWDA`）

> **pyGWRx 模型编号 15｜类别：局部分类**
> 本文同时说明原始方法与当前 pyGWRx 实现。凡实现与论文求解器不同之处均会明确指出。

正式来源：[Brunsdon, Fotheringham & Charlton (2007), *Geographically Weighted Discriminant Analysis*](https://doi.org/10.1111/j.1538-4632.2007.00709.x)。


## 1. 模型要解决的核心问题

空间统计中最危险的假设之一，是默认一组回归关系在整个研究区完全相同。若某个变量在城市中心、郊区、沿海和山区产生不同影响，一个全局系数只能给出平均效应，局部差异会被平均掉。地理加权方法的共同思想是：在每个目标位置建立一个局部窗口，让距离更近、时间更近、属性更相似或属于同一机制的观测获得更高权重，再估计该位置的局部统计量或局部模型。

需要强调：局部模型不是自动的因果模型。它首先是一种用于描述、探索和预测空间异质性的工具。带宽、核函数、局部共线性、异常值、残差空间相关和多重检验均会改变解释，必须与诊断一起使用。


## 2. 一句话思想

不同地区的类别特征分布可能不同。GWDA 在每个位置局部估计类别均值、协方差和先验，再进行 LDA 或 QDA 分类。

## 3. 数学模型

局部类别 $c$ 的权重和为 $n_{ic}^w=\sum_{j:y_j=c}w_{ij}$，局部均值为

$$
\mu_{ic}=\frac{\sum_{j:y_j=c}w_{ij}x_j}{n_{ic}^w}.
$$

LDA 使用共享局部协方差 $\Sigma_i$：

$$
\delta_{ic}(x)=x^\top\Sigma_i^{-1}\mu_{ic}
-\frac12\mu_{ic}^\top\Sigma_i^{-1}\mu_{ic}
+\log\pi_{ic}.
$$

QDA 则为每类使用 $\Sigma_{ic}$ 并加入 $-\frac12\log|\Sigma_{ic}|$。

## 4. 算法流程

1. 按类别检查每个局部邻域的有效样本。
2. 选择带宽，可用 LOOCV 分类准确率。
3. 局部估计类均值、协方差和先验。
4. 计算判别分数与后验概率。
5. 输出类别、置信度、熵和混淆矩阵。

## 5. pyGWRx 当前实现

```python
from pygwrx import GWDA

model = GWDA(kernel='bisquare', bandwidth='cv', adaptive=True, quadratic=False, local_mean=True, local_cov=True, local_prior=True, prior=None, regularization=0.0, verbose=False)
```

pyGWRx 支持局部 LDA/QDA、局部/全局均值协方差先验开关、协方差正则、LOOCV 带宽选择、概率与熵输出。

### 5.1 输入语义

- `X`：自变量矩阵；启用 `fit_intercept=True` 时由模型统一添加截距。
- `y`：响应或类别，具体形状和分布要求由模型决定。
- `coords`：校准位置坐标；经纬度数据在使用欧氏距离前应投影，或选择模型支持的相应距离语义。
- 带宽：固定模式表示距离阈值，自适应模式通常表示近邻数量；两者不能混为同一单位。

### 5.2 结果与诊断

应优先检查：拟合值与残差、局部系数、带宽或尺度、有效参数个数、AICc/CV、标准误与显著性、局部共线性、影响度、残差空间结构。模型专用结果请结合下方图件和 `pygwrx.diagnostics` 使用。

## 6. 适用场景

类别边界和类条件分布具有空间异质性的选举、土地利用、疾病类型等分类问题。

## 7. 关键局限与误用风险

局部类别稀少会使协方差不可估；QDA 参数更多；准确率可能受空间泄漏影响；应使用空间分块验证并报告类别不平衡。

## 8. 推荐可视化

![15 gwda class](../../assets/figures/specialized/15_gwda_class.png)

![16 gwda confidence](../../assets/figures/specialized/16_gwda_confidence.png)

![17 gwda confusion](../../assets/figures/specialized/17_gwda_confusion.png)

## 9. 最小工作流

```python
# 以下是接口结构示意；不同模型的 fit 参数可能包括 times、attributes 或阶段列表。
model = GWDA(...)
model.fit(X, y, coords)

# 常见结果
# model.fitted_values_
# model.residuals_
# model.local_parameters_ / model.coef_
# model.diagnostics_
```

推荐工作顺序：全局模型 → 带宽/尺度选择 → 局部拟合 → 推断校正 → 局部共线性和影响诊断 → 空间分块验证 → 图件与结论。

## 10. 主要参考资料

- [Brunsdon, Fotheringham & Charlton (2007), *Geographically Weighted Discriminant Analysis*](https://doi.org/10.1111/j.1538-4632.2007.00709.x)

---

**版本说明：** 本文依据当前 pyGWRx 0.1.2 Alpha 源码与算法知识库整理。它描述的是当前已验证实现，而不是对任意同名软件的泛化说明。


## 11. 当前能力边界

- **输入：** X, class labels, coordinates
- **主要操作：** fit, predict, predict_proba
- **新位置能力：** Validated class labels and local class probabilities.
- **安装分组：** `base`
- **英文模型指南：** [打开](../../models/gwda.md)
- **API：** [打开](../../api/models/gwda.md)

## 12. 完整可运行示例

```python
# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Fit geographically weighted discriminant analysis."""

from pygwrx import GWDA
from _common import classification_data

X, y, coords = classification_data()
model = GWDA(bandwidth=28, adaptive=True, quadratic=False).fit(X, y, coords)
print(model.summary())
print("classes=", model.classes_)
print("predictions=", model.predict(X.iloc[:5], coords.iloc[:5]))
print("probabilities=", model.predict_proba(X.iloc[:5], coords.iloc[:5]))
```

该脚本是项目 API—示例覆盖检查所使用的正式示例，可通过 `python examples/run_all.py` 批量运行。
