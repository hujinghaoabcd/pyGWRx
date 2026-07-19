# pyGWRx 全模型算法百科

> 本文汇总 19 个正式模型。每章均以原始方法为背景，并以当前 pyGWRx 源码为实现边界。


---

# 第 1 章 地理加权回归（GWR）

> **pyGWRx 模型编号 01｜类别：基础局部回归**
> 本文同时说明原始方法与当前 pyGWRx 实现。凡实现与论文求解器不同之处均会明确指出。

经典来源：[Brunsdon, Fotheringham & Charlton (1996), *Geographically Weighted Regression: A Method for Exploring Spatial Nonstationarity*](https://doi.org/10.1111/j.1538-4632.1996.tb00936.x)；系统专著：[Fotheringham, Brunsdon & Charlton (2002), *Geographically Weighted Regression: The Analysis of Spatially Varying Relationships*](https://www.wiley.com/en-us/Geographically+Weighted+Regression%3A+The+Analysis+of+Spatially+Varying+Relationships-p-9780471496168)。


## 1. 模型要解决的核心问题

空间统计中最危险的假设之一，是默认一组回归关系在整个研究区完全相同。若某个变量在城市中心、郊区、沿海和山区产生不同影响，一个全局系数只能给出平均效应，局部差异会被平均掉。地理加权方法的共同思想是：在每个目标位置建立一个局部窗口，让距离更近、时间更近、属性更相似或属于同一机制的观测获得更高权重，再估计该位置的局部统计量或局部模型。

需要强调：局部模型不是自动的因果模型。它首先是一种用于描述、探索和预测空间异质性的工具。带宽、核函数、局部共线性、异常值、残差空间相关和多重检验均会改变解释，必须与诊断一起使用。


## 2. 一句话思想

把一套全局回归拆成在每个空间位置标定的一组局部加权最小二乘回归。它回答的不是“全区域平均关系是多少”，而是“关系在每个位置附近是什么样”。

## 3. 数学模型

设观测位置为 $s_i=(u_i,v_i)$，设计矩阵为 $X$，响应为 $y$。位置 $s_i$ 的局部系数为

$$
\hat{\boldsymbol\beta}(s_i)
=\left(X^\top W_iX\right)^{-1}X^\top W_i y,
$$

其中 $W_i=\operatorname{diag}(w_{i1},\ldots,w_{in})$。常见核函数为

$$
\text{Gaussian: }w_{ij}=\exp\!\left[-\frac12(d_{ij}/h_i)^2\right],
$$

$$
\text{bisquare: }w_{ij}=\left[1-(d_{ij}/h_i)^2\right]^2\mathbf 1(d_{ij}<h_i),
$$

$$
\text{exponential: }w_{ij}=\exp(-d_{ij}/h_i).
$$

固定带宽中 $h_i=h$ 是距离；自适应带宽中 $h_i$ 是位置 $i$ 到第 $k$ 个近邻的距离。

## 4. 算法流程

1. 验证坐标、响应和自变量并决定是否添加截距。
2. 计算空间距离矩阵。
3. 通过 CV 或 AICc 选择固定距离带宽或自适应近邻数。
4. 在每个位置形成局部权重并求解 WLS。
5. 构造帽子矩阵，计算 ENP、AIC/AICc/BIC、局部标准误、t 值、Local R²、影响度和 Cook’s D。
6. 预测时在新位置重新形成权重并标定局部系数。

## 5. pyGWRx 当前实现

```python
from pygwrx import GWR

model = GWR(kernel='gaussian', bandwidth='cv', bandwidth_method='cv', adaptive=False, bandwidth_range=None, optimization_method='golden_section', fit_intercept=True, distance_metric='euclidean', sigma2_v1=True, verbose=False)
```

pyGWRx 的 `GWR` 是其他回归模型的统一基线。它支持 Gaussian、bisquare、exponential 或可调用核；固定/自适应带宽；CV/AICc；欧氏距离等；拟合后保存系数、预测、残差、帽子矩阵、推断统计量和局部诊断。`predict_result()` 返回新位置的系数与预测，而不是简单插值训练系数。

### 5.1 输入语义

- `X`：自变量矩阵；启用 `fit_intercept=True` 时由模型统一添加截距。
- `y`：响应或类别，具体形状和分布要求由模型决定。
- `coords`：校准位置坐标；经纬度数据在使用欧氏距离前应投影，或选择模型支持的相应距离语义。
- 带宽：固定模式表示距离阈值，自适应模式通常表示近邻数量；两者不能混为同一单位。

### 5.2 结果与诊断

应优先检查：拟合值与残差、局部系数、带宽或尺度、有效参数个数、AICc/CV、标准误与显著性、局部共线性、影响度、残差空间结构。模型专用结果请结合下方图件和 `pygwrx.diagnostics` 使用。

## 6. 适用场景

适合连续响应、样本位置明确、关系可能平滑地随空间变化、且研究目标需要局部解释或局部预测的场景。建议先拟合 OLS，再检查非平稳性、残差和带宽。

## 7. 关键局限与误用风险

GWR 假定所有系数共享同一空间尺度；局部样本过少会造成不稳定；局部共线性可能放大系数；大量逐位置检验存在多重比较问题；残差空间相关意味着遗漏结构；它不应仅凭较高的样本内 $R^2$ 被判为优越。

## 8. 推荐可视化

![01 coefficient](../assets/figures/core/01_coefficient.png)

![02 coefficient significant](../assets/figures/core/02_coefficient_significant.png)

![04 local r2](../assets/figures/core/04_local_r2.png)

![05 standardized residual](../assets/figures/core/05_standardized_residual.png)

![10 kernel weights](../assets/figures/core/10_kernel_weights.png)

![12 diagnostic panel](../assets/figures/core/12_diagnostic_panel.png)

## 9. 最小工作流

```python
# 以下是接口结构示意；不同模型的 fit 参数可能包括 times、attributes 或阶段列表。
model = GWR(...)
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
- [Fotheringham, Brunsdon & Charlton (2002), *Geographically Weighted Regression: The Analysis of Spatially Varying Relationships*](https://www.wiley.com/en-us/Geographically+Weighted+Regression%3A+The+Analysis+of+Spatially+Varying+Relationships-p-9780471496168)
- [Comber et al. (2022), *A route map for the informed application of GWR*](https://doi.org/10.1111/gean.12316)

---

**版本说明：** 本文依据 `pyGWRx_diagnostics_visualization_complete` 源码生成。它描述的是当前已验证实现，而不是对任意同名软件的泛化说明。


---

# 第 2 章 多尺度地理加权回归（MGWR）

> **pyGWRx 模型编号 02｜类别：多尺度局部回归**
> 本文同时说明原始方法与当前 pyGWRx 实现。凡实现与论文求解器不同之处均会明确指出。

正式来源：[Fotheringham, Yang & Kang (2017), *Multiscale Geographically Weighted Regression (MGWR)*](https://doi.org/10.1080/24694452.2017.1352480)。


## 1. 模型要解决的核心问题

空间统计中最危险的假设之一，是默认一组回归关系在整个研究区完全相同。若某个变量在城市中心、郊区、沿海和山区产生不同影响，一个全局系数只能给出平均效应，局部差异会被平均掉。地理加权方法的共同思想是：在每个目标位置建立一个局部窗口，让距离更近、时间更近、属性更相似或属于同一机制的观测获得更高权重，再估计该位置的局部统计量或局部模型。

需要强调：局部模型不是自动的因果模型。它首先是一种用于描述、探索和预测空间异质性的工具。带宽、核函数、局部共线性、异常值、残差空间相关和多重检验均会改变解释，必须与诊断一起使用。


## 2. 一句话思想

GWR 用一把尺子测量所有关系，而 MGWR 允许每个系数拥有自己的空间作用尺度。一个变量可以是几乎全局的，另一个变量只在很小邻域内变化。

## 3. 数学模型

MGWR 的加性形式为

$$
y_i=\sum_{k=0}^{p}\beta_k(s_i)x_{ik}+\varepsilon_i,
$$

但第 $k$ 个系数通过自己的权重矩阵 $W_{ik}(h_k)$ 估计。反向拟合中，对第 $k$ 项构造部分残差

$$
r_i^{(-k)}=y_i-\sum_{\ell\ne k}x_{i\ell}\hat\beta_\ell(s_i),
$$

再用带宽 $h_k$ 的单变量局部回归更新

$$
\hat\beta_k(s_i)
=\frac{\sum_jw_{ij}(h_k)x_{jk}r_j^{(-k)}}
{\sum_jw_{ij}(h_k)x_{jk}^2}.
$$

迭代直至系数面、残差平方和或带宽稳定。每个 $h_k$ 可解释为对应过程的空间尺度。

## 4. 算法流程

1. 先用单带宽 GWR 或指定带宽初始化。
2. 逐变量构造部分残差。
3. 为每个变量独立搜索 AICc/CV 最优带宽。
4. 用该变量的带宽更新局部系数面。
5. 循环反向拟合，直至收敛或带宽连续多轮不变。
6. 基于各项平滑矩阵构造精确或近似推断与 ENP。

## 5. pyGWRx 当前实现

```python
from pygwrx import MGWR

model = MGWR(kernel='bisquare', bandwidths=None, bandwidth_method='aicc', adaptive=True, bandwidth_range=None, bandwidth_ranges=None, init_bandwidth=None, optimization_method='golden_section', search_tol=1e-6, search_max_iter=200, max_iter=200, tol=1e-5, rss_score=False, bws_same_times=5, fit_intercept=True, distance_metric='euclidean', sigma2_v1=True, verbose=False)
```

pyGWRx 的 `MGWR` 采用 Gaussian 加性 MGWR，支持每个设计列独立带宽、自动或手动范围、反向拟合、精确平滑矩阵与协方差诊断和校准位置结果表；当前不支持独立新位置预测。截距若启用也具有自己的带宽。

### 5.1 输入语义

- `X`：自变量矩阵；启用 `fit_intercept=True` 时由模型统一添加截距。
- `y`：响应或类别，具体形状和分布要求由模型决定。
- `coords`：校准位置坐标；经纬度数据在使用欧氏距离前应投影，或选择模型支持的相应距离语义。
- 带宽：固定模式表示距离阈值，自适应模式通常表示近邻数量；两者不能混为同一单位。

### 5.2 结果与诊断

应优先检查：拟合值与残差、局部系数、带宽或尺度、有效参数个数、AICc/CV、标准误与显著性、局部共线性、影响度、残差空间结构。模型专用结果请结合下方图件和 `pygwrx.diagnostics` 使用。

## 6. 适用场景

当理论上不同驱动因素作用于不同空间尺度，或 GWR 的统一带宽导致某些系数过度平滑、另一些系数过度波动时使用。

## 7. 关键局限与误用风险

计算成本高于 GWR；带宽之间可能相互影响；局部系数仍可能共线；极大的带宽更接近全局效应，但不等于严格的全局固定系数；解释时必须把系数大小和带宽尺度同时考虑。

## 8. 推荐可视化

![09 mgwr bandwidths](../assets/figures/core/09_mgwr_bandwidths.png)

![11 gwr mgwr comparison](../assets/figures/core/11_gwr_mgwr_comparison.png)

## 9. 最小工作流

```python
# 以下是接口结构示意；不同模型的 fit 参数可能包括 times、attributes 或阶段列表。
model = MGWR(...)
model.fit(X, y, coords)

# 常见结果
# model.fitted_values_
# model.residuals_
# model.local_parameters_ / model.coef_
# model.diagnostics_
```

推荐工作顺序：全局模型 → 带宽/尺度选择 → 局部拟合 → 推断校正 → 局部共线性和影响诊断 → 空间分块验证 → 图件与结论。

## 10. 主要参考资料

- [Fotheringham, Yang & Kang (2017), *Multiscale Geographically Weighted Regression (MGWR)*](https://doi.org/10.1080/24694452.2017.1352480)
- [Comber et al. (2022), *A route map for the informed application of GWR*](https://doi.org/10.1111/gean.12316)

---

**版本说明：** 本文依据 `pyGWRx_diagnostics_visualization_complete` 源码生成。它描述的是当前已验证实现，而不是对任意同名软件的泛化说明。


---

# 第 3 章 稳健地理加权回归（RGWR）

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

![01 rgwr weights](../assets/figures/specialized/01_rgwr_weights.png)

![02 rgwr convergence](../assets/figures/specialized/02_rgwr_convergence.png)

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

**版本说明：** 本文依据 `pyGWRx_diagnostics_visualization_complete` 源码生成。它描述的是当前已验证实现，而不是对任意同名软件的泛化说明。


---

# 第 4 章 局部条件岭地理加权回归（LCR-GWR）

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

![07 gwr condition number](../assets/figures/core/07_gwr_condition_number.png)

![08 lcr lambda](../assets/figures/core/08_lcr_lambda.png)

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

**版本说明：** 本文依据 `pyGWRx_diagnostics_visualization_complete` 源码生成。它描述的是当前已验证实现，而不是对任意同名软件的泛化说明。


---

# 第 5 章 地理加权广义线性模型（GWGLM）

> **pyGWRx 模型编号 05｜类别：非高斯局部回归**
> 本文同时说明原始方法与当前 pyGWRx 实现。凡实现与论文求解器不同之处均会明确指出。

计数响应的重要基础来源：[Nakaya et al. (2005), *Geographically weighted Poisson regression for disease association mapping*](https://doi.org/10.1002/sim.2129)。


## 1. 模型要解决的核心问题

空间统计中最危险的假设之一，是默认一组回归关系在整个研究区完全相同。若某个变量在城市中心、郊区、沿海和山区产生不同影响，一个全局系数只能给出平均效应，局部差异会被平均掉。地理加权方法的共同思想是：在每个目标位置建立一个局部窗口，让距离更近、时间更近、属性更相似或属于同一机制的观测获得更高权重，再估计该位置的局部统计量或局部模型。

需要强调：局部模型不是自动的因果模型。它首先是一种用于描述、探索和预测空间异质性的工具。带宽、核函数、局部共线性、异常值、残差空间相关和多重检验均会改变解释，必须与诊断一起使用。


## 2. 一句话思想

将 GWR 的空间局部化与 GLM 的链接函数和方差函数结合，使计数、比例和二元结果也可具有空间变化系数。

## 3. 数学模型

对位置 $s_i$，局部线性预测子为

$$
\eta_j(s_i)=x_j^\top\beta(s_i),\qquad
\mu_j(s_i)=g^{-1}(\eta_j(s_i)).
$$

局部对数似然由空间权重加权：

$$
\ell_i(\beta)=\sum_j w_{ij}\,\ell(y_j;\mu_j,\phi).
$$

IWLS 第 $t$ 步形成工作响应和工作权重

$$
z_j^{(t)}=\eta_j^{(t)}+(y_j-\mu_j^{(t)})\frac{d\eta}{d\mu},
$$

$$
\omega_{ij}^{(t)}=w_{ij}\left[\operatorname{Var}(Y_j)
\left(\frac{d\eta}{d\mu}\right)^2\right]^{-1},
$$

再做加权最小二乘更新。

## 4. 算法流程

1. 选择 Gaussian/Poisson/Binomial 家族及链接。
2. 选择空间核与带宽。
3. 在每个目标位置进行局部 IWLS。
4. 用离差、对数似然和 AICc 选择带宽并检查收敛。
5. 输出均值尺度预测、Pearson/Deviance 残差和局部系数。

## 5. pyGWRx 当前实现

```python
from pygwrx import GWGLM

model = GWGLM(family='gaussian', kernel='bisquare', bandwidth='cv', bandwidth_method='aicc', adaptive=False, max_iter=100, tol=1e-6, ...)
```

pyGWRx 支持 Gaussian identity、Poisson log 和 Binomial logit。非高斯模型使用局部 IWLS，提供曝光量/offset 语义、离差残差、预测概率或期望计数，并对不收敛和非法响应做严格检查。

### 5.1 输入语义

- `X`：自变量矩阵；启用 `fit_intercept=True` 时由模型统一添加截距。
- `y`：响应或类别，具体形状和分布要求由模型决定。
- `coords`：校准位置坐标；经纬度数据在使用欧氏距离前应投影，或选择模型支持的相应距离语义。
- 带宽：固定模式表示距离阈值，自适应模式通常表示近邻数量；两者不能混为同一单位。

### 5.2 结果与诊断

应优先检查：拟合值与残差、局部系数、带宽或尺度、有效参数个数、AICc/CV、标准误与显著性、局部共线性、影响度、残差空间结构。模型专用结果请结合下方图件和 `pygwrx.diagnostics` 使用。

## 6. 适用场景

事件计数、疾病发生数、二元分类概率或比例结果具有空间异质性时使用。

## 7. 关键局限与误用风险

局部稀有事件会导致分离或奇异；Poisson 过度离散需额外模型；局部样本必须覆盖响应类别；IWLS 的带宽搜索成本高；不可把概率地图直接解释为因果风险。

## 8. 推荐可视化

![03 gwglm residuals](../assets/figures/specialized/03_gwglm_residuals.png)

## 9. 最小工作流

```python
# 以下是接口结构示意；不同模型的 fit 参数可能包括 times、attributes 或阶段列表。
model = GWGLM(...)
model.fit(X, y, coords)

# 常见结果
# model.fitted_values_
# model.residuals_
# model.local_parameters_ / model.coef_
# model.diagnostics_
```

推荐工作顺序：全局模型 → 带宽/尺度选择 → 局部拟合 → 推断校正 → 局部共线性和影响诊断 → 空间分块验证 → 图件与结论。

## 10. 主要参考资料

- [Nakaya et al. (2005), *Geographically weighted Poisson regression for disease association mapping*](https://doi.org/10.1002/sim.2129)
- [Gollini et al. (2015), *GWmodel: an R Package for Exploring Spatial Heterogeneity*](https://doi.org/10.18637/jss.v063.i17)

---

**版本说明：** 本文依据 `pyGWRx_diagnostics_visualization_complete` 源码生成。它描述的是当前已验证实现，而不是对任意同名软件的泛化说明。


---

# 第 6 章 地理与时间加权回归（GTWR）

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

![18 gtwr slices](../assets/figures/specialized/18_gtwr_slices.png)

![19 gtwr trajectory](../assets/figures/specialized/19_gtwr_trajectory.png)

![20 gtwr residuals](../assets/figures/specialized/20_gtwr_residuals.png)

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

**版本说明：** 本文依据 `pyGWRx_diagnostics_visualization_complete` 源码生成。它描述的是当前已验证实现，而不是对任意同名软件的泛化说明。


---

# 第 7 章 多尺度地理与时间加权回归（MGTWR）

> **pyGWRx 模型编号 07｜类别：多尺度时空局部回归**
> 本文同时说明原始方法与当前 pyGWRx 实现。凡实现与论文求解器不同之处均会明确指出。

名称在文献中存在歧义。本项目的 MGTWR 指“每个系数拥有独立空间带宽和时空比例/时间尺度”的多尺度 GTWR，而不是部分文献中 global/local 混合系数的 mixed GTWR。


## 1. 模型要解决的核心问题

空间统计中最危险的假设之一，是默认一组回归关系在整个研究区完全相同。若某个变量在城市中心、郊区、沿海和山区产生不同影响，一个全局系数只能给出平均效应，局部差异会被平均掉。地理加权方法的共同思想是：在每个目标位置建立一个局部窗口，让距离更近、时间更近、属性更相似或属于同一机制的观测获得更高权重，再估计该位置的局部统计量或局部模型。

需要强调：局部模型不是自动的因果模型。它首先是一种用于描述、探索和预测空间异质性的工具。带宽、核函数、局部共线性、异常值、残差空间相关和多重检验均会改变解释，必须与诊断一起使用。


## 2. 一句话思想

将 MGWR 的“变量各有尺度”推广到时空域：不同驱动因素不仅空间作用范围不同，时间记忆长度也可以不同。

## 3. 数学模型

模型仍是加性变化系数形式

$$
y_i=\sum_{k=0}^{p}x_{ik}\beta_k(s_i,t_i)+\varepsilon_i,
$$

但第 $k$ 项使用独立时空权重

$$
w_{ij,k}=K\!\left(
\frac{\sqrt{(d_{ij}^{S})^2+\tau_k(d_{ij}^{T})^2}}{h_k}
\right),
$$

其中 $h_k$ 是第 $k$ 项的邻域尺度，$\tau_k$ 控制时间距离相对于空间距离的缩放。反向拟合使用部分残差逐项更新 $(h_k,\tau_k,\beta_k)$。

## 4. 算法流程

1. 初始化所有变量的空间带宽与时间尺度。
2. 构造每个变量的部分残差。
3. 对当前变量搜索 $(h_k,\tau_k)$。
4. 在该变量的时空权重下更新系数面。
5. 循环直到 RSS、系数和尺度稳定。
6. 组合各项平滑矩阵进行推断。

## 5. pyGWRx 当前实现

```python
from pygwrx import MGTWR

model = MGTWR(bandwidths=None, taus=None, kernel='bisquare', adaptive=True, bandwidth_method='aicc', tau_range=(0.0, 4.0), tol_multi=1e-5, max_iter=200, calculate_inference=True, ...)
```

pyGWRx 的 `MGTWR` 是 Gaussian 多尺度时空回归，支持手动或 AICc 选择每列 `bandwidths` 与 `taus`，反向拟合、可选完整推断和分块计算。

### 5.1 输入语义

- `X`：自变量矩阵；启用 `fit_intercept=True` 时由模型统一添加截距。
- `y`：响应或类别，具体形状和分布要求由模型决定。
- `coords`：校准位置坐标；经纬度数据在使用欧氏距离前应投影，或选择模型支持的相应距离语义。
- 带宽：固定模式表示距离阈值，自适应模式通常表示近邻数量；两者不能混为同一单位。

### 5.2 结果与诊断

应优先检查：拟合值与残差、局部系数、带宽或尺度、有效参数个数、AICc/CV、标准误与显著性、局部共线性、影响度、残差空间结构。模型专用结果请结合下方图件和 `pygwrx.diagnostics` 使用。

## 6. 适用场景

同一数据中存在短期局部效应、长期区域效应和近似全局趋势并存时使用。

## 7. 关键局限与误用风险

参数空间大、计算昂贵；时间尺度与带宽可能相互补偿；数据时间跨度不足时无法识别长时间尺度；必须明确本项目命名与其他 mixed-GTWR 文献的差异。

## 8. 推荐可视化

![21 mgtwr scales](../assets/figures/specialized/21_mgtwr_scales.png)

## 9. 最小工作流

```python
# 以下是接口结构示意；不同模型的 fit 参数可能包括 times、attributes 或阶段列表。
model = MGTWR(...)
model.fit(X, y, coords)

# 常见结果
# model.fitted_values_
# model.residuals_
# model.local_parameters_ / model.coef_
# model.diagnostics_
```

推荐工作顺序：全局模型 → 带宽/尺度选择 → 局部拟合 → 推断校正 → 局部共线性和影响诊断 → 空间分块验证 → 图件与结论。

## 10. 主要参考资料

- [Fotheringham, Yang & Kang (2017), *Multiscale Geographically Weighted Regression (MGWR)*](https://doi.org/10.1080/24694452.2017.1352480)
- [Huang, Wu & Barry (2010), *Geographically and temporally weighted regression for modeling spatio-temporal variation in house prices*](https://doi.org/10.1080/13658810802672469)

---

**版本说明：** 本文依据 `pyGWRx_diagnostics_visualization_complete` 源码生成。它描述的是当前已验证实现，而不是对任意同名软件的泛化说明。


---

# 第 8 章 时空加权回归（STWR）

> **pyGWRx 模型编号 08｜类别：变化率驱动时空回归**
> 本文同时说明原始方法与当前 pyGWRx 实现。凡实现与论文求解器不同之处均会明确指出。

正式来源：[Que, Ma, Ma & Chen (2020), *A spatiotemporal weighted regression model (STWR v1.0)*](https://doi.org/10.5194/gmd-13-6149-2020)。


## 1. 模型要解决的核心问题

空间统计中最危险的假设之一，是默认一组回归关系在整个研究区完全相同。若某个变量在城市中心、郊区、沿海和山区产生不同影响，一个全局系数只能给出平均效应，局部差异会被平均掉。地理加权方法的共同思想是：在每个目标位置建立一个局部窗口，让距离更近、时间更近、属性更相似或属于同一机制的观测获得更高权重，再估计该位置的局部统计量或局部模型。

需要强调：局部模型不是自动的因果模型。它首先是一种用于描述、探索和预测空间异质性的工具。带宽、核函数、局部共线性、异常值、残差空间相关和多重检验均会改变解释，必须与诊断一起使用。


## 2. 一句话思想

GTWR 主要依据经过了多久来衡量时间距离；STWR 进一步问“过程改变了多少”。过去时段的观测是否有用，不只由时间间隔决定，还由响应值的变化率决定。

## 3. 数学模型

对当前时段 $t$ 的焦点 $i$ 和过去第 $q$ 个时段的样本 $j$，响应变化率型时间距离可写为

$$
d_{ij}^{T}=
\frac{\Delta t_{\mathrm{all}}}{\Delta t_q}
\left|\frac{y_{j,t-q}-y_{i,t}}{y_{j,t-q}}\right|.
$$

时间作用通过 sigmoid/tanh 型映射进入权重，例如

$$
K_T(d^T)=2\sigma(d^T)-1
=\frac{2}{1+e^{-d^T}}-1.
$$

空间核与时间项按 $\alpha$ 组合。历史阶段的空间带宽按

$$
h_{t-q}=h_t-\tan(\theta)\,\Delta t_q
$$

演化，并受最小可识别邻域约束。模型用最近 `tick_nums` 个阶段为最新阶段标定局部系数。

## 4. 算法流程

1. 把数据按时间阶段组织为坐标、X、y 列表。
2. 决定使用多少历史阶段。
3. 计算当前点到各历史阶段的空间距离。
4. 根据当前与历史响应构造变化率时间距离。
5. 结合 $\alpha$、$\theta$ 和阶段带宽形成时空权重。
6. 用当前阶段为校准位置、历史阶段为信息源做局部 WLS。
7. 通过 CV/AICc 候选搜索选择参数。

## 5. pyGWRx 当前实现

```python
from pygwrx import STWR

model = STWR(spatial_bandwidth='cv', adaptive=True, kernel='bisquare', alpha=0.3, theta=0.0, tick_nums=None, bandwidth_candidates=None, alpha_candidates=None, theta_candidates=None, tick_candidates=None, ...)
```

pyGWRx 依据 Que 等 2020 的正式 STWR 和作者公开代码重建：保留阶段顺序、变化率时间距离、sigmoid 时间效应、历史带宽演化和最新阶段预测；实现为确定性的 NumPy/SciPy 版本。

### 5.1 输入语义

- `X`：自变量矩阵；启用 `fit_intercept=True` 时由模型统一添加截距。
- `y`：响应或类别，具体形状和分布要求由模型决定。
- `coords`：校准位置坐标；经纬度数据在使用欧氏距离前应投影，或选择模型支持的相应距离语义。
- 带宽：固定模式表示距离阈值，自适应模式通常表示近邻数量；两者不能混为同一单位。

### 5.2 结果与诊断

应优先检查：拟合值与残差、局部系数、带宽或尺度、有效参数个数、AICc/CV、标准误与显著性、局部共线性、影响度、残差空间结构。模型专用结果请结合下方图件和 `pygwrx.diagnostics` 使用。

## 6. 适用场景

过程的时间相似性主要体现为变化状态而不只是时间间隔，例如环境变量、城市变化和动态社会经济关系。

## 7. 关键局限与误用风险

时间距离使用响应值，因此预测真正未知未来时需估计参考响应；接近零的历史响应需要稳定分母；阶段划分影响结果；它不是简单的连续时间 GTWR。

## 8. 推荐可视化

![25 stwr weights](../assets/figures/specialized/25_stwr_weights.png)

## 9. 最小工作流

```python
# 以下是接口结构示意；不同模型的 fit 参数可能包括 times、attributes 或阶段列表。
model = STWR(...)
model.fit(X, y, coords)

# 常见结果
# model.fitted_values_
# model.residuals_
# model.local_parameters_ / model.coef_
# model.diagnostics_
```

推荐工作顺序：全局模型 → 带宽/尺度选择 → 局部拟合 → 推断校正 → 局部共线性和影响诊断 → 空间分块验证 → 图件与结论。

## 10. 主要参考资料

- [Que, Ma, Ma & Chen (2020), *A spatiotemporal weighted regression model (STWR v1.0)*](https://doi.org/10.5194/gmd-13-6149-2020)

---

**版本说明：** 本文依据 `pyGWRx_diagnostics_visualization_complete` 源码生成。它描述的是当前已验证实现，而不是对任意同名软件的泛化说明。


---

# 第 9 章 相似性地理与时间加权回归（SGTWR）

> **pyGWRx 模型编号 09｜类别：空间—时间—属性三邻近回归**
> 本文同时说明原始方法与当前 pyGWRx 实现。凡实现与论文求解器不同之处均会明确指出。

本项目对应的正式来源：[Li et al. (2025), *SGTWR Model with Spatial-Temporal Heterogeneity and Attribute Similarity*](https://doi.org/10.3390/su172310773)。


## 1. 模型要解决的核心问题

空间统计中最危险的假设之一，是默认一组回归关系在整个研究区完全相同。若某个变量在城市中心、郊区、沿海和山区产生不同影响，一个全局系数只能给出平均效应，局部差异会被平均掉。地理加权方法的共同思想是：在每个目标位置建立一个局部窗口，让距离更近、时间更近、属性更相似或属于同一机制的观测获得更高权重，再估计该位置的局部统计量或局部模型。

需要强调：局部模型不是自动的因果模型。它首先是一种用于描述、探索和预测空间异质性的工具。带宽、核函数、局部共线性、异常值、残差空间相关和多重检验均会改变解释，必须与诊断一起使用。


## 2. 一句话思想

SGTWR 同时接受三种“近”：地理位置近、时间近、属性结构像。它适合某些远距离城市在同一发展阶段表现相似，而同城不同年代又可能差异明显的情形。

## 3. 数学模型

空间—时间 Gaussian 权重为

$$
w_{ij}^{ST}=\exp\left[-\frac12\left(
\left(\frac{d_{ij}^{S}}{h_i^{S}}\right)^2+
\left(\frac{d_{ij}^{T}}{h^{T}}\right)^2
\right)\right].
$$

属性相似性权重为

$$
w_{ij}^{A}=\exp\left[-\left(
\frac1m\sum_k|z_{ik}-z_{jk}|
\right)^2\right].
$$

综合权重为

$$
w_{ij}=\alpha w_{ij}^{ST}+(1-\alpha)w_{ij}^{A}.
$$

本项目分别选择空间带宽、时间带宽和 $\alpha$，而不把空间与时间预先压缩成单一距离。

## 4. 算法流程

1. 统一时间尺度并标准化相似属性。
2. 生成空间带宽候选、时间带宽候选和 $\alpha$ 候选。
3. 计算独立的空间—时间 Gaussian 权重。
4. 计算属性相似性权重。
5. 组合后标定局部 WLS并计算 AICc。
6. 确定性搜索最优参数；可启用 causal 过滤。

## 5. pyGWRx 当前实现

```python
from pygwrx import SGTWR

model = SGTWR(spatial_bandwidth='aicc', temporal_bandwidth='aicc', adaptive=True, alpha='aicc', similarity_vars=None, standardize_similarity=True, causal=False, time_unit='auto', ...)
```

pyGWRx 按 2025 论文公式实现，但参数求解采用可复现的 AICc 候选搜索，而不是论文案例中的遗传算法；这样易测试、结果确定，但大候选网格会更慢。

### 5.1 输入语义

- `X`：自变量矩阵；启用 `fit_intercept=True` 时由模型统一添加截距。
- `y`：响应或类别，具体形状和分布要求由模型决定。
- `coords`：校准位置坐标；经纬度数据在使用欧氏距离前应投影，或选择模型支持的相应距离语义。
- 带宽：固定模式表示距离阈值，自适应模式通常表示近邻数量；两者不能混为同一单位。

### 5.2 结果与诊断

应优先检查：拟合值与残差、局部系数、带宽或尺度、有效参数个数、AICc/CV、标准误与显著性、局部共线性、影响度、残差空间结构。模型专用结果请结合下方图件和 `pygwrx.diagnostics` 使用。

## 6. 适用场景

长时间、多地区面板数据中，空间、时间和城市/区域属性相似性共同塑造局部关系时使用。

## 7. 关键局限与误用风险

论文较新，外部复现仍少；三类权重可能相互补偿；相似属性选择影响极大；计算和内存高于 GTWR/SGWR；因果预测必须避免未来样本。

## 8. 推荐可视化

![22 sgtwr scales](../assets/figures/specialized/22_sgtwr_scales.png)

![26 sgtwr weights](../assets/figures/specialized/26_sgtwr_weights.png)

## 9. 最小工作流

```python
# 以下是接口结构示意；不同模型的 fit 参数可能包括 times、attributes 或阶段列表。
model = SGTWR(...)
model.fit(X, y, coords)

# 常见结果
# model.fitted_values_
# model.residuals_
# model.local_parameters_ / model.coef_
# model.diagnostics_
```

推荐工作顺序：全局模型 → 带宽/尺度选择 → 局部拟合 → 推断校正 → 局部共线性和影响诊断 → 空间分块验证 → 图件与结论。

## 10. 主要参考资料

- [Li et al. (2025), *SGTWR Model with Spatial-Temporal Heterogeneity and Attribute Similarity*](https://doi.org/10.3390/su172310773)
- [Lessani & Li (2024), *SGWR: similarity and geographically weighted regression*](https://doi.org/10.1080/13658816.2024.2342319)
- [Huang, Wu & Barry (2010), *Geographically and temporally weighted regression for modeling spatio-temporal variation in house prices*](https://doi.org/10.1080/13658810802672469)

---

**版本说明：** 本文依据 `pyGWRx_diagnostics_visualization_complete` 源码生成。它描述的是当前已验证实现，而不是对任意同名软件的泛化说明。


---

# 第 10 章 相似性与地理加权回归（SGWR）

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

![23 sgwr weights](../assets/figures/specialized/23_sgwr_weights.png)

![24 sgwr profiles](../assets/figures/specialized/24_sgwr_profiles.png)

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

**版本说明：** 本文依据 `pyGWRx_diagnostics_visualization_complete` 源码生成。它描述的是当前已验证实现，而不是对任意同名软件的泛化说明。


---

# 第 11 章 可扩展地理加权回归（ScaGWR）

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
- 带宽：固定模式表示距离阈值，自适应模式通常表示近邻数量；两者不能混为同一单位。

### 5.2 结果与诊断

应优先检查：拟合值与残差、局部系数、带宽或尺度、有效参数个数、AICc/CV、标准误与显著性、局部共线性、影响度、残差空间结构。模型专用结果请结合下方图件和 `pygwrx.diagnostics` 使用。

## 6. 适用场景

样本达到数万、数十万甚至更大，标准 GWR 距离矩阵不可承受，但仍需局部线性解释时使用。

## 7. 关键局限与误用风险

它是对核权重结构的受限近似，不与任意经典 GWR 完全相同；近邻数和多项式阶数影响精度；坐标近邻查询仍受维度和数据密度影响。

## 8. 推荐可视化

![10 scalable kernel](../assets/figures/specialized/10_scalable_kernel.png)

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

**版本说明：** 本文依据 `pyGWRx_diagnostics_visualization_complete` 源码生成。它描述的是当前已验证实现，而不是对任意同名软件的泛化说明。


---

# 第 12 章 Bootstrap GWR 非平稳性检验

> **pyGWRx 模型编号 12｜类别：空间非平稳统计检验**
> 本文同时说明原始方法与当前 pyGWRx 实现。凡实现与论文求解器不同之处均会明确指出。

方法来源：[Harris et al. (2017), *Introducing bootstrap methods to investigate coefficient non-stationarity*](https://doi.org/10.1016/j.spasta.2017.07.006)。


## 1. 模型要解决的核心问题

空间统计中最危险的假设之一，是默认一组回归关系在整个研究区完全相同。若某个变量在城市中心、郊区、沿海和山区产生不同影响，一个全局系数只能给出平均效应，局部差异会被平均掉。地理加权方法的共同思想是：在每个目标位置建立一个局部窗口，让距离更近、时间更近、属性更相似或属于同一机制的观测获得更高权重，再估计该位置的局部统计量或局部模型。

需要强调：局部模型不是自动的因果模型。它首先是一种用于描述、探索和预测空间异质性的工具。带宽、核函数、局部共线性、异常值、残差空间相关和多重检验均会改变解释，必须与诊断一起使用。


## 2. 一句话思想

看到系数地图有起伏，不等于真实非平稳。BootstrapGWR 在“全局系数不变”的零假设下反复生成数据，判断观察到的局部系数波动是否大于随机噪声可解释的程度。

## 3. 数学模型

先拟合零假设 OLS：

$$
y=X\hat\beta_{OLS}+\varepsilon,
\qquad \hat\varepsilon\sim(0,\hat\sigma^2).
$$

第 $b$ 次参数 bootstrap 为

$$
y^{*(b)}=X\hat\beta_{OLS}+\varepsilon^{*(b)},
\qquad \varepsilon^{*(b)}\sim N(0,\hat\sigma^2).
$$

每个样本重新拟合 GWR。全局修正统计量可取局部 pseudo-$t$ 面的空间标准差：

$$
T_k=\operatorname{SD}_i\left(\frac{\hat\beta_{ik}}{\widehat{SE}_{ik}}\right).
$$

有限样本 plus-one p 值为

$$
p_k=\frac{1+\sum_b\mathbf1(T_k^{*(b)}\ge T_k)}{B+1}.
$$

## 4. 算法流程

1. 拟合 OLS 零模型与原始 GWR。
2. 计算观察统计量。
3. 从零模型生成 bootstrap 响应。
4. 每次可重新选择带宽并拟合 GWR。
5. 形成全局和局部分布，计算 plus-one p 值。
6. 报告 Monte Carlo 误差并进行多重检验校正。

## 5. pyGWRx 当前实现

```python
from pygwrx import BootstrapGWR

model = BootstrapGWR(bandwidth='aicc', adaptive=False, kernel='bisquare', n_bootstrap=99, reselect_bandwidth=True, pvalue_method='plus_one', ...)
```

pyGWRx 支持全局、局部或两类检验；OLS 参数零模型；带宽重选；双侧局部尾部；随机种子；可选保存局部 bootstrap 数组。

### 5.1 输入语义

- `X`：自变量矩阵；启用 `fit_intercept=True` 时由模型统一添加截距。
- `y`：响应或类别，具体形状和分布要求由模型决定。
- `coords`：校准位置坐标；经纬度数据在使用欧氏距离前应投影，或选择模型支持的相应距离语义。
- 带宽：固定模式表示距离阈值，自适应模式通常表示近邻数量；两者不能混为同一单位。

### 5.2 结果与诊断

应优先检查：拟合值与残差、局部系数、带宽或尺度、有效参数个数、AICc/CV、标准误与显著性、局部共线性、影响度、残差空间结构。模型专用结果请结合下方图件和 `pygwrx.diagnostics` 使用。

## 6. 适用场景

需要判断某个系数面是否真的空间变化，而不是仅展示地图时使用。

## 7. 关键局限与误用风险

计算量约为 $B$ 次完整 GWR；p 值精度受 $B$ 限制；零模型若遗漏空间误差会影响检验；局部检验仍需校正。

## 8. 推荐可视化

![08 bootstrap pvalues](../assets/figures/specialized/08_bootstrap_pvalues.png)

![09 bootstrap bandwidths](../assets/figures/specialized/09_bootstrap_bandwidths.png)

## 9. 最小工作流

```python
# 以下是接口结构示意；不同模型的 fit 参数可能包括 times、attributes 或阶段列表。
model = BootstrapGWR(...)
model.fit(X, y, coords)

# 常见结果
# model.fitted_values_
# model.residuals_
# model.local_parameters_ / model.coef_
# model.diagnostics_
```

推荐工作顺序：全局模型 → 带宽/尺度选择 → 局部拟合 → 推断校正 → 局部共线性和影响诊断 → 空间分块验证 → 图件与结论。

## 10. 主要参考资料

- [Harris et al. (2017), *Introducing bootstrap methods to investigate coefficient non-stationarity*](https://doi.org/10.1016/j.spasta.2017.07.006)
- [Brunsdon, Fotheringham & Charlton (1996), *Geographically Weighted Regression: A Method for Exploring Spatial Nonstationarity*](https://doi.org/10.1111/j.1538-4632.1996.tb00936.x)

---

**版本说明：** 本文依据 `pyGWRx_diagnostics_visualization_complete` 源码生成。它描述的是当前已验证实现，而不是对任意同名软件的泛化说明。


---

# 第 13 章 地理加权汇总统计（GWSS）

> **pyGWRx 模型编号 13｜类别：局部探索性统计**
> 本文同时说明原始方法与当前 pyGWRx 实现。凡实现与论文求解器不同之处均会明确指出。

正式来源：[Brunsdon, Fotheringham & Charlton (2002), *Geographically weighted summary statistics*](https://doi.org/10.1016/S0198-9715(01)00009-6)。


## 1. 模型要解决的核心问题

空间统计中最危险的假设之一，是默认一组回归关系在整个研究区完全相同。若某个变量在城市中心、郊区、沿海和山区产生不同影响，一个全局系数只能给出平均效应，局部差异会被平均掉。地理加权方法的共同思想是：在每个目标位置建立一个局部窗口，让距离更近、时间更近、属性更相似或属于同一机制的观测获得更高权重，再估计该位置的局部统计量或局部模型。

需要强调：局部模型不是自动的因果模型。它首先是一种用于描述、探索和预测空间异质性的工具。带宽、核函数、局部共线性、异常值、残差空间相关和多重检验均会改变解释，必须与诊断一起使用。


## 2. 一句话思想

在回归之前先回答更基础的问题：均值、方差、偏度、分位数和变量相关性是否随空间变化。GWSS 是局部探索分析，而不是预测模型。

## 3. 数学模型

位置 $i$ 的归一化权重为 $\tilde w_{ij}=w_{ij}/\sum_jw_{ij}$。局部均值：

$$
\bar x_i=\sum_j\tilde w_{ij}x_j.
$$

带有效样本量修正的局部协方差可写为

$$
\operatorname{Cov}_i(x,y)=
\frac{\sum_j\tilde w_{ij}(x_j-\bar x_i)(y_j-\bar y_i)}
{1-\sum_j\tilde w_{ij}^2}.
$$

局部相关为协方差除以局部标准差乘积。加权分位数通过按值排序并累计归一化权重获得。

## 4. 算法流程

1. 选择核与带宽。
2. 为每个位置归一化权重。
3. 计算局部位置、离散、形状与分位统计量。
4. 对变量对计算局部协方差/相关。
5. 绘制地图识别异质性、异常区域和后续模型需求。

## 5. pyGWRx 当前实现

```python
from pygwrx import GWSS

model = GWSS(kernel='bisquare', bandwidth=None, adaptive=False, quantile=False, verbose=False)
```

pyGWRx 支持局部均值、方差、标准差、偏度、分位数、协方差和相关；自适应带宽严格按近邻数解释；协方差采用有效权重修正。

### 5.1 输入语义

- `X`：自变量矩阵；启用 `fit_intercept=True` 时由模型统一添加截距。
- `y`：响应或类别，具体形状和分布要求由模型决定。
- `coords`：校准位置坐标；经纬度数据在使用欧氏距离前应投影，或选择模型支持的相应距离语义。
- 带宽：固定模式表示距离阈值，自适应模式通常表示近邻数量；两者不能混为同一单位。

### 5.2 结果与诊断

应优先检查：拟合值与残差、局部系数、带宽或尺度、有效参数个数、AICc/CV、标准误与显著性、局部共线性、影响度、残差空间结构。模型专用结果请结合下方图件和 `pygwrx.diagnostics` 使用。

## 6. 适用场景

模型构建前的空间 EDA、变量分布非平稳检查、局部相关探索。

## 7. 关键局限与误用风险

局部相关不是回归效应或因果；多张地图会放大偶然模式；边缘区域权重不对称；描述性统计不提供 GWR 的 AIC/帽子矩阵。

## 8. 推荐可视化

![11 gwss mean](../assets/figures/specialized/11_gwss_mean.png)

![12 gwss correlation](../assets/figures/specialized/12_gwss_correlation.png)

## 9. 最小工作流

```python
# 以下是接口结构示意；不同模型的 fit 参数可能包括 times、attributes 或阶段列表。
model = GWSS(...)
model.fit(X, y, coords)

# 常见结果
# model.fitted_values_
# model.residuals_
# model.local_parameters_ / model.coef_
# model.diagnostics_
```

推荐工作顺序：全局模型 → 带宽/尺度选择 → 局部拟合 → 推断校正 → 局部共线性和影响诊断 → 空间分块验证 → 图件与结论。

## 10. 主要参考资料

- [Brunsdon, Fotheringham & Charlton (2002), *Geographically weighted summary statistics*](https://doi.org/10.1016/S0198-9715(01)00009-6)

---

**版本说明：** 本文依据 `pyGWRx_diagnostics_visualization_complete` 源码生成。它描述的是当前已验证实现，而不是对任意同名软件的泛化说明。


---

# 第 14 章 地理加权主成分分析（GWPCA）

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

![13 gwpca variance](../assets/figures/specialized/13_gwpca_variance.png)

![14 gwpca loading](../assets/figures/specialized/14_gwpca_loading.png)

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

**版本说明：** 本文依据 `pyGWRx_diagnostics_visualization_complete` 源码生成。它描述的是当前已验证实现，而不是对任意同名软件的泛化说明。


---

# 第 15 章 地理加权判别分析（GWDA）

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

![15 gwda class](../assets/figures/specialized/15_gwda_class.png)

![16 gwda confidence](../assets/figures/specialized/16_gwda_confidence.png)

![17 gwda confusion](../assets/figures/specialized/17_gwda_confusion.png)

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

**版本说明：** 本文依据 `pyGWRx_diagnostics_visualization_complete` 源码生成。它描述的是当前已验证实现，而不是对任意同名软件的泛化说明。


---

# 第 16 章 混合地理加权回归（Mixed GWR）

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

![07 mixed coefficients](../assets/figures/specialized/07_mixed_coefficients.png)

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

**版本说明：** 本文依据 `pyGWRx_diagnostics_visualization_complete` 源码生成。它描述的是当前已验证实现，而不是对任意同名软件的泛化说明。


---

# 第 17 章 地理加权 Lasso（GWLasso）

> **pyGWRx 模型编号 17｜类别：局部稀疏回归**
> 本文同时说明原始方法与当前 pyGWRx 实现。凡实现与论文求解器不同之处均会明确指出。

正式来源：[Wheeler (2009), *The Geographically Weighted Lasso*](https://doi.org/10.1068/a40256)。


## 1. 模型要解决的核心问题

空间统计中最危险的假设之一，是默认一组回归关系在整个研究区完全相同。若某个变量在城市中心、郊区、沿海和山区产生不同影响，一个全局系数只能给出平均效应，局部差异会被平均掉。地理加权方法的共同思想是：在每个目标位置建立一个局部窗口，让距离更近、时间更近、属性更相似或属于同一机制的观测获得更高权重，再估计该位置的局部统计量或局部模型。

需要强调：局部模型不是自动的因果模型。它首先是一种用于描述、探索和预测空间异质性的工具。带宽、核函数、局部共线性、异常值、残差空间相关和多重检验均会改变解释，必须与诊断一起使用。


## 2. 一句话思想

不同地区可能不仅系数大小不同，真正起作用的变量集合也不同。GWLasso 在每个位置做带地理权重的 L1 正则化，使部分局部系数精确收缩到零。

## 3. 数学模型

位置 $s$ 的局部目标为

$$
\min_{\beta_0(s),\beta(s)}
\frac{1}{2\sum_iw_i(s)}
\sum_iw_i(s)\left[y_i-\beta_0(s)-x_i^\top\beta(s)\right]^2
+\lambda(s)\|\beta(s)\|_1.
$$

截距不惩罚。为让惩罚可比较，局部自变量通常按加权均值和加权尺度标准化，再将系数还原到原始量纲。KKT 条件决定某个局部系数是否为零。

## 4. 算法流程

1. 确定地理带宽。
2. 在每个目标位置计算空间权重。
3. 对局部 X、y 进行加权中心化和标准化。
4. 通过局部 CV 或给定 alpha 选择惩罚。
5. 坐标下降求解局部 Lasso。
6. 还原系数，绘制变量激活区域和选择频率。

## 5. pyGWRx 当前实现

```python
from pygwrx import GWLasso

model = GWLasso(kernel='exponential', bandwidth='cv', alpha='cv', alpha_grid=None, n_alphas=30, cv_folds=5, standardize=True, adaptive=False, ...)
```

pyGWRx 支持全局或局部 alpha、局部标准化、截距不惩罚、alpha 网格和交叉验证、变量重要性与活跃矩阵。

### 5.1 输入语义

- `X`：自变量矩阵；启用 `fit_intercept=True` 时由模型统一添加截距。
- `y`：响应或类别，具体形状和分布要求由模型决定。
- `coords`：校准位置坐标；经纬度数据在使用欧氏距离前应投影，或选择模型支持的相应距离语义。
- 带宽：固定模式表示距离阈值，自适应模式通常表示近邻数量；两者不能混为同一单位。

### 5.2 结果与诊断

应优先检查：拟合值与残差、局部系数、带宽或尺度、有效参数个数、AICc/CV、标准误与显著性、局部共线性、影响度、残差空间结构。模型专用结果请结合下方图件和 `pygwrx.diagnostics` 使用。

## 6. 适用场景

解释变量较多、局部共线性明显、需要识别“哪里哪些变量有效”时使用。

## 7. 关键局限与误用风险

Lasso 在高度相关变量中可能任意选择一个；局部选择结果可不稳定；alpha 与空间带宽共同决定稀疏度；常规标准误不适用于选择后的系数。

## 8. 推荐可视化

![04 gwlasso frequency](../assets/figures/specialized/04_gwlasso_frequency.png)

![05 gwlasso active](../assets/figures/specialized/05_gwlasso_active.png)

![06 gwlasso alpha](../assets/figures/specialized/06_gwlasso_alpha.png)

## 9. 最小工作流

```python
# 以下是接口结构示意；不同模型的 fit 参数可能包括 times、attributes 或阶段列表。
model = GWLasso(...)
model.fit(X, y, coords)

# 常见结果
# model.fitted_values_
# model.residuals_
# model.local_parameters_ / model.coef_
# model.diagnostics_
```

推荐工作顺序：全局模型 → 带宽/尺度选择 → 局部拟合 → 推断校正 → 局部共线性和影响诊断 → 空间分块验证 → 图件与结论。

## 10. 主要参考资料

- [Wheeler (2009), *The Geographically Weighted Lasso*](https://doi.org/10.1068/a40256)

---

**版本说明：** 本文依据 `pyGWRx_diagnostics_visualization_complete` 源码生成。它描述的是当前已验证实现，而不是对任意同名软件的泛化说明。


---

# 第 18 章 潜在几何地理加权回归（LG-GWR）

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

![27 lggwr latent](../assets/figures/specialized/27_lggwr_latent.png)

![28 lggwr metric](../assets/figures/specialized/28_lggwr_metric.png)

![29 lggwr training](../assets/figures/specialized/29_lggwr_training.png)

![30 lggwr neighbours](../assets/figures/specialized/30_lggwr_neighbours.png)

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

**版本说明：** 本文依据 `pyGWRx_diagnostics_visualization_complete` 源码生成。它描述的是当前已验证实现，而不是对任意同名软件的泛化说明。


---

# 第 19 章 地理机制分区加权回归（GR-GWR）

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

![31 grgwr regimes](../assets/figures/specialized/31_grgwr_regimes.png)

![32 grgwr convergence](../assets/figures/specialized/32_grgwr_convergence.png)

![33 grgwr sizes](../assets/figures/specialized/33_grgwr_sizes.png)

![34 grgwr coefficient](../assets/figures/specialized/34_grgwr_coefficient.png)

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

**版本说明：** 本文依据 `pyGWRx_diagnostics_visualization_complete` 源码生成。它描述的是当前已验证实现，而不是对任意同名软件的泛化说明。
