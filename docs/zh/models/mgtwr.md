# 多尺度地理与时间加权回归（`MGTWR`）

> **类别：多尺度时空局部回归**  
> 当前实现完全位于 pyGWRx 内部，不依赖任何 MGTWR 专用外部包。

## 1. 核心思想

GTWR 为全部系数使用同一个时空邻域，而 MGTWR 允许每个系数拥有独立的空间带宽和时间尺度。这样，短期局部效应、长期区域效应以及近似全局趋势可以同时存在于一个模型中。

## 2. 数学模型

模型为

$$
y_i=\sum_{k=0}^{p}x_{ik}\beta_k(s_i,t_i)+\varepsilon_i,
$$

第 $k$ 个系数使用

$$
w_{ij,k}=K\!\left(
\frac{\sqrt{(d_{ij}^{S})^2+\tau_k(d_{ij}^{T})^2}}{h_k}
\right),
$$

其中 $h_k$ 是空间带宽，$\tau_k\geq0$ 控制时间距离的贡献。当 $\tau_k=0$ 时，该系数退化为空间多尺度项。

## 3. pyGWRx 求解流程

1. 使用一个公共的 GTWR 尺度初始化各加性项。
2. 对当前变量构造部分残差。
3. 搜索或应用该变量的 $(h_k,\tau_k)$。
4. 使用项目统一的核函数和加权最小二乘更新局部系数。
5. 逐变量循环，直到变化分数低于 `tol_multi`，或达到 `max_iter`。
6. `calculate_inference=True` 时，沿最终反向拟合历史传播平滑算子，计算有效参数量、影响度、标准误、t 值和信息准则。

`n_chunks` 只控制精确推断的内存分块，不改变拟合系数。

## 4. 构造函数

```python
MGTWR(
    bandwidths=None,
    taus=None,
    *,
    kernel="bisquare",
    adaptive=True,
    fit_intercept=True,
    bandwidth_method="aicc",
    bandwidth_range=None,
    tau_range=(0.0, 4.0),
    init_bandwidth=None,
    init_tau=None,
    tol=1e-6,
    tol_multi=1e-5,
    max_iter=200,
    rss_score=False,
    calculate_inference=True,
    n_chunks=1,
    verbose=False,
)
```

`bandwidths` 与 `taus` 必须同时提供或同时省略。标量会扩展到所有拟合参数；序列长度必须等于拟合参数个数，启用截距时包括截距。自适应带宽表示整数近邻数。自动选择在 `bandwidth_range` 和 `tau_range` 内依据 AIC、AICc、BIC 或 CV 执行确定性的“粗网格 + 局部细化”候选搜索；它会检查范围边界，但不宣称穷举证明全局最优。

## 5. 数值验证

`tests/reference_data/mgtwr_fixed_gaussian_reference.json` 保存了一次性独立实现生成的固定尺度基准。项目测试在不导入、不安装任何 MGTWR 专用外部包的情况下，逐项核对局部系数、拟合值、残差、逐变量 ENP、标准误、t 值、信息准则和反向拟合迭代次数。更广泛的一次性对比矩阵中，局部系数最大绝对差为 `4.17e-8`，拟合值最大绝对差为 `7.29e-8`。

## 6. 尺度与性能注意事项

空间距离和时间距离直接组合，因此 `tau` 依赖坐标与时间的量纲。论文或报告必须同时说明坐标单位、时间单位和 `tau_range`。自动选择使用有界候选网格与局部细化，尺度落在边界时应专门检查。精确平滑矩阵推断明显比仅拟合系数更昂贵；`n_chunks` 只降低峰值内存，不会把校准过程并行化。

## 7. 完整可运行示例

```python
from pygwrx import MGTWR
from _common import print_model_result, temporal_regression

X, y, coords, times = temporal_regression(n=20, p=2)
model = MGTWR(
    bandwidths=[12, 12, 12],
    taus=[1.0, 1.0, 1.0],
    adaptive=True,
    calculate_inference=False,
).fit(X, y, coords, times)

print_model_result(model)
print("空间带宽：", model.bandwidths_)
print("时间尺度：", model.taus_)
```

项目根目录运行：

```bash
python examples/models/17_mgtwr.py
```

## 8. 主要结果

- `bandwidths_`、`taus_`、`temporal_bandwidths_`；
- `bandwidth_history_`、`tau_history_`、`convergence_history_`；
- `params_`、`intercept_`、`coef_`、`fitted_values_`、`residuals_`；
- 开启推断时的逐变量有效参数量、标准误、t 值、AIC、AICc 和 BIC。

`summary()` 返回字符表格，`to_frame()` 返回校准位置结果表，并保留每行时间值。

## 9. 验证与解释

必须检查：

- 是否收敛以及实际迭代次数；
- 带宽或 tau 是否落在搜索边界；
- 时间单位和坐标单位改变后结果是否稳定；
- 局部共线性和异常影响点；
- 空间与时间分块验证结果。

`predict()` 不支持独立新位置，会明确抛出 `NotImplementedError`。这属于当前模型能力边界，不会静默回退到其他算法。校准位置使用 `fitted_values_`。

## 10. 论文或报告中应说明

- 坐标系、时间编码及单位；
- 核函数和固定/自适应带宽语义；
- 每个参数的带宽与 tau；
- 初始尺度、搜索范围、停止规则、迭代次数和收敛状态；
- 是否开启精确推断及 `n_chunks`；
- 验证方式与仅支持校准位置结果的边界。

## 11. 参考文献

- Wu, C., Ren, F., Hu, W., & Du, Q. (2019). Multiscale geographically and temporally weighted regression: exploring the spatiotemporal determinants of housing prices. *International Journal of Geographical Information Science*, 33(3), 489–511. https://doi.org/10.1080/13658816.2018.1545158
- Fotheringham, A. S., Yang, W., & Kang, W. (2017). Multiscale geographically weighted regression (MGWR). *Annals of the American Association of Geographers*, 107(6), 1247–1265. https://doi.org/10.1080/24694452.2017.1352480

## 12. 相关文档

- [英文模型指南](../../models/mgtwr.md)
- [MGTWR API](../../api/models/mgtwr.md)
- [核函数与带宽](../../guides/kernels-and-bandwidths.md)
