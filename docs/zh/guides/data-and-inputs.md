# 数据、坐标与输入规范

pyGWRx 的模型共享一套基本数据契约，但不同模型对响应分布、时间、属性相似度和阶段结构有额外要求。输入正确比调参更重要；很多所谓“模型不稳定”其实来自坐标单位、索引错位或数据泄漏。

## 1. 基础输入

| 输入 | 常见形状 | 含义 |
|---|---:|---|
| `X` | `(n, p)` | 自变量；可使用 NumPy 数组或 pandas DataFrame |
| `y` | `(n,)` | 连续响应、计数、二元响应或类别标签 |
| `coords` | `(n, 2)` | 每个观测的二维位置，行顺序必须与 `X`、`y` 完全一致 |
| `times` | `(n,)` | GTWR、MGTWR、SGTWR 等模型的时间坐标 |
| `attributes` | `(n, q)` | SGWR/SGTWR 的非地理相似性变量 |

DataFrame 输入会保留变量名，推荐用于正式分析。不要在 `X` 中手工添加常数列后又启用 `fit_intercept=True`。

## 2. 坐标与 CRS

经纬度是角度，不是米。若使用欧氏距离，先把 GeoDataFrame 投影到适合研究区的投影坐标系：

```python
import geopandas as gpd

projected = gdf.to_crs("EPSG:32650")
coords = projected.geometry.get_coordinates().to_numpy()
```

跨大范围研究可采用模型支持的球面距离语义。报告中必须说明 CRS、距离单位和是否使用固定或自适应带宽。

## 3. 索引与缺失值

在传给模型前，应一次性构造共同有效掩码，避免分别删除 `X`、`y`、坐标中的缺失值造成行错位：

```python
columns = ["y", "x1", "x2", "geometry"]
analysis = gdf[columns].dropna().copy()
X = analysis[["x1", "x2"]]
y = analysis["y"]
coords = analysis.geometry.get_coordinates().to_numpy()
```

不要在拟合后重新排序结果表而不保留原索引。地图连接应使用稳定的观测 ID。

## 4. 不同模型的响应要求

- GWR、MGWR、RGWR、LCRGWR、MixedGWR 等：连续响应。
- GWGLM：Gaussian、Poisson 或 Binomial；Poisson 可提供 exposure/offset。
- GWDA：类别标签，需保证每个局部邻域包含足够类别样本。
- GWPCA、GWSS：主要输入是多变量特征，任务不是回归预测。
- BootstrapGWR：连续响应，用于非平稳性推断而不是常规预测。

## 5. 时空模型

时间变量必须具有清楚单位。GTWR/MGTWR/SGTWR 使用点级时间；STWR 使用多阶段列表与阶段间隔。时间标准化会改变空间—时间权衡，因此应报告原始单位、缩放方式和是否采用因果历史窗口。

## 6. 训练与验证

随机 K 折通常会把相邻样本同时放入训练和验证集，夸大性能。推荐空间分块、缓冲留出或区域外推验证；时空模型还应采用时间滚动或过去预测未来的划分。

## 7. 输入检查清单

- 行数和行顺序完全一致；
- 坐标单位与距离语义一致；
- 自变量没有重复常数列；
- 缺失值和无穷值已统一处理；
- 类别/计数响应满足模型假设；
- 带宽搜索范围不会产生局部样本不足；
- 验证划分符合真实使用场景。
