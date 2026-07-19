# Housing — Neighborhood house prices / 社区房价

**EN** — Neighborhood-level variation in housing prices and structural property
characteristics (King County / Seattle area), for spatial analysis and regression
modelling. Response is z-scored in the SGWR benchmark. Because n is large and pure
Python GWR is O(n²), this dataset is usually subsampled.

**中文** — 社区尺度的房价与住房结构特征的空间分布(西雅图 King County 一带),用于
空间统计与回归建模。SGWR 基准中因变量做了 z-score 标准化。因样本量大、纯 Python
GWR 为 O(n²),通常需子采样使用。

- **File / 文件**: `Housing.csv`
- **Spatial unit / 空间单位**: Neighborhood / 社区
- **N**: ≈20,833

## Fields / 字段

| Field | Meaning (EN) | 含义(中文) |
|---|---|---|
| `x_coor`, `y_coor` | X / Y coordinates | X / Y 坐标 |
| `price` | Housing price ★response | 房价 ★因变量 |
| `bedrooms` | Number of bedrooms | 卧室数量 |
| `bathrooms` | Number of bathrooms | 卫生间数量 |
| `sqft_lot` | Lot size | 占地面积 |
| `grade` | Housing quality grade | 建筑质量等级 |
| `sqft_living15` | Average living area (15 nearest neighbors) | 周边(15 邻域)居住面积均值 |
| `sqft_lot15` | Average lot size (15 nearest neighbors) | 周边(15 邻域)地块面积均值 |

## Model variables / 模型变量
- **Y**: `price`
- **X**: `bedrooms`, `bathrooms`, `sqft_lot`, `grade`, `sqft_living15`, `sqft_lot15`

> ⚠️ Use **bisquare + adaptive**; a Gaussian kernel is unstable on this dataset
> (optimal bandwidth collapses). / 建议用 **bisquare + 自适应**;高斯核在本数据上不稳定。

## Source / 来源
Lessani, M. N., & Li, Z. (2025). *Enhancing the computational efficiency of the
SGWR model and introducing its software implementation.* Annals of GIS, 31(4),
635–650. https://doi.org/10.1080/19475683.2025.2523739


## Exact source, licence, processing, and integrity

- Licence: `MIT`
- Upstream snapshot: FastSGWR commit `b63064938a2ba5a1eb27cc7bdb642eaa62cb5de6`
- Upstream path: `Data/Housing.csv`
- Upstream/local Git blob SHA-1: `35f4a3e7f8fea05d8f34a0c2bd03312afe74559e`
- Evidence reviewed: `2026-07-19`
- pyGWRx processing: none; local bytes match the pinned upstream Git blob.
- Integrity: see `DATA_PROVENANCE.md` and `DATA_HASHES.sha256`.
