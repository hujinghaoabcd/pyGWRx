# Crime — County-level crime rate / 县级犯罪率

**EN** — County-level crime rates and related socioeconomic characteristics in the
USA, for spatial analysis and regression modelling. Response is z-scored in the
SGWR benchmark.

**中文** — 美国县级犯罪率及相关社会经济特征的空间分布,用于空间统计与回归建模。
在 SGWR 基准中因变量做了 z-score 标准化。

- **File / 文件**: `Crime.csv`
- **Spatial unit / 空间单位**: County / 县
- **N**: 2841

## Fields / 字段

| Field | Meaning (EN) | 含义(中文) |
|---|---|---|
| `X`, `Y` | X / Y coordinates | X / Y 坐标 |
| `Five_ave_crime` | Five-year average crime rate ★response | 五年平均犯罪率 ★因变量 |
| `PopulationDensity` | Population density | 人口密度 |
| `PopulationFemale` | Female population rate | 女性人口比例 |
| `X.Black` | Black population rate | 黑人人口比例 |
| `Neighbor_Disadvantage` | Neighborhood disadvantage index | 社区弱势指数 |
| `Casinos` | Casino visits (POI-based) | 赌场访问量(基于 POI 数据) |

## Model variables / 模型变量
- **Y**: `Five_ave_crime`
- **X**: `PopulationDensity`, `PopulationFemale`, `X.Black`, `Neighbor_Disadvantage`, `Casinos`

## Source / 来源
Lessani, M. N., & Li, Z. (2025). *Enhancing the computational efficiency of the
SGWR model and introducing its software implementation.* Annals of GIS, 31(4),
635–650. https://doi.org/10.1080/19475683.2025.2523739


## Exact source, licence, processing, and integrity

- Licence: `MIT`
- Upstream snapshot: FastSGWR commit `b63064938a2ba5a1eb27cc7bdb642eaa62cb5de6`
- Upstream path: `Data/Crime.csv`
- Upstream/local Git blob SHA-1: `ac8ac10e020232a5293e7984c9e90ac440f91414`
- Evidence reviewed: `2026-07-19`
- pyGWRx processing: none; local bytes match the pinned upstream Git blob.
- Integrity: see `DATA_PROVENANCE.md` and `DATA_HASHES.sha256`.
