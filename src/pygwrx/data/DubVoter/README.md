# DubVoter — Greater Dublin voter turnout / 都柏林选民投票率

**EN** — Spatial variation in voter turnout and socio-demographic structure across
Electoral Divisions (ED) in Greater Dublin, Ireland. A canonical GWR benchmark
dataset (shipped with the GWmodel R package).

**中文** — 爱尔兰大都柏林地区各选区(Electoral Division）的选民投票率与社会人口
结构的空间差异。GWR 的经典基准数据集(随 GWmodel R 包分发)。

- **Files / 文件**: `Dub.voter.shp` (+ `.dbf/.shx/.prj/.xml`), a polygon shapefile.
- **Spatial unit / 空间单位**: Electoral Division (ED) / 选区
- **N**: 322
- **CRS**: EPSG:29902 (Irish grid). Coordinates use polygon centroids / 坐标取多边形质心。

## Fields / 字段

| Field | Meaning (EN) | 含义(中文) |
|---|---|---|
| `GenEl2004` | Turnout in the 2004 parliamentary election ★response | 2004 年议会选举投票率 ★因变量 |
| `DiffAdd` | Share of residents who moved in during the last year | 一年内迁入人口比例 |
| `LARent` | Share of public/social housing tenants | 公屋(社会住房)租户比例 |
| `SC1` | Share of high social class population | 高社会阶层人口比例 |
| `Unempl` | Unemployment rate | 失业人口比例 |
| `LowEduc` | Share of population with no formal education | 无正式教育人口比例 |
| `Age18_24` | Population aged 18–24 (%) | 18–24 岁人口比例 |
| `Age25_44` | Population aged 25–44 (%) | 25–44 岁人口比例 |
| `Age45_64` | Population aged 45–64 (%) | 45–64 岁人口比例 |

## Model variables / 模型变量
- **Y (response / 因变量)**: `GenEl2004`
- **X (predictors / 自变量)**: `DiffAdd`, `LARent`, `SC1`, `Unempl`, `LowEduc`, `Age18_24`, `Age25_44`, `Age45_64`

## Loading / 加载
```python
from pygwrx.io import load_dublin_voter
X, y, coords = load_dublin_voter(return_type="arrays")
```

## Source / 来源
Gollini, I., Lu, B., Charlton, M., Brunsdon, C., & Harris, P. (2015).
*GWmodel: An R Package for Exploring Spatial Heterogeneity Using Geographically
Weighted Models.* Journal of Statistical Software, 63(17).
https://doi.org/10.18637/jss.v063.i17


## Exact source, licence, processing, and integrity

- Licence: `GPL-2.0-or-later`
- Upstream snapshot: CRAN `GWmodel` `2.4-1`, published `2024-09-07`
- Upstream object: `DubVoter` / `Dub.voter`
- Archived source: <https://cran.r-project.org/src/contrib/Archive/GWmodel/GWmodel_2.4-1.tar.gz>
- Evidence reviewed: `2026-07-19`
- pyGWRx processing: retained the shapefile components and normalized CRS metadata to `EPSG:29902`.
- Integrity: see `DATA_PROVENANCE.md` and `DATA_HASHES.sha256`.
