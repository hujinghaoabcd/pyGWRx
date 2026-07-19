# GeorgiaEduc — Georgia educational attainment / 佐治亚州教育程度

**EN** — County census data from the state of Georgia, USA, widely used to
demonstrate GWR (the response is the percentage of the population with a bachelor's
degree). A canonical dataset from Fotheringham, Brunsdon & Charlton (2002); also
found in GWR 3, spgwr, GWmodel and mgwr.

**中文** — 美国佐治亚州县级人口普查数据,GWR 的经典演示数据(因变量为拥有学士学位
的人口比例)。出自 Fotheringham、Brunsdon & Charlton(2002),也见于 GWR 3、spgwr、
GWmodel、mgwr 等。

- **Files / 文件**: `GeorgiaEduc.shp` (+ `.dbf/.shx/.prj/.xml`), a point/polygon shapefile.
- **Spatial unit / 空间单位**: County / 县
- **N**: 159 counties / 159 个县。

## Fields / 字段(主要)

| Field | Meaning (EN) | 含义(中文) |
|---|---|---|
| `X`, `Y` / `X_COORD`, `Y_COORD` | Projected X / Y coordinates | 投影 X / Y 坐标 |
| `Latitude`, `Longitud` | Latitude / longitude of centroid | 质心经纬度 |
| `PctBach` | % population with a bachelor's degree ★response | 拥有学士学位人口比例 ★因变量 |
| `PctRural` | % population defined as rural | 农村人口比例 |
| `PctPov` | % population below the poverty line | 贫困线以下人口比例 |
| `PctBlack` | % population who are black | 黑人人口比例 |
| `PctEld` | % population aged 65+ | 65 岁以上人口比例 |
| `PctFB` | % population born outside the US | 外国出生人口比例 |
| `TotPop90` | Total population (1990) | 1990 年总人口 |

## Model variables / 模型变量
- **Y**: `PctBach`
- **X**: `PctRural`, `PctPov`, `PctBlack`, `PctEld`, `PctFB`

## Source / 来源
Fotheringham, A. S., Brunsdon, C., & Charlton, M. E. (2002). *Geographically
Weighted Regression: The Analysis of Spatially Varying Relationships.* Chichester:
Wiley. (Dataset also distributed with the GWmodel R package.)


## Exact source, licence, processing, and integrity

- Licence: `GPL-2.0-or-later`
- Upstream snapshot: CRAN `GWmodel` `2.4-1`, published `2024-09-07`
- Upstream object: `Georgia` / Georgia counties example data
- Archived source: <https://cran.r-project.org/src/contrib/Archive/GWmodel/GWmodel_2.4-1.tar.gz>
- Evidence reviewed: `2026-07-19`
- pyGWRx processing: duplicate county keys were consolidated during preparation; projected centroids and coordinate fields were refreshed. The distributed shapefile contains the canonical 159 counties in `EPSG:32616`.
- Integrity: see `DATA_PROVENANCE.md` and `DATA_HASHES.sha256`.
