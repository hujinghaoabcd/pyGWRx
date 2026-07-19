# EWHP — England & Wales house prices / 英格兰与威尔士房价

**EN** — A house-price dataset for England and Wales from 2001 with 9 hedonic
(explanatory) variables. Used to demonstrate GWR spatial prediction. Bundled with
the GWmodel R package. `EWOutline.csv` is the accompanying England & Wales boundary
outline for mapping.

**中文** — 2001 年英格兰与威尔士的房价数据,含 9 个特征(享乐)变量,用于演示 GWR
空间预测。随 GWmodel R 包分发。`EWOutline.csv` 为配套的英格兰与威尔士边界轮廓,用于
制图。

- **Files / 文件**: `EWHP.csv`(data / 数据), `EWOutline.csv`(boundary / 边界)
- **N**: 519

## Fields / 字段

| Field | Meaning (EN) | 含义(中文) |
|---|---|---|
| `Easting`, `Northing` | X / Y coordinates | X / Y 坐标 |
| `PurPrice` | Purchase price of the property ★response | 房产成交价 ★因变量 |
| `BldIntWr` | 1 if built during the world war, else 0 | 战时建造为 1,否则 0 |
| `BldPostW` | 1 if built after the world war, else 0 | 战后建造为 1,否则 0 |
| `Bld60s` | 1 if built 1960–1969, else 0 | 1960–1969 年建造为 1 |
| `Bld70s` | 1 if built 1970–1979, else 0 | 1970–1979 年建造为 1 |
| `Bld80s` | 1 if built 1980–1989, else 0 | 1980–1989 年建造为 1 |
| `TypDetch` | 1 if detached house, else 0 | 独栋住宅为 1 |
| `TypSemiD` | 1 if semi-detached, else 0 | 半独栋为 1 |
| `TypFlat` | 1 if a flat / apartment, else 0 | 公寓为 1 |
| `FlrArea` | Floor area in square metres | 建筑面积(平方米) |

## Model variables / 模型变量
- **Y**: `PurPrice`
- **X**: the 9 hedonic dummies + `FlrArea` / 9 个享乐虚拟变量 + `FlrArea`

## Source / 来源
Fotheringham, A. S., Brunsdon, C., & Charlton, M. E. (2002). *Geographically
Weighted Regression: The Analysis of Spatially Varying Relationships.* Chichester:
Wiley. (Dataset distributed with the GWmodel R package; author: Binbin Lu.)


## Exact source, licence, processing, and integrity

- Licence: `GPL-2.0-or-later`
- Upstream snapshot: CRAN `GWmodel` `2.4-1`, published `2024-09-07`
- Upstream objects: `EWHP` and `EWOutline`
- Archived source: <https://cran.r-project.org/src/contrib/Archive/GWmodel/GWmodel_2.4-1.tar.gz>
- Evidence reviewed: `2026-07-19`
- pyGWRx processing: retained modelling values and the mapping outline in CSV form.
- Integrity: see `DATA_PROVENANCE.md` and `DATA_HASHES.sha256`.
