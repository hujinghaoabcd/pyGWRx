# Columbus — Columbus (OH) neighborhood crime / 哥伦布市社区犯罪

**EN** — The classic Columbus, Ohio neighborhood crime dataset from Anselin (1988),
a standard spatial-econometrics benchmark (bundled with GeoDa, PySAL/libpysal,
spData). 49 neighborhoods.

**中文** — Anselin(1988)的经典哥伦布市(俄亥俄州)社区犯罪数据,空间计量的标准
基准数据集(随 GeoDa、PySAL/libpysal、spData 分发),共 49 个社区。

- **File / 文件**: `columbus.csv`
- **Spatial unit / 空间单位**: Neighborhood / 社区
- **N**: 49

## Fields / 字段(常用)

| Field | Meaning (EN) | 含义(中文) |
|---|---|---|
| `X`, `Y` | Centroid coordinates | 质心坐标 |
| `CRIME` | Residential burglaries & vehicle thefts per 1,000 households ★response | 每千户住宅入室盗窃与车辆盗窃数 ★因变量 |
| `INC` | Household income (in $1,000 USD) | 家庭收入(千美元) |
| `HOVAL` | Housing value (in $1,000 USD) | 住房价值(千美元) |
| `OPEN` | Open space in the neighborhood | 社区开放空间 |
| `PLUMB` | % housing units without plumbing | 无管道设施住房比例 |
| `DISCBD` | Distance to the central business district (CBD) | 到中央商务区(CBD)的距离 |

## Model variables / 模型变量(常用设定)
- **Y**: `CRIME`
- **X**: `INC`, `HOVAL` (other covariates available: `OPEN`, `PLUMB`, `DISCBD` …)

## Source / 来源
Anselin, L. (1988). *Spatial Econometrics: Methods and Models.* Dordrecht: Kluwer
Academic Publishers. (Distributed with GeoDa / PySAL / spData.)


## Exact source, licence, processing, and integrity

- Licence: `CC0-1.0`
- Upstream snapshot: CRAN `spData` `2.3.5`, published `2026-05-04`
- Upstream object: `columbus` (`data/columbus.rda`; spatial form also documented as `inst/shapes/columbus.gpkg`)
- Upstream package page: <https://cran.r-project.org/package=spData>
- Evidence reviewed: `2026-07-19`
- pyGWRx processing: retained a regression-ready tabular extract; R row names, polygon geometry, and neighbour objects are not bundled in this CSV.
- Integrity: see `DATA_PROVENANCE.md` and `DATA_HASHES.sha256`.
