# HIV — County-level HIV prevalence / 县级 HIV 发病率

**EN** — County-level spatial variation in HIV prevalence in the USA, for spatial
analysis and regression modelling. Response is z-scored in the SGWR benchmark.

**中文** — 美国县级 HIV 发病率的空间分布,用于空间统计与回归建模。在 SGWR 基准中
因变量做了 z-score 标准化。

- **File / 文件**: `HIV.csv`
- **Spatial unit / 空间单位**: County / 县
- **N**: 2526

## Fields / 字段

| Field | Meaning (EN) | 含义(中文) |
|---|---|---|
| `X`, `Y` | X / Y coordinates | X / Y 坐标 |
| `Rate_per_100000` | HIV rate per 100,000 ★response | HIV 发病率(每 10 万人)★因变量 |
| `Walkability` | Walkability score | 步行便利度指数 |
| `PopulationFemale` | Female population rate | 女性人口比例 |
| `NoHealthInsurance` | Uninsured population rate | 未投保人口比例 |
| `X.Black` | Black population rate | 黑人人口比例 |

## Model variables / 模型变量
- **Y**: `Rate_per_100000`
- **X**: `Walkability`, `PopulationFemale`, `NoHealthInsurance`, `X.Black`

## Source / 来源
Lessani, M. N., & Li, Z. (2025). *Enhancing the computational efficiency of the
SGWR model and introducing its software implementation.* Annals of GIS, 31(4),
635–650. https://doi.org/10.1080/19475683.2025.2523739


## Exact source, licence, processing, and integrity

- Licence: `MIT`
- Upstream snapshot: FastSGWR commit `b63064938a2ba5a1eb27cc7bdb642eaa62cb5de6`
- Upstream path: `Data/HIV.csv`
- Upstream/local Git blob SHA-1: `cbe28a992be30dab5f7913f277d87672d5865d13`
- Evidence reviewed: `2026-07-19`
- pyGWRx processing: none; local bytes match the pinned upstream Git blob.
- Integrity: see `DATA_PROVENANCE.md` and `DATA_HASHES.sha256`.
