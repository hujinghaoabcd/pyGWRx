# Data and input contracts

## Aligned rows

For row-wise models, row `i` in every input must describe the same observation.

| Argument | Shape | Typical types | Meaning |
|---|---:|---|---|
| `X` | `(n, p)` | NumPy array, pandas DataFrame | predictors or multivariate features |
| `y` | `(n,)` | array-like | continuous response, count, or class labels depending on model |
| `coords` | `(n, 2)` | array, DataFrame | numeric calibration coordinates |
| `times` | `(n,)` | numeric, datetime-like | row-wise time for GTWR/SGTWR/MGTWR |
| `attributes` | `(n, q)` | array, DataFrame | latent-geometry or contextual attributes |
| `exposure` | `(n,)` | positive numeric | Poisson exposure for GWGLM |

```python
assert len(X) == len(y) == len(coords)
```

The validation layer rejects NaN and infinite values. Handle missingness before fitting and document the strategy.

## DataFrames and feature names

DataFrames preserve feature names in summaries, result tables, diagnostics, and plotting calls.

```python
X = frame[["income", "access", "density"]]
y = frame["response"].to_numpy()
coords = frame[["east", "north"]]
```

Avoid duplicate column names. Preserve a stable observation ID outside `X` for joining outputs.

## Intercepts

Most regression estimators add an intercept when `fit_intercept=True`. Do not manually add a constant column unless the model documentation explicitly requires it or you disable automatic intercept handling.

## Coordinate systems

For Euclidean modelling, use projected coordinates with meaningful distance units. Longitude/latitude degrees are not metres.

```python
# GeoPandas example
projected = gdf.to_crs(gdf.estimate_utm_crs())
coords = projected.geometry.get_coordinates()[["x", "y"]]
```

For polygons, define a representative-point rule such as centroid, point-on-surface, or a population-weighted centroid. The choice becomes part of the spatial model.

## Spatiotemporal row-wise models

GTWR, SGTWR, and MGTWR use one time value per row:

```python
model.fit(X, y, coords, times)
```

Repeated coordinates across times are allowed when they represent repeated measurements. Sort order is not a substitute for an explicit `times` vector.

Document:

- numeric versus datetime input;
- time unit or conversion;
- irregular intervals;
- repeated locations;
- causal versus non-causal weighting;
- prediction-time availability of all features.

## STWR stage lists

STWR uses ordered snapshots:

```python
model.fit(X_list, y_list, coords_list, intervals)
```

Each list element is a stage. Validate the row count and alignment inside every stage. `intervals` describe the temporal relation between stages according to the STWR API; they are not interchangeable with a row-wise time vector.

## Classification and generalized responses

- `GWDA.fit(X, labels, coords)` accepts class labels.
- `GWGLM(family="binomial")` requires valid binary/binomial response semantics.
- `GWGLM(family="poisson")` requires non-negative count responses; exposure must be positive.
- `GWGLM(family="gaussian")` behaves as a local Gaussian identity-link model.

## Multivariate methods

- `GWPCA.fit(X, coords)` has no response.
- `GWSS.fit(X, coords)` has no response.
- Scaling is usually important because covariance and penalization depend on variable units.

## Similarity and latent attributes

Similarity variables and LGGWR attributes must be available at prediction time. Do not include the response, future information, or variables derived from the held-out target outcome.

## Optional geospatial integration

GeoPandas and Shapely are included in the standard installation. Install `.[parquet]` only when Parquet or GeoParquet persistence is required.

Use:

```python
from pygwrx.core import extract_geopandas_coords
from pygwrx.io import to_geodataframe, from_geodataframe, save_results
```

See [Geospatial I/O](../guides/geospatial-io.md) and the [I/O API](../api/io/index.md).
