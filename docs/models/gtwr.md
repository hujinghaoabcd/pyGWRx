# Geographically and Temporally Weighted Regression (`GTWR`)

<div class="model-hero" markdown>

**Task:** continuous-response local regression with one observation-level time coordinate per row  
**Core mechanism:** define one combined space-time distance, then fit local weighted regressions  
**Required inputs:** `X`, `y`, spatial `coords`, and row-wise `times`  
**Independent-target prediction:** supported at new coordinates and times

</div>

[API reference](../api/models/gtwr.md){ .md-button .md-button--primary }
[STWR manual](stwr.md){ .md-button }
[MGTWR manual](mgtwr.md){ .md-button }

## What GTWR is for

GTWR extends GWR by allowing proximity to depend jointly on spatial separation and temporal separation. Each row has its own coordinate and time. A local regression at target $(s_i,t_i)$ uses observations that are near under the selected combined distance.

pyGWRx provides two distance formulations.

### GWmodel-compatible distance

With `distance_combination="gwmodel"`,

$$
d_{st}=\lambda d_s+(1-\lambda)d_t
+2\sqrt{\lambda(1-\lambda)d_s d_t}\cos(\xi).
$$

- `lambda_st=1` produces the spatial component;
- `lambda_st=0` produces the temporal component;
- intermediate values combine both;
- `ksi` controls the interaction term through its angle in radians.

This formulation follows `GWmodel::st.dist`. Spatial and temporal units directly affect the result, so coordinate scale and time scale must be documented.

### Euclidean space-time distance

With `distance_combination="euclidean"`,

$$
d_{st}=\sqrt{d_s^2+\tau d_t^2}.
$$

`tau` determines the contribution of temporal distance. Larger `tau` makes a given temporal difference more distant. `lambda_st` and `ksi` are not the scientific scale controls for this branch.

## GTWR, STWR, and MGTWR are different models

| Model | Data organisation | Time mechanism | Number of scales | Target prediction |
|---|---|---|---|---|
| GTWR | One row per observation with one time value | Combined spatial and absolute temporal distance | One shared bandwidth; one balance/temporal-scale setting | Supported |
| [`STWR`](stwr.md) | Ordered lists of stage-specific datasets | Stage intervals plus response-value variation rate | One current spatial bandwidth, `alpha`, `theta`, and stage count | Latest-stage only |
| [`MGTWR`](mgtwr.md) | One row per observation with numeric time | Euclidean space-time distance per coefficient | One spatial bandwidth and one `tau` per fitted parameter | Not exposed |

Use GTWR when row-wise time and one common space-time neighbourhood are appropriate. Do not reshape arbitrary stage data into GTWR merely because the class accepts a time column.

## Causal versus symmetric time

The published/default GWmodel distance uses absolute temporal differences. Consequently, with `causal=False`, an observation later than a calibration or prediction time can contribute if it is close in absolute time.

Set `causal=True` for history-only forecasting. Future training observations relative to each target are assigned an extremely large temporal distance and effectively excluded.

!!! warning "`causal=False` can leak future information"
    The default is retained for compatibility with standard retrospective GTWR. It is not appropriate for forecasting claims unless future observations are already excluded by the training design.

## Time input

`times` may be:

- finite numeric values;
- pandas/Python/NumPy datetime-like values.

For numeric time, pyGWRx does **not** rescale values. A difference of 1 means whatever unit the user defined.

For datetime-like time, `time_unit` controls conversion to elapsed numeric values relative to a fitted origin:

```text
seconds, minutes, hours, days, weeks, or auto
```

`auto` chooses a stable unit from the training span. Prediction datetimes are converted using the fitted origin and fitted unit. Do not mix numeric training times with datetime prediction times.

## Installation

```bash
pip install pygwrx
```

## Self-contained example

```python
import numpy as np
import pandas as pd

from pygwrx import GTWR

rng = np.random.default_rng(99)
n = 90
coords = rng.uniform(0.0, 100.0, size=(n, 2))
times = pd.date_range("2024-01-01", periods=n, freq="12h")
time_index = np.arange(n, dtype=float) / 2.0

X = pd.DataFrame(
    {
        "access": rng.normal(size=n),
        "density": rng.normal(size=n),
    }
)

beta_access = 0.8 + 0.006 * coords[:, 0] + 0.015 * time_index
beta_density = -0.7 + 0.004 * coords[:, 1]
y = (
    3.0
    + beta_access * X["access"].to_numpy()
    + beta_density * X["density"].to_numpy()
    + rng.normal(0.0, 0.35, size=n)
)

model = GTWR(
    kernel="bisquare",
    bandwidth="aicc",
    adaptive=True,
    distance_combination="gwmodel",
    lambda_st="auto",
    lambda_grid_size=9,
    causal=True,
    time_unit="days",
).fit(
    X,
    y,
    coords,
    times,
    compute_hat_matrix=False,
)

print("bandwidth:", model.bandwidth_)
print("lambda:", model.lambda_st_)
print("time unit:", model.time_unit_)
print(model.to_frame().head())

X_new = pd.DataFrame({"access": [0.4], "density": [-0.2]})
coords_new = np.array([[60.0, 45.0]])
times_new = pd.to_datetime(["2024-02-20"])

result = model.predict_result(X_new, coords_new, times_new)
print(result.to_frame())
```

## Constructor

```python
GTWR(
    kernel="bisquare",
    bandwidth="cv",
    bandwidth_method="cv",
    adaptive=False,
    bandwidth_range=None,
    lambda_st=0.05,
    lambda_range=(0.0, 1.0),
    lambda_grid_size=11,
    ksi=0.0,
    distance_combination="gwmodel",
    tau=1.0,
    causal=False,
    time_unit="auto",
    optimization_method="golden_section",
    search_grid_size=25,
    search_tol=1e-5,
    search_max_iter=100,
    fit_intercept=True,
    distance_metric="euclidean",
    sigma2_v1=False,
    verbose=False,
)
```

## Constructor parameters

### Space-time distance

| Parameter | Default | Meaning | Guidance |
|---|---:|---|---|
| `distance_combination` | `"gwmodel"` | Selects GWmodel generalized distance or Euclidean space-time distance. | Do not compare fitted parameters across formulations without holding spatial/time units fixed. |
| `lambda_st` | `0.05` | GWmodel spatial-temporal balance in `[0,1]`, or `"auto"`. | Applies to the GWmodel formulation. Automatic mode evaluates a deterministic lambda grid and selects bandwidth jointly for each candidate. |
| `lambda_range` | `(0,1)` | Bounds for automatic lambda candidates. | Restrict only with scientific justification. Boundary selection suggests the model is approaching a predominantly spatial or temporal distance. |
| `lambda_grid_size` | `11` | Number of lambda candidates. | Larger grids improve resolution but multiply bandwidth searches. |
| `ksi` | `0.0` | GWmodel interaction angle in `[0, pi]`. | It is an angle in radians, not a temporal decay. `cos(ksi)` changes the sign and magnitude of the cross term. |
| `tau` | `1.0` | Non-negative Euclidean temporal scale. | `tau=0` removes temporal distance. Its numerical meaning depends on both coordinate and time units. |
| `causal` | `False` | Excludes observations later than each regression target when true. | Use true for forecasting and temporally ordered evaluation. |
| `time_unit` | `"auto"` | Datetime conversion unit. | Numeric times remain unchanged. Record the resolved `time_unit_`. |

### Kernel and search

| Parameter | Default | Meaning and guidance |
|---|---:|---|
| `kernel` | `"bisquare"` | Kernel applied to combined space-time distance. |
| `bandwidth` | `"cv"` | Numeric value, `"cv"`, `"aicc"`, or `None`. `"adaptive"` is rejected; use `adaptive=True`. |
| `bandwidth_method` | `"cv"` | Criterion when `bandwidth=None`; only CV and AICc are substantive GTWR criteria. `"aic"` is normalised to AICc in selection. |
| `adaptive` | `False` | Numeric bandwidth is a nearest-neighbour count in combined distance when true. |
| `bandwidth_range` | `None` | Optional search bounds. Adaptive lower support must exceed design requirements. |
| `optimization_method` | `"golden_section"` | Grid, golden section, or Brent. Brent applies to fixed bandwidths. |
| `search_grid_size` | `25` | Fixed-bandwidth grid candidates when grid search is used. |
| `search_tol` | `1e-5` | Continuous search tolerance. |
| `search_max_iter` | `100` | Maximum search iterations. |
| `distance_metric` | `"euclidean"` | Spatial distance metric; affects only $d_s$. |
| `fit_intercept` | `True` | Fits a local space-time intercept. |
| `sigma2_v1` | `False` | Residual variance denominator. Default false matches the GWmodel-style `n - 2 trace(S) + trace(S'S)` convention. |
| `verbose` | `False` | Prints lambda/bandwidth selection and fit progress. |

## Fitting

```python
model.fit(
    X,
    y,
    coords,
    times,
    compute_hat_matrix=True,
    compute_local_r2=True,
    compute_inference=True,
    compute_hat_matrix_flag=None,
    verbose=None,
)
```

Smoother traces, influence, information criteria, and residual variance are computed even when the full hat matrix is not retained. Set `compute_hat_matrix=False` for larger samples.

The class stores full spatial, temporal, and combined training distance matrices. Each is approximately `8 × n²` bytes. Standard GTWR can therefore become memory intensive even when `hat_matrix_` is disabled.

## Prediction

```python
pred = model.predict(X_new, coords_new, times_new)
result = model.predict_result(X_new, coords_new, times_new)
params = model.get_local_parameters(coords_new, times_new)
```

Prediction recomputes target-to-training space-time distances and locally recalibrates coefficients. With `causal=True`, only training rows at or before each target time can receive ordinary temporal distance.

A target earlier than most training observations may have insufficient effective support. Forecasting validation should train only on historically available rows and move the origin forward in time.

## Main fitted attributes

| Attribute | Meaning |
|---|---|
| `bandwidth_` | Selected fixed combined distance or adaptive neighbour count. |
| `lambda_st_` | Fitted GWmodel balance parameter. |
| `tau_`, `ksi_` | Resolved Euclidean temporal scale and GWmodel angle. |
| `time_unit_`, `time_origin_` | Fitted time-conversion state. |
| `times_train_` | Numeric times after validation/conversion. |
| `spatial_distance_matrix_` | Pairwise spatial distances. |
| `temporal_distance_matrix_` | Pairwise temporal distances after causal treatment where applicable. |
| `spatiotemporal_distance_matrix_` | Combined training distance matrix. |
| `lambda_selection_history_` | Lambda, bandwidth, and score records. |
| `bandwidth_selection_result_`, `bandwidth_score_` | Optimizer details and final selection score. |
| `coef_`, `intercept_`, `fitted_values_`, `residuals_` | Local regression results. |
| `influence_`, `standardized_residuals_`, `cooks_distance_` | Smoother-based influence diagnostics. |
| `coef_se_`, `coef_t_`, `intercept_se_`, `intercept_t_` | Local inference arrays when enabled. |

## Interpretation and validation

1. Standardise the scientific meaning of spatial and temporal units before fitting.
2. Compare GWR and GTWR using the same response and predictors.
3. Inspect lambda/tau boundary behaviour and bandwidth support.
4. Compare `causal=False` retrospective fit with a strictly causal forecasting design where relevant.
5. Map coefficient changes over both space and time rather than only space.
6. Examine residuals within temporal slices and spatial regions.
7. Use rolling-origin or temporally ordered validation, ideally combined with spatial blocks.

A better in-sample AICc does not demonstrate forecasting skill. Symmetric GTWR can use future observations unless the data split or `causal=True` prevents it.

## Common mistakes

| Mistake | Correction |
|---|---|
| Treating GTWR as stage-based STWR | GTWR requires one time per row and uses direct time distance. |
| Mixing datetime and numeric times across fit/predict | Use one input kind consistently. |
| Ignoring resolved time units | Record `time_unit_`; unit changes alter lambda/tau interpretation. |
| Using `causal=False` for forecasting | Enable causal mode and use ordered training windows. |
| Interpreting `lambda_st` under Euclidean distance | Use `tau` as the Euclidean temporal scale. |
| Comparing tau values after changing coordinate or time units | Tau is unit-dependent. |
| Using `bandwidth="adaptive"` | Set `adaptive=True` and supply/select a valid bandwidth. |
| Disabling the hat matrix but assuming all quadratic memory is removed | Three pairwise distance matrices are still stored. |
| Using random cross-validation | Use temporal ordering and spatial separation appropriate to deployment. |

## What to report

Report:

- row-wise time definition and units;
- datetime conversion and resolved `time_unit_`;
- coordinate reference system and spatial metric;
- distance formulation and complete lambda/ksi or tau specification;
- causal setting;
- kernel, fixed/adaptive bandwidth, search criterion and bounds;
- selected lambda and bandwidth history when automatic;
- residual variance convention;
- influence, residual, and local inference diagnostics;
- spatial-temporal validation design;
- memory-related fit switches;
- pyGWRx version.

## References

- Huang, B., Wu, B., & Barry, M. (2010). Geographically and temporally weighted regression for modeling spatio-temporal variation in house prices. *International Journal of Geographical Information Science*, 24(3), 383–401. [`10.1080/13658810802672469`](https://doi.org/10.1080/13658810802672469)

## Related documentation

- [Generated GTWR API](../api/models/gtwr.md)
- [STWR](stwr.md)
- [MGTWR](mgtwr.md)
- [Spatiotemporal data](../guides/spatiotemporal-data.md)