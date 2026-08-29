# Standard Geographically Weighted Regression (`GWR`)

<div class="model-hero" markdown>

**Task:** continuous-response local linear regression  
**Core assumption:** all local coefficient surfaces share one spatial bandwidth  
**Required inputs:** predictor matrix `X`, numeric response `y`, coordinates `coords`  
**Independent-target prediction:** supported by local recalibration at target coordinates

</div>

[API reference](../api/models/gwr.md){ .md-button .md-button--primary }
[Model selection guide](../getting-started/choosing-a-model.md){ .md-button }
[Kernels and bandwidths](../guides/kernels-and-bandwidths.md){ .md-button }

## What GWR is for

Standard GWR is an exploratory spatially varying-coefficient regression. Instead of estimating one coefficient vector for the whole study area, it estimates a weighted local regression at each calibration location. The original method was introduced to investigate spatial non-stationarity: the possibility that the relationship between a continuous response and its predictors changes across geographic space.

At target location $s_i$, the fitted parameter vector is

$$
\widehat{\boldsymbol\beta}(s_i)
=\left(X^\top W_iX\right)^{-1}X^\top W_i y,
$$

where $W_i$ is a diagonal matrix of kernel weights determined by the distances from $s_i$ to the observations and by one shared bandwidth.

### Use GWR when

- the response is continuous and a local linear relationship is scientifically plausible;
- the main question is whether coefficient magnitude or sign varies smoothly over space;
- one common spatial scale is an acceptable first approximation;
- a transparent spatial baseline is needed before fitting MGWR or a more specialised model;
- local coefficients, fitted values and diagnostics at observed or new locations are required.

### Do not make GWR the first choice when

| Situation | Better starting point |
|---|---|
| The response is a count, rate or binary outcome | [`GWGLM`](gwglm.md) |
| Different predictors clearly operate at different spatial scales | [`MGWR`](mgwr.md) |
| Some coefficients should be constant over the whole study area | [`MixedGWR`](mixed-gwr.md) |
| Strong response outliers dominate local fits | [`RGWR`](rgwr.md) |
| Local collinearity is the main problem | [`LCRGWR`](lcr-gwr.md) or [`GWLasso`](gw-lasso.md) |
| Relationships vary materially through time | [`GTWR`](gtwr.md), [`STWR`](stwr.md) or [`MGTWR`](mgtwr.md) |
| The sample is too large for repeated conventional local fits | [`ScalableGWR`](scalable-gwr.md) |

!!! warning "GWR is not automatically causal"
    A mapped local coefficient is a conditional association produced by a chosen neighbourhood and model specification. Spatial variation may also reflect omitted variables, collinearity, outliers, sampling density, boundary effects or residual dependence.

## What pyGWRx implements

The published GWR idea and the pyGWRx class are closely aligned, but the software contract is more specific:

- Gaussian local weighted least squares;
- fixed-distance or adaptive-neighbour bandwidths;
- built-in Gaussian, bisquare, exponential, tricube and boxcar kernels, or a callable kernel;
- automatic bandwidth selection by CV, AIC, AICc or BIC;
- fitted-model bandwidth-search provenance including criterion, search method, search range, evaluated trace, best score and boundary-solution flag;
- optional storage of the full hat matrix while always retaining its traces and influence diagnostics;
- local numerical-rank and condition diagnostics for every fitted weighted design;
- local standard errors and t statistics only where the local parameterisation is numerically identifiable;
- local R², standardised residuals and Cook's distance;
- target-location prediction by recalibrating local coefficients from the stored training data.

Prediction is **not** interpolation of the coefficient maps. For every target coordinate, pyGWRx recomputes its distances to the training observations, constructs a new kernel, solves a new weighted regression, and applies those local coefficients to the supplied target predictors.

## Installation

```bash
pip install pygwrx
```

GeoPandas is only needed for GeoDataFrame-oriented output. NumPy arrays and pandas DataFrames are sufficient for fitting and prediction.

## Input data contract

| Input | Shape | Meaning | Important checks |
|---|---:|---|---|
| `X` | `(n, p)` | Numeric predictors. Do not manually add an intercept when `fit_intercept=True`. | No missing or infinite values; DataFrame column order is preserved and checked during prediction. |
| `y` | `(n,)` | Continuous numeric response. | Same row order and sample count as `X` and `coords`. |
| `coords` | `(n, 2)` | Two-dimensional coordinates used to calculate neighbourhood distances. | Use a projected CRS for planar metrics, or `[longitude, latitude]` in degrees with `distance_metric="haversine"`. |

For Euclidean distance, the bandwidth has the same unit as the coordinates. Coordinates in metres produce a fixed bandwidth in metres. The Haversine implementation expects longitude first, latitude second, and returns kilometres with its default Earth radius.

## Minimal self-contained example

This example creates a spatially varying synthetic relationship, selects an adaptive AICc bandwidth, fits GWR without storing the full hat matrix, and predicts at new locations.

```python
import numpy as np
import pandas as pd

from pygwrx import GWR

rng = np.random.default_rng(42)
n = 80

coords = rng.uniform(0.0, 100.0, size=(n, 2))
X = pd.DataFrame(
    {
        "income": rng.normal(size=n),
        "access": rng.normal(size=n),
    }
)

# Create coefficients that vary smoothly with x and y coordinates.
beta_income = 1.2 + 0.012 * coords[:, 0]
beta_access = -0.9 + 0.010 * coords[:, 1]
y = (
    4.0
    + beta_income * X["income"].to_numpy()
    + beta_access * X["access"].to_numpy()
    + rng.normal(0.0, 0.35, size=n)
)

model = GWR(
    kernel="bisquare",
    bandwidth="aicc",
    adaptive=True,
).fit(
    X,
    y,
    coords,
    compute_hat_matrix=False,
)

print("selected neighbour count:", model.bandwidth_)
print("bandwidth search:", model.bandwidth_search_)
print("rank-deficient locations:", model.rank_deficient_.sum())
print(model.get_diagnostics())
print(model.to_frame().head())

X_new = pd.DataFrame(
    {
        "income": [0.25, -0.40],
        "access": [1.10, 0.30],
    }
)
coords_new = np.array([[25.0, 30.0], [75.0, 65.0]])

prediction = model.predict_result(X_new, coords_new)
print(prediction.to_frame())
```

For a first real-data run, replace the generated `X`, `y` and `coords` while preserving row alignment and coordinate units.

## Constructor

```python
GWR(
    kernel="gaussian",
    bandwidth="cv",
    bandwidth_method="cv",
    adaptive=False,
    bandwidth_range=None,
    optimization_method="golden_section",
    fit_intercept=True,
    distance_metric="euclidean",
    sigma2_v1=True,
    verbose=False,
)
```

### Constructor parameters

| Parameter | Accepted values and default | What it controls | How to use it |
|---|---|---|---|
| `kernel` | Built-in name or callable; default `"gaussian"` | Converts distance into non-negative local weights. | Gaussian and exponential have non-zero tails. Bisquare and tricube are compact and make observations outside the local bandwidth exactly zero. Boxcar gives equal weight inside the neighbourhood. Compare sensitivity rather than selecting from appearance alone. |
| `bandwidth` | Positive number, `"cv"`, `"aic"`, `"aicc"`, `"bic"`, or `None`; default `"cv"` | Sets or selects the shared spatial scale. | With `adaptive=False`, a number is a distance. With `adaptive=True`, it must be an integer neighbour count. A string directly chooses the automatic criterion. `None` delegates to `bandwidth_method`. |
| `bandwidth_method` | `"cv"`, `"aic"`, `"aicc"`, `"bic"`; default `"cv"` | Automatic criterion used only when `bandwidth=None`. | It does not override `bandwidth="aicc"` or another explicit criterion string. AICc is often a useful inferential default; CV directly targets leave-one-out prediction error. Record the chosen criterion. |
| `adaptive` | Boolean; default `False` | Chooses fixed-distance versus nearest-neighbour bandwidth semantics. | Compare adaptive bandwidths when sampling density is uneven. Never interpret an adaptive bandwidth as metres or kilometres. |
| `bandwidth_range` | `(lower, upper)` or `None` | Restricts automatic search. | Use scientifically defensible bounds or bounds that prevent underdetermined local fits. Adaptive bounds must represent integer neighbour counts. Without explicit fixed bounds, pyGWRx searches from half the smallest positive pairwise distance to twice the largest pairwise distance so isolated locations are not removed by percentile trimming. Check whether the selected value lies on a boundary. |
| `optimization_method` | `"golden_section"`, `"brent"`, or `"grid"`; default `"golden_section"` | Numerical search used for automatic bandwidth selection. | Grid search is transparent and useful for sensitivity checks. Continuous methods are usually faster for fixed-distance bandwidths. Adaptive searches evaluate every valid integer neighbour count regardless of this setting. |
| `fit_intercept` | Boolean; default `True` | Adds a spatially varying intercept. | Leave enabled unless the scientific model genuinely requires a zero response at zero predictors. Do not add an all-ones column to `X` when it is enabled. |
| `distance_metric` | `"euclidean"`, `"manhattan"`/`"cityblock"`, `"chebyshev"`, `"minkowski"`, `"haversine"`; default `"euclidean"` | Defines geographic proximity. | Euclidean is appropriate for projected planar coordinates. Haversine expects `[longitude, latitude]` degrees. Alternative metrics must be scientifically justified because they change the neighbourhood itself. |
| `sigma2_v1` | Boolean; default `True` | Selects the residual-variance denominator. | `True` uses `RSS / (n - trace(S))`; `False` uses `RSS / (n - 2 trace(S) + trace(S'S))`. Keep this setting fixed when comparing reported standard errors across models. |
| `verbose` | Boolean; default `False` | Prints bandwidth and fit progress. | Enable during slow searches or debugging; it does not change the estimator. |

### Understanding bandwidth size and local rank

A smaller bandwidth gives a more local and flexible surface but uses less information per fit. It can increase variance, local singularity and sensitivity to individual observations. A larger bandwidth smooths coefficients and approaches a more global relationship.

With an adaptive bandwidth, the local distance threshold changes by location so that each local regression is based on approximately the same number of nearest observations. With compact kernels, the number of positive-weight observations is especially important. A local design can also be rank deficient even when the raw positive-weight observation count exceeds the number of parameters, for example because two predictors become locally collinear or one predictor has effectively no local variation.

pyGWRx solves each unpenalized weighted design with one SVD. When the weighted design is rank deficient, the coefficient vector remains the Moore-Penrose minimum-norm solution and fitted values remain numerically defined. That does **not** make the individual coefficients identifiable. Such locations are marked in `rank_deficient_`; `local_rank_` and `local_condition_number_` expose the numerical diagnosis; coefficient covariance, standard errors and t statistics are returned as `NaN` at those locations. Increase the bandwidth, simplify the design or diagnose local collinearity before interpreting the affected coefficient surfaces.

## Fitting

```python
model.fit(
    X,
    y,
    coords,
    compute_hat_matrix=True,
    compute_local_r2=True,
    compute_inference=True,
    verbose=None,
)
```

| Fit argument | Default | Meaning and practical choice |
|---|---:|---|
| `compute_hat_matrix` | `True` | Stores the full `n × n` smoother matrix. Set to `False` for larger samples. The trace, `trace(S'S)`, influence, AIC/AICc/BIC and effective-parameter diagnostics are still computed. |
| `compute_local_r2` | `True` | Computes a weighted local R² at every calibration location. Disable only when it is not needed and fit time matters. Local R² is descriptive and should not replace residual checks. |
| `compute_inference` | `True` | Computes covariance diagonals, local standard errors and local t statistics where the local weighted design is identifiable. Rank diagnostics are retained even when this option is disabled. |
| `verbose` | `None` | Optional per-fit override of the constructor setting. |

`compute_hat_matrix_flag` remains accepted only as a backward-compatibility alias for older pyGWRx code; new workflows should use `compute_hat_matrix`.

### Memory guidance

The full hat matrix requires roughly `8 × n²` bytes before Python-array overhead. For example, `n=10,000` implies about 800 MB for one float64 matrix. Use `compute_hat_matrix=False` unless the actual matrix entries are required; diagnostics do not require it to be retained. Standard GWR currently still forms a full pairwise distance matrix during calibration, so disabling hat-matrix storage does **not** yet remove all `O(n²)` memory use.

## Prediction and target-location coefficients

```python
pred = model.predict(X_new, coords_new)
result = model.predict_result(X_new, coords_new)
params = model.get_local_parameters(coords_new)
```

| Method | Returns | Use case |
|---|---|---|
| `predict()` | One prediction per target row | Standard numeric prediction. |
| `predict_result()` | Predictions, local slopes, intercepts, coordinates and optional standard errors/t statistics | Auditable prediction and coefficient inspection. Rank-deficient target recalibrations keep predictions but expose `NaN` coefficient inference. |
| `get_local_parameters()` | Dictionary containing target intercepts, slopes, coordinates, local rank, local condition number and rank-deficiency flag | Coefficient surfaces and numerical identifiability at target locations without applying target `X`. |
| `get_local_coefficients()` | Slopes only | Compatibility helper; prefer `get_local_parameters()` when the intercept or rank diagnosis matters. |

The rows and columns of `X_new` must correspond to `coords_new` and to the training predictors. When DataFrames are used, column names and order must match the fitted model.

## Main fitted attributes

| Attribute | Shape or type | Interpretation |
|---|---|---|
| `bandwidth_` | scalar | Selected fixed distance or adaptive neighbour count. |
| `bandwidth_search_` | dictionary or `None` | Automatic-search provenance: criterion, search method/range, selected value, best score, evaluated trace and `boundary_solution`. It is `None` when a numeric bandwidth is supplied directly. |
| `coef_` | `(n, p)` | Local slope estimates at calibration locations. Rank-deficient locations contain the minimum-norm coefficient vector and require the rank flags below for interpretation. |
| `intercept_` | `(n,)` | Local intercepts, or zeros when no intercept is fitted. |
| `fitted_values_` | `(n,)` | Calibration-location fitted responses. |
| `residuals_` | `(n,)` | `y - fitted_values_`. |
| `local_rank_` | `(n,)` integer array | Numerical rank of each weighted local design matrix. |
| `local_condition_number_` | `(n,)` | Condition number of each weighted local design; rank-deficient locations are reported as infinity. |
| `rank_deficient_` | `(n,)` boolean array | `True` where local rank is smaller than the number of fitted parameters. |
| `local_r2_` | `(n,)` or `None` | Weighted local R². Interpret together with local sample support. |
| `diagnostics_` | dictionary | Global fit and GWR smoother diagnostics, including information criteria and effective complexity where available. |
| `influence_` | `(n,)` | Diagonal of the smoother matrix. Large values indicate locally influential observations. |
| `standardized_residuals_` | `(n,)` | Residuals adjusted for fitted variance and leverage. |
| `cooks_distance_` | `(n,)` | Local influence summary based on standardised residuals and leverage. |
| `coef_se_`, `intercept_se_` | local arrays or `None` | Local standard errors when inference is enabled; rows are `NaN` at rank-deficient local designs. |
| `coef_t_`, `intercept_t_` | local arrays or `None` | Local coefficient-to-standard-error ratios; rows are `NaN` where the corresponding local parameterisation is not identifiable. Account for multiple local comparisons. |
| `hat_matrix_` | `(n, n)` or `None` | Stored smoother matrix only when requested. `S_matrix_` is a compatibility alias. |

`to_frame()` combines coordinates, coefficients, fitted values, residuals, local R², inference arrays, influence measures and local rank diagnostics into one location-indexed pandas DataFrame.

## How to interpret a fitted GWR responsibly

### 1. Establish a global baseline

Fit and inspect an ordinary linear model using the same response and predictors. GWR should answer a spatial non-stationarity question, not merely replace an unexamined global model.

### 2. Inspect the selected bandwidth

- Inspect `bandwidth_search_` rather than only `bandwidth_`; it records the criterion, evaluated search trace and whether the selected value is a search-boundary solution.
- A selected boundary value indicates that the search range may be constraining the result.
- A very large adaptive bandwidth suggests weak evidence for strongly local variation.
- A very small bandwidth can produce unstable coefficients even when fit statistics improve.

### 3. Check numerical identifiability before interpreting coefficients

Inspect `rank_deficient_`, `local_rank_` and `local_condition_number_` before mapping or testing coefficients. A minimum-norm coefficient at a rank-deficient location is a valid numerical solution for prediction, but the individual parameter decomposition is not uniquely identified. Do not interpret its sign, magnitude or significance as if it were an ordinary full-rank local estimate.

### 4. Check coefficient stability and uncertainty

Map coefficients together with standard errors or adjusted significance information only after excluding or resolving rank-deficient locations. Compare local condition diagnostics and coefficient correlations. Abrupt isolated coefficient changes are often a warning rather than a substantive finding.

### 5. Examine influence and residuals

Use `standardized_residuals_`, `cooks_distance_` and residual maps. Remaining residual spatial structure means that local coefficient variation has not explained all spatial dependence.

### 6. Validate the intended use

Random train/test splitting can leak spatial information. For claims about transfer to new places, use spatial blocks or held-out regions and compare against global and simpler spatial baselines.

## Common mistakes and corrections

| Mistake | Why it is a problem | Correction |
|---|---|---|
| Passing longitude/latitude to Euclidean distance | Degrees are not a uniform planar distance unit. | Project the data or use Haversine deliberately. |
| Setting `bandwidth=30` with `adaptive=False` while intending 30 neighbours | The value is interpreted as 30 coordinate units. | Set `adaptive=True`. |
| Using `bandwidth="adaptive"` | This string is intentionally rejected. | Use `adaptive=True` with a numeric or automatically selected bandwidth. |
| Adding an intercept column while `fit_intercept=True` | Creates a duplicate constant and local singularity. | Supply predictors only. |
| Treating a finite minimum-norm coefficient as identified after a rank warning | Rank deficiency means multiple coefficient vectors can represent the same local fit. | Inspect `rank_deficient_`; change the bandwidth/design or treat affected coefficient inference as unavailable. |
| Mapping coefficients without uncertainty or collinearity | Attractive surfaces may be numerically unstable or statistically weak. | Pair coefficient maps with SE/t information, rank/condition diagnostics and residuals. |
| Treating local t values as independent tests | Many overlapping local models create a multiple-comparison problem. | Use adjusted procedures and interpret spatial patterns, not isolated threshold crossings. |
| Comparing fixed and adaptive bandwidth numbers directly | They have different units and meanings. | Compare fitted neighbourhoods, criteria and effective support, not raw numbers. |
| Calling in-sample `score()` predictive validation | Recalibration at training locations is not out-of-area validation. | Use a spatially structured holdout design. |

## What to report

A reproducible GWR analysis should report:

- response and predictor definitions and preprocessing;
- sample size and coordinate reference system;
- distance metric;
- kernel;
- fixed or adaptive bandwidth semantics;
- automatic criterion, optimisation method and search bounds;
- selected bandwidth and whether it reached a boundary;
- intercept and residual-variance conventions;
- number and spatial distribution of rank-deficient local fits, if any;
- effective parameter count, AICc/CV and global comparison;
- local coefficient summaries with uncertainty and collinearity/rank checks;
- influence and residual diagnostics;
- validation design;
- pyGWRx version and relevant fit switches.

## Published method versus pyGWRx

| Topic | Published GWR concept | pyGWRx contract |
|---|---|---|
| Coefficients | Location-specific weighted regressions | Gaussian local WLS at calibration or target locations; one SVD supplies the minimum-norm solution and numerical rank. |
| Spatial scale | One kernel bandwidth | Fixed distance or adaptive neighbour count; manual or CV/AIC/AICc/BIC selected. Automatic searches retain their evaluated provenance on the fitted model. |
| Prediction | Local calibration may be performed at arbitrary locations | Explicit `predict_result()` recalibrates coefficients from stored training observations. |
| Diagnostics | Weighting-function and bandwidth choice are central | Smoother traces, information criteria, local R², inference, leverage, Cook's distance, local rank/condition diagnostics and export helpers. |
| Rank-deficient local fit | Local normal equations may be singular or ill conditioned | Preserve the Moore-Penrose minimum-norm fitted solution, flag the location, and suppress coefficient SE/t inference with `NaN`. |
| Large samples | Conventional repeated local fitting can be expensive | Full distance calculations remain part of standard GWR; use memory switches or `ScalableGWR` when needed. |

## References

- Brunsdon, C., Fotheringham, A. S., & Charlton, M. E. (1996). Geographically Weighted Regression: A Method for Exploring Spatial Nonstationarity. *Geographical Analysis*, 28, 281–298. [`10.1111/j.1538-4632.1996.tb00936.x`](https://doi.org/10.1111/j.1538-4632.1996.tb00936.x)
- Fotheringham, A. S., Brunsdon, C., & Charlton, M. (2002). *Geographically Weighted Regression: The Analysis of Spatially Varying Relationships*. Wiley.
- Comber, A. et al. (2022). A route map for the informed application of Geographically Weighted Regression. *Geographical Analysis*, 55, 155–178. [`10.1111/gean.12316`](https://doi.org/10.1111/gean.12316)

## Related documentation

- [Generated `GWR` API](../api/models/gwr.md)
- [MGWR user manual](mgwr.md)
- [Kernels and bandwidths](../guides/kernels-and-bandwidths.md)
- [Diagnostics and inference](../guides/diagnostics.md)
- [Prediction and result objects](../guides/prediction-and-results.md)