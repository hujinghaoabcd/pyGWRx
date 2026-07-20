# Spatiotemporal Weighted Regression (`STWR`)

<div class="model-hero" markdown>

**Task:** latest-stage continuous-response regression using current and recent historical snapshots  
**Core mechanism:** combine a stage-specific spatial kernel with a response-variation temporal effect  
**Required inputs:** ordered lists of `X`, `y`, and coordinates plus inter-stage time intervals  
**Independent-target prediction:** supported only for the latest modeled stage and requires a current-stage response reference

</div>

[API reference](../api/models/stwr.md){ .md-button .md-button--primary }
[GTWR manual](gtwr.md){ .md-button }
[MGTWR manual](mgtwr.md){ .md-button }

## What makes STWR different

STWR is not ordinary GTWR applied to repeated snapshots. Data are organised into ordered stages, and the model is calibrated only for the **latest stage**. Recent historical observations are included according to:

- their spatial kernel weight;
- elapsed stage intervals;
- the relative change between their response value and the current query response reference;
- a current-to-past spatial-bandwidth slope.

For a current query $i$ and a past observation $j$ at lagged stage $q$, the public STWR v1.0 formulation uses

$$
d^T_{ij}=\frac{\Delta t_{\mathrm{all}}}{\Delta t_q}
\left|\frac{y_{j,t-q}-y_{i,t}}{y_{j,t-q}}\right|,
$$

then maps it through

$$
T_{ij}=\tanh\left(\frac{d^T_{ij}}{2}\right).
$$

The combined weight is

$$
w_{ij}=(1-\alpha)w^{S}_{ij}+\alpha T_{ij}
$$

when more than one stage is used.

!!! warning "This temporal effect is not conventional decay"
    Larger response variation produces a larger `tanh` temporal effect, approaching 1. The implementation follows the public STWR v1.0 formula. Do not describe `alpha` as a simple preference for recent observations or as exponential time decay.

## Data organisation

Supply stages from **oldest to latest**:

```python
X_list = [X_t0, X_t1, X_t2]
y_list = [y_t0, y_t1, y_t2]
coords_list = [coords_t0, coords_t1, coords_t2]
```

Each stage may contain a different number of rows and different coordinates, but all `X` stages must have the same feature count and, for DataFrames, the same columns in the same order.

`time_intervals` accepts either:

- one interval between each consecutive stage, length `n_stages - 1`; or
- one zero-prefixed value per stage, length `n_stages`.

For three stages:

```python
[7.0, 14.0]
```

is converted internally to:

```python
[0.0, 7.0, 14.0]
```

Values after the initial zero must be positive. They are interval lengths, not absolute timestamps.

## When to use STWR

Use it when:

- the data naturally arrive as repeated stages or snapshots;
- the latest stage is the inferential/prediction target;
- response-change behaviour is central to temporal borrowing;
- sample locations or sample counts may differ by stage;
- only a limited number of recent stages should influence the current fit.

Do not use it when every row already has a meaningful continuous timestamp and ordinary temporal distance is the intended mechanism; use [`GTWR`](gtwr.md). Do not use it for future-stage prediction because the current class predicts only within the latest modeled stage.

## Installation

```bash
pip install pygwrx
```

## Self-contained example

```python
import numpy as np
import pandas as pd

from pygwrx import STWR

rng = np.random.default_rng(111)
X_list = []
y_list = []
coords_list = []

for stage in range(3):
    n = 45
    coords = rng.uniform(0.0, 100.0, size=(n, 2))
    X = pd.DataFrame(
        {
            "access": rng.normal(size=n),
            "density": rng.normal(size=n),
        }
    )
    beta_access = 0.8 + 0.15 * stage + 0.006 * coords[:, 0]
    y = (
        3.0
        + beta_access * X["access"].to_numpy()
        - 0.7 * X["density"].to_numpy()
        + rng.normal(0.0, 0.35, size=n)
    )
    X_list.append(X)
    y_list.append(y)
    coords_list.append(coords)

model = STWR(
    spatial_bandwidth="cv",
    adaptive=True,
    kernel="bisquare",
    alpha="cv",
    theta=0.0,
    tick_nums="cv",
    alpha_candidates=[0.0, 0.3, 0.6],
    tick_candidates=[1, 2, 3],
    store_weights=False,
).fit(
    X_list,
    y_list,
    coords_list,
    time_intervals=[7.0, 7.0],
)

print("bandwidth:", model.spatial_bandwidth_)
print("alpha:", model.alpha_)
print("theta:", model.theta_)
print("stages used:", model.tick_nums_)
print(model.get_results().head())

X_new = pd.DataFrame({"access": [0.4], "density": [-0.2]})
coords_new = np.array([[55.0, 40.0]])

# Supply a scientifically available latest-stage reference response when possible.
result = model.predict_result(
    X_new,
    coords_new,
    reference_y=np.array([3.5]),
)
print(result.to_frame())
```

## Constructor

```python
STWR(
    spatial_bandwidth="cv",
    *,
    adaptive=True,
    kernel="bisquare",
    alpha=0.3,
    theta=0.0,
    tick_nums=None,
    bandwidth_candidates=None,
    alpha_candidates=None,
    theta_candidates=None,
    tick_candidates=None,
    fit_intercept=True,
    distance_metric="euclidean",
    sigma2_v1=True,
    ridge=0.0,
    store_weights=True,
    verbose=False,
)
```

## Constructor parameters

| Parameter | Default | Meaning | Guidance |
|---|---:|---|---|
| `spatial_bandwidth` | `"cv"` | Latest-stage fixed distance or adaptive neighbour count; `cv`, `auto`, `loocv`, or `None` triggers candidate search. | Adaptive bandwidth is constrained by the latest-stage sample size, not the total historical rows. |
| `adaptive` | `True` | Interprets the latest-stage bandwidth as a neighbour count before conversion to a local distance threshold. | Useful under uneven latest-stage sampling. |
| `kernel` | `"bisquare"` | Spatial kernel; allowed values are bisquare, Gaussian, and exponential. | Compact support can make historical stages locally unsupported. |
| `alpha` | `0.3` | Convex-combination weight on the response-variation temporal effect; numeric `[0,1]` or automatic token. | `alpha=0` uses spatial weights only. Larger alpha does not mean faster decay; it gives more weight to `tanh` variation effects. |
| `theta` | `0.0` | Spatial-bandwidth slope angle, strictly between `-pi/2` and `pi/2`, or automatic token. | Earlier-stage bandwidth equals current bandwidth minus `tan(theta) × elapsed`. Positive theta narrows older-stage bandwidths; negative theta widens them. |
| `tick_nums` | `None` | Number of most recent stages included. `None` uses all stages; automatic token selects from candidates. | `1` reduces the model to latest-stage spatial weighting and ignores alpha/theta temporal borrowing. |
| `bandwidth_candidates` | `None` | Explicit candidate bandwidths. | Default adaptive search uses at most 10 values; supply a denser scientifically meaningful grid for final analysis. |
| `alpha_candidates` | `None` | Candidate alphas. | Default automatic grid is 0.0 to 0.9 in steps of 0.1. |
| `theta_candidates` | `None` | Candidate theta values. | Default automatic behaviour evaluates only 0.0 unless explicit candidates are supplied. |
| `tick_candidates` | `None` | Candidate recent-stage counts. | Default automatic search checks all counts from 1 to available stages. |
| `fit_intercept` | `True` | Fits a local intercept. |
| `distance_metric` | `"euclidean"` | Spatial distance metric: Euclidean, Manhattan/cityblock, Chebyshev, or Haversine. |
| `sigma2_v1` | `True` | Uses residual denominator `n_latest - trace(S)`; false uses the alternative smoother denominator. |
| `ridge` | `0.0` | Optional numerical ridge on slopes; intercept is unpenalised. | Positive ridge is a pyGWRx stabilisation choice and must be reported. |
| `store_weights` | `True` | Stores combined, spatial, and temporal latest-to-history weight matrices. | Disable for large stage collections when matrices are not needed. |
| `verbose` | `False` | Prints selection and fit information. |

## Parameter selection

Any subset of bandwidth, alpha, theta, and tick count may be fixed while the others are selected. pyGWRx evaluates the Cartesian product of candidate sets using latest-stage leave-one-out squared error.

At each latest-stage query, its own current-stage source weight is set to zero. Historical rows remain available according to the candidate weights.

`selection_history_` records every candidate combination and CV score. Total work grows multiplicatively:

```text
n_bandwidths × n_alphas × n_thetas × n_tick_counts
```

Use small exploratory grids first, then refine around plausible values. A large unrestricted grid can become expensive.

## Fitted outputs

| Attribute | Meaning |
|---|---|
| `spatial_bandwidth_` | Selected latest-stage bandwidth. |
| `alpha_`, `theta_`, `tick_nums_` | Final temporal/spatial-history controls. |
| `selection_history_` | Candidate combinations and latest-stage CV scores. |
| `stage_slices_` | Source-array slices ordered current stage first, then older stages. |
| `coef_`, `intercept_` | Local parameters at latest-stage coordinates only. |
| `fitted_values_`, `residuals_` | Latest-stage fitted responses and residuals. |
| `smoother_rows_`, `influence_` | Latest-stage smoother rows over selected current/history sources. |
| `coef_se_`, `coef_t_`, `intercept_se_`, `intercept_t_` | Local inference arrays. |
| `weights_`, `spatial_weights_`, `temporal_weights_` | Stored only when `store_weights=True`. |
| `diagnostics_`, `sigma2_` | Latest-stage smoother diagnostics and residual variance. |

`get_results()` returns a latest-stage DataFrame. Earlier-stage coefficients are not fitted outputs.

## Prediction and `reference_y`

STWR temporal weights require a current-stage response reference for each new location. `predict_result()` accepts:

```python
reference_y=np.array([...])
```

When omitted, pyGWRx estimates it from latest-stage observed responses using inverse-distance weighting, matching the public STWR prediction strategy.

This creates an important interpretation boundary:

- supplied `reference_y` means prediction is conditional on an externally available current-stage response baseline;
- omitted `reference_y` means the prediction partly depends on a spatial interpolation of observed latest-stage response.

Do not call the latter a pure predictor using only `X` and coordinates. Report how `reference_y` was obtained.

## Memory and numerical support

With `m` latest-stage query locations and `N` rows across selected stages, each stored weight matrix is approximately `8 × m × N` bytes. Three matrices are retained when `store_weights=True`.

Singular local systems raise an explicit error. Remedies include:

- increasing spatial bandwidth;
- reducing `tick_nums` when historical design creates instability;
- removing collinear predictors;
- using a small reported ridge value.

## Common mistakes

| Mistake | Correction |
|---|---|
| Passing absolute timestamps as `time_intervals` | Supply consecutive positive interval lengths with optional leading zero. |
| Ordering stages newest to oldest | Supply oldest to latest; the class internally reverses selected stages for source fitting. |
| Calling alpha a time-decay rate | It weights the `tanh` response-variation effect. |
| Assuming `theta="cv"` searches many values automatically | Supply `theta_candidates`; the default candidate set is only 0.0. |
| Predicting a future stage | The class predicts locations in the latest modeled stage only. |
| Omitting `reference_y` without disclosure | The model then uses IDW of latest-stage responses. |
| Comparing results with different interval units | The variation scaling and bandwidth slope depend on interval magnitudes. |
| Storing large weight matrices by default | Set `store_weights=False` unless decomposition is needed. |
| Interpreting historical response use as leakage-free forecasting | The method explicitly uses historical responses and may use a current-stage reference; validation must mirror deployment data availability. |

## What to report

Report:

- stage definition and oldest-to-latest ordering;
- rows and coordinate support per stage;
- inter-stage intervals and units;
- selected recent-stage count;
- spatial metric, kernel, fixed/adaptive bandwidth;
- alpha and its exact response-variation interpretation;
- theta in radians and historical bandwidth behaviour;
- candidate grids and CV selection size;
- ridge and variance convention;
- source of prediction `reference_y`;
- weight storage and memory choices;
- latest-stage residual/inference diagnostics;
- temporally realistic validation design;
- pyGWRx version.

## References

- Que, X., Ma, X., Ma, C., & Chen, Q. (2020). A spatiotemporal weighted regression model (STWR v1.0) for analyzing local nonstationarity in space and time. *Geoscientific Model Development*, 13, 6149–6164. [`10.5194/gmd-13-6149-2020`](https://doi.org/10.5194/gmd-13-6149-2020)

## Related documentation

- [Generated STWR API](../api/models/stwr.md)
- [GTWR](gtwr.md)
- [MGTWR](mgtwr.md)
- [Spatiotemporal data](../guides/spatiotemporal-data.md)