# Similarity Geographically and Temporally Weighted Regression (`SGTWR`)

<div class="model-hero" markdown>

**Task:** continuous-response local Gaussian regression using spatial proximity, temporal proximity, and predictor-space similarity  
**Core mechanism:** combine a two-bandwidth Gaussian space-time kernel with the SGWR attribute-similarity kernel  
**Required inputs:** `X`, `y`, coordinates, row-wise times, and defensible similarity variables  
**Independent-target prediction:** supported at new space-time locations, with optional history-only filtering

</div>

[API reference](../api/models/sgtwr.md){ .md-button .md-button--primary }
[SGWR manual](sgwr.md){ .md-button }
[GTWR manual](gtwr.md){ .md-button }

## Why SGTWR exists

GTWR defines neighbours through space and time. SGWR defines neighbours through geography and attribute similarity. SGTWR keeps both ideas separate and then combines them, allowing a geographically or temporally distant observation to contribute when its selected contextual attributes resemble the target.

The spatiotemporal component is

$$
w_{ij}^{ST}=\exp\left[-\frac12\left\{
\left(\frac{d_{ij}^{S}}{h_i^{S}}\right)^2+
\left(\frac{d_{ij}^{T}}{h^{T}}\right)^2
\right\}\right],
$$

where $h_i^S$ is a fixed spatial distance or a target-specific adaptive-neighbour distance and $h^T$ is one positive temporal bandwidth.

For standardized similarity variables $z$,

$$
d_{ij}^{A}=\frac{1}{m}\sum_{k=1}^{m}|z_{ik}-z_{jk}|,
\qquad
w_{ij}^{A}=\exp[-(d_{ij}^{A})^2].
$$

The final weight is

$$
w_{ij}=\alpha w_{ij}^{ST}+(1-\alpha)w_{ij}^{A}.
$$

Therefore:

- `alpha=1` gives a pure two-bandwidth spatiotemporal model;
- `alpha=0` gives similarity-only weighting;
- intermediate values combine both components.

Unlike GTWR's single combined-distance kernel, SGTWR normalizes spatial and temporal distances with separate bandwidths before adding their squared effects.

## When to use SGTWR

Use SGTWR when:

- the response is continuous;
- every row has a spatial coordinate and time;
- observations can be related through functional/contextual similarity beyond physical proximity;
- the selected similarity variables are available at prediction time;
- separate spatial and temporal bandwidths are scientifically meaningful;
- comparison with GWR, SGWR, and GTWR is part of the analysis.

Do not select SGTWR merely because it has more tuning parameters. Its neighbourhood is dense and flexible, and weak validation can make in-sample gains look stronger than genuine transfer performance.

| Situation | Better starting point |
|---|---|
| Time is irrelevant | [`SGWR`](sgwr.md) |
| Attribute similarity is not scientifically justified | [`GTWR`](gtwr.md) |
| Each coefficient requires a different space-time scale | [`MGTWR`](mgtwr.md), which does not include similarity |
| Data are ordered stages with response-change temporal effects | [`STWR`](stwr.md) |
| Future predictor attributes are unavailable | Do not use those variables for target similarity. |

## Published method versus pyGWRx selection

The 2025 SGTWR paper combines a Gaussian space-time kernel with SGWR-style attribute similarity and tunes spatial bandwidth, temporal bandwidth, and component weights with a genetic algorithm.

pyGWRx keeps the published weight formulas but replaces the genetic algorithm with a deterministic AICc candidate search. This difference is deliberate:

- repeated runs are reproducible;
- every evaluated combination is recorded;
- tests can verify exact candidate behaviour;
- the result is the best supplied/default candidate combination, not a continuous genetic-algorithm optimum.

Do not report pyGWRx as using a genetic algorithm.

## Similarity variables and leakage

`similarity_vars` accepts predictor names or indices; `None` uses all predictors. As in SGWR, selected variables affect both the local regression design and the neighbourhood.

With `standardize_similarity=True`, training means and population standard deviations are stored and reused for target rows. A zero-variance variable receives scale 1 and contributes no attribute separation.

Similarity variables must be:

- known at each prediction time;
- computed without future observations when forecasting;
- independent of the response definition;
- measured consistently across time;
- substantively meaningful as a functional-neighbourhood definition.

A time-varying predictor can make two observations similar because their contemporaneous values match. That is legitimate only when the target-time value is available under the deployment scenario.

## Causal filtering

With `causal=False`, temporal distance uses absolute time difference, so later observations can influence earlier calibration or target times.

With `causal=True`, pyGWRx sets future-source weights to zero in:

1. the spatiotemporal component; and
2. the final combined matrix.

The second step is essential: it prevents the similarity component from reintroducing future observations after the space-time component has excluded them.

!!! warning "Causal weighting is not a complete forecasting design"
    You must still ensure that predictor values, similarity attributes, preprocessing statistics, and candidate selection are based only on information available at each forecast origin.

## Time input

SGTWR reuses GTWR's time converter.

- Numeric time remains in the supplied unit.
- Datetime-like input is converted relative to a fitted origin.
- `time_unit="auto"` chooses seconds, minutes, hours, days, or weeks from the training span.
- Target datetimes reuse the fitted origin and unit.

Record `time_unit_`. Spatial bandwidth, temporal bandwidth, and alpha cannot be interpreted reproducibly without coordinate and time units.

## Installation

```bash
pip install pygwrx
```

## Self-contained example

The example uses restricted candidate sets so the fitting cost is visible and controlled.

```python
import numpy as np
import pandas as pd

from pygwrx import SGTWR

rng = np.random.default_rng(151)
n = 72
coords = rng.uniform(0.0, 100.0, size=(n, 2))
times = pd.date_range("2023-01-01", periods=n, freq="7D")
time_index = np.arange(n, dtype=float)

functional_group = rng.integers(0, 2, size=n)
X = pd.DataFrame(
    {
        "income": rng.normal(functional_group, 0.6, size=n),
        "climate": rng.normal(1 - functional_group, 0.6, size=n),
        "density": rng.normal(size=n),
    }
)

beta_income = 0.7 + 0.010 * coords[:, 0] + 0.004 * time_index
beta_climate = np.where(functional_group == 0, -1.0, -0.4)
y = (
    3.0
    + beta_income * X["income"].to_numpy()
    + beta_climate * X["climate"].to_numpy()
    + 0.4 * X["density"].to_numpy()
    + rng.normal(0.0, 0.35, size=n)
)

model = SGTWR(
    spatial_bandwidth="aicc",
    temporal_bandwidth="aicc",
    alpha="aicc",
    adaptive=True,
    spatial_bandwidth_candidates=[28, 42, 58],
    temporal_bandwidth_candidates=[35.0, 70.0, 140.0],
    alpha_candidates=[0.25, 0.50, 0.75, 1.0],
    similarity_vars=["income", "climate"],
    standardize_similarity=True,
    causal=True,
    time_unit="days",
    store_weights=False,
).fit(X, y, coords, times)

print("spatial bandwidth:", model.spatial_bandwidth_)
print("temporal bandwidth:", model.temporal_bandwidth_)
print("spatiotemporal alpha:", model.alpha_)
print("time unit:", model.time_unit_)
print(pd.DataFrame(model.selection_history_).sort_values("aicc").head())
print(model.get_results().head())

X_new = pd.DataFrame(
    {
        "income": [0.4, 1.2],
        "climate": [0.9, 0.2],
        "density": [0.1, -0.3],
    }
)
coords_new = np.array([[25.0, 30.0], [75.0, 65.0]])
times_new = pd.to_datetime(["2024-06-01", "2024-08-01"])

prediction = model.predict_result(X_new, coords_new, times_new)
print(prediction.to_frame())
```

The temporal candidates in this example are days because `time_unit="days"`. They are not dates and do not have spatial units.

## Constructor

```python
SGTWR(
    spatial_bandwidth="aicc",
    *,
    temporal_bandwidth="aicc",
    adaptive=True,
    alpha="aicc",
    similarity_vars=None,
    standardize_similarity=True,
    spatial_bandwidth_candidates=None,
    temporal_bandwidth_candidates=None,
    alpha_candidates=None,
    causal=False,
    time_unit="auto",
    fit_intercept=True,
    distance_metric="euclidean",
    sigma2_v1=True,
    ridge=0.0,
    store_weights=True,
    verbose=False,
)
```

## Constructor parameters

### Spatiotemporal component

| Parameter | Default | Meaning | Guidance and failure modes |
|---|---:|---|---|
| `spatial_bandwidth` | `"aicc"` | Positive fixed distance, adaptive integer count, `None`, or automatic AICc token. | Adaptive minimum is `n_parameters + 1`. It normalizes only spatial distance in the space-time Gaussian kernel. |
| `temporal_bandwidth` | `"aicc"` | Positive time bandwidth, `None`, or automatic AICc token. | Its unit is the fitted numeric time unit. Small values sharply restrict temporal influence. |
| `adaptive` | `True` | Makes the spatial bandwidth a target-specific nearest-neighbour distance. | Temporal bandwidth remains one fixed positive value. |
| `causal` | `False` | Excludes later source rows when true. | Use true for forecasting; retain false only for clearly retrospective/symmetric analyses. |
| `time_unit` | `"auto"` | Datetime conversion convention. | Numeric times are not rescaled. Record the resolved `time_unit_`. |
| `distance_metric` | `"euclidean"` | Euclidean, Manhattan/cityblock, Chebyshev, or Haversine spatial distance. | This affects spatial distance only; temporal distance remains absolute numeric difference. |

### Similarity and mixing

| Parameter | Default | Meaning and guidance |
|---|---:|---|
| `alpha` | `"aicc"` | Contribution of the spatiotemporal component in `[0,1]`. | `1` is pure space-time weighting; `0` is similarity-only weighting. It is not an explained-variance share. |
| `similarity_vars` | `None` | Predictor names or indices defining similarity. | `None` uses every predictor. Avoid leakage and variables unavailable at target time. |
| `standardize_similarity` | `True` | Applies training Z-score-style transformation before mean absolute differences. | Keep true unless raw-unit dominance is scientifically intentional. |

### Candidate search

| Parameter | Default | Automatic candidate behaviour |
|---|---:|---|
| `spatial_bandwidth_candidates` | `None` | Adaptive: up to 10 rounded values from `n_parameters + 1` to `n`; fixed: 8 quantiles from 0.2 to 0.9 of positive spatial distances. |
| `temporal_bandwidth_candidates` | `None` | 8 quantiles from 0.2 to 1.0 of positive pairwise temporal differences; `[1.0]` when all times are equal. |
| `alpha_candidates` | `None` | 11 values from 0 to 1 inclusive. |

A numeric/explicit value for one parameter collapses only that candidate dimension. Other automatic dimensions are still searched.

Default automatic search can evaluate roughly:

```text
10 spatial × 8 temporal × 11 alpha = 880 full SGTWR fits
```

Each fit builds a full smoother matrix. Supply coarse candidate sets first, then refine around stable regions.

### Estimation and storage

| Parameter | Default | Meaning and guidance |
|---|---:|---|
| `fit_intercept` | `True` | Fits a local intercept. Do not add a manual constant. |
| `sigma2_v1` | `True` | Uses `RSS / (n - trace(S))`; false uses the alternative smoother denominator. |
| `ridge` | `0.0` | Optional ridge on slope diagonals; intercept remains unpenalized. | Positive ridge is a pyGWRx stabilisation extension and must be reported. |
| `store_weights` | `True` | Retains spatiotemporal, similarity, and combined `n × n` matrices. | Disable unless component decomposition is needed. The full hat matrix remains retained. |
| `verbose` | `False` | Prints selected scales, alpha, and AICc. |

## How parameter selection works

pyGWRx evaluates the complete Cartesian product of candidate lists. For each combination, it:

1. constructs the spatiotemporal Gaussian matrix;
2. applies causal filtering when enabled;
3. combines it with the similarity matrix;
4. fits local regressions at all training rows;
5. calculates the full smoother-matrix AICc;
6. stores spatial bandwidth, temporal bandwidth, alpha, and AICc in `selection_history_`.

The smallest finite AICc wins. Ties retain the first combination encountered in sorted candidate order.

This is **not** leave-one-out forecasting validation. Each training row is included in its own local fit. Use temporal/spatial holdouts in addition to AICc.

## Fitting and memory

```python
model.fit(X, y, coords, times)
```

The fit method always computes and retains:

- the full `n × n` hat matrix;
- local covariance factors, standard errors, and t values;
- influence and global smoother diagnostics.

With `store_weights=True`, three more `n × n` matrices are retained. At `n=10,000`, the four float64 matrices alone require approximately 3.2 GB before temporaries and overhead.

The model does not currently expose switches for omitting the hat matrix or inference. Use smaller candidate grids and `store_weights=False` when memory is limited.

## Prediction semantics

```python
pred = model.predict(X_new, coords_new, times_new)
result = model.predict_result(X_new, coords_new, times_new)
```

For each target, pyGWRx:

1. converts target time with the fitted time origin and unit;
2. validates target predictor columns;
3. computes target-to-training space-time weights;
4. transforms target similarity variables with training statistics;
5. computes target-to-training attribute weights;
6. removes future sources from both components when causal;
7. combines weights with fitted alpha;
8. recalibrates local coefficients from the training response.

Target predictor values can therefore affect both the local design and the neighbourhood. All selected similarity attributes must be known at target time.

## Main fitted attributes

| Attribute | Meaning |
|---|---|
| `spatial_bandwidth_` | Selected fixed spatial distance or adaptive neighbour count. |
| `temporal_bandwidth_` | Selected positive time scale in `time_unit_`. |
| `alpha_` | Fitted spatiotemporal mixing proportion. |
| `selection_history_` | Every candidate combination and AICc. Empty when all three parameters are explicit. |
| `time_unit_`, `time_converter_`, `times_train_` | Fitted time conversion and numeric training times. |
| `similarity_indices_`, `similarity_feature_names_` | Variables defining attribute distance. |
| `similarity_mean_`, `similarity_scale_` | Training transformation reused for targets. |
| `spatiotemporal_weights_`, `similarity_weights_`, `combined_weights_` | Optional component matrices. |
| `coef_`, `intercept_`, `fitted_values_`, `residuals_` | Calibration-location regression output. |
| `parameter_standard_errors_`, `parameter_t_values_` | Local inference arrays. |
| `influence_`, `diagnostics_`, `sigma2_`, `hat_matrix_` | Smoother-based diagnostics. |

Use `get_results()` for calibration output. It includes coordinates, numeric time, prediction, coefficients, intercept, and residual. The fitted class does not expose a `summary()` method.

## Interpreting the three selected parameters

The parameters interact and should not be interpreted independently.

- A small spatial bandwidth with a large temporal bandwidth emphasizes local geography across a broader time window.
- A large spatial bandwidth with a small temporal bandwidth emphasizes recent observations across a broader region.
- A small alpha allows attribute similarity to create long-range and cross-time influence, subject to causal filtering.
- A large alpha approaches a conventional two-bandwidth spatiotemporal kernel.

A selected alpha near zero does not prove that space and time are irrelevant. It can reflect:

- strong attribute proxies for geography/time;
- leakage or post-outcome variables;
- candidate grids that inadequately represent space-time scales;
- in-sample overfitting through dense similarity links.

Inspect actual weight decomposition and validate forward in time.

## Recommended validation

1. Fit GWR, SGWR, and a pure space-time model under matched predictors.
2. Define similarity variables before looking at target performance.
3. Use `causal=True` for forecasting.
4. Fit on an earlier time window and predict a later window.
5. Add spatial blocking when deployment includes new regions.
6. Compare explicit `alpha=1`, selected alpha, and similarity-heavy settings.
7. Inspect component neighbour profiles and remote high-weight observations.
8. Repeat under candidate-grid refinements and alternative similarity sets.
9. Check local coefficient and residual stability, not only overall AICc.

## Common mistakes

| Mistake | Correction |
|---|---|
| Reporting a genetic algorithm | pyGWRx uses deterministic Cartesian AICc candidate search. |
| Using default `causal=False` for forecasting | Enable causal filtering and enforce time-ordered preprocessing/validation. |
| Assuming causal zeroing applies only to the space-time component | pyGWRx also zeros future rows after mixing, so similarity cannot reintroduce them. |
| Using target-derived or future attributes for similarity | Restrict similarity to features available at prediction time. |
| Interpreting alpha as explained-variance share | It mixes two weight matrices. |
| Changing time units without changing candidate bandwidths | Temporal bandwidth values are unit-dependent. |
| Launching the default 880-combination search without cost planning | Use coarse explicit grids, inspect history, then refine. |
| Calling `summary()` or `to_frame()` on the fitted model | Use `get_results()`; prediction result objects have `to_frame()`. |
| Setting `store_weights=False` and assuming all quadratic memory is removed | The full hat matrix is still retained. |
| Treating candidate AICc as forecasting evidence | Add forward temporal and spatial validation. |

## What to report

Report:

- response, predictors, and exact similarity-variable set;
- leakage controls and target-time feature availability;
- similarity standardization and attribute-distance formula;
- coordinate system, spatial distance metric, and spatial bandwidth semantics;
- time representation, origin, resolved unit, and temporal bandwidth;
- causal setting;
- complete candidate lists and number of combinations;
- deterministic AICc selection rather than genetic algorithm;
- selected spatial bandwidth, temporal bandwidth, alpha, and boundary behaviour;
- ridge and residual-variance convention;
- weight-storage and memory choices;
- component/combined neighbour examples;
- comparisons with GWR, SGWR, and pure space-time weighting;
- forward temporal and spatial validation design;
- pyGWRx version.

## References

- Li, M., Du, W., Yu, S., Hong, Z., Zhang, D., He, Y., & De, L. (2025). SGTWR Model with Spatial-Temporal Heterogeneity and Attribute Similarity for Urban Traffic Carbon Emission Driver Analysis. *Sustainability*, 17(23), 10773. [`10.3390/su172310773`](https://doi.org/10.3390/su172310773)
- Lessani, M. N., & Li, Z. (2024). SGWR: similarity and geographically weighted regression. *International Journal of Geographical Information Science*, 38(7), 1232–1255. [`10.1080/13658816.2024.2342319`](https://doi.org/10.1080/13658816.2024.2342319)

## Related documentation

- [Generated SGTWR API](../api/models/sgtwr.md)
- [SGWR](sgwr.md)
- [GTWR](gtwr.md)
- [MGTWR](mgtwr.md)
- [Spatiotemporal data](../guides/spatiotemporal-data.md)