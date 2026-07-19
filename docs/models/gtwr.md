# Geographically and Temporally Weighted Regression (`GTWR`)

<div class="model-hero" markdown>

**Family:** Row-wise spatiotemporal regression
**Install:** `pip install -e ".[all]"`
**Required data:** X, y, coordinates, and row-wise times
**Primary operations:** fit, score, predict, predict_result
**New-location capability:** Validated at new space-time targets; causal filtering is available when configured.

</div>

[API reference](../api/models/gtwr.md){ .md-button .md-button--primary }
[Runnable source](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/models/05_gtwr.py){ .md-button }
[Choose a model](../getting-started/choosing-a-model.md){ .md-button }

## Why this model exists

Use GTWR when coefficients vary jointly over space and continuous or row-wise time, and neighbourhoods should be defined by a combined space-time distance.

!!! note "One-sentence idea"
    GTWR extends GWR by replacing purely spatial distance with a configurable space-time distance. The space-time balance therefore directly controls which observations become local neighbours.

## Statistical formulation

pyGWRx supports a GWmodel-style combination,

$$
d_{ij}^{ST}=\lambda d_{ij}^{S}+(1-\lambda)d_{ij}^{T}
+2\sqrt{\lambda(1-\lambda)d_{ij}^{S}d_{ij}^{T}}\cos(\xi),
$$

and a scaled Euclidean alternative. The resulting $d_{ij}^{ST}$ is passed to the selected kernel. With `causal=True`, observations later than the focal time receive zero weight.

## How pyGWRx fits the model

1. Normalize or convert the time vector according to `time_unit`.
2. Compute spatial and temporal distance components.
3. Apply or search the space-time balance parameter and kernel bandwidth.
4. Build local space-time weights, optionally excluding future observations.
5. Solve local regressions and calculate time-aware diagnostics.
6. For new targets, recompute space-time weights using their coordinates and times.

## Constructor and important controls

```python
GTWR(kernel: 'Union[str, Callable[[np.ndarray, float], np.ndarray]]' = 'bisquare', bandwidth: 'Union[float, int, str, None]' = 'cv', bandwidth_method: 'str' = 'cv', adaptive: 'bool' = False, bandwidth_range: 'Optional[Tuple[float, float]]' = None, lambda_st: 'Union[float, str]' = 0.05, lambda_range: 'Tuple[float, float]' = (0.0, 1.0), lambda_grid_size: 'int' = 11, ksi: 'float' = 0.0, distance_combination: 'str' = 'gwmodel', tau: 'float' = 1.0, causal: 'bool' = False, time_unit: 'str' = 'auto', optimization_method: 'str' = 'golden_section', search_grid_size: 'int' = 25, search_tol: 'float' = 1e-05, search_max_iter: 'int' = 100, fit_intercept: 'bool' = True, distance_metric: 'str' = 'euclidean', sigma2_v1: 'bool' = False verbose: 'bool' = False) -> 'None'
```

The API page documents every parameter and fitted attribute. In practice, start by deciding the **data contract**, **neighbourhood definition**, **selection criterion**, and **prediction/inference goal** before tuning secondary controls.

| Decision | Questions to answer |
|---|---|
| Data | Are rows independent observations, ordered stages, classes, counts, or multivariate features? |
| Distance | Are coordinates projected? Is time or contextual similarity part of the neighbourhood? |
| Bandwidth | Fixed distance or adaptive neighbours? Supplied value or selected criterion? |
| Inference | Are local uncertainty, non-stationarity tests, or only prediction required? |
| Validation | Does the split respect spatial and, where relevant, temporal dependence? |

## Complete runnable example

The following is the exact maintained example used by the API-coverage checks.

```python
# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Fit and predict with geographically and temporally weighted regression."""

from pygwrx import GTWR, GTWRPredictionResult
from _common import print_model_result, temporal_regression

X, y, coords, times = temporal_regression()
model = GTWR(kernel="bisquare", bandwidth=24, adaptive=True, lambda_st=0.3).fit(
    X, y, coords, times
)
print_model_result(model)
print("score=", model.score(X, y, coords, times=times))
result = model.predict_result(X.iloc[:3], coords.iloc[:3], times[:3])
assert isinstance(result, GTWRPredictionResult)
print(result.to_frame())
```

Run it from the `examples/models` directory or through `python examples/run_all.py`.

## Reading the fitted result

**Main outputs:** Space-time coefficient surfaces, `lambda_st_`/time-scaling state, fitted values, residuals, diagnostics, `to_frame()`, and `GTWRPredictionResult`.

Available high-level methods detected in the current class are: `fit()`, `score()`, `predict()`, `predict_result()`, `summary()`, `to_frame()`.

A safe inspection sequence is:

```python
# 1. Human-readable overview
print(model.summary()) if hasattr(model, "summary") else None

# 2. Location-indexed table when supported
frame = model.to_frame() if hasattr(model, "to_frame") else None

# 3. Explicitly inspect the model-specific state
print([name for name in vars(model) if name.endswith("_")])
```

Do not assume that every model exposes the same outputs. Regression, classification, transformation, descriptive-statistics, and inference models have different result semantics.

## Diagnostics and interpretation

Examine coefficient slices, temporal trajectories, temporal residual patterns, selected balance parameters, and leakage-safe validation splits.

The common diagnostics layer can be used where the fitted model provides the required fields:

```python
from pygwrx.diagnostics import diagnostics_frame, local_diagnostic_frame

print(diagnostics_frame([model], labels=["GTWR"]))
try:
    print(local_diagnostic_frame(model).head())
except (AttributeError, NotImplementedError, ValueError) as exc:
    print("This model exposes a different diagnostic contract:", exc)
```

See [Diagnostics and inference](../guides/diagnostics.md) for model-aware checks and interpretation rules.

## Recommended visual checks


<div class="figure-grid" markdown>

<figure markdown>
  ![18 gtwr slices](../assets/figures/specialized/18_gtwr_slices.png){ loading=lazy }
  <figcaption>18 Gtwr Slices</figcaption>
</figure>

<figure markdown>
  ![19 gtwr trajectory](../assets/figures/specialized/19_gtwr_trajectory.png){ loading=lazy }
  <figcaption>19 Gtwr Trajectory</figcaption>
</figure>

<figure markdown>
  ![20 gtwr residuals](../assets/figures/specialized/20_gtwr_residuals.png){ loading=lazy }
  <figcaption>20 Gtwr Residuals</figcaption>
</figure>

</div>


The figures are generated from deterministic examples and are illustrative; they are not benchmark claims.

## Common mistakes

- Mixing incompatible spatial and temporal units without checking scaling.
- Random train/test splits that leak future information.
- Leaving `causal=False` for a forecasting interpretation.
- Treating time as periodic without explicitly encoding periodicity.

## What to report in a paper or technical report

- Time representation and unit.
- Space-time distance formula, lambda/tau/ksi, and causal setting.
- Bandwidth-selection procedure.
- Temporal validation strategy.
- Coefficient and residual evolution through time.

## References

- [Huang, Wu & Barry (2010), *Geographically and temporally weighted regression for modeling spatio-temporal variation in house prices*](https://doi.org/10.1080/13658810802672469)

## Related documentation

- [Detailed API for `GTWR`](../api/models/gtwr.md)
- [Maintained example source](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/models/05_gtwr.py)
- [Kernels and bandwidths](../guides/kernels-and-bandwidths.md)
- [Prediction and result objects](../guides/prediction-and-results.md)
- [Complete Chinese model guide](../zh/models/gtwr.md)
