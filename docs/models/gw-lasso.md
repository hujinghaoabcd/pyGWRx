# Geographically Weighted Lasso (`GWLasso`)

<div class="model-hero" markdown>

**Family:** Locally regularized regression
**Install:** `pip install -e ".[ml]"`
**Required data:** X, y, coordinates
**Primary operations:** fit, score, predict
**New-location capability:** Validated local prediction with the learned local penalties and scaling state.

</div>

[API reference](../api/models/gw-lasso.md){ .md-button .md-button--primary }
[Runnable source](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/models/07_gw_lasso.py){ .md-button }
[Choose a model](../getting-started/choosing-a-model.md){ .md-button }

## Why this model exists

Use GWLasso when local models contain many or correlated candidate predictors and spatially varying variable selection is part of the research question.

!!! note "One-sentence idea"
    Every focal location solves a geographically weighted Lasso. Coefficients can shrink to zero locally, allowing the active predictor set to change over space.

## Statistical formulation

At focal location $s_i$, the objective is

$$
\min_{\beta_i}\;\frac{1}{2}\sum_j w_{ij}(y_j-x_j^\top\beta_i)^2
+\alpha_i\lVert\beta_i\rVert_1.
$$

The penalty may be supplied or selected. Standardization is important because the $L_1$ penalty is scale-sensitive.

## How pyGWRx fits the model

1. Validate and optionally standardize predictors.
2. Construct geographic weights at each location.
3. Use a supplied alpha or search an alpha grid with local/cross-validation logic.
4. Solve the weighted Lasso problem at every location.
5. Transform coefficients back to the original predictor scale.
6. Summarize active sets and selection frequencies and use the local model for prediction.

## Constructor and important controls

```python
GWLasso(kernel: 'Union[str, Callable]' = 'exponential', bandwidth: 'Union[float, int, str, None]' = 'cv', alpha: 'AlphaLike' = 'cv', alpha_grid: 'Optional[Sequence[float]]' = None, n_alphas: 'int' = 30, alpha_min_ratio: 'float' = 0.001, cv_folds: 'int' = 5, standardize: 'bool' = True, adaptive: 'bool' = False, bandwidth_range: 'Optional[Tuple[float, float]]' = None, n_bandwidths: 'int' = 8, max_iter: 'int' = 5000, tol: 'float' = 1e-06, active_tol: 'float' = 1e-08, fit_intercept: 'bool' = True, distance_metric: 'str' = 'euclidean' random_state: 'Optional[int]' = 0, verbose: 'bool' = False) -> 'None'
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

"""Fit geographically weighted Lasso with a fixed local penalty."""

from pygwrx import GWLasso
from _common import print_model_result, spatial_regression

X, y, coords = spatial_regression(n=48, p=3)
model = GWLasso(
    bandwidth=24, adaptive=True, alpha=0.06, max_iter=1000, random_state=0
).fit(X, y, coords)
print_model_result(model)
print("selection_frequency=", model.selection_frequency_)
print("predictions=", model.predict(X.iloc[:3], coords.iloc[:3]))
```

Run it from the `examples/models` directory or through `python examples/run_all.py`.

## Reading the fitted result

**Main outputs:** Local sparse coefficients, active masks, `selection_frequency_`, local or global alpha information, fitted values, predictions, and `to_frame()`.

Available high-level methods detected in the current class are: `fit()`, `score()`, `predict()`, `to_frame()`.

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

Map selection frequencies and active sets, inspect alpha sensitivity, and compare sparse surfaces with standard GWR. Stability across resamples or neighbouring bandwidths is more informative than one zero/non-zero map.

The common diagnostics layer can be used where the fitted model provides the required fields:

```python
from pygwrx.diagnostics import diagnostics_frame, local_diagnostic_frame

print(diagnostics_frame([model], labels=["GWLasso"]))
try:
    print(local_diagnostic_frame(model).head())
except (AttributeError, NotImplementedError, ValueError) as exc:
    print("This model exposes a different diagnostic contract:", exc)
```

See [Diagnostics and inference](../guides/diagnostics.md) for model-aware checks and interpretation rules.

## Recommended visual checks


<div class="figure-grid" markdown>

<figure markdown>
  ![04 gwlasso frequency](../assets/figures/specialized/04_gwlasso_frequency.png){ loading=lazy }
  <figcaption>04 Gwlasso Frequency</figcaption>
</figure>

<figure markdown>
  ![05 gwlasso active](../assets/figures/specialized/05_gwlasso_active.png){ loading=lazy }
  <figcaption>05 Gwlasso Active</figcaption>
</figure>

<figure markdown>
  ![06 gwlasso alpha](../assets/figures/specialized/06_gwlasso_alpha.png){ loading=lazy }
  <figcaption>06 Gwlasso Alpha</figcaption>
</figure>

</div>


The figures are generated from deterministic examples and are illustrative; they are not benchmark claims.

## Common mistakes

- Interpreting a zero coefficient as proof of no relationship.
- Using unstandardized predictors with very different units.
- Selecting alpha and bandwidth on the same data without acknowledging tuning optimism.
- Ignoring unstable active sets among correlated predictors.

## What to report in a paper or technical report

- Standardization and alpha-selection strategy.
- Bandwidth and kernel.
- Selection frequency and coefficient distribution.
- Stability or sensitivity analysis.
- Predictive comparison with non-regularized alternatives.

## References

- [Wheeler (2009), *The Geographically Weighted Lasso*](https://doi.org/10.1068/a40256)

## Related documentation

- [Detailed API for `GWLasso`](../api/models/gw-lasso.md)
- [Maintained example source](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/models/07_gw_lasso.py)
- [Kernels and bandwidths](../guides/kernels-and-bandwidths.md)
- [Prediction and result objects](../guides/prediction-and-results.md)
- [Complete Chinese model guide](../zh/models/gw-lasso.md)
