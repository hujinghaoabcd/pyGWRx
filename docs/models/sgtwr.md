# Similarity Geographically and Temporally Weighted Regression (`SGTWR`)

<div class="model-hero" markdown>

**Family:** Geography-time-similarity regression
**Install:** `pip install -e ".[all]"`
**Required data:** X, y, coordinates, times, and similarity variables
**Primary operations:** fit, predict, predict_result
**New-location capability:** Validated at target space-time points with optional causal filtering.

</div>

[API reference](../api/models/sgtwr.md){ .md-button .md-button--primary }
[Runnable source](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/models/16_sgtwr.py){ .md-button }
[Choose a model](../getting-started/choosing-a-model.md){ .md-button }

## Why this model exists

Use SGTWR when geographic proximity, temporal proximity, and contextual similarity all define relevant neighbours.

!!! note "One-sentence idea"
    SGTWR maintains separate space-time and attribute-similarity components and combines them with alpha instead of forcing all notions of proximity into one distance.

## Statistical formulation

A representative space-time component is

$$
w_{ij}^{ST}=\exp\left[-\frac12\left((d_{ij}^{S}/h_i^S)^2+(d_{ij}^{T}/h^T)^2\right)\right],
$$

combined with an attribute-similarity weight $w_{ij}^{A}$ as $w_{ij}=\alpha w_{ij}^{ST}+(1-\alpha)w_{ij}^{A}$. Spatial bandwidth, temporal bandwidth, and alpha are separately configurable or selectable.

## How pyGWRx fits the model

1. Normalize time and selected similarity attributes.
2. Construct candidate spatial bandwidths, temporal bandwidths, and alpha values when selecting.
3. Compute space-time and attribute-similarity weights separately.
4. Apply causal filtering when the analysis is predictive.
5. Combine weights and solve local regressions.
6. Inspect parameter selection and weight decomposition.

## Constructor and important controls

```python
SGTWR(spatial_bandwidth: 'SelectionValue' = 'aicc', *, temporal_bandwidth: 'SelectionValue' = 'aicc', adaptive: 'bool' = True, alpha: 'SelectionValue' = 'aicc', similarity_vars: 'Optional[Sequence[Union[int, str]]]' = None, standardize_similarity: 'bool' = True, spatial_bandwidth_candidates: 'Optional[Sequence[Number]]' = None, temporal_bandwidth_candidates: 'Optional[Sequence[Number]]' = None, alpha_candidates: 'Optional[Sequence[Number]]' = None, causal: 'bool' = False, time_unit: 'str' = 'auto', fit_intercept: 'bool' = True, distance_metric: 'str' = 'euclidean', sigma2_v1: 'bool' = True, ridge: 'float' = 0.0, store_weights: 'bool' = True, verbose: 'bool' = False) -> 'None'
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

"""Fit similarity and geographically-temporally weighted regression."""

from pygwrx import SGTWR, SGTWRPredictionResult
from _common import print_model_result, temporal_regression

X, y, coords, times = temporal_regression(n=48, p=3)
model = SGTWR(
    spatial_bandwidth=24,
    temporal_bandwidth=2.0,
    adaptive=True,
    alpha=0.5,
    similarity_vars=["x1", "x2"],
    store_weights=True,
).fit(X, y, coords, times)
print_model_result(model)
print("combined_weights_shape=", model.combined_weights_.shape)
result = model.predict_result(X.iloc[:3], coords.iloc[:3], times[:3])
assert isinstance(result, SGTWRPredictionResult)
print(result.to_frame())
```

Run it from the `examples/models` directory or through `python examples/run_all.py`.

## Reading the fitted result

**Main outputs:** Space-time local coefficients, selected spatial/temporal bandwidths and alpha, stored weight components, fitted values, predictions, and `SGTWRPredictionResult`.

Available high-level methods detected in the current class are: `fit()`, `predict()`, `predict_result()`.

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

Plot spatial and temporal scales, decompose weights, inspect temporal trajectories, and evaluate forecasting performance with forward or rolling splits.

The common diagnostics layer can be used where the fitted model provides the required fields:

```python
from pygwrx.diagnostics import diagnostics_frame, local_diagnostic_frame

print(diagnostics_frame([model], labels=["SGTWR"]))
try:
    print(local_diagnostic_frame(model).head())
except (AttributeError, NotImplementedError, ValueError) as exc:
    print("This model exposes a different diagnostic contract:", exc)
```

See [Diagnostics and inference](../guides/diagnostics.md) for model-aware checks and interpretation rules.

## Recommended visual checks


<div class="figure-grid" markdown>

<figure markdown>
  ![22 sgtwr scales](../assets/figures/specialized/22_sgtwr_scales.png){ loading=lazy }
  <figcaption>22 Sgtwr Scales</figcaption>
</figure>

<figure markdown>
  ![26 sgtwr weights](../assets/figures/specialized/26_sgtwr_weights.png){ loading=lazy }
  <figcaption>26 Sgtwr Weights</figcaption>
</figure>

</div>


The figures are generated from deterministic examples and are illustrative; they are not benchmark claims.

## Common mistakes

- Outcome leakage through similarity variables.
- Non-causal use of future observations.
- Large candidate grids with unreported computational cost.
- Interpreting three smoothing parameters independently when they interact.

## What to report in a paper or technical report

- Spatial and temporal units.
- Similarity variables and standardization.
- Spatial/temporal bandwidths, alpha, candidates, and causal setting.
- Weight decomposition and temporal validation.
- Sensitivity to each neighbourhood component.

## References

- [Li et al. (2025), *SGTWR Model with Spatial-Temporal Heterogeneity and Attribute Similarity*](https://doi.org/10.3390/su172310773)
- [Lessani & Li (2024), *SGWR: similarity and geographically weighted regression*](https://doi.org/10.1080/13658816.2024.2342319)
- [Huang, Wu & Barry (2010), *Geographically and temporally weighted regression for modeling spatio-temporal variation in house prices*](https://doi.org/10.1080/13658810802672469)

## Related documentation

- [Detailed API for `SGTWR`](../api/models/sgtwr.md)
- [Maintained example source](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/models/16_sgtwr.py)
- [Kernels and bandwidths](../guides/kernels-and-bandwidths.md)
- [Prediction and result objects](../guides/prediction-and-results.md)
- [Complete Chinese model guide](../zh/models/sgtwr.md)
