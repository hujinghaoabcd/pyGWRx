# Mixed Geographically Weighted Regression (`MixedGWR`)

<div class="model-hero" markdown>

**Family:** Semiparametric global-local regression
**Install:** `pip install -e ".[all]"`
**Required data:** X, y, coordinates, and global/local variable assignments
**Primary operations:** fit, score, predict
**New-location capability:** Validated using global coefficients and re-estimated local components.

</div>

[API reference](../api/models/mixed-gwr.md){ .md-button .md-button--primary }
[Runnable source](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/models/08_mixed_gwr.py){ .md-button }
[Choose a model](../getting-started/choosing-a-model.md){ .md-button }

## Why this model exists

Use MixedGWR when theory indicates that some effects should be spatially constant while others should vary locally.

!!! note "One-sentence idea"
    MixedGWR decomposes the linear predictor into global coefficients estimated from all observations and local coefficients estimated with geographical weights.

## Statistical formulation

A semiparametric form is

$$
y_i=X_{i,G}\gamma+X_{i,L}\beta(s_i)+\varepsilon_i,
$$

where $\gamma$ is global and $\beta(s_i)$ varies spatially. Estimation alternates between the global and local components until the combined fit stabilizes.

## How pyGWRx fits the model

1. Resolve variable names or indices into disjoint global and local sets.
2. Initialize the global and local components.
3. Estimate local coefficients conditional on the current global component.
4. Update global coefficients conditional on the current local component.
5. Iterate to convergence and compute combined fitted values and diagnostics.
6. Predict using the global component plus local calibration at target coordinates.

## Constructor and important controls

```python
MixedGWR(kernel: 'Union[str, Callable]' = 'bisquare', bandwidth: 'Union[float, int, str, None]' = 'aicc', bandwidth_method: 'str' = 'aicc', adaptive: 'bool' = True, local_vars: 'VariableSpec' = None, global_vars: 'VariableSpec' = None, intercept_fixed: 'bool' = True, ridge: 'float' = 0.0, fit_intercept: 'bool' = True, bandwidth_range: 'Optional[tuple[float, float]]' = None, optimization_method: 'str' = 'golden_section', distance_metric: 'str' = 'euclidean', verbose: 'bool' = False) -> 'None'
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

"""Fit a semiparametric Mixed GWR with global and local predictors."""

from pygwrx import MixedGWR
from _common import mixed_regression, print_model_result

X, y, coords = mixed_regression()
model = MixedGWR(
    bandwidth=28,
    adaptive=True,
    global_vars=["global_x"],
    local_vars=["local_x"],
    intercept_fixed=True,
).fit(X, y, coords, compute_enp=False)
print_model_result(model)
print("global_coefficients=", model.coef_global_)
print("predictions=", model.predict(X.iloc[:3], coords.iloc[:3]))
```

Run it from the `examples/models` directory or through `python examples/run_all.py`.

## Reading the fitted result

**Main outputs:** `coef_global_`, local coefficient surfaces, combined fitted values and residuals, diagnostics, predictions, and `to_frame()`.

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

Compare global estimates with a standard global regression, map only the local terms, and assess whether forcing a term global materially worsens fit or residual structure.

The common diagnostics layer can be used where the fitted model provides the required fields:

```python
from pygwrx.diagnostics import diagnostics_frame, local_diagnostic_frame

print(diagnostics_frame([model], labels=["MixedGWR"]))
try:
    print(local_diagnostic_frame(model).head())
except (AttributeError, NotImplementedError, ValueError) as exc:
    print("This model exposes a different diagnostic contract:", exc)
```

See [Diagnostics and inference](../guides/diagnostics.md) for model-aware checks and interpretation rules.

## Recommended visual checks


<div class="figure-grid" markdown>

<figure markdown>
  ![07 mixed coefficients](../assets/figures/specialized/07_mixed_coefficients.png){ loading=lazy }
  <figcaption>07 Mixed Coefficients</figcaption>
</figure>

</div>


The figures are generated from deterministic examples and are illustrative; they are not benchmark claims.

## Common mistakes

- Assigning the same variable to both global and local sets.
- Treating the global/local choice as purely data-driven when it should reflect theory and validation.
- Ignoring intercept semantics when `intercept_fixed` changes.

## What to report in a paper or technical report

- Global and local variable sets and rationale.
- Intercept treatment.
- Bandwidth and convergence settings.
- Global estimates with uncertainty and local surface summaries.
- Comparison with all-global and all-local alternatives.

## References

- [Mei et al. (2004), *A Note on the Mixed Geographically Weighted Regression Model*](https://doi.org/10.1111/j.1085-9489.2004.00331.x)
- [Comber et al. (2022), *A route map for the informed application of GWR*](https://doi.org/10.1111/gean.12316)

## Related documentation

- [Detailed API for `MixedGWR`](../api/models/mixed-gwr.md)
- [Maintained example source](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/models/08_mixed_gwr.py)
- [Kernels and bandwidths](../guides/kernels-and-bandwidths.md)
- [Prediction and result objects](../guides/prediction-and-results.md)
- [Complete Chinese model guide](../zh/models/mixed-gwr.md)
