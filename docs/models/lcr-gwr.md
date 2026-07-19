# Locally Compensated Ridge GWR (`LCRGWR`)

<div class="model-hero" markdown>

**Family:** Collinearity-compensated local regression
**Install:** `pip install -e ".[all]"`
**Required data:** X, y, coordinates
**Primary operations:** fit, score, predict, predict_result
**New-location capability:** Validated local prediction with fitted or locally adjusted ridge terms.

</div>

[API reference](../api/models/lcr-gwr.md){ .md-button .md-button--primary }
[Runnable source](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/models/13_lcr_gwr.py){ .md-button }
[Choose a model](../getting-started/choosing-a-model.md){ .md-button }

## Why this model exists

Use LCRGWR when local—not merely global—collinearity makes standard GWR coefficient surfaces unstable.

!!! note "One-sentence idea"
    LCRGWR measures local condition numbers and adds the minimum location-specific ridge penalty needed to reduce excessive local ill-conditioning.

## Statistical formulation

If the local cross-product eigenvalues are $d_{i,\max}$ and $d_{i,\min}$ and the target condition number is $\kappa^*$, the compensating penalty is

$$
\lambda_i=\max\left\{0,\frac{d_{i,\max}-\kappa^*d_{i,\min}}{\kappa^*-1}\right\}.
$$

The local estimator becomes $(X^\top W_iX+\lambda_iP)^{-1}X^\top W_iy$, usually leaving the intercept unpenalized.

## How pyGWRx fits the model

1. Compute each local weighted design matrix.
2. Evaluate local condition numbers and related collinearity measures.
3. Compare the condition number with `cn_thresh`.
4. Apply a fixed ridge penalty or calculate the minimal local compensation.
5. Refit local coefficients with the ridge-adjusted system.
6. Compare compensated and uncompensated surfaces and predictions.

## Constructor and important controls

```python
LCRGWR(kernel: 'Union[str, Callable[[np.ndarray, float], np.ndarray]]' = 'bisquare', bandwidth: 'Union[float, int, str, None]' = 'cv', bandwidth_method: 'str' = 'cv', adaptive: 'bool' = False, bandwidth_range: 'Optional[Tuple[float, float]]' = None, optimization_method: 'str' = 'golden_section', lambda_ridge: 'float' = 0.0, lambda_adjust: 'bool' = True, cn_thresh: 'float' = 30.0, fit_intercept: 'bool' = True, distance_metric: 'str' = 'euclidean', sigma2_v1: 'bool' = True verbose: 'bool' = False) -> 'None'
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

"""Fit locally compensated ridge GWR for collinear predictors."""

from pygwrx import LCRGWR
from _common import collinear_regression, print_model_result

X, y, coords = collinear_regression()
model = LCRGWR(bandwidth=28, adaptive=True, cn_thresh=15.0, lambda_adjust=True).fit(
    X, y, coords
)
print_model_result(model)
print("local_condition_numbers=", model.local_condition_numbers_[:5])
print("local_lambdas=", model.local_lambdas_[:5])
```

Run it from the `examples/models` directory or through `python examples/run_all.py`.

## Reading the fitted result

**Main outputs:** GWR-style outputs plus local condition numbers, `local_lambdas_`, compensated coefficient surfaces, diagnostics, predictions, and result tables.

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

Map condition numbers and local lambdas together. Coefficient stabilization should be evaluated alongside predictive performance and substantive plausibility.

The common diagnostics layer can be used where the fitted model provides the required fields:

```python
from pygwrx.diagnostics import diagnostics_frame, local_diagnostic_frame

print(diagnostics_frame([model], labels=["LCRGWR"]))
try:
    print(local_diagnostic_frame(model).head())
except (AttributeError, NotImplementedError, ValueError) as exc:
    print("This model exposes a different diagnostic contract:", exc)
```

See [Diagnostics and inference](../guides/diagnostics.md) for model-aware checks and interpretation rules.

## Recommended visual checks


<div class="figure-grid" markdown>

<figure markdown>
  ![07 gwr condition number](../assets/figures/core/07_gwr_condition_number.png){ loading=lazy }
  <figcaption>07 Gwr Condition Number</figcaption>
</figure>

<figure markdown>
  ![08 lcr lambda](../assets/figures/core/08_lcr_lambda.png){ loading=lazy }
  <figcaption>08 Lcr Lambda</figcaption>
</figure>

</div>


The figures are generated from deterministic examples and are illustrative; they are not benchmark claims.

## Common mistakes

- Using ridge compensation as a substitute for understanding redundant predictors.
- Choosing an arbitrary threshold without sensitivity analysis.
- Comparing penalized coefficient magnitude directly with unpenalized estimates.
- Ignoring whether only a small region requires compensation.

## What to report in a paper or technical report

- Condition-number definition and threshold.
- Fixed versus adaptive lambda strategy.
- Spatial distribution of local lambdas.
- Coefficient and prediction sensitivity.
- Comparison with variable removal or MGWR alternatives.

## References

- [Wheeler (2007), *Diagnostic Tools and a Remedial Method for Collinearity in GWR*](https://doi.org/10.1068/a38325)
- [Gollini et al. (2015), *GWmodel: an R Package for Exploring Spatial Heterogeneity*](https://doi.org/10.18637/jss.v063.i17)

## Related documentation

- [Detailed API for `LCRGWR`](../api/models/lcr-gwr.md)
- [Maintained example source](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/models/13_lcr_gwr.py)
- [Kernels and bandwidths](../guides/kernels-and-bandwidths.md)
- [Prediction and result objects](../guides/prediction-and-results.md)
- [Complete Chinese model guide](../zh/models/lcr-gwr.md)
