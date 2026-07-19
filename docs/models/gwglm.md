# Geographically Weighted Generalized Linear Models (`GWGLM`)

<div class="model-hero" markdown>

**Family:** Generalized local regression
**Install:** `pip install -e ".[all]"`
**Required data:** X, response, coordinates; optional exposure for Poisson
**Primary operations:** fit, score, predict, predict_result
**New-location capability:** Validated for Gaussian means, binomial probabilities, and Poisson means.

</div>

[API reference](../api/models/gwglm.md){ .md-button .md-button--primary }
[Runnable source](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/models/06_gwglm.py){ .md-button }
[Choose a model](../getting-started/choosing-a-model.md){ .md-button }

## Why this model exists

Use GWGLM when the response distribution is Gaussian, binary/binomial, or count-valued and its predictor effects may vary over space.

!!! note "One-sentence idea"
    GWGLM combines spatial kernels with generalized linear model links and variance functions. Each focal location maximizes a geographically weighted local likelihood using iterative weighted least squares.

## Statistical formulation

At focal location $s_i$,

$$
\eta_j(s_i)=x_j^\top\beta(s_i),\qquad \mu_j(s_i)=g^{-1}(\eta_j(s_i)),
$$

and the local log-likelihood is $\ell_i(\beta)=\sum_j w_{ij}\ell(y_j;\mu_j,\phi)$. Non-Gaussian families are solved with local IWLS using working responses and combined spatial/variance weights.

## How pyGWRx fits the model

1. Choose `gaussian`, `binomial`, or `poisson` and validate the response domain.
2. Build geographic kernel weights for each focal location.
3. Initialize the local linear predictor.
4. Iterate local working responses and working weights until convergence.
5. Calculate family-appropriate likelihood, deviance, residuals, and information criteria.
6. Predict on the response-mean scale; supply exposure where required.

## Constructor and important controls

```python
GWGLM(family: 'FamilyName' = 'gaussian', kernel: 'KernelLike' = 'bisquare', bandwidth: 'BandwidthLike' = 'cv', bandwidth_method: 'str' = 'aicc', adaptive: 'bool' = False, bandwidth_range: 'Optional[Tuple[float, float]]' = None, optimization_method: 'str' = 'golden_section', max_iter: 'int' = 100, tol: 'float' = 1e-06, fit_intercept: 'bool' = True, distance_metric: 'str' = 'euclidean', sigma2_v1: 'bool' = True verbose: 'bool' = False) -> 'None'
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

"""Fit Gaussian, binomial, and Poisson GWGLM families."""

import numpy as np
from pygwrx import GWGLM, GWGLMPredictionResult
from _common import count_regression, print_model_result, spatial_regression

X, y, coords = spatial_regression(p=2)
gaussian = GWGLM(family="gaussian", bandwidth=24, adaptive=True).fit(X, y, coords)
print_model_result(gaussian)

binary = (y > np.median(y)).astype(int)
binomial = GWGLM(family="binomial", bandwidth=24, adaptive=True).fit(X, binary, coords)
binomial_result = binomial.predict_result(X.iloc[:3], coords.iloc[:3])
assert isinstance(binomial_result, GWGLMPredictionResult)
print(binomial_result.to_frame())

Xc, counts, coordsc, exposure = count_regression()
poisson = GWGLM(family="poisson", bandwidth=24, adaptive=True).fit(
    Xc, counts, coordsc, exposure=exposure
)
print(
    "poisson means=",
    poisson.predict(Xc.iloc[:3], coordsc.iloc[:3], exposure=exposure[:3]),
)
```

Run it from the `examples/models` directory or through `python examples/run_all.py`.

## Reading the fitted result

**Main outputs:** Local coefficients, mean-scale fitted values, family-specific residuals and deviance, convergence information, diagnostics, result frames, and `GWGLMPredictionResult`.

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

Use deviance and Pearson residuals for non-Gaussian families, check local convergence, inspect extreme fitted probabilities or means, and compare against a global GLM.

The common diagnostics layer can be used where the fitted model provides the required fields:

```python
from pygwrx.diagnostics import diagnostics_frame, local_diagnostic_frame

print(diagnostics_frame([model], labels=["GWGLM"]))
try:
    print(local_diagnostic_frame(model).head())
except (AttributeError, NotImplementedError, ValueError) as exc:
    print("This model exposes a different diagnostic contract:", exc)
```

See [Diagnostics and inference](../guides/diagnostics.md) for model-aware checks and interpretation rules.

## Recommended visual checks


<div class="figure-grid" markdown>

<figure markdown>
  ![03 gwglm residuals](../assets/figures/specialized/03_gwglm_residuals.png){ loading=lazy }
  <figcaption>03 Gwglm Residuals</figcaption>
</figure>

</div>


The figures are generated from deterministic examples and are illustrative; they are not benchmark claims.

## Common mistakes

- Treating binomial probabilities as unconstrained linear predictions.
- Omitting or inconsistently applying Poisson exposure.
- Using Gaussian residual diagnostics without considering the selected family.
- Ignoring separation, sparse local events, or local non-convergence.

## What to report in a paper or technical report

- Family, link, and response coding.
- Exposure or offset definition.
- Bandwidth criterion and local IWLS tolerance.
- Deviance, information criteria, convergence, and residual diagnostics.
- Prediction scale used in interpretation.

## References

- [Nakaya et al. (2005), *Geographically weighted Poisson regression for disease association mapping*](https://doi.org/10.1002/sim.2129)
- [Gollini et al. (2015), *GWmodel: an R Package for Exploring Spatial Heterogeneity*](https://doi.org/10.18637/jss.v063.i17)

## Related documentation

- [Detailed API for `GWGLM`](../api/models/gwglm.md)
- [Maintained example source](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/models/06_gwglm.py)
- [Kernels and bandwidths](../guides/kernels-and-bandwidths.md)
- [Prediction and result objects](../guides/prediction-and-results.md)
- [Complete Chinese model guide](../zh/models/gwglm.md)
