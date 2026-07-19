# Geo-Regime Geographically Weighted Regression (`GRGWR`)

<div class="model-hero" markdown>

**Family:** Original research model
**Install:** `pip install -e ".[ml]"`
**Required data:** X, y, coordinates, regime count, and connectivity settings
**Primary operations:** fit, predict, predict_result
**New-location capability:** Validated using learned regime structure and target assignment logic.

</div>

[API reference](../api/models/gr-gwr.md){ .md-button .md-button--primary }
[Runnable source](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/models/19_gr_gwr.py){ .md-button }
[Choose a model](../getting-started/choosing-a-model.md){ .md-button }

## Why this model exists

Use GRGWR as a research model when contiguous spatial regimes and abrupt mechanism boundaries are scientifically more plausible than a completely smooth coefficient surface.

!!! note "One-sentence idea"
    GRGWR alternates between local/regime-conditioned regression and spatially constrained regime assignment, producing connected regions with internally coherent coefficient behaviour.

## Statistical formulation

A conceptual objective combines fit and boundary regularization,

$$
\mathcal L=\sum_i\left(y_i-x_i^\top\beta_{r_i}(s_i)\right)^2
+\lambda_B\sum_{(i,j)\in E}\mathbf 1(r_i\ne r_j),
$$

subject to regime-size and optional connectivity constraints. The implementation iteratively updates model parameters and regime labels.

## How pyGWRx fits the model

1. Construct a spatial-neighbour graph.
2. Initialize regime labels reproducibly.
3. Estimate regime-conditioned/local coefficient structures.
4. Reassign observations using fit and spatial-boundary costs.
5. Enforce minimum size and connectivity where configured.
6. Repeat until assignments or the objective stabilize, then expose regime-aware predictions.

## Constructor and important controls

```python
GRGWR(n_regimes: 'int' = 3, bandwidth: 'BandwidthLike' = 20, kernel: 'str' = 'bisquare', lambda_boundary: 'float' = 1.0, max_iter: 'int' = 10, tol: 'float' = 0.0001, spatial_constraint_weight: 'float' = 0.5, fit_intercept: 'bool' = True, verbose: 'bool' = False, *, n_neighbors: 'int' = 8, min_regime_size: 'Optional[int]' = None, enforce_connectivity: 'bool' = True, random_state: 'Optional[int]' = 42) -> 'None'
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

"""Fit geo-regime GWR and inspect connected spatial regimes."""

from pygwrx import GRGWR, GRGWRPredictionResult
from _common import print_model_result, regime_regression

X, y, coords, truth = regime_regression(n=56)
model = GRGWR(n_regimes=2, bandwidth=18, max_iter=2, random_state=0).fit(X, y, coords)
print_model_result(model)
print("regime_sizes=", model.regime_sizes_)
print(
    "truth_agreement_or_label_swap=",
    max((model.regimes_ == truth).mean(), (model.regimes_ != truth).mean()),
)
result = model.predict_result(X.iloc[:3], coords.iloc[:3])
assert isinstance(result, GRGWRPredictionResult)
print(result.to_frame())
```

Run it from the `examples/models` directory or through `python examples/run_all.py`.

## Reading the fitted result

**Main outputs:** `regimes_`, regime sizes, boundary information, convergence history, coefficient surfaces, diagnostics, predictions, result frames, and `GRGWRPredictionResult`.

Available high-level methods detected in the current class are: `fit()`, `predict()`, `predict_result()`, `summary()`, `to_frame()`.

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

Map regimes and boundaries, inspect regime sizes and connectivity, compare multiple random starts and regime counts, and evaluate whether boundaries are stable under perturbation.

The common diagnostics layer can be used where the fitted model provides the required fields:

```python
from pygwrx.diagnostics import diagnostics_frame, local_diagnostic_frame

print(diagnostics_frame([model], labels=["GRGWR"]))
try:
    print(local_diagnostic_frame(model).head())
except (AttributeError, NotImplementedError, ValueError) as exc:
    print("This model exposes a different diagnostic contract:", exc)
```

See [Diagnostics and inference](../guides/diagnostics.md) for model-aware checks and interpretation rules.

## Recommended visual checks


<div class="figure-grid" markdown>

<figure markdown>
  ![31 grgwr regimes](../assets/figures/specialized/31_grgwr_regimes.png){ loading=lazy }
  <figcaption>31 Grgwr Regimes</figcaption>
</figure>

<figure markdown>
  ![32 grgwr convergence](../assets/figures/specialized/32_grgwr_convergence.png){ loading=lazy }
  <figcaption>32 Grgwr Convergence</figcaption>
</figure>

<figure markdown>
  ![33 grgwr sizes](../assets/figures/specialized/33_grgwr_sizes.png){ loading=lazy }
  <figcaption>33 Grgwr Sizes</figcaption>
</figure>

<figure markdown>
  ![34 grgwr coefficient](../assets/figures/specialized/34_grgwr_coefficient.png){ loading=lazy }
  <figcaption>34 Grgwr Coefficient</figcaption>
</figure>

</div>


The figures are generated from deterministic examples and are illustrative; they are not benchmark claims.

## Common mistakes

- Treating estimated regimes as objectively true administrative regions.
- Selecting regime count solely by visual appeal.
- Ignoring disconnected or very small regimes when constraints are disabled.
- Using outcome-driven regimes without honest validation.

## What to report in a paper or technical report

- Regime count, initialization, graph construction, neighbours, and random seed.
- Boundary penalty, minimum size, and connectivity enforcement.
- Convergence and regime stability.
- Coefficient differences and boundary interpretation.
- Sensitivity across regime counts and starts.

## References

- [Brunsdon, Fotheringham & Charlton (1996), *Geographically Weighted Regression: A Method for Exploring Spatial Nonstationarity*](https://doi.org/10.1111/j.1538-4632.1996.tb00936.x)
- [Mei et al. (2004), *A Note on the Mixed Geographically Weighted Regression Model*](https://doi.org/10.1111/j.1085-9489.2004.00331.x)
- [Comber et al. (2022), *A route map for the informed application of GWR*](https://doi.org/10.1111/gean.12316)

## Related documentation

- [Detailed API for `GRGWR`](../api/models/gr-gwr.md)
- [Maintained example source](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/models/19_gr_gwr.py)
- [Kernels and bandwidths](../guides/kernels-and-bandwidths.md)
- [Prediction and result objects](../guides/prediction-and-results.md)
- [Complete Chinese model guide](../zh/models/gr-gwr.md)
