# Multiscale Geographically Weighted Regression (`MGWR`)

<div class="model-hero" markdown>

**Family:** Multiscale local regression
**Install:** `pip install -e ".[all]"`
**Required data:** X, y, coordinates
**Primary operations:** fit, score, calibration-location results
**New-location capability:** Independent-target prediction is intentionally unavailable in the current validated API.

</div>

[API reference](../api/models/mgwr.md){ .md-button .md-button--primary }
[Runnable source](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/models/02_mgwr.py){ .md-button }
[Choose a model](../getting-started/choosing-a-model.md){ .md-button }

## Why this model exists

Use MGWR when predictors plausibly operate at different spatial scales and a single shared GWR bandwidth would hide those differences.

!!! note "One-sentence idea"
    MGWR gives every design column—including the intercept when fitted—its own spatial bandwidth. Large bandwidths indicate broad or near-global processes; small bandwidths indicate more local variation.

## Statistical formulation

The model remains additive,

$$
y_i=\sum_{k=0}^{p} x_{ik}\beta_k(s_i)+\varepsilon_i,
$$

but coefficient $k$ is updated with its own weight matrix $W_{ik}(h_k)$. Backfitting cycles through partial residuals, searches or applies $h_k$, and updates one coefficient surface at a time until the surfaces and residual sum of squares stabilize.

## How pyGWRx fits the model

1. Initialize coefficients and variable-specific bandwidths, usually from a GWR-scale solution.
2. Construct the partial residual for one design column.
3. Search or apply that column’s bandwidth.
4. Update the local coefficient surface for the selected column.
5. Repeat across columns and backfitting iterations until convergence.
6. Combine component smoothers to obtain diagnostics and optional inference.

## Constructor and important controls

```python
MGWR(kernel: 'Union[str, Callable[[np.ndarray, float], np.ndarray]]' = 'bisquare', bandwidths: 'BandwidthInput' = None, bandwidth_method: 'str' = 'aicc', adaptive: 'bool' = True, bandwidth_range: 'BandwidthRange' = None, bandwidth_ranges: 'BandwidthRanges' = None, init_bandwidth: 'Optional[Bandwidth]' = None, optimization_method: 'str' = 'golden_section', search_tol: 'float' = 1e-06, search_max_iter: 'int' = 200, max_iter: 'int' = 200, tol: 'float' = 1e-05, rss_score: 'bool' = False, bws_same_times: 'int' = 5, fit_intercept: 'bool' = True, distance_metric: 'str' = 'euclidean', sigma2_v1: 'bool' = True verbose: 'bool' = False) -> 'None'
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

"""Fit MGWR with fixed variable-specific bandwidths."""

from pygwrx import MGWR
from _common import print_model_result, spatial_regression

X, y, coords = spatial_regression(n=48, p=2)
model = MGWR(bandwidths=[24, 26, 28], adaptive=True, max_iter=8, tol=0.5).fit(
    X, y, coords, compute_inference=True
)
print_model_result(model)
try:
    model.predict(X.iloc[:2], coords.iloc[:2])
except NotImplementedError as exc:
    print("Expected MGWR prediction limitation:", exc)
```

Run it from the `examples/models` directory or through `python examples/run_all.py`.

## Reading the fitted result

**Main outputs:** `bandwidths_`, variable-specific coefficient surfaces, convergence information, fitted values, residuals, diagnostics, optional inference, and `to_frame()`.

Available high-level methods detected in the current class are: `fit()`, `score()`, `predict()`, `summary()`, `to_frame()`.

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

Interpret bandwidths together with coefficient surfaces. Check whether any bandwidth sticks to a search boundary, whether backfitting converged, and whether highly correlated predictors produce unstable scale estimates.

The common diagnostics layer can be used where the fitted model provides the required fields:

```python
from pygwrx.diagnostics import diagnostics_frame, local_diagnostic_frame

print(diagnostics_frame([model], labels=["MGWR"]))
try:
    print(local_diagnostic_frame(model).head())
except (AttributeError, NotImplementedError, ValueError) as exc:
    print("This model exposes a different diagnostic contract:", exc)
```

See [Diagnostics and inference](../guides/diagnostics.md) for model-aware checks and interpretation rules.

## Recommended visual checks


<div class="figure-grid" markdown>

<figure markdown>
  ![09 mgwr bandwidths](../assets/figures/core/09_mgwr_bandwidths.png){ loading=lazy }
  <figcaption>09 Mgwr Bandwidths</figcaption>
</figure>

<figure markdown>
  ![11 gwr mgwr comparison](../assets/figures/core/11_gwr_mgwr_comparison.png){ loading=lazy }
  <figcaption>11 Gwr Mgwr Comparison</figcaption>
</figure>

</div>


The figures are generated from deterministic examples and are illustrative; they are not benchmark claims.

## Common mistakes

- Reading a bandwidth as an exact physical process scale without sensitivity analysis.
- Using narrow search bounds that force boundary solutions.
- Assuming `predict()` supports independent new targets; the current method rejects that operation.
- Comparing coefficient surfaces without accounting for their different effective scales.

## What to report in a paper or technical report

- Each coefficient’s bandwidth and unit.
- Initialization, search ranges, convergence tolerance, and iterations.
- AICc/CV and effective parameter count.
- Whether inference was computed and how uncertainty was handled.
- Why multiscale structure is preferable to standard GWR.

## References

- [Fotheringham, Yang & Kang (2017), *Multiscale Geographically Weighted Regression (MGWR)*](https://doi.org/10.1080/24694452.2017.1352480)
- [Comber et al. (2022), *A route map for the informed application of GWR*](https://doi.org/10.1111/gean.12316)

## Related documentation

- [Detailed API for `MGWR`](../api/models/mgwr.md)
- [Maintained example source](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/models/02_mgwr.py)
- [Kernels and bandwidths](../guides/kernels-and-bandwidths.md)
- [Prediction and result objects](../guides/prediction-and-results.md)
- [Complete Chinese model guide](../zh/models/mgwr.md)
