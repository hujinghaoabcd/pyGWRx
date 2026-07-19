# Geographically Weighted Summary Statistics (`GWSS`)

<div class="model-hero" markdown>

**Family:** Local descriptive statistics
**Install:** `pip install -e ".[all]"`
**Required data:** Multivariate X and coordinates
**Primary operations:** fit, select_bandwidth, summary
**New-location capability:** Not applicable; this is a local-statistics estimator.

</div>

[API reference](../api/models/gwss.md){ .md-button .md-button--primary }
[Runnable source](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/models/11_gwss.py){ .md-button }
[Choose a model](../getting-started/choosing-a-model.md){ .md-button }

## Why this model exists

Use GWSS before modelling to examine how means, dispersion, covariance, correlation, and optional quantiles vary across space.

!!! note "One-sentence idea"
    GWSS applies the same geographic weighting principle as GWR, but computes descriptive statistics instead of response-model coefficients.

## Statistical formulation

A local weighted mean is

$$
\bar x_i=\frac{\sum_jw_{ij}x_j}{\sum_jw_{ij}},
$$

and local covariance/correlation follow from the weighted centered products. With quantile mode, weighted local distribution summaries are also computed.

## How pyGWRx fits the model

1. Validate numeric variables and coordinates.
2. Apply or select a bandwidth.
3. Build local geographic weights.
4. Compute local means, variances, standard deviations, covariance, and correlation.
5. Optionally compute weighted quantiles.
6. Export location-indexed statistics for mapping or downstream model design.

## Constructor and important controls

```python
GWSS(kernel: 'str | Any' = 'bisquare', bandwidth: 'float | int | None' = None, adaptive: 'bool' = False, quantile: 'bool' = False, verbose: 'bool' = False) -> 'None'
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

"""Compute geographically weighted summary statistics."""

from pygwrx import GWSS
from _common import spatial_regression

X, _, coords = spatial_regression(n=48, p=3)
model = GWSS(bandwidth=24, adaptive=True, quantile=True).fit(X, coords)
print(model.summary())
print("local_means_shape=", model.local_mean_.shape)
print("local_correlation_pairs=", sorted(model.local_corr_))
print("first_correlation_shape=", next(iter(model.local_corr_.values())).shape)
```

Run it from the `examples/models` directory or through `python examples/run_all.py`.

## Reading the fitted result

**Main outputs:** `local_mean_`, local variance/standard deviation, local covariance and correlation collections, optional quantiles, summaries, and tabular outputs.

Available high-level methods detected in the current class are: `fit()`, `select_bandwidth()`, `summary()`.

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

Treat GWSS as exploratory. Compare bandwidths, inspect effective local sample sizes, and avoid over-interpreting noisy local correlation where variation is low.

The common diagnostics layer can be used where the fitted model provides the required fields:

```python
from pygwrx.diagnostics import diagnostics_frame, local_diagnostic_frame

print(diagnostics_frame([model], labels=["GWSS"]))
try:
    print(local_diagnostic_frame(model).head())
except (AttributeError, NotImplementedError, ValueError) as exc:
    print("This model exposes a different diagnostic contract:", exc)
```

See [Diagnostics and inference](../guides/diagnostics.md) for model-aware checks and interpretation rules.

## Recommended visual checks


<div class="figure-grid" markdown>

<figure markdown>
  ![11 gwss mean](../assets/figures/specialized/11_gwss_mean.png){ loading=lazy }
  <figcaption>11 Gwss Mean</figcaption>
</figure>

<figure markdown>
  ![12 gwss correlation](../assets/figures/specialized/12_gwss_correlation.png){ loading=lazy }
  <figcaption>12 Gwss Correlation</figcaption>
</figure>

</div>


The figures are generated from deterministic examples and are illustrative; they are not benchmark claims.

## Common mistakes

- Using local correlations as causal evidence.
- Ignoring that local statistics are smoothed and overlapping.
- Computing quantiles with too few effective observations.
- Selecting variables after inspecting many local maps without multiplicity awareness.

## What to report in a paper or technical report

- Variables and transformations.
- Kernel and bandwidth.
- Whether quantiles were calculated.
- Effective neighbourhood size and sensitivity.
- Which local statistics were mapped and why.

## References

- [Brunsdon, Fotheringham & Charlton (2002), *Geographically weighted summary statistics*](https://doi.org/10.1016/S0198-9715(01)00009-6)

## Related documentation

- [Detailed API for `GWSS`](../api/models/gwss.md)
- [Maintained example source](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/models/11_gwss.py)
- [Kernels and bandwidths](../guides/kernels-and-bandwidths.md)
- [Prediction and result objects](../guides/prediction-and-results.md)
- [Complete Chinese model guide](../zh/models/gwss.md)
