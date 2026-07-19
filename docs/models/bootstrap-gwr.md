# Bootstrap Tests for GWR Non-stationarity (`BootstrapGWR`)

<div class="model-hero" markdown>

**Family:** Spatial inference
**Install:** `pip install -e ".[all]"`
**Required data:** X, y, coordinates
**Primary operations:** fit, summary, to_frame
**New-location capability:** Not applicable; the estimator performs coefficient-variability inference.

</div>

[API reference](../api/models/bootstrap-gwr.md){ .md-button .md-button--primary }
[Runnable source](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/models/14_bootstrap_gwr.py){ .md-button }
[Choose a model](../getting-started/choosing-a-model.md){ .md-button }

## Why this model exists

Use BootstrapGWR when the main question is whether apparent local coefficient variation exceeds what would be expected under a specified null model.

!!! note "One-sentence idea"
    BootstrapGWR repeatedly simulates or resamples under a null structure, refits GWR, and compares observed coefficient variability with the bootstrap distribution.

## Statistical formulation

For coefficient $k$, an observed spatial-variability statistic $T_k$ is compared with bootstrap replicates $T_k^{*(b)}$. A plus-one Monte Carlo p-value is

$$
p_k=\frac{1+\sum_{b=1}^{B}\mathbf 1(T_k^{*(b)}\ge T_k)}{B+1}.
$$

Optional localized tests compare observed local deviations with location-wise bootstrap distributions.

## How pyGWRx fits the model

1. Fit the observed GWR and calculate coefficient-variability statistics.
2. Use the validated OLS null model and specify the bootstrap count.
3. Generate bootstrap responses under the null.
4. Refit GWR for every replicate, optionally reselecting the bandwidth.
5. Calculate global and optional localized p-values.
6. Summarize evidence for coefficient non-stationarity and retain bootstrap distributions when requested.

## Constructor and important controls

```python
BootstrapGWR(bandwidth: 'Union[float, int, str, None]' = 'aicc', adaptive: 'bool' = False, kernel: 'str' = 'bisquare', bandwidth_method: 'str' = 'aicc', bandwidth_range: 'Optional[Tuple[float, float]]' = None, optimization_method: 'str' = 'golden_section', fit_intercept: 'bool' = True, distance_metric: 'str' = 'euclidean', n_bootstrap: 'int' = 99, reselect_bandwidth: 'bool' = True, pvalue_method: 'str' = 'plus_one', localized_tail: 'str' = 'two-sided', store_local_bootstrap: 'bool' = False, random_state: 'Optional[Union[int, np.random.Generator]]' = None, verbose: 'bool' = False) -> 'None'
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

"""Run coefficient-wise bootstrap tests for spatial variability."""

from pygwrx import BootstrapGWR
from _common import print_model_result, spatial_regression

X, y, coords = spatial_regression(n=42, p=2)
model = BootstrapGWR(
    bandwidth=22,
    adaptive=True,
    n_bootstrap=9,
    reselect_bandwidth=False,
    store_local_bootstrap=True,
    random_state=0,
).fit(X, y, coords)
print_model_result(model)
print("modified_pvalues=", model.modified_p_values_)
print("localized_p_values_shape=", model.localized_p_values_.shape)
```

Run it from the `examples/models` directory or through `python examples/run_all.py`.

## Reading the fitted result

**Main outputs:** Observed test statistics, modified/global p-values, optional localized p-values, bootstrap bandwidths, stored coefficient replicates, summaries, and result frames.

Available high-level methods detected in the current class are: `fit()`, `summary()`, `to_frame()`.

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

Check Monte Carlo resolution, random-seed reproducibility, sensitivity to bandwidth reselection, and the multiplicity of localized tests.

The common diagnostics layer can be used where the fitted model provides the required fields:

```python
from pygwrx.diagnostics import diagnostics_frame, local_diagnostic_frame

print(diagnostics_frame([model], labels=["BootstrapGWR"]))
try:
    print(local_diagnostic_frame(model).head())
except (AttributeError, NotImplementedError, ValueError) as exc:
    print("This model exposes a different diagnostic contract:", exc)
```

See [Diagnostics and inference](../guides/diagnostics.md) for model-aware checks and interpretation rules.

## Recommended visual checks


<div class="figure-grid" markdown>

<figure markdown>
  ![08 bootstrap pvalues](../assets/figures/specialized/08_bootstrap_pvalues.png){ loading=lazy }
  <figcaption>08 Bootstrap Pvalues</figcaption>
</figure>

<figure markdown>
  ![09 bootstrap bandwidths](../assets/figures/specialized/09_bootstrap_bandwidths.png){ loading=lazy }
  <figcaption>09 Bootstrap Bandwidths</figcaption>
</figure>

</div>


The figures are generated from deterministic examples and are illustrative; they are not benchmark claims.

## Common mistakes

- Using too few replicates for precise p-values.
- Claiming stationarity because a low-powered test is non-significant.
- Ignoring multiple testing for localized p-values.
- Treating the fixed OLS null as interchangeable with unsupported spatial-error or spatial-lag nulls.

## What to report in a paper or technical report

- The validated OLS null model and both coefficient-wise/global and localized statistics.
- Number of bootstrap replicates and random seed.
- Bandwidth reselection choice.
- Global and localized p-value treatment.
- Monte Carlo uncertainty and multiplicity correction.

## References

- [Harris et al. (2017), *Introducing bootstrap methods to investigate coefficient non-stationarity*](https://doi.org/10.1016/j.spasta.2017.07.006)
- [Brunsdon, Fotheringham & Charlton (1996), *Geographically Weighted Regression: A Method for Exploring Spatial Nonstationarity*](https://doi.org/10.1111/j.1538-4632.1996.tb00936.x)

## Related documentation

- [Detailed API for `BootstrapGWR`](../api/models/bootstrap-gwr.md)
- [Maintained example source](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/models/14_bootstrap_gwr.py)
- [Kernels and bandwidths](../guides/kernels-and-bandwidths.md)
- [Prediction and result objects](../guides/prediction-and-results.md)
- [Complete Chinese model guide](../zh/models/bootstrap-gwr.md)
