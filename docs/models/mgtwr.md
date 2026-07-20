# Multiscale Geographically and Temporally Weighted Regression (`MGTWR`)

<div class="model-hero" markdown>

**Task:** continuous-response local regression with parameter-specific spatial and temporal scales  
**Core mechanism:** additive backfitting with one spatial bandwidth and one temporal scale `tau` per fitted parameter  
**Required inputs:** `X`, `y`, two-dimensional spatial coordinates, and numeric row-wise times  
**Independent-target prediction:** intentionally unavailable

</div>

[API reference](../api/models/mgtwr.md){ .md-button .md-button--primary }
[GTWR manual](gtwr.md){ .md-button }
[MGWR manual](mgwr.md){ .md-button }

## Why MGTWR exists

GTWR forces the intercept and all predictor effects to use one common space-time neighbourhood. MGTWR allows every fitted parameter to operate at its own spatial and temporal scale.

For parameter $j$, pyGWRx combines distances as

$$
d_{ij}^{(j)}=\sqrt{d_{s,ij}^2+\tau_j d_{t,ij}^2}.
$$

Each coefficient receives:

- spatial bandwidth $b_j$;
- non-negative temporal scale $\tau_j$.

`tau_j=0` removes temporal distance for that coefficient. Larger `tau_j` increases temporal separation and therefore produces stronger temporal locality for a fixed spatial bandwidth.

The equivalent temporal bandwidth reported by pyGWRx is

$$
b_{t,j}=\frac{b_j}{\sqrt{\tau_j}},
$$

and is infinite when `tau_j=0`.

## MGTWR is both multiscale and coupled

The model is fitted by additive backfitting. One coefficient contribution is updated from its partial residual while all other contributions are held fixed. Its spatial bandwidth and temporal scale are selected or applied during that update. Iteration continues until the score of change converges.

Thus, the scale pairs are not independent one-variable fits. They must be interpreted as one coupled model.

## When to use MGTWR

Use it when:

- the response is continuous;
- every row has a numeric time coordinate;
- different predictors plausibly operate at different spatial and temporal scales;
- coefficient-specific scale interpretation is a main research objective;
- calibration-location inference is sufficient.

Do not start with MGTWR before fitting simpler baselines. Its extra scale flexibility increases computation, identifiability risk, and sensitivity to units.

| Situation | Better starting point |
|---|---|
| One common space-time scale is acceptable | [`GTWR`](gtwr.md) |
| Data are ordered snapshots and response-change rates define history | [`STWR`](stwr.md) |
| Time is irrelevant | [`MGWR`](mgwr.md) |
| Independent target prediction is required | GTWR currently provides the validated target operator. |
| Times are datetime objects | Convert them deliberately to numeric units before MGTWR. |

## Units determine the meaning of `tau`

MGTWR accepts numeric time only and does not rescale it. Spatial distance is Euclidean. Therefore `tau` changes meaning when either coordinate units or time units change.

For example, converting time from days to hours multiplies temporal differences by 24. To represent the same combined distance, `tau` would need to shrink by $24^2$.

Before fitting:

- use projected coordinates;
- choose one explicit numeric time unit;
- record the time origin and conversion;
- keep units identical across all comparisons.

## Installation

```bash
pip install pygwrx
```

## Self-contained example: automatic scale selection

```python
import numpy as np
import pandas as pd

from pygwrx import MGTWR

rng = np.random.default_rng(123)
n = 72
coords = rng.uniform(0.0, 100.0, size=(n, 2))
times = np.linspace(0.0, 12.0, n)  # months since study origin

X = pd.DataFrame(
    {
        "local_fast": rng.normal(size=n),
        "broad_slow": rng.normal(size=n),
    }
)

beta_fast = 0.8 + 0.7 * np.sin(coords[:, 0] / 15.0) + 0.08 * times
beta_slow = -1.0 + 0.004 * coords[:, 1] + 0.01 * times
y = (
    3.0
    + beta_fast * X["local_fast"].to_numpy()
    + beta_slow * X["broad_slow"].to_numpy()
    + rng.normal(0.0, 0.30, size=n)
)

model = MGTWR(
    bandwidths=None,
    taus=None,
    kernel="bisquare",
    adaptive=True,
    bandwidth_method="aicc",
    bandwidth_range=(20, 65),
    tau_range=(0.0, 3.0),
    tol=1e-4,
    tol_multi=1e-5,
    max_iter=100,
    calculate_inference=True,
    n_chunks=2,
).fit(X, y, coords, times)

names = ["Intercept", *X.columns]
for name, bandwidth, tau, temporal_bw in zip(
    names,
    model.bandwidths_,
    model.taus_,
    model.temporal_bandwidths_,
):
    print(
        f"{name:>12}: spatial={bandwidth}, "
        f"tau={tau:.4f}, temporal_bw={temporal_bw:.4f}"
    )

print("converged:", model.converged_)
print("iterations:", model.n_iter_)
print(model.to_frame().head())
```

Automatic fitting is computationally intensive. Use a small, defensible predictor set and coarse exploratory tolerances before final sensitivity runs.

## Self-contained example: manual scale pairs

`bandwidths` and `taus` must be supplied together.

```python
manual = MGTWR(
    # Intercept, local_fast, broad_slow
    bandwidths=[55, 25, 65],
    taus=[0.2, 2.0, 0.05],
    adaptive=True,
    init_bandwidth=45,
    init_tau=0.5,
    calculate_inference=False,
).fit(X, y, coords, times)

print(manual.bandwidths_)
print(manual.taus_)
```

A scalar bandwidth and scalar tau are repeated for all fitted parameters. This is useful for controlled reproduction, but it removes the main multiscale distinction.

## Constructor

```python
MGTWR(
    bandwidths=None,
    taus=None,
    *,
    kernel="bisquare",
    adaptive=True,
    fit_intercept=True,
    bandwidth_method="aicc",
    bandwidth_range=None,
    tau_range=(0.0, 4.0),
    init_bandwidth=None,
    init_tau=None,
    tol=1e-6,
    tol_multi=1e-5,
    max_iter=200,
    rss_score=False,
    calculate_inference=True,
    n_chunks=1,
    verbose=False,
)
```

## Constructor parameters

### Final and initial scales

| Parameter | Default | Meaning | Guidance |
|---|---:|---|---|
| `bandwidths` | `None` | Scalar or one final spatial bandwidth per fitted parameter. | Must be supplied together with `taus`. Sequence order includes the intercept. Adaptive values are neighbour counts in combined space-time distance. |
| `taus` | `None` | Scalar or one non-negative temporal scale per fitted parameter. | Must match bandwidth count. `tau=0` removes temporal distance for that coefficient. |
| `init_bandwidth` | `None` | Common spatial bandwidth for the initial GTWR fit. | When omitted, it is selected or derived from manual final bandwidths. It is not the final multiscale result. |
| `init_tau` | `None` | Common temporal scale for the initial GTWR fit. | When omitted, it is selected or derived from manual taus. |
| `bandwidth_range` | `None` | Common automatic spatial search bounds. | Adaptive bounds are integer neighbour counts. Check each final parameter for boundary selection. |
| `tau_range` | `(0,4)` | Common automatic temporal-scale bounds. | Boundary `tau=0` implies no temporal distance; an upper-bound solution may mean the search needs sensitivity analysis. |

### Backfitting and search

| Parameter | Default | Meaning and guidance |
|---|---:|---|
| `kernel` | `"bisquare"` | Gaussian, bisquare, or exponential kernel for every parameter. |
| `adaptive` | `True` | Spatial bandwidth is a neighbour count in each parameter's combined distance. |
| `bandwidth_method` | `"aicc"` | AICc, AIC, BIC, or CV scale-selection criterion. |
| `tol` | `1e-6` | Resolution target for the deterministic two-dimensional bandwidth/tau search. | Smaller values increase candidate refinement and cost. It is not the backfitting convergence tolerance. |
| `tol_multi` | `1e-5` | Additive backfitting score-of-change tolerance. | Always report convergence status and history. |
| `max_iter` | `200` | Maximum backfitting iterations and an input to scale-search limits. |
| `rss_score` | `False` | Uses relative RSS change instead of smooth-function change when true. | Hold fixed across comparisons. |
| `fit_intercept` | `True` | Fits a varying intercept with its own bandwidth and tau. | Manual vectors then have `p + 1` entries. |
| `verbose` | `False` | Prints search and backfitting progress. |

### Exact inference

| Parameter | Default | Meaning and guidance |
|---|---:|---|
| `calculate_inference` | `True` | Computes exact smoother traces, variable-specific effective parameters, local SEs/t values, and information criteria. | Disable for exploratory scale timing or large samples when exact inference is too expensive. |
| `n_chunks` | `1` | Number of column chunks in exact smoother calculation. | Increasing it reduces peak temporary memory without changing fitted coefficients or final mathematics. |

Unlike MGWR, these switches are constructor parameters rather than `fit()` arguments.

## Fitting

```python
model.fit(X, y, coords, times)
```

Input requirements:

- `X`, `y`, `coords`, and `times` must have equal row counts;
- `coords` are two-dimensional and use Euclidean distance;
- `times` must be finite one-dimensional numeric values;
- sample size must exceed fitted parameter count by more than two;
- DataFrame feature order defines manual scale-vector order.

Automatic scale selection uses a deterministic coarse-to-fine two-dimensional candidate search with explicit boundary evaluation. It is not an exhaustive proof of the global optimum.

## Scale interpretation

| Output | Interpretation |
|---|---|
| Small `bandwidths_[j]` | Parameter uses a local combined space-time neighbourhood. |
| Large `bandwidths_[j]` | Broader combined neighbourhood. Under adaptive mode, this is neighbour count, not pure spatial distance. |
| `taus_[j] == 0` | Temporal difference is ignored for parameter `j`. |
| Large `taus_[j]` | Temporal differences are amplified, producing stronger temporal locality for a fixed bandwidth. |
| Small `temporal_bandwidths_[j]` | Narrower equivalent time scale under the fitted units. |
| Infinite temporal bandwidth | Tau is zero; the coefficient has no temporal-distance term. |

Because adaptive neighbours are chosen using combined distance, `bandwidths_` should not be described as a purely spatial neighbourhood size without also reporting `taus_`.

## Main fitted attributes

| Attribute | Meaning |
|---|---|
| `initial_bandwidth_`, `initial_tau_` | Common starting GTWR scales. |
| `bandwidth_` | Compatibility field equal to the initial shared bandwidth, not the final vector. |
| `bandwidths_`, `taus_` | Final parameter-specific scale pairs. |
| `temporal_bandwidths_` | Derived `bandwidth / sqrt(tau)` values. |
| `bandwidth_history_`, `tau_history_` | Scale vectors from each backfitting iteration. |
| `convergence_history_`, `n_iter_`, `converged_` | Backfitting stability information. |
| `params_`, `coef_`, `intercept_` | Calibration-location local parameters. |
| `parameter_contributions_` | Additive contribution of each parameter; row sum equals `fitted_values_`. |
| `effective_params_by_variable_` / `ENP_j_` | Exact complexity by parameter when inference is enabled. |
| `parameter_standard_errors_`, `parameter_t_values_` | Local inference arrays when enabled. |
| `adjusted_alpha_by_variable_`, `critical_t_values_` | Variable-specific multiple-comparison aids. |
| `rss_`, `r2_`, `aic_`, `aicc_`, `bic_` | Fit diagnostics; information criteria require exact inference. |

`to_frame()` adds numeric time to the inherited calibration-location coefficient table.

## Prediction limitation

```python
model.predict(X_new, coords_new, times_new)
```

raises `NotImplementedError`. This is deliberate: the package does not expose an unvalidated shortcut for independently supplied target locations.

Use:

```python
model.fitted_values_
model.to_frame()
```

for calibration results. For transfer assessment, create spatial-temporal holdouts, refit on each training partition, and evaluate through a separately justified prediction protocol. Do not report calibration fitted values as future or out-of-region prediction.

## Computational implications

MGTWR repeatedly searches two scales for every parameter and every backfitting iteration. Exact inference then propagates the full additive smoother process in chunks.

Cost increases with:

- sample size;
- number of predictors;
- scale-search resolution;
- number of backfitting iterations;
- exact inference;
- smaller chunks reducing memory but increasing loop overhead.

The class does not retain full or partial hat matrices after exact inference, but intermediate operations remain computationally substantial.

## Recommended workflow

1. Define numeric spatial and time units.
2. Fit global regression and GWR.
3. Fit GTWR with one common scale.
4. Fit MGWR to inspect purely spatial multiscale structure.
5. Fit MGTWR with conservative bounds and predictor count.
6. Check convergence and scale histories.
7. Inspect every bandwidth/tau boundary.
8. Compare coefficient scales, ENP, uncertainty, and residuals together.
9. Repeat under plausible initial scales and unit-preserving bounds.
10. Use refitted spatial-temporal holdouts for transfer claims.

## Common mistakes

| Mistake | Correction |
|---|---|
| Passing datetime values directly | Convert to an explicit numeric unit and record the origin. |
| Supplying bandwidths without taus | Both must be supplied together. |
| Forgetting the intercept entry | With `fit_intercept=True`, vectors require `p + 1` values, intercept first. |
| Reading `bandwidth_` as the final MGTWR scale | Use `bandwidths_`; `bandwidth_` is the initial shared value. |
| Calling adaptive bandwidth purely spatial | Neighbours are ordered by combined space-time distance. |
| Comparing tau after changing units | Tau is unit-dependent; keep units fixed. |
| Ignoring `converged_=False` | Inspect both bandwidth and tau histories and rerun sensitivity analyses. |
| Treating automatic search as a guaranteed global optimum | It is deterministic coarse-to-fine search with boundaries, not exhaustive optimisation. |
| Calling `predict()` | Independent-target prediction is intentionally unavailable. |
| Disabling inference and then interpreting AICc/ENP fields as exact | Exact smoother diagnostics require `calculate_inference=True`. |

## What to report

Report:

- spatial projection and units;
- numeric time origin and units;
- kernel and adaptive/fixed semantics;
- criterion, bandwidth range, tau range, and search tolerance;
- initial shared bandwidth and tau;
- final parameter order, bandwidth vector, tau vector, and derived temporal bandwidths;
- backfitting tolerance, score type, iterations, convergence status, and histories;
- inference/chunk settings and effective parameters;
- boundary and initialisation sensitivity;
- comparison with GWR, GTWR, and MGWR;
- explicit absence of independent-target prediction;
- refitted validation design;
- pyGWRx version.

## References

- Wu, C., Ren, F., Hu, W., & Du, Q. (2019). Multiscale geographically and temporally weighted regression: exploring the spatiotemporal determinants of housing prices. *International Journal of Geographical Information Science*, 33(3), 489–511. [`10.1080/13658816.2018.1545158`](https://doi.org/10.1080/13658816.2018.1545158)

## Related documentation

- [Generated MGTWR API](../api/models/mgtwr.md)
- [GTWR](gtwr.md)
- [STWR](stwr.md)
- [MGWR](mgwr.md)
- [Spatiotemporal data](../guides/spatiotemporal-data.md)