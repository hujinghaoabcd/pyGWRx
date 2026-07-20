# Robust Geographically Weighted Regression (`RGWR`)

<div class="model-hero" markdown>

**Task:** continuous-response local regression resistant to high-residual observations  
**Core mechanism:** multiply each spatial kernel by an observation-level robust residual weight  
**Required inputs:** predictor matrix `X`, numeric response `y`, coordinates `coords`  
**Independent-target prediction:** supported using the final robust calibration weights

</div>

[API reference](../api/models/rgwr.md){ .md-button .md-button--primary }
[GWR manual](gwr.md){ .md-button }
[Diagnostics guide](../guides/diagnostics.md){ .md-button }

## What problem RGWR solves

Every observation participates in many overlapping GWR calibrations. A response outlier can therefore distort not only the coefficient estimate at its own location but also many neighbouring coefficient estimates. RGWR reduces that contamination by combining ordinary spatial weights with a second observation-level weight derived from residual behaviour.

At focal location $s_i$, observation $j$ receives

$$
w_{ij}^{\mathrm{effective}}=w_{ij}^{\mathrm{spatial}}r_j,
$$

where $r_j\in[0,1]$ is shared across all local regressions. A low robust weight means that the observation contributes less wherever it falls inside a spatial neighbourhood.

RGWR is a response-outlier remedy. It is not a substitute for diagnosing local predictor collinearity, incorrect response distributions, omitted variables or spatially structured residuals.

## When to use RGWR

Use RGWR when:

- the response is continuous and Gaussian local regression is otherwise appropriate;
- a small number of observations have unusually large residuals;
- ordinary GWR coefficient surfaces change markedly when those observations are removed;
- measurement error, recording anomalies or rare response events could contaminate neighbouring fits;
- the robust-weight map is itself useful for auditing the data.

Do not use RGWR as the automatic default merely because residuals are imperfect. When broad model misspecification causes many large residuals, robust downweighting can conceal the problem rather than solve it.

| Main problem | More appropriate response |
|---|---|
| Correlated local predictors | [`LCRGWR`](lcr-gwr.md) or [`GWLasso`](gw-lasso.md) |
| Count or binary response | [`GWGLM`](gwglm.md) |
| Different predictor scales | [`MGWR`](mgwr.md) |
| A known global/local coefficient partition | [`MixedGWR`](mixed-gwr.md) |
| Spatial transfer prediction rather than coefficient robustness | Validate GWR and RGWR with spatial holdouts; do not rely on in-sample fit alone. |

## The two robust methods

pyGWRx implements the two classical procedures exposed by `GWmodel::gwr.robust`.

### `method="automatic"`

1. Fit an initial ordinary GWR and retain its selected bandwidth.
2. Divide residual magnitudes by the root mean squared residual.
3. Convert those scores to robust weights:
   - score $\leq$ `cut1`: weight 1;
   - `cut1` < score $\leq$ `cut2`: smooth bisquare transition;
   - score > `cut2`: weight 0.
4. Refit every local regression using spatial weight × robust weight.
5. Repeat until relative MSE change is no greater than `tol` or `max_iter` is reached.

This method can partially downweight an observation before assigning zero weight.

### `method="filtered"`

1. Fit an initial ordinary GWR with the full hat matrix.
2. Calculate GWmodel-style studentised residuals.
3. Set the weight to zero when `abs(studentised residual) >= cut_filter`; otherwise retain weight 1.
4. Refit once.

This method performs a single hard filtering step. It is easier to audit but more sensitive to the selected cutoff.

!!! important "Bandwidth selection precedes robust reweighting"
    The bandwidth is selected from the initial ordinary GWR and then retained during robust fitting. The final robust weights do not trigger a second automatic bandwidth search.

## Installation

```bash
pip install pygwrx
```

## Self-contained example

```python
import numpy as np
import pandas as pd

from pygwrx import RGWR

rng = np.random.default_rng(12)
n = 70
coords = rng.uniform(0.0, 100.0, size=(n, 2))
X = pd.DataFrame(
    {
        "income": rng.normal(size=n),
        "access": rng.normal(size=n),
    }
)

beta = 1.0 + 0.01 * coords[:, 0]
y = (
    3.0
    + beta * X["income"].to_numpy()
    - 0.8 * X["access"].to_numpy()
    + rng.normal(0.0, 0.35, size=n)
)

# Inject two large response anomalies.
y[[8, 51]] += np.array([8.0, -7.0])

model = RGWR(
    kernel="bisquare",
    bandwidth="aicc",
    adaptive=True,
    method="automatic",
    cut1=2.0,
    cut2=3.0,
    max_iter=20,
    tol=1e-5,
).fit(
    X,
    y,
    coords,
    compute_hat_matrix=False,
)

print("bandwidth:", model.bandwidth_)
print("robust refits:", model.n_iter_)
print("converged:", model.converged_)
print("zero-weight observations:", np.flatnonzero(model.outlier_mask_))
print(model.to_frame().sort_values("robust_weight").head())
```

For an auditable hard-filter alternative:

```python
filtered = RGWR(
    kernel="bisquare",
    bandwidth=model.bandwidth_,
    adaptive=True,
    method="filtered",
    cut_filter=3.0,
).fit(X, y, coords, compute_hat_matrix=False)

print(filtered.to_frame().sort_values("initial_studentized_residual").tail())
```

Filtered RGWR internally stores the initial full hat matrix long enough to calculate its studentised residuals even when the final `compute_hat_matrix=False`.

## Constructor

```python
RGWR(
    kernel="gaussian",
    bandwidth="cv",
    bandwidth_method="cv",
    adaptive=False,
    bandwidth_range=None,
    optimization_method="golden_section",
    fit_intercept=True,
    distance_metric="euclidean",
    sigma2_v1=True,
    method="automatic",
    max_iter=20,
    tol=1e-5,
    cut1=2.0,
    cut2=3.0,
    cut_filter=3.0,
    verbose=False,
)
```

## Constructor parameters

### Spatial GWR parameters

| Parameter | Default | Meaning and use |
|---|---:|---|
| `kernel` | `"gaussian"` | Spatial kernel. Compact kernels make the interaction between neighbourhood support and zero robust weights especially visible. |
| `bandwidth` | `"cv"` | Numeric fixed/adaptive bandwidth or CV/AIC/AICc/BIC token. Selection uses the initial standard GWR only. |
| `bandwidth_method` | `"cv"` | Criterion used when `bandwidth=None`. |
| `adaptive` | `False` | Numeric bandwidth is a neighbour count when true and a coordinate distance when false. |
| `bandwidth_range` | `None` | Optional automatic search bounds. Inspect whether the selected initial-GWR bandwidth reaches a bound. |
| `optimization_method` | `"golden_section"` | `"golden_section"`, `"brent"` or `"grid"`. |
| `fit_intercept` | `True` | Fits a local intercept. Do not add a manual constant column. |
| `distance_metric` | `"euclidean"` | Defines spatial proximity. Coordinate and bandwidth units must be consistent. |
| `sigma2_v1` | `True` | Final robust-GWR residual variance convention used for standard errors. |

See the [GWR manual](gwr.md) for detailed fixed/adaptive bandwidth and distance guidance.

### Robust parameters

| Parameter | Default | Meaning | Selection guidance and failure mode |
|---|---:|---|---|
| `method` | `"automatic"` | Chooses iterative smooth downweighting or one-step hard filtering. | Use automatic when gradual influence reduction is preferred. Use filtered when a clear residual cutoff and one auditable refit are required. Compare both in sensitivity analysis when conclusions depend on a few observations. |
| `max_iter` | `20` | Maximum robust refits for automatic RGWR. | Increase only after checking that convergence history is steadily decreasing. Filtered RGWR always performs one robust refit. |
| `tol` | `1e-5` | Relative MSE-change stopping tolerance for automatic RGWR. | A smaller value demands tighter convergence. Always report `converged_`; reaching `max_iter` generates a warning. |
| `cut1` | `2.0` | Automatic score below which an observation keeps full weight. | Lower values start downweighting more observations. Excessively low values can remove normal residual variation. Must be non-negative. |
| `cut2` | `3.0` | Automatic score above which robust weight becomes zero. | Must be greater than `cut1`. Lower values reject more observations and can leave too little effective local support. |
| `cut_filter` | `3.0` | Absolute studentised-residual cutoff for filtered RGWR. | Smaller values remove more observations. Inspect the actual studentised-residual distribution rather than treating 3 as universally correct. |
| `verbose` | `False` | Prints automatic relative-MSE progress. | Useful for convergence diagnosis. |

## Fitting

```python
model.fit(
    X,
    y,
    coords,
    compute_hat_matrix=True,
    compute_local_r2=True,
    compute_inference=True,
    compute_hat_matrix_flag=None,
    verbose=None,
)
```

The fit arguments have the same meaning as GWR, with two qualifications:

- `compute_hat_matrix` controls storage of the **final robust** smoother matrix;
- filtered RGWR temporarily requires the initial full GWR matrix regardless of that final storage setting.

Robust filtering can fail with `RuntimeError` when too few observations retain positive weight for the number of design columns. The correct response is to inspect the data, increase bandwidth, reduce predictors or relax robust thresholds—not to suppress the error.

## Main fitted attributes

| Attribute | Meaning |
|---|---|
| `robust_method_` | Final normalised method name. |
| `robust_weights_` | Final observation-level weights in `[0, 1]`. |
| `downweighted_mask_` | Observations with weight below 1. |
| `outlier_mask_` | Observations with final weight equal to zero. |
| `n_iter_` | Number of robust refits after the initial GWR. |
| `converged_` | Automatic tolerance reached, or true after the filtered one-refit procedure. |
| `weight_history_` | All robust weight vectors, beginning with all ones. |
| `mse_history_` | Initial and successive robust-fit MSE values. |
| `convergence_history_` | Relative MSE changes for automatic RGWR. |
| `initial_fitted_values_`, `initial_residuals_` | Ordinary GWR state before robust weighting. |
| `initial_diagnostics_` | Initial GWR diagnostics for direct before/after comparison. |
| `initial_studentized_residuals_` | Initial filtered-method residual scores; `None` for automatic RGWR. |
| `robust_residual_scores_` | Final residuals divided by root MSE. |
| `coef_`, `intercept_`, `fitted_values_`, `residuals_` | Final robust local model results. |

`to_frame()` appends `robust_weight`, `downweighted`, `robust_outlier`, robust residual score and, for filtered RGWR, the initial studentised residual.

## Prediction semantics

RGWR inherits GWR target-location recalibration. The target spatial kernel is multiplied by the **final training-observation robust weights**. Prediction therefore preserves the fitted assessment of which calibration observations were unreliable.

```python
pred = model.predict(X_new, coords_new)
result = model.predict_result(X_new, coords_new)
```

This does not identify outliers in the new target response because target responses are not supplied. A prediction workflow must separately monitor target-domain anomalies and data drift.

## Interpretation workflow

1. Fit ordinary GWR with the same spatial specification.
2. Identify large residuals, leverage and Cook's distance.
3. Investigate data quality and scientific plausibility before downweighting.
4. Fit RGWR and compare initial versus final coefficient surfaces.
5. Map robust weights and the number of zero/downweighted observations.
6. Check automatic convergence or filtered threshold sensitivity.
7. Examine whether local conclusions depend on a small set of removed observations.
8. Validate GWR and RGWR using the same spatial holdouts.

A lower robust MSE is not sufficient evidence that the model is preferable: downweighting changes the fitting objective. The important questions are whether influential anomalies were handled transparently and whether substantive surfaces became more stable.

## Common mistakes

| Mistake | Correction |
|---|---|
| Calling every low robust weight a data error | Treat it as an influence signal; investigate the observation and local model. |
| Using RGWR to fix predictor collinearity | Use local condition-number diagnostics and LCRGWR/GWLasso. |
| Selecting bandwidth after removing residual outliers without reporting it | pyGWRx intentionally selects from the initial GWR; report that workflow. |
| Lowering `cut2` until coefficient maps look smooth | Predefine or sensitivity-test thresholds; do not tune by preferred visual outcome. |
| Ignoring `converged_=False` | Inspect `convergence_history_`, adjust iteration settings and report instability. |
| Comparing final RGWR diagnostics with GWR as if the objective were unchanged | Include initial diagnostics, robust weights and held-out validation. |
| Forgetting that one zero-weight observation affects many local fits | Map its neighbourhood influence and compare surfaces before/after robust fitting. |

## What to report

Report all ordinary GWR spatial settings plus:

- robust method;
- `cut1`, `cut2` or `cut_filter`;
- iteration limit, tolerance, iterations completed and convergence status;
- number and spatial distribution of downweighted and zero-weight observations;
- minimum/summary robust weights;
- comparison of initial and final diagnostics and coefficient surfaces;
- threshold sensitivity;
- investigation of flagged observations;
- spatial validation design;
- pyGWRx version.

## References

- Harris, P., Fotheringham, A. S., & Juggins, S. (2010). Robust geographically weighted regression: a technique for quantifying spatial relationships between freshwater acidification critical loads and catchment attributes. *Annals of the Association of American Geographers*, 100(2), 286–306. [`10.1080/00045600903550378`](https://doi.org/10.1080/00045600903550378)
- Lu, B., Harris, P., Charlton, M., & Brunsdon, C. (2014). The GWmodel R package: further topics for exploring spatial heterogeneity using geographically weighted models. *Geo-spatial Information Science*, 17(2), 85–101. [`10.1080/10095020.2014.917453`](https://doi.org/10.1080/10095020.2014.917453)

## Related documentation

- [Generated RGWR API](../api/models/rgwr.md)
- [Standard GWR](gwr.md)
- [LCRGWR](lcr-gwr.md)
- [Diagnostics and inference](../guides/diagnostics.md)