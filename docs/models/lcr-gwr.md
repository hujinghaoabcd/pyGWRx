# Locally Compensated Ridge GWR (`LCRGWR`)

<div class="model-hero" markdown>

**Task:** continuous-response local regression with location-specific collinearity diagnosis and ridge compensation  
**Core mechanism:** apply a positive local ridge parameter only where the unpenalised local condition number exceeds a threshold  
**Required inputs:** predictor matrix `X`, numeric response `y`, coordinates `coords`  
**Independent-target prediction:** supported with target-specific condition numbers and ridge terms

</div>

[API reference](../api/models/lcr-gwr.md){ .md-button .md-button--primary }
[GWR manual](gwr.md){ .md-button }
[GWLasso manual](gw-lasso.md){ .md-button }

## What problem LCRGWR solves

GWR fits many local regressions from overlapping weighted samples. Even when predictors are not severely collinear globally, a predictor can become locally constant or locally redundant inside particular neighbourhoods. The resulting local normal equations can be unstable, producing large coefficient variance, abrupt sign reversals and extreme local estimates.

LCRGWR first diagnoses the unpenalised weighted design at every location. When the local condition number exceeds `cn_thresh`, it calculates a location-specific ridge value intended to reduce the condition number toward the threshold.

With locally normalised singular values $d_{\max}$ and $d_{\min}$, the classical compensation rule is

$$
\lambda_i=
\frac{d_{\max}-\kappa^* d_{\min}}
{\kappa^*-1},
$$

where $\kappa^*$ is `cn_thresh`. Locations below the threshold retain the baseline `lambda_ridge`, usually zero.

!!! important "Pre- and post-compensation condition numbers are different diagnostics"
    `condition_numbers_` describes the unpenalised local design and therefore remains high at locations that required compensation. Use `compensated_condition_numbers_` and `penalized_system_condition_numbers_` to inspect the implied and actual post-penalty systems.

## When to use LCRGWR

Use LCRGWR when:

- the response is continuous and GWR is otherwise appropriate;
- coefficient surfaces show instability associated with local predictor correlation;
- local condition numbers exceed a pre-specified diagnostic threshold;
- retaining all predictors is scientifically preferable to local variable deletion;
- a transparent location-specific stabilisation map is required.

Do not use it merely because ordinary GWR coefficients vary. Spatial variation and numerical collinearity are different phenomena.

| Main objective | Better starting point |
|---|---|
| Reduce influence of response outliers | [`RGWR`](rgwr.md) |
| Select a sparse subset of predictors at each location | [`GWLasso`](gw-lasso.md) |
| Assign known variables to global and local groups | [`MixedGWR`](mixed-gwr.md) |
| Allow each predictor a different bandwidth | [`MGWR`](mgwr.md) |
| Diagnose collinearity without changing estimates | Fit GWR and use local collinearity diagnostics first. |

## What pyGWRx implements

The class follows the classical GWR-LCR workflow and the `GWmodel::gwr.lcr` conventions, with one explicit consistency improvement: pyGWRx constructs the hat matrix from the actual penalised estimator, so smoother traces, information criteria, influence and standard errors correspond to the ridge-adjusted fit.

The implementation provides:

- strict leave-one-out CV bandwidth selection for automatic LCRGWR;
- fixed or adaptive spatial bandwidths;
- a constant baseline ridge term through `lambda_ridge`;
- threshold-triggered local compensation through `lambda_adjust=True`;
- pre-compensation, formula-implied post-compensation and actual penalised-system condition numbers;
- location-specific lambda values and masks;
- optional final-bandwidth LOOCV residuals;
- target-location predictions and target local diagnostics.

The intercept is penalised together with the slopes, matching the reference GWmodel convention.

## Installation

```bash
pip install pygwrx
```

## Self-contained example

```python
import numpy as np
import pandas as pd

from pygwrx import LCRGWR

rng = np.random.default_rng(21)
n = 90
coords = rng.uniform(0.0, 100.0, size=(n, 2))

x1 = rng.normal(size=n)
# Strong overall correlation plus a spatially varying disturbance.
x2 = 0.97 * x1 + 0.08 * rng.normal(size=n) + 0.002 * coords[:, 0]
x3 = rng.normal(size=n)
X = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3})

y = 2.0 + 1.4 * x1 - 1.0 * x2 + 0.6 * x3 + rng.normal(0.0, 0.3, size=n)

model = LCRGWR(
    kernel="bisquare",
    bandwidth="cv",
    adaptive=True,
    cn_thresh=30.0,
    lambda_ridge=0.0,
    lambda_adjust=True,
).fit(
    X,
    y,
    coords,
    compute_hat_matrix=False,
    compute_cv=True,
)

print("bandwidth:", model.bandwidth_)
print("compensated locations:", model.locally_compensated_mask_.sum())
print("maximum original CN:", model.condition_numbers_.max())
print("maximum local lambda:", model.local_lambda_.max())
print(model.to_frame().head())
```

Target-location prediction and diagnostics are separate:

```python
X_new = X.iloc[:3].copy()
coords_new = np.array([[20.0, 20.0], [50.0, 50.0], [80.0, 80.0]])

pred = model.predict_result(X_new, coords_new)
local_diag = model.get_local_diagnostics(coords_new)
print(pred.to_frame())
print(local_diag)
```

## Constructor

```python
LCRGWR(
    kernel="bisquare",
    bandwidth="cv",
    bandwidth_method="cv",
    adaptive=False,
    bandwidth_range=None,
    optimization_method="golden_section",
    lambda_ridge=0.0,
    lambda_adjust=True,
    cn_thresh=30.0,
    fit_intercept=True,
    distance_metric="euclidean",
    sigma2_v1=True,
    verbose=False,
)
```

## Constructor parameters

### Spatial and bandwidth parameters

| Parameter | Default | Meaning and use |
|---|---:|---|
| `kernel` | `"bisquare"` | Spatial kernel used by every local penalised regression. Compact support can expose locally unsupported predictors more sharply. |
| `bandwidth` | `"cv"` | Numeric fixed/adaptive bandwidth, `"cv"`, or `None`. Classical automatic LCRGWR selection is strict leave-one-out CV. |
| `bandwidth_method` | `"cv"` | Automatic criterion when `bandwidth=None`. Only `"cv"` is supported for the classical LCR algorithm. |
| `adaptive` | `False` | Interprets numeric bandwidth as a neighbour count when true. Adaptive neighbourhoods are often useful under uneven sampling density. |
| `bandwidth_range` | `None` | Optional CV search bounds. Check whether selection reaches a boundary. |
| `optimization_method` | `"golden_section"` | `"golden_section"`, `"brent"`, or `"grid"`. Grid search provides the clearest sensitivity trace. |
| `fit_intercept` | `True` | Adds a local intercept, which is included in the penalised design under the reference convention. |
| `distance_metric` | `"euclidean"` | Defines spatial proximity and fixed-bandwidth units. |
| `sigma2_v1` | `True` | Residual variance convention for local standard errors. |

### Ridge and condition-number parameters

| Parameter | Default | Meaning | How to choose and what can go wrong |
|---|---:|---|---|
| `lambda_ridge` | `0.0` | Non-negative ridge value applied at every location before optional compensation. | Keep zero for classical threshold-only LCRGWR. A positive value creates globally present local ridge regularisation and means `ridge_applied_mask_` may be true even below the condition threshold. |
| `lambda_adjust` | `True` | Enables threshold-triggered replacement of `lambda_ridge` with a location-specific compensation value. | Set false only for a controlled constant-ridge comparison. With false and `lambda_ridge=0`, the model is effectively unpenalised under the LCR fitting conventions. |
| `cn_thresh` | `30.0` | Desired maximum local condition number used by the compensation formula. | Values around 20–30 are common diagnostic conventions, not universal laws. Pre-specify or sensitivity-test the threshold. A lower threshold applies ridge at more locations and increases bias. Must be greater than one. |
| `verbose` | `False` | Prints bandwidth and fit information. | Useful for CV and threshold diagnostics. |

## How condition numbers are calculated

pyGWRx follows the GWmodel/Belsley-style convention used by LCRGWR:

1. multiply the design columns by local weights;
2. normalise columns by their local Euclidean norms;
3. obtain singular values;
4. calculate largest / smallest singular value.

A locally constant or unsupported predictor can produce a zero smallest singular value and an infinite pre-compensation condition number. That is a substantive warning about local information support, not merely a numerical inconvenience.

## Fitting

```python
model.fit(
    X,
    y,
    coords,
    compute_hat_matrix=True,
    compute_local_r2=True,
    compute_inference=True,
    compute_cv=True,
    verbose=None,
)
```

| Fit argument | Default | Meaning and guidance |
|---|---:|---|
| `compute_hat_matrix` | `True` | Stores the complete penalised smoother matrix. Traces remain available when false. Disable for larger samples unless matrix entries are required. |
| `compute_local_r2` | `True` | Computes local R². It does not diagnose collinearity and should be read alongside condition-number maps. |
| `compute_inference` | `True` | Computes penalised-estimator covariance factors, local SEs and t values. Ridge introduces bias, so ordinary unpenalised inferential interpretations require caution. |
| `compute_cv` | `True` | Computes final-bandwidth leave-one-out residuals, squared contributions and total CV score. Disable when those outputs are unnecessary after a supplied bandwidth. |
| `verbose` | `None` | Per-fit verbosity override. |

## Main fitted attributes

| Attribute | Meaning |
|---|---|
| `condition_numbers_` / `local_condition_numbers_` | Unpenalised local condition numbers. |
| `local_lambda_` / `local_lambdas_` | Ridge value actually used at every calibration location. |
| `compensated_condition_numbers_` | Condition numbers implied by the classical compensation formula. |
| `penalized_system_condition_numbers_` | Numerical condition numbers of the actual penalised normal systems. |
| `locally_compensated_mask_` | Locations above `cn_thresh` where adaptive compensation was applied. |
| `ridge_applied_mask_` | All locations with positive final lambda, including constant baseline ridge. |
| `design_scales_` | Global design-column scales used by the estimator. |
| `cv_residuals_`, `cv_contributions_` | Final-bandwidth LOOCV residuals and their squared contributions. |
| `bandwidth_cv_score_` | Sum of squared LOOCV residuals. |
| `bandwidth_selection_result_` | Search value, score, evaluations, convergence and method. |
| `coef_`, `intercept_`, `fitted_values_`, `residuals_` | Penalised local fit results. |

`to_frame()` exports coefficient results together with local condition numbers, lambdas and compensation masks. `get_local_diagnostics(coords)` computes the same diagnostic family at arbitrary target coordinates.

## Interpreting the outputs

### Diagnose before celebrating smoother coefficients

A ridge penalty usually reduces coefficient variance and extreme values. That visual stability is not proof that the original scientific specification was sound. Inspect which predictors are locally unsupported and whether the bandwidth or design should change.

### Compare three condition-number fields

- `condition_numbers_`: severity of the original local design problem;
- `compensated_condition_numbers_`: result implied by the classical singular-value formula;
- `penalized_system_condition_numbers_`: condition of the actual matrix solved by pyGWRx.

They answer different questions and need not be numerically identical.

### Interpret lambda spatially

A clustered high-lambda region can indicate:

- locally redundant predictor patterns;
- insufficient variation within the chosen bandwidth;
- boundary effects;
- sparse sampling;
- a predictor whose spatial support differs from the shared GWR bandwidth.

### Ridge changes the estimand

Penalised coefficients are biased toward zero. Compare signs, magnitudes and uncertainty with ordinary GWR, but do not describe LCR estimates as if the penalty had no inferential consequence.

## LCRGWR versus GWLasso

| Feature | LCRGWR | GWLasso |
|---|---|---|
| Main problem | Numerical instability from local collinearity | Local shrinkage and variable selection |
| Penalty | L2 ridge | L1 Lasso |
| Activation | Condition-number threshold or constant lambda | Fixed or locally CV-selected alpha |
| Coefficients exactly zero | Generally no | Yes |
| Keeps all predictors | Yes | Not necessarily |
| Primary diagnostic | Condition numbers and local lambda | Selection frequency, active sets and local alpha |

## Common mistakes

| Mistake | Correction |
|---|---|
| Expecting `condition_numbers_` to fall below the threshold | It records the pre-compensation design. Inspect post-compensation fields. |
| Treating `cn_thresh=30` as universally optimal | Justify and sensitivity-test thresholds such as 20, 25 and 30. |
| Using LCRGWR to address response outliers | Use RGWR and residual diagnostics. |
| Interpreting ridge-stabilised t values exactly like ordinary GWR t values | Acknowledge penalisation bias and focus on stability/sensitivity. |
| Ignoring infinite local condition numbers | Check locally constant predictors, bandwidth support and data preparation. |
| Setting a positive `lambda_ridge` while claiming threshold-only compensation | Report the constant baseline penalty explicitly. |
| Disabling `compute_cv` and then expecting `cv_residuals_` | Leave it enabled when final LOOCV diagnostics are required. |
| Comparing target predictions without target diagnostic maps | Use `get_local_diagnostics()` to show target condition numbers and lambdas. |

## What to report

Report:

- all ordinary GWR spatial settings;
- local condition-number definition;
- `cn_thresh`, `lambda_ridge` and `lambda_adjust`;
- selected bandwidth and CV search details;
- number and spatial pattern of compensated locations;
- distributions/maps of original, compensated and actual system condition numbers;
- local lambda distribution;
- coefficient comparison with ordinary GWR;
- ridge-aware inference caveats;
- final-bandwidth CV and spatial validation results;
- pyGWRx version.

## References

- Wheeler, D. C. (2007). Diagnostic tools and a remedial method for collinearity in geographically weighted regression. *Environment and Planning A*, 39(10), 2464–2481. [`10.1068/a38325`](https://doi.org/10.1068/a38325)
- Gollini, I., Lu, B., Charlton, M., Brunsdon, C., & Harris, P. (2015). GWmodel: An R Package for Exploring Spatial Heterogeneity Using Geographically Weighted Models. *Journal of Statistical Software*, 63(17), 1–50. [`10.18637/jss.v063.i17`](https://doi.org/10.18637/jss.v063.i17)

## Related documentation

- [Generated LCRGWR API](../api/models/lcr-gwr.md)
- [Standard GWR](gwr.md)
- [Robust GWR](rgwr.md)
- [Geographically Weighted Lasso](gw-lasso.md)
- [Diagnostics and inference](../guides/diagnostics.md)