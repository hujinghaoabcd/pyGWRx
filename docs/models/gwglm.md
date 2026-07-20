# Geographically Weighted Generalized Linear Models (`GWGLM`)

<div class="model-hero" markdown>

**Task:** local Gaussian, Poisson, or Bernoulli regression  
**Core mechanism:** combine spatial kernel weights with family-specific local likelihood and IWLS weights  
**Required inputs:** predictor matrix `X`, response `y`, coordinates `coords`; optional Poisson exposure or offset  
**Independent-target prediction:** supported on the family-appropriate conditional-mean scale

</div>

[API reference](../api/models/gwglm.md){ .md-button .md-button--primary }
[GWR manual](gwr.md){ .md-button }
[Diagnostics guide](../guides/diagnostics.md){ .md-button }

## Why GWGLM exists

Standard GWR assumes a continuous response with Gaussian errors and an identity link. That is unsuitable for many spatial outcomes:

- event counts cannot be negative;
- binary probabilities must remain between zero and one;
- rates may require different population, area, or time exposures.

GWGLM retains the geographically weighted coefficient idea while changing the response distribution and link function. At focal location $s_i$,

$$
g\{\mu_j(s_i)\}=x_j^\top\beta(s_i),
$$

and the local likelihood is weighted by geographic proximity. Poisson and Bernoulli models are solved by local iteratively weighted least squares (IWLS), where spatial weights and GLM working weights act together.

## Supported families

| `family` | Response contract | Link | `predict()` returns | Important limitation |
|---|---|---|---|---|
| `"gaussian"` | Finite continuous values | Identity | Conditional mean on the response scale | This path delegates to standard GWR. |
| `"poisson"` | Finite non-negative values | Log | Expected count, including supplied exposure or offset | The class validates non-negativity but does not require integer-valued counts. Scientific use should still match a Poisson mean-variance interpretation. |
| `"binomial"` | Exactly 0 and 1, with both classes present | Logit | Probability of outcome 1 | Grouped binomial successes/trials are not supported; this is Bernoulli only. |

`family="bernoulli"` is accepted as an alias and normalized to `"binomial"`. Gamma is intentionally not exposed because it has not been validated against a mature geographically weighted reference implementation.

## When to use GWGLM

Use it when:

- the response distribution matches one of the supported families;
- predictor effects may vary smoothly over space;
- local conditional means or probabilities are scientifically meaningful;
- a common spatial bandwidth across coefficients is acceptable;
- family-specific deviance, Pearson residuals, and convergence diagnostics are required.

Do not choose the family from whichever produces the best map. Choose it from the response-generating process and data support.

| Situation | Better action or model |
|---|---|
| Continuous Gaussian response | Begin with [`GWR`](gwr.md); Gaussian GWGLM is mainly a unified interface. |
| Multiclass categorical response | [`GWDA`](gwda.md) |
| Count data show strong overdispersion or zero inflation | Diagnose explicitly; the current class does not expose negative-binomial or zero-inflated families. |
| Different predictors need different scales | The current GWGLM is single-bandwidth; do not infer multiscale behaviour from it. |
| Local events are extremely sparse | Increase support, simplify predictors, or reconsider whether local likelihood estimation is identifiable. |

## Poisson exposure and offset

For a Poisson model,

$$
\log\mu_i = x_i^\top\beta(s_i)+\log(e_i),
$$

where $e_i>0$ is exposure. In pyGWRx:

- `exposure=e` supplies positive exposure directly;
- `offset=np.log(e)` supplies the same quantity on the linear-predictor scale;
- only one of `exposure` and `offset` may be supplied;
- omitting both uses exposure 1;
- exposure/offset is rejected for Gaussian and Bernoulli families.

Exposure must be supplied consistently at fitting and prediction. A model fitted with population exposure but predicted with exposure 1 returns expected counts for a unit population, not directly comparable totals.

## Installation

```bash
pip install pygwrx
```

## Self-contained examples

### Poisson counts with exposure

```python
import numpy as np
import pandas as pd

from pygwrx import GWGLM

rng = np.random.default_rng(55)
n = 90
coords = rng.uniform(0.0, 100.0, size=(n, 2))
X = pd.DataFrame(
    {
        "deprivation": rng.normal(size=n),
        "access": rng.normal(size=n),
    }
)
exposure = rng.uniform(500.0, 2500.0, size=n)

beta_deprivation = 0.20 + 0.004 * coords[:, 0]
log_rate = -6.2 + beta_deprivation * X["deprivation"] - 0.25 * X["access"]
mu = exposure * np.exp(log_rate)
y = rng.poisson(mu)

model = GWGLM(
    family="poisson",
    kernel="bisquare",
    bandwidth="aicc",
    adaptive=True,
    max_iter=100,
    tol=1e-6,
).fit(
    X,
    y,
    coords,
    exposure=exposure,
    compute_hat_matrix=False,
)

print("bandwidth:", model.bandwidth_)
print("all local fits converged:", model.converged_)
print("deviance explained:", model.percent_deviance_)
print(model.to_frame().head())

X_new = X.iloc[:2].copy()
coords_new = np.array([[25.0, 40.0], [75.0, 60.0]])
new_exposure = np.array([1000.0, 2000.0])
result = model.predict_result(
    X_new,
    coords_new,
    exposure=new_exposure,
)
print(result.to_frame())
```

The returned Poisson predictions are expected counts for `new_exposure`. Divide by exposure only when a rate is the intended reporting scale.

### Bernoulli probabilities

```python
linear = -0.4 + 1.1 * X["deprivation"] - 0.7 * X["access"]
probability = 1.0 / (1.0 + np.exp(-linear))
y_binary = rng.binomial(1, probability)

classifier = GWGLM(
    family="binomial",
    bandwidth="cv",
    adaptive=True,
).fit(X, y_binary, coords)

prob = classifier.predict(X.iloc[:3], coords[:3])
labels = (prob >= 0.5).astype(int)
print(prob)
print(labels)
```

`predict()` returns probabilities, not thresholded class labels. Threshold selection is a separate classification decision and should be validated using appropriate metrics.

## Constructor

```python
GWGLM(
    family="gaussian",
    kernel="bisquare",
    bandwidth="cv",
    bandwidth_method="aicc",
    adaptive=False,
    bandwidth_range=None,
    optimization_method="golden_section",
    max_iter=100,
    tol=1e-6,
    fit_intercept=True,
    distance_metric="euclidean",
    sigma2_v1=True,
    verbose=False,
)
```

## Constructor parameters

### Distribution and local optimisation

| Parameter | Default | Meaning | How to use and what can go wrong |
|---|---:|---|---|
| `family` | `"gaussian"` | Response distribution and canonical link. | Use `gaussian`, `poisson`, or `binomial`; `bernoulli` aliases binomial. A wrong family produces misleading means, residuals, uncertainty, and bandwidth selection. |
| `max_iter` | `100` | Maximum IWLS iterations for every local Poisson/Bernoulli fit. | Inspect `local_converged_` and `iteration_counts_`. Increasing the limit does not fix separation or inadequate local support. |
| `tol` | `1e-6` | Local coefficient-update stopping tolerance. | Smaller values increase work. Keep it fixed in comparisons and report it. The implementation follows the established Python reference stopping convention based on the smallest absolute coefficient update. |
| `fit_intercept` | `True` | Fits a local intercept. | Do not add a manual constant column. For Poisson, the intercept is a local baseline log rate when exposure is used. |
| `sigma2_v1` | `True` | Gaussian-only residual-variance convention inherited from GWR. | It does not control Poisson/Bernoulli dispersion; non-Gaussian inference uses unit dispersion. |

### Spatial parameters

| Parameter | Default | Meaning and guidance |
|---|---:|---|
| `kernel` | `"bisquare"` | Spatial kernel. Compact kernels can create locations with too few events or outcome classes. |
| `bandwidth` | `"cv"` | Numeric bandwidth or `cv`, `aic`, `aicc`, `bic`. For Gaussian, standard GWR selection is used; non-Gaussian families use family-specific local likelihood/deviance calculations. |
| `bandwidth_method` | `"aicc"` | Criterion used only when `bandwidth=None`. Because the explicit default is `"cv"`, changing `bandwidth_method` alone does not change the default selection. |
| `adaptive` | `False` | Numeric bandwidth is an integer neighbour count when true. Adaptive non-Gaussian bandwidth must be at least `n_parameters + 1`. |
| `bandwidth_range` | `None` | Optional search interval. Check boundary selection and local event support. |
| `optimization_method` | `"golden_section"` | `golden_section`, `brent`, or `grid`; adaptive searches use integer-aware golden/grid behaviour. Brent is used only for fixed bandwidths. |
| `distance_metric` | `"euclidean"` | Spatial distance definition. Use projected coordinates or deliberate Haversine input. |
| `verbose` | `False` | Prints candidate scores and fitting progress. |

## Fitting

```python
model.fit(
    X,
    y,
    coords,
    exposure=None,
    offset=None,
    compute_hat_matrix=False,
    compute_inference=True,
    compute_local_r2=True,
)
```

| Fit argument | Families | Meaning and guidance |
|---|---|---|
| `exposure` | Poisson only | Positive scalar or one value per row. Converted internally to a log offset. |
| `offset` | Poisson only | Additive log-scale offset. Supply at most one of exposure/offset. |
| `compute_hat_matrix` | All | Retains the complete local smoother matrix. Trace statistics are still calculated for non-Gaussian diagnostics when false. |
| `compute_inference` | All | Computes local SEs and t/z-compatible arrays. Non-Gaussian output should be interpreted as Wald z statistics. |
| `compute_local_r2` | Gaussian only | Forwarded to GWR. Non-Gaussian local R² is not computed; use deviance diagnostics instead. |

Non-Gaussian fitting includes a tiny deterministic numerical ridge in local IWLS systems. This is a solver stabiliser, not a user-tuned penalised regression model.

## Prediction and scoring

```python
pred = model.predict(X_new, coords_new, exposure=None, offset=None)
result = model.predict_result(X_new, coords_new, exposure=None, offset=None)
score = model.score(X_test, y_test, coords_test, exposure=None, offset=None)
```

| Family | `predict()` scale | `score()` |
|---|---|---|
| Gaussian | Response mean | R² |
| Poisson | Expected count including target exposure/offset | Deviance explained against a Poisson null mean |
| Bernoulli | Probability of outcome 1 | Deviance explained, not classification accuracy |

For non-Gaussian targets, pyGWRx recalibrates local parameters using the stored training response and training exposure. Target exposure enters only when converting the target linear predictor to an expected count.

## Main fitted attributes

| Attribute | Meaning |
|---|---|
| `family_` | Normalized fitted family. |
| `fitted_values_` / `mu_` | Conditional means on the response scale. |
| `linear_predictor_` | Fitted values on the link scale, excluding the stored Poisson offset field. |
| `exposure_train_`, `offset_train_` | Poisson exposure and its log; ones/zeros for other families. |
| `deviance_`, `null_deviance_` | Family-specific fitted and null deviance. |
| `percent_deviance_` | `1 - deviance / null_deviance`. |
| `adjusted_percent_deviance_` | Effective-complexity-adjusted deviance measure. |
| `log_likelihood_` | Family-specific log likelihood. |
| `deviance_residuals_` | Signed deviance residuals. |
| `pearson_residuals_` | Residuals scaled by the family variance. |
| `iteration_counts_` | IWLS iterations by calibration location. |
| `local_converged_` | Local convergence flags. |
| `converged_` | True only when every local fit converged. |
| `final_working_weights_` | Final GLM working weights at calibration locations. |
| `parameter_standard_errors_` | Local parameter SEs. |
| `parameter_z_values_` | Non-Gaussian local Wald z values; Gaussian compatibility values mirror GWR t fields. |
| `bandwidth_selection_result_` | Non-Gaussian optimizer result when automatic selection is used. |
| `cv_residuals_`, `cv_contributions_` | Stored only when a non-Gaussian CV bandwidth was selected. |

`to_frame()` adds link-scale values, deviance/Pearson residuals, influence, IWLS iterations, convergence flags, Poisson exposure/offset, and local SE/z fields.

## Family-specific interpretation

### Gaussian

Interpret as standard GWR. Local R² and Gaussian residual variance are available. Use the [GWR manual](gwr.md) for coefficient and influence safeguards.

### Poisson

A slope coefficient is a local log-rate ratio when exposure is used. `exp(beta)` is the multiplicative change in expected rate for a one-unit predictor increase, holding other variables and exposure fixed. Check overdispersion, zero patterns, exposure quality, influential high counts, and residual spatial structure.

### Bernoulli

A slope is a local log-odds coefficient. `exp(beta)` is a local odds ratio, not a risk ratio. Extreme probabilities and non-convergence can signal local separation or insufficient outcome variation. Evaluate calibration, discrimination, class imbalance, threshold sensitivity, and spatially structured validation.

## Common mistakes

| Mistake | Correction |
|---|---|
| Supplying exposure for Gaussian or Bernoulli | Exposure/offset is Poisson-only. |
| Supplying both exposure and offset | Choose one; they encode the same model term. |
| Fitting grouped successes/trials with `family="binomial"` | The current implementation accepts Bernoulli 0/1 rows only. |
| Thresholding Bernoulli probabilities and calling `score()` accuracy | `score()` returns deviance explained. Compute classification metrics separately. |
| Interpreting Poisson output as a rate when exposure was supplied | `predict()` returns expected counts; divide by the desired exposure unit for rates. |
| Ignoring `local_converged_` | Map failed/high-iteration locations and inspect separation or sparse events. |
| Using local R² for Poisson/Bernoulli | It is intentionally `None`; use deviance and family residuals. |
| Comparing Gaussian AICc and Poisson AICc across different response definitions | Information criteria compare models fitted to the same response likelihood. |
| Using very small compact neighbourhoods with rare outcomes | Increase support or simplify the local model. |

## What to report

Report:

- family, link, response coding, and scientific distribution justification;
- exposure or offset definition and units;
- kernel, coordinate system, distance metric, fixed/adaptive semantics;
- bandwidth criterion, bounds, optimizer, and selected bandwidth;
- IWLS tolerance, iteration limit, and local convergence rate;
- deviance, null deviance, deviance explained, AIC/AICc/BIC;
- family-specific residual diagnostics;
- local coefficients on link and transformed scales where appropriate;
- treatment of sparse events, separation, overdispersion, or class imbalance;
- spatial validation design;
- pyGWRx version.

## References

- Nakaya, T., Fotheringham, A. S., Brunsdon, C., & Charlton, M. (2005). Geographically weighted Poisson regression for disease association mapping. *Statistics in Medicine*, 24, 2695–2717. [`10.1002/sim.2129`](https://doi.org/10.1002/sim.2129)
- Gollini, I., Lu, B., Charlton, M., Brunsdon, C., & Harris, P. (2015). GWmodel: An R Package for Exploring Spatial Heterogeneity Using Geographically Weighted Models. *Journal of Statistical Software*, 63(17). [`10.18637/jss.v063.i17`](https://doi.org/10.18637/jss.v063.i17)

## Related documentation

- [Generated GWGLM API](../api/models/gwglm.md)
- [Standard GWR](gwr.md)
- [GWDA classification](gwda.md)
- [Prediction and results](../guides/prediction-and-results.md)