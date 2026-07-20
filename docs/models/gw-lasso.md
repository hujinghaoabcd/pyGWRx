# Geographically Weighted Lasso (`GWLasso`)

<div class="model-hero" markdown>

**Task:** continuous-response local regression with coefficient shrinkage and location-specific variable selection  
**Core mechanism:** solve one locally weighted L1-penalised regression at every calibration or target location  
**Required inputs:** predictor matrix `X`, numeric response `y`, coordinates `coords`  
**Independent-target prediction:** supported by recalibrating a local Lasso at each target location

</div>

[API reference](../api/models/gw-lasso.md){ .md-button .md-button--primary }
[LCRGWR manual](lcr-gwr.md){ .md-button }
[Model selection guide](../getting-started/choosing-a-model.md){ .md-button }

## What GWLasso is for

Ordinary GWR estimates every predictor at every location. Under local collinearity or a large candidate predictor set, coefficient surfaces can become unstable and difficult to interpret. GWLasso adds an L1 penalty to each local regression so that weak local coefficients are shrunk and may become exactly zero.

At evaluation location $s$, the fitted model solves

$$
\frac{1}{2\sum_i w_i(s)}
\sum_i w_i(s)
\left[y_i-\beta_0(s)-x_i^\top\beta(s)\right]^2
+\lambda(s)\lVert\beta^*(s)\rVert_1,
$$

where $\beta^*(s)$ refers to coefficients on locally standardised predictors. The intercept is not penalised.

The selected variables may differ across locations. This makes GWLasso useful for exploratory local variable selection, but it also introduces selection instability and a second tuning problem beyond spatial bandwidth.

## When to use GWLasso

Use it when:

- the response is continuous;
- the candidate predictor set is larger than is comfortable for ordinary local regression;
- local variable selection is scientifically meaningful;
- predictor relevance may differ by location;
- exact zeros and local selection-frequency summaries are desired;
- local standardisation is acceptable and will be reported.

Do not treat GWLasso as a universal cure for collinearity. If the scientific objective is to retain every predictor and stabilise estimates, [`LCRGWR`](lcr-gwr.md) is more directly aligned.

| Main objective | Better model or action |
|---|---|
| Retain all predictors while compensating high local condition numbers | [`LCRGWR`](lcr-gwr.md) |
| Reduce response-outlier influence | [`RGWR`](rgwr.md) |
| Use a known global/local partition | [`MixedGWR`](mixed-gwr.md) |
| Estimate different spatial scales without sparsity | [`MGWR`](mgwr.md) |
| Formal post-selection inference | Requires a dedicated inferential design beyond the current class. |

## Two nested tuning problems

GWLasso has two conceptually different tuning layers.

### 1. Spatial bandwidth

`bandwidth="cv"` performs a global leave-one-out search over `n_bandwidths` candidates. At each candidate bandwidth, one focal observation is removed and a local Lasso is fitted for its location. The selected bandwidth minimises LOOCV RMSE.

### 2. Local Lasso penalty

`alpha="cv"` selects a separate penalty at every calibration or target location using deterministic shuffled folds controlled by `random_state`. A local logarithmic penalty path is generated unless `alpha_grid` is supplied.

This means an automatic fit can involve many bandwidth candidates × many locations × many local alpha candidates × CV folds. Computational cost can rise quickly.

!!! warning "Local selection does not imply local causation"
    A variable can be selected because it proxies for another process or because local sample support changes. Selection maps require stability analysis, not only visual interpretation.

## Installation

GWLasso requires the machine-learning extra:

```bash
pip install "pygwrx[ml]"
```

## Self-contained example

```python
import numpy as np
import pandas as pd

from pygwrx import GWLasso

rng = np.random.default_rng(33)
n = 80
coords = rng.uniform(0.0, 100.0, size=(n, 2))
X = pd.DataFrame(
    {
        "x1": rng.normal(size=n),
        "x2": rng.normal(size=n),
        "x3": rng.normal(size=n),
        "noise": rng.normal(size=n),
    }
)

# x1 matters mostly in the west, x2 mostly in the east, x3 everywhere.
west = coords[:, 0] < 50.0
beta1 = np.where(west, 1.6, 0.0)
beta2 = np.where(west, 0.0, -1.4)
y = (
    2.0
    + beta1 * X["x1"].to_numpy()
    + beta2 * X["x2"].to_numpy()
    + 0.7 * X["x3"].to_numpy()
    + rng.normal(0.0, 0.35, size=n)
)

model = GWLasso(
    kernel="exponential",
    bandwidth="cv",
    alpha="cv",
    adaptive=True,
    n_bandwidths=8,
    n_alphas=25,
    cv_folds=5,
    standardize=True,
    random_state=42,
).fit(X, y, coords)

print("selected bandwidth:", model.bandwidth_)
print("selection frequency:")
print(pd.Series(model.selection_frequency_, index=X.columns))
print("all local fits converged:", model.converged_.all())
print(model.to_frame().head())
```

Prediction recalibrates both local coefficients and, when `alpha="cv"`, the target-location penalty:

```python
X_new = X.iloc[:2].copy()
coords_new = np.array([[25.0, 40.0], [75.0, 40.0]])

pred = model.predict(X_new, coords_new)
params = model.predict_parameters(coords_new)
print(pred)
print(params.to_frame())
```

## Constructor

```python
GWLasso(
    kernel="exponential",
    bandwidth="cv",
    alpha="cv",
    alpha_grid=None,
    n_alphas=30,
    alpha_min_ratio=1e-3,
    cv_folds=5,
    standardize=True,
    adaptive=False,
    bandwidth_range=None,
    n_bandwidths=8,
    max_iter=5000,
    tol=1e-6,
    active_tol=1e-8,
    fit_intercept=True,
    distance_metric="euclidean",
    random_state=0,
    verbose=False,
)
```

## Constructor parameters

### Spatial neighbourhood

| Parameter | Default | Meaning and guidance |
|---|---:|---|
| `kernel` | `"exponential"` | Spatial weighting function. Wheeler's original implementation used an exponential kernel; pyGWRx supports all standard kernels. Compact kernels may leave small local samples and increase CV failures. |
| `bandwidth` | `"cv"` | Numeric bandwidth, `"cv"`, `"adaptive"`, or `None`. `"adaptive"` is a convenience token that enables adaptive-neighbour CV. |
| `adaptive` | `False` | Numeric bandwidth is an integer neighbour count when true. The `"adaptive"` bandwidth token sets this effectively to true. |
| `bandwidth_range` | `None` | Optional lower and upper bounds for the global bandwidth candidate grid. |
| `n_bandwidths` | `8` | Number of global bandwidth candidates. | More candidates improve search resolution but multiply total local fits. Always inspect `bandwidth_scores_`. |
| `distance_metric` | `"euclidean"` | Spatial distance definition. Use projected coordinates or deliberate Haversine coordinates. |

Automatic GWLasso bandwidth selection supports CV, not AIC/AICc/BIC tokens.

### Local Lasso penalty

| Parameter | Default | Meaning | How to choose and what can go wrong |
|---|---:|---|---|
| `alpha` | `"cv"` | Non-negative fixed local Lasso penalty, or locally selected penalty when `"cv"`. | A fixed value gives the same penalty strength everywhere after local standardisation. Local CV is more flexible but substantially more expensive and may produce noisy alpha surfaces. `alpha=0` gives local weighted least squares under the same centring/scaling conventions. |
| `alpha_grid` | `None` | Explicit positive penalty candidates used at every local CV problem. | Supply for reproducibility or controlled comparisons. Values may be ascending or descending. An inappropriate grid can place the optimum at an edge. |
| `n_alphas` | `30` | Number of generated logarithmic candidates when `alpha_grid=None`. | More candidates increase resolution and cost. Must be at least 2. |
| `alpha_min_ratio` | `1e-3` | Smallest generated alpha as a fraction of local `alpha_max`. | Smaller ratios explore weaker penalties and denser models. Must lie strictly between 0 and 1. |
| `cv_folds` | `5` | Local deterministic shuffled folds for alpha selection. | Must be at least 2 and feasible for effective local support. Small neighbourhoods can make fold estimates unstable. |
| `standardize` | `True` | Locally centre and scale predictors before penalisation. | Usually keep true because L1 penalties depend on variable scale. Turning it off makes penalty effects depend on raw units. |

### Optimisation and selection reporting

| Parameter | Default | Meaning and guidance |
|---|---:|---|
| `max_iter` | `5000` | Maximum coordinate-descent iterations for each local Lasso. | Inspect `converged_`; isolated failures may indicate poor scaling, too-tight tolerance or difficult local design. |
| `tol` | `1e-6` | Coordinate-descent convergence tolerance. | Smaller values increase precision and cost. Keep fixed across model comparisons. |
| `active_tol` | `1e-8` | Absolute final coefficient threshold used to mark a variable selected. | This affects selection indicators and frequencies but not the fitted optimisation itself. Report it when interpreting selection maps. |
| `fit_intercept` | `True` | Fits an unpenalised local intercept. | Do not manually add a constant. |
| `random_state` | `0` | Seed for deterministic local CV folds. | Retain and report it. Repeat with alternative seeds when selection stability is important. |
| `verbose` | `False` | Prints bandwidth candidates and fit progress. | Useful because automatic fitting can be lengthy. |

## Fitting

```python
model.fit(X, y, coords)
```

Unlike GWR, the public `fit()` method has no separate hat-matrix or inference switches. The class focuses on local penalised estimation and variable selection rather than classical smoother-matrix inference.

At fit time pyGWRx stores:

- local coefficient and intercept arrays;
- one local alpha per calibration location;
- active-variable indices;
- local objective, iteration count and convergence flag;
- local alpha-CV score;
- bandwidth candidate table;
- selection frequency by predictor;
- approximate global diagnostics using the mean active-variable count.

!!! caution "Diagnostics use approximate model complexity"
    The reported diagnostic feature count is based on the mean number of active variables plus the intercept. It is not an exact Lasso degrees-of-freedom or post-selection inference calculation.

## Main fitted attributes

| Attribute | Shape/type | Interpretation |
|---|---|---|
| `bandwidth_` | scalar | Selected fixed distance or adaptive neighbour count. |
| `bandwidth_scores_` | DataFrame or `None` | Candidate bandwidth, LOOCV RMSE and failed-location count. |
| `coef_` | `(n, p)` | Local coefficients on the original predictor scale. |
| `intercept_` | `(n,)` | Local unpenalised intercepts. |
| `alpha_` | `(n,)` | Fixed or locally selected penalty at each calibration location. |
| `active_vars_` | list of arrays | Selected predictor indices at every location. |
| `selection_frequency_` | `(p,)` | Fraction of calibration locations with coefficient magnitude above `active_tol`. |
| `mean_active_variables_` | scalar | Mean selected-predictor count per location. |
| `local_objective_` | `(n,)` | Final local penalised objective. |
| `n_iter_` | `(n,)` | Coordinate-descent iterations per location. |
| `converged_` | `(n,)` boolean | Local convergence flags. |
| `local_alpha_cv_score_` | `(n,)` | Local alpha CV score where alpha selection is used. |
| `diagnostics_` | dictionary | Fit diagnostics based on approximate active-variable complexity. |

`get_variable_importance()` returns a copy of `selection_frequency_`. It is a selection-frequency summary, not a causal importance score.

## Prediction semantics

`predict_parameters(coords_new)` recalibrates a local Lasso using the stored training data and target spatial weights. `predict(X_new, coords_new)` then applies those target coefficients.

When `alpha="cv"`, a new local CV problem is solved at every target coordinate. Consequently:

- target prediction can be computationally expensive;
- target alpha values may differ from nearby calibration alpha values;
- predictions remain tied to the training response and local support;
- coefficient maps should not be interpolated manually as a substitute.

## How to interpret selection maps

### Selection frequency

A frequency near 1 means a predictor is active at most calibration locations under the fitted tuning specification. It does not mean the coefficient has a stable sign or magnitude. Always examine both selection and coefficient maps.

### Local alpha

Larger alpha generally produces more shrinkage and fewer active variables. Spatial alpha patterns can also reflect sampling density, predictor variance, noise and fold instability—not only substantive process differences.

### Zero coefficients

An exact zero is conditional on:

- bandwidth and kernel;
- local standardisation;
- alpha candidates and CV folds;
- random seed;
- correlated alternatives;
- `active_tol` for the reported selection flag.

Correlated predictors can substitute for one another across adjacent locations, creating patchy selection maps.

## Recommended stability analysis

1. Fit a global Lasso and ordinary GWR baseline.
2. Inspect bandwidth candidate scores and failed locations.
3. Confirm every local fit converged.
4. Repeat local alpha selection with several seeds.
5. Vary `n_alphas`, `alpha_min_ratio` or an explicit grid.
6. Compare selection maps under plausible bandwidths and kernels.
7. Map coefficient sign and magnitude alongside selection.
8. Use spatially blocked validation for predictive claims.

## GWLasso versus LCRGWR

| Feature | GWLasso | LCRGWR |
|---|---|---|
| Penalty | L1 | L2 |
| Main goal | Local sparsity and variable selection | Stabilise locally collinear designs while retaining variables |
| Local tuning | Alpha fixed or locally CV-selected | Lambda determined by condition threshold or fixed baseline |
| Exact zero coefficients | Yes | Usually no |
| Classical local SE/t output | Not exposed | Available with penalisation caveats |
| Key audit output | Active sets, selection frequency, alpha, convergence | Condition numbers, local lambda, compensation masks |

## Common mistakes

| Mistake | Correction |
|---|---|
| Installing only `pygwrx` and encountering a missing ML dependency | Install `pygwrx[ml]`. |
| Comparing raw alpha values after changing `standardize` | Local scaling changes penalty meaning; hold the setting fixed. |
| Treating `selection_frequency_` as effect size | Examine coefficient magnitude/sign and stability separately. |
| Ignoring local non-convergence | Map `converged_`, inspect scaling and adjust optimisation settings. |
| Using too few bandwidth candidates and calling the result optimal | Inspect `bandwidth_scores_` and refine around promising values. |
| Treating CV-selected local models as formally inference-ready | Post-selection uncertainty is not provided by this class. |
| Interpreting patchy selection as sharp process boundaries | Check correlated-variable substitution and seed/tuning stability. |
| Forgetting that target prediction reruns local alpha selection | Budget computation and inspect target `alphas` from `predict_parameters()`. |

## What to report

Report:

- candidate predictors and preprocessing;
- coordinate system, distance metric, kernel and bandwidth semantics;
- bandwidth candidate range/count and selected bandwidth;
- fixed or local-CV alpha strategy;
- explicit alpha grid or generated path settings;
- local standardisation setting;
- CV folds and random seed;
- optimisation tolerance, iteration limit and convergence rate;
- `active_tol`;
- selection-frequency and coefficient stability analyses;
- predictive validation design;
- approximate nature of fit-complexity diagnostics;
- pyGWRx version and ML extra.

## References

- Wheeler, D. C. (2009). Simultaneous coefficient penalization and model selection in geographically weighted regression: The geographically weighted lasso. *Environment and Planning A*, 41(3), 722–742. [`10.1068/a40256`](https://doi.org/10.1068/a40256)

## Related documentation

- [Generated GWLasso API](../api/models/gw-lasso.md)
- [LCRGWR](lcr-gwr.md)
- [Standard GWR](gwr.md)
- [Model selection](../getting-started/choosing-a-model.md)