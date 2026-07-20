# Multiscale Geographically Weighted Regression (`MGWR`)

<div class="model-hero" markdown>

**Task:** continuous-response local additive regression with coefficient-specific spatial scales  
**Core assumption:** the intercept and each predictor may operate at a different bandwidth  
**Required inputs:** predictor matrix `X`, numeric response `y`, coordinates `coords`  
**Independent-target prediction:** intentionally not exposed in the current validated API

</div>

[API reference](../api/models/mgwr.md){ .md-button .md-button--primary }
[GWR manual](gwr.md){ .md-button }
[Kernels and bandwidths](../guides/kernels-and-bandwidths.md){ .md-button }

## Why MGWR exists

Standard GWR estimates all local coefficients with one shared bandwidth. That assumption means every relationship is forced to vary over the same spatial scale. MGWR relaxes it by assigning a separate bandwidth to every fitted parameter, including the intercept when an intercept is fitted.

For predictors $x_{ij}$, MGWR represents the response as

$$
y_i = \beta_0(s_i) + \sum_{j=1}^{p} x_{ij}\beta_j(s_i) + \varepsilon_i,
$$

but each coefficient surface $\beta_j(s)$ is smoothed with its own bandwidth $b_j$. The model is calibrated through additive backfitting: one coefficient contribution is updated from a partial residual while the other contributions are held fixed, and the process repeats until the score of change converges.

This distinction is substantive rather than cosmetic. A small bandwidth suggests a relationship supported at a relatively local scale; a large bandwidth suggests broad-scale or nearly global variation. The bandwidth is part of the scientific result, not merely a tuning parameter.

## Use MGWR when

- a continuous-response linear model is appropriate;
- theory or preliminary evidence suggests that predictors operate at different spatial scales;
- the scale of each relationship is itself an important result;
- standard GWR appears to over-localise some coefficients and over-smooth others;
- calibration-location coefficient surfaces and multiscale diagnostics are the primary outputs.

## Do not make MGWR the first choice when

| Situation | Better action or model |
|---|---|
| The global linear model and GWR have not yet been examined | Begin with a global model and [`GWR`](gwr.md). |
| New-location prediction is the main production requirement | Use a model with a validated target prediction operator, such as GWR, or design a separate validation workflow. |
| Some effects are known to be exactly global rather than merely broad-scale | Use [`MixedGWR`](mixed-gwr.md) with an explicit global/local partition. |
| The response is non-Gaussian | Use [`GWGLM`](gwglm.md); the current MGWR class is Gaussian. |
| Response outliers or local collinearity dominate the analysis | Diagnose those problems first; consider [`RGWR`](rgwr.md), [`LCRGWR`](lcr-gwr.md) or [`GWLasso`](gw-lasso.md). |
| The dataset is very large | MGWR backfitting and exact smoother diagnostics can be expensive. Reduce the design, test on a representative subset, or use a scalable alternative where scientifically acceptable. |

!!! important "MGWR is not GWR with a list of arbitrary neighbourhoods"
    The variable-specific bandwidths are coupled through additive backfitting. Each term is estimated from partial residuals that depend on the other terms, so bandwidths and coefficient surfaces must be interpreted as one fitted system.

## Published method and pyGWRx scope

The 2017 MGWR method formalised the idea that different processes can operate at different geographic scales. The widely used Python `mgwr` implementation later documented a backfitting-based software workflow. pyGWRx follows the same central model structure while defining its own explicit public contract:

- Gaussian additive coefficient-specific local regressions;
- one bandwidth for the intercept and one for every predictor when `fit_intercept=True`;
- manual shared, manual variable-specific, or automatically selected bandwidths;
- fixed-distance or adaptive-neighbour bandwidth semantics;
- CV, AIC, AICc or BIC bandwidth criteria;
- stored bandwidth and convergence histories;
- exact smoother traces and variable-specific effective parameter counts;
- optional local covariance, standard-error and t-statistic computation;
- no independent-target `predict()` contract.

The absence of target prediction is deliberate. A stable, validated operator for independently supplied locations is not currently part of the package contract. Use `fitted_values_` for calibration-location estimates and evaluate transfer with a refitted spatial holdout workflow rather than calling `predict()`.

## Installation

```bash
pip install pygwrx
```

## Input data contract

| Input | Shape | Meaning | MGWR-specific concern |
|---|---:|---|---|
| `X` | `(n, p)` | Numeric predictors | Predictor order determines the order of slope bandwidths. DataFrame columns are strongly recommended. |
| `y` | `(n,)` | Continuous numeric response | Must align row-by-row with `X` and `coords`. |
| `coords` | `(n, d)`; normally `(n, 2)` | Locations used for spatial distances | Distance units define fixed bandwidth units. Use projected coordinates or deliberately select Haversine distance. |

When `fit_intercept=True` and `X` has columns `income` and `access`, the fitted parameter order is:

```text
[Intercept, income, access]
```

Therefore a manual bandwidth vector must contain three values in exactly that order.

## Self-contained example: automatic multiscale selection

```python
import numpy as np
import pandas as pd

from pygwrx import MGWR

rng = np.random.default_rng(7)
n = 72
coords = rng.uniform(0.0, 100.0, size=(n, 2))

X = pd.DataFrame(
    {
        "local_driver": rng.normal(size=n),
        "broad_driver": rng.normal(size=n),
    }
)

# One coefficient changes quickly with x; the other changes gradually with y.
beta_local = 0.8 + 0.9 * np.sin(coords[:, 0] / 13.0)
beta_broad = -1.2 + 0.006 * coords[:, 1]
y = (
    3.5
    + beta_local * X["local_driver"].to_numpy()
    + beta_broad * X["broad_driver"].to_numpy()
    + rng.normal(0.0, 0.30, size=n)
)

model = MGWR(
    kernel="bisquare",
    bandwidths=None,          # select one bandwidth per fitted parameter
    bandwidth_method="aicc",
    adaptive=True,
    max_iter=100,
    tol=1e-5,
).fit(
    X,
    y,
    coords,
    compute_hat_matrix=False,
    store_partial_hat_matrices=False,
    compute_inference=True,
    n_chunks=2,
)

parameter_names = ["Intercept", *X.columns]
for name, bandwidth, enp in zip(
    parameter_names,
    model.bandwidths_,
    model.effective_params_by_variable_,
):
    print(f"{name:>12}: bandwidth={bandwidth}, ENP={enp:.3f}")

print("converged:", model.converged_)
print("iterations:", model.n_iter_)
print(model.to_frame().head())
```

Automatic MGWR selection is substantially more expensive than one GWR fit because it repeatedly searches and updates coefficient-specific smoothers. Begin with a modest predictor set and inspect convergence before scaling up.

## Self-contained example: fixed bandwidth vector

Manual bandwidths are useful for reproducing a published specification, sensitivity analysis, or separating coefficient fitting from bandwidth search.

```python
from pygwrx import MGWR

manual = MGWR(
    kernel="bisquare",
    adaptive=True,
    # Intercept, local_driver, broad_driver
    bandwidths=[50, 24, 60],
    init_bandwidth=40,
    max_iter=100,
    tol=1e-5,
).fit(
    X,
    y,
    coords,
    compute_hat_matrix=False,
    store_partial_hat_matrices=False,
    compute_inference=False,
)

print(manual.bandwidths_)
print(manual.converged_)
```

A scalar `bandwidths=40` is accepted and repeats the same value for every parameter. That is mainly useful for controlled comparisons; it removes the central multiscale advantage and should not be described as automatically calibrated MGWR.

## Constructor

```python
MGWR(
    kernel="bisquare",
    bandwidths=None,
    bandwidth_method="aicc",
    adaptive=True,
    bandwidth_range=None,
    bandwidth_ranges=None,
    init_bandwidth=None,
    optimization_method="golden_section",
    search_tol=1e-6,
    search_max_iter=200,
    max_iter=200,
    tol=1e-5,
    rss_score=False,
    bws_same_times=5,
    fit_intercept=True,
    distance_metric="euclidean",
    sigma2_v1=True,
    verbose=False,
)
```

## Constructor parameters

### Neighbourhood and bandwidth specification

| Parameter | Default | Meaning | How to choose and what can go wrong |
|---|---:|---|---|
| `kernel` | `"bisquare"` | Common kernel shape used for every coefficient-specific smoother. | Bisquare is compact and makes local sample support explicit. Gaussian/exponential tails use every observation with decreasing weight. Kernel sensitivity should be checked after the main bandwidth specification is stable. |
| `bandwidths` | `None` | Final manual bandwidth specification. `None` requests variable-specific selection; a scalar repeats across parameters; a sequence supplies one value per fitted parameter. | With an intercept, sequence length must equal `p + 1`. Adaptive values must be integer neighbour counts. A wrong order assigns the intended scale to the wrong coefficient. |
| `bandwidth_method` | `"aicc"` | Criterion for the initial GWR bandwidth and coefficient-specific searches. | Supported: CV, AIC, AICc and BIC. Keep the criterion fixed when comparing bandwidth patterns. AICc is the default because effective complexity matters strongly in local smoothers. |
| `adaptive` | `True` | Interprets all spatial bandwidths as nearest-neighbour counts instead of coordinate distances. | Often appropriate for uneven sampling density. Raw fixed and adaptive bandwidth values are not comparable. |
| `bandwidth_range` | `None` | Common lower and upper search bounds for all parameters. | Useful for preventing unsupported extremely local fits. It is overridden parameter-by-parameter where `bandwidth_ranges` is supplied. |
| `bandwidth_ranges` | `None` | One search interval per fitted parameter. | The list order includes the intercept. Use only when domain knowledge supports parameter-specific limits; restrictive ranges can force boundary solutions. |
| `init_bandwidth` | `None` | Shared bandwidth for the initial GWR that starts backfitting. | `None` selects it automatically. A sensible manual value can improve reproducibility and speed, but it does not replace the final `bandwidths_`. |
| `optimization_method` | `"golden_section"` | One-dimensional search method used inside scale selection. | Grid search is useful for transparent sensitivity checks; continuous methods can be faster for fixed bandwidths. |
| `search_tol` | `1e-6` | Numerical resolution target for each bandwidth search. | Smaller values increase search precision and cost. It is distinct from the outer backfitting `tol`. |
| `search_max_iter` | `200` | Maximum iterations for each one-dimensional bandwidth search. | Increase only when searches fail to settle; inspect boundary behaviour before merely raising the limit. |

### Backfitting and inference configuration

| Parameter | Default | Meaning | How to choose and what can go wrong |
|---|---:|---|---|
| `max_iter` | `200` | Maximum additive backfitting iterations. | A reached limit sets `converged_=False` and emits a warning. Do not report final bandwidths as stable without checking convergence. |
| `tol` | `1e-5` | Outer score-of-change convergence tolerance. | Smaller values demand tighter convergence and more work. Avoid very loose values solely to make an example run quickly in research analysis. |
| `rss_score` | `False` | Uses relative RSS change when true; otherwise uses smooth-function change. | Keep the default for the standard coefficient-surface convergence check unless reproducing an RSS-based specification. Report the choice. |
| `bws_same_times` | `5` | Allows stopping repeated bandwidth searches after the complete bandwidth vector is unchanged for this many iterations. | Set to `0` to disable this shortcut. An unchanged bandwidth vector does not by itself prove that coefficient contributions are fully converged; inspect both histories. |
| `fit_intercept` | `True` | Fits a spatially varying intercept with its own bandwidth. | Do not add a constant column to `X`. When disabled, all bandwidth-vector lengths decrease by one. |
| `distance_metric` | `"euclidean"` | Spatial distance definition shared by all coefficient smoothers. | Use projected coordinates for planar distance or Haversine for longitude/latitude. A metric change alters every fitted scale. |
| `sigma2_v1` | `True` | Residual-variance denominator used for local inference. | Keep consistent across compared models. `False` uses the alternative denominator involving both `trace(S)` and `trace(S'S)`. |
| `verbose` | `False` | Prints initial bandwidth and backfitting progress. | Enable when diagnosing slow scale searches or convergence. |

## Fitting

```python
model.fit(
    X,
    y,
    coords,
    compute_hat_matrix=False,
    store_partial_hat_matrices=False,
    compute_inference=True,
    n_chunks=1,
    verbose=None,
)
```

| Fit argument | Default | Meaning and practical guidance |
|---|---:|---|
| `compute_hat_matrix` | `False` | Retains the complete final `n × n` smoother matrix. Leave false unless matrix entries are explicitly required. Exact traces are still computed. |
| `store_partial_hat_matrices` | `False` | Retains an `n × n` smoother matrix for every fitted parameter. This can be extremely memory intensive. It is not required for ordinary coefficient, ENP or inference output. |
| `compute_inference` | `True` | Computes local covariance diagonals, standard errors and t statistics. Exact smoother traces and effective parameter counts are computed regardless. |
| `n_chunks` | `1` | Divides the exact smoother calculation by columns to reduce peak working memory. | Increase for larger `n` when memory is constrained. Chunking changes memory scheduling, not the mathematical result. |
| `verbose` | `None` | Optional per-fit override of constructor verbosity. |

### Memory implications

For $q$ fitted parameters, storing partial smoother matrices requires approximately

$$
8n^2q\ \text{bytes}
$$

before array overhead. With `n=5,000` and `q=6`, this is about 1.2 GB. The full hat matrix adds another roughly 200 MB. Leave both storage switches false unless a downstream method explicitly requires the matrices.

Chunking does not remove the computational cost of exact inference, but it can reduce peak temporary memory.

## Reading the result

### Bandwidth attributes

| Attribute | Meaning |
|---|---|
| `bandwidths_` | Final coefficient-specific bandwidth vector. This is the main multiscale result. |
| `initial_bandwidth_` | Shared GWR bandwidth used to initialise backfitting. |
| `bandwidth_` | Compatibility field equal to `initial_bandwidth_`; **it is not the final MGWR bandwidth vector**. |
| `bandwidth_history_` | Bandwidth vector recorded at each backfitting iteration. |
| `convergence_history_` | Score of change at each iteration. |
| `n_iter_` | Number of completed iterations. |
| `converged_` | Whether the stopping criterion was met before `max_iter`. |

### Coefficients, contributions and inference

| Attribute | Shape | Interpretation |
|---|---:|---|
| `intercept_` | `(n,)` | Local intercept surface when fitted. |
| `coef_` | `(n, p)` | Local slope surfaces. |
| `parameter_contributions_` | `(n, q)` | Additive contribution of every fitted parameter to each fitted response. Their row-wise sum is `fitted_values_`. |
| `fitted_values_` | `(n,)` | Calibration-location fitted responses. |
| `residuals_` | `(n,)` | Calibration residuals. |
| `effective_params_by_variable_` | `(q,)` | Effective model complexity assigned to each coefficient surface. `ENP_j_` is an alias. |
| `parameter_standard_errors_` | `(n, q)` or `None` | Local standard errors when inference is enabled. |
| `parameter_t_values_` | `(n, q)` or `None` | Local coefficient-to-SE ratios. Multiple local comparisons still require caution. |
| `adjusted_alpha_by_variable_` | `(q,)` or `None` | Variable-specific adjusted significance levels where available. |
| `critical_t_values_` | `(q,)` or `None` | Corresponding variable-specific critical values. |
| `diagnostics_` | dictionary | Fit, information-criterion and exact smoother diagnostics. |

`to_frame()` exports location-indexed coefficients, fitted values, residuals and available inference fields. Always pair the table with the ordered parameter names used to interpret `bandwidths_` and `effective_params_by_variable_`.

## How to interpret MGWR bandwidths

### Small bandwidth

A smaller bandwidth indicates a coefficient surface estimated from a more local neighbourhood. It may represent a genuinely local process, but it may also reflect noise, collinearity, boundary effects or a search-range constraint.

### Large bandwidth

A large bandwidth indicates a broad-scale relationship. In an adaptive model, a bandwidth close to the sample size often suggests a nearly global effect, but it is not mathematically identical to declaring the coefficient constant. Use MixedGWR when an exactly global coefficient is part of the model specification.

### Intercept bandwidth

The intercept absorbs spatially varying baseline structure not explained by slopes. A small intercept bandwidth alongside broad slope bandwidths can indicate missing local context. It should not be ignored simply because the substantive interest lies in predictor coefficients.

### Boundary bandwidths

For every parameter, check whether the selected bandwidth equals its lower or upper bound. Boundary selection may indicate:

- an insufficient search interval;
- inadequate information to identify the scale;
- pressure toward a global effect;
- an unstable extremely local fit.

Do not rank variables by raw bandwidth alone when their uncertainty and effective complexity differ.

## Convergence checks

Run these checks after every automatic fit:

```python
print("converged:", model.converged_)
print("iterations:", model.n_iter_)
print("final score:", model.convergence_history_[-1])
print("bandwidth history:")
print(model.bandwidth_history_)
```

A scientifically usable result should have:

- `converged_ == True`, or a clearly justified sensitivity analysis when it is false;
- a decreasing or stabilising convergence history;
- stable final bandwidths across nearby initial values or reasonable search settings;
- no unexplained boundary solutions;
- coefficient surfaces that remain interpretable under plausible kernel and criterion choices.

## Prediction limitation

The current class intentionally raises `NotImplementedError` for independent-target prediction. This prevents a user from applying an unvalidated shortcut to target locations.

For calibration-location evaluation:

```python
fitted = model.fitted_values_
residuals = model.residuals_
```

For spatial transfer assessment, split the data by spatial blocks, refit MGWR on each training partition, and evaluate predictions through a separately defined and validated workflow. Do not present in-sample fitted values as out-of-area predictive performance.

## GWR versus MGWR

| Question | GWR | MGWR |
|---|---|---|
| Number of spatial bandwidths | One shared bandwidth | One per fitted parameter |
| Calibration | Independent multivariate local regressions | Coupled additive backfitting |
| Main scientific output | Local coefficient surfaces under one scale | Local coefficient surfaces **and** their variable-specific scales |
| Computational cost | Lower | Substantially higher due to repeated searches and exact multiscale inference |
| Target prediction in pyGWRx | Supported | Not exposed |
| Best use | Transparent baseline and single-scale exploration | Explicit investigation of process-specific spatial scales |

MGWR should normally be compared with GWR using the same response, predictors, coordinate treatment, kernel family and validation design. Better AICc alone is not sufficient; the selected scales and coefficient surfaces must also be stable and scientifically credible.

## Common mistakes and corrections

| Mistake | Consequence | Correction |
|---|---|---|
| Supplying two bandwidths for two predictors while fitting an intercept | The class expects three values and raises an error. | Include the intercept bandwidth first or set `fit_intercept=False`. |
| Interpreting `model.bandwidth_` as the MGWR result | It stores the initial shared GWR scale for compatibility. | Use `model.bandwidths_`. |
| Treating a sample-sized bandwidth as proof of an exactly constant coefficient | Broad smoothing is not the same as an explicit global parameter. | Compare with MixedGWR or a global restriction. |
| Reporting bandwidths without convergence status | The vector may be an unfinished backfitting state. | Report `converged_`, `n_iter_` and sensitivity to initial/search settings. |
| Enabling partial hat-matrix storage by default | Memory use grows as `n² × q`. | Keep `store_partial_hat_matrices=False` unless essential. |
| Calling `predict()` | Independent-target prediction is intentionally unavailable. | Use calibration outputs or a refitted spatial holdout workflow. |
| Using a large predictor set without collinearity checks | Backfitting can produce unstable or difficult-to-identify scales. | Reduce redundant predictors and inspect local/global collinearity. |
| Comparing bandwidths across different coordinate units or metrics | Scale values no longer have the same meaning. | Hold the coordinate and distance specification fixed. |
| Interpreting coefficient maps without ENP and uncertainty | Smoothness and effective complexity differ by variable. | Examine `effective_params_by_variable_`, SE/t fields and adjusted critical values. |

## Recommended analysis sequence

1. Fit a global linear baseline.
2. Fit standard GWR and diagnose bandwidth, residuals, influence and collinearity.
3. Fit MGWR with automatic bandwidths and conservative matrix-storage settings.
4. Check convergence and every bandwidth boundary.
5. Compare GWR and MGWR information criteria under the same specification.
6. Examine each variable's bandwidth, ENP, coefficient distribution and uncertainty together.
7. Repeat with plausible initial bandwidths, search ranges or kernels to assess stability.
8. Validate transfer claims with spatially structured refitting.

## What to report

A reproducible MGWR analysis should include:

- all data preparation and coordinate information required for GWR;
- kernel and fixed/adaptive semantics;
- bandwidth criterion and optimisation method;
- common and variable-specific search ranges;
- initial bandwidth;
- final bandwidth vector with parameter order and units;
- convergence tolerance, iteration count and convergence status;
- `rss_score` and `bws_same_times` settings;
- effective parameter count by variable;
- inference and matrix-storage switches;
- AICc/CV comparison with global regression and GWR;
- coefficient, residual, influence and collinearity diagnostics;
- boundary and sensitivity analysis;
- explicit acknowledgement that independent-target prediction was not used;
- pyGWRx version.

## References

- Fotheringham, A. S., Yang, W., & Kang, W. (2017). Multiscale Geographically Weighted Regression (MGWR). *Annals of the American Association of Geographers*, 107(6), 1247–1265. [`10.1080/24694452.2017.1352480`](https://doi.org/10.1080/24694452.2017.1352480)
- Oshan, T. M., Li, Z., Kang, W., Wolf, L. J., & Fotheringham, A. S. (2019). mgwr: A Python Implementation of Multiscale Geographically Weighted Regression for Investigating Process Spatial Heterogeneity and Scale. *ISPRS International Journal of Geo-Information*, 8(6), 269. [`10.3390/ijgi8060269`](https://doi.org/10.3390/ijgi8060269)
- Comber, A. et al. (2022). A route map for the informed application of Geographically Weighted Regression. *Geographical Analysis*, 55, 155–178. [`10.1111/gean.12316`](https://doi.org/10.1111/gean.12316)

## Related documentation

- [Generated `MGWR` API](../api/models/mgwr.md)
- [Standard GWR manual](gwr.md)
- [Mixed GWR manual](mixed-gwr.md)
- [Kernels and bandwidths](../guides/kernels-and-bandwidths.md)
- [Diagnostics and inference](../guides/diagnostics.md)