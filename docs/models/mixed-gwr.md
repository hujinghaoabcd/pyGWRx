# Mixed Geographically Weighted Regression (`MixedGWR`)

<div class="model-hero" markdown>

**Task:** continuous-response semiparametric regression with explicitly global and geographically varying coefficients  
**Core mechanism:** partition predictors before fitting, then estimate constant and local components by partial regression  
**Required inputs:** predictor matrix `X`, numeric response `y`, coordinates `coords`, and a scientifically justified variable partition  
**Independent-target prediction:** supported using constant global coefficients and target-location local recalibration

</div>

[API reference](../api/models/mixed-gwr.md){ .md-button .md-button--primary }
[GWR manual](gwr.md){ .md-button }
[MGWR manual](mgwr.md){ .md-button }

## What MixedGWR is for

Standard GWR allows every coefficient to vary geographically. That can be unnecessarily flexible when some relationships are expected to be stable over the whole study area. Mixed GWR—also called semiparametric GWR—splits predictors into two groups:

- **global variables:** one coefficient for the entire dataset;
- **local variables:** one coefficient at every calibration or target location.

The model can be written as

$$
y_i=X_{G,i}\gamma + X_{L,i}\beta(s_i)+\varepsilon_i,
$$

where $\gamma$ is globally constant and $\beta(s_i)$ varies geographically.

pyGWRx uses a partial-regression algorithm aligned with the `GWmodel::gwr.mixed` workflow. It is not MGWR backfitting: the user supplies the global/local partition, and all local variables share one spatial bandwidth.

## When to use MixedGWR

Use it when:

- the response is continuous;
- theory identifies some relationships as global and others as spatially varying;
- a simpler model than full GWR is desired;
- an exactly global coefficient is more appropriate than an MGWR coefficient with a very large bandwidth;
- the variable partition can be justified before or through a transparent model-comparison procedure.

Do not choose the partition solely from whichever coefficient map looks most attractive.

| Situation | Better starting point |
|---|---|
| No defensible global/local partition exists | Begin with GWR and spatial-variation diagnostics. |
| Each predictor may have a distinct but not necessarily global scale | [`MGWR`](mgwr.md) |
| Local outliers dominate | [`RGWR`](rgwr.md) |
| Local collinearity is the main issue | [`LCRGWR`](lcr-gwr.md) or [`GWLasso`](gw-lasso.md) |
| The response is Poisson or binary | [`GWGLM`](gwglm.md) |

## MixedGWR versus MGWR

| Feature | MixedGWR | MGWR |
|---|---|---|
| Who determines global/local status? | User supplies an explicit partition | No exact partition; scales are estimated continuously |
| Local bandwidths | One shared local bandwidth | One bandwidth per fitted parameter |
| Global coefficients | Exactly constant | A very large bandwidth can be broad but is not an explicit global restriction |
| Calibration algorithm | Partial regression | Additive backfitting |
| Independent-target prediction in pyGWRx | Supported | Not exposed |
| Best use | Theory-driven global/local separation | Process-scale investigation |

## Variable partition rules

`local_vars` and `global_vars` accept either integer indices or DataFrame column names.

- When both are `None`, all features are local.
- When only one group is supplied, the other is its complement.
- When both are supplied, together they must cover every feature exactly once.
- At least one feature must be local.
- Names require `X` to be a pandas DataFrame.
- A group cannot mix names and indices.

With

```python
X.columns == ["income", "access", "elevation"]
```

these are equivalent:

```python
local_vars=["access", "elevation"]
global_vars=["income"]
```

and

```python
local_vars=[1, 2]
global_vars=[0]
```

Column names are safer and make the scientific partition auditable.

## Intercept placement

When `fit_intercept=True`:

- `intercept_fixed=True` places the intercept in the global design and `intercept_` is a scalar;
- `intercept_fixed=False` places it in the local design and `intercept_` is a length-`n` array.

A global intercept assumes one common baseline after accounting for all predictors. A local intercept absorbs unexplained spatially varying baseline structure. This choice can materially change local slope surfaces and should be made scientifically, not as a cosmetic option.

## Installation

```bash
pip install pygwrx
```

## Self-contained example

```python
import numpy as np
import pandas as pd

from pygwrx import MixedGWR

rng = np.random.default_rng(44)
n = 85
coords = rng.uniform(0.0, 100.0, size=(n, 2))
X = pd.DataFrame(
    {
        "global_income": rng.normal(size=n),
        "local_access": rng.normal(size=n),
        "local_density": rng.normal(size=n),
    }
)

beta_access = 0.5 + 0.012 * coords[:, 0]
beta_density = -1.0 + 0.009 * coords[:, 1]
y = (
    3.0
    + 1.3 * X["global_income"].to_numpy()
    + beta_access * X["local_access"].to_numpy()
    + beta_density * X["local_density"].to_numpy()
    + rng.normal(0.0, 0.35, size=n)
)

model = MixedGWR(
    kernel="bisquare",
    bandwidth="aicc",
    adaptive=True,
    global_vars=["global_income"],
    local_vars=["local_access", "local_density"],
    intercept_fixed=True,
).fit(X, y, coords, compute_enp=True)

print("selected bandwidth:", model.bandwidth_)
print("global coefficient:", model.coef_global_)
print("local variable indices:", model.local_var_indices_)
print("AICc:", model.aicc_)
print(model.to_frame().head())
```

Prediction at new locations preserves the fitted global coefficients and recalibrates the local component:

```python
X_new = pd.DataFrame(
    {
        "global_income": [0.2, -0.3],
        "local_access": [1.0, 0.5],
        "local_density": [-0.2, 0.8],
    }
)
coords_new = np.array([[25.0, 30.0], [75.0, 65.0]])
print(model.predict(X_new, coords_new))
```

Prediction DataFrame columns must match the training names and order exactly.

## Constructor

```python
MixedGWR(
    kernel="bisquare",
    bandwidth="aicc",
    bandwidth_method="aicc",
    adaptive=True,
    local_vars=None,
    global_vars=None,
    intercept_fixed=True,
    ridge=0.0,
    fit_intercept=True,
    bandwidth_range=None,
    optimization_method="golden_section",
    distance_metric="euclidean",
    verbose=False,
)
```

## Constructor parameters

### Spatial parameters

| Parameter | Default | Meaning and guidance |
|---|---:|---|
| `kernel` | `"bisquare"` | Spatial kernel for all local variables. Global variables are not spatially weighted coefficients. |
| `bandwidth` | `"aicc"` | Numeric bandwidth or CV/AIC/AICc/BIC token. Automatic selection uses a full-GWR bandwidth under the same predictors, following the published/reference workflow. |
| `bandwidth_method` | `"aicc"` | Criterion when `bandwidth=None` or the legacy `"adaptive"` token is supplied. |
| `adaptive` | `True` | Numeric bandwidth is an integer neighbour count when true. |
| `bandwidth_range` | `None` | Optional automatic search bounds. |
| `optimization_method` | `"golden_section"` | `"golden_section"`, `"brent"`, or `"grid"` through the shared selector. |
| `distance_metric` | `"euclidean"` | Spatial distance used for the local component. |

!!! note "Automatic bandwidth selection is not mixed-model-specific optimisation"
    The selected bandwidth comes from the corresponding full GWR criterion and is then used by MixedGWR. Report this clearly when comparing partitions.

### Partition and estimation parameters

| Parameter | Default | Meaning | How to use and what can go wrong |
|---|---:|---|---|
| `local_vars` | `None` | Names or indices assigned geographically varying coefficients. | At least one local variable is required. When omitted with a supplied global group, it becomes the complement. |
| `global_vars` | `None` | Names or indices assigned constant coefficients. | When omitted with a local group, it becomes the complement. Overlap or incomplete coverage raises an error when both groups are supplied. |
| `intercept_fixed` | `True` | Places the intercept in the global design when true, local design when false. | Has no effect when `fit_intercept=False`. Examine whether a local intercept is absorbing omitted spatial context. |
| `ridge` | `0.0` | Non-negative regularisation added to both local and global normal equations. | Zero reproduces the unregularised reference algorithm with deterministic pseudo-inverse fallback. A positive value is a numerical/user-specified extension and must be reported; it is not LCR threshold compensation. |
| `fit_intercept` | `True` | Fits an intercept in the selected global or local component. | Do not add a constant column to `X`. |
| `verbose` | `False` | Prints fit progress. |

## Fitting

```python
model.fit(X, y, coords, compute_enp=True)
```

| Fit argument | Default | Meaning and guidance |
|---|---:|---|
| `compute_enp` | `True` | Computes and stores the complete mixed-model hat matrix, effective parameter count `trace(S)`, `trace(S'S)` and GWR-style information criteria. | Set false for larger samples when the full `n × n` matrix is too expensive. Diagnostics then use the number of design columns rather than exact mixed smoother complexity. |

Unlike GWR, `compute_enp=True` currently retains the full hat matrix rather than only calculating traces. Memory grows quadratically with sample size.

## Main fitted attributes

| Attribute | Shape/type | Interpretation |
|---|---|---|
| `local_var_indices_` | integer array | Original feature positions assigned to the local component. |
| `global_var_indices_` | integer array | Original feature positions assigned to the global component. |
| `coef_local_` | `(n, n_local)` | Local slopes in local-variable order. |
| `coef_global_` | `(n_global,)` | Constant slopes in global-variable order. |
| `intercept_` | scalar, `(n,)`, or `0.0` | Global, local or absent intercept according to configuration. |
| `coef_` | `(n, p)` | Full coefficient surface in original feature order; global columns repeat the constant coefficient. |
| `bandwidth_` | scalar | Shared local-component bandwidth. |
| `fitted_values_`, `residuals_` | `(n,)` | Calibration results. |
| `hat_matrix_` | `(n, n)` or `None` | Mixed smoother matrix when `compute_enp=True`. |
| `enp_` | scalar or `None` | Exact effective parameter count `trace(S)`. |
| `trace_StS_` | scalar or `None` | Squared smoother trace. |
| `aic_`, `aicc_`, `bic_` | scalars | Diagnostics from exact smoother complexity when enabled, otherwise design-column approximation. |

`to_frame()` exports the full coefficient surface in original feature order, fitted values and residuals. Because global columns are repeated down the table, also report `coef_global_` directly.

## Prediction semantics

For every target set, pyGWRx:

1. rebuilds local and global designs using the fitted partition;
2. recomputes target-to-training spatial distances;
3. refits the local coefficient component at target locations;
4. deterministically recomputes and verifies the global coefficient vector;
5. combines local and global predictions.

The fitted training state is not modified. Prediction is not interpolation of local coefficient surfaces.

## How to choose the partition

A defensible workflow is:

1. fit a global linear model;
2. fit and diagnose standard GWR;
3. identify variables with theory-supported constant effects or weak spatial variation;
4. define a small number of candidate partitions before comparing them;
5. keep kernel, bandwidth criterion, coordinate treatment and validation design fixed;
6. compare information criteria and spatially blocked prediction;
7. examine whether local coefficient surfaces and residuals remain stable;
8. avoid stepwise searching over many partitions without accounting for selection.

`test_spatial_variation()` returns descriptive local coefficient variances and the global vector. Despite the method name, it is **not a formal hypothesis test**.

## Common mistakes

| Mistake | Correction |
|---|---|
| Assuming `local_vars=None` means no local variables | With both groups omitted, all features are local. |
| Supplying names with a NumPy array | Use integer indices or pass a DataFrame. |
| Supplying overlapping or incomplete groups | When both groups are present, they must partition all features exactly once. |
| Comparing a large MGWR bandwidth with an exactly global MixedGWR coefficient as identical | Broad smoothing and an explicit constant restriction are different models. |
| Calling `test_spatial_variation()` a formal significance test | Treat it only as descriptive variance. |
| Using `ridge>0` without reporting it | It changes both local and global estimates and is not part of the unregularised reference workflow. |
| Setting `compute_enp=False` but interpreting AICc as exact mixed smoother AICc | Complexity is then approximated from design columns. |
| Changing partitions and bandwidth criteria simultaneously | Hold spatial settings fixed to isolate the partition effect. |
| Using a local intercept to absorb all unexplained spatial structure | Diagnose omitted variables and residuals; justify `intercept_fixed`. |

## What to report

Report:

- the scientific basis for every global/local assignment;
- variable names and exact partition;
- intercept placement;
- kernel, fixed/adaptive semantics and distance metric;
- bandwidth criterion, search range and selected bandwidth;
- the fact that automatic bandwidth is based on full GWR selection;
- ridge value;
- whether exact ENP/hat-matrix diagnostics were computed;
- global coefficients and local coefficient summaries/maps;
- candidate-partition comparison and selection procedure;
- residual, collinearity and spatial validation diagnostics;
- pyGWRx version.

## References

- Brunsdon, C., Fotheringham, A. S., & Charlton, M. (1999). Some notes on parametric significance tests for geographically weighted regression. *Journal of Regional Science*, 39(3), 497–524. [`10.1111/0022-4146.00146`](https://doi.org/10.1111/0022-4146.00146)
- Mei, C.-L., He, S.-Y., & Fang, K.-T. (2004). A note on the mixed geographically weighted regression model. *Journal of Regional Science*, 44(1), 143–157. [`10.1111/j.1085-9489.2004.00331.x`](https://doi.org/10.1111/j.1085-9489.2004.00331.x)
- Fotheringham, A. S., Brunsdon, C., & Charlton, M. (2002). *Geographically Weighted Regression: The Analysis of Spatially Varying Relationships*. Wiley.

## Related documentation

- [Generated MixedGWR API](../api/models/mixed-gwr.md)
- [Standard GWR](gwr.md)
- [MGWR](mgwr.md)
- [Diagnostics and inference](../guides/diagnostics.md)