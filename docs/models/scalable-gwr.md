# Scalable Geographically Weighted Regression (`ScalableGWR`)

<div class="model-hero" markdown>

**Task:** large-sample continuous-response local regression using the published ScaGWR approximation  
**Core mechanism:** compress Q-nearest-neighbour polynomial-kernel cross-products once, then optimise two global approximation parameters  
**Required inputs:** predictor matrix `X`, numeric response `y`, and coordinates `coords`  
**Independent-target prediction:** supported through the same compressed polynomial-kernel estimator

</div>

[API reference](../api/models/scalable-gwr.md){ .md-button .md-button--primary }
[GWR manual](gwr.md){ .md-button }
[Performance guide](../guides/performance-and-reproducibility.md){ .md-button }

## What ScalableGWR actually is

ScalableGWR implements the ScaGWR estimator introduced by Murakami and colleagues. It is not ordinary GWR with a smaller distance matrix, and it is not exact GWR evaluated on only a random subset.

Conventional GWR repeatedly builds location-specific weighted cross-products. ScaGWR instead:

1. finds a fixed number \(Q\) of nearest neighbours for every evaluation location;
2. represents a continuous Gaussian or exponential kernel by a finite polynomial basis;
3. pre-compresses local \(X^\top W X\), \(X^\top W y\), and inference moments for each basis term;
4. optimises only a global kernel-scale parameter and a global shrinkage parameter;
5. assembles local systems from the compressed moments without revisiting all pairwise distances.

For a fixed predictor count, polynomial degree, and neighbour count, this avoids an \(n\times n\) distance matrix and gives the algorithm its approximately linear sample-size scaling.

## Polynomial-kernel approximation

For each target \(i\), neighbour \(j\), and basis term \(r\), pyGWRx forms a transformed base-kernel value \(\phi_r(d_{ij})\) and precomputes

$$
A_{i,r}=\sum_{j\in\mathcal N_Q(i)}\phi_r(d_{ij})x_jx_j^\top,
\qquad
b_{i,r}=\sum_{j\in\mathcal N_Q(i)}\phi_r(d_{ij})x_jy_j.
$$

The fitted `scale_` determines non-negative basis coefficients that sum to one. The assembled local system is

$$
\left\{\sum_r c_r A_{i,r}+\lambda X^\top X\right\}\beta_i
=
\sum_r c_r b_{i,r}+\lambda X^\top y,
$$

where `penalty_` is \(\lambda\ge 0\).

The penalty is not a conventional local ridge term toward zero. It adds the global OLS cross-products and therefore shrinks unstable local estimates toward the global relationship. `numerical_jitter`, by contrast, is a separate diagonal stabiliser added directly to every local system.

## When to use ScalableGWR

Use it when:

- the response is continuous and a Gaussian local linear model is appropriate;
- standard GWR's quadratic distance/matrix cost is prohibitive;
- a polynomial-kernel approximation is acceptable;
- the nearest-neighbour count \(Q\) can be fixed in advance or sensitivity-tested;
- target-location coefficients and predictions are required at large scale.

Do not select it merely because its class name contains “Scalable”. It estimates a different approximation model from conventional GWR.

| Situation | Better starting point |
|---|---|
| Exact conventional GWR remains computationally feasible | Fit [`GWR`](gwr.md) as the principal reference. |
| Different predictors require different scales | [`MGWR`](mgwr.md), with a much higher computational cost. |
| Local collinearity is the primary issue | [`LCRGWR`](lcr-gwr.md) or [`GWLasso`](gw-lasso.md). |
| A compact bisquare/tricube kernel is scientifically required | Current ScalableGWR supports only continuous Gaussian and exponential base kernels. |
| A fixed-distance rather than Q-neighbour approximation is required | Use conventional GWR; ScaGWR fixes an adaptive neighbour count. |

## Critical terminology: `bandwidth` is Q

In this class, `bandwidth` means the number of nearest neighbours \(Q\). It is always an integer and `adaptive` is always true internally.

`optimize_bandwidth=True` is retained as the public parameter name, but it does **not** optimise Q. It optimises:

- `scale_`: the global polynomial-basis mixing scale;
- `penalty_`: the global OLS-shrinkage strength.

The fitted neighbour count remains:

```python
model.bandwidth_ == model.bandwidth
```

A complete sensitivity analysis must therefore refit several Q values explicitly.

## Installation

```bash
pip install pygwrx
```

## Self-contained example

```python
import numpy as np
import pandas as pd

from pygwrx import ScalableGWR

rng = np.random.default_rng(161)
n = 2500
coords = rng.uniform(0.0, 1000.0, size=(n, 2))
X = pd.DataFrame(
    {
        "income": rng.normal(size=n),
        "access": rng.normal(size=n),
        "density": rng.normal(size=n),
    }
)

beta_income = 0.8 + 0.0012 * coords[:, 0]
beta_access = -1.1 + 0.0008 * coords[:, 1]
y = (
    3.0
    + beta_income * X["income"].to_numpy()
    + beta_access * X["access"].to_numpy()
    + 0.5 * X["density"].to_numpy()
    + rng.normal(0.0, 0.4, size=n)
)

model = ScalableGWR(
    bandwidth=120,          # fixed Q-neighbour count
    kernel="gaussian",
    polynomial=4,
    criterion="cv",
    optimize_bandwidth=True,
    sample_size=600,        # calibration targets only; all rows remain neighbours
    random_state=42,
).fit(X, y, coords)

print("Q:", model.bandwidth_)
print("scale:", model.scale_)
print("global shrinkage penalty:", model.penalty_)
print("full-data CV RMSE:", model.cv_score_)
print(model.summary())
print(model.to_frame().head())

X_new = pd.DataFrame(
    {
        "income": [0.2, -0.4],
        "access": [0.9, 0.1],
        "density": [0.0, 0.5],
    }
)
coords_new = np.array([[250.0, 300.0], [750.0, 650.0]])

prediction = model.predict_result(
    X_new,
    coords_new,
    return_standard_errors=True,
)
print(prediction.to_frame())
```

`sample_size` reduces only the number of target locations used to optimise `scale` and `penalty` under CV. All training rows remain available as neighbours and in the global shrinkage term. The final fit, full-data CV RMSE, coefficients, and inference use all observations.

## Constructor

```python
ScalableGWR(
    bandwidth=100,
    kernel="gaussian",
    polynomial=4,
    criterion="cv",
    optimize_bandwidth=True,
    scale=None,
    penalty=None,
    fit_intercept=True,
    sample_size=None,
    random_state=None,
    optimizer_maxiter=200,
    numerical_jitter=0.0,
    verbose=False,
)
```

## Constructor parameters

### Approximation structure

| Parameter | Default | Meaning | How to use and what can go wrong |
|---|---:|---|---|
| `bandwidth` | `100` | Fixed number \(Q\) of nearest neighbours used to construct compressed local moments. | Must be at least 2, smaller than `n_samples`, and greater than the number of design columns. It is not automatically selected. Small Q increases locality and instability; large Q increases work and smoothness. |
| `kernel` | `"gaussian"` | Continuous base kernel, `gaussian`/`gau` or `exponential`/`exp`. | Compact kernels are not supported because the published polynomial construction targets continuous kernels. |
| `polynomial` | `4` | Positive polynomial degree; the implementation stores `polynomial + 1` basis terms. | Higher degree increases compressed-moment memory and computation. It may improve approximation flexibility but does not guarantee better transfer performance. |
| `fit_intercept` | `True` | Adds a spatially varying intercept to the design. | Do not add a constant predictor manually. Constant predictor columns are rejected even when the intercept is disabled. |

### Scale and global shrinkage

| Parameter | Default | Meaning and guidance |
|---|---:|---|
| `optimize_bandwidth` | `True` | Optimises `scale` and `penalty` with L-BFGS-B in log space. | The name does not mean Q is optimised. Set false for a fully fixed ScaGWR specification. |
| `scale` | `None` | Positive fixed scale when optimisation is disabled, or initial value when enabled. | `None` starts from 1.0. The fitted value controls polynomial-basis coefficients, not a coordinate-distance bandwidth. |
| `penalty` | `None` | Non-negative global OLS-shrinkage value, fixed or used as optimiser start. | `None` starts from 0.01. A larger value pulls local systems more strongly toward the global cross-products. It is distinct from `numerical_jitter`. |
| `criterion` | `"cv"` | Optimisation objective: leave-one-out residual sum of squares or ScaGWR AICc. | CV excludes each calibration target from its Q-neighbour set. AICc uses in-sample compressed moments and smoother trace. Compare criteria only under the same Q and polynomial structure. |
| `optimizer_maxiter` | `200` | Maximum L-BFGS-B iterations. | A non-converged optimiser with a finite solution emits a warning. Inspect `optimization_result_` rather than assuming convergence. |
| `numerical_jitter` | `0.0` | Non-negative diagonal value added after the published global penalty term. | Use only for explicit numerical stabilisation. Positive values alter the estimator and must be reported separately from `penalty_`. |

### Calibration subsampling

| Parameter | Default | Meaning and guidance |
|---|---:|---|
| `sample_size` | `None` | Random number of target sites used to optimise scale/penalty under CV. | All observations still contribute as neighbours and to global moments. It is ignored with AICc and triggers a warning. Minimum is 2. |
| `random_state` | `None` | Seed for selecting calibration targets when `sample_size<n`. | Set for reproducibility and repeat with alternative seeds to assess calibration stability. |
| `verbose` | `False` | Prints final Q, scale, penalty, and CV RMSE. |

## How the base bandwidth is derived

ScalableGWR internally calculates `base_bandwidth_` from the median distance to a reference neighbour rank:

- reference rank is `min(50, Q) - 1` among leave-one-out neighbours;
- Gaussian base bandwidth is reference distance divided by `sqrt(3)`;
- exponential base bandwidth is reference distance divided by 3.

This value anchors the polynomial basis. It is not a user-selected final bandwidth and should not be confused with Q or `scale_`.

Duplicated coordinates are allowed only when the data still contain positive neighbour distances from which this reference can be derived. Completely coincident coordinates raise an error.

## Fitting

```python
model.fit(X, y, coords)
```

The fit always performs the following:

- validates and stores the complete training design;
- builds a `cKDTree`, not a full pairwise distance matrix;
- calibrates or fixes scale and penalty;
- computes final coefficients for every training location;
- calculates standard errors, t values, p values, smoother traces, effective degrees of freedom, AIC/AICc, R², adjusted R², and full-data LOOCV RMSE.

There are no switches for disabling inference. Memory is dominated by Q-neighbour indices/distances and compressed arrays rather than \(n^2\) matrices.

Approximate compressed cross-product storage scales as

$$
O\{n(P+1)k^2\},
$$

where \(P\) is polynomial degree and \(k\) is the design-column count. Neighbour data scale as \(O(nQ)\). A large predictor count can therefore still be expensive even when sample scaling is linear.

## Prediction and coefficient-only evaluation

```python
pred = model.predict(X_new, coords_new)
result = model.predict_result(
    X_new,
    coords_new,
    return_standard_errors=False,
)
```

`predict_result()` also accepts `X=None` to estimate target coefficients without predictions:

```python
coefficient_surface = model.predict_result(
    None,
    coords_new,
    return_standard_errors=True,
)
```

When `X=None`, the returned coefficient matrix is valid, but `predictions` is `None`. Target standard errors require squared compressed moments and extra computation.

Prediction uses exactly Q nearest training observations plus the fitted global penalty term. It does not rebuild or optimise `scale_` and `penalty_` for each target.

## Main fitted attributes

| Attribute | Meaning |
|---|---|
| `bandwidth_` | Fixed Q-neighbour count. |
| `base_bandwidth_` | Reference distance scale used by the polynomial basis. |
| `scale_` | Fitted or fixed global polynomial-basis scale. |
| `penalty_` | Fitted or fixed global OLS-shrinkage strength. |
| `optimization_result_` | SciPy L-BFGS-B result, or `None` when optimisation is disabled. |
| `coefficients_` | Full local parameter matrix including intercept when fitted. |
| `coef_`, `intercept_` | Local slopes and intercepts. |
| `standard_errors_`, `t_values_`, `p_values_` | Local inference arrays. |
| `trace_S_`, `trace_StS_` | ScaGWR smoother traces. |
| `effective_n_params_` | `2 trace(S) - trace(S'S)`. |
| `effective_df_` | `n - effective_n_params_`. |
| `sigma_` | Residual standard deviation using effective residual degrees of freedom. |
| `cv_score_` | Full-sample LOOCV RMSE, regardless of calibration subsampling. |
| `aic_`, `aicc_`, `r2_`, `adjusted_r2_` | Global fit diagnostics. |
| `global_cross_product_`, `global_response_product_` | Global OLS moments used by the shrinkage term. |

`to_frame()` includes coordinates, observed/fitted/residual values, every coefficient, standard error, t statistic, and p value.

## Interpreting scale and penalty

### Scale

`scale_` changes the relative polynomial-basis coefficients. It should not be reported as metres, kilometres, or neighbour count. Its meaning is conditional on:

- base kernel;
- polynomial degree;
- internally derived `base_bandwidth_`;
- fixed Q.

### Penalty

`penalty_` controls borrowing from the global OLS cross-products. A large value can stabilise local systems and make coefficient surfaces more global. It is an empirical-Bayes-style global shrinkage component of ScaGWR, not evidence that spatial non-stationarity is absent.

### Q sensitivity

Because Q is fixed outside optimisation, repeat the complete fit over several plausible neighbour counts. Track:

- CV RMSE and AICc;
- scale and penalty;
- coefficient stability;
- optimiser convergence;
- prediction performance under spatial blocks;
- runtime and memory.

## ScalableGWR versus conventional GWR

| Feature | ScalableGWR | GWR |
|---|---|---|
| Distance storage | KD-tree Q-neighbour queries | Conventional pairwise/local distance operations |
| Neighbour count | Fixed Q, manually selected | Fixed distance or adaptive count automatically selectable |
| Kernel | Polynomial approximation to Gaussian/exponential | Direct Gaussian, bisquare, exponential, tricube, boxcar, or callable |
| Global shrinkage | Published penalty toward global OLS moments | None in standard GWR |
| Scale optimisation | Two global approximation parameters | Geographic bandwidth criterion |
| Exact equality | Approximation estimator | Conventional local weighted least squares |
| Large-n use | Designed for it | Can become quadratic in time/memory |

A benchmark should compare both predictive accuracy and coefficient recovery, not runtime alone.

## Common mistakes

| Mistake | Correction |
|---|---|
| Saying `optimize_bandwidth=True` selects Q | It optimises scale and penalty only; refit Q explicitly. |
| Interpreting `scale_` as a distance bandwidth | It controls polynomial-basis mixing and has no direct coordinate unit. |
| Treating `penalty_` as ordinary ridge-to-zero | It adds global OLS cross-products and shrinks toward the global relationship. |
| Using `numerical_jitter` without reporting it | It is a separate diagonal modification of every local system. |
| Setting `bandwidth >= n` | Q must be smaller than sample size because CV needs non-self neighbours. |
| Setting Q barely above the design size | Increase Q to provide stable local support and sensitivity-test it. |
| Using constant predictor columns | They are rejected; remove them and let `fit_intercept` handle the intercept. |
| Assuming `sample_size` reduces the final fitted dataset | It only subsamples CV calibration targets; final fitting and diagnostics use all rows. |
| Using `sample_size` with AICc | It is ignored. |
| Claiming exact equivalence to GWR | ScaGWR is a polynomial-kernel approximation with global shrinkage. |
| Ignoring optimiser warnings | Inspect `optimization_result_` and test starting values/Q. |

## Recommended validation workflow

1. Fit conventional GWR on a manageable subset or smaller benchmark dataset.
2. Choose several Q values larger than the design dimension.
3. Fit ScalableGWR under CV and/or AICc for each Q.
4. Inspect optimiser convergence, scale, penalty, and base bandwidth.
5. Compare coefficient surfaces with exact GWR where feasible.
6. Use spatially blocked target prediction.
7. Repeat calibration subsampling with several seeds when `sample_size` is used.
8. Report speed, peak memory, accuracy, and approximation differences together.

## What to report

Report:

- sample size, predictor count, and coordinate system;
- fixed Q and its sensitivity analysis;
- base kernel and polynomial degree;
- criterion and whether calibration targets were subsampled;
- random seed and sample size;
- optimiser start values, iteration limit, convergence status, fitted scale, and fitted penalty;
- `base_bandwidth_`;
- numerical jitter;
- effective parameters, CV RMSE, AICc, and coefficient uncertainty;
- comparison with conventional GWR;
- spatial validation, runtime, and memory measurements;
- pyGWRx version.

## References

- Murakami, D., Yoshida, T., Seya, H., Griffith, D. A., & Yamagata, Y. (2021). Scalable GWR: A Linear-Time Algorithm for Large-Scale Geographically Weighted Regression with Polynomial Kernels. *Annals of the American Association of Geographers*, 111(2), 459–480. [`10.1080/24694452.2020.1774350`](https://doi.org/10.1080/24694452.2020.1774350)

## Related documentation

- [Generated ScalableGWR API](../api/models/scalable-gwr.md)
- [Standard GWR](gwr.md)
- [Performance and reproducibility](../guides/performance-and-reproducibility.md)
- [Prediction and result objects](../guides/prediction-and-results.md)