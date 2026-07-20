# Similarity and Geographically Weighted Regression (`SGWR`)

<div class="model-hero" markdown>

**Task:** continuous-response local Gaussian regression using both geographic proximity and predictor-space similarity  
**Core mechanism:** combine a geographic kernel with a standardized attribute-similarity kernel  
**Required inputs:** `X`, `y`, coordinates, and a defensible set of similarity variables  
**Independent-target prediction:** supported by recomputing both weight components against the training sample

</div>

[API reference](../api/models/sgwr.md){ .md-button .md-button--primary }
[GWR manual](gwr.md){ .md-button }
[SGTWR manual](sgtwr.md){ .md-button }

## Why SGWR exists

Standard GWR assumes that geographic proximity is the only neighbourhood principle: nearby observations receive greater influence. SGWR adds a second principle—observations with similar contextual attributes may also be informative even when they are geographically distant.

For selected, standardized similarity variables $z$, pyGWRx calculates

$$
d_{ij}^{A}=\frac{1}{m}\sum_{r=1}^{m}|z_{ir}-z_{jr}|,
$$

then converts this mean absolute attribute distance into

$$
w_{ij}^{A}=\exp\left[-(d_{ij}^{A})^2\right].
$$

The final local weight is

$$
w_{ij}=\alpha w_{ij}^{G}+(1-\alpha)w_{ij}^{A}.
$$

Therefore:

- `alpha=1` gives ordinary geographic GWR;
- `alpha=0` gives similarity-only local regression;
- intermediate values combine the two neighbourhood concepts.

The combined row is divided by its maximum before local regression. This normalisation changes only the common scale of one local weight vector, not its relative weights or weighted least-squares coefficient estimate.

## When to use SGWR

Use SGWR when:

- the response is continuous and local Gaussian regression is appropriate;
- geographic proximity alone is scientifically incomplete;
- non-adjacent observations can be meaningfully related through pre-specified contextual attributes;
- the similarity variables are available for both training and target locations;
- comparison with pure GWR is part of the analysis.

Examples can include cities with similar socioeconomic structure, catchments with similar environmental conditions, or markets with similar functional characteristics despite physical separation.

Do not introduce similarity only because it improves in-sample AICc. A similarity variable defines who influences whom and therefore requires stronger justification than an ordinary predictor choice.

| Situation | Better action or model |
|---|---|
| Geography alone is a defensible neighbourhood | Begin with [`GWR`](gwr.md). |
| Time also determines relevance | [`SGTWR`](sgtwr.md) |
| Every predictor should have its own geographic/similarity balance | Current SGWR has one shared alpha; do not describe it as multiscale similarity regression. |
| Similarity must be learned from separate contextual attributes not used as predictors | The current class selects similarity variables from `X`; construct and validate the design accordingly. |
| The response or post-outcome variables are used to define similarity | Do not fit: this creates target leakage. |

## Similarity variables are part of the model specification

`similarity_vars` accepts DataFrame column names or zero-based integer indices. `None` uses every predictor.

The selected variables serve two roles when they are also regressors:

1. they enter the local regression design;
2. they determine attribute similarity and therefore the local sample weights.

This dual role is valid only when scientifically intended. A variable may be a useful predictor but a poor definition of functional similarity.

Use DataFrame names for auditability:

```python
similarity_vars=["income", "land_use", "climate"]
```

Avoid:

- variables calculated from `y`;
- future information unavailable at prediction time;
- identifiers or arbitrary encodings;
- highly duplicated representations of the same concept;
- unstable variables whose measurement scale changes between training and deployment.

## Standardization and similarity scale

With `standardize_similarity=True`, pyGWRx stores the training-sample mean and population standard deviation (`ddof=0`) for each selected variable. Prediction rows are transformed with those same values.

A zero-variance similarity variable receives scale 1, so its standardized difference remains zero and it contributes no discrimination. It should usually be removed because it adds no neighbourhood information.

With standardization disabled, variables remain in raw units. A high-range variable can then dominate the mean absolute attribute distance. Disable standardization only when raw-unit weighting is deliberate and justified.

The published SGWR kernel has no user-exposed similarity bandwidth. The similarity scale is determined by:

- selected variables;
- their standardization;
- the mean absolute difference definition;
- the mixing parameter `alpha`.

## How automatic selection works

Automatic SGWR fitting is sequential, not joint.

### Step 1: select geographic bandwidth

When `bandwidth=None` or `"aicc"`, pyGWRx fits a pure GWR and selects its geographic bandwidth by AICc. The selected bandwidth is then held fixed.

`bandwidth_kernel` may differ from the final `kernel`. This supports the documented hybrid workflow of selecting an adaptive bisquare bandwidth before fitting a different final geographic kernel. When the kernels differ, report both explicitly.

### Step 2: select alpha

When `alpha=None` or `"aicc"`, pyGWRx:

1. evaluates `alpha_grid_size` evenly spaced candidates inside `alpha_range`;
2. identifies the best finite AICc;
3. performs bounded scalar refinement between the neighbouring grid values;
4. stores all evaluations in `alpha_search_history_`.

The default `alpha_range=(0.01, 1.0)` excludes pure similarity-only regression. To allow `alpha=0` during automatic selection, use a range beginning at zero.

Automatic AICc uses the full SGWR smoother matrix. It is an in-sample complexity-adjusted criterion, not spatial transfer validation.

## Installation

```bash
pip install pygwrx
```

## Self-contained example

```python
import numpy as np
import pandas as pd

from pygwrx import SGWR

rng = np.random.default_rng(141)
n = 84
coords = rng.uniform(0.0, 100.0, size=(n, 2))

# Two spatially separated functional groups.
group = rng.integers(0, 2, size=n)
X = pd.DataFrame(
    {
        "income": rng.normal(loc=group * 1.2, scale=0.7, size=n),
        "access": rng.normal(loc=(1 - group) * 0.9, scale=0.6, size=n),
        "density": rng.normal(size=n),
    }
)

beta_income = np.where(group == 0, 0.7, 1.5)
beta_access = np.where(group == 0, -1.2, -0.4)
y = (
    3.0
    + beta_income * X["income"].to_numpy()
    + beta_access * X["access"].to_numpy()
    + 0.5 * X["density"].to_numpy()
    + rng.normal(0.0, 0.35, size=n)
)

model = SGWR(
    bandwidth="aicc",
    adaptive=True,
    kernel="bisquare",
    alpha="aicc",
    alpha_range=(0.0, 1.0),
    alpha_grid_size=15,
    similarity_vars=["income", "access"],
    standardize_similarity=True,
    store_weights=False,
).fit(X, y, coords)

print("geographic bandwidth:", model.bandwidth_)
print("geographic alpha:", model.alpha_)
print("similarity variables:", model.similarity_feature_names_)
print(model.summary())
print(model.results_frame().head())

X_new = pd.DataFrame(
    {
        "income": [0.2, 1.4],
        "access": [0.9, 0.1],
        "density": [0.0, 0.5],
    }
)
coords_new = np.array([[20.0, 25.0], [80.0, 70.0]])

prediction = model.predict_result(X_new, coords_new)
print(prediction.to_frame())
```

A good sensitivity analysis fits at least:

```python
pure_gwr = SGWR(
    bandwidth=model.bandwidth_,
    adaptive=True,
    alpha=1.0,
    similarity_vars=["income", "access"],
    store_weights=False,
).fit(X, y, coords)

pure_similarity = SGWR(
    bandwidth=model.bandwidth_,
    adaptive=True,
    alpha=0.0,
    similarity_vars=["income", "access"],
    store_weights=False,
).fit(X, y, coords)
```

The pure-similarity fit still requires coordinates because the class validates the complete SGWR data contract, although geographic weights receive zero mixing weight.

## Constructor

```python
SGWR(
    bandwidth="aicc",
    adaptive=True,
    kernel="bisquare",
    alpha="aicc",
    similarity_vars=None,
    *,
    standardize_similarity=True,
    bandwidth_kernel=None,
    bandwidth_range=None,
    alpha_range=(0.01, 1.0),
    alpha_grid_size=21,
    fit_intercept=True,
    distance_metric="euclidean",
    sigma2_v1=True,
    ridge=0.0,
    store_weights=True,
    verbose=False,
)
```

## Constructor parameters

### Geographic component

| Parameter | Default | Meaning and guidance |
|---|---:|---|
| `bandwidth` | `"aicc"` | Positive fixed distance, adaptive integer neighbour count, `None`, or automatic `"aicc"` token. | Only AICc automatic selection is accepted. It is performed on a pure GWR before alpha selection. |
| `adaptive` | `True` | Interprets numeric bandwidth as a one-based neighbour count. | Adaptive count must be at least `n_design_columns + 1`. |
| `kernel` | `"bisquare"` | Geographic kernel used in the final SGWR fit. | The similarity kernel remains fixed as `exp(-d²)` and is not changed by this parameter. |
| `bandwidth_kernel` | `None` | Optional kernel used only by the automatic pure-GWR bandwidth selector. | `None` uses the final kernel. A different value creates a deliberate hybrid specification. |
| `bandwidth_range` | `None` | Optional pure-GWR AICc search bounds. | Check whether the selected bandwidth reaches a boundary. |
| `distance_metric` | `"euclidean"` | Euclidean, Manhattan/cityblock, Chebyshev, or Haversine geographic distance. | Haversine expects longitude/latitude ordering and changes fixed-bandwidth units. |

### Similarity component and mixing

| Parameter | Default | Meaning | Guidance and failure modes |
|---|---:|---|---|
| `alpha` | `"aicc"` | Geographic mixing proportion in `[0,1]`, or automatic AICc selection. | Interpret alpha only together with selected variables and standardization. It is not the percentage of explained variance due to geography. |
| `similarity_vars` | `None` | Predictor names or indices used for attribute distance. | `None` uses all predictors. Select variables from scientific reasoning and deployment availability. Empty selection is rejected. |
| `standardize_similarity` | `True` | Uses training mean and population SD before attribute differences. | Keep true when scales differ. Prediction always reuses fitted training statistics. |
| `alpha_range` | `(0.01,1.0)` | Bounds for automatic alpha search. | Include zero explicitly when similarity-only regression should be a candidate. A boundary optimum requires sensitivity analysis. |
| `alpha_grid_size` | `21` | Number of deterministic coarse candidates before local refinement. | Larger values improve initial resolution and increase repeated full local fits. Minimum is 3. |

### Estimation and storage

| Parameter | Default | Meaning and guidance |
|---|---:|---|
| `fit_intercept` | `True` | Fits a local intercept. Do not add a manual constant. |
| `sigma2_v1` | `True` | Uses `RSS / (n - trace(S))`; false uses the alternative smoother denominator. |
| `ridge` | `0.0` | Optional non-negative ridge on slope diagonals; intercept is unpenalized. | This is a numerical/user extension, not the published SGWR similarity mechanism. Report positive values. |
| `store_weights` | `True` | Retains spatial, similarity, and combined `n × n` training matrices. | Disable when weight decomposition is unnecessary. The full `hat_matrix_` is still retained by the current implementation. |
| `verbose` | `False` | Prints selection and final AICc. |

## Fitting and memory

```python
model.fit(X, y, coords)
```

The current public fit method has no switches for hat-matrix storage or inference. It always calculates and stores:

- the complete `n × n` hat matrix;
- local covariance factors, standard errors, and t values;
- smoother diagnostics, influence, Cook's distance, and local R².

With `store_weights=True`, three additional `n × n` matrices are retained. Approximate float64 storage is therefore:

```text
hat matrix:                 8 n² bytes
three component matrices: 24 n² bytes
```

before temporary arrays and overhead. At `n=10,000`, those four matrices alone are roughly 3.2 GB. Set `store_weights=False` for larger problems, but recognize that standard SGWR still retains the hat matrix.

## Prediction semantics

```python
pred = model.predict(X_new, coords_new)
result = model.predict_result(X_new, coords_new)
```

For each target row, pyGWRx:

1. validates predictor columns against training order;
2. transforms target similarity variables with training mean and scale;
3. computes target-to-training geographic weights;
4. computes target-to-training similarity weights;
5. combines them with fitted `alpha_`;
6. recalibrates a local weighted regression from the stored training response.

The target predictor values therefore affect both the regression prediction and, when selected as similarity variables, the neighbourhood used to fit the target coefficients.

This creates a deployment requirement: every similarity variable must be known reliably at prediction time. A future or unavailable attribute makes the prediction contract invalid.

## Main fitted attributes

| Attribute | Meaning |
|---|---|
| `bandwidth_` | Selected geographic distance or adaptive count. |
| `alpha_`, `alpha_score_` | Final geographic mixing proportion and selected AICc. |
| `alpha_search_history_` | Every evaluated alpha/AICc pair, including bounded refinement evaluations. |
| `similarity_indices_`, `similarity_feature_names_` | Selected similarity-variable positions and names. |
| `similarity_mean_`, `similarity_scale_` | Training transformation reused for targets. |
| `bandwidth_selector_` | Fitted pure GWR selector when bandwidth was automatic. |
| `spatial_weights_`, `similarity_weights_`, `combined_weights_` | Optional component matrices. |
| `coef_`, `intercept_`, `fitted_values_`, `residuals_` | Local calibration results. |
| `parameter_standard_errors_`, `parameter_t_values_` | Local inference arrays. |
| `influence_`, `standardized_residuals_`, `cooks_distance_`, `local_r2_` | Local diagnostics. |
| `diagnostics_`, `sigma2_`, `hat_matrix_` | Global smoother diagnostics and stored full smoother. |

Use `results_frame()`, not `to_frame()`, for calibration-location output. Prediction result objects provide their own `to_frame()`.

## Interpreting alpha and neighbour profiles

`alpha_` is the relative mixing coefficient between two weight matrices. It does **not** decompose model variance or prove that geography accounts for a particular percentage of the outcome.

A small alpha means attribute similarity has strong influence under the chosen variables and standardization. It can also indicate:

- a geographically misspecified model;
- broad omitted spatial structure;
- similarity variables that partially encode the response;
- overfitting to dense long-range connections.

Inspect actual neighbour profiles:

```python
location = 0
weight_table = pd.DataFrame(
    {
        "geographic": model.spatial_weights_[location],
        "similarity": model.similarity_weights_[location],
        "combined": model.combined_weights_[location],
    }
)
print(weight_table.sort_values("combined", ascending=False).head(10))
```

This requires `store_weights=True`. Compare whether high combined-weight remote observations are scientifically plausible.

## Recommended validation

1. Fit global regression and ordinary GWR.
2. Predefine candidate similarity-variable sets.
3. Fit SGWR and inspect geographic/similarity/combined neighbour profiles.
4. Compare `alpha=1`, fitted alpha, and `alpha=0` where estimable.
5. Repeat under alternative standardization and variable sets.
6. Use spatially separated validation, not only AICc.
7. Ensure target similarity variables are computed without outcome or future leakage.
8. Check coefficient, influence, and residual stability.

Because similarity creates long-range links, ordinary random cross-validation can be especially optimistic: a held-out location may still be closely connected in attribute space to training observations.

## Common mistakes

| Mistake | Correction |
|---|---|
| Using `y`, residuals, or outcome-derived classes as similarity variables | Define similarity only from legitimate predictors/context available at deployment. |
| Assuming `similarity_vars=None` means no similarity | It means all predictors define similarity. |
| Interpreting alpha as explained-variance share | It is a weight-matrix mixing coefficient. |
| Forgetting default alpha search excludes zero | Set `alpha_range=(0.0, 1.0)` when pure similarity should be considered. |
| Claiming bandwidth and alpha were jointly optimized | Bandwidth is selected by pure GWR first; alpha is selected second. |
| Using different final and selector kernels without reporting both | Report `kernel` and `bandwidth_kernel`. |
| Disabling standardization while mixing variables with different units | Standardize or justify raw-unit dominance. |
| Calling `to_frame()` on the fitted model | Use `results_frame()`. |
| Setting `store_weights=False` and assuming quadratic memory disappears | The full hat matrix remains stored. |
| Validating with random splits only | Use spatial and functional-similarity-aware holdouts. |

## What to report

Report:

- response and all predictors;
- exact similarity-variable set and scientific rationale;
- similarity standardization and training-only transformation;
- attribute-distance and similarity-kernel definitions;
- coordinate system, distance metric, geographic kernel, and fixed/adaptive bandwidth;
- pure-GWR bandwidth criterion, bounds, and selector kernel;
- alpha search range, grid size, final alpha, and boundary behaviour;
- ridge and residual-variance convention;
- stored-weight and memory settings;
- geographic/similarity/combined neighbour examples;
- comparison with GWR and similarity-only sensitivity;
- leakage controls and spatial/functional validation design;
- pyGWRx version.

## References

- Lessani, M. N., & Li, Z. (2024). SGWR: similarity and geographically weighted regression. *International Journal of Geographical Information Science*, 38(7), 1232–1255. [`10.1080/13658816.2024.2342319`](https://doi.org/10.1080/13658816.2024.2342319)

## Related documentation

- [Generated SGWR API](../api/models/sgwr.md)
- [Standard GWR](gwr.md)
- [SGTWR](sgtwr.md)
- [Kernels and bandwidths](../guides/kernels-and-bandwidths.md)
- [Prediction and result objects](../guides/prediction-and-results.md)