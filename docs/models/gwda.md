# Geographically Weighted Discriminant Analysis (`GWDA`)

<div class="model-hero" markdown>

**Task:** multiclass spatial classification with geographically varying class distributions  
**Core mechanism:** estimate local class means, covariances, and priors, then apply local Gaussian discriminant rules  
**Required inputs:** numeric feature matrix `X`, class labels `y`, and coordinates `coords`  
**Independent-target prediction:** supported through `predict()`, `predict_proba()`, and `predict_entropy()`

</div>

[API reference](../api/models/gwda.md){ .md-button .md-button--primary }
[GWGLM manual](gwglm.md){ .md-button }
[Model selection guide](../getting-started/choosing-a-model.md){ .md-button }

## What GWDA is for

GWDA is a spatial extension of linear or quadratic discriminant analysis. It is designed for classification problems where class centres, class covariance structure, or class prevalence may vary geographically.

At prediction location $u$, class $g$ receives the Gaussian discriminant cost

$$
d_g(x,u)=
\frac12\log|\Sigma_g(u)|
+\frac12\{x-\mu_g(u)\}^\top\Sigma_g(u)^{-1}\{x-\mu_g(u)\}
-\log\pi_g(u).
$$

The class with the smallest cost is predicted. Probabilities are obtained by normalising the exponentiated negative costs.

- **WLDA** (`quadratic=False`) uses one locally pooled covariance matrix for all classes.
- **WQDA** (`quadratic=True`) uses one local covariance matrix per class.

WQDA is more flexible but requires substantially more local class support.

## When to use GWDA

Use GWDA when:

- the target is categorical with two or more classes;
- predictor distributions differ by class;
- class separation or prevalence may vary over space;
- local probabilities and classification uncertainty are required;
- Gaussian discriminant assumptions are a defensible approximation.

Do not choose GWDA solely because the outcome is categorical. A binary outcome modelled through local log odds may be better represented by [`GWGLM`](gwglm.md). GWDA models class-conditional predictor distributions; Bernoulli GWGLM models the conditional probability of the response.

| Situation | Better action or model |
|---|---|
| Binary response with coefficient interpretation in log odds | Bernoulli [`GWGLM`](gwglm.md) |
| Nonlinear class boundaries unrelated to Gaussian class distributions | Compare a non-spatial nonlinear classifier and spatial validation design. |
| Very small classes or rare local classes | Increase support, simplify features, or avoid WQDA. |
| Need regression coefficients rather than class-distribution summaries | Use an appropriate regression model. |
| Need unsupervised local structure | [`GWPCA`](gwpca.md) or [`GWSS`](gwss.md) |

## WLDA versus WQDA

| Property | WLDA | WQDA |
|---|---|---|
| Constructor | `quadratic=False` | `quadratic=True` |
| Covariance | One locally pooled covariance | Separate covariance for each class |
| Boundary | Linear in feature space at each location | Quadratic in feature space at each location |
| Parameters | Fewer | More |
| Local data requirement | Lower | Higher |
| Singular-covariance risk | Lower | Higher |

Begin with WLDA. Move to WQDA only when class-specific covariance differences are scientifically important and locally supported.

## Local and global class statistics

The three switches are independent:

| Parameter | `True` | `False` |
|---|---|---|
| `local_mean` | Class means are estimated with spatial weights at each evaluation location. | Each class uses one global mean. |
| `local_cov` | Class covariance matrices are estimated with spatial weights. | Each class uses one global covariance. |
| `local_prior` | Class priors follow local weighted class prevalence. | Priors follow global class proportions, unless `prior` is supplied. |

This supports controlled comparisons such as local means with global covariance and priors. A supplied `prior` overrides local/global empirical prior calculation.

## Important pyGWRx implementation detail

The maintained `GWmodel::gwda` workflow is the main reference for local class statistics and classification ordering. pyGWRx uses the standard Gaussian log-determinant discriminant formula shown above so that `predict_proba()` has a clear probabilistic interpretation. This intentionally differs from a matrix-norm term present in the published R source.

The class currently uses the shared Euclidean coordinate-distance utility and does not expose a `distance_metric` parameter. Use projected coordinates with meaningful planar units.

## Installation

```bash
pip install pygwrx
```

## Self-contained example

```python
import numpy as np
import pandas as pd

from pygwrx import GWDA

rng = np.random.default_rng(66)
n_per_class = 45
n = 2 * n_per_class
coords = rng.uniform(0.0, 100.0, size=(n, 2))
y = np.array(["A"] * n_per_class + ["B"] * n_per_class)

# Class centres change with the east-west coordinate.
class_sign = np.where(y == "A", -1.0, 1.0)
X = pd.DataFrame(
    {
        "feature_1": class_sign * (1.0 + 0.008 * coords[:, 0])
        + rng.normal(0.0, 0.65, size=n),
        "feature_2": class_sign * 0.7
        + 0.005 * coords[:, 1]
        + rng.normal(0.0, 0.55, size=n),
    }
)

model = GWDA(
    kernel="bisquare",
    bandwidth="cv",
    adaptive=True,
    quadratic=False,
    local_mean=True,
    local_cov=True,
    local_prior=True,
    regularization=1e-6,
).fit(
    X,
    y,
    coords,
    validate=True,
)

print("selected neighbours:", model.bandwidth_)
print("LOOCV accuracy:", model.correct_ratio_)
print("confusion matrix with totals:")
print(model.confusion_matrix_)
print("mean normalized entropy:", model.entropy_.mean())

X_new = pd.DataFrame(
    {
        "feature_1": [-0.8, 1.1],
        "feature_2": [-0.5, 0.9],
    }
)
coords_new = np.array([[20.0, 30.0], [80.0, 65.0]])

print("labels:", model.predict(X_new, coords_new))
print("probabilities:")
print(pd.DataFrame(model.predict_proba(X_new, coords_new), columns=model.classes_))
print("entropy:", model.predict_entropy(X_new, coords_new))
```

`model.classes_` is sorted. Probability columns follow that exact class order.

## Constructor

```python
GWDA(
    kernel="bisquare",
    bandwidth="cv",
    adaptive=True,
    quadratic=False,
    local_mean=True,
    local_cov=True,
    local_prior=True,
    prior=None,
    regularization=0.0,
    verbose=False,
)
```

## Constructor parameters

| Parameter | Default | Meaning | How to use and what can go wrong |
|---|---:|---|---|
| `kernel` | `"bisquare"` | Converts Euclidean coordinate distance into local weights. | Compact support makes local class absence and singular covariance more likely. Compare with broader support when classes are sparse. |
| `bandwidth` | `"cv"` | Positive fixed distance, adaptive integer neighbour count, `None`, or `"cv"`. | Automatic selection maximises leave-one-out classification accuracy. No AIC/AICc/BIC tokens are accepted. |
| `adaptive` | `True` | Interprets numeric bandwidth as nearest-neighbour count. | Recommended when sampling density varies. Fixed bandwidths have coordinate units. |
| `quadratic` | `False` | Selects WLDA or WQDA. | Use WQDA only with sufficient class-specific local support and covariance stability. |
| `local_mean` | `True` | Makes class centres location-specific. | Disable for a controlled model with global class centres. |
| `local_cov` | `True` | Makes class covariance estimates location-specific. | Local covariance is data hungry. Disabling it can greatly stabilise classification. |
| `local_prior` | `True` | Uses local weighted class prevalence. | Local priors can dominate predictions in imbalanced regions; inspect `class_priors_`. |
| `prior` | `None` | Fixed priors in sorted class order. | Values must be positive, finite, and sum to one. Supplying priors overrides empirical local/global priors. |
| `regularization` | `0.0` | Adds a constant ridge to covariance diagonals. | Zero reproduces the unregularised method and raises on singular/non-positive-definite covariance. A positive value is an explicit stabilising extension and must be reported. |
| `verbose` | `False` | Prints fitted method, bandwidth, and evaluation mode. |

## Data requirements

- `X` must contain at least two numeric features.
- Every class must contain at least `n_features + 1` observations globally.
- Local covariance estimation may require considerably more than that global minimum.
- Class labels may be numeric or strings but cannot contain missing values.
- DataFrame columns used for prediction must match training columns.
- Coordinates should be projected because the class does not expose an alternative distance metric.

A globally valid class can still have zero or nearly zero local kernel weight. The resulting non-positive prior or unstable covariance raises an error. Increase bandwidth or simplify the model rather than masking the failure.

## Bandwidth selection

```python
selected = model.select_bandwidth(
    X,
    y,
    coords,
    bounds=None,
)
```

The criterion is leave-one-out classification accuracy. During each candidate evaluation, the focal training observation receives zero spatial weight.

- Adaptive searches examine all integer values when the range is at most 180 neighbours; wider ranges use bounded optimisation plus a local integer check.
- Fixed searches use bounded scalar optimisation.
- Ties prefer the smaller candidate because candidates are ordered by `(-accuracy, bandwidth)`.
- `bandwidth_scores_` stores evaluated `(bandwidth, accuracy)` pairs.
- Candidate failures caused by local covariance/support problems score as zero accuracy.

A single accuracy optimum may be unstable under class imbalance. Inspect balanced accuracy, class-specific recall, probabilities, and bandwidth sensitivity separately.

## Fitting and validation modes

```python
model.fit(
    X,
    y,
    coords,
    X_pred=None,
    coords_pred=None,
    validate=True,
)
```

| Configuration | `validation_mode_` | Stored result |
|---|---|---|
| No prediction rows, `validate=True` | `"leave-one-out"` | LOOCV labels, probabilities, entropy, confusion matrix, and `correct_ratio_`. |
| No prediction rows, `validate=False` | `"training"` | In-sample evaluation including each focal observation. This is optimistic and does not set formal validation accuracy. |
| `X_pred` and `coords_pred` supplied together | `"prediction"` | Results for supplied rows; training state remains available for later prediction methods. |

For ordinary usage, fit with `validate=True`, then call prediction methods for new rows. Supplying `X_pred` inside `fit()` is available for reference-workflow compatibility but is not required.

## Prediction methods

| Method | Returns |
|---|---|
| `predict(X, coords)` | Class labels. |
| `predict_proba(X, coords)` | Probability matrix `(n_targets, n_classes)` in `classes_` order. |
| `predict_entropy(X, coords)` | Normalised Shannon entropy from 0 to 1. |
| `get_entropy()` | Copy of entropy stored by the most recent fit evaluation. |

Entropy near zero means one class dominates the fitted probability vector. Entropy near one means probabilities are close to uniform. Low entropy is not automatically correctness; a confidently wrong local model also has low entropy.

GWDA currently does not expose `to_frame()`. Build a result table explicitly:

```python
prob = model.predict_proba(X_new, coords_new)
result = pd.DataFrame(prob, columns=[f"prob_{c}" for c in model.classes_])
result["prediction"] = model.predict(X_new, coords_new)
result["entropy"] = model.predict_entropy(X_new, coords_new)
```

## Main fitted attributes

| Attribute | Meaning |
|---|---|
| `classes_` | Sorted class labels and probability-column order. |
| `class_counts_` | Global observations per class. |
| `fixed_prior_` | Validated user prior or `None`. |
| `bandwidth_`, `bandwidth_scores_` | Final bandwidth and evaluated CV results. |
| `class_means_` | Dictionary mapping each class to local/global mean arrays. |
| `class_covariances_` / `class_covs_` | Class covariance arrays; second name is a compatibility alias. |
| `pooled_covariances_` | Local pooled covariance for WLDA; `None` for WQDA. |
| `class_priors_` | Dictionary of class-prior vectors by evaluation location. |
| `discriminant_scores_` / `log_posteriors_` | Gaussian costs; lower is better. The alias name should not be read as literal unnormalised log probability. |
| `predictions_` | Labels for the fit evaluation rows. |
| `probabilities_` | Normalised class probabilities for fit evaluation rows. |
| `entropy_` | Normalised classification entropy. |
| `confusion_matrix_` | GWmodel-style predicted-row/observed-column matrix with totals, available for LOOCV validation. |
| `correct_ratio_` | LOOCV accuracy when validation is performed. |

## Interpretation and validation

1. Compare a global LDA/QDA classifier under the same features.
2. Inspect class counts and spatial class distribution.
3. Fit WLDA before WQDA.
4. Examine bandwidth accuracy curves, not only the optimum.
5. Map local priors and entropy alongside predictions.
6. Evaluate class-specific recall, precision, calibration, and confusion.
7. Repeat with global/local mean, covariance, and prior switches.
8. Use spatially blocked validation for claims about new regions.

Leave-one-out validation tests omission of one row but still uses nearby observations. It is not equivalent to transferring to a geographically separated area.

## Common mistakes

| Mistake | Correction |
|---|---|
| Reading probability columns without `classes_` | Always label columns using the sorted fitted class order. |
| Using WQDA with sparse local classes | Increase bandwidth or use WLDA/global covariance. |
| Adding regularization only until fitting stops failing | Predefine and sensitivity-test it; report the value. |
| Treating local prior variation as predictor evidence | Priors reflect class prevalence, not feature separation. |
| Calling `validate=False` performance validation | It is in-sample classification. |
| Reporting accuracy alone for imbalanced classes | Include class-wise and probability-based metrics. |
| Using longitude/latitude as ordinary Euclidean coordinates | Project coordinates before fitting. |
| Interpreting low entropy as guaranteed correctness | Compare entropy with observed errors and calibration. |
| Expecting a regression-style `to_frame()` | Construct a class-probability table explicitly. |

## What to report

Report:

- class definitions, counts, and feature preprocessing;
- WLDA or WQDA;
- local/global settings for means, covariances, and priors;
- fixed priors and their class order, when used;
- covariance regularization;
- coordinate reference system, kernel, fixed/adaptive bandwidth, search bounds, and selected bandwidth;
- validation mode and whether LOOCV or spatial blocks were used;
- confusion matrix and class-specific metrics;
- probability calibration and entropy analysis;
- sensitivity to bandwidth, covariance mode, and priors;
- pyGWRx version.

## References

- Brunsdon, C., Fotheringham, A. S., & Charlton, M. E. (2007). Geographically weighted discriminant analysis. *Geographical Analysis*, 39(4), 376–396. [`10.1111/j.1538-4632.2007.00709.x`](https://doi.org/10.1111/j.1538-4632.2007.00709.x)
- Gollini, I., Lu, B., Charlton, M., Brunsdon, C., & Harris, P. (2015). GWmodel: An R Package for Exploring Spatial Heterogeneity Using Geographically Weighted Models. *Journal of Statistical Software*, 63(17). [`10.18637/jss.v063.i17`](https://doi.org/10.18637/jss.v063.i17)

## Related documentation

- [Generated GWDA API](../api/models/gwda.md)
- [Bernoulli GWGLM](gwglm.md)
- [GWSS](gwss.md)
- [GWPCA](gwpca.md)