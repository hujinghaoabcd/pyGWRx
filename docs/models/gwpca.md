# Geographically Weighted Principal Component Analysis (`GWPCA`)

<div class="model-hero" markdown>

**Task:** exploratory local dimension reduction and covariance-structure analysis  
**Core mechanism:** perform a geographically weighted, locally centred SVD at each evaluation location  
**Required inputs:** multivariate numeric matrix `X` and observation coordinates `coords`  
**Transformation capability:** scores are available only for locations whose loading matrices were explicitly fitted

</div>

[API reference](../api/models/gwpca.md){ .md-button .md-button--primary }
[GWSS manual](gwss.md){ .md-button }
[Model selection guide](../getting-started/choosing-a-model.md){ .md-button }

## What GWPCA is for

Global PCA assumes one covariance structure and one loading matrix for the entire study area. GWPCA asks whether dominant multivariate dimensions and variable contributions change geographically.

At evaluation location $u$, pyGWRx decomposes

$$
\sqrt{W(u)}\{X^*-\bar X_w(u)\}=UDV^\top,
$$

where $X^*$ is the globally centred or standardised data matrix and $\bar X_w(u)$ is a local weighted mean. Columns of $V$ are local loading vectors. Local component variances are calculated from squared singular values divided by the local weight sum.

GWPCA has no response variable. It is not a regression model and its component scores are not predictions of an outcome.

## When to use GWPCA

Use it when:

- several numeric variables may share local latent dimensions;
- global PCA could conceal changing covariance structure;
- local loading maps and local explained variance are substantively meaningful;
- dimension reduction is exploratory rather than a fixed global preprocessing step;
- a small number of components can be supported inside every local neighbourhood.

| Goal | Better method |
|---|---|
| Describe local means, dispersion, or pairwise correlations directly | [`GWSS`](gwss.md) |
| Explain or predict a continuous response | A regression model such as [`GWR`](gwr.md) |
| Classify labelled observations | [`GWDA`](gwda.md) |
| Transform arbitrary future coordinates without refitting | Current GWPCA does not interpolate loading surfaces; fit those coordinates through `eval_coords`. |

## Global preprocessing followed by local centring

The `scaling` option controls the first stage:

- `scaling=True`: subtract the global sample mean and divide by global sample standard deviation;
- `scaling=False`: subtract the global mean only.

Both paths then subtract a separate local weighted mean before SVD. Therefore `local_means_` are means on the globally processed scale, not raw-variable means.

Use scaling when variables have different units or variances. Without scaling, high-variance variables can dominate local components. A zero-variance variable makes `scaling=True` invalid and must be removed or reconsidered.

## Sign indeterminacy

Principal-component signs are arbitrary: multiplying a loading vector and its scores by -1 represents the same component. pyGWRx applies a deterministic convention in which the largest absolute loading in each component is made positive. This improves reproducibility but does not create an intrinsically positive scientific direction.

Do not interpret a sign difference between software packages before accounting for sign alignment.

## Installation

GWPCA uses scikit-learn for the stored global PCA reference:

```bash
pip install "pygwrx[ml]"
```

## Self-contained example

```python
import numpy as np
import pandas as pd

from pygwrx import GWPCA

rng = np.random.default_rng(88)
n = 90
coords = rng.uniform(0.0, 100.0, size=(n, 2))

latent_west = rng.normal(size=n)
latent_east = rng.normal(size=n)
east_weight = coords[:, 0] / 100.0
west_weight = 1.0 - east_weight

X = pd.DataFrame(
    {
        "housing": west_weight * latent_west + 0.2 * rng.normal(size=n),
        "income": west_weight * 0.8 * latent_west + 0.3 * rng.normal(size=n),
        "industry": east_weight * latent_east + 0.2 * rng.normal(size=n),
        "emissions": east_weight * 0.9 * latent_east + 0.3 * rng.normal(size=n),
    }
)

model = GWPCA(
    n_components=2,
    kernel="bisquare",
    bandwidth="cv",
    adaptive=True,
    scaling=True,
    compute_scores=False,
).fit(
    X,
    coords,
    compute_cv=True,
)

print("selected neighbours:", model.bandwidth_)
print(model.summary())
print(model.to_frame().head())

# Local PC scores for the same observation/evaluation locations.
focal_scores = model.transform(X, coords)
print(focal_scores[:5])

pc1_winner = pd.Series(
    np.asarray(model.feature_names_)[model.get_winning_variable(0)]
)
print(pc1_winner.value_counts())
```

## Evaluation locations and transformation

By default, local loadings are fitted at the observation coordinates. A separate set of reporting locations can be supplied:

```python
reporting_coords = np.array([[20.0, 20.0], [50.0, 50.0], [80.0, 80.0]])
reporting_model = GWPCA(
    n_components=2,
    bandwidth=35,
    adaptive=True,
).fit(
    X,
    coords,
    eval_coords=reporting_coords,
)
```

To score rows at these locations, supply exactly one row per fitted evaluation coordinate:

```python
X_reporting = X.iloc[:3].copy()  # illustrative feature rows
scores = reporting_model.transform(X_reporting, reporting_coords)
```

`transform()` does not find the nearest fitted loading or interpolate between loading matrices. Every supplied coordinate must match exactly one fitted `eval_coords_` row. To transform a new set of locations, refit with those locations in `eval_coords`.

## Constructor

```python
GWPCA(
    n_components=2,
    kernel="bisquare",
    bandwidth="cv",
    adaptive=True,
    scaling=True,
    compute_scores=False,
    verbose=False,
)
```

## Constructor parameters

| Parameter | Default | Meaning | How to use and what can go wrong |
|---|---:|---|---|
| `n_components` | `2` | Number of local loading vectors retained. | Must be between 1 and the number of variables. Every positive-weight local window needs at least `n_components + 1` observations. More components increase storage and reduce the dimensionality advantage. |
| `kernel` | `"bisquare"` | Euclidean spatial weighting kernel. | Compact support can produce too few positive-weight observations. Gaussian/exponential support is broader but still bandwidth-sensitive. |
| `bandwidth` | `"cv"` | Positive fixed distance, adaptive integer, `None`, or `"cv"`. | Only CV string selection is supported. `"adaptive"` is deliberately rejected; use `adaptive=True`. |
| `adaptive` | `True` | Interprets numeric bandwidth as neighbour count. | Useful for uneven sampling density. Adaptive values must be integers from 2 to sample size, but practical support must exceed the component minimum. |
| `scaling` | `True` | Globally standardises variables before local centring and SVD. | Usually retain when units differ. Disable only when raw variance magnitude is scientifically intended. |
| `compute_scores` | `False` | Stores local score matrices for every positively weighted observation in every evaluation window. | This can consume substantial memory and creates a list whose row counts vary by location. It is not required for focal scores or `transform()`. |
| `verbose` | `False` | Prints fit dimensions and bandwidth. |

The class currently uses Euclidean coordinate distance and does not expose `distance_metric`.

## Bandwidth selection

```python
bandwidth = model.select_bandwidth(X, coords)
```

Selection follows the GWmodel-compatible leave-one-out criterion and golden-section routine.

For each omitted observation, the retained local components reconstruct its processed feature row. The stored benchmark criterion is the **square of the sum of reconstruction residual components**, not the more common sum of squared component-wise residuals. This definition is retained intentionally for numerical compatibility with published/reference GWmodel examples.

Consequences:

- the selected bandwidth is tied to `n_components` and `scaling`;
- changing component count requires a new bandwidth search;
- the adaptive golden search is not exhaustive and follows the reference floor/round update sequence;
- CV should be interpreted as a reference-compatible reconstruction criterion, not downstream-response prediction error.

## Fitting

```python
model.fit(
    X,
    coords,
    eval_coords=None,
    compute_cv=False,
)
```

| Fit argument | Meaning |
|---|---|
| `X` | Numeric matrix with at least two rows and two variables. DataFrame names are retained. |
| `coords` | Observation coordinates aligned with `X`. |
| `eval_coords` | Optional coordinates where local loading matrices are calibrated. Original observations remain the weighted data. |
| `compute_cv` | Stores one leave-one-out CV contribution per observation at the final bandwidth. It is separate from automatic bandwidth selection. |

`compute_cv=False` does not prevent automatic bandwidth selection when `bandwidth="cv"`; it only controls whether final-bandwidth contributions are retained in `cv_scores_`.

## Main fitted attributes

| Attribute | Shape/type | Interpretation |
|---|---:|---|
| `global_mean_`, `global_scale_` | `(p,)` | Fitted preprocessing parameters reused by `transform()`. |
| `pca_global_` | sklearn PCA object | Global reference PCA on processed data. |
| `eval_coords_` | `(m, 2)` | Locations of local loading matrices. |
| `loadings_` | `(m, p, k)` | Local loading vectors. |
| `var_` | `(m, p)` | Local component variance array; first `k` values correspond to retained components, remaining values retain total-variance information. |
| `local_pv_` | `(m, k)` | Percent local variance explained by each retained component. |
| `cumulative_pv_` | `(m,)` | Total percent variance explained by retained components. |
| `local_means_` | `(m, p)` | Local means on the globally processed scale. |
| `focal_scores_` | `(n, k)` or `None` | Scores for observations at their own coordinates, available only when `eval_coords` is omitted. |
| `scores_` | list or `None` | Per-location score matrices for all positive-weight observations when `compute_scores=True`. |
| `weights_` | `(m, n)` | Raw local spatial weights. |
| `cv_scores_` | `(n,)` or `None` | Final-bandwidth LOOCV contributions. |
| `feature_names_` | list | Variable names used for winning-variable reporting. |

## Methods and exports

| Method | Returns and limitation |
|---|---|
| `transform(X, coords=None)` | Locally centred scores using already fitted loading matrices. Coordinates must match fitted evaluation locations exactly. |
| `get_winning_variable(component=0)` | Index of the largest absolute loading for each evaluation location. |
| `to_frame()` | Local percentage variance by retained component, cumulative percentage, and PC1 winning-variable name. It does not include coordinates or all loadings. |
| `summary()` | Global explained variance and local cumulative-variance distribution. |

To create a complete spatial loading table, combine `eval_coords_` and `loadings_` explicitly:

```python
records = []
for location, coord in enumerate(model.eval_coords_):
    for variable, name in enumerate(model.feature_names_):
        records.append(
            {
                "x": coord[0],
                "y": coord[1],
                "variable": name,
                "loading_pc1": model.loadings_[location, variable, 0],
                "loading_pc2": model.loadings_[location, variable, 1],
            }
        )
loading_frame = pd.DataFrame(records)
```

## Interpretation workflow

1. Inspect distributions and global PCA first.
2. Fit GWPCA with scaling justified by units.
3. Check selected bandwidth and local support.
4. Map cumulative explained variance to identify where the retained dimension is adequate.
5. Map loading magnitude, sign-aligned patterns, and winning variables.
6. Compare neighbouring loading vectors after accounting for possible component ordering changes.
7. Repeat across plausible bandwidths and component counts.
8. Validate any downstream use separately; GWPCA exploration is not predictive validation.

Local components can switch order when eigenvalues are close. A “PC1” surface may therefore represent different combinations across space. Examine local variance gaps and full loading vectors rather than only the winning variable.

## Common mistakes

| Mistake | Correction |
|---|---|
| Calling GWPCA a regression model | It has no response and estimates local multivariate structure. |
| Treating loading signs as intrinsically meaningful | Signs are arbitrary; pyGWRx only applies a reproducible convention. |
| Transforming arbitrary new coordinates | Refit with those coordinates in `eval_coords`. |
| Assuming `to_frame()` includes coordinates/loadings | Join `eval_coords_` and `loadings_` manually. |
| Using variables with incompatible scales while `scaling=False` | Standardise or justify raw-variance dominance. |
| Keeping a component count that explains little variance in some regions | Map `cumulative_pv_` and reconsider local dimensionality. |
| Ignoring component-order switching | Compare eigenvalue/variance separation and full loading patterns. |
| Enabling `compute_scores=True` for a dense grid without memory planning | Store only what downstream analysis requires. |
| Interpreting CV as response-prediction performance | It is a reference-compatible PCA reconstruction criterion. |
| Using longitude/latitude directly | Project coordinates because distance is Euclidean. |

## What to report

Report:

- variables, units, transformations, and missing-value treatment;
- global centring/scaling choice;
- number of retained components;
- coordinate reference system;
- kernel and fixed/adaptive bandwidth;
- automatic CV or manual bandwidth and its reference-compatible criterion;
- observation and evaluation-location design;
- local percentage/cumulative variance patterns;
- loading sign convention and component-order checks;
- bandwidth and component-count sensitivity;
- whether local score matrices or final CV contributions were stored;
- exploratory status and downstream validation procedure;
- pyGWRx version and `[ml]` installation extra.

## References

- Harris, P., Brunsdon, C., & Charlton, M. (2011). Geographically weighted principal components analysis. *International Journal of Geographical Information Science*, 25(10), 1717–1736. [`10.1080/13658816.2011.554838`](https://doi.org/10.1080/13658816.2011.554838)
- Gollini, I., Lu, B., Charlton, M., Brunsdon, C., & Harris, P. (2015). GWmodel: An R Package for Exploring Spatial Heterogeneity Using Geographically Weighted Models. *Journal of Statistical Software*, 63(17). [`10.18637/jss.v063.i17`](https://doi.org/10.18637/jss.v063.i17)

## Related documentation

- [Generated GWPCA API](../api/models/gwpca.md)
- [GWSS](gwss.md)
- [GWDA](gwda.md)
- [Model selection](../getting-started/choosing-a-model.md)