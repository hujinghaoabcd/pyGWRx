# Geographically Weighted Summary Statistics (`GWSS`)

<div class="model-hero" markdown>

**Task:** exploratory local descriptive statistics for one or more numeric variables  
**Core mechanism:** calculate weighted moments, pairwise associations, and optional quantiles at each summary location  
**Required inputs:** numeric data matrix `X` and observation coordinates `coords`  
**Evaluation away from observations:** supported through `summary_coords`; this is not response prediction

</div>

[API reference](../api/models/gwss.md){ .md-button .md-button--primary }
[GWPCA manual](gwpca.md){ .md-button }
[Model selection guide](../getting-started/choosing-a-model.md){ .md-button }

## What GWSS is for

GWSS describes how a variable distribution or pairwise association changes across space before a regression, classification, or dimension-reduction model is imposed. It can reveal:

- spatial variation in local level and dispersion;
- regions with asymmetric local distributions;
- changing pairwise covariance and correlation;
- differences between Pearson and rank association;
- local median and interquartile structure that is less driven by extremes.

GWSS does not estimate response coefficients, make class predictions, or test causality. It is an exploratory local-statistics estimator.

## Statistics calculated

For every summary location and variable, pyGWRx can calculate:

| Attribute | Statistic | Interpretation caution |
|---|---|---|
| `local_mean_` | Normalised weighted mean | Sensitive to extreme observations and bandwidth. |
| `local_var_` | Normalised weighted second central moment | This moment definition is not the same denominator used by pairwise covariance. |
| `local_std_` | Square root of local variance | In original variable units. |
| `local_skewness_` | Weighted third central moment divided by local SD cubed | Undefined when local SD is zero. |
| `local_cv_` | Local SD divided by local mean | Unstable or undefined near a zero mean; often inappropriate for variables that can be negative. |
| `local_median_` | Weighted median | Available only with `quantile=True`. |
| `local_iqr_` | Weighted Q3 minus Q1 | Available only with `quantile=True`. |
| `local_qi_` | `(2 median - Q3 - Q1) / IQR` | Signed quantile imbalance under the GWmodel definition; undefined when IQR is zero. |
| `local_cov_` | Unbiased weighted pairwise covariance | Stored in a dictionary keyed by variable-index pairs. |
| `local_corr_` | Weighted Pearson correlation | Sensitive to local outliers and nonlinearity. |
| `local_corr_spearman_` | Weighted correlation of globally ranked variables | Captures monotonic association; ranking is calculated over the full dataset before local weighting. |

The implementation follows `GWmodel::gwss` conventions. In particular, univariate local variance uses a normalized weighted moment, whereas pairwise covariance uses the unbiased denominator `1 - sum(w²)`.

## When to use GWSS

Use GWSS when:

- exploring a multivariate spatial dataset before modelling;
- deciding whether global summaries hide important local variation;
- checking where predictor distributions or correlations change;
- comparing mean-based and quantile-based local structure;
- selecting candidate variables or regions for further investigation.

Do not use local correlations as evidence that one variable causes another. Overlapping spatial windows create smooth-looking maps even under weak evidence.

| Goal | More appropriate method |
|---|---|
| Estimate a continuous response relationship | [`GWR`](gwr.md), [`MGWR`](mgwr.md), or another regression model |
| Classify observations | [`GWDA`](gwda.md) |
| Summarise local multivariate dimensions | [`GWPCA`](gwpca.md) |
| Formal hypothesis testing of local statistics | Requires a dedicated inferential or permutation design not exposed by this class. |

## Coordinate and distance assumptions

GWSS currently uses the shared Euclidean distance utility and does not expose `distance_metric`. Supply projected coordinates in meaningful planar units. A fixed bandwidth then has the same unit as those coordinates.

`summary_coords` may differ from observation coordinates. Each summary location receives weights over the original observations, allowing a regular grid or selected reporting points. It does not create new observations or interpolate raw variables.

## Installation

```bash
pip install pygwrx
```

## Self-contained example

```python
import numpy as np
import pandas as pd

from pygwrx import GWSS

rng = np.random.default_rng(77)
n = 100
coords = rng.uniform(0.0, 100.0, size=(n, 2))

X = pd.DataFrame(
    {
        "income": 40.0 + 0.20 * coords[:, 0] + rng.normal(0.0, 5.0, size=n),
        "access": 70.0 - 0.25 * coords[:, 1] + rng.normal(0.0, 6.0, size=n),
        "density": np.exp(1.5 + 0.008 * coords[:, 0] + rng.normal(0.0, 0.25, size=n)),
    }
)

# Evaluate on a small reporting grid rather than only at observations.
grid_axis = np.linspace(10.0, 90.0, 5)
summary_coords = np.array(
    [(x, y) for y in grid_axis for x in grid_axis],
    dtype=float,
)

model = GWSS(
    kernel="bisquare",
    bandwidth=None,       # automatically select one shared mean-CV bandwidth
    adaptive=True,
    quantile=True,
).fit(
    X,
    coords,
    summary_coords=summary_coords,
)

print("selected neighbours:", model.bandwidth_)
print(model.summary())
result = model.to_dataframe()
print(result.head())
```

For a median-oriented bandwidth, select it explicitly and refit:

```python
selector = GWSS(kernel="bisquare", adaptive=True, quantile=True)
median_bw = selector.select_bandwidth(X, coords, statistic="median")
median_model = GWSS(
    kernel="bisquare",
    bandwidth=median_bw,
    adaptive=True,
    quantile=True,
).fit(X, coords, summary_coords=summary_coords)
```

## Constructor

```python
GWSS(
    kernel="bisquare",
    bandwidth=None,
    adaptive=False,
    quantile=False,
    verbose=False,
)
```

## Constructor parameters

| Parameter | Default | Meaning | How to use and what can go wrong |
|---|---:|---|---|
| `kernel` | `"bisquare"` | Converts Euclidean distance to local weights. | Compact support makes local summaries easy to interpret but may create insufficient effective support. Gaussian/exponential kernels use all observations with decaying weight. |
| `bandwidth` | `None` | Positive fixed distance, adaptive integer neighbour count, or automatic selection when `None`. | `fit()` with `None` always selects one shared bandwidth using mean-based leave-one-out CV, even when `quantile=True`. Select a median bandwidth separately when that is the target statistic. |
| `adaptive` | `False` | Numeric bandwidth is a neighbour count when true. | Useful under uneven sampling density. Adaptive bandwidth must be an integer of at least 2 and no larger than sample size. |
| `quantile` | `False` | Adds local median, IQR, and quantile imbalance. | Increases computation, especially during median bandwidth selection. Use for skewed or outlier-prone variables. |
| `verbose` | `False` | Prints the number of fitted summary locations. |

No `distance_metric`, per-variable bandwidth, missing-value handling, or inferential p-value parameter is exposed.

## Bandwidth selection

```python
bandwidth = model.select_bandwidth(
    X,
    coords,
    statistic="mean",  # or "median"
)
```

The selector returns one shared bandwidth for all variables and all reported statistics.

The criterion compares each full local mean/median with its leave-one-out counterpart and sums squared changes over observations and variables. It follows the GWmodel mean/median CV convention; it is not a regression prediction error.

- Adaptive selection exhaustively checks integer neighbour counts from 2 to `n`.
- Fixed selection uses bounded scalar optimisation between `max_distance / 5000` and `max_distance`.
- `statistic="median"` is available only through an explicit selector call; `fit(bandwidth=None)` uses `"mean"`.

A bandwidth that stabilises local means may not be ideal for correlations, skewness, or quantiles. Treat the selected value as a common exploratory scale and conduct sensitivity analysis.

## Fitting

```python
model.fit(
    X,
    coords,
    summary_coords=None,
)
```

| Argument | Meaning |
|---|---|
| `X` | One-dimensional arrays are accepted and reshaped to one variable; DataFrame names are preserved. Missing and infinite values are rejected. |
| `coords` | Observation coordinates aligned row-by-row with `X`. |
| `summary_coords` | Optional locations where statistics are evaluated. When omitted, observation coordinates are used. Coordinate dimension must match `coords`. |

All variables share one weight matrix `weights_` with shape `(n_summary_locations, n_observations)`. Storing it requires roughly `8 × m × n` bytes before overhead. A very dense reporting grid can therefore be memory intensive.

## Outputs and export

`to_dataframe()`—not `to_frame()`—returns GWmodel-compatible columns:

- `x`, `y` summary coordinates;
- `<variable>_LM`, `_LSD`, `_LVar`, `_LSKe`, `_LCV`;
- optional `<variable>_Median`, `_IQR`, `_QI`;
- `Cov_left.right`, `Corr_left.right`, and `Spearman_rho_left.right`.

Main fitted state:

| Attribute | Shape/type |
|---|---|
| `coords_data_` | Observation coordinates. |
| `coords_summary_` | Evaluation coordinates. |
| `weights_` | `(m, n)` normalized weight matrix. |
| `var_names_` | Variable names used by export. |
| `local_mean_`, `local_var_`, `local_std_`, `local_skewness_`, `local_cv_` | `(m, p)` arrays. |
| `local_median_`, `local_iqr_`, `local_qi_` | `(m, p)` arrays or `None`. |
| `local_cov_`, `local_corr_`, `local_corr_spearman_` | Dictionaries keyed by `(j, k)`, each value length `m`. |
| `bandwidth_` | Shared fixed distance or adaptive neighbour count. |

## Interpretation workflow

1. Inspect global distributions, units, missingness, and transformations first.
2. Fit GWSS at a scientifically interpretable bandwidth.
3. Map local means/medians and dispersion together.
4. Compare Pearson and Spearman association maps.
5. Inspect effective neighbourhood support and boundary regions.
6. Repeat at larger and smaller plausible bandwidths.
7. Use findings to motivate—not automatically determine—subsequent model specification.

Spatial patterns in local statistics are partly induced by overlapping kernels. Adjacent summary locations are not independent estimates.

## Common mistakes

| Mistake | Correction |
|---|---|
| Calling `to_frame()` | Use `to_dataframe()`. |
| Treating `summary_coords` as new predicted data | They are locations for weighted summaries of the original observations. |
| Using longitude/latitude directly | Project coordinates; distance is Euclidean. |
| Interpreting `local_cv_` near zero means | CV becomes unstable or undefined; inspect mean and SD separately. |
| Assuming `quantile=True` selects a median bandwidth | `fit()` still uses mean CV when bandwidth is `None`. Select median bandwidth explicitly. |
| Comparing correlations without local variance/support | Correlation can be undefined or unstable when local variance is tiny. |
| Treating pairwise correlation maps as hypothesis tests | No p-values or multiple-testing procedure is provided. |
| Using a separate “optimal” bandwidth for every plotted statistic without disclosure | The current class stores one shared bandwidth; report sensitivity clearly. |

## What to report

Report:

- variables, units, transformations, and sample size;
- coordinate reference system;
- kernel and fixed/adaptive bandwidth semantics;
- manual or mean/median CV selection and selected bandwidth;
- observation versus summary-location design;
- whether quantiles were calculated;
- local statistic definitions, especially variance/covariance denominators;
- bandwidth sensitivity and boundary effects;
- treatment of undefined CV, skewness, or correlation values;
- exploratory rather than inferential status;
- pyGWRx version.

## References

- Brunsdon, C., Fotheringham, A. S., & Charlton, M. E. (2002). Geographically weighted summary statistics—a framework for localised exploratory data analysis. *Computers, Environment and Urban Systems*, 26(6), 501–524. [`10.1016/S0198-9715(01)00009-6`](https://doi.org/10.1016/S0198-9715(01)00009-6)
- Gollini, I., Lu, B., Charlton, M., Brunsdon, C., & Harris, P. (2015). GWmodel: An R Package for Exploring Spatial Heterogeneity Using Geographically Weighted Models. *Journal of Statistical Software*, 63(17). [`10.18637/jss.v063.i17`](https://doi.org/10.18637/jss.v063.i17)

## Related documentation

- [Generated GWSS API](../api/models/gwss.md)
- [GWPCA](gwpca.md)
- [GWDA](gwda.md)
- [Model selection](../getting-started/choosing-a-model.md)