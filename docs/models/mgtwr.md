# Multiscale Geographically and Temporally Weighted Regression (`MGTWR`)

<div class="model-hero" markdown>

**Family:** Multiscale spatiotemporal regression  
**Install:** base installation  
**Required data:** X, y, coordinates, and numeric times  
**Primary operations:** fit, summary, and calibration-location result tables  
**New-location capability:** independent-target prediction is intentionally unavailable

</div>

[API reference](../api/models/mgtwr.md){ .md-button .md-button--primary }
[Runnable source](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/models/17_mgtwr.py){ .md-button }
[Choose a model](../getting-started/choosing-a-model.md){ .md-button }

## Why this model exists

MGTWR is appropriate when predictor effects vary over space and time and different coefficients operate at different scales. Unlike a single-bandwidth GTWR, each fitted term receives its own spatial bandwidth and temporal scale.

## Statistical formulation

For coefficient $k$, pyGWRx uses

$$
w_{ij,k}=K\!\left(
\frac{\sqrt{(d_{ij}^{S})^2+\tau_k(d_{ij}^{T})^2}}{h_k}
\right),
$$

where $h_k$ is the coefficient-specific bandwidth and $\tau_k\geq0$ controls the contribution of temporal distance. Setting $\tau_k=0$ reduces that coefficient to a spatial-only multiscale term.

The response is represented as an additive collection of local terms. Calibration begins with a common GTWR fit. Backfitting then forms the partial residual for one term, selects or applies $(h_k,\tau_k)$, updates its local coefficient surface, and cycles through all terms until the score of change meets `tol_multi` or `max_iter` is reached.

## Implementation status

`MGTWR` is implemented entirely inside pyGWRx. It reuses the package's distance utilities, kernels, weighted least-squares solver, diagnostics, and multiscale fitted-state conventions. There is no MGTWR-specific external runtime, development, or reference dependency.

When `calculate_inference=True`, pyGWRx propagates the additive smoother operators through the final backfitting history to obtain coefficient-specific effective parameter counts, influence values, covariance factors, standard errors, t statistics, and information criteria. `n_chunks` controls memory partitioning of that exact smoother calculation; it does not change the fitted coefficients.

## Constructor

```python
MGTWR(
    bandwidths=None,
    taus=None,
    *,
    kernel="bisquare",
    adaptive=True,
    fit_intercept=True,
    bandwidth_method="aicc",
    bandwidth_range=None,
    tau_range=(0.0, 4.0),
    init_bandwidth=None,
    init_tau=None,
    tol=1e-6,
    tol_multi=1e-5,
    max_iter=200,
    rss_score=False,
    calculate_inference=True,
    n_chunks=1,
    verbose=False,
)
```

`bandwidths` and `taus` must be supplied together. A scalar is expanded to every fitted parameter; a sequence must contain one value per fitted parameter, including the intercept when `fit_intercept=True`. With adaptive kernels, bandwidths are integer neighbour counts. Automatic selection performs a deterministic coarse-to-fine candidate search inside `bandwidth_range` and `tau_range` using the selected AIC, AICc, BIC, or CV criterion. It evaluates the configured boundaries, but it is not advertised as an exhaustive proof of the global optimum.

## Numerical validation

The frozen fixed-scale regression fixture in `tests/reference_data/mgtwr_fixed_gaussian_reference.json` was produced once by an independent implementation outside the repository. The project test compares local coefficients, fitted values, residuals, coefficient-specific ENP, standard errors, t statistics, information criteria, and backfitting iterations without importing or installing any external MGTWR package. Across the broader one-time comparison matrix, the largest observed coefficient difference was `4.17e-8` and the largest fitted-value difference was `7.29e-8`.

## Scale and performance cautions

`tau` is unit-dependent because spatial and temporal distances are combined numerically. Coordinate units, time units, and `tau_range` must therefore be reported and chosen together. Automatic selection uses bounded candidate grids with local refinement, so boundary selections should be inspected rather than interpreted as proof of an interior optimum. Exact smoother inference is substantially more expensive than coefficient fitting; `n_chunks` reduces peak memory but does not parallelize calibration.

## Complete runnable example

```python
from pygwrx import MGTWR
from _common import print_model_result, temporal_regression

X, y, coords, times = temporal_regression(n=20, p=2)
model = MGTWR(
    bandwidths=[12, 12, 12],
    taus=[1.0, 1.0, 1.0],
    adaptive=True,
    calculate_inference=False,
).fit(X, y, coords, times)

print_model_result(model)
print("spatial_bandwidths=", model.bandwidths_)
print("temporal_scales=", model.taus_)
```

Run the maintained script from the project root:

```bash
python examples/models/17_mgtwr.py
```

## Reading the fitted result

Important attributes include:

- `bandwidths_`, `taus_`, and `temporal_bandwidths_`;
- `bandwidth_history_`, `tau_history_`, and `convergence_history_`;
- `params_`, `intercept_`, `coef_`, `fitted_values_`, and `residuals_`;
- `effective_params_by_variable_`, standard errors, t values, AIC, AICc, and BIC when inference is enabled.

Use `summary()` for a text table and `to_frame()` for calibration-location coefficients and diagnostics. The frame includes the original time value for every row.

## Validation and interpretation

Check convergence, boundary scale selections, sensitivity to time units, and local collinearity before interpreting coefficient surfaces. A large `tau` does not have an absolute meaning without the time encoding and spatial coordinate units. Compare candidate encodings and report them explicitly.

`predict()` raises `NotImplementedError` for independently supplied target locations. This is a deliberate capability boundary, not a hidden fallback. Use `fitted_values_` for calibration locations.

## What to report

- coordinate reference system and time encoding;
- kernel and fixed/adaptive interpretation;
- per-parameter bandwidths and taus;
- search ranges, initial scales, stopping rule, iterations, and convergence;
- whether exact inference was enabled and the selected `n_chunks`;
- validation design and the calibration-only prediction boundary.

## References

- Wu, C., Ren, F., Hu, W., & Du, Q. (2019). Multiscale geographically and temporally weighted regression: exploring the spatiotemporal determinants of housing prices. *International Journal of Geographical Information Science*, 33(3), 489–511. https://doi.org/10.1080/13658816.2018.1545158
- Fotheringham, A. S., Yang, W., & Kang, W. (2017). Multiscale geographically weighted regression (MGWR). *Annals of the American Association of Geographers*, 107(6), 1247–1265. https://doi.org/10.1080/24694452.2017.1352480

## Related documentation

- [Detailed API for `MGTWR`](../api/models/mgtwr.md)
- [Kernels and bandwidths](../guides/kernels-and-bandwidths.md)
- [Prediction and result objects](../guides/prediction-and-results.md)
- [Chinese model guide](../zh/models/mgtwr.md)
