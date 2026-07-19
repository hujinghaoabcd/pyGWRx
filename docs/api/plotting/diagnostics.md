# Regression diagnostics

This page documents **9** public symbols. Each entry includes its purpose, import path, full API docstring, and the maintained example that exercises it.

[Conceptual guide](../../guides/visualization.md){ .md-button }

## `plot_residuals`

Plot residuals against fitted values with a binned mean trend.

| Property | Value |
|---|---|
| Type | `function` |
| Import | `from pygwrx.plotting import plot_residuals` |
| Signature | `plot_residuals(fitted_values, residuals, figsize: 'Optional[Tuple[float, float]]' = None, *, theme: 'str' = 'default', ax: 'Optional[plt.Axes]' = None, title: 'str' = 'Residuals vs fitted values', **kwargs)` |
| Maintained example | [`examples/plotting/02_diagnostics_and_comparison.py`](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/plotting/02_diagnostics_and_comparison.py) |

::: pygwrx.plotting.plot_residuals


## `plot_residual_histogram`

Plot a residual histogram and optional fitted normal density.

| Property | Value |
|---|---|
| Type | `function` |
| Import | `from pygwrx.plotting import plot_residual_histogram` |
| Signature | `plot_residual_histogram(residuals, bins: 'int' = 30, density: 'bool' = True, figsize: 'Optional[Tuple[float, float]]' = None, *, theme: 'str' = 'default', ax: 'Optional[plt.Axes]' = None, title: 'str' = 'Residual distribution', **kwargs)` |
| Maintained example | [`examples/plotting/02_diagnostics_and_comparison.py`](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/plotting/02_diagnostics_and_comparison.py) |

::: pygwrx.plotting.plot_residual_histogram


## `plot_qq`

Create a normal Q–Q plot for residuals.

| Property | Value |
|---|---|
| Type | `function` |
| Import | `from pygwrx.plotting import plot_qq` |
| Signature | `plot_qq(residuals, figsize: 'Optional[Tuple[float, float]]' = None, *, theme: 'str' = 'default', ax: 'Optional[plt.Axes]' = None, title: 'str' = 'Normal Q–Q plot', **kwargs)` |
| Maintained example | [`examples/plotting/02_diagnostics_and_comparison.py`](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/plotting/02_diagnostics_and_comparison.py) |

::: pygwrx.plotting.plot_qq


## `plot_spatial_residuals`

Map residuals using calibration point coordinates.

| Property | Value |
|---|---|
| Type | `function` |
| Import | `from pygwrx.plotting import plot_spatial_residuals` |
| Signature | `plot_spatial_residuals(coords, residuals, cmap: 'str' = 'RdBu_r', figsize: 'Optional[Tuple[float, float]]' = None, *, theme: 'str' = 'default', ax: 'Optional[plt.Axes]' = None, title: 'str' = 'Spatial residual pattern', **kwargs)` |
| Maintained example | [`examples/plotting/02_diagnostics_and_comparison.py`](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/plotting/02_diagnostics_and_comparison.py) |

::: pygwrx.plotting.plot_spatial_residuals


## `plot_observed_vs_predicted`

Plot observed and predicted values with a one-to-one reference.

| Property | Value |
|---|---|
| Type | `function` |
| Import | `from pygwrx.plotting import plot_observed_vs_predicted` |
| Signature | `plot_observed_vs_predicted(y_true, y_pred, figsize: 'Optional[Tuple[float, float]]' = None, *, theme: 'str' = 'default', ax: 'Optional[plt.Axes]' = None, title: 'str' = 'Observed vs predicted', **kwargs)` |
| Maintained example | [`examples/plotting/02_diagnostics_and_comparison.py`](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/plotting/02_diagnostics_and_comparison.py) |

::: pygwrx.plotting.plot_observed_vs_predicted


## `plot_bandwidth_selection`

Compatibility wrapper for :func:`pygwrx.plotting.bandwidth.plot_bandwidth_selection`.

| Property | Value |
|---|---|
| Type | `function` |
| Import | `from pygwrx.plotting import plot_bandwidth_selection` |
| Signature | `plot_bandwidth_selection(bandwidths, scores, selected_bandwidth, criterion: 'str' = 'CV', figsize: 'Optional[Tuple[float, float]]' = None, *, theme: 'str' = 'default', ax: 'Optional[plt.Axes]' = None, **kwargs)` |
| Maintained example | [`examples/plotting/02_diagnostics_and_comparison.py`](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/plotting/02_diagnostics_and_comparison.py) |

::: pygwrx.plotting.plot_bandwidth_selection


## `plot_coefficient_variability`

Compare distributions of local coefficients across variables.

| Property | Value |
|---|---|
| Type | `function` |
| Import | `from pygwrx.plotting import plot_coefficient_variability` |
| Signature | `plot_coefficient_variability(coefficients, feature_names: 'Optional[Sequence[str]]' = None, figsize: 'Optional[Tuple[float, float]]' = None, *, theme: 'str' = 'default', ax: 'Optional[plt.Axes]' = None, kind: 'str' = 'box', global_coefficients: 'Optional[Sequence[float]]' = None, **kwargs)` |
| Maintained example | [`examples/plotting/02_diagnostics_and_comparison.py`](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/plotting/02_diagnostics_and_comparison.py) |

::: pygwrx.plotting.plot_coefficient_variability


## `plot_diagnostic_panel`

Create a complete calibration-diagnostic panel without displaying it.

| Property | Value |
|---|---|
| Type | `function` |
| Import | `from pygwrx.plotting import plot_diagnostic_panel` |
| Signature | `plot_diagnostic_panel(y_true, y_pred=None, residuals=None, coords=None, figsize: 'Tuple[float, float]' = (14, 9), *, theme: 'str' = 'default')` |
| Maintained example | [`examples/plotting/02_diagnostics_and_comparison.py`](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/plotting/02_diagnostics_and_comparison.py) |

::: pygwrx.plotting.plot_diagnostic_panel


## `plot_local_diagnostics`

Plot several local diagnostic arrays on a common coordinate set.

| Property | Value |
|---|---|
| Type | `function` |
| Import | `from pygwrx.plotting import plot_local_diagnostics` |
| Signature | `plot_local_diagnostics(coords, diagnostics: 'Mapping[str, Sequence[float]]', figsize: 'Optional[Tuple[float, float]]' = None, *, theme: 'str' = 'default', ncols: 'int' = 3)` |
| Maintained example | [`examples/plotting/02_diagnostics_and_comparison.py`](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/plotting/02_diagnostics_and_comparison.py) |

::: pygwrx.plotting.plot_local_diagnostics


## Runnable examples used on this page

??? example "`examples/plotting/02_diagnostics_and_comparison.py`"

    ```python
    # SPDX-FileCopyrightText: 2026 Jinghao Hu
    # SPDX-License-Identifier: MIT
    
    """All general residual, bandwidth, comparison, and collinearity plots."""
    
    # Allow this script to run directly from any working directory.
    import sys
    from pathlib import Path
    
    _PROJECT_ROOT = Path(__file__).resolve().parents[2]
    _EXAMPLES_ROOT = _PROJECT_ROOT / "examples"
    _SRC_ROOT = _PROJECT_ROOT / "src"
    for _path in (_SRC_ROOT, _EXAMPLES_ROOT):
        if str(_path) not in sys.path:
            sys.path.insert(0, str(_path))
    
    import matplotlib
    
    matplotlib.use("Agg", force=True)
    from _common import save_plot
    from _models import surface_models
    
    from pygwrx.plotting import (
        compare_coefficient_surfaces,
        compare_model_diagnostics,
        plot_bandwidth_selection,
        plot_coefficient_variability,
        plot_diagnostic_panel,
        plot_kernel_weights,
        plot_local_collinearity,
        plot_local_diagnostics,
        plot_mgwr_bandwidths,
        plot_observed_vs_predicted,
        plot_qq,
        plot_residual_histogram,
        plot_residuals,
        plot_spatial_residuals,
    )
    
    X, y, coords, gwr, mgwr, lcr = surface_models()
    plots = {
        "compare_surfaces.png": compare_coefficient_surfaces([gwr, mgwr], "x1"),
        "compare_diagnostics.png": compare_model_diagnostics([gwr, mgwr]),
        "kernel_weights.png": plot_kernel_weights(gwr, focus=3),
        "mgwr_bandwidths.png": plot_mgwr_bandwidths(mgwr),
        "residuals.png": plot_residuals(gwr.fitted_values_, gwr.residuals_),
        "residual_histogram.png": plot_residual_histogram(gwr.residuals_),
        "qq.png": plot_qq(gwr.residuals_),
        "spatial_residuals.png": plot_spatial_residuals(coords, gwr.residuals_),
        "observed_predicted.png": plot_observed_vs_predicted(y, gwr.fitted_values_),
        "bandwidth_selection.png": plot_bandwidth_selection(
            [10, 15, 20, 25], [14.0, 9.0, 7.5, 8.2], 20, criterion="AICc"
        ),
        "coefficient_variability.png": plot_coefficient_variability(
            gwr.coef_, feature_names=["x1", "x2"]
        ),
        "diagnostic_panel_arrays.png": plot_diagnostic_panel(
            y, gwr.fitted_values_, gwr.residuals_, coords
        ),
        "diagnostic_panel_model.png": plot_diagnostic_panel(gwr),
        "local_diagnostics.png": plot_local_diagnostics(
            coords, {"local_r2": gwr.local_r2_, "influence": gwr.influence_}
        ),
        "collinearity_gwr.png": plot_local_collinearity(gwr, "condition_number"),
        "collinearity_lcr.png": plot_local_collinearity(lcr, "local_lambda"),
    }
    for name, result in plots.items():
        print(save_plot(result, name))
    ```
