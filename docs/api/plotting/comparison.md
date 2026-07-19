# Model comparison

This page documents **2** public symbols. Each entry includes its purpose, import path, full API docstring, and the maintained example that exercises it.

[Conceptual guide](../../guides/visualization.md){ .md-button }

## `compare_coefficient_surfaces`

Compare the same local coefficient across two or more fitted models.

| Property | Value |
|---|---|
| Type | `function` |
| Import | `from pygwrx.plotting import compare_coefficient_surfaces` |
| Signature | `compare_coefficient_surfaces(models: 'Sequence[object]', feature, *, geometry=None, labels: 'Optional[Sequence[str]]' = None, significance: 'Optional[str]' = None, alpha: 'float' = 0.05, shared_scale: 'bool' = True, theme: 'str' = 'default', figsize: 'Optional[Tuple[float, float]]' = None)` |
| Maintained example | [`examples/plotting/02_diagnostics_and_comparison.py`](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/plotting/02_diagnostics_and_comparison.py) |

::: pygwrx.plotting.compare_coefficient_surfaces


## `compare_model_diagnostics`

Compare normalized global diagnostics across fitted models.

| Property | Value |
|---|---|
| Type | `function` |
| Import | `from pygwrx.plotting import compare_model_diagnostics` |
| Signature | `compare_model_diagnostics(models: 'Sequence[object]', *, metrics: 'Sequence[str]' = ('r2', 'rmse', 'aicc', 'enp'), labels: 'Optional[Sequence[str]]' = None, theme: 'str' = 'default', figsize: 'Optional[Tuple[float, float]]' = None)` |
| Maintained example | [`examples/plotting/02_diagnostics_and_comparison.py`](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/plotting/02_diagnostics_and_comparison.py) |

::: pygwrx.plotting.compare_model_diagnostics


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
