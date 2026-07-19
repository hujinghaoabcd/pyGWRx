# Coefficient and diagnostic surfaces

This page documents **5** public symbols. Each entry includes its purpose, import path, full API docstring, and the maintained example that exercises it.

[Conceptual guide](../../guides/visualization.md){ .md-button }

## `plot_coefficient_map`

Plot one fitted local coefficient surface.

| Property | Value |
|---|---|
| Type | `function` |
| Import | `from pygwrx.plotting import plot_coefficient_map` |
| Signature | `plot_coefficient_map(model: 'Any', feature: 'Any', *, geometry: 'Any' = None, significance: 'Optional[str]' = None, alpha: 'float' = 0.05, theme: 'str' = 'default', ax: 'Optional[plt.Axes]' = None, figsize: 'Optional[Tuple[float, float]]' = None, cmap: 'Optional[str]' = None, vmin: 'Optional[float]' = None, vmax: 'Optional[float]' = None, marker_size: 'float' = 45.0, title: 'Optional[str]' = None) -> 'Tuple[plt.Figure, plt.Axes]'` |
| Maintained example | [`examples/plotting/01_surfaces_and_arrays.py`](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/plotting/01_surfaces_and_arrays.py) |

::: pygwrx.plotting.plot_coefficient_map


## `plot_significance_map`

Dispatch to model-aware or historical array-based significance mapping.

| Property | Value |
|---|---|
| Type | `function` |
| Import | `from pygwrx.plotting import plot_significance_map` |
| Signature | `plot_significance_map(first, *args, **kwargs)` |
| Maintained example | [`examples/plotting/01_surfaces_and_arrays.py`](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/plotting/01_surfaces_and_arrays.py) |

::: pygwrx.plotting.plot_significance_map


## `plot_model_significance_map`

Map negative-significant, non-significant, and positive-significant areas.

| Property | Value |
|---|---|
| Type | `function` |
| Import | `from pygwrx.plotting import plot_model_significance_map` |
| Signature | `plot_model_significance_map(model, feature, *, geometry=None, correction: 'str' = 'adjusted', alpha: 'float' = 0.05, theme: 'str' = 'default', ax: 'Optional[plt.Axes]' = None, figsize: 'Optional[Tuple[float, float]]' = None, marker_size: 'float' = 45.0, title: 'Optional[str]' = None)` |
| Maintained example | [`examples/plotting/01_surfaces_and_arrays.py`](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/plotting/01_surfaces_and_arrays.py) |

::: pygwrx.plotting.plot_model_significance_map


## `plot_local_diagnostic_map`

Plot a fitted local diagnostic such as Local R² or Cook's distance.

| Property | Value |
|---|---|
| Type | `function` |
| Import | `from pygwrx.plotting import plot_local_diagnostic_map` |
| Signature | `plot_local_diagnostic_map(model, metric: 'str', *, geometry=None, theme: 'str' = 'default', ax: 'Optional[plt.Axes]' = None, figsize: 'Optional[Tuple[float, float]]' = None, cmap: 'Optional[str]' = None, vmin: 'Optional[float]' = None, vmax: 'Optional[float]' = None, marker_size: 'float' = 45.0, title: 'Optional[str]' = None)` |
| Maintained example | [`examples/plotting/01_surfaces_and_arrays.py`](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/plotting/01_surfaces_and_arrays.py) |

::: pygwrx.plotting.plot_local_diagnostic_map


## `plot_local_collinearity`

Plot local condition numbers or LCR-GWR ridge compensation.

| Property | Value |
|---|---|
| Type | `function` |
| Import | `from pygwrx.plotting import plot_local_collinearity` |
| Signature | `plot_local_collinearity(model, metric: 'str' = 'condition_number', *, geometry=None, theme: 'str' = 'default', ax: 'Optional[plt.Axes]' = None, figsize: 'Optional[Tuple[float, float]]' = None, cmap: 'str' = 'magma', marker_size: 'float' = 45.0, show_threshold: 'bool' = True, title: 'Optional[str]' = None)` |
| Maintained example | [`examples/plotting/02_diagnostics_and_comparison.py`](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/plotting/02_diagnostics_and_comparison.py) |

::: pygwrx.plotting.plot_local_collinearity


## Runnable examples used on this page

??? example "`examples/plotting/01_surfaces_and_arrays.py`"

    ```python
    # SPDX-FileCopyrightText: 2026 Jinghao Hu
    # SPDX-License-Identifier: MIT
    
    """Model-aware coefficient maps plus all historical array-based maps."""
    
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
    import geopandas as gpd
    import numpy as np
    from _common import save_plot
    from _models import surface_models
    from shapely.geometry import Point
    
    from pygwrx.plotting import (
        create_choropleth,
        plot_array_significance_map,
        plot_bandwidth,
        plot_coefficient_map,
        plot_coefficient_surface,
        plot_local_coefficients,
        plot_local_diagnostic_map,
        plot_local_r2,
        plot_model_significance_map,
        plot_multiple_coefficients,
        plot_significance_map,
    )
    
    X, y, coords, gwr, _, _ = surface_models()
    coords_array = coords.to_numpy()
    p_values = np.full_like(gwr.coef_, 0.02)
    plots = {
        "coefficient_map.png": plot_coefficient_map(gwr, "x1", theme="paper"),
        "model_significance.png": plot_model_significance_map(gwr, "x1", correction="raw"),
        "dispatch_model_significance.png": plot_significance_map(gwr, "x1"),
        "local_diagnostic.png": plot_local_diagnostic_map(gwr, "local_r2"),
        "array_significance.png": plot_array_significance_map(
            coords_array, p_values, feature_idx=0, coefficients=gwr.coef_
        ),
        "dispatch_array_significance.png": plot_significance_map(
            coords_array, p_values, feature_idx=0, coefficients=gwr.coef_
        ),
        "local_coefficients.png": plot_local_coefficients(coords_array, gwr.coef_, 0, "x1"),
        "coefficient_surface.png": plot_coefficient_surface(
            coords_array, gwr.coef_, 0, interpolation="nearest"
        ),
        "array_local_r2.png": plot_local_r2(coords_array, gwr.local_r2_),
        "bandwidth_map.png": plot_bandwidth(
            coords_array, 2.0, sample_locations=coords_array[:3]
        ),
        "multiple_coefficients.png": plot_multiple_coefficients(
            coords_array, gwr.coef_, feature_names=["x1", "x2"], shared_scale=True
        ),
    }
    for name, result in plots.items():
        print(save_plot(result, name))
    gdf = gpd.GeoDataFrame(
        {"value": gwr.coef_[:, 0]},
        geometry=[Point(x, y) for x, y in coords_array],
        crs="EPSG:3857",
    )
    print(save_plot(create_choropleth(gdf, "value"), "choropleth.png"))
    ```

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
