# Array-based maps

This page documents **7** public symbols. Each entry includes its purpose, import path, full API docstring, and the maintained example that exercises it.

[Conceptual guide](../../guides/visualization.md){ .md-button }

## `plot_array_significance_map`

Plot significant and non-significant locations from local p-values.

| Property | Value |
|---|---|
| Type | `function` |
| Import | `from pygwrx.plotting import plot_array_significance_map` |
| Signature | `plot_array_significance_map(coords, p_values, alpha: 'float' = 0.05, feature_idx: 'int' = 0, figsize: 'Optional[Tuple[float, float]]' = None, *, coefficients=None, theme: 'str' = 'default', ax: 'Optional[plt.Axes]' = None, **kwargs)` |
| Maintained example | [`examples/plotting/01_surfaces_and_arrays.py`](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/plotting/01_surfaces_and_arrays.py) |

::: pygwrx.plotting.plot_array_significance_map


## `plot_local_coefficients`

Plot one column of a local coefficient array.

| Property | Value |
|---|---|
| Type | `function` |
| Import | `from pygwrx.plotting import plot_local_coefficients` |
| Signature | `plot_local_coefficients(coords, coefficients, feature_idx: 'int' = 0, feature_name: 'Optional[str]' = None, cmap: 'Optional[str]' = None, figsize: 'Optional[Tuple[float, float]]' = None, title: 'Optional[str]' = None, basemap: 'Optional[gpd.GeoDataFrame]' = None, *, theme: 'str' = 'default', ax: 'Optional[plt.Axes]' = None, **kwargs)` |
| Maintained example | [`examples/plotting/01_surfaces_and_arrays.py`](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/plotting/01_surfaces_and_arrays.py) |

::: pygwrx.plotting.plot_local_coefficients


## `plot_coefficient_surface`

Interpolate a local coefficient array to a regular plotting grid.

| Property | Value |
|---|---|
| Type | `function` |
| Import | `from pygwrx.plotting import plot_coefficient_surface` |
| Signature | `plot_coefficient_surface(coords, coefficients, feature_idx: 'int' = 0, method: 'str' = 'contourf', n_levels: 'int' = 20, figsize: 'Optional[Tuple[float, float]]' = None, *, theme: 'str' = 'default', ax: 'Optional[plt.Axes]' = None, grid_size: 'int' = 100, interpolation: 'str' = 'linear', cmap: 'Optional[str]' = None, **kwargs)` |
| Maintained example | [`examples/plotting/01_surfaces_and_arrays.py`](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/plotting/01_surfaces_and_arrays.py) |

::: pygwrx.plotting.plot_coefficient_surface


## `plot_local_r2`

Plot spatial local R² values.

| Property | Value |
|---|---|
| Type | `function` |
| Import | `from pygwrx.plotting import plot_local_r2` |
| Signature | `plot_local_r2(coords, local_r2, cmap: 'str' = 'YlOrRd', figsize: 'Optional[Tuple[float, float]]' = None, *, theme: 'str' = 'default', ax: 'Optional[plt.Axes]' = None, **kwargs)` |
| Maintained example | [`examples/plotting/01_surfaces_and_arrays.py`](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/plotting/01_surfaces_and_arrays.py) |

::: pygwrx.plotting.plot_local_r2


## `plot_bandwidth`

Visualize fixed-distance bandwidth footprints at selected locations.

| Property | Value |
|---|---|
| Type | `function` |
| Import | `from pygwrx.plotting import plot_bandwidth` |
| Signature | `plot_bandwidth(coords, bandwidth: 'Union[float, np.ndarray]', kernel: 'str' = 'gaussian', sample_locations=None, figsize: 'Optional[Tuple[float, float]]' = None, *, theme: 'str' = 'default', ax: 'Optional[plt.Axes]' = None, **kwargs)` |
| Maintained example | [`examples/plotting/01_surfaces_and_arrays.py`](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/plotting/01_surfaces_and_arrays.py) |

::: pygwrx.plotting.plot_bandwidth


## `create_choropleth`

Create a validated GeoDataFrame choropleth.

| Property | Value |
|---|---|
| Type | `function` |
| Import | `from pygwrx.plotting import create_choropleth` |
| Signature | `create_choropleth(gdf: "'gpd.GeoDataFrame'", column: 'str', cmap: 'str' = 'viridis', legend: 'bool' = True, figsize: 'Optional[Tuple[float, float]]' = None, *, theme: 'str' = 'default', ax: 'Optional[plt.Axes]' = None, title: 'Optional[str]' = None, **kwargs)` |
| Maintained example | [`examples/plotting/01_surfaces_and_arrays.py`](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/plotting/01_surfaces_and_arrays.py) |

::: pygwrx.plotting.create_choropleth


## `plot_multiple_coefficients`

Create a panel containing every coefficient column.

| Property | Value |
|---|---|
| Type | `function` |
| Import | `from pygwrx.plotting import plot_multiple_coefficients` |
| Signature | `plot_multiple_coefficients(coords, coefficients, feature_names: 'Optional[List[str]]' = None, ncols: 'int' = 2, figsize: 'Optional[Tuple[float, float]]' = None, *, theme: 'str' = 'default', shared_scale: 'bool' = False, **kwargs)` |
| Maintained example | [`examples/plotting/01_surfaces_and_arrays.py`](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/plotting/01_surfaces_and_arrays.py) |

::: pygwrx.plotting.plot_multiple_coefficients


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
