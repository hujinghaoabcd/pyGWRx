# GRGWR regimes

This page documents **4** public symbols. Each entry includes its purpose, import path, full API docstring, and the maintained example that exercises it.

[Conceptual guide](../../guides/visualization.md){ .md-button }

## `plot_grgwr_regimes`

Map final mechanism regimes and optional boundary graph edges.

| Property | Value |
|---|---|
| Type | `function` |
| Import | `from pygwrx.plotting import plot_grgwr_regimes` |
| Signature | `plot_grgwr_regimes(model, *, geometry=None, show_boundaries: 'bool' = True, theme: 'str' = 'default', ax: 'Optional[plt.Axes]' = None, figsize: 'Optional[Tuple[float, float]]' = None, title: 'str' = 'GR-GWR spatial regimes')` |
| Maintained example | [`examples/plotting/06_lggwr_and_grgwr.py`](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/plotting/06_lggwr_and_grgwr.py) |

::: pygwrx.plotting.plot_grgwr_regimes


## `plot_grgwr_convergence`

Plot the accepted penalized objective sequence.

| Property | Value |
|---|---|
| Type | `function` |
| Import | `from pygwrx.plotting import plot_grgwr_convergence` |
| Signature | `plot_grgwr_convergence(model, *, theme: 'str' = 'default', ax: 'Optional[plt.Axes]' = None, figsize: 'Optional[Tuple[float, float]]' = None, title: 'str' = 'GR-GWR ICM convergence')` |
| Maintained example | [`examples/plotting/06_lggwr_and_grgwr.py`](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/plotting/06_lggwr_and_grgwr.py) |

::: pygwrx.plotting.plot_grgwr_convergence


## `plot_grgwr_regime_sizes`

Plot sample counts and optional RMSE by final regime.

| Property | Value |
|---|---|
| Type | `function` |
| Import | `from pygwrx.plotting import plot_grgwr_regime_sizes` |
| Signature | `plot_grgwr_regime_sizes(model, *, theme: 'str' = 'default', ax: 'Optional[plt.Axes]' = None, figsize: 'Optional[Tuple[float, float]]' = None, title: 'str' = 'GR-GWR regime sizes')` |
| Maintained example | [`examples/plotting/06_lggwr_and_grgwr.py`](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/plotting/06_lggwr_and_grgwr.py) |

::: pygwrx.plotting.plot_grgwr_regime_sizes


## `plot_grgwr_coefficient_surface`

Map a GR-GWR coefficient while overlaying discovered regime boundaries.

| Property | Value |
|---|---|
| Type | `function` |
| Import | `from pygwrx.plotting import plot_grgwr_coefficient_surface` |
| Signature | `plot_grgwr_coefficient_surface(model, feature, *, show_boundaries: 'bool' = True, theme: 'str' = 'default', ax: 'Optional[plt.Axes]' = None, figsize: 'Optional[Tuple[float, float]]' = None, title: 'Optional[str]' = None)` |
| Maintained example | [`examples/plotting/06_lggwr_and_grgwr.py`](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/plotting/06_lggwr_and_grgwr.py) |

::: pygwrx.plotting.plot_grgwr_coefficient_surface


## Runnable examples used on this page

??? example "`examples/plotting/06_lggwr_and_grgwr.py`"

    ```python
    # SPDX-FileCopyrightText: 2026 Jinghao Hu
    # SPDX-License-Identifier: MIT
    
    """All visualization functions for the two original research models."""
    
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
    from _models import original_models
    
    from pygwrx.plotting import (
        plot_grgwr_coefficient_surface,
        plot_grgwr_convergence,
        plot_grgwr_regime_sizes,
        plot_grgwr_regimes,
        plot_lggwr_latent_geometry,
        plot_lggwr_metric_matrix,
        plot_lggwr_neighbourhood_comparison,
        plot_lggwr_training,
    )
    
    lggwr, grgwr = original_models()
    plots = {
        "lggwr_geometry.png": plot_lggwr_latent_geometry(lggwr),
        "lggwr_metric.png": plot_lggwr_metric_matrix(lggwr),
        "lggwr_training.png": plot_lggwr_training(lggwr),
        "lggwr_neighbours.png": plot_lggwr_neighbourhood_comparison(lggwr, 0),
        "grgwr_regimes.png": plot_grgwr_regimes(grgwr),
        "grgwr_convergence.png": plot_grgwr_convergence(grgwr),
        "grgwr_sizes.png": plot_grgwr_regime_sizes(grgwr),
        "grgwr_surface.png": plot_grgwr_coefficient_surface(grgwr, "x1"),
    }
    for name, result in plots.items():
        print(save_plot(result, name))
    ```
