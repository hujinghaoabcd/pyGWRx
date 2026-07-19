# Robust and GLM plots

This page documents **3** public symbols. Each entry includes its purpose, import path, full API docstring, and the maintained example that exercises it.

[Conceptual guide](../../guides/visualization.md){ .md-button }

## `plot_rgwr_weights`

Map final robust weights and outline completely rejected observations.

| Property | Value |
|---|---|
| Type | `function` |
| Import | `from pygwrx.plotting import plot_rgwr_weights` |
| Signature | `plot_rgwr_weights(model, *, geometry=None, theme: 'str' = 'default', ax: 'Optional[plt.Axes]' = None, figsize: 'Optional[Tuple[float, float]]' = None, cmap: 'str' = 'viridis', title: 'Optional[str]' = None)` |
| Maintained example | [`examples/plotting/03_robust_regularized_bootstrap.py`](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/plotting/03_robust_regularized_bootstrap.py) |

::: pygwrx.plotting.plot_rgwr_weights


## `plot_rgwr_convergence`

Plot iteration MSE and the number of downweighted observations.

| Property | Value |
|---|---|
| Type | `function` |
| Import | `from pygwrx.plotting import plot_rgwr_convergence` |
| Signature | `plot_rgwr_convergence(model, *, theme: 'str' = 'default', ax: 'Optional[plt.Axes]' = None, figsize: 'Optional[Tuple[float, float]]' = None, title: 'str' = 'Robust GWR convergence')` |
| Maintained example | [`examples/plotting/03_robust_regularized_bootstrap.py`](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/plotting/03_robust_regularized_bootstrap.py) |

::: pygwrx.plotting.plot_rgwr_convergence


## `plot_gwglm_residuals`

Map Pearson, deviance, or raw residuals from a fitted GWGLM.

| Property | Value |
|---|---|
| Type | `function` |
| Import | `from pygwrx.plotting import plot_gwglm_residuals` |
| Signature | `plot_gwglm_residuals(model, *, residual: 'str' = 'deviance', geometry=None, theme: 'str' = 'default', ax: 'Optional[plt.Axes]' = None, figsize: 'Optional[Tuple[float, float]]' = None, title: 'Optional[str]' = None)` |
| Maintained example | [`examples/plotting/03_robust_regularized_bootstrap.py`](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/plotting/03_robust_regularized_bootstrap.py) |

::: pygwrx.plotting.plot_gwglm_residuals


## Runnable examples used on this page

??? example "`examples/plotting/03_robust_regularized_bootstrap.py`"

    ```python
    # SPDX-FileCopyrightText: 2026 Jinghao Hu
    # SPDX-License-Identifier: MIT
    
    """All robust, GLM, Lasso, mixed, bootstrap, and scalable plots."""
    
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
    from _models import regularized_models
    
    from pygwrx.plotting import (
        plot_bootstrap_bandwidths,
        plot_bootstrap_pvalues,
        plot_gwglm_residuals,
        plot_gwlasso_active_map,
        plot_gwlasso_alpha,
        plot_gwlasso_selection_frequency,
        plot_mixed_gwr_coefficients,
        plot_rgwr_convergence,
        plot_rgwr_weights,
        plot_scalable_gwr_kernel,
    )
    
    X, y, coords, rgwr, gwglm, gwlasso, mixed, bootstrap, scalable = regularized_models()
    plots = {
        "rgwr_weights.png": plot_rgwr_weights(rgwr),
        "rgwr_convergence.png": plot_rgwr_convergence(rgwr),
        "gwglm_residuals.png": plot_gwglm_residuals(gwglm),
        "gwlasso_frequency.png": plot_gwlasso_selection_frequency(gwlasso),
        "gwlasso_active.png": plot_gwlasso_active_map(gwlasso, "x1"),
        "gwlasso_alpha.png": plot_gwlasso_alpha(gwlasso),
        "mixed_coefficients.png": plot_mixed_gwr_coefficients(mixed),
        "bootstrap_pvalues.png": plot_bootstrap_pvalues(bootstrap, "x1"),
        "bootstrap_bandwidths.png": plot_bootstrap_bandwidths(bootstrap),
        "scalable_kernel.png": plot_scalable_gwr_kernel(scalable),
    }
    for name, result in plots.items():
        print(save_plot(result, name))
    ```
