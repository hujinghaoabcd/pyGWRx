# Bootstrap plots

This page documents **2** public symbols. Each entry includes its purpose, import path, full API docstring, and the maintained example that exercises it.

[Conceptual guide](../../guides/visualization.md){ .md-button }

## `plot_bootstrap_pvalues`

Map localized bootstrap p values or show a global modified-test p value.

| Property | Value |
|---|---|
| Type | `function` |
| Import | `from pygwrx.plotting import plot_bootstrap_pvalues` |
| Signature | `plot_bootstrap_pvalues(model, feature, *, test: 'str' = 'localized', geometry=None, alpha: 'float' = 0.05, theme: 'str' = 'default', ax: 'Optional[plt.Axes]' = None, figsize: 'Optional[Tuple[float, float]]' = None, title: 'Optional[str]' = None)` |
| Maintained example | [`examples/plotting/03_robust_regularized_bootstrap.py`](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/plotting/03_robust_regularized_bootstrap.py) |

::: pygwrx.plotting.plot_bootstrap_pvalues


## `plot_bootstrap_bandwidths`

Plot bandwidth variability across bootstrap replications.

| Property | Value |
|---|---|
| Type | `function` |
| Import | `from pygwrx.plotting import plot_bootstrap_bandwidths` |
| Signature | `plot_bootstrap_bandwidths(model, *, theme: 'str' = 'default', ax: 'Optional[plt.Axes]' = None, figsize: 'Optional[Tuple[float, float]]' = None, title: 'str' = 'Bootstrap bandwidth distribution')` |
| Maintained example | [`examples/plotting/03_robust_regularized_bootstrap.py`](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/plotting/03_robust_regularized_bootstrap.py) |

::: pygwrx.plotting.plot_bootstrap_bandwidths


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
