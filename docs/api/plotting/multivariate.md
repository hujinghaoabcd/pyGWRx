# Multivariate and classification plots

This page documents **5** public symbols. Each entry includes its purpose, import path, full API docstring, and the maintained example that exercises it.

[Conceptual guide](../../guides/visualization.md){ .md-button }

## `plot_gwss_statistic`

Map a local GWSS univariate or pairwise summary statistic.

| Property | Value |
|---|---|
| Type | `function` |
| Import | `from pygwrx.plotting import plot_gwss_statistic` |
| Signature | `plot_gwss_statistic(model, statistic: 'str' = 'mean', feature=0, *, second_feature=None, geometry=None, theme: 'str' = 'default', ax: 'Optional[plt.Axes]' = None, figsize: 'Optional[Tuple[float, float]]' = None, title: 'Optional[str]' = None)` |
| Maintained example | [`examples/plotting/04_multivariate_and_classification.py`](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/plotting/04_multivariate_and_classification.py) |

::: pygwrx.plotting.plot_gwss_statistic


## `plot_gwpca_explained_variance`

Map local explained variance for one component or cumulatively.

| Property | Value |
|---|---|
| Type | `function` |
| Import | `from pygwrx.plotting import plot_gwpca_explained_variance` |
| Signature | `plot_gwpca_explained_variance(model, component: 'int' = 0, *, cumulative: 'bool' = False, geometry=None, theme: 'str' = 'default', ax: 'Optional[plt.Axes]' = None, figsize: 'Optional[Tuple[float, float]]' = None, title: 'Optional[str]' = None)` |
| Maintained example | [`examples/plotting/04_multivariate_and_classification.py`](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/plotting/04_multivariate_and_classification.py) |

::: pygwrx.plotting.plot_gwpca_explained_variance


## `plot_gwpca_loading`

Map a local principal-component loading surface.

| Property | Value |
|---|---|
| Type | `function` |
| Import | `from pygwrx.plotting import plot_gwpca_loading` |
| Signature | `plot_gwpca_loading(model, feature=0, component: 'int' = 0, *, geometry=None, theme: 'str' = 'default', ax: 'Optional[plt.Axes]' = None, figsize: 'Optional[Tuple[float, float]]' = None, title: 'Optional[str]' = None)` |
| Maintained example | [`examples/plotting/04_multivariate_and_classification.py`](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/plotting/04_multivariate_and_classification.py) |

::: pygwrx.plotting.plot_gwpca_loading


## `plot_gwda_classification`

Map predicted classes or maximum class probability for GWDA.

| Property | Value |
|---|---|
| Type | `function` |
| Import | `from pygwrx.plotting import plot_gwda_classification` |
| Signature | `plot_gwda_classification(model, *, geometry=None, confidence: 'bool' = False, theme: 'str' = 'default', ax: 'Optional[plt.Axes]' = None, figsize: 'Optional[Tuple[float, float]]' = None, title: 'Optional[str]' = None)` |
| Maintained example | [`examples/plotting/04_multivariate_and_classification.py`](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/plotting/04_multivariate_and_classification.py) |

::: pygwrx.plotting.plot_gwda_classification


## `plot_gwda_confusion_matrix`

Plot a calibration/validation confusion matrix when labels are available.

| Property | Value |
|---|---|
| Type | `function` |
| Import | `from pygwrx.plotting import plot_gwda_confusion_matrix` |
| Signature | `plot_gwda_confusion_matrix(model, *, normalize: 'bool' = False, theme: 'str' = 'default', ax: 'Optional[plt.Axes]' = None, figsize: 'Optional[Tuple[float, float]]' = None, title: 'str' = 'GWDA confusion matrix')` |
| Maintained example | [`examples/plotting/04_multivariate_and_classification.py`](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/plotting/04_multivariate_and_classification.py) |

::: pygwrx.plotting.plot_gwda_confusion_matrix


## Runnable examples used on this page

??? example "`examples/plotting/04_multivariate_and_classification.py`"

    ```python
    # SPDX-FileCopyrightText: 2026 Jinghao Hu
    # SPDX-License-Identifier: MIT
    
    """All GWSS, GWPCA, and GWDA visualization functions."""
    
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
    from _models import multivariate_models
    
    from pygwrx.plotting import (
        plot_gwda_classification,
        plot_gwda_confusion_matrix,
        plot_gwpca_explained_variance,
        plot_gwpca_loading,
        plot_gwss_statistic,
    )
    
    X, coords, gwss, gwpca, Xc, yc, cc, gwda = multivariate_models()
    plots = {
        "gwss_mean.png": plot_gwss_statistic(gwss, "mean", "x1"),
        "gwss_correlation.png": plot_gwss_statistic(
            gwss, "correlation", "x1", second_feature="x2"
        ),
        "gwpca_variance.png": plot_gwpca_explained_variance(gwpca, 0),
        "gwpca_cumulative.png": plot_gwpca_explained_variance(gwpca, 0, cumulative=True),
        "gwpca_loading.png": plot_gwpca_loading(gwpca, "x1", 0),
        "gwda_classification.png": plot_gwda_classification(gwda),
        "gwda_confidence.png": plot_gwda_classification(gwda, confidence=True),
        "gwda_confusion.png": plot_gwda_confusion_matrix(gwda, normalize=True),
    }
    for name, result in plots.items():
        print(save_plot(result, name))
    ```
