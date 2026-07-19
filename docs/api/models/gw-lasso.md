# GWLasso

This page documents **1** public symbols. Each entry includes its purpose, import path, full API docstring, and the maintained example that exercises it.

[Conceptual guide](../../models/index.md){ .md-button }

## `GWLasso`

Geographically weighted Lasso regression.

| Property | Value |
|---|---|
| Type | `class` |
| Import | `from pygwrx.models import GWLasso` |
| Signature | `GWLasso(kernel: 'Union[str, Callable]' = 'exponential', bandwidth: 'Union[float, int, str, None]' = 'cv', alpha: 'AlphaLike' = 'cv', alpha_grid: 'Optional[Sequence[float]]' = None, n_alphas: 'int' = 30, alpha_min_ratio: 'float' = 0.001, cv_folds: 'int' = 5, standardize: 'bool' = True, adaptive: 'bool' = False, bandwidth_range: 'Optional[Tuple[float, float]]' = None, n_bandwidths: 'int' = 8, max_iter: 'int' = 5000, tol: 'float' = 1e-06, active_tol: 'float' = 1e-08, fit_intercept: 'bool' = True, distance_metric: 'str' = 'euclidean', random_state: 'Optional[int]' = 0, verbose: 'bool' = False) -> 'None'` |
| Maintained example | [`examples/models/07_gw_lasso.py`](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/models/07_gw_lasso.py) |

::: pygwrx.models.GWLasso


## Runnable examples used on this page

??? example "`examples/models/07_gw_lasso.py`"

    ```python
    # SPDX-FileCopyrightText: 2026 Jinghao Hu
    # SPDX-License-Identifier: MIT
    
    """Fit geographically weighted Lasso with a fixed local penalty."""
    
    # Allow this script to run directly from any working directory.
    import sys
    from pathlib import Path
    
    _PROJECT_ROOT = Path(__file__).resolve().parents[2]
    _EXAMPLES_ROOT = _PROJECT_ROOT / "examples"
    _SRC_ROOT = _PROJECT_ROOT / "src"
    for _path in (_SRC_ROOT, _EXAMPLES_ROOT):
        if str(_path) not in sys.path:
            sys.path.insert(0, str(_path))
    
    from _common import print_model_result, spatial_regression
    
    from pygwrx import GWLasso
    
    X, y, coords = spatial_regression(n=48, p=3)
    model = GWLasso(
        bandwidth=24, adaptive=True, alpha=0.06, max_iter=1000, random_state=0
    ).fit(X, y, coords)
    print_model_result(model)
    print("selection_frequency=", model.selection_frequency_)
    print("predictions=", model.predict(X.iloc[:3], coords.iloc[:3]))
    ```
