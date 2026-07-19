# RGWR

This page documents **1** public symbols. Each entry includes its purpose, import path, full API docstring, and the maintained example that exercises it.

[Conceptual guide](../../models/index.md){ .md-button }

## `RGWR`

Classical robust geographically weighted regression.

| Property | Value |
|---|---|
| Type | `class` |
| Import | `from pygwrx.models import RGWR` |
| Signature | `RGWR(kernel: 'Union[str, Callable[[np.ndarray, float], np.ndarray]]' = 'gaussian', bandwidth: 'Union[float, int, str, None]' = 'cv', bandwidth_method: 'str' = 'cv', adaptive: 'bool' = False, bandwidth_range: 'Optional[Tuple[float, float]]' = None, optimization_method: 'str' = 'golden_section', fit_intercept: 'bool' = True, distance_metric: 'str' = 'euclidean', sigma2_v1: 'bool' = True, method: 'str' = 'automatic', max_iter: 'int' = 20, tol: 'float' = 1e-05, cut1: 'float' = 2.0, cut2: 'float' = 3.0, cut_filter: 'float' = 3.0, verbose: 'bool' = False) -> 'None'` |
| Maintained example | [`examples/models/03_rgwr.py`](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/models/03_rgwr.py) |

::: pygwrx.models.RGWR


## Runnable examples used on this page

??? example "`examples/models/03_rgwr.py`"

    ```python
    # SPDX-FileCopyrightText: 2026 Jinghao Hu
    # SPDX-License-Identifier: MIT
    
    """Fit robust GWR in automatic down-weighting mode."""
    
    # Allow this script to run directly from any working directory.
    import sys
    from pathlib import Path
    
    _PROJECT_ROOT = Path(__file__).resolve().parents[2]
    _EXAMPLES_ROOT = _PROJECT_ROOT / "examples"
    _SRC_ROOT = _PROJECT_ROOT / "src"
    for _path in (_SRC_ROOT, _EXAMPLES_ROOT):
        if str(_path) not in sys.path:
            sys.path.insert(0, str(_path))
    
    import numpy as np
    from _common import print_model_result, spatial_regression
    
    from pygwrx import RGWR
    
    X, y, coords = spatial_regression()
    y = y.copy()
    y[[2, 20]] += np.array([5.0, -4.0])
    model = RGWR(bandwidth=24, adaptive=True, max_iter=8).fit(X, y, coords)
    print_model_result(model)
    print("robust_weights=", model.robust_weights_[:8])
    print("predictions=", model.predict(X.iloc[:3], coords.iloc[:3]))
    ```
