# LCRGWR

This page documents **1** public symbols. Each entry includes its purpose, import path, full API docstring, and the maintained example that exercises it.

[Conceptual guide](../../models/index.md){ .md-button }

## `LCRGWR`

Locally compensated ridge geographically weighted regression.

| Property | Value |
|---|---|
| Type | `class` |
| Import | `from pygwrx.models import LCRGWR` |
| Signature | `LCRGWR(kernel: 'Union[str, Callable[[np.ndarray, float], np.ndarray]]' = 'bisquare', bandwidth: 'Union[float, int, str, None]' = 'cv', bandwidth_method: 'str' = 'cv', adaptive: 'bool' = False, bandwidth_range: 'Optional[Tuple[float, float]]' = None, optimization_method: 'str' = 'golden_section', lambda_ridge: 'float' = 0.0, lambda_adjust: 'bool' = True, cn_thresh: 'float' = 30.0, fit_intercept: 'bool' = True, distance_metric: 'str' = 'euclidean', sigma2_v1: 'bool' = True, verbose: 'bool' = False) -> 'None'` |
| Maintained example | [`examples/models/13_lcr_gwr.py`](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/models/13_lcr_gwr.py) |

::: pygwrx.models.LCRGWR


## Runnable examples used on this page

??? example "`examples/models/13_lcr_gwr.py`"

    ```python
    # SPDX-FileCopyrightText: 2026 Jinghao Hu
    # SPDX-License-Identifier: MIT
    
    """Fit locally compensated ridge GWR for collinear predictors."""
    
    # Allow this script to run directly from any working directory.
    import sys
    from pathlib import Path
    
    _PROJECT_ROOT = Path(__file__).resolve().parents[2]
    _EXAMPLES_ROOT = _PROJECT_ROOT / "examples"
    _SRC_ROOT = _PROJECT_ROOT / "src"
    for _path in (_SRC_ROOT, _EXAMPLES_ROOT):
        if str(_path) not in sys.path:
            sys.path.insert(0, str(_path))
    
    from _common import collinear_regression, print_model_result
    
    from pygwrx import LCRGWR
    
    X, y, coords = collinear_regression()
    model = LCRGWR(bandwidth=28, adaptive=True, cn_thresh=15.0, lambda_adjust=True).fit(
        X, y, coords
    )
    print_model_result(model)
    print("local_condition_numbers=", model.local_condition_numbers_[:5])
    print("local_lambdas=", model.local_lambdas_[:5])
    ```
