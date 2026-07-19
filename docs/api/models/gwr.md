# GWR

This page documents **2** public symbols. Each entry includes its purpose, import path, full API docstring, and the maintained example that exercises it.

[Conceptual guide](../../models/index.md){ .md-button }

## `GWR`

Gaussian geographically weighted regression.

| Property | Value |
|---|---|
| Type | `class` |
| Import | `from pygwrx.models import GWR` |
| Signature | `GWR(kernel: 'Union[str, Callable[[np.ndarray, float], np.ndarray]]' = 'gaussian', bandwidth: 'Union[float, int, str, None]' = 'cv', bandwidth_method: 'str' = 'cv', adaptive: 'bool' = False, bandwidth_range: 'Optional[Tuple[float, float]]' = None, optimization_method: 'str' = 'golden_section', fit_intercept: 'bool' = True, distance_metric: 'str' = 'euclidean', sigma2_v1: 'bool' = True, verbose: 'bool' = False) -> 'None'` |
| Maintained example | [`examples/models/01_gwr.py`](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/models/01_gwr.py) |

::: pygwrx.models.GWR


## `GWRPredictionResult`

Rich prediction result returned by :meth:`GWR.predict_result`.

| Property | Value |
|---|---|
| Type | `class` |
| Import | `from pygwrx.models import GWRPredictionResult` |
| Signature | `GWRPredictionResult(predictions: 'np.ndarray', coef: 'np.ndarray', intercept: 'np.ndarray', coords: 'np.ndarray', feature_names: 'Tuple[str, ...]', coef_standard_errors: 'Optional[np.ndarray]' = None, intercept_standard_errors: 'Optional[np.ndarray]' = None, coef_t_values: 'Optional[np.ndarray]' = None, intercept_t_values: 'Optional[np.ndarray]' = None) -> None` |
| Maintained example | [`examples/models/01_gwr.py`](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/models/01_gwr.py) |

::: pygwrx.models.GWRPredictionResult


## Runnable examples used on this page

??? example "`examples/models/01_gwr.py`"

    ```python
    # SPDX-FileCopyrightText: 2026 Jinghao Hu
    # SPDX-License-Identifier: MIT
    
    """Load a bundled real dataset, fit GWR, inspect it, and predict."""
    
    from __future__ import annotations
    
    # Allow this script to run directly from any working directory.
    import sys
    from pathlib import Path
    
    _PROJECT_ROOT = Path(__file__).resolve().parents[2]
    _EXAMPLES_ROOT = _PROJECT_ROOT / "examples"
    _SRC_ROOT = _PROJECT_ROOT / "src"
    for _path in (_SRC_ROOT, _EXAMPLES_ROOT):
        if str(_path) not in sys.path:
            sys.path.insert(0, str(_path))
    
    from pygwrx import GWR, GWRPredictionResult
    from pygwrx.io import load_columbus
    
    bundle = load_columbus(return_type="dict")
    X = bundle["data"]
    y = bundle["target"]
    coords = bundle["coords"]
    
    print("dataset=", bundle["description"])
    print("features=", bundle["feature_names"])
    print("license=", bundle["license"])
    
    model = GWR(kernel="bisquare", bandwidth=24, adaptive=True).fit(X, y, coords)
    print(model.summary())
    print("score=", model.score(X, y, coords))
    
    result = model.predict_result(X[:3], coords[:3])
    assert isinstance(result, GWRPredictionResult)
    print(result.to_frame())
    ```
