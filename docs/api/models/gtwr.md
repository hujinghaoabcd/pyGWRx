# GTWR

This page documents **2** public symbols. Each entry includes its purpose, import path, full API docstring, and the maintained example that exercises it.

[Conceptual guide](../../models/index.md){ .md-button }

## `GTWR`

Geographically and temporally weighted regression.

| Property | Value |
|---|---|
| Type | `class` |
| Import | `from pygwrx.models import GTWR` |
| Signature | `GTWR(kernel: 'Union[str, Callable[[np.ndarray, float], np.ndarray]]' = 'bisquare', bandwidth: 'Union[float, int, str, None]' = 'cv', bandwidth_method: 'str' = 'cv', adaptive: 'bool' = False, bandwidth_range: 'Optional[Tuple[float, float]]' = None, lambda_st: 'Union[float, str]' = 0.05, lambda_range: 'Tuple[float, float]' = (0.0, 1.0), lambda_grid_size: 'int' = 11, ksi: 'float' = 0.0, distance_combination: 'str' = 'gwmodel', tau: 'float' = 1.0, causal: 'bool' = False, time_unit: 'str' = 'auto', optimization_method: 'str' = 'golden_section', search_grid_size: 'int' = 25, search_tol: 'float' = 1e-05, search_max_iter: 'int' = 100, fit_intercept: 'bool' = True, distance_metric: 'str' = 'euclidean', sigma2_v1: 'bool' = False, verbose: 'bool' = False) -> 'None'` |
| Maintained example | [`examples/models/05_gtwr.py`](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/models/05_gtwr.py) |

::: pygwrx.models.GTWR


## `GTWRPredictionResult`

Rich prediction result returned by :meth:`GTWR.predict_result`.

| Property | Value |
|---|---|
| Type | `class` |
| Import | `from pygwrx.models import GTWRPredictionResult` |
| Signature | `GTWRPredictionResult(predictions: 'np.ndarray', coef: 'np.ndarray', intercept: 'np.ndarray', coords: 'np.ndarray', times: 'np.ndarray', feature_names: 'Tuple[str, ...]', coef_standard_errors: 'Optional[np.ndarray]' = None, intercept_standard_errors: 'Optional[np.ndarray]' = None, coef_t_values: 'Optional[np.ndarray]' = None, intercept_t_values: 'Optional[np.ndarray]' = None) -> None` |
| Maintained example | [`examples/models/05_gtwr.py`](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/models/05_gtwr.py) |

::: pygwrx.models.GTWRPredictionResult


## Runnable examples used on this page

??? example "`examples/models/05_gtwr.py`"

    ```python
    # SPDX-FileCopyrightText: 2026 Jinghao Hu
    # SPDX-License-Identifier: MIT
    
    """Fit and predict with geographically and temporally weighted regression."""
    
    # Allow this script to run directly from any working directory.
    import sys
    from pathlib import Path
    
    _PROJECT_ROOT = Path(__file__).resolve().parents[2]
    _EXAMPLES_ROOT = _PROJECT_ROOT / "examples"
    _SRC_ROOT = _PROJECT_ROOT / "src"
    for _path in (_SRC_ROOT, _EXAMPLES_ROOT):
        if str(_path) not in sys.path:
            sys.path.insert(0, str(_path))
    
    from _common import print_model_result, temporal_regression
    
    from pygwrx import GTWR, GTWRPredictionResult
    
    X, y, coords, times = temporal_regression()
    model = GTWR(kernel="bisquare", bandwidth=24, adaptive=True, lambda_st=0.3).fit(
        X, y, coords, times
    )
    print_model_result(model)
    print("score=", model.score(X, y, coords, times=times))
    result = model.predict_result(X.iloc[:3], coords.iloc[:3], times[:3])
    assert isinstance(result, GTWRPredictionResult)
    print(result.to_frame())
    ```
