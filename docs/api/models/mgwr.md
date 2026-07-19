# MGWR

This page documents **1** public symbols. Each entry includes its purpose, import path, full API docstring, and the maintained example that exercises it.

[Conceptual guide](../../models/index.md){ .md-button }

## `MGWR`

Gaussian multiscale geographically weighted regression.

| Property | Value |
|---|---|
| Type | `class` |
| Import | `from pygwrx.models import MGWR` |
| Signature | `MGWR(kernel: 'Union[str, Callable[[np.ndarray, float], np.ndarray]]' = 'bisquare', bandwidths: 'BandwidthInput' = None, bandwidth_method: 'str' = 'aicc', adaptive: 'bool' = True, bandwidth_range: 'BandwidthRange' = None, bandwidth_ranges: 'BandwidthRanges' = None, init_bandwidth: 'Optional[Bandwidth]' = None, optimization_method: 'str' = 'golden_section', search_tol: 'float' = 1e-06, search_max_iter: 'int' = 200, max_iter: 'int' = 200, tol: 'float' = 1e-05, rss_score: 'bool' = False, bws_same_times: 'int' = 5, fit_intercept: 'bool' = True, distance_metric: 'str' = 'euclidean', sigma2_v1: 'bool' = True, verbose: 'bool' = False) -> 'None'` |
| Maintained example | [`examples/models/02_mgwr.py`](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/models/02_mgwr.py) |

::: pygwrx.models.MGWR


## Runnable examples used on this page

??? example "`examples/models/02_mgwr.py`"

    ```python
    # SPDX-FileCopyrightText: 2026 Jinghao Hu
    # SPDX-License-Identifier: MIT
    
    """Fit MGWR with fixed variable-specific bandwidths."""
    
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
    
    from pygwrx import MGWR
    
    X, y, coords = spatial_regression(n=48, p=2)
    model = MGWR(bandwidths=[24, 26, 28], adaptive=True, max_iter=8, tol=0.5).fit(
        X, y, coords, compute_inference=True
    )
    print_model_result(model)
    try:
        model.predict(X.iloc[:2], coords.iloc[:2])
    except NotImplementedError as exc:
        print("Expected MGWR prediction limitation:", exc)
    ```
