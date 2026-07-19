# GWGLM

This page documents **2** public symbols. Each entry includes its purpose, import path, full API docstring, and the maintained example that exercises it.

[Conceptual guide](../../models/index.md){ .md-button }

## `GWGLM`

Geographically weighted generalized linear model.

| Property | Value |
|---|---|
| Type | `class` |
| Import | `from pygwrx.models import GWGLM` |
| Signature | `GWGLM(family: 'FamilyName' = 'gaussian', kernel: 'KernelLike' = 'bisquare', bandwidth: 'BandwidthLike' = 'cv', bandwidth_method: 'str' = 'aicc', adaptive: 'bool' = False, bandwidth_range: 'Optional[Tuple[float, float]]' = None, optimization_method: 'str' = 'golden_section', max_iter: 'int' = 100, tol: 'float' = 1e-06, fit_intercept: 'bool' = True, distance_metric: 'str' = 'euclidean', sigma2_v1: 'bool' = True, verbose: 'bool' = False) -> 'None'` |
| Maintained example | [`examples/models/06_gwglm.py`](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/models/06_gwglm.py) |

::: pygwrx.models.GWGLM


## `GWGLMPredictionResult`

Rich prediction result returned by :meth:`GWGLM.predict_result`.

| Property | Value |
|---|---|
| Type | `class` |
| Import | `from pygwrx.models import GWGLMPredictionResult` |
| Signature | `GWGLMPredictionResult(predictions: 'np.ndarray', linear_predictor: 'np.ndarray', coef: 'np.ndarray', intercept: 'np.ndarray', coords: 'np.ndarray', feature_names: 'Tuple[str, ...]', family: 'str', exposure: 'Optional[np.ndarray]' = None, coef_standard_errors: 'Optional[np.ndarray]' = None, intercept_standard_errors: 'Optional[np.ndarray]' = None, coef_z_values: 'Optional[np.ndarray]' = None, intercept_z_values: 'Optional[np.ndarray]' = None) -> None` |
| Maintained example | [`examples/models/06_gwglm.py`](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/models/06_gwglm.py) |

::: pygwrx.models.GWGLMPredictionResult


## Runnable examples used on this page

??? example "`examples/models/06_gwglm.py`"

    ```python
    # SPDX-FileCopyrightText: 2026 Jinghao Hu
    # SPDX-License-Identifier: MIT
    
    """Fit Gaussian, binomial, and Poisson GWGLM families."""
    
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
    from _common import count_regression, print_model_result, spatial_regression
    
    from pygwrx import GWGLM, GWGLMPredictionResult
    
    X, y, coords = spatial_regression(p=2)
    gaussian = GWGLM(family="gaussian", bandwidth=24, adaptive=True).fit(X, y, coords)
    print_model_result(gaussian)
    
    binary = (y > np.median(y)).astype(int)
    binomial = GWGLM(family="binomial", bandwidth=24, adaptive=True).fit(X, binary, coords)
    binomial_result = binomial.predict_result(X.iloc[:3], coords.iloc[:3])
    assert isinstance(binomial_result, GWGLMPredictionResult)
    print(binomial_result.to_frame())
    
    Xc, counts, coordsc, exposure = count_regression()
    poisson = GWGLM(family="poisson", bandwidth=24, adaptive=True).fit(
        Xc, counts, coordsc, exposure=exposure
    )
    print(
        "poisson means=",
        poisson.predict(Xc.iloc[:3], coordsc.iloc[:3], exposure=exposure[:3]),
    )
    ```
