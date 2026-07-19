# LGGWR

This page documents **2** public symbols. Each entry includes its purpose, import path, full API docstring, and the maintained example that exercises it.

[Conceptual guide](../../models/index.md){ .md-button }

## `LGGWR`

Latent-Geometry Geographically Weighted Regression.

| Property | Value |
|---|---|
| Type | `class` |
| Import | `from pygwrx.models import LGGWR` |
| Signature | `LGGWR(latent_dim: 'int' = 2, bandwidth: 'BandwidthLike' = None, adaptive: 'bool' = False, kernel: 'str' = 'gaussian', geometry: 'str' = 'joint', learning_rate: 'float' = 0.05, max_iter: 'int' = 100, tol: 'float' = 1e-06, lambda_reg: 'float' = 0.0, orthogonal_constraint: 'Optional[bool]' = None, grad_clip: 'float' = 10.0, patience: 'int' = 20, select_bandwidth: 'bool' = True, random_state: 'Optional[int]' = None, verbose: 'bool' = False, *, fit_intercept: 'bool' = True, standardize_geometry: 'bool' = True, initialization: 'str' = 'coordinate', n_restarts: 'int' = 1, scale_constraint: 'str' = 'frobenius', bandwidth_updates: 'int' = 1) -> 'None'` |
| Maintained example | [`examples/models/18_lg_gwr.py`](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/models/18_lg_gwr.py) |

::: pygwrx.models.LGGWR


## `LGGWRPredictionResult`

Detailed LG-GWR predictions at evaluation locations.

| Property | Value |
|---|---|
| Type | `class` |
| Import | `from pygwrx.models import LGGWRPredictionResult` |
| Signature | `LGGWRPredictionResult(predictions: 'np.ndarray', coefficients: 'np.ndarray', intercepts: 'np.ndarray', coords: 'np.ndarray', latent_coords: 'np.ndarray', feature_names: 'Tuple[str, ...]') -> None` |
| Maintained example | [`examples/models/18_lg_gwr.py`](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/models/18_lg_gwr.py) |

::: pygwrx.models.LGGWRPredictionResult


## Runnable examples used on this page

??? example "`examples/models/18_lg_gwr.py`"

    ```python
    # SPDX-FileCopyrightText: 2026 Jinghao Hu
    # SPDX-License-Identifier: MIT
    
    """Fit latent-geometry GWR with auxiliary contextual attributes."""
    
    # Allow this script to run directly from any working directory.
    import sys
    from pathlib import Path
    
    _PROJECT_ROOT = Path(__file__).resolve().parents[2]
    _EXAMPLES_ROOT = _PROJECT_ROOT / "examples"
    _SRC_ROOT = _PROJECT_ROOT / "src"
    for _path in (_SRC_ROOT, _EXAMPLES_ROOT):
        if str(_path) not in sys.path:
            sys.path.insert(0, str(_path))
    
    from _common import latent_regression, print_model_result
    
    from pygwrx import LGGWR, LGGWRPredictionResult
    
    X, y, coords, attributes = latent_regression()
    model = LGGWR(
        latent_dim=2, bandwidth=2.5, select_bandwidth=False, max_iter=8, random_state=0
    ).fit(X, y, coords, attributes)
    print_model_result(model)
    print("latent_coordinates_shape=", model.latent_coords_.shape)
    result = model.predict_result(X.iloc[:3], coords.iloc[:3], attributes.iloc[:3])
    assert isinstance(result, LGGWRPredictionResult)
    print(result.to_frame())
    ```
