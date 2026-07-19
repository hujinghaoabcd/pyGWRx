# SGWR

This page documents **1** public symbols. Each entry includes its purpose, import path, full API docstring, and the maintained example that exercises it.

[Conceptual guide](../../models/index.md){ .md-button }

## `SGWR`

Similarity and geographically weighted regression.

| Property | Value |
|---|---|
| Type | `class` |
| Import | `from pygwrx.models import SGWR` |
| Signature | `SGWR(bandwidth: 'Bandwidth' = 'aicc', adaptive: 'bool' = True, kernel: 'str' = 'bisquare', alpha: 'Alpha' = 'aicc', similarity_vars: 'Optional[Sequence[Union[int, str]]]' = None, *, standardize_similarity: 'bool' = True, bandwidth_kernel: 'Optional[str]' = None, bandwidth_range: 'Optional[Tuple[float, float]]' = None, alpha_range: 'Tuple[float, float]' = (0.01, 1.0), alpha_grid_size: 'int' = 21, fit_intercept: 'bool' = True, distance_metric: 'str' = 'euclidean', sigma2_v1: 'bool' = True, ridge: 'float' = 0.0, store_weights: 'bool' = True, verbose: 'bool' = False) -> 'None'` |
| Maintained example | [`examples/models/15_sgwr.py`](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/models/15_sgwr.py) |

::: pygwrx.models.SGWR


## Runnable examples used on this page

??? example "`examples/models/15_sgwr.py`"

    ```python
    # SPDX-FileCopyrightText: 2026 Jinghao Hu
    # SPDX-License-Identifier: MIT
    
    """Fit similarity and geographically weighted regression."""
    
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
    
    from pygwrx import SGWR
    
    X, y, coords = spatial_regression(n=48, p=3)
    model = SGWR(
        bandwidth=24,
        adaptive=True,
        alpha=0.45,
        similarity_vars=["x1", "x2"],
        store_weights=True,
    ).fit(X, y, coords)
    print_model_result(model)
    print("combined_weights_shape=", model.combined_weights_.shape)
    print("predictions=", model.predict(X.iloc[:3], coords.iloc[:3]))
    ```
