# GWPCA

This page documents **1** public symbols. Each entry includes its purpose, import path, full API docstring, and the maintained example that exercises it.

[Conceptual guide](../../models/index.md){ .md-button }

## `GWPCA`

Fit a basic geographically weighted principal component analysis.

| Property | Value |
|---|---|
| Type | `class` |
| Import | `from pygwrx.models import GWPCA` |
| Signature | `GWPCA(n_components: 'int' = 2, kernel: 'str \| Any' = 'bisquare', bandwidth: 'float \| int \| str \| None' = 'cv', adaptive: 'bool' = True, scaling: 'bool' = True, compute_scores: 'bool' = False, verbose: 'bool' = False) -> 'None'` |
| Maintained example | [`examples/models/09_gwpca.py`](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/models/09_gwpca.py) |

::: pygwrx.models.GWPCA


## Runnable examples used on this page

??? example "`examples/models/09_gwpca.py`"

    ```python
    # SPDX-FileCopyrightText: 2026 Jinghao Hu
    # SPDX-License-Identifier: MIT
    
    """Fit GWPCA, inspect local loadings, and transform observations."""
    
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
    
    from pygwrx import GWPCA
    
    X, _, coords = spatial_regression(n=48, p=3)
    model = GWPCA(n_components=2, bandwidth=24, adaptive=True).fit(
        X, coords, compute_cv=True
    )
    print_model_result(model)
    print("scores_shape=", model.transform(X, coords).shape)
    print("explained_variance_first_location=", model.local_pv_[0])
    ```
