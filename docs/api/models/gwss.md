# GWSS

This page documents **1** public symbols. Each entry includes its purpose, import path, full API docstring, and the maintained example that exercises it.

[Conceptual guide](../../models/index.md){ .md-button }

## `GWSS`

Compute geographically weighted summary statistics.

| Property | Value |
|---|---|
| Type | `class` |
| Import | `from pygwrx.models import GWSS` |
| Signature | `GWSS(kernel: 'str \| Any' = 'bisquare', bandwidth: 'float \| int \| None' = None, adaptive: 'bool' = False, quantile: 'bool' = False, verbose: 'bool' = False) -> 'None'` |
| Maintained example | [`examples/models/11_gwss.py`](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/models/11_gwss.py) |

::: pygwrx.models.GWSS


## Runnable examples used on this page

??? example "`examples/models/11_gwss.py`"

    ```python
    # SPDX-FileCopyrightText: 2026 Jinghao Hu
    # SPDX-License-Identifier: MIT
    
    """Compute geographically weighted summary statistics."""
    
    # Allow this script to run directly from any working directory.
    import sys
    from pathlib import Path
    
    _PROJECT_ROOT = Path(__file__).resolve().parents[2]
    _EXAMPLES_ROOT = _PROJECT_ROOT / "examples"
    _SRC_ROOT = _PROJECT_ROOT / "src"
    for _path in (_SRC_ROOT, _EXAMPLES_ROOT):
        if str(_path) not in sys.path:
            sys.path.insert(0, str(_path))
    
    from _common import spatial_regression
    
    from pygwrx import GWSS
    
    X, _, coords = spatial_regression(n=48, p=3)
    model = GWSS(bandwidth=24, adaptive=True, quantile=True).fit(X, coords)
    print(model.summary())
    print("local_means_shape=", model.local_mean_.shape)
    print("local_correlation_pairs=", sorted(model.local_corr_))
    print("first_correlation_shape=", next(iter(model.local_corr_.values())).shape)
    ```
