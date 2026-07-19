# Bandwidth selection

This page documents **5** public symbols. Each entry includes its purpose, import path, full API docstring, and the maintained example that exercises it.

[Conceptual guide](../../guides/core-numerics.md){ .md-button }

## `BandwidthSelector`

Abstract base class for bandwidth selection methods.

| Property | Value |
|---|---|
| Type | `class` |
| Import | `from pygwrx.core import BandwidthSelector` |
| Signature | `BandwidthSelector()` |
| Maintained example | [`examples/core/07_bandwidth_selectors.py`](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/core/07_bandwidth_selectors.py) |

::: pygwrx.core.BandwidthSelector


## `CrossValidationSelector`

Select bandwidth by strict leave-one-out squared prediction error.

| Property | Value |
|---|---|
| Type | `class` |
| Import | `from pygwrx.core import CrossValidationSelector` |
| Signature | `CrossValidationSelector(n_intervals: 'int' = 20, optimization_method: 'str' = 'golden_section', adaptive: 'bool' = False, verbose: 'bool' = False) -> 'None'` |
| Maintained example | [`examples/core/07_bandwidth_selectors.py`](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/core/07_bandwidth_selectors.py) |

::: pygwrx.core.CrossValidationSelector


## `AICSelector`

Select bandwidth using Gaussian GWR AIC or AICc.

| Property | Value |
|---|---|
| Type | `class` |
| Import | `from pygwrx.core import AICSelector` |
| Signature | `AICSelector(n_intervals: 'int' = 20, corrected: 'bool' = True, adaptive: 'bool' = False, optimization_method: 'str' = 'golden_section', verbose: 'bool' = False) -> 'None'` |
| Maintained example | [`examples/core/07_bandwidth_selectors.py`](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/core/07_bandwidth_selectors.py) |

::: pygwrx.core.AICSelector


## `BICSelector`

Select bandwidth using Gaussian GWR BIC.

| Property | Value |
|---|---|
| Type | `class` |
| Import | `from pygwrx.core import BICSelector` |
| Signature | `BICSelector(n_intervals: 'int' = 20, optimization_method: 'str' = 'golden_section', adaptive: 'bool' = False, verbose: 'bool' = False) -> 'None'` |
| Maintained example | [`examples/core/07_bandwidth_selectors.py`](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/core/07_bandwidth_selectors.py) |

::: pygwrx.core.BICSelector


## `get_bandwidth_selector`

Create a bandwidth selector by method name.

| Property | Value |
|---|---|
| Type | `function` |
| Import | `from pygwrx.core import get_bandwidth_selector` |
| Signature | `get_bandwidth_selector(method: 'str', **kwargs) -> 'BandwidthSelector'` |
| Maintained example | [`examples/core/07_bandwidth_selectors.py`](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/core/07_bandwidth_selectors.py) |

::: pygwrx.core.get_bandwidth_selector


## Runnable examples used on this page

??? example "`examples/core/07_bandwidth_selectors.py`"

    ```python
    # SPDX-FileCopyrightText: 2026 Jinghao Hu
    # SPDX-License-Identifier: MIT
    
    """Select bandwidths with CV, AIC/AICc, and BIC selectors."""
    
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
    from _common import spatial_regression
    
    from pygwrx.core import (
        AICSelector,
        BandwidthSelector,
        BICSelector,
        CrossValidationSelector,
        gaussian_kernel,
        get_bandwidth_selector,
    )
    
    X, y, coords = spatial_regression(n=28, p=2)
    Xa, ya, ca = X.to_numpy(), np.asarray(y), coords.to_numpy()
    selectors = [
        CrossValidationSelector(n_intervals=5, adaptive=True, verbose=False),
        AICSelector(n_intervals=5, corrected=False, adaptive=True, verbose=False),
        AICSelector(n_intervals=5, corrected=True, adaptive=True, verbose=False),
        BICSelector(n_intervals=5, adaptive=True, verbose=False),
    ]
    for selector in selectors:
        print(
            type(selector).__name__,
            selector.select(Xa, ya, ca, gaussian_kernel, bandwidth_range=(10, 18)),
        )
    print("factory=", type(get_bandwidth_selector("aicc", adaptive=True)).__name__)
    print("abstract_base=", BandwidthSelector)
    ```
