# Optimization

This page documents **3** public symbols. Each entry includes its purpose, import path, full API docstring, and the maintained example that exercises it.

[Conceptual guide](../../guides/core-numerics.md){ .md-button }

## `OptimizationResult`

Result returned by a one-dimensional optimizer.

| Property | Value |
|---|---|
| Type | `class` |
| Import | `from pygwrx.core import OptimizationResult` |
| Signature | `OptimizationResult(value: 'Union[float, int]', score: 'float', iterations: 'int', converged: 'bool', evaluations: 'int' = 0, message: 'str' = '') -> None` |
| Maintained example | [`examples/core/06_optimization.py`](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/core/06_optimization.py) |

::: pygwrx.core.OptimizationResult


## `GoldenSectionSearch`

Golden-section search for one-dimensional minimization.

| Property | Value |
|---|---|
| Type | `class` |
| Import | `from pygwrx.core import GoldenSectionSearch` |
| Signature | `GoldenSectionSearch(tol: 'float' = 1e-05, max_iter: 'int' = 100, verbose: 'bool' = True)` |
| Maintained example | [`examples/core/06_optimization.py`](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/core/06_optimization.py) |

::: pygwrx.core.GoldenSectionSearch


## `BrentSearch`

Brent's bounded method for continuous one-dimensional minimization.

| Property | Value |
|---|---|
| Type | `class` |
| Import | `from pygwrx.core import BrentSearch` |
| Signature | `BrentSearch(tol: 'float' = 1e-05, max_iter: 'int' = 100, verbose: 'bool' = True)` |
| Maintained example | [`examples/core/06_optimization.py`](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/core/06_optimization.py) |

::: pygwrx.core.BrentSearch


## Runnable examples used on this page

??? example "`examples/core/06_optimization.py`"

    ```python
    # SPDX-FileCopyrightText: 2026 Jinghao Hu
    # SPDX-License-Identifier: MIT
    
    """Use both public scalar optimizers and the OptimizationResult container."""
    
    # Allow this script to run directly from any working directory.
    import sys
    from pathlib import Path
    
    _PROJECT_ROOT = Path(__file__).resolve().parents[2]
    _EXAMPLES_ROOT = _PROJECT_ROOT / "examples"
    _SRC_ROOT = _PROJECT_ROOT / "src"
    for _path in (_SRC_ROOT, _EXAMPLES_ROOT):
        if str(_path) not in sys.path:
            sys.path.insert(0, str(_path))
    
    from pygwrx.core import BrentSearch, GoldenSectionSearch, OptimizationResult
    
    
    def objective(x):
        """Simple convex objective with a known minimum."""
        return (x - 2.5) ** 2 + 1.0
    
    
    golden = GoldenSectionSearch(tol=1e-7, max_iter=100, verbose=False)
    brent = BrentSearch(tol=1e-7, max_iter=100, verbose=False)
    print("golden=", golden.minimize(objective, 0.0, 5.0))
    print("brent=", brent.minimize(objective, 0.0, 5.0))
    print("manual_result=", OptimizationResult(2.5, 1.0, 10, True, evaluations=12))
    ```
