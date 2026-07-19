# Kernels

This page documents **6** public symbols. Each entry includes its purpose, import path, full API docstring, and the maintained example that exercises it.

[Conceptual guide](../../guides/core-numerics.md){ .md-button }

## `gaussian_kernel`

Compute Gaussian kernel weights.

| Property | Value |
|---|---|
| Type | `function` |
| Import | `from pygwrx.core import gaussian_kernel` |
| Signature | `gaussian_kernel(distances: 'np.ndarray', bandwidth: 'float') -> 'np.ndarray'` |
| Maintained example | [`examples/core/01_kernels.py`](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/core/01_kernels.py) |

::: pygwrx.core.gaussian_kernel


## `bisquare_kernel`

Compute bi-square (quartic) kernel weights.

| Property | Value |
|---|---|
| Type | `function` |
| Import | `from pygwrx.core import bisquare_kernel` |
| Signature | `bisquare_kernel(distances: 'np.ndarray', bandwidth: 'float') -> 'np.ndarray'` |
| Maintained example | [`examples/core/01_kernels.py`](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/core/01_kernels.py) |

::: pygwrx.core.bisquare_kernel


## `exponential_kernel`

Compute exponential kernel weights.

| Property | Value |
|---|---|
| Type | `function` |
| Import | `from pygwrx.core import exponential_kernel` |
| Signature | `exponential_kernel(distances: 'np.ndarray', bandwidth: 'float') -> 'np.ndarray'` |
| Maintained example | [`examples/core/01_kernels.py`](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/core/01_kernels.py) |

::: pygwrx.core.exponential_kernel


## `tricube_kernel`

Compute tri-cube kernel weights.

| Property | Value |
|---|---|
| Type | `function` |
| Import | `from pygwrx.core import tricube_kernel` |
| Signature | `tricube_kernel(distances: 'np.ndarray', bandwidth: 'float') -> 'np.ndarray'` |
| Maintained example | [`examples/core/01_kernels.py`](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/core/01_kernels.py) |

::: pygwrx.core.tricube_kernel


## `boxcar_kernel`

Compute boxcar (uniform) kernel weights.

| Property | Value |
|---|---|
| Type | `function` |
| Import | `from pygwrx.core import boxcar_kernel` |
| Signature | `boxcar_kernel(distances: 'np.ndarray', bandwidth: 'float') -> 'np.ndarray'` |
| Maintained example | [`examples/core/01_kernels.py`](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/core/01_kernels.py) |

::: pygwrx.core.boxcar_kernel


## `get_kernel_function`

Return a built-in kernel by name or validate a custom callable.

| Property | Value |
|---|---|
| Type | `function` |
| Import | `from pygwrx.core import get_kernel_function` |
| Signature | `get_kernel_function(kernel: 'KernelLike') -> 'KernelCallable'` |
| Maintained example | [`examples/core/01_kernels.py`](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/core/01_kernels.py) |

::: pygwrx.core.get_kernel_function


## Runnable examples used on this page

??? example "`examples/core/01_kernels.py`"

    ```python
    # SPDX-FileCopyrightText: 2026 Jinghao Hu
    # SPDX-License-Identifier: MIT
    
    """Evaluate every public kernel and resolve kernels by name or callable."""
    
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
    
    from pygwrx.core import (
        bisquare_kernel,
        boxcar_kernel,
        exponential_kernel,
        gaussian_kernel,
        get_kernel_function,
        tricube_kernel,
    )
    
    distances = np.array([0.0, 0.5, 1.0, 2.0])
    for kernel in (
        gaussian_kernel,
        bisquare_kernel,
        exponential_kernel,
        tricube_kernel,
        boxcar_kernel,
    ):
        print(kernel.__name__, kernel(distances, bandwidth=1.5))
    print("resolved=", get_kernel_function("bisquare").__name__)
    print("callable_passthrough=", get_kernel_function(gaussian_kernel) is gaussian_kernel)
    ```
