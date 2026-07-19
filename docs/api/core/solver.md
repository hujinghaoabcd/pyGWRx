# Local solvers

This page documents **4** public symbols. Each entry includes its purpose, import path, full API docstring, and the maintained example that exercises it.

[Conceptual guide](../../guides/core-numerics.md){ .md-button }

## `weighted_least_squares`

Solve a weighted least-squares problem.

| Property | Value |
|---|---|
| Type | `function` |
| Import | `from pygwrx.core import weighted_least_squares` |
| Signature | `weighted_least_squares(X: 'np.ndarray', y: 'np.ndarray', weights: 'np.ndarray', *, ridge: 'float' = 1e-08) -> 'Tuple[np.ndarray, np.ndarray]'` |
| Maintained example | [`examples/core/04_solver.py`](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/core/04_solver.py) |

::: pygwrx.core.weighted_least_squares


## `local_regression`

Perform local weighted regression at target locations.

| Property | Value |
|---|---|
| Type | `function` |
| Import | `from pygwrx.core import local_regression` |
| Signature | `local_regression(X: 'np.ndarray', y: 'np.ndarray', coords: 'np.ndarray', target_coords: 'np.ndarray', kernel_func: 'KernelFunction', bandwidth: 'float', distance_metric: 'str' = 'euclidean', adaptive: 'bool' = False, *, ridge: 'float' = 1e-08) -> 'np.ndarray'` |
| Maintained example | [`examples/core/04_solver.py`](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/core/04_solver.py) |

::: pygwrx.core.local_regression


## `compute_hat_matrix`

Compute the GWR hat matrix ``S`` such that ``y_hat = S @ y``.

| Property | Value |
|---|---|
| Type | `function` |
| Import | `from pygwrx.core import compute_hat_matrix` |
| Signature | `compute_hat_matrix(X: 'np.ndarray', coords: 'np.ndarray', kernel_func: 'KernelFunction', bandwidth: 'float', distance_metric: 'str' = 'euclidean', adaptive: 'bool' = False, *, ridge: 'float' = 1e-08) -> 'np.ndarray'` |
| Maintained example | [`examples/core/04_solver.py`](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/core/04_solver.py) |

::: pygwrx.core.compute_hat_matrix


## `adaptive_bandwidth_weights`

Convert an adaptive neighbour-order bandwidth into a distance scale.

| Property | Value |
|---|---|
| Type | `function` |
| Import | `from pygwrx.core import adaptive_bandwidth_weights` |
| Signature | `adaptive_bandwidth_weights(distances: 'np.ndarray', k_nearest: 'int') -> 'float'` |
| Maintained example | [`examples/core/04_solver.py`](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/core/04_solver.py) |

::: pygwrx.core.adaptive_bandwidth_weights


## Runnable examples used on this page

??? example "`examples/core/04_solver.py`"

    ```python
    # SPDX-FileCopyrightText: 2026 Jinghao Hu
    # SPDX-License-Identifier: MIT
    
    """Run all public local-regression solver utilities."""
    
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
        adaptive_bandwidth_weights,
        compute_hat_matrix,
        gaussian_kernel,
        local_regression,
        weighted_least_squares,
    )
    
    rng = np.random.default_rng(0)
    coords = rng.uniform(0.0, 5.0, size=(20, 2))
    x = rng.normal(size=20)
    X = np.column_stack((np.ones(20), x))
    y = 1.0 + 2.0 * x + rng.normal(0.0, 0.05, 20)
    distances = np.linalg.norm(coords - coords[0], axis=1)
    weights = gaussian_kernel(distances, bandwidth=2.0)
    beta, covariance = weighted_least_squares(X, y, weights)
    print("beta=", beta)
    print("covariance_shape=", covariance.shape)
    print("adaptive_scale=", adaptive_bandwidth_weights(distances, 8))
    print(
        "local_parameters=",
        local_regression(X, y, coords, coords[:3], gaussian_kernel, 2.0),
    )
    hat = compute_hat_matrix(X, coords, gaussian_kernel, 2.0)
    print("hat_shape_trace=", hat.shape, np.trace(hat))
    ```
