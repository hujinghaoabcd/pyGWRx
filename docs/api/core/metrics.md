# Metrics

This page documents **11** public symbols. Each entry includes its purpose, import path, full API docstring, and the maintained example that exercises it.

[Conceptual guide](../../guides/core-numerics.md){ .md-button }

## `compute_r_squared`

Compute the coefficient of determination, R².

| Property | Value |
|---|---|
| Type | `function` |
| Import | `from pygwrx.core import compute_r_squared` |
| Signature | `compute_r_squared(y_true: 'np.ndarray', y_pred: 'np.ndarray') -> 'float'` |
| Maintained example | [`examples/core/05_metrics.py`](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/core/05_metrics.py) |

::: pygwrx.core.compute_r_squared


## `compute_adjusted_r_squared`

Compute GWR adjusted R² from residual effective degrees of freedom.

| Property | Value |
|---|---|
| Type | `function` |
| Import | `from pygwrx.core import compute_adjusted_r_squared` |
| Signature | `compute_adjusted_r_squared(y_true: 'np.ndarray', y_pred: 'np.ndarray', edf: 'float') -> 'float'` |
| Maintained example | [`examples/core/05_metrics.py`](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/core/05_metrics.py) |

::: pygwrx.core.compute_adjusted_r_squared


## `compute_aic`

Compute Gaussian GWR AIC using trace(S) as the complexity term.

| Property | Value |
|---|---|
| Type | `function` |
| Import | `from pygwrx.core import compute_aic` |
| Signature | `compute_aic(y_true: 'np.ndarray', y_pred: 'np.ndarray', n_params: 'float') -> 'float'` |
| Maintained example | [`examples/core/05_metrics.py`](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/core/05_metrics.py) |

::: pygwrx.core.compute_aic


## `compute_aicc`

Compute Gaussian GWR corrected AIC (AICc). Compute the corrected Akaike information criterion for Gaussian GWR.

| Property | Value |
|---|---|
| Type | `function` |
| Import | `from pygwrx.core import compute_aicc` |
| Signature | `compute_aicc(y_true: 'np.ndarray', y_pred: 'np.ndarray', n_params: 'float') -> 'float'` |
| Maintained example | [`examples/core/05_metrics.py`](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/core/05_metrics.py) |

::: pygwrx.core.compute_aicc


## `compute_bic`

Compute Gaussian GWR BIC using trace(S).

| Property | Value |
|---|---|
| Type | `function` |
| Import | `from pygwrx.core import compute_bic` |
| Signature | `compute_bic(y_true: 'np.ndarray', y_pred: 'np.ndarray', trace_S: 'float') -> 'float'` |
| Maintained example | [`examples/core/05_metrics.py`](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/core/05_metrics.py) |

::: pygwrx.core.compute_bic


## `compute_local_r_squared`

Compute local weighted R² values. Compute a locally weighted coefficient of determination.

| Property | Value |
|---|---|
| Type | `function` |
| Import | `from pygwrx.core import compute_local_r_squared` |
| Signature | `compute_local_r_squared(y_true: 'np.ndarray', y_pred: 'np.ndarray', weights: 'np.ndarray') -> 'np.ndarray'` |
| Maintained example | [`examples/core/05_metrics.py`](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/core/05_metrics.py) |

::: pygwrx.core.compute_local_r_squared


## `compute_effective_parameters`

Return trace(S), the first common effective-parameter convention.

| Property | Value |
|---|---|
| Type | `function` |
| Import | `from pygwrx.core import compute_effective_parameters` |
| Signature | `compute_effective_parameters(hat_matrix: 'np.ndarray') -> 'float'` |
| Maintained example | [`examples/core/05_metrics.py`](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/core/05_metrics.py) |

::: pygwrx.core.compute_effective_parameters


## `compute_diagnostics`

Compute diagnostic statistics for a Gaussian GWR-style model.

| Property | Value |
|---|---|
| Type | `function` |
| Import | `from pygwrx.core import compute_diagnostics` |
| Signature | `compute_diagnostics(y_true: 'np.ndarray', y_pred: 'np.ndarray', hat_matrix: 'Optional[np.ndarray]' = None, n_features: 'Optional[int]' = None, compute_gwr_stats: 'bool' = False, *, trace_S: 'Optional[float]' = None, trace_StS: 'Optional[float]' = None) -> 'Dict[str, float]'` |
| Maintained example | [`examples/core/05_metrics.py`](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/core/05_metrics.py) |

::: pygwrx.core.compute_diagnostics


## `compute_trace_statistics`

Compute trace(S) and trace(S'S) from a validated hat matrix.

| Property | Value |
|---|---|
| Type | `function` |
| Import | `from pygwrx.core import compute_trace_statistics` |
| Signature | `compute_trace_statistics(hat_matrix: 'np.ndarray') -> 'Dict[str, float]'` |
| Maintained example | [`examples/core/05_metrics.py`](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/core/05_metrics.py) |

::: pygwrx.core.compute_trace_statistics


## `compute_edf`

Compute residual effective degrees of freedom using the GWmodel convention.

| Property | Value |
|---|---|
| Type | `function` |
| Import | `from pygwrx.core import compute_edf` |
| Signature | `compute_edf(n: 'int', trace_S: 'float', trace_StS: 'float') -> 'float'` |
| Maintained example | [`examples/core/05_metrics.py`](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/core/05_metrics.py) |

::: pygwrx.core.compute_edf


## `compute_enp`

Compute the GWmodel-style effective number of parameters. Compute the effective parameter count using the GWmodel convention.

| Property | Value |
|---|---|
| Type | `function` |
| Import | `from pygwrx.core import compute_enp` |
| Signature | `compute_enp(trace_S: 'float', trace_StS: 'float') -> 'float'` |
| Maintained example | [`examples/core/05_metrics.py`](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/core/05_metrics.py) |

::: pygwrx.core.compute_enp


## Runnable examples used on this page

??? example "`examples/core/05_metrics.py`"

    ```python
    # SPDX-FileCopyrightText: 2026 Jinghao Hu
    # SPDX-License-Identifier: MIT
    
    """Calculate every public model-fit and effective-parameter metric."""
    
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
        compute_adjusted_r_squared,
        compute_aic,
        compute_aicc,
        compute_bic,
        compute_diagnostics,
        compute_edf,
        compute_effective_parameters,
        compute_enp,
        compute_local_r_squared,
        compute_r_squared,
        compute_trace_statistics,
    )
    
    y = np.array([1.0, 2.0, 2.8, 4.2, 5.0])
    yhat = np.array([1.1, 1.9, 3.0, 4.0, 4.9])
    hat = np.eye(5) * 0.4
    weights = np.vstack([np.linspace(1.0, 0.2, 5)] * 5)
    trace = compute_trace_statistics(hat)
    print("r2=", compute_r_squared(y, yhat))
    print("adjusted_r2=", compute_adjusted_r_squared(y, yhat, edf=3.0))
    print("aic=", compute_aic(y, yhat, n_params=2.0))
    print("aicc=", compute_aicc(y, yhat, n_params=2.0))
    print("bic=", compute_bic(y, yhat, trace_S=2.0))
    print("local_r2=", compute_local_r_squared(y, yhat, weights))
    print("effective_parameters=", compute_effective_parameters(hat))
    print("trace_statistics=", trace)
    print("edf=", compute_edf(5, trace["trace_S"], trace["trace_StS"]))
    print("enp=", compute_enp(trace["trace_S"], trace["trace_StS"]))
    print(
        "diagnostics=",
        compute_diagnostics(y, yhat, hat, n_features=1, compute_gwr_stats=True),
    )
    ```
