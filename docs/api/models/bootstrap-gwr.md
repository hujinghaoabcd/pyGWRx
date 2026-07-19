# BootstrapGWR

This page documents **1** public symbols. Each entry includes its purpose, import path, full API docstring, and the maintained example that exercises it.

[Conceptual guide](../../models/index.md){ .md-button }

## `BootstrapGWR`

Test GWR coefficient non-stationarity by parametric bootstrap.

| Property | Value |
|---|---|
| Type | `class` |
| Import | `from pygwrx.models import BootstrapGWR` |
| Signature | `BootstrapGWR(bandwidth: 'Union[float, int, str, None]' = 'aicc', adaptive: 'bool' = False, kernel: 'str' = 'bisquare', bandwidth_method: 'str' = 'aicc', bandwidth_range: 'Optional[Tuple[float, float]]' = None, optimization_method: 'str' = 'golden_section', fit_intercept: 'bool' = True, distance_metric: 'str' = 'euclidean', n_bootstrap: 'int' = 99, reselect_bandwidth: 'bool' = True, pvalue_method: 'str' = 'plus_one', localized_tail: 'str' = 'two-sided', store_local_bootstrap: 'bool' = False, random_state: 'Optional[Union[int, np.random.Generator]]' = None, verbose: 'bool' = False) -> 'None'` |
| Maintained example | [`examples/models/14_bootstrap_gwr.py`](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/models/14_bootstrap_gwr.py) |

::: pygwrx.models.BootstrapGWR


## Runnable examples used on this page

??? example "`examples/models/14_bootstrap_gwr.py`"

    ```python
    # SPDX-FileCopyrightText: 2026 Jinghao Hu
    # SPDX-License-Identifier: MIT
    
    """Run coefficient-wise bootstrap tests for spatial variability."""
    
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
    
    from pygwrx import BootstrapGWR
    
    X, y, coords = spatial_regression(n=42, p=2)
    model = BootstrapGWR(
        bandwidth=22,
        adaptive=True,
        n_bootstrap=9,
        reselect_bandwidth=False,
        store_local_bootstrap=True,
        random_state=0,
    ).fit(X, y, coords)
    print_model_result(model)
    print("modified_pvalues=", model.modified_p_values_)
    print("localized_p_values_shape=", model.localized_p_values_.shape)
    ```
