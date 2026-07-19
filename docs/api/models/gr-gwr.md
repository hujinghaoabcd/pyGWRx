# GRGWR

This page documents **2** public symbols. Each entry includes its purpose, import path, full API docstring, and the maintained example that exercises it.

[Conceptual guide](../../models/index.md){ .md-button }

## `GRGWR`

Geo-Regime Geographically Weighted Regression.

| Property | Value |
|---|---|
| Type | `class` |
| Import | `from pygwrx.models import GRGWR` |
| Signature | `GRGWR(n_regimes: 'int' = 3, bandwidth: 'BandwidthLike' = 20, kernel: 'str' = 'bisquare', lambda_boundary: 'float' = 1.0, max_iter: 'int' = 10, tol: 'float' = 0.0001, spatial_constraint_weight: 'float' = 0.5, fit_intercept: 'bool' = True, verbose: 'bool' = False, *, n_neighbors: 'int' = 8, min_regime_size: 'Optional[int]' = None, enforce_connectivity: 'bool' = True, random_state: 'Optional[int]' = 42) -> 'None'` |
| Maintained example | [`examples/models/19_gr_gwr.py`](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/models/19_gr_gwr.py) |

::: pygwrx.models.GRGWR


## `GRGWRPredictionResult`

Detailed GR-GWR predictions at evaluation locations.

| Property | Value |
|---|---|
| Type | `class` |
| Import | `from pygwrx.models import GRGWRPredictionResult` |
| Signature | `GRGWRPredictionResult(predictions: 'np.ndarray', coefficients: 'np.ndarray', intercepts: 'np.ndarray', regimes: 'np.ndarray', coords: 'np.ndarray', feature_names: 'Tuple[str, ...]') -> None` |
| Maintained example | [`examples/models/19_gr_gwr.py`](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/models/19_gr_gwr.py) |

::: pygwrx.models.GRGWRPredictionResult


## Runnable examples used on this page

??? example "`examples/models/19_gr_gwr.py`"

    ```python
    # SPDX-FileCopyrightText: 2026 Jinghao Hu
    # SPDX-License-Identifier: MIT
    
    """Fit geo-regime GWR and inspect connected spatial regimes."""
    
    # Allow this script to run directly from any working directory.
    import sys
    from pathlib import Path
    
    _PROJECT_ROOT = Path(__file__).resolve().parents[2]
    _EXAMPLES_ROOT = _PROJECT_ROOT / "examples"
    _SRC_ROOT = _PROJECT_ROOT / "src"
    for _path in (_SRC_ROOT, _EXAMPLES_ROOT):
        if str(_path) not in sys.path:
            sys.path.insert(0, str(_path))
    
    from _common import print_model_result, regime_regression
    
    from pygwrx import GRGWR, GRGWRPredictionResult
    
    X, y, coords, truth = regime_regression(n=56)
    model = GRGWR(n_regimes=2, bandwidth=18, max_iter=2, random_state=0).fit(X, y, coords)
    print_model_result(model)
    print("regime_sizes=", model.regime_sizes_)
    print(
        "truth_agreement_or_label_swap=",
        max((model.regimes_ == truth).mean(), (model.regimes_ != truth).mean()),
    )
    result = model.predict_result(X.iloc[:3], coords.iloc[:3])
    assert isinstance(result, GRGWRPredictionResult)
    print(result.to_frame())
    ```
