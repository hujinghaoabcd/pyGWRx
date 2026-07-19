# SGTWR

This page documents **2** public symbols. Each entry includes its purpose, import path, full API docstring, and the maintained example that exercises it.

[Conceptual guide](../../models/index.md){ .md-button }

## `SGTWR`

Spatiotemporal geographically weighted regression with similarity.

| Property | Value |
|---|---|
| Type | `class` |
| Import | `from pygwrx.models import SGTWR` |
| Signature | `SGTWR(spatial_bandwidth: 'SelectionValue' = 'aicc', *, temporal_bandwidth: 'SelectionValue' = 'aicc', adaptive: 'bool' = True, alpha: 'SelectionValue' = 'aicc', similarity_vars: 'Optional[Sequence[Union[int, str]]]' = None, standardize_similarity: 'bool' = True, spatial_bandwidth_candidates: 'Optional[Sequence[Number]]' = None, temporal_bandwidth_candidates: 'Optional[Sequence[Number]]' = None, alpha_candidates: 'Optional[Sequence[Number]]' = None, causal: 'bool' = False, time_unit: 'str' = 'auto', fit_intercept: 'bool' = True, distance_metric: 'str' = 'euclidean', sigma2_v1: 'bool' = True, ridge: 'float' = 0.0, store_weights: 'bool' = True, verbose: 'bool' = False) -> 'None'` |
| Maintained example | [`examples/models/16_sgtwr.py`](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/models/16_sgtwr.py) |

::: pygwrx.models.SGTWR


## `SGTWRPredictionResult`

Detailed predictions from a fitted SGTWR model.

| Property | Value |
|---|---|
| Type | `class` |
| Import | `from pygwrx.models import SGTWRPredictionResult` |
| Signature | `SGTWRPredictionResult(predictions: 'np.ndarray', coef: 'np.ndarray', intercept: 'np.ndarray', coords: 'np.ndarray', times: 'np.ndarray', feature_names: 'Tuple[str, ...]') -> None` |
| Maintained example | [`examples/models/16_sgtwr.py`](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/models/16_sgtwr.py) |

::: pygwrx.models.SGTWRPredictionResult


## Runnable examples used on this page

??? example "`examples/models/16_sgtwr.py`"

    ```python
    # SPDX-FileCopyrightText: 2026 Jinghao Hu
    # SPDX-License-Identifier: MIT
    
    """Fit similarity and geographically-temporally weighted regression."""
    
    # Allow this script to run directly from any working directory.
    import sys
    from pathlib import Path
    
    _PROJECT_ROOT = Path(__file__).resolve().parents[2]
    _EXAMPLES_ROOT = _PROJECT_ROOT / "examples"
    _SRC_ROOT = _PROJECT_ROOT / "src"
    for _path in (_SRC_ROOT, _EXAMPLES_ROOT):
        if str(_path) not in sys.path:
            sys.path.insert(0, str(_path))
    
    from _common import print_model_result, temporal_regression
    
    from pygwrx import SGTWR, SGTWRPredictionResult
    
    X, y, coords, times = temporal_regression(n=48, p=3)
    model = SGTWR(
        spatial_bandwidth=24,
        temporal_bandwidth=2.0,
        adaptive=True,
        alpha=0.5,
        similarity_vars=["x1", "x2"],
        store_weights=True,
    ).fit(X, y, coords, times)
    print_model_result(model)
    print("combined_weights_shape=", model.combined_weights_.shape)
    result = model.predict_result(X.iloc[:3], coords.iloc[:3], times[:3])
    assert isinstance(result, SGTWRPredictionResult)
    print(result.to_frame())
    ```
