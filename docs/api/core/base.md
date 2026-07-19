# Base classes

This page documents **11** public symbols. Each entry includes its purpose, import path, full API docstring, and the maintained example that exercises it.

[Conceptual guide](../../guides/core-numerics.md){ .md-button }

## `BaseSpatialEstimator`

Root class for all spatial estimators.

| Property | Value |
|---|---|
| Type | `class` |
| Import | `from pygwrx.core import BaseSpatialEstimator` |
| Signature | `BaseSpatialEstimator(*, distance_metric: 'str' = 'euclidean', random_state: 'Optional[int]' = None, verbose: 'bool' = False) -> 'None'` |
| Maintained example | [`examples/core/08_base_classes.py`](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/core/08_base_classes.py) |

::: pygwrx.core.BaseSpatialEstimator


## `BaseSpatialRegressor`

Base class for geographically weighted spatial regressors.

| Property | Value |
|---|---|
| Type | `class` |
| Import | `from pygwrx.core import BaseSpatialRegressor` |
| Signature | `BaseSpatialRegressor(kernel: 'KernelLike' = 'gaussian', bandwidth: 'BandwidthLike' = 'cv', bandwidth_method: 'str' = 'cv', fit_intercept: 'bool' = True, distance_metric: 'str' = 'euclidean', adaptive: 'bool' = False, bandwidth_range: 'Optional[Tuple[float, float]]' = None, optimization_method: 'str' = 'golden_section', random_state: 'Optional[int]' = None, verbose: 'bool' = False) -> 'None'` |
| Maintained example | [`examples/core/08_base_classes.py`](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/core/08_base_classes.py) |

::: pygwrx.core.BaseSpatialRegressor


## `BaseGWR`

Base class for geographically weighted spatial regressors.

| Property | Value |
|---|---|
| Type | `class` |
| Import | `from pygwrx.core import BaseGWR` |
| Signature | `BaseGWR(kernel: 'KernelLike' = 'gaussian', bandwidth: 'BandwidthLike' = 'cv', bandwidth_method: 'str' = 'cv', fit_intercept: 'bool' = True, distance_metric: 'str' = 'euclidean', adaptive: 'bool' = False, bandwidth_range: 'Optional[Tuple[float, float]]' = None, optimization_method: 'str' = 'golden_section', random_state: 'Optional[int]' = None, verbose: 'bool' = False) -> 'None'` |
| Maintained example | [`examples/core/08_base_classes.py`](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/core/08_base_classes.py) |

::: pygwrx.core.BaseGWR


## `SpatiotemporalMixin`

No summary is available.

| Property | Value |
|---|---|
| Type | `class` |
| Import | `from pygwrx.core import SpatiotemporalMixin` |
| Signature | `SpatiotemporalMixin()` |
| Maintained example | [`examples/core/08_base_classes.py`](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/core/08_base_classes.py) |

::: pygwrx.core.SpatiotemporalMixin


## `MultiscaleMixin`

No summary is available.

| Property | Value |
|---|---|
| Type | `class` |
| Import | `from pygwrx.core import MultiscaleMixin` |
| Signature | `MultiscaleMixin()` |
| Maintained example | [`examples/core/08_base_classes.py`](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/core/08_base_classes.py) |

::: pygwrx.core.MultiscaleMixin


## `BaseSpatiotemporalRegressor`

Base for spatiotemporal GWR-family regressors.

| Property | Value |
|---|---|
| Type | `class` |
| Import | `from pygwrx.core import BaseSpatiotemporalRegressor` |
| Signature | `BaseSpatiotemporalRegressor(kernel: 'KernelLike' = 'gaussian', bandwidth: 'BandwidthLike' = 'cv', bandwidth_method: 'str' = 'cv', fit_intercept: 'bool' = True, distance_metric: 'str' = 'euclidean', adaptive: 'bool' = False, bandwidth_range: 'Optional[Tuple[float, float]]' = None, optimization_method: 'str' = 'golden_section', random_state: 'Optional[int]' = None, verbose: 'bool' = False) -> 'None'` |
| Maintained example | [`examples/core/08_base_classes.py`](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/core/08_base_classes.py) |

::: pygwrx.core.BaseSpatiotemporalRegressor


## `BaseMultiscaleRegressor`

Base for one-bandwidth-per-coefficient regressors.

| Property | Value |
|---|---|
| Type | `class` |
| Import | `from pygwrx.core import BaseMultiscaleRegressor` |
| Signature | `BaseMultiscaleRegressor(*args: 'Any', **kwargs: 'Any') -> 'None'` |
| Maintained example | [`examples/core/08_base_classes.py`](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/core/08_base_classes.py) |

::: pygwrx.core.BaseMultiscaleRegressor


## `BaseSpatialClassifier`

Base class for spatial classifiers such as GWDA.

| Property | Value |
|---|---|
| Type | `class` |
| Import | `from pygwrx.core import BaseSpatialClassifier` |
| Signature | `BaseSpatialClassifier(**kwargs: 'Any') -> 'None'` |
| Maintained example | [`examples/core/08_base_classes.py`](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/core/08_base_classes.py) |

::: pygwrx.core.BaseSpatialClassifier


## `BaseSpatialTransformer`

Root class for all spatial estimators.

| Property | Value |
|---|---|
| Type | `class` |
| Import | `from pygwrx.core import BaseSpatialTransformer` |
| Signature | `BaseSpatialTransformer(*, distance_metric: 'str' = 'euclidean', random_state: 'Optional[int]' = None, verbose: 'bool' = False) -> 'None'` |
| Maintained example | [`examples/core/08_base_classes.py`](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/core/08_base_classes.py) |

::: pygwrx.core.BaseSpatialTransformer


## `BaseSpatialStatistics`

Root class for all spatial estimators.

| Property | Value |
|---|---|
| Type | `class` |
| Import | `from pygwrx.core import BaseSpatialStatistics` |
| Signature | `BaseSpatialStatistics(*, distance_metric: 'str' = 'euclidean', random_state: 'Optional[int]' = None, verbose: 'bool' = False) -> 'None'` |
| Maintained example | [`examples/core/08_base_classes.py`](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/core/08_base_classes.py) |

::: pygwrx.core.BaseSpatialStatistics


## `BaseSpatialInference`

Root class for all spatial estimators.

| Property | Value |
|---|---|
| Type | `class` |
| Import | `from pygwrx.core import BaseSpatialInference` |
| Signature | `BaseSpatialInference(*, distance_metric: 'str' = 'euclidean', random_state: 'Optional[int]' = None, verbose: 'bool' = False) -> 'None'` |
| Maintained example | [`examples/core/08_base_classes.py`](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/core/08_base_classes.py) |

::: pygwrx.core.BaseSpatialInference


## Runnable examples used on this page

??? example "`examples/core/08_base_classes.py`"

    ```python
    # SPDX-FileCopyrightText: 2026 Jinghao Hu
    # SPDX-License-Identifier: MIT
    
    """Implement minimal concrete estimators from every public base class/mixin."""
    
    from __future__ import annotations
    
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
    import pandas as pd
    
    from pygwrx.core import (
        BaseGWR,
        BaseMultiscaleRegressor,
        BaseSpatialClassifier,
        BaseSpatialEstimator,
        BaseSpatialInference,
        BaseSpatialRegressor,
        BaseSpatialStatistics,
        BaseSpatialTransformer,
        BaseSpatiotemporalRegressor,
        MultiscaleMixin,
        SpatiotemporalMixin,
    )
    
    
    class MeanRegressor(BaseSpatialRegressor):
        """Minimal concrete spatial regressor for base-contract demonstration."""
    
        def fit(self, X, y, coords, **kwargs):
            Xa, ya, ca = self._validate_inputs(X, y, coords)
            self._store_training_data(Xa, ya, ca)
            self.mean_ = float(np.mean(ya))
            self._mark_fitted()
            return self
    
        def predict(self, X, coords, **kwargs):
            Xa, _ = self._validate_prediction_inputs(X, coords)
            return np.full(Xa.shape[0], self.mean_)
    
    
    class TinyGWR(BaseSpatialRegressor):
        """Minimal concrete GWR-style regressor."""
    
        def fit(self, X, y, coords, **kwargs):
            Xa, ya, ca = self._validate_inputs(X, y, coords)
            self._store_training_data(Xa, ya, ca)
            self.bandwidth_ = 1.0
            self.coef_ = np.zeros_like(Xa)
            self.intercept_ = np.full(len(ya), ya.mean())
            self.fitted_values_ = self.intercept_.copy()
            self.residuals_ = ya - self.fitted_values_
            self._mark_fitted()
            return self
    
        def predict(self, X, coords, **kwargs):
            Xa, _ = self._validate_prediction_inputs(X, coords)
            return np.full(len(Xa), self.y_train_.mean())
    
    
    class TimeRegressor(SpatiotemporalMixin, MeanRegressor):
        """Concrete demonstration of the spatiotemporal mixin."""
    
    
    class MultiRegressor(MultiscaleMixin, MeanRegressor):
        """Concrete demonstration of the multiscale mixin."""
    
    
    class ConcreteSTR(BaseSpatiotemporalRegressor, MeanRegressor):
        """Concrete base spatiotemporal regressor."""
    
        fit = MeanRegressor.fit
        predict = MeanRegressor.predict
    
    
    class ConcreteMSR(BaseMultiscaleRegressor, MeanRegressor):
        """Concrete base multiscale regressor."""
    
        fit = MeanRegressor.fit
        predict = MeanRegressor.predict
    
    
    class MajorityClassifier(BaseSpatialClassifier):
        """Minimal majority-class spatial classifier."""
    
        def fit(self, X, y, coords, **kwargs):
            self._validate_spatial_inputs(X, coords, reset=True)
            values, counts = np.unique(y, return_counts=True)
            self.classes_ = values
            self.majority_ = values[np.argmax(counts)]
            self._mark_fitted()
            return self
    
        def predict(self, X, coords, **kwargs):
            Xa, _ = self._validate_spatial_inputs(X, coords, reset=False)
            return np.repeat(self.majority_, len(Xa))
    
        def predict_proba(self, X, coords, **kwargs):
            prediction = self.predict(X, coords)
            return np.column_stack(
                [prediction == class_value for class_value in self.classes_]
            ).astype(float)
    
    
    class IdentityTransformer(BaseSpatialTransformer):
        """Minimal identity spatial transformer."""
    
        def fit(self, X, coords, **kwargs):
            self._validate_spatial_inputs(X, coords, reset=True)
            self._mark_fitted()
            return self
    
        def transform(self, X, coords, **kwargs):
            Xa, _ = self._validate_spatial_inputs(X, coords, reset=False)
            return Xa
    
    
    class ColumnStatistics(BaseSpatialStatistics):
        """Minimal column-mean spatial statistics estimator."""
    
        def fit(self, X, coords, **kwargs):
            Xa, ca = self._validate_spatial_inputs(X, coords, reset=True)
            self.means_ = Xa.mean(axis=0)
            self.coords_train_ = ca
            self._mark_fitted()
            return self
    
        def to_frame(self):
            self._check_is_fitted()
            return pd.DataFrame({"mean": self.means_})
    
    
    class BasicInference(BaseSpatialInference):
        """Minimal inference result container."""
    
        def fit(self, X, y, coords, **kwargs):
            self.n_samples_ = len(y)
            self._mark_fitted()
            return self
    
        def summary(self):
            self._check_is_fitted()
            return f"n_samples={self.n_samples_}"
    
    
    # BaseGWR remains an identity alias for backward compatibility.
    assert BaseGWR is BaseSpatialRegressor
    
    
    X = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]})
    y = np.array([1.0, 2.0, 2.0, 3.0])
    coords = np.column_stack((np.arange(4), np.zeros(4)))
    
    for model in (
        MeanRegressor(),
        TinyGWR(bandwidth=1.0),
        TimeRegressor(),
        MultiRegressor(),
        ConcreteSTR(),
        ConcreteMSR(),
    ):
        model.fit(X, y, coords)
        print(type(model).__name__, model.predict(X.iloc[:2], coords[:2]))
    
    print("root_estimator=", BaseSpatialEstimator.__name__)
    classifier = MajorityClassifier().fit(X, np.array([0, 1, 1, 1]), coords)
    print("classifier=", classifier.predict(X, coords), classifier.predict_proba(X, coords))
    print("transformer=", IdentityTransformer().fit(X, coords).transform(X, coords))
    print("statistics=", ColumnStatistics().fit(X, coords).to_frame())
    print("inference=", BasicInference().fit(X, y, coords).summary())
    ```
