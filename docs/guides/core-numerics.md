# Core numerical functions

The core layer contains the reusable numerical building blocks behind the model classes. Most users should start with a model class; use these functions directly when developing a new estimator, reproducing a formula, or testing one numerical component in isolation.

## Functional map

| Area | What it provides | Typical use |
|---|---|---|
| Kernels | Gaussian, bisquare, exponential, tricube, boxcar | Convert distance to local weight |
| Distances | Euclidean, Manhattan, Chebyshev, Minkowski, haversine | Define the neighbourhood geometry |
| Bandwidth selection | CV, AIC, BIC selectors | Select distance or neighbour-count scale |
| Optimization | Golden-section and Brent search | Optimize scalar criteria |
| Solvers | Weighted least squares, local regression, hat matrix | Build custom local estimators |
| Metrics | R², information criteria, ENP/EDF, trace statistics | Diagnose fitted smoothers |
| Validation/base | Input contracts and estimator base classes | Extend pyGWRx safely |

!!! warning
    Core functions expose lower-level contracts. Check shapes, intercept handling, rank, and fixed/adaptive bandwidth semantics explicitly.

## Kernels

The maintained example below exercises the public symbols in this group. [Open source](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/core/01_kernels.py).

```python
# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Evaluate every public kernel and resolve kernels by name or callable."""

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

## Distances and validation

The maintained example below exercises the public symbols in this group. [Open source](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/core/02_distances_and_validation.py).

```python
# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Use all public distance, validation, caching, and chunk helpers."""

import numpy as np
import pandas as pd
from pygwrx.core import (
    DistanceCache,
    add_intercept,
    chebyshev_distance,
    chunked_computation,
    compute_distance_matrix,
    euclidean_distance,
    haversine_distance,
    manhattan_distance,
    minkowski_distance,
    validate_coords,
    validate_data,
)

a = np.array([[0.0, 0.0], [1.0, 2.0]])
b = np.array([[2.0, 1.0], [3.0, 4.0]])
print("euclidean=", euclidean_distance(a, b))
print("manhattan=", manhattan_distance(a, b))
print("chebyshev=", chebyshev_distance(a, b))
print("minkowski_p3=", minkowski_distance(a, b, p=3.0))
print(
    "haversine_km=",
    haversine_distance(np.array([[116.4, 39.9]]), np.array([[121.5, 31.2]])),
)
print("matrix=", compute_distance_matrix(a, metric="euclidean"))

X, y = validate_data(pd.DataFrame({"x": [1, 2]}), pd.Series([3, 4]))
coords = validate_coords(pd.DataFrame(a, columns=["x", "y"]))
print("validated_shapes=", X.shape, y.shape, coords.shape)
print("with_intercept=", add_intercept(X))
print("chunks=", list(chunked_computation(10, chunk_size=4)))
print("cache_memory=", DistanceCache.estimate_memory(100, 50))
print("cache_strategy=", DistanceCache.get_strategy(100, 50, task="gwr"))
print("should_cache=", DistanceCache.should_cache(100, 50))
DistanceCache.print_recommendation(100, 50)
```

## GeoPandas coordinates

The maintained example below exercises the public symbols in this group. [Open source](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/core/03_geopandas_coordinates.py).

```python
# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Extract coordinates from a GeoDataFrame using the base installation."""

import geopandas as gpd
from shapely.geometry import Point
from pygwrx.core import extract_geopandas_coords

gdf = gpd.GeoDataFrame(
    {"name": ["a", "b"]},
    geometry=[Point(0.0, 1.0), Point(2.0, 3.0)],
    crs="EPSG:3857",
)
print(extract_geopandas_coords(gdf))
```

## Local solvers

The standard solver is unpenalized weighted least squares. Coefficients are estimated from `sqrt(W) @ X` using SVD-backed least squares, exact zero weights are preserved, and no hidden ridge penalty is added. The low-level `ridge=` argument remains an explicit opt-in.

The maintained example below exercises the public symbols in this group. [Open source](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/core/04_solver.py).

```python
# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Run all public local-regression solver utilities."""

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

## Metrics

The maintained example below exercises the public symbols in this group. [Open source](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/core/05_metrics.py).

```python
# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Calculate every public model-fit and effective-parameter metric."""

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

## Optimization

The maintained example below exercises the public symbols in this group. [Open source](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/core/06_optimization.py).

```python
# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Use both public scalar optimizers and the OptimizationResult container."""

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

## Bandwidth selectors

The maintained example below exercises the public symbols in this group. [Open source](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/core/07_bandwidth_selectors.py).

```python
# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Select bandwidths with CV, AIC/AICc, and BIC selectors."""

import numpy as np
from pygwrx.core import (
    AICSelector,
    BICSelector,
    BandwidthSelector,
    CrossValidationSelector,
    gaussian_kernel,
    get_bandwidth_selector,
)
from _common import spatial_regression

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

## Base classes

The maintained example below exercises the public symbols in this group. [Open source](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/core/08_base_classes.py).

```python
# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Implement minimal concrete estimators from every public base class/mixin."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pygwrx.core import (
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

print("root_estimator_params=", {"contract": "task-specific spatial estimator"})
classifier = MajorityClassifier().fit(X, np.array([0, 1, 1, 1]), coords)
print("classifier=", classifier.predict(X, coords), classifier.predict_proba(X, coords))
print("transformer=", IdentityTransformer().fit(X, coords).transform(X, coords))
print("statistics=", ColumnStatistics().fit(X, coords).to_frame())
print("inference=", BasicInference().fit(X, y, coords).summary())
```

## API

See the [Core API reference](../api/core/index.md), where every symbol includes its full docstring, signature, and mapped runnable example.

## Regression base hierarchy

`BaseSpatialRegressor` is the single shared base for the geographically
weighted regression family. It owns the common regression state, kernel
and bandwidth configuration, local parameter computation, prediction
validation, and result export behavior. The deprecated compatibility alias
has been removed from the current public API.

