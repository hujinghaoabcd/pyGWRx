# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Base classes for spatial estimators and GWR-family models.

The hierarchy in this module standardizes input validation, fitted-state management, prediction checks, and result export across pyGWRx estimators.

Author:
    Jinghao Hu
"""

from __future__ import annotations

__author__ = "Jinghao Hu"
__license__ = "MIT"

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd

KernelLike = Union[str, Callable[[np.ndarray, float], np.ndarray]]
BandwidthLike = Union[float, int, str, None]
ArrayLike = Union[np.ndarray, pd.DataFrame, pd.Series]


class BaseSpatialEstimator(ABC):
    """Root class for all spatial estimators.

    The base class uses a single NumPy/SciPy numerical implementation.
    PyGWRx uses NumPy/SciPy internally; future acceleration should be implemented
    behind the numerical routines rather than exposed as an estimator parameter.
    """

    def __init__(
        self,
        *,
        distance_metric: str = "euclidean",
        random_state: Optional[int] = None,
        verbose: bool = False,
    ) -> None:
        self._validate_spatial_estimator_parameters(
            distance_metric=distance_metric,
            random_state=random_state,
            verbose=verbose,
        )
        self.distance_metric = distance_metric.strip().lower()
        self.random_state = random_state
        self.verbose = bool(verbose)

        self._is_fitted = False
        self.n_samples_: Optional[int] = None
        self.n_features_in_: Optional[int] = None
        self.feature_names_in_: Optional[np.ndarray] = None

    @staticmethod
    def _validate_spatial_estimator_parameters(
        *,
        distance_metric: str,
        random_state: Optional[int],
        verbose: bool,
    ) -> None:
        if not isinstance(distance_metric, str) or not distance_metric.strip():
            raise ValueError("distance_metric must be a non-empty string.")

        if random_state is not None:
            if isinstance(random_state, (bool, np.bool_)) or not isinstance(
                random_state, (int, np.integer)
            ):
                raise TypeError("random_state must be an integer or None.")

        if not isinstance(verbose, (bool, np.bool_)):
            raise TypeError("verbose must be boolean.")

    @property
    def is_fitted_(self) -> bool:
        """Whether the estimator completed a successful fit."""
        return bool(self._is_fitted)

    def _mark_fitted(self) -> None:
        self._is_fitted = True

    def _mark_unfitted(self) -> None:
        self._is_fitted = False

    def _check_is_fitted(self) -> None:
        if not self._is_fitted:
            raise ValueError(
                f"{self.__class__.__name__} is not fitted yet. Call 'fit' first."
            )

    def _validate_feature_matrix(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        *,
        reset: bool,
    ) -> np.ndarray:
        from pygwrx.core.utils import validate_data

        if (
            isinstance(X, pd.DataFrame)
            and not reset
            and self.feature_names_in_ is not None
        ):
            received = np.asarray(X.columns, dtype=object)
            expected = np.asarray(self.feature_names_in_, dtype=object)
            if received.shape != expected.shape or not np.array_equal(
                received, expected
            ):
                raise ValueError(
                    "Prediction DataFrame columns must match the training columns in "
                    f"the same order. Expected {expected.tolist()}, got {received.tolist()}."
                )

        n_rows = X.shape[0] if hasattr(X, "shape") and len(X.shape) > 0 else len(X)
        X_arr, _ = validate_data(X, np.zeros(n_rows, dtype=float))

        if reset:
            self.n_samples_ = int(X_arr.shape[0])
            self.n_features_in_ = int(X_arr.shape[1])
            self.feature_names_in_ = (
                np.asarray(X.columns, dtype=object)
                if isinstance(X, pd.DataFrame)
                else None
            )
        elif self.n_features_in_ is not None and X_arr.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X_arr.shape[1]} features, but the fitted estimator expects "
                f"{self.n_features_in_}."
            )
        return X_arr

    def _validate_spatial_inputs(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        coords: Union[np.ndarray, pd.DataFrame],
        *,
        reset: bool,
    ) -> Tuple[np.ndarray, np.ndarray]:
        from pygwrx.core.utils import validate_coords

        X_arr = self._validate_feature_matrix(X, reset=reset)
        coords_arr = validate_coords(coords)
        if X_arr.shape[0] != coords_arr.shape[0]:
            raise ValueError(
                "X and coords must contain the same number of samples; "
                f"got {X_arr.shape[0]} and {coords_arr.shape[0]}."
            )
        return X_arr, coords_arr


class BaseSpatialRegressor(BaseSpatialEstimator):
    """Base class for geographically weighted spatial regressors.

    This class combines the common regression contract with kernel,
    bandwidth, local-parameter, prediction, fitted-state, and result-
    export behavior used throughout the pyGWRx regression family.
    """

    def __init__(
        self,
        kernel: KernelLike = "gaussian",
        bandwidth: BandwidthLike = "cv",
        bandwidth_method: str = "cv",
        fit_intercept: bool = True,
        distance_metric: str = "euclidean",
        adaptive: bool = False,
        bandwidth_range: Optional[Tuple[float, float]] = None,
        optimization_method: str = "golden_section",
        random_state: Optional[int] = None,
        verbose: bool = False,
    ) -> None:
        super().__init__(
            distance_metric=distance_metric,
            random_state=random_state,
            verbose=verbose,
        )
        if not isinstance(fit_intercept, (bool, np.bool_)):
            raise TypeError("fit_intercept must be boolean.")
        self.fit_intercept = bool(fit_intercept)
        self._reset_regression_state()

        self._validate_gwr_parameters(
            kernel=kernel,
            bandwidth=bandwidth,
            bandwidth_method=bandwidth_method,
            adaptive=adaptive,
            bandwidth_range=bandwidth_range,
            optimization_method=optimization_method,
        )
        self.kernel = kernel
        self.bandwidth = bandwidth
        self.bandwidth_method = bandwidth_method.strip().lower()
        self.adaptive = bool(adaptive)
        self.bandwidth_range = bandwidth_range
        self.optimization_method = optimization_method
        self._reset_gwr_state()

    def _reset_regression_state(self) -> None:
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: Optional[np.ndarray] = None
        self.fitted_values_: Optional[np.ndarray] = None
        self.residuals_: Optional[np.ndarray] = None
        self.diagnostics_: Optional[Dict[str, Any]] = None
        self.local_r2_: Optional[np.ndarray] = None
        self.X_train_: Optional[np.ndarray] = None
        self.y_train_: Optional[np.ndarray] = None
        self.coords_train_: Optional[np.ndarray] = None
        self.times_train_: Optional[np.ndarray] = None
        self.context_train_: Optional[np.ndarray] = None

    def _reset_gwr_state(self) -> None:
        self.bandwidth_: Optional[Union[float, int]] = None
        self.kernel_func_: Optional[Callable] = None
        self.hat_matrix_: Optional[np.ndarray] = None

    @staticmethod
    def _validate_gwr_parameters(
        *,
        kernel: KernelLike,
        bandwidth: BandwidthLike,
        bandwidth_method: str,
        adaptive: bool,
        bandwidth_range: Optional[Tuple[float, float]],
        optimization_method: str,
    ) -> None:
        if not isinstance(kernel, str) and not callable(kernel):
            raise TypeError("kernel must be a string name or callable.")
        if isinstance(kernel, str) and not kernel.strip():
            raise ValueError("kernel name cannot be empty.")

        if bandwidth is not None and not isinstance(
            bandwidth, (str, int, float, np.integer, np.floating)
        ):
            raise TypeError("bandwidth must be numeric, a selection token, or None.")
        if isinstance(bandwidth, (bool, np.bool_)):
            raise TypeError("bandwidth must not be boolean.")
        if isinstance(bandwidth, str):
            token = bandwidth.strip().lower()
            if token not in {"cv", "aic", "aicc", "bic", "adaptive"}:
                raise ValueError(
                    "bandwidth string must be one of 'cv', 'aic', 'aicc', 'bic', 'adaptive'."
                )
        elif bandwidth is not None:
            value = float(bandwidth)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(
                    "numeric bandwidth must be finite and greater than zero."
                )
            if adaptive and not value.is_integer():
                raise ValueError(
                    "adaptive numeric bandwidth must be an integer neighbour count."
                )

        if not isinstance(bandwidth_method, str) or not bandwidth_method.strip():
            raise ValueError("bandwidth_method must be a non-empty string.")
        if not isinstance(adaptive, (bool, np.bool_)):
            raise TypeError("adaptive must be boolean.")

        if bandwidth_range is not None:
            if (
                not isinstance(bandwidth_range, (tuple, list))
                or len(bandwidth_range) != 2
            ):
                raise TypeError(
                    "bandwidth_range must be a two-element tuple/list or None."
                )
            lower, upper = float(bandwidth_range[0]), float(bandwidth_range[1])
            if not np.isfinite(lower) or not np.isfinite(upper):
                raise ValueError("bandwidth_range values must be finite.")
            if lower <= 0 or upper <= 0 or lower > upper:
                raise ValueError("bandwidth_range must satisfy 0 < lower <= upper.")
            if adaptive and (not lower.is_integer() or not upper.is_integer()):
                raise ValueError("adaptive bandwidth_range values must be integers.")

        valid_optimizers = {"grid", "golden_section", "brent"}
        if optimization_method not in valid_optimizers:
            raise ValueError(
                f"optimization_method must be one of {sorted(valid_optimizers)}."
            )

    @abstractmethod
    def fit(self, X: ArrayLike, y: ArrayLike, coords: ArrayLike, **kwargs: Any):
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: ArrayLike, coords: ArrayLike, **kwargs: Any) -> np.ndarray:
        raise NotImplementedError

    def _validate_inputs(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        coords: Union[np.ndarray, pd.DataFrame],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        from pygwrx.core.utils import validate_coords, validate_data

        feature_names = (
            np.asarray(X.columns, dtype=object) if isinstance(X, pd.DataFrame) else None
        )
        X_arr, y_arr = validate_data(X, y)
        coords_arr = validate_coords(coords)
        if X_arr.shape[0] != coords_arr.shape[0]:
            raise ValueError(
                "X, y, and coords must contain the same number of samples; "
                f"got {X_arr.shape[0]}, {y_arr.shape[0]}, and {coords_arr.shape[0]}."
            )
        self.n_samples_ = int(X_arr.shape[0])
        self.n_features_in_ = int(X_arr.shape[1])
        self.feature_names_in_ = feature_names
        return X_arr, y_arr, coords_arr

    def _validate_prediction_inputs(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        coords: Union[np.ndarray, pd.DataFrame],
    ) -> Tuple[np.ndarray, np.ndarray]:
        return self._validate_spatial_inputs(X, coords, reset=False)

    def _store_training_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        coords: np.ndarray,
        *,
        times: Optional[np.ndarray] = None,
        context: Optional[np.ndarray] = None,
        copy: bool = True,
    ) -> None:
        self.X_train_ = np.array(X, dtype=float, copy=copy)
        self.y_train_ = np.array(y, dtype=float, copy=copy).reshape(-1)
        self.coords_train_ = np.array(coords, dtype=float, copy=copy)
        self.times_train_ = (
            None
            if times is None
            else np.array(times, dtype=float, copy=copy).reshape(-1)
        )
        self.context_train_ = (
            None if context is None else np.array(context, dtype=float, copy=copy)
        )
        self.n_samples_ = int(self.X_train_.shape[0])
        self.n_features_in_ = int(self.X_train_.shape[1])

    def score(
        self, X: ArrayLike, y: ArrayLike, coords: ArrayLike, **kwargs: Any
    ) -> float:
        from pygwrx.core.metrics import compute_r_squared

        return float(compute_r_squared(y, self.predict(X, coords, **kwargs)))

    def get_diagnostics(self) -> Dict[str, Any]:
        self._check_is_fitted()
        return {} if self.diagnostics_ is None else dict(self.diagnostics_)

    def to_frame(self) -> pd.DataFrame:
        """Return training-location coefficients and diagnostics as a DataFrame."""
        self._check_is_fitted()
        if self.X_train_ is None:
            raise RuntimeError("Training data are unavailable.")

        n = self.X_train_.shape[0]
        output: Dict[str, Any] = {}
        if self.coords_train_ is not None:
            for j in range(self.coords_train_.shape[1]):
                output[f"coord_{j}"] = self.coords_train_[:, j]

        if self.intercept_ is not None:
            values = np.asarray(self.intercept_).reshape(-1)
            if values.size == n:
                output["intercept"] = values

        if self.coef_ is not None:
            coef = np.asarray(self.coef_)
            if coef.ndim == 1:
                coef = coef.reshape(-1, 1)
            names = (
                [str(name) for name in self.feature_names_in_]
                if self.feature_names_in_ is not None
                and len(self.feature_names_in_) == coef.shape[1]
                else [f"x{j}" for j in range(coef.shape[1])]
            )
            for j, name in enumerate(names):
                output[f"coef_{name}"] = coef[:, j]

        for name, values in (
            ("fitted", self.fitted_values_),
            ("residual", self.residuals_),
            ("local_r2", self.local_r2_),
        ):
            if values is not None:
                array = np.asarray(values).reshape(-1)
                if array.size == n:
                    output[name] = array
        return pd.DataFrame(output)

    def to_geodataframe(self, crs: Optional[Union[str, int]] = None):
        """Return training-location results as a point GeoDataFrame."""
        from pygwrx.io import to_geodataframe

        frame = self.to_frame()
        if self.coords_train_ is None:
            raise RuntimeError("Training coordinates are unavailable.")
        data_columns = [
            column for column in frame.columns if not column.startswith("coord_")
        ]
        return to_geodataframe(
            frame[data_columns].to_numpy(dtype=float),
            None,
            self.coords_train_,
            feature_names=data_columns,
            crs=crs,
        )

    def _predict_basic(self, X: ArrayLike, coords: ArrayLike) -> np.ndarray:
        params = self._compute_local_parameters(coords)
        X_arr, _ = self._validate_prediction_inputs(X, coords)
        return np.einsum("ij,ij->i", X_arr, params["coef"]) + params["intercept"]

    def _compute_local_parameters(self, coords: ArrayLike) -> Dict[str, np.ndarray]:
        from pygwrx.core.kernels import get_kernel_function
        from pygwrx.core.solver import local_regression
        from pygwrx.core.utils import add_intercept, validate_coords

        self._check_is_fitted()
        if self.X_train_ is None or self.y_train_ is None or self.coords_train_ is None:
            raise RuntimeError(
                "Stored training data are required for local prediction."
            )
        if self.bandwidth_ is None:
            raise RuntimeError("The fitted model does not define bandwidth_.")

        coords_arr = validate_coords(coords)
        X_design = add_intercept(self.X_train_) if self.fit_intercept else self.X_train_
        kernel_func = self.kernel_func_ or get_kernel_function(self.kernel)
        local_coefs = local_regression(
            X_design,
            self.y_train_,
            self.coords_train_,
            coords_arr,
            kernel_func,
            self.bandwidth_,
            distance_metric=self.distance_metric,
            adaptive=self.adaptive,
        )
        if self.fit_intercept:
            intercept = local_coefs[:, 0]
            coef = local_coefs[:, 1:]
        else:
            intercept = np.zeros(coords_arr.shape[0], dtype=float)
            coef = local_coefs
        return {"intercept": intercept, "coef": coef, "coords": coords_arr}


# Backward-compatible public alias retained for the 0.1.x series.
BaseGWR = BaseSpatialRegressor


class SpatiotemporalMixin:
    times_train_: Optional[np.ndarray] = None
    spatial_bandwidth_: Optional[Union[float, int]] = None
    temporal_bandwidth_: Optional[Union[float, int]] = None

    def _validate_spatiotemporal_inputs(
        self, X: ArrayLike, coords: ArrayLike, times: ArrayLike, *, reset: bool
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        X_arr, coords_arr = self._validate_spatial_inputs(X, coords, reset=reset)
        times_arr = np.asarray(times, dtype=float).reshape(-1)
        if times_arr.shape[0] != X_arr.shape[0]:
            raise ValueError(
                "X, coords, and times must contain the same number of samples."
            )
        if not np.all(np.isfinite(times_arr)):
            raise ValueError("times contains NaN or infinite values.")
        return X_arr, coords_arr, times_arr


class MultiscaleMixin:
    bandwidths_: Optional[np.ndarray] = None
    bandwidth_history_: Optional[Any] = None
    convergence_history_: Optional[Any] = None

    def _reset_multiscale_state(self) -> None:
        self.bandwidths_ = None
        self.bandwidth_history_ = None
        self.convergence_history_ = None


class BaseSpatiotemporalRegressor(SpatiotemporalMixin, BaseSpatialRegressor):
    """Base for spatiotemporal GWR-family regressors."""


class BaseMultiscaleRegressor(MultiscaleMixin, BaseSpatialRegressor):
    """Base for one-bandwidth-per-coefficient regressors."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._reset_multiscale_state()


class BaseSpatialClassifier(BaseSpatialEstimator):
    """Base class for spatial classifiers such as GWDA."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.classes_: Optional[np.ndarray] = None
        self.diagnostics_: Optional[Dict[str, Any]] = None

    @abstractmethod
    def fit(self, X: ArrayLike, y: ArrayLike, coords: ArrayLike, **kwargs: Any):
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: ArrayLike, coords: ArrayLike, **kwargs: Any) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def predict_proba(
        self, X: ArrayLike, coords: ArrayLike, **kwargs: Any
    ) -> np.ndarray:
        raise NotImplementedError

    def score(
        self, X: ArrayLike, y: ArrayLike, coords: ArrayLike, **kwargs: Any
    ) -> float:
        truth = np.asarray(y).reshape(-1)
        predicted = np.asarray(self.predict(X, coords, **kwargs)).reshape(-1)
        if truth.shape != predicted.shape:
            raise ValueError("y and predictions must have the same shape.")
        return float(np.mean(truth == predicted))


class BaseSpatialTransformer(BaseSpatialEstimator):
    @abstractmethod
    def fit(self, X: ArrayLike, coords: ArrayLike, **kwargs: Any):
        raise NotImplementedError

    @abstractmethod
    def transform(self, X: ArrayLike, coords: ArrayLike, **kwargs: Any) -> np.ndarray:
        raise NotImplementedError

    def fit_transform(
        self, X: ArrayLike, coords: ArrayLike, **kwargs: Any
    ) -> np.ndarray:
        return self.fit(X, coords, **kwargs).transform(X, coords, **kwargs)


class BaseSpatialStatistics(BaseSpatialEstimator):
    @abstractmethod
    def fit(self, X: ArrayLike, coords: ArrayLike, **kwargs: Any):
        raise NotImplementedError

    @abstractmethod
    def to_frame(self) -> pd.DataFrame:
        raise NotImplementedError


class BaseSpatialInference(BaseSpatialEstimator):
    @abstractmethod
    def fit(self, X: ArrayLike, y: ArrayLike, coords: ArrayLike, **kwargs: Any):
        raise NotImplementedError

    @abstractmethod
    def summary(self) -> str:
        raise NotImplementedError


__all__ = [
    "BaseSpatialEstimator",
    "BaseSpatialRegressor",
    "BaseGWR",
    "SpatiotemporalMixin",
    "MultiscaleMixin",
    "BaseSpatiotemporalRegressor",
    "BaseMultiscaleRegressor",
    "BaseSpatialClassifier",
    "BaseSpatialTransformer",
    "BaseSpatialStatistics",
    "BaseSpatialInference",
]
