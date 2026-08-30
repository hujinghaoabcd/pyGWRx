# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Standard geographically and temporally weighted regression.

This module implements Gaussian GTWR with the generalized spatiotemporal
metric used by GWmodel, an optional history-only causal extension, and a
Euclidean space-time metric compatible with the public Python ``gtwr`` package.

Author:
    Jinghao Hu
"""

from __future__ import annotations

__author__ = "Jinghao Hu"
__license__ = "MIT"

import warnings
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from pygwrx.core.base import BaseSpatiotemporalRegressor
from pygwrx.core.kernels import get_kernel_function
from pygwrx.core.metrics import compute_diagnostics
from pygwrx.core.optimization import (
    BrentSearch,
    GoldenSectionSearch,
    OptimizationResult,
)
from pygwrx.core.solver import adaptive_bandwidth_weights, weighted_least_squares
from pygwrx.core.time import (
    _TIME_UNIT_ALIASES,
    TimeAxis,
    auto_time_unit,
    looks_datetime_like,
    normalize_prediction_times,
    normalize_training_times,
)
from pygwrx.core.utils import add_intercept, compute_distance_matrix

_FUTURE_DISTANCE = 1.0e50


@dataclass(frozen=True)
class GTWRPredictionResult:
    """Rich prediction result returned by :meth:`GTWR.predict_result`."""

    predictions: np.ndarray
    coef: np.ndarray
    intercept: np.ndarray
    coords: np.ndarray
    times: np.ndarray
    feature_names: Tuple[str, ...]
    coef_standard_errors: Optional[np.ndarray] = None
    intercept_standard_errors: Optional[np.ndarray] = None
    coef_t_values: Optional[np.ndarray] = None
    intercept_t_values: Optional[np.ndarray] = None

    def to_frame(self) -> pd.DataFrame:
        """Return prediction results as a pandas DataFrame."""
        data: Dict[str, np.ndarray] = {
            "coord_0": self.coords[:, 0],
            "coord_1": self.coords[:, 1],
            "time": self.times,
            "prediction": self.predictions,
            "intercept": self.intercept,
        }
        if self.intercept_standard_errors is not None:
            data["intercept_se"] = self.intercept_standard_errors
        if self.intercept_t_values is not None:
            data["intercept_t"] = self.intercept_t_values
        for index, name in enumerate(self.feature_names):
            data[f"coef_{name}"] = self.coef[:, index]
            if self.coef_standard_errors is not None:
                data[f"se_{name}"] = self.coef_standard_errors[:, index]
            if self.coef_t_values is not None:
                data[f"t_{name}"] = self.coef_t_values[:, index]
        return pd.DataFrame(data)

    def to_geodataframe(self, crs: Optional[Union[str, int]] = None):
        """Return prediction results as a point GeoDataFrame."""
        from pygwrx.io import to_geodataframe

        frame = self.to_frame()
        columns = [
            column for column in frame.columns if not column.startswith("coord_")
        ]
        return to_geodataframe(
            frame[columns].to_numpy(dtype=float),
            None,
            self.coords,
            feature_names=columns,
            crs=crs,
        )


@dataclass
class _GTWRLocalFit:
    params: np.ndarray
    fitted_values: np.ndarray
    influence: np.ndarray
    trace_S: float
    trace_StS: float
    covariance_factors: Optional[np.ndarray]
    hat_matrix: Optional[np.ndarray]


class GTWR(BaseSpatiotemporalRegressor):
    r"""Geographically and temporally weighted regression.

    The default ``distance_combination="gwmodel"`` follows the generalized
    spatiotemporal distance implemented by ``GWmodel::st.dist``:

    .. math::

        d_{st} = \lambda d_s + (1-\lambda)d_t
        + 2\sqrt{\lambda(1-\lambda)d_s d_t}\cos(\xi).

    GWmodel uses absolute temporal differences, so the standard default is
    ``causal=False``. With ``causal=True``, observations later than the regression
    time receive a very large temporal distance as an optional history-only
    extension for leakage-safe forecasting.

    ``distance_combination="euclidean"`` instead uses
    :math:`\sqrt{d_s^2 + \tau d_t^2}` and is provided for transparent comparison
    with the public Python ``gtwr`` package.

    Args:
        kernel: Kernel name or callable accepting ``(distances, bandwidth)``.
        bandwidth: Numeric bandwidth or ``"cv"``/``"aicc"`` for automatic search.
        bandwidth_method: Selection criterion used when ``bandwidth=None``.
        adaptive: Interpret bandwidth as an integer nearest-neighbour count.
        bandwidth_range: Optional lower and upper search bounds.
        lambda_st: GWmodel spatial-temporal balance in ``[0, 1]`` or ``"auto"``.
        lambda_range: Search interval used only when ``lambda_st="auto"``.
        lambda_grid_size: Number of deterministic lambda candidates.
        ksi: GWmodel interaction angle in radians, constrained to ``[0, pi]``.
        distance_combination: ``"gwmodel"`` or ``"euclidean"``.
        tau: Non-negative temporal scale used by the Euclidean combination.
        causal: Whether future observations should be temporally remote. The
            default ``False`` matches ``GWmodel::st.dist``.
        time_unit: Unit used to convert datetime-like times. ``"auto"`` chooses
            a stable unit from the training span. Numeric times are not rescaled.
        optimization_method: ``"grid"``, ``"golden_section"``, or ``"brent"``.
        search_grid_size: Number of fixed-bandwidth candidates for grid search.
        search_tol: Tolerance for continuous bandwidth optimization.
        search_max_iter: Maximum continuous search iterations.
        fit_intercept: Whether to include a local intercept.
        distance_metric: Spatial distance metric.
        sigma2_v1: Residual variance convention. ``False`` matches GWmodel's
            ``RSS / (n - 2 trace(S) + trace(S'S))`` default diagnostic.
        verbose: Whether to print selection and fitting progress.

    Attributes:
        bandwidth_: Selected fixed distance or adaptive neighbour count.
        lambda_st_: Fitted GWmodel balance parameter.
        tau_: Fitted Euclidean temporal scale.
        times_train_: Numeric training times in ``time_unit_``.
        time_unit_: Resolved datetime unit or ``"numeric"``.
        spatiotemporal_distance_matrix_: Training target-to-observation distances.
        coef_: Local slopes with shape ``(n_samples, n_features)``.
        intercept_: Local intercepts with shape ``(n_samples,)``.
        fitted_values_: Fitted responses at calibration locations.
        diagnostics_: Gaussian GWR-style diagnostics based on smoother traces.
    """

    def __init__(
        self,
        kernel: Union[str, Callable[[np.ndarray, float], np.ndarray]] = "bisquare",
        bandwidth: Union[float, int, str, None] = "cv",
        bandwidth_method: str = "cv",
        adaptive: bool = False,
        bandwidth_range: Optional[Tuple[float, float]] = None,
        lambda_st: Union[float, str] = 0.05,
        lambda_range: Tuple[float, float] = (0.0, 1.0),
        lambda_grid_size: int = 11,
        ksi: float = 0.0,
        distance_combination: str = "gwmodel",
        tau: float = 1.0,
        causal: bool = False,
        time_unit: str = "auto",
        optimization_method: str = "golden_section",
        search_grid_size: int = 25,
        search_tol: float = 1e-5,
        search_max_iter: int = 100,
        fit_intercept: bool = True,
        distance_metric: str = "euclidean",
        sigma2_v1: bool = False,
        verbose: bool = False,
    ) -> None:
        if isinstance(bandwidth, str) and bandwidth.strip().lower() == "adaptive":
            raise ValueError(
                "GTWR uses adaptive=True for nearest-neighbour bandwidths; "
                "bandwidth must be numeric, None, 'cv', or 'aicc'."
            )
        super().__init__(
            kernel=kernel,
            bandwidth=bandwidth,
            bandwidth_method=bandwidth_method,
            adaptive=adaptive,
            bandwidth_range=bandwidth_range,
            optimization_method=optimization_method,
            fit_intercept=fit_intercept,
            distance_metric=distance_metric,
            verbose=verbose,
        )
        self.lambda_st = lambda_st
        self.lambda_range = lambda_range
        self.lambda_grid_size = lambda_grid_size
        self.ksi = ksi
        self.distance_combination = distance_combination
        self.tau = tau
        self.causal = causal
        self.time_unit = time_unit
        self.search_grid_size = search_grid_size
        self.search_tol = search_tol
        self.search_max_iter = search_max_iter
        self.sigma2_v1 = sigma2_v1
        self._validate_gtwr_parameters()
        self._reset_gtwr_state()

    def _reset_gtwr_state(self) -> None:
        self.lambda_st_: Optional[float] = None
        self.tau_: Optional[float] = None
        self.ksi_: Optional[float] = None
        self.time_unit_: Optional[str] = None
        self.time_origin_: Optional[pd.Timestamp] = None
        self.time_input_kind_: Optional[str] = None
        self._time_axis: Optional[TimeAxis] = None
        self.spatiotemporal_distance_matrix_: Optional[np.ndarray] = None
        self.spatial_distance_matrix_: Optional[np.ndarray] = None
        self.temporal_distance_matrix_: Optional[np.ndarray] = None
        self.bandwidth_selection_result_: Optional[OptimizationResult] = None
        self.bandwidth_score_: Optional[float] = None
        self.lambda_selection_history_: List[Dict[str, float]] = []
        self.sigma2_: Optional[float] = None
        self.influence_: Optional[np.ndarray] = None
        self.standardized_residuals_: Optional[np.ndarray] = None
        self.cooks_distance_: Optional[np.ndarray] = None
        self.parameter_covariance_diagonal_: Optional[np.ndarray] = None
        self.parameter_standard_errors_: Optional[np.ndarray] = None
        self.parameter_t_values_: Optional[np.ndarray] = None
        self.intercept_se_: Optional[np.ndarray] = None
        self.coef_se_: Optional[np.ndarray] = None
        self.intercept_t_: Optional[np.ndarray] = None
        self.coef_t_: Optional[np.ndarray] = None
        self.inference_enabled_: bool = False
        self.S_matrix_: Optional[np.ndarray] = None

    def _reset_fit_state(self) -> None:
        self._mark_unfitted()
        self._reset_regression_state()
        self._reset_gwr_state()
        self._reset_gtwr_state()

    def _validate_gtwr_parameters(self) -> None:
        if isinstance(self.lambda_st, str):
            if self.lambda_st.strip().lower() != "auto":
                raise ValueError("lambda_st must be a number in [0, 1] or 'auto'.")
        elif isinstance(self.lambda_st, (bool, np.bool_)):
            raise TypeError("lambda_st must be numeric or 'auto', not bool.")
        else:
            value = float(self.lambda_st)
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError("lambda_st must lie in [0, 1].")

        if (
            not isinstance(self.lambda_range, (tuple, list))
            or len(self.lambda_range) != 2
        ):
            raise TypeError("lambda_range must be a two-element tuple/list.")
        lambda_lower, lambda_upper = map(float, self.lambda_range)
        if not (0.0 <= lambda_lower <= lambda_upper <= 1.0):
            raise ValueError("lambda_range must satisfy 0 <= lower <= upper <= 1.")
        if isinstance(self.lambda_grid_size, (bool, np.bool_)) or not isinstance(
            self.lambda_grid_size, (int, np.integer)
        ):
            raise TypeError("lambda_grid_size must be an integer.")
        if int(self.lambda_grid_size) < 2:
            raise ValueError("lambda_grid_size must be at least 2.")

        self.ksi = float(self.ksi)
        if not np.isfinite(self.ksi) or not 0.0 <= self.ksi <= np.pi:
            raise ValueError("ksi must lie in [0, pi].")
        if not isinstance(self.distance_combination, str):
            raise TypeError("distance_combination must be a string.")
        self.distance_combination = self.distance_combination.strip().lower()
        if self.distance_combination not in {"gwmodel", "euclidean"}:
            raise ValueError("distance_combination must be 'gwmodel' or 'euclidean'.")

        if isinstance(self.tau, (bool, np.bool_)):
            raise TypeError("tau must be a non-negative real scalar.")
        self.tau = float(self.tau)
        if not np.isfinite(self.tau) or self.tau < 0.0:
            raise ValueError("tau must be finite and non-negative.")
        if self.distance_combination == "euclidean" and isinstance(self.lambda_st, str):
            raise ValueError(
                "lambda_st='auto' is only available for the GWmodel distance."
            )

        if not isinstance(self.causal, (bool, np.bool_)):
            raise TypeError("causal must be boolean.")
        self.causal = bool(self.causal)
        if not isinstance(self.time_unit, str) or not self.time_unit.strip():
            raise ValueError("time_unit must be a non-empty string.")
        unit = self.time_unit.strip().lower()
        if unit != "auto" and unit not in _TIME_UNIT_ALIASES:
            raise ValueError(
                "time_unit must be 'auto', seconds, minutes, hours, days, or weeks."
            )
        self.time_unit = unit

        for name, value, minimum in (
            ("search_grid_size", self.search_grid_size, 3),
            ("search_max_iter", self.search_max_iter, 1),
        ):
            if isinstance(value, (bool, np.bool_)) or not isinstance(
                value, (int, np.integer)
            ):
                raise TypeError(f"{name} must be an integer.")
            if int(value) < minimum:
                raise ValueError(f"{name} must be at least {minimum}.")
        self.search_grid_size = int(self.search_grid_size)
        self.search_max_iter = int(self.search_max_iter)
        if isinstance(self.search_tol, (bool, np.bool_)):
            raise TypeError("search_tol must be a positive real scalar.")
        self.search_tol = float(self.search_tol)
        if not np.isfinite(self.search_tol) or self.search_tol <= 0.0:
            raise ValueError("search_tol must be finite and greater than zero.")
        if not isinstance(self.sigma2_v1, (bool, np.bool_)):
            raise TypeError("sigma2_v1 must be boolean.")
        self.sigma2_v1 = bool(self.sigma2_v1)

    @staticmethod
    def _looks_datetime_like(times: object) -> bool:
        return looks_datetime_like(times)

    @staticmethod
    def _auto_time_unit(span_seconds: float) -> str:
        return auto_time_unit(span_seconds)

    def _convert_times(self, times: object, *, reset: bool) -> np.ndarray:
        if reset:
            axis = normalize_training_times(times, time_unit=self.time_unit)
            self._time_axis = axis
            self.time_origin_ = axis.origin
            self.time_unit_ = axis.unit
            self.time_input_kind_ = "datetime" if axis.datetime_like else "numeric"
            return axis.values.copy()

        axis = self._time_axis
        if axis is None:
            if self.time_unit_ is None or self.time_input_kind_ is None:
                raise ValueError(
                    "Prediction times must be datetime-like because the model was fitted "
                    "with datetime-like times."
                )
            axis = TimeAxis(
                values=np.empty(0, dtype=float),
                unit=self.time_unit_,
                origin=self.time_origin_,
                datetime_like=self.time_input_kind_ == "datetime",
            )
        return normalize_prediction_times(times, axis=axis)

    def _validate_fit_inputs(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        coords: Union[np.ndarray, pd.DataFrame],
        times: object,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        X_arr, coords_arr = self._validate_spatial_inputs(X, coords, reset=True)
        y_arr = np.asarray(y, dtype=float)
        if y_arr.ndim == 2 and 1 in y_arr.shape:
            y_arr = y_arr.reshape(-1)
        if y_arr.ndim != 1:
            raise ValueError("y must be one-dimensional.")
        if y_arr.shape[0] != X_arr.shape[0]:
            raise ValueError("X and y must contain the same number of samples.")
        if not np.all(np.isfinite(y_arr)):
            raise ValueError("y contains NaN or infinite values.")
        times_arr = self._convert_times(times, reset=True)
        if times_arr.shape[0] != X_arr.shape[0]:
            raise ValueError("X, y, coords, and times must have the same length.")
        return X_arr, y_arr, coords_arr, times_arr

    def _validate_prediction_inputs(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        coords: Union[np.ndarray, pd.DataFrame],
        times: object,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        self._check_is_fitted()
        X_arr, coords_arr = self._validate_spatial_inputs(X, coords, reset=False)
        times_arr = self._convert_times(times, reset=False)
        if times_arr.shape[0] != X_arr.shape[0]:
            raise ValueError("X, coords, and times must have the same length.")
        return X_arr, coords_arr, times_arr

    def _temporal_distances(
        self,
        target_times: np.ndarray,
        source_times: np.ndarray,
    ) -> np.ndarray:
        delta = target_times[:, None] - source_times[None, :]
        if self.causal:
            return np.where(delta >= 0.0, delta, _FUTURE_DISTANCE)
        return np.abs(delta)

    def _combine_distances(
        self,
        spatial: np.ndarray,
        temporal: np.ndarray,
        *,
        lambda_value: float,
    ) -> np.ndarray:
        if self.distance_combination == "euclidean":
            # Calculate only on ordinary values and restore the finite future sentinel
            # afterwards to avoid overflow when squaring 1e50.
            future = temporal >= _FUTURE_DISTANCE * 0.5
            safe_temporal = np.where(future, 0.0, temporal)
            combined = np.sqrt(spatial**2 + self.tau * safe_temporal**2)
            if self.causal and self.tau > 0.0:
                combined[future] = _FUTURE_DISTANCE
            return combined

        cross = (
            2.0
            * np.sqrt(
                np.maximum(
                    lambda_value * (1.0 - lambda_value) * spatial * temporal,
                    0.0,
                )
            )
            * np.cos(self.ksi)
        )
        combined = lambda_value * spatial + (1.0 - lambda_value) * temporal + cross
        # The quadratic-form expression is non-negative analytically; clipping
        # removes only floating-point roundoff near zero when ksi is pi.
        return np.maximum(combined, 0.0)

    def _distance_matrices(
        self,
        source_coords: np.ndarray,
        source_times: np.ndarray,
        target_coords: np.ndarray,
        target_times: np.ndarray,
        *,
        lambda_value: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        spatial = np.asarray(
            compute_distance_matrix(
                target_coords,
                source_coords,
                metric=self.distance_metric,
            ),
            dtype=float,
        )
        temporal = self._temporal_distances(target_times, source_times)
        combined = self._combine_distances(
            spatial,
            temporal,
            lambda_value=lambda_value,
        )
        if not np.all(np.isfinite(combined)) or np.any(combined < 0.0):
            raise ValueError(
                "The spatiotemporal distance calculation produced invalid values."
            )
        return spatial, temporal, combined

    def _weights(
        self,
        distances: np.ndarray,
        bandwidth: Union[int, float],
    ) -> np.ndarray:
        if self.kernel_func_ is None:
            raise RuntimeError("The kernel function is unavailable.")
        if self.adaptive:
            k = int(bandwidth)
            local_bandwidth = adaptive_bandwidth_weights(distances, k)
        else:
            local_bandwidth = float(bandwidth)
        weights = np.asarray(self.kernel_func_(distances, local_bandwidth), dtype=float)
        if weights.shape != distances.shape:
            raise ValueError("The kernel returned an unexpected weight shape.")
        if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
            raise ValueError("The kernel returned invalid weights.")
        if not np.any(weights > 0.0):
            raise ValueError("The local spatiotemporal kernel has no positive weights.")
        return weights

    def _minimum_available_observations(
        self,
        distance_matrix: np.ndarray,
        *,
        lambda_value: float,
    ) -> int:
        temporal_active = (
            self.distance_combination == "gwmodel" and lambda_value < 1.0
        ) or (self.distance_combination == "euclidean" and self.tau > 0.0)
        if not self.causal or not temporal_active:
            return distance_matrix.shape[1]
        return int(np.min(np.sum(distance_matrix < _FUTURE_DISTANCE * 0.5, axis=1)))

    def _validate_numeric_bandwidth(
        self,
        bandwidth: Union[int, float],
        X_design: np.ndarray,
        distance_matrix: np.ndarray,
        *,
        lambda_value: float,
    ) -> Union[int, float]:
        value = float(bandwidth)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError("numeric bandwidth must be finite and greater than zero.")
        if self.adaptive:
            if not value.is_integer():
                raise ValueError(
                    "adaptive bandwidth must be an integer neighbour count."
                )
            k = int(value)
            minimum = X_design.shape[1] + 1
            maximum = self._minimum_available_observations(
                distance_matrix, lambda_value=lambda_value
            )
            if k < minimum:
                raise ValueError(
                    f"adaptive bandwidth must be at least {minimum} for "
                    f"{X_design.shape[1]} design columns."
                )
            if k > maximum:
                raise ValueError(
                    "adaptive bandwidth exceeds the minimum number of temporally "
                    f"available observations across targets ({maximum})."
                )
            return k
        return value

    def _criterion_name(self) -> str:
        method = (
            self.bandwidth.strip().lower()
            if isinstance(self.bandwidth, str)
            else self.bandwidth_method.strip().lower()
        )
        if method == "aic":
            method = "aicc"
        if method not in {"cv", "aicc"}:
            raise ValueError(
                "GTWR automatic bandwidth selection supports only 'cv' or 'aicc'."
            )
        return method

    def _candidate_bounds(
        self,
        X_design: np.ndarray,
        distance_matrix: np.ndarray,
        *,
        lambda_value: float,
    ) -> Tuple[float, float]:
        if self.bandwidth_range is not None:
            lower, upper = map(float, self.bandwidth_range)
        elif self.adaptive:
            lower = float(X_design.shape[1] + 1)
            upper = float(
                self._minimum_available_observations(
                    distance_matrix, lambda_value=lambda_value
                )
            )
        else:
            ordinary = distance_matrix[
                (distance_matrix > np.finfo(float).eps)
                & (distance_matrix < _FUTURE_DISTANCE * 0.5)
            ]
            if ordinary.size == 0:
                raise ValueError(
                    "No positive finite spatiotemporal distances are available."
                )
            lower = float(np.min(ordinary))
            upper = float(np.max(ordinary))
        if lower > upper:
            raise ValueError(
                f"No valid bandwidth interval remains; got lower={lower}, upper={upper}."
            )
        return lower, upper

    def _selection_score(
        self,
        bandwidth: Union[int, float],
        X_design: np.ndarray,
        y: np.ndarray,
        distance_matrix: np.ndarray,
        *,
        method: str,
        lambda_value: float,
    ) -> float:
        try:
            bandwidth = self._validate_numeric_bandwidth(
                bandwidth,
                X_design,
                distance_matrix,
                lambda_value=lambda_value,
            )
        except (TypeError, ValueError):
            return np.inf

        n_samples = X_design.shape[0]
        fitted = np.empty(n_samples, dtype=float)
        trace_s = 0.0
        trace_sts = 0.0
        for index, distance_row in enumerate(distance_matrix):
            try:
                weights = self._weights(distance_row, bandwidth)
                if method == "cv":
                    weights = weights.copy()
                    weights[index] = 0.0
                if np.count_nonzero(weights > 0.0) < X_design.shape[1]:
                    return np.inf
                beta, inverse_normal = weighted_least_squares(X_design, y, weights)
                fitted[index] = float(X_design[index] @ beta)
                if method == "aicc":
                    inverse_xtx_xtw = inverse_normal @ (X_design.T * weights)
                    hat_row = X_design[index] @ inverse_xtx_xtw
                    trace_s += float(hat_row[index])
                    trace_sts += float(np.dot(hat_row, hat_row))
            except (np.linalg.LinAlgError, ValueError, FloatingPointError):
                return np.inf

        if method == "cv":
            residuals = y - fitted
            return float(np.dot(residuals, residuals))
        diagnostics = compute_diagnostics(
            y,
            fitted,
            compute_gwr_stats=True,
            trace_S=trace_s,
            trace_StS=trace_sts,
        )
        return float(diagnostics["aicc"])

    def _select_bandwidth(
        self,
        X_design: np.ndarray,
        y: np.ndarray,
        distance_matrix: np.ndarray,
        *,
        lambda_value: float,
    ) -> Tuple[Union[int, float], float, OptimizationResult]:
        method = self._criterion_name()
        lower, upper = self._candidate_bounds(
            X_design,
            distance_matrix,
            lambda_value=lambda_value,
        )

        def objective(value: Union[int, float]) -> float:
            return self._selection_score(
                value,
                X_design,
                y,
                distance_matrix,
                method=method,
                lambda_value=lambda_value,
            )

        if self.optimization_method == "grid":
            if self.adaptive:
                candidates = np.arange(int(np.ceil(lower)), int(np.floor(upper)) + 1)
            else:
                candidates = np.linspace(lower, upper, self.search_grid_size)
            if candidates.size == 0:
                raise ValueError(
                    "The bandwidth search interval contains no candidates."
                )
            scores = np.asarray([objective(candidate) for candidate in candidates])
            best_index = int(np.argmin(scores))
            value = (
                int(candidates[best_index])
                if self.adaptive
                else float(candidates[best_index])
            )
            result = OptimizationResult(
                value=value,
                score=float(scores[best_index]),
                iterations=int(candidates.size),
                converged=bool(np.isfinite(scores[best_index])),
                evaluations=int(candidates.size),
                message="Deterministic grid search completed.",
            )
        else:
            # Brent's continuous optimizer cannot search integer neighbour counts.
            # Preserve the public option while routing adaptive searches through the
            # discrete golden-section implementation used elsewhere in pyGWRx.
            if self.adaptive or self.optimization_method == "golden_section":
                optimizer = GoldenSectionSearch(
                    tol=self.search_tol,
                    max_iter=self.search_max_iter,
                    verbose=self.verbose,
                )
                result = optimizer.minimize(
                    objective,
                    lower,
                    upper,
                    adaptive=self.adaptive,
                )
            else:
                optimizer = BrentSearch(
                    tol=self.search_tol,
                    max_iter=self.search_max_iter,
                    verbose=self.verbose,
                )
                result = optimizer.minimize(objective, lower, upper)
            value = int(result.value) if self.adaptive else float(result.value)

        if not result.converged or not np.isfinite(result.score):
            raise RuntimeError(
                "GTWR bandwidth selection did not find a finite solution."
            )
        return value, float(result.score), result

    def _resolve_lambda_and_bandwidth(
        self,
        X_design: np.ndarray,
        y: np.ndarray,
        coords: np.ndarray,
        times: np.ndarray,
    ) -> Tuple[float, Union[int, float], np.ndarray, np.ndarray, np.ndarray]:
        lambda_candidates = (
            np.linspace(
                float(self.lambda_range[0]),
                float(self.lambda_range[1]),
                int(self.lambda_grid_size),
            )
            if isinstance(self.lambda_st, str)
            else np.asarray([float(self.lambda_st)])
        )

        best: Optional[
            Tuple[
                float,
                Union[int, float],
                float,
                np.ndarray,
                np.ndarray,
                np.ndarray,
                Optional[OptimizationResult],
            ]
        ] = None
        self.lambda_selection_history_ = []
        for lambda_value in lambda_candidates:
            spatial, temporal, combined = self._distance_matrices(
                coords,
                times,
                coords,
                times,
                lambda_value=float(lambda_value),
            )
            if isinstance(self.bandwidth, str) or self.bandwidth is None:
                bandwidth, score, result = self._select_bandwidth(
                    X_design,
                    y,
                    combined,
                    lambda_value=float(lambda_value),
                )
            else:
                bandwidth = self._validate_numeric_bandwidth(
                    self.bandwidth,
                    X_design,
                    combined,
                    lambda_value=float(lambda_value),
                )
                result = None
                if len(lambda_candidates) > 1:
                    # A score is required only when lambda itself is selected. A
                    # manually supplied lambda and bandwidth should fit even when a
                    # leave-one-out criterion is undefined for a very small sample.
                    method = self.bandwidth_method.strip().lower()
                    if method == "aic":
                        method = "aicc"
                    if method not in {"cv", "aicc"}:
                        method = "cv"
                    score = self._selection_score(
                        bandwidth,
                        X_design,
                        y,
                        combined,
                        method=method,
                        lambda_value=float(lambda_value),
                    )
                else:
                    score = np.nan

            record = {
                "lambda_st": float(lambda_value),
                "bandwidth": float(bandwidth),
                "score": float(score),
            }
            self.lambda_selection_history_.append(record)
            if self.verbose and len(lambda_candidates) > 1:
                print(
                    f"GTWR lambda={lambda_value:.6g}, bandwidth={bandwidth}, "
                    f"score={score:.6g}"
                )
            if (
                best is None
                or (np.isfinite(score) and not np.isfinite(best[2]))
                or (np.isfinite(score) and score < best[2])
            ):
                best = (
                    float(lambda_value),
                    bandwidth,
                    float(score),
                    spatial,
                    temporal,
                    combined,
                    result,
                )

        if best is None or (len(lambda_candidates) > 1 and not np.isfinite(best[2])):
            raise RuntimeError(
                "GTWR lambda/bandwidth selection found no finite solution."
            )
        (
            lambda_value,
            bandwidth,
            score,
            spatial,
            temporal,
            combined,
            result,
        ) = best
        self.bandwidth_score_ = score
        self.bandwidth_selection_result_ = result
        return lambda_value, bandwidth, spatial, temporal, combined

    def _fit_training_locations(
        self,
        X_design: np.ndarray,
        distance_matrix: np.ndarray,
        *,
        store_hat_matrix: bool,
        compute_inference: bool,
    ) -> _GTWRLocalFit:
        if self.y_train_ is None or self.bandwidth_ is None:
            raise RuntimeError("Training response or fitted bandwidth is unavailable.")
        n_samples, n_parameters = X_design.shape
        params = np.empty((n_samples, n_parameters), dtype=float)
        fitted = np.empty(n_samples, dtype=float)
        influence = np.empty(n_samples, dtype=float)
        covariance_factors = (
            np.empty((n_samples, n_parameters), dtype=float)
            if compute_inference
            else None
        )
        hat_matrix = (
            np.empty((n_samples, n_samples), dtype=float) if store_hat_matrix else None
        )
        trace_sts = 0.0
        for index, distance_row in enumerate(distance_matrix):
            weights = self._weights(distance_row, self.bandwidth_)
            n_positive = int(np.count_nonzero(weights > 0.0))
            if n_positive < n_parameters:
                warnings.warn(
                    f"Location {index}: only {n_positive} positive-weight observations "
                    f"are available for {n_parameters} design columns. The local "
                    "rank-aware WLS solver returns a minimum-norm unpenalized local "
                    "solution; consider increasing the bandwidth.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            beta, inverse_normal = weighted_least_squares(
                X_design,
                self.y_train_,
                weights,
            )
            inverse_xtx_xtw = inverse_normal @ (X_design.T * weights)
            hat_row = X_design[index] @ inverse_xtx_xtw
            params[index] = beta
            fitted[index] = float(X_design[index] @ beta)
            influence[index] = float(hat_row[index])
            trace_sts += float(np.dot(hat_row, hat_row))
            if hat_matrix is not None:
                hat_matrix[index] = hat_row
            if covariance_factors is not None:
                covariance_factors[index] = np.sum(inverse_xtx_xtw**2, axis=1)
        return _GTWRLocalFit(
            params=params,
            fitted_values=fitted,
            influence=influence,
            trace_S=float(np.sum(influence)),
            trace_StS=float(trace_sts),
            covariance_factors=covariance_factors,
            hat_matrix=hat_matrix,
        )

    def _compute_local_r2(self, distance_matrix: np.ndarray) -> np.ndarray:
        if self.y_train_ is None or self.residuals_ is None or self.bandwidth_ is None:
            raise RuntimeError("Fitted results are unavailable.")
        output = np.full(self.y_train_.shape[0], np.nan, dtype=float)
        residual_sq = self.residuals_**2
        for index, distance_row in enumerate(distance_matrix):
            weights = self._weights(distance_row, self.bandwidth_)
            weight_sum = float(np.sum(weights))
            if weight_sum <= np.finfo(float).eps:
                continue
            local_mean = float(np.dot(weights, self.y_train_) / weight_sum)
            local_tss = float(np.dot(weights, (self.y_train_ - local_mean) ** 2))
            local_rss = float(np.dot(weights, residual_sq))
            if local_tss <= np.finfo(float).eps:
                output[index] = 1.0 if local_rss <= np.finfo(float).eps else 0.0
            else:
                output[index] = 1.0 - local_rss / local_tss
        return output

    def _set_inference(
        self,
        covariance_factors: Optional[np.ndarray],
        *,
        trace_S: float,
        trace_StS: float,
    ) -> None:
        if self.residuals_ is None or self.coef_ is None or self.intercept_ is None:
            raise RuntimeError("Fitted regression results are unavailable.")
        rss = float(np.dot(self.residuals_, self.residuals_))
        denominator = (
            self.n_samples_ - trace_S
            if self.sigma2_v1
            else self.n_samples_ - 2.0 * trace_S + trace_StS
        )
        self.sigma2_ = rss / denominator if denominator > 0.0 else np.nan
        leverage_term = 1.0 - np.asarray(self.influence_, dtype=float)
        self.standardized_residuals_ = np.full(self.n_samples_, np.nan, dtype=float)
        valid = leverage_term > np.finfo(float).eps
        if np.isfinite(self.sigma2_) and self.sigma2_ >= 0.0:
            self.standardized_residuals_[valid] = self.residuals_[valid] / np.sqrt(
                self.sigma2_ * leverage_term[valid]
            )
        self.cooks_distance_ = np.full(self.n_samples_, np.nan, dtype=float)
        if trace_S > np.finfo(float).eps:
            self.cooks_distance_[valid] = (
                self.standardized_residuals_[valid] ** 2
                * self.influence_[valid]
                / (trace_S * leverage_term[valid])
            )
        if covariance_factors is None or not np.isfinite(self.sigma2_):
            return
        covariance_diagonal = np.maximum(covariance_factors * self.sigma2_, 0.0)
        standard_errors = np.sqrt(covariance_diagonal)
        full_params = (
            np.column_stack([self.intercept_, self.coef_])
            if self.fit_intercept
            else np.asarray(self.coef_)
        )
        t_values = np.full_like(full_params, np.nan, dtype=float)
        np.divide(
            full_params,
            standard_errors,
            out=t_values,
            where=standard_errors > np.finfo(float).eps,
        )
        self.parameter_covariance_diagonal_ = covariance_diagonal
        self.parameter_standard_errors_ = standard_errors
        self.parameter_t_values_ = t_values
        if self.fit_intercept:
            self.intercept_se_ = standard_errors[:, 0]
            self.coef_se_ = standard_errors[:, 1:]
            self.intercept_t_ = t_values[:, 0]
            self.coef_t_ = t_values[:, 1:]
        else:
            self.intercept_se_ = np.zeros(self.n_samples_, dtype=float)
            self.coef_se_ = standard_errors
            self.intercept_t_ = np.full(self.n_samples_, np.nan, dtype=float)
            self.coef_t_ = t_values

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        coords: Union[np.ndarray, pd.DataFrame],
        times: object,
        *,
        compute_hat_matrix: bool = True,
        compute_local_r2: bool = True,
        compute_inference: bool = True,
        compute_hat_matrix_flag: Optional[bool] = None,
        verbose: Optional[bool] = None,
    ) -> "GTWR":
        """Fit GTWR and return ``self``.

        Smoother traces are calculated even when the full hat matrix is not
        retained, preserving valid AICc, effective-parameter, influence, and
        residual-variance diagnostics.
        """
        if compute_hat_matrix_flag is not None:
            if not isinstance(compute_hat_matrix_flag, (bool, np.bool_)):
                raise TypeError("compute_hat_matrix_flag must be boolean or None.")
            compute_hat_matrix = bool(compute_hat_matrix_flag)
        for name, value in (
            ("compute_hat_matrix", compute_hat_matrix),
            ("compute_local_r2", compute_local_r2),
            ("compute_inference", compute_inference),
        ):
            if not isinstance(value, (bool, np.bool_)):
                raise TypeError(f"{name} must be boolean.")
        if verbose is not None:
            if not isinstance(verbose, (bool, np.bool_)):
                raise TypeError("verbose must be boolean or None.")
            self.verbose = bool(verbose)

        self._validate_gwr_parameters(
            kernel=self.kernel,
            bandwidth=self.bandwidth,
            bandwidth_method=self.bandwidth_method,
            adaptive=self.adaptive,
            bandwidth_range=self.bandwidth_range,
            optimization_method=self.optimization_method,
        )
        self._validate_gtwr_parameters()
        self._reset_fit_state()
        try:
            X_arr, y_arr, coords_arr, times_arr = self._validate_fit_inputs(
                X, y, coords, times
            )
            feature_names = (
                None
                if self.feature_names_in_ is None
                else self.feature_names_in_.copy()
            )
            self.X_train_ = X_arr.copy()
            self.y_train_ = y_arr.copy()
            self.coords_train_ = coords_arr.copy()
            self.times_train_ = times_arr.copy()
            self.feature_names_in_ = feature_names
            self.n_samples_ = int(X_arr.shape[0])
            self.n_features_in_ = int(X_arr.shape[1])
            X_design = add_intercept(X_arr) if self.fit_intercept else X_arr
            self.kernel_func_ = get_kernel_function(self.kernel)

            (
                self.lambda_st_,
                self.bandwidth_,
                self.spatial_distance_matrix_,
                self.temporal_distance_matrix_,
                self.spatiotemporal_distance_matrix_,
            ) = self._resolve_lambda_and_bandwidth(
                X_design,
                y_arr,
                coords_arr,
                times_arr,
            )
            self.tau_ = float(self.tau)
            self.ksi_ = float(self.ksi)

            if self.verbose:
                kind = "adaptive neighbours" if self.adaptive else "fixed distance"
                print(
                    f"Fitting GTWR with bandwidth={self.bandwidth_} ({kind}), "
                    f"distance={self.distance_combination}, lambda={self.lambda_st_:.6g}, "
                    f"tau={self.tau_:.6g}, ksi={self.ksi_:.6g}."
                )

            self.inference_enabled_ = bool(compute_inference)
            local_fit = self._fit_training_locations(
                X_design,
                self.spatiotemporal_distance_matrix_,
                store_hat_matrix=bool(compute_hat_matrix),
                compute_inference=self.inference_enabled_,
            )
            if self.fit_intercept:
                self.intercept_ = local_fit.params[:, 0].copy()
                self.coef_ = local_fit.params[:, 1:].copy()
            else:
                self.intercept_ = np.zeros(self.n_samples_, dtype=float)
                self.coef_ = local_fit.params.copy()
            self.fitted_values_ = local_fit.fitted_values.copy()
            self.residuals_ = y_arr - self.fitted_values_
            self.influence_ = local_fit.influence.copy()
            self.hat_matrix_ = local_fit.hat_matrix
            self.S_matrix_ = self.hat_matrix_
            self.diagnostics_ = compute_diagnostics(
                y_arr,
                self.fitted_values_,
                compute_gwr_stats=True,
                trace_S=local_fit.trace_S,
                trace_StS=local_fit.trace_StS,
            )
            self.local_r2_ = (
                self._compute_local_r2(self.spatiotemporal_distance_matrix_)
                if compute_local_r2
                else None
            )
            self._set_inference(
                local_fit.covariance_factors,
                trace_S=local_fit.trace_S,
                trace_StS=local_fit.trace_StS,
            )
            self._mark_fitted()
            return self
        except Exception:
            self._reset_fit_state()
            raise

    def _prediction_parameters(
        self,
        coords: np.ndarray,
        times: np.ndarray,
    ) -> Dict[str, Optional[np.ndarray]]:
        self._check_is_fitted()
        if (
            self.X_train_ is None
            or self.y_train_ is None
            or self.coords_train_ is None
            or self.times_train_ is None
            or self.bandwidth_ is None
            or self.lambda_st_ is None
        ):
            raise RuntimeError("Stored GTWR training state is incomplete.")
        X_design = add_intercept(self.X_train_) if self.fit_intercept else self.X_train_
        _, _, distances = self._distance_matrices(
            self.coords_train_,
            self.times_train_,
            coords,
            times,
            lambda_value=self.lambda_st_,
        )
        full_params = np.empty((coords.shape[0], X_design.shape[1]), dtype=float)
        covariance_factors = (
            np.empty_like(full_params) if self.inference_enabled_ else None
        )
        for index, distance_row in enumerate(distances):
            weights = self._weights(distance_row, self.bandwidth_)
            beta, inverse_normal = weighted_least_squares(
                X_design,
                self.y_train_,
                weights,
            )
            inverse_xtx_xtw = inverse_normal @ (X_design.T * weights)
            full_params[index] = beta
            if covariance_factors is not None:
                covariance_factors[index] = np.sum(inverse_xtx_xtw**2, axis=1)
        if self.fit_intercept:
            intercept = full_params[:, 0]
            coef = full_params[:, 1:]
        else:
            intercept = np.zeros(coords.shape[0], dtype=float)
            coef = full_params
        standard_errors: Optional[np.ndarray] = None
        t_values: Optional[np.ndarray] = None
        if (
            covariance_factors is not None
            and self.sigma2_ is not None
            and np.isfinite(self.sigma2_)
        ):
            standard_errors = np.sqrt(
                np.maximum(covariance_factors * self.sigma2_, 0.0)
            )
            t_values = np.full_like(full_params, np.nan, dtype=float)
            np.divide(
                full_params,
                standard_errors,
                out=t_values,
                where=standard_errors > np.finfo(float).eps,
            )
        return {
            "coef": coef,
            "intercept": intercept,
            "standard_errors": standard_errors,
            "t_values": t_values,
        }

    def predict(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        coords: Union[np.ndarray, pd.DataFrame],
        times: object,
    ) -> np.ndarray:
        """Predict responses at new space-time locations."""
        return self.predict_result(X, coords, times).predictions

    def predict_result(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        coords: Union[np.ndarray, pd.DataFrame],
        times: object,
    ) -> GTWRPredictionResult:
        """Return predictions, local parameters, and optional inference."""
        X_arr, coords_arr, times_arr = self._validate_prediction_inputs(
            X, coords, times
        )
        params = self._prediction_parameters(coords_arr, times_arr)
        coef = np.asarray(params["coef"], dtype=float)
        intercept = np.asarray(params["intercept"], dtype=float)
        predictions = np.einsum("ij,ij->i", X_arr, coef) + intercept
        names = (
            tuple(str(name) for name in self.feature_names_in_)
            if self.feature_names_in_ is not None
            else tuple(f"x{index}" for index in range(X_arr.shape[1]))
        )
        full_se = params["standard_errors"]
        full_t = params["t_values"]
        if full_se is not None:
            if self.fit_intercept:
                intercept_se = full_se[:, 0]
                coef_se = full_se[:, 1:]
            else:
                intercept_se = np.zeros(coords_arr.shape[0], dtype=float)
                coef_se = full_se
        else:
            intercept_se = None
            coef_se = None
        if full_t is not None:
            if self.fit_intercept:
                intercept_t = full_t[:, 0]
                coef_t = full_t[:, 1:]
            else:
                intercept_t = np.full(coords_arr.shape[0], np.nan, dtype=float)
                coef_t = full_t
        else:
            intercept_t = None
            coef_t = None
        return GTWRPredictionResult(
            predictions=np.asarray(predictions, dtype=float),
            coef=coef,
            intercept=intercept,
            coords=coords_arr.copy(),
            times=times_arr.copy(),
            feature_names=names,
            coef_standard_errors=coef_se,
            intercept_standard_errors=intercept_se,
            coef_t_values=coef_t,
            intercept_t_values=intercept_t,
        )

    def get_local_parameters(
        self,
        coords: Union[np.ndarray, pd.DataFrame],
        times: object,
    ) -> Dict[str, np.ndarray]:
        """Return local parameters at arbitrary space-time locations."""
        dummy = np.zeros((len(coords), self.n_features_in_ or 0), dtype=float)
        _, coords_arr, times_arr = self._validate_prediction_inputs(
            dummy, coords, times
        )
        params = self._prediction_parameters(coords_arr, times_arr)
        return {
            "intercept": np.asarray(params["intercept"], dtype=float).copy(),
            "coef": np.asarray(params["coef"], dtype=float).copy(),
            "coords": coords_arr.copy(),
            "times": times_arr.copy(),
        }

    def to_frame(self) -> pd.DataFrame:
        """Return training-location estimates and diagnostics as a DataFrame."""
        frame = super().to_frame()
        if self.times_train_ is not None:
            frame.insert(2, "time", self.times_train_)
        if self.intercept_se_ is not None:
            frame["intercept_se"] = self.intercept_se_
        if self.intercept_t_ is not None:
            frame["intercept_t"] = self.intercept_t_
        feature_names = (
            [str(name) for name in self.feature_names_in_]
            if self.feature_names_in_ is not None
            else [f"x{index}" for index in range(self.n_features_in_ or 0)]
        )
        if self.coef_se_ is not None:
            for index, name in enumerate(feature_names):
                frame[f"se_{name}"] = self.coef_se_[:, index]
        if self.coef_t_ is not None:
            for index, name in enumerate(feature_names):
                frame[f"t_{name}"] = self.coef_t_[:, index]
        for name, values in (
            ("influence", self.influence_),
            ("standardized_residual", self.standardized_residuals_),
            ("cooks_distance", self.cooks_distance_),
        ):
            if values is not None:
                frame[name] = values
        return frame

    def summary(self) -> str:
        """Return a stable text summary of the fitted GTWR model."""
        self._check_is_fitted()
        if self.diagnostics_ is None or self.X_train_ is None:
            raise RuntimeError("Fitted diagnostics are unavailable.")
        lines = [
            "=" * 78,
            "Geographically and Temporally Weighted Regression (GTWR)",
            "=" * 78,
            f"Samples: {self.n_samples_}",
            f"Predictors: {self.n_features_in_}",
            f"Kernel: {self.kernel}",
            f"Bandwidth: {self.bandwidth_} ({'adaptive neighbours' if self.adaptive else 'fixed distance'})",
            f"Bandwidth criterion score: {self.bandwidth_score_}",
            f"Distance combination: {self.distance_combination}",
            f"lambda_st: {self.lambda_st_}",
            f"tau: {self.tau_}",
            f"ksi: {self.ksi_}",
            f"Causal history-only weighting: {self.causal}",
            f"Time unit: {self.time_unit_}",
            f"R-squared: {self.diagnostics_.get('r2', np.nan):.6f}",
            f"Adjusted R-squared: {self.diagnostics_.get('adj_r2', np.nan):.6f}",
            f"AIC: {self.diagnostics_.get('aic', np.nan):.6f}",
            f"AICc: {self.diagnostics_.get('aicc', np.nan):.6f}",
            f"BIC: {self.diagnostics_.get('bic', np.nan):.6f}",
            f"trace(S): {self.diagnostics_.get('trace_S', np.nan):.6f}",
            f"trace(S'S): {self.diagnostics_.get('trace_StS', np.nan):.6f}",
            f"Residual variance (sigma^2): {self.sigma2_:.6f}",
            "=" * 78,
        ]
        return "\n".join(lines)
