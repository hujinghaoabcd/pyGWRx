# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Spatiotemporal weighted regression proposed by Que et al. (2020).

The implementation follows the public STWR v1.0 formulation: observations are
organized into ordered time stages; past observations receive a temporal effect
based on the response-value variation rate rather than the raw time interval;
and spatial and temporal kernels are combined by a weighted average.

Author:
    Jinghao Hu
"""

from __future__ import annotations

__author__ = "Jinghao Hu"
__license__ = "MIT"

from dataclasses import dataclass
from numbers import Integral, Real
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from pygwrx.core.kernels import get_kernel_function
from pygwrx.core.metrics import compute_diagnostics
from pygwrx.core.solver import adaptive_bandwidth_weights
from pygwrx.core.utils import add_intercept, compute_distance_matrix, validate_coords

Number = Union[int, float]
Bandwidth = Union[int, float, str, None]
SelectionValue = Union[int, float, str, None]


@dataclass(frozen=True)
class STWRPredictionResult:
    """Detailed predictions produced at the latest modeled time stage."""

    predictions: np.ndarray
    coef: np.ndarray
    intercept: np.ndarray
    coords: np.ndarray
    feature_names: Tuple[str, ...]
    reference_y: np.ndarray

    def to_frame(self) -> pd.DataFrame:
        """Return predictions and local parameters as a DataFrame."""
        data: Dict[str, np.ndarray] = {
            "coord_0": self.coords[:, 0],
            "coord_1": self.coords[:, 1],
            "prediction": self.predictions,
            "reference_y": self.reference_y,
            "intercept": self.intercept,
        }
        for index, name in enumerate(self.feature_names):
            data[f"coef_{name}"] = self.coef[:, index]
        return pd.DataFrame(data)


@dataclass
class _STWRLocalFit:
    parameters: np.ndarray
    fitted_values: np.ndarray
    smoother_rows: np.ndarray
    covariance_diagonal: Optional[np.ndarray]


class STWR:
    r"""Spatiotemporal weighted regression.

    STWR uses the response-value variation rate as its time distance. For a
    current regression point :math:`i` and a past observation :math:`j`, the
    public STWR v1.0 code computes

    .. math::

        d^T_{ij} = \frac{\Delta t_{\mathrm{all}}}{\Delta t_q}
        \left|\frac{y_{j,t-q}-y_{i,t}}{y_{j,t-q}}\right|,

    and maps it through :math:`\tanh(d^T_{ij}/2)`. The final weight is a convex
    combination of a spatial kernel and that temporal effect. The spatial
    bandwidth at earlier stages is changed linearly by ``tan(theta)``.

    Args:
        spatial_bandwidth: Current-stage fixed distance or adaptive neighbour
            count. ``"cv"`` selects from ``bandwidth_candidates``.
        adaptive: Interpret spatial bandwidths as neighbour counts at the latest
            stage before conversion to local distance scales.
        kernel: Spatial kernel. The published implementation primarily uses
            ``"bisquare"`` and ``"gaussian"``.
        alpha: Temporal contribution in ``[0, 1]``. ``"cv"`` selects from
            ``alpha_candidates``.
        theta: Spatial-bandwidth time slope in radians. Earlier bandwidths equal
            the latest bandwidth minus ``tan(theta) * elapsed_time``. ``"cv"``
            selects from ``theta_candidates``.
        tick_nums: Number of most recent stages used. ``None`` uses all stages;
            ``"cv"`` selects from ``tick_candidates``.
        bandwidth_candidates: Optional candidates for automatic bandwidth search.
        alpha_candidates: Optional candidates for automatic alpha search.
        theta_candidates: Optional candidates for automatic theta search.
        tick_candidates: Optional candidates for automatic stage-count search.
        fit_intercept: Include a local intercept.
        distance_metric: Spatial distance metric supported by pyGWRx.
        sigma2_v1: Use ``n - trace(S)`` rather than the v2 residual denominator.
        ridge: Non-negative numerical ridge added to local normal matrices. The
            intercept is not penalized.
        store_weights: Store the final latest-stage-to-history weight matrix.
        verbose: Print selection and fit information.

    References:
        Que, X., Ma, X., Ma, C., & Chen, Q. (2020). A spatiotemporal
        weighted regression model (STWR v1.0) for analyzing local
        nonstationarity in space and time. *Geoscientific Model Development*,
        13, 6149-6164.
    """

    _AUTO_NAMES = {"cv", "auto", "loocv"}

    def __init__(
        self,
        spatial_bandwidth: Bandwidth = "cv",
        *,
        adaptive: bool = True,
        kernel: str = "bisquare",
        alpha: SelectionValue = 0.3,
        theta: SelectionValue = 0.0,
        tick_nums: Union[int, str, None] = None,
        bandwidth_candidates: Optional[Sequence[Number]] = None,
        alpha_candidates: Optional[Sequence[Number]] = None,
        theta_candidates: Optional[Sequence[Number]] = None,
        tick_candidates: Optional[Sequence[int]] = None,
        fit_intercept: bool = True,
        distance_metric: str = "euclidean",
        sigma2_v1: bool = True,
        ridge: float = 0.0,
        store_weights: bool = True,
        verbose: bool = False,
    ) -> None:
        self.spatial_bandwidth = spatial_bandwidth
        self.adaptive = self._boolean(adaptive, "adaptive")
        self.kernel = self._kernel_name(kernel)
        self.alpha = alpha
        self.theta = theta
        self.tick_nums = tick_nums
        self.bandwidth_candidates = bandwidth_candidates
        self.alpha_candidates = alpha_candidates
        self.theta_candidates = theta_candidates
        self.tick_candidates = tick_candidates
        self.fit_intercept = self._boolean(fit_intercept, "fit_intercept")
        self.distance_metric = self._distance_metric(distance_metric)
        self.sigma2_v1 = self._boolean(sigma2_v1, "sigma2_v1")
        self.ridge = self._nonnegative(ridge, "ridge")
        self.store_weights = self._boolean(store_weights, "store_weights")
        self.verbose = self._boolean(verbose, "verbose")
        self.kernel_func_ = get_kernel_function(self.kernel)
        self._reset_fit_state()

    @staticmethod
    def _boolean(value: bool, name: str) -> bool:
        if not isinstance(value, (bool, np.bool_)):
            raise TypeError(f"{name} must be boolean.")
        return bool(value)

    @staticmethod
    def _kernel_name(value: str) -> str:
        if not isinstance(value, str):
            raise TypeError("kernel must be a string.")
        name = value.strip().lower()
        if name not in {"bisquare", "gaussian", "exponential"}:
            raise ValueError("STWR kernel must be bisquare, gaussian, or exponential.")
        get_kernel_function(name)
        return name

    @staticmethod
    def _distance_metric(value: str) -> str:
        if not isinstance(value, str):
            raise TypeError("distance_metric must be a string.")
        name = value.strip().lower()
        allowed = {"euclidean", "manhattan", "cityblock", "chebyshev", "haversine"}
        if name not in allowed:
            raise ValueError(f"distance_metric must be one of {sorted(allowed)}.")
        return name

    @staticmethod
    def _nonnegative(value: Real, name: str) -> float:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise TypeError(f"{name} must be a real scalar.")
        result = float(value)
        if not np.isfinite(result) or result < 0.0:
            raise ValueError(f"{name} must be finite and non-negative.")
        return result

    @staticmethod
    def _numeric_vector(value: object, name: str) -> np.ndarray:
        try:
            array = np.asarray(value, dtype=float)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{name} must be numeric array-like data.") from exc
        if array.ndim == 2 and 1 in array.shape:
            array = array.reshape(-1)
        if array.ndim != 1 or array.size == 0:
            raise ValueError(f"{name} must be a non-empty one-dimensional vector.")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must contain only finite values.")
        return array

    @staticmethod
    def _coerce_X_stage(
        value: Union[np.ndarray, pd.DataFrame],
        *,
        expected_names: Optional[Tuple[str, ...]] = None,
    ) -> Tuple[np.ndarray, Tuple[str, ...]]:
        if isinstance(value, pd.DataFrame):
            names = tuple(str(column) for column in value.columns)
            if expected_names is not None and names != expected_names:
                raise ValueError(
                    "All X stages must use the same DataFrame columns and order."
                )
            array = value.to_numpy(dtype=float)
        else:
            array = np.asarray(value, dtype=float)
            if array.ndim != 2:
                raise ValueError("Each X stage must be a two-dimensional array.")
            names = (
                expected_names
                if expected_names is not None
                else tuple(f"x{index}" for index in range(array.shape[1]))
            )
        if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
            raise ValueError("Each X stage must be a non-empty two-dimensional array.")
        if not np.all(np.isfinite(array)):
            raise ValueError("X stages must contain only finite values.")
        if expected_names is not None and array.shape[1] != len(expected_names):
            raise ValueError("All X stages must have the same number of columns.")
        return array, names

    def _reset_fit_state(self) -> None:
        self._is_fitted = False
        self.feature_names_: Tuple[str, ...] = ()
        self.X_stages_: Optional[List[np.ndarray]] = None
        self.y_stages_: Optional[List[np.ndarray]] = None
        self.coords_stages_: Optional[List[np.ndarray]] = None
        self.time_intervals_: Optional[np.ndarray] = None
        self.spatial_bandwidth_: Optional[Union[int, float]] = None
        self.alpha_: Optional[float] = None
        self.theta_: Optional[float] = None
        self.tick_nums_: Optional[int] = None
        self.selection_history_: List[Dict[str, float]] = []
        self.parameters_: Optional[np.ndarray] = None
        self.coef_: Optional[np.ndarray] = None
        self.coefficients_: Optional[np.ndarray] = None
        self.intercept_: Optional[np.ndarray] = None
        self.fitted_values_: Optional[np.ndarray] = None
        self.residuals_: Optional[np.ndarray] = None
        self.smoother_rows_: Optional[np.ndarray] = None
        self.influence_: Optional[np.ndarray] = None
        self.parameter_covariance_diagonal_: Optional[np.ndarray] = None
        self.parameter_standard_errors_: Optional[np.ndarray] = None
        self.parameter_t_values_: Optional[np.ndarray] = None
        self.coef_se_: Optional[np.ndarray] = None
        self.intercept_se_: Optional[np.ndarray] = None
        self.coef_t_: Optional[np.ndarray] = None
        self.intercept_t_: Optional[np.ndarray] = None
        self.sigma2_: Optional[float] = None
        self.diagnostics_: Optional[Dict[str, float]] = None
        self.weights_: Optional[np.ndarray] = None
        self.spatial_weights_: Optional[np.ndarray] = None
        self.temporal_weights_: Optional[np.ndarray] = None
        self.stage_slices_: Optional[Tuple[slice, ...]] = None

    @classmethod
    def _is_auto(cls, value: object) -> bool:
        return isinstance(value, str) and value.strip().lower() in cls._AUTO_NAMES

    def _validate_stages(
        self,
        X_list: Sequence[Union[np.ndarray, pd.DataFrame]],
        y_list: Sequence[Union[np.ndarray, pd.Series]],
        coords_list: Sequence[Union[np.ndarray, pd.DataFrame]],
        time_intervals: Sequence[Number],
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], np.ndarray]:
        if not isinstance(X_list, Sequence) or isinstance(X_list, (str, bytes)):
            raise TypeError("X_list must be a sequence of time-stage matrices.")
        n_stages = len(X_list)
        if n_stages == 0 or len(y_list) != n_stages or len(coords_list) != n_stages:
            raise ValueError(
                "X_list, y_list, and coords_list must have equal non-zero length."
            )
        X_stages: List[np.ndarray] = []
        y_stages: List[np.ndarray] = []
        coords_stages: List[np.ndarray] = []
        names: Optional[Tuple[str, ...]] = None
        for stage, (X_value, y_value, coord_value) in enumerate(
            zip(X_list, y_list, coords_list)
        ):
            X_stage, names = self._coerce_X_stage(X_value, expected_names=names)
            y_stage = self._numeric_vector(y_value, f"y_list[{stage}]")
            coords_stage = validate_coords(coord_value)
            if (
                X_stage.shape[0] != y_stage.size
                or X_stage.shape[0] != coords_stage.shape[0]
            ):
                raise ValueError(
                    f"Stage {stage} has inconsistent X, y, and coordinate rows."
                )
            X_stages.append(X_stage)
            y_stages.append(y_stage)
            coords_stages.append(coords_stage)
        self.feature_names_ = names or ()

        intervals = np.asarray(time_intervals, dtype=float).reshape(-1)
        if intervals.size == n_stages - 1:
            intervals = np.r_[0.0, intervals]
        if intervals.size != n_stages:
            raise ValueError(
                "time_intervals must contain one zero-prefixed value per stage or "
                "one interval between each pair of stages."
            )
        if not np.all(np.isfinite(intervals)) or np.any(intervals < 0.0):
            raise ValueError("time_intervals must be finite and non-negative.")
        if not np.isclose(intervals[0], 0.0):
            raise ValueError("The first time interval must be zero.")
        if n_stages > 1 and np.any(intervals[1:] <= 0.0):
            raise ValueError("Intervals after the first stage must be positive.")
        return X_stages, y_stages, coords_stages, intervals

    @staticmethod
    def _explicit_unit_interval(value: object, name: str) -> Optional[float]:
        if value is None or STWR._is_auto(value):
            return None
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise TypeError(f"{name} must be numeric or 'cv'.")
        result = float(value)
        if not np.isfinite(result) or not 0.0 <= result <= 1.0:
            raise ValueError(f"{name} must lie in [0, 1].")
        return result

    @staticmethod
    def _explicit_theta(value: object) -> Optional[float]:
        if value is None or STWR._is_auto(value):
            return None
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise TypeError("theta must be numeric or 'cv'.")
        result = float(value)
        if not np.isfinite(result) or not (-np.pi / 2.0 < result < np.pi / 2.0):
            raise ValueError("theta must lie strictly between -pi/2 and pi/2.")
        return result

    def _explicit_tick_nums(self, n_stages: int) -> Optional[int]:
        value = self.tick_nums
        if value is None:
            return n_stages
        if self._is_auto(value):
            return None
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
            raise TypeError("tick_nums must be an integer, None, or 'cv'.")
        result = int(value)
        if not 1 <= result <= n_stages:
            raise ValueError("tick_nums must lie between 1 and the number of stages.")
        return result

    def _explicit_bandwidth(self, n_latest: int) -> Optional[Union[int, float]]:
        value = self.spatial_bandwidth
        if value is None or self._is_auto(value):
            return None
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise TypeError("spatial_bandwidth must be numeric or 'cv'.")
        if self.adaptive:
            if not float(value).is_integer():
                raise ValueError("Adaptive spatial_bandwidth must be an integer count.")
            result: Union[int, float] = int(value)
            if not 2 <= result <= n_latest:
                raise ValueError("Adaptive spatial_bandwidth must be in [2, n_latest].")
            return result
        result = float(value)
        if not np.isfinite(result) or result <= 0.0:
            raise ValueError("Fixed spatial_bandwidth must be positive.")
        return result

    def _candidate_bandwidths(
        self,
        n_latest: int,
        coords: np.ndarray,
    ) -> List[Union[int, float]]:
        if self.bandwidth_candidates is not None:
            candidates = list(self.bandwidth_candidates)
        elif self.adaptive:
            lower = min(n_latest, max(4, len(self.feature_names_) + 2))
            candidates = (
                np.unique(
                    np.rint(np.linspace(lower, n_latest, min(10, n_latest - lower + 1)))
                )
                .astype(int)
                .tolist()
            )
        else:
            distances = compute_distance_matrix(
                coords, coords, metric=self.distance_metric
            )
            positive = distances[distances > 0.0]
            if positive.size == 0:
                raise ValueError(
                    "Fixed bandwidth selection requires distinct coordinates."
                )
            candidates = np.linspace(
                float(np.quantile(positive, 0.2)),
                float(np.quantile(positive, 0.9)),
                10,
            ).tolist()
        validated: List[Union[int, float]] = []
        for candidate in candidates:
            original = self.spatial_bandwidth
            try:
                self.spatial_bandwidth = candidate
                value = self._explicit_bandwidth(n_latest)
            finally:
                self.spatial_bandwidth = original
            if value is not None and value not in validated:
                validated.append(value)
        if not validated:
            raise ValueError("No valid spatial bandwidth candidates were supplied.")
        return validated

    def _candidate_alphas(self) -> List[float]:
        explicit = self._explicit_unit_interval(self.alpha, "alpha")
        if explicit is not None:
            return [explicit]
        values = (
            self.alpha_candidates
            if self.alpha_candidates is not None
            else np.linspace(0.0, 0.9, 10)
        )
        result = [float(value) for value in values]
        if not result or any(
            not np.isfinite(value) or not 0.0 <= value <= 1.0 for value in result
        ):
            raise ValueError("alpha_candidates must contain values in [0, 1].")
        return sorted(set(result))

    def _candidate_thetas(self) -> List[float]:
        explicit = self._explicit_theta(self.theta)
        if explicit is not None:
            return [explicit]
        values = self.theta_candidates if self.theta_candidates is not None else [0.0]
        result = [float(value) for value in values]
        if not result or any(
            not np.isfinite(value) or not (-np.pi / 2.0 < value < np.pi / 2.0)
            for value in result
        ):
            raise ValueError("theta_candidates contain an invalid slope angle.")
        return sorted(set(result))

    def _candidate_ticks(self, n_stages: int) -> List[int]:
        explicit = self._explicit_tick_nums(n_stages)
        if explicit is not None:
            return [explicit]
        values = (
            self.tick_candidates
            if self.tick_candidates is not None
            else range(1, n_stages + 1)
        )
        result = sorted(set(int(value) for value in values))
        if not result or any(value < 1 or value > n_stages for value in result):
            raise ValueError("tick_candidates must lie within the available stages.")
        return result

    def _source_arrays(
        self,
        X_stages: Sequence[np.ndarray],
        y_stages: Sequence[np.ndarray],
        tick_nums: int,
    ) -> Tuple[np.ndarray, np.ndarray, Tuple[slice, ...]]:
        ordered_X = [X_stages[-1 - offset] for offset in range(tick_nums)]
        ordered_y = [y_stages[-1 - offset] for offset in range(tick_nums)]
        slices: List[slice] = []
        start = 0
        for stage in ordered_X:
            stop = start + stage.shape[0]
            slices.append(slice(start, stop))
            start = stop
        return np.vstack(ordered_X), np.concatenate(ordered_y), tuple(slices)

    def _current_bandwidths(
        self,
        distances: np.ndarray,
        bandwidth: Union[int, float],
    ) -> np.ndarray:
        if self.adaptive:
            return np.asarray(
                [adaptive_bandwidth_weights(row, int(bandwidth)) for row in distances],
                dtype=float,
            )
        return np.full(distances.shape[0], float(bandwidth), dtype=float)

    @staticmethod
    def _safe_response_denominator(values: np.ndarray) -> np.ndarray:
        epsilon = 1.0e-6
        return np.where(
            np.abs(values) < epsilon,
            np.where(values < 0.0, -epsilon, epsilon),
            values,
        )

    def _weight_components(
        self,
        query_coords: np.ndarray,
        query_reference_y: np.ndarray,
        coords_stages: Sequence[np.ndarray],
        y_stages: Sequence[np.ndarray],
        intervals: np.ndarray,
        *,
        bandwidth: Union[int, float],
        alpha: float,
        theta: float,
        tick_nums: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[slice, ...]]:
        selected_coords = [coords_stages[-1 - offset] for offset in range(tick_nums)]
        selected_y = [y_stages[-1 - offset] for offset in range(tick_nums)]
        # Source arrays are ordered from the current stage to older stages.
        slices_list: List[slice] = []
        start = 0
        for stage in selected_coords:
            stop = start + stage.shape[0]
            slices_list.append(slice(start, stop))
            start = stop
        slices = tuple(slices_list)
        n_query = query_coords.shape[0]
        n_source = sum(stage.shape[0] for stage in selected_coords)
        spatial = np.zeros((n_query, n_source), dtype=float)
        temporal = np.zeros_like(spatial)
        combined = np.zeros_like(spatial)

        current_distances = np.asarray(
            compute_distance_matrix(
                query_coords,
                selected_coords[0],
                metric=self.distance_metric,
            ),
            dtype=float,
        )
        current_bandwidths = self._current_bandwidths(current_distances, bandwidth)
        total_interval = float(np.sum(intervals[-tick_nums:]))

        elapsed = 0.0
        for offset, (stage_coords, stage_y, stage_slice) in enumerate(
            zip(selected_coords, selected_y, slices)
        ):
            distances = np.asarray(
                compute_distance_matrix(
                    query_coords,
                    stage_coords,
                    metric=self.distance_metric,
                ),
                dtype=float,
            )
            if offset == 0:
                stage_bandwidth = current_bandwidths.copy()
                temporal_effect = np.zeros_like(distances)
            else:
                interval_index = len(intervals) - offset
                elapsed += float(intervals[interval_index])
                stage_bandwidth = current_bandwidths - np.tan(theta) * elapsed
                positive = distances[distances > 0.0]
                minimum = (
                    np.finfo(float).eps
                    if positive.size == 0
                    else max(float(np.min(positive)), np.finfo(float).eps)
                )
                stage_bandwidth = np.maximum(stage_bandwidth, minimum)
                denominator = self._safe_response_denominator(stage_y)
                variation = np.abs(
                    (stage_y[None, :] - query_reference_y[:, None])
                    / denominator[None, :]
                )
                if elapsed <= 0.0:
                    raise ValueError(
                        "Past STWR stages require a positive elapsed time."
                    )
                scale = 1.0 if total_interval <= 0.0 else total_interval / elapsed
                time_distance = scale * variation
                temporal_effect = np.tanh(0.5 * time_distance)

            stage_spatial = np.empty_like(distances)
            for row in range(n_query):
                stage_spatial[row] = self.kernel_func_(
                    distances[row], float(stage_bandwidth[row])
                )
            stage_combined = (
                stage_spatial
                if tick_nums == 1
                else (1.0 - alpha) * stage_spatial + alpha * temporal_effect
            )
            spatial[:, stage_slice] = stage_spatial
            temporal[:, stage_slice] = temporal_effect
            combined[:, stage_slice] = stage_combined

        if np.any(np.sum(combined > 0.0, axis=1) < 1):
            raise ValueError("An STWR calibration location has no positive weights.")
        return spatial, temporal, combined, slices

    def _system_matrix(self, X_design: np.ndarray, weights: np.ndarray) -> np.ndarray:
        system = X_design.T @ (X_design * weights[:, None])
        if self.ridge > 0.0:
            penalty = np.eye(system.shape[0], dtype=float) * self.ridge
            if self.fit_intercept:
                penalty[0, 0] = 0.0
            system = system + penalty
        return system

    def _fit_weight_matrix(
        self,
        X_design: np.ndarray,
        y_source: np.ndarray,
        weights: np.ndarray,
        query_design: np.ndarray,
        *,
        compute_covariance: bool,
    ) -> _STWRLocalFit:
        n_locations = query_design.shape[0]
        n_parameters = X_design.shape[1]
        parameters = np.empty((n_locations, n_parameters), dtype=float)
        fitted = np.empty(n_locations, dtype=float)
        smoother = np.empty((n_locations, X_design.shape[0]), dtype=float)
        covariance = (
            np.empty((n_locations, n_parameters), dtype=float)
            if compute_covariance
            else None
        )
        for location in range(n_locations):
            local_weights = weights[location]
            system = self._system_matrix(X_design, local_weights)
            xtw = X_design.T * local_weights
            try:
                operator = np.linalg.solve(system, xtw)
            except np.linalg.LinAlgError as exc:
                raise np.linalg.LinAlgError(
                    "STWR local system is singular at calibration location "
                    f"{location}; increase the bandwidth, reduce tick_nums, remove "
                    "collinear variables, or set a small ridge value."
                ) from exc
            beta = operator @ y_source
            parameters[location] = beta
            fitted[location] = float(query_design[location] @ beta)
            smoother[location] = query_design[location] @ operator
            if covariance is not None:
                covariance[location] = np.sum(operator**2, axis=1)
        return _STWRLocalFit(parameters, fitted, smoother, covariance)

    def _cv_score(
        self,
        X_stages: Sequence[np.ndarray],
        y_stages: Sequence[np.ndarray],
        coords_stages: Sequence[np.ndarray],
        intervals: np.ndarray,
        *,
        bandwidth: Union[int, float],
        alpha: float,
        theta: float,
        tick_nums: int,
    ) -> float:
        X_source, y_source, _ = self._source_arrays(X_stages, y_stages, tick_nums)
        X_design = add_intercept(X_source) if self.fit_intercept else X_source.copy()
        query_X = X_stages[-1]
        query_design = add_intercept(query_X) if self.fit_intercept else query_X.copy()
        try:
            _, _, weights, _ = self._weight_components(
                coords_stages[-1],
                y_stages[-1],
                coords_stages,
                y_stages,
                intervals,
                bandwidth=bandwidth,
                alpha=alpha,
                theta=theta,
                tick_nums=tick_nums,
            )
            n_latest = y_stages[-1].size
            weights[np.arange(n_latest), np.arange(n_latest)] = 0.0
            fit = self._fit_weight_matrix(
                X_design,
                y_source,
                weights,
                query_design,
                compute_covariance=False,
            )
            residuals = y_stages[-1] - fit.fitted_values
            score = float(np.dot(residuals, residuals))
        except (ValueError, np.linalg.LinAlgError):
            score = np.inf
        self.selection_history_.append(
            {
                "spatial_bandwidth": float(bandwidth),
                "alpha": float(alpha),
                "theta": float(theta),
                "tick_nums": float(tick_nums),
                "cv": score,
            }
        )
        return score

    def _select_parameters(
        self,
        X_stages: Sequence[np.ndarray],
        y_stages: Sequence[np.ndarray],
        coords_stages: Sequence[np.ndarray],
        intervals: np.ndarray,
    ) -> Tuple[Union[int, float], float, float, int]:
        n_stages = len(X_stages)
        n_latest = X_stages[-1].shape[0]
        explicit_bandwidth = self._explicit_bandwidth(n_latest)
        bandwidths = (
            [explicit_bandwidth]
            if explicit_bandwidth is not None
            else self._candidate_bandwidths(n_latest, coords_stages[-1])
        )
        alphas = self._candidate_alphas()
        thetas = self._candidate_thetas()
        ticks = self._candidate_ticks(n_stages)
        if len(bandwidths) == len(alphas) == len(thetas) == len(ticks) == 1:
            return bandwidths[0], alphas[0], thetas[0], ticks[0]
        self.selection_history_ = []
        best: Optional[Tuple[float, Union[int, float], float, float, int]] = None
        for tick_nums in ticks:
            for theta in thetas:
                for alpha in alphas:
                    for bandwidth in bandwidths:
                        score = self._cv_score(
                            X_stages,
                            y_stages,
                            coords_stages,
                            intervals,
                            bandwidth=bandwidth,
                            alpha=alpha,
                            theta=theta,
                            tick_nums=tick_nums,
                        )
                        if best is None or score < best[0]:
                            best = (score, bandwidth, alpha, theta, tick_nums)
        if best is None or not np.isfinite(best[0]):
            raise RuntimeError("STWR parameter selection found no estimable model.")
        return best[1], best[2], best[3], best[4]

    def fit(
        self,
        X_list: Sequence[Union[np.ndarray, pd.DataFrame]],
        y_list: Sequence[Union[np.ndarray, pd.Series]],
        coords_list: Sequence[Union[np.ndarray, pd.DataFrame]],
        time_intervals: Sequence[Number],
    ) -> "STWR":
        """Fit STWR for the latest time stage using recent historical stages."""
        self._reset_fit_state()
        try:
            X_stages, y_stages, coords_stages, intervals = self._validate_stages(
                X_list, y_list, coords_list, time_intervals
            )
            bandwidth, alpha, theta, tick_nums = self._select_parameters(
                X_stages, y_stages, coords_stages, intervals
            )
            X_source, y_source, stage_slices = self._source_arrays(
                X_stages, y_stages, tick_nums
            )
            X_design = (
                add_intercept(X_source) if self.fit_intercept else X_source.copy()
            )
            query_X = X_stages[-1]
            query_design = (
                add_intercept(query_X) if self.fit_intercept else query_X.copy()
            )
            spatial, temporal, weights, _ = self._weight_components(
                coords_stages[-1],
                y_stages[-1],
                coords_stages,
                y_stages,
                intervals,
                bandwidth=bandwidth,
                alpha=alpha,
                theta=theta,
                tick_nums=tick_nums,
            )
            local_fit = self._fit_weight_matrix(
                X_design,
                y_source,
                weights,
                query_design,
                compute_covariance=True,
            )
            y_latest = y_stages[-1]
            residuals = y_latest - local_fit.fitted_values
            n_latest = y_latest.size
            trace_s = float(
                np.sum(
                    local_fit.smoother_rows[np.arange(n_latest), np.arange(n_latest)]
                )
            )
            trace_sts = float(np.sum(local_fit.smoother_rows**2))
            diagnostics = compute_diagnostics(
                y_latest,
                local_fit.fitted_values,
                trace_S=max(trace_s, 0.0),
                trace_StS=max(trace_sts, 0.0),
                compute_gwr_stats=True,
            )
            denominator = (
                n_latest - trace_s
                if self.sigma2_v1
                else n_latest - 2.0 * trace_s + trace_sts
            )
            if denominator <= 0.0:
                raise ValueError(
                    "STWR residual effective degrees of freedom are not positive; "
                    "increase the bandwidth or use fewer historical stages."
                )
            sigma2 = float(np.dot(residuals, residuals) / denominator)
            if local_fit.covariance_diagonal is None:
                raise RuntimeError("STWR covariance factors were not computed.")
            parameter_se = np.sqrt(
                np.maximum(local_fit.covariance_diagonal * sigma2, 0.0)
            )
            parameter_t = np.divide(
                local_fit.parameters,
                parameter_se,
                out=np.full_like(local_fit.parameters, np.nan),
                where=parameter_se > 0.0,
            )

            self.X_stages_ = [stage.copy() for stage in X_stages]
            self.y_stages_ = [stage.copy() for stage in y_stages]
            self.coords_stages_ = [stage.copy() for stage in coords_stages]
            self.time_intervals_ = intervals.copy()
            self.spatial_bandwidth_ = bandwidth
            self.alpha_ = alpha
            self.theta_ = theta
            self.tick_nums_ = tick_nums
            self.stage_slices_ = stage_slices
            self.parameters_ = local_fit.parameters
            if self.fit_intercept:
                self.intercept_ = local_fit.parameters[:, 0]
                self.coef_ = local_fit.parameters[:, 1:]
                self.intercept_se_ = parameter_se[:, 0]
                self.coef_se_ = parameter_se[:, 1:]
                self.intercept_t_ = parameter_t[:, 0]
                self.coef_t_ = parameter_t[:, 1:]
            else:
                self.intercept_ = np.zeros(n_latest, dtype=float)
                self.coef_ = local_fit.parameters
                self.intercept_se_ = np.zeros(n_latest, dtype=float)
                self.coef_se_ = parameter_se
                self.intercept_t_ = np.full(n_latest, np.nan)
                self.coef_t_ = parameter_t
            self.coefficients_ = self.coef_
            self.fitted_values_ = local_fit.fitted_values
            self.residuals_ = residuals
            self.smoother_rows_ = local_fit.smoother_rows
            self.influence_ = local_fit.smoother_rows[
                np.arange(n_latest), np.arange(n_latest)
            ]
            self.parameter_covariance_diagonal_ = local_fit.covariance_diagonal
            self.parameter_standard_errors_ = parameter_se
            self.parameter_t_values_ = parameter_t
            self.sigma2_ = sigma2
            self.diagnostics_ = diagnostics
            if self.store_weights:
                self.weights_ = weights
                self.spatial_weights_ = spatial
                self.temporal_weights_ = temporal
            self._is_fitted = True
        except Exception:
            self._reset_fit_state()
            raise
        if self.verbose:
            print(
                "STWR fitted: "
                f"bandwidth={self.spatial_bandwidth_}, alpha={self.alpha_:.4f}, "
                f"theta={self.theta_:.4f}, tick_nums={self.tick_nums_}, "
                f"AICc={self.diagnostics_['aicc']:.6f}"
            )
        return self

    def _require_fitted(self) -> None:
        if not self._is_fitted:
            raise ValueError("STWR is not fitted. Call fit() first.")

    def _estimate_reference_y(self, coords_new: np.ndarray) -> np.ndarray:
        if self.coords_stages_ is None or self.y_stages_ is None:
            raise RuntimeError("STWR training state is incomplete.")
        distances = np.asarray(
            compute_distance_matrix(
                coords_new,
                self.coords_stages_[-1],
                metric=self.distance_metric,
            ),
            dtype=float,
        )
        epsilon = np.finfo(float).eps
        inverse = 1.0 / np.maximum(distances, epsilon)
        exact = distances <= epsilon
        result = np.empty(coords_new.shape[0], dtype=float)
        for row in range(coords_new.shape[0]):
            if np.any(exact[row]):
                result[row] = float(np.mean(self.y_stages_[-1][exact[row]]))
            else:
                result[row] = float(
                    np.dot(inverse[row], self.y_stages_[-1]) / np.sum(inverse[row])
                )
        return result

    def predict_result(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        coords: Union[np.ndarray, pd.DataFrame],
        *,
        reference_y: Optional[Union[np.ndarray, pd.Series]] = None,
    ) -> STWRPredictionResult:
        """Predict at new locations in the latest modeled time stage.

        ``reference_y`` supplies the current-stage response baseline required by
        the STWR variation-rate time distance. When omitted, it is estimated by
        inverse-distance weighting from the latest observed responses, following
        the prediction strategy in the public STWR code.
        """
        self._require_fitted()
        if (
            self.X_stages_ is None
            or self.y_stages_ is None
            or self.coords_stages_ is None
            or self.time_intervals_ is None
            or self.spatial_bandwidth_ is None
            or self.alpha_ is None
            or self.theta_ is None
            or self.tick_nums_ is None
        ):
            raise RuntimeError("STWR training state is incomplete.")
        X_arr, _ = self._coerce_X_stage(X, expected_names=self.feature_names_)
        coords_arr = validate_coords(coords)
        if X_arr.shape[0] != coords_arr.shape[0]:
            raise ValueError("X and coords must contain the same number of rows.")
        reference = (
            self._estimate_reference_y(coords_arr)
            if reference_y is None
            else self._numeric_vector(reference_y, "reference_y")
        )
        if reference.size != X_arr.shape[0]:
            raise ValueError("reference_y must contain one value per prediction row.")
        X_source, y_source, _ = self._source_arrays(
            self.X_stages_, self.y_stages_, self.tick_nums_
        )
        X_design = add_intercept(X_source) if self.fit_intercept else X_source.copy()
        query_design = add_intercept(X_arr) if self.fit_intercept else X_arr.copy()
        _, _, weights, _ = self._weight_components(
            coords_arr,
            reference,
            self.coords_stages_,
            self.y_stages_,
            self.time_intervals_,
            bandwidth=self.spatial_bandwidth_,
            alpha=self.alpha_,
            theta=self.theta_,
            tick_nums=self.tick_nums_,
        )
        local_fit = self._fit_weight_matrix(
            X_design,
            y_source,
            weights,
            query_design,
            compute_covariance=False,
        )
        if self.fit_intercept:
            intercept = local_fit.parameters[:, 0]
            coef = local_fit.parameters[:, 1:]
        else:
            intercept = np.zeros(X_arr.shape[0], dtype=float)
            coef = local_fit.parameters
        return STWRPredictionResult(
            predictions=local_fit.fitted_values,
            coef=coef,
            intercept=intercept,
            coords=coords_arr.copy(),
            feature_names=self.feature_names_,
            reference_y=reference.copy(),
        )

    def predict(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        coords: Union[np.ndarray, pd.DataFrame],
        *,
        reference_y: Optional[Union[np.ndarray, pd.Series]] = None,
    ) -> np.ndarray:
        """Return STWR predictions at latest-stage locations."""
        return self.predict_result(X, coords, reference_y=reference_y).predictions

    def get_results(self) -> pd.DataFrame:
        """Return latest-stage fitted values and local coefficients."""
        self._require_fitted()
        if self.coords_stages_ is None or self.fitted_values_ is None:
            raise RuntimeError("STWR training results are incomplete.")
        result = STWRPredictionResult(
            predictions=self.fitted_values_,
            coef=self.coef_,
            intercept=self.intercept_,
            coords=self.coords_stages_[-1],
            feature_names=self.feature_names_,
            reference_y=self.y_stages_[-1].copy(),
        ).to_frame()
        result["residual"] = self.residuals_
        return result
