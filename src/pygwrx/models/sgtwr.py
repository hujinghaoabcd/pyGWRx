# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Spatiotemporal GWR with static attribute-similarity weighting.

This module implements the SGTWR formulation published by Li et al. (2025).
The model combines a two-bandwidth Gaussian spatiotemporal kernel with the
static attribute-similarity kernel used by SGWR.

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

from pygwrx.core.metrics import compute_diagnostics
from pygwrx.core.solver import adaptive_bandwidth_weights
from pygwrx.core.utils import add_intercept, compute_distance_matrix, validate_coords
from pygwrx.models.gtwr import GTWR

Number = Union[int, float]
SelectionValue = Union[int, float, str, None]


@dataclass(frozen=True)
class SGTWRPredictionResult:
    """Detailed predictions from a fitted SGTWR model."""

    predictions: np.ndarray
    coef: np.ndarray
    intercept: np.ndarray
    coords: np.ndarray
    times: np.ndarray
    feature_names: Tuple[str, ...]

    def to_frame(self) -> pd.DataFrame:
        """Return predictions and local parameters as a DataFrame."""
        data: Dict[str, np.ndarray] = {
            "coord_0": self.coords[:, 0],
            "coord_1": self.coords[:, 1],
            "time": self.times,
            "prediction": self.predictions,
            "intercept": self.intercept,
        }
        for index, name in enumerate(self.feature_names):
            data[f"coef_{name}"] = self.coef[:, index]
        return pd.DataFrame(data)


@dataclass
class _SGTWRLocalFit:
    parameters: np.ndarray
    fitted_values: np.ndarray
    hat_matrix: np.ndarray
    covariance_diagonal: Optional[np.ndarray]


class SGTWR:
    r"""Spatiotemporal geographically weighted regression with similarity.

    The published spatiotemporal weight is

    .. math::

        W_{ST,ij}=\exp\left[-\frac{1}{2}\left(
        \left(\frac{d^S_{ij}}{h^S_i}\right)^2+
        \left(\frac{d^T_{ij}}{h^T}\right)^2\right)\right].

    Static attribute similarity follows SGWR:

    .. math::

        W_{S,ij}=\exp\left[-\left(
        \frac{1}{m}\sum_{k=1}^{m}|z_{ik}-z_{jk}|\right)^2\right].

    The final local weight is

    .. math::

        W_{ij}=\alpha W_{ST,ij}+(1-\alpha)W_{S,ij}.

    The paper uses a genetic algorithm to tune the spatial neighbour count,
    temporal bandwidth, and mixing coefficient. pyGWRx uses a deterministic
    AICc candidate search so fitted results are reproducible and testable.

    Args:
        spatial_bandwidth: Fixed spatial distance or adaptive neighbour count.
            ``"aicc"`` selects from ``spatial_bandwidth_candidates``.
        temporal_bandwidth: Positive temporal bandwidth. ``"aicc"`` selects
            from ``temporal_bandwidth_candidates``.
        adaptive: Interpret ``spatial_bandwidth`` as a neighbour count.
        alpha: Spatiotemporal contribution in ``[0, 1]``. ``"aicc"`` selects
            from ``alpha_candidates``.
        similarity_vars: Predictor names or indices used for similarity.
            ``None`` uses every predictor.
        standardize_similarity: Standardize selected variables before taking
            absolute attribute differences. This matches the paper's Z-score
            preprocessing and the standardized SGWR implementation.
        spatial_bandwidth_candidates: Optional spatial candidates for AICc
            selection.
        temporal_bandwidth_candidates: Optional temporal candidates for AICc
            selection.
        alpha_candidates: Optional mixing candidates for AICc selection.
        causal: Exclude source observations later than a regression time.
        time_unit: Numeric unit or datetime conversion convention used by GTWR.
        fit_intercept: Include a local intercept.
        distance_metric: Spatial distance metric supported by pyGWRx.
        sigma2_v1: Use ``n - trace(S)`` for residual variance when true.
        ridge: Non-negative stabilization added to slope normal equations. The
            intercept is not penalized.
        store_weights: Store component and combined weight matrices.
        verbose: Print selected parameters and AICc.

    References:
        Li, M., Du, W., Yu, S., Hong, Z., Zhang, D., He, Y., & De, L. (2025).
        SGTWR Model with Spatial-Temporal Heterogeneity and Attribute Similarity
        for Urban Traffic Carbon Emission Driver Analysis. *Sustainability*,
        17(23), 10773.
    """

    _AUTO_NAMES = {"aicc", "auto", "optimize", "optimise"}

    def __init__(
        self,
        spatial_bandwidth: SelectionValue = "aicc",
        *,
        temporal_bandwidth: SelectionValue = "aicc",
        adaptive: bool = True,
        alpha: SelectionValue = "aicc",
        similarity_vars: Optional[Sequence[Union[int, str]]] = None,
        standardize_similarity: bool = True,
        spatial_bandwidth_candidates: Optional[Sequence[Number]] = None,
        temporal_bandwidth_candidates: Optional[Sequence[Number]] = None,
        alpha_candidates: Optional[Sequence[Number]] = None,
        causal: bool = False,
        time_unit: str = "auto",
        fit_intercept: bool = True,
        distance_metric: str = "euclidean",
        sigma2_v1: bool = True,
        ridge: float = 0.0,
        store_weights: bool = True,
        verbose: bool = False,
    ) -> None:
        self.spatial_bandwidth = spatial_bandwidth
        self.temporal_bandwidth = temporal_bandwidth
        self.adaptive = self._boolean(adaptive, "adaptive")
        self.alpha = alpha
        self.similarity_vars = similarity_vars
        self.standardize_similarity = self._boolean(
            standardize_similarity,
            "standardize_similarity",
        )
        self.spatial_bandwidth_candidates = spatial_bandwidth_candidates
        self.temporal_bandwidth_candidates = temporal_bandwidth_candidates
        self.alpha_candidates = alpha_candidates
        self.causal = self._boolean(causal, "causal")
        self.time_unit = self._time_unit(time_unit)
        self.fit_intercept = self._boolean(fit_intercept, "fit_intercept")
        self.distance_metric = self._distance_metric(distance_metric)
        self.sigma2_v1 = self._boolean(sigma2_v1, "sigma2_v1")
        self.ridge = self._nonnegative(ridge, "ridge")
        self.store_weights = self._boolean(store_weights, "store_weights")
        self.verbose = self._boolean(verbose, "verbose")
        self._reset_fit_state()

    @staticmethod
    def _boolean(value: bool, name: str) -> bool:
        if not isinstance(value, (bool, np.bool_)):
            raise TypeError(f"{name} must be boolean.")
        return bool(value)

    @staticmethod
    def _distance_metric(value: str) -> str:
        if not isinstance(value, str):
            raise TypeError("distance_metric must be a string.")
        name = value.strip().lower()
        allowed = {
            "euclidean",
            "manhattan",
            "cityblock",
            "chebyshev",
            "haversine",
        }
        if name not in allowed:
            raise ValueError(f"distance_metric must be one of {sorted(allowed)}.")
        return name

    @staticmethod
    def _time_unit(value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise TypeError("time_unit must be a non-empty string.")
        name = value.strip().lower()
        allowed = {
            "auto",
            "second",
            "seconds",
            "minute",
            "minutes",
            "hour",
            "hours",
            "day",
            "days",
            "week",
            "weeks",
        }
        if name not in allowed:
            raise ValueError(f"time_unit must be one of {sorted(allowed)}.")
        return name

    @staticmethod
    def _nonnegative(value: Real, name: str) -> float:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise TypeError(f"{name} must be a real scalar.")
        result = float(value)
        if not np.isfinite(result) or result < 0.0:
            raise ValueError(f"{name} must be finite and non-negative.")
        return result

    @classmethod
    def _is_auto(cls, value: object) -> bool:
        return isinstance(value, str) and value.strip().lower() in cls._AUTO_NAMES

    def _reset_fit_state(self) -> None:
        self._is_fitted = False
        self.feature_names_: Tuple[str, ...] = ()
        self.similarity_indices_: Tuple[int, ...] = ()
        self.similarity_feature_names_: Tuple[str, ...] = ()
        self.similarity_mean_: Optional[np.ndarray] = None
        self.similarity_scale_: Optional[np.ndarray] = None
        self.X_train_: Optional[np.ndarray] = None
        self.X_design_: Optional[np.ndarray] = None
        self.y_train_: Optional[np.ndarray] = None
        self.coords_train_: Optional[np.ndarray] = None
        self.times_train_: Optional[np.ndarray] = None
        self.time_converter_: Optional[GTWR] = None
        self.time_unit_: Optional[str] = None
        self.spatial_bandwidth_: Optional[Union[int, float]] = None
        self.temporal_bandwidth_: Optional[float] = None
        self.alpha_: Optional[float] = None
        self.selection_history_: List[Dict[str, float]] = []
        self.parameters_: Optional[np.ndarray] = None
        self.coef_: Optional[np.ndarray] = None
        self.coefficients_: Optional[np.ndarray] = None
        self.intercept_: Optional[np.ndarray] = None
        self.fitted_values_: Optional[np.ndarray] = None
        self.residuals_: Optional[np.ndarray] = None
        self.hat_matrix_: Optional[np.ndarray] = None
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
        self.spatiotemporal_weights_: Optional[np.ndarray] = None
        self.similarity_weights_: Optional[np.ndarray] = None
        self.combined_weights_: Optional[np.ndarray] = None

    @staticmethod
    def _coerce_X(
        value: Union[np.ndarray, pd.DataFrame],
        *,
        expected_names: Optional[Tuple[str, ...]] = None,
    ) -> Tuple[np.ndarray, Tuple[str, ...]]:
        if isinstance(value, pd.DataFrame):
            names = tuple(str(column) for column in value.columns)
            if expected_names is not None and names != expected_names:
                raise ValueError(
                    "Prediction DataFrame columns must match training order."
                )
            array = value.to_numpy(dtype=float)
        else:
            array = np.asarray(value, dtype=float)
            if array.ndim != 2:
                raise ValueError("X must be a two-dimensional array.")
            names = expected_names or tuple(
                f"x{index}" for index in range(array.shape[1])
            )
        if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
            raise ValueError("X must be a non-empty two-dimensional array.")
        if not np.all(np.isfinite(array)):
            raise ValueError("X must contain only finite values.")
        if expected_names is not None and array.shape[1] != len(expected_names):
            raise ValueError("X has the wrong number of columns.")
        return array, names

    @staticmethod
    def _coerce_y(
        value: Union[np.ndarray, pd.Series],
        n_rows: int,
    ) -> np.ndarray:
        array = np.asarray(value, dtype=float)
        if array.ndim == 2 and 1 in array.shape:
            array = array.reshape(-1)
        if array.ndim != 1 or array.size != n_rows:
            raise ValueError("y must contain one value per row of X.")
        if not np.all(np.isfinite(array)):
            raise ValueError("y must contain only finite values.")
        return array

    def _fit_times(self, value: object) -> np.ndarray:
        converter = GTWR(time_unit=self.time_unit)
        converted = converter._convert_times(value, reset=True)
        self.time_converter_ = converter
        self.time_unit_ = converter.time_unit_
        return converted

    def _predict_times(self, value: object) -> np.ndarray:
        if self.time_converter_ is None:
            raise RuntimeError("The SGTWR time converter is unavailable.")
        return self.time_converter_._convert_times(value, reset=False)

    def _resolve_similarity_indices(
        self,
        feature_names: Tuple[str, ...],
        n_features: int,
    ) -> Tuple[int, ...]:
        if self.similarity_vars is None:
            return tuple(range(n_features))
        if isinstance(self.similarity_vars, (str, bytes)):
            requested: Sequence[Union[int, str]] = [self.similarity_vars]
        else:
            requested = self.similarity_vars
        indices: List[int] = []
        for value in requested:
            if isinstance(value, str):
                if value not in feature_names:
                    raise ValueError(f"Unknown similarity variable {value!r}.")
                index = feature_names.index(value)
            elif isinstance(value, Integral) and not isinstance(
                value,
                (bool, np.bool_),
            ):
                index = int(value)
                if not 0 <= index < n_features:
                    raise ValueError("A similarity variable index is out of range.")
            else:
                raise TypeError(
                    "similarity_vars entries must be names or integer indices."
                )
            if index not in indices:
                indices.append(index)
        if not indices:
            raise ValueError("similarity_vars cannot be empty.")
        return tuple(indices)

    def _fit_similarity_scaler(self, X: np.ndarray) -> np.ndarray:
        if self.standardize_similarity:
            mean = np.mean(X, axis=0)
            scale = np.std(X, axis=0, ddof=0)
            scale = np.where(scale <= np.finfo(float).eps, 1.0, scale)
        else:
            mean = np.zeros(X.shape[1], dtype=float)
            scale = np.ones(X.shape[1], dtype=float)
        self.similarity_mean_ = mean
        self.similarity_scale_ = scale
        return (X - mean) / scale

    def _transform_similarity(self, X: np.ndarray) -> np.ndarray:
        if self.similarity_mean_ is None or self.similarity_scale_ is None:
            raise RuntimeError("Similarity scaler is unavailable.")
        selected = X[:, self.similarity_indices_]
        return (selected - self.similarity_mean_) / self.similarity_scale_

    @staticmethod
    def _similarity_weights(
        query: np.ndarray,
        source: np.ndarray,
    ) -> np.ndarray:
        distance = np.mean(
            np.abs(query[:, None, :] - source[None, :, :]),
            axis=2,
        )
        return np.exp(-(distance**2))

    def _explicit_spatial_bandwidth(
        self,
        n_samples: int,
        n_parameters: int,
    ) -> Optional[Union[int, float]]:
        value = self.spatial_bandwidth
        if value is None or self._is_auto(value):
            return None
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise TypeError("spatial_bandwidth must be numeric or 'aicc'.")
        numeric = float(value)
        if not np.isfinite(numeric) or numeric <= 0.0:
            raise ValueError("spatial_bandwidth must be positive and finite.")
        if self.adaptive:
            if not numeric.is_integer():
                raise ValueError("Adaptive spatial_bandwidth must be an integer count.")
            result = int(numeric)
            minimum = n_parameters + 1
            if result < minimum or result > n_samples:
                raise ValueError(
                    "Adaptive spatial_bandwidth must lie between "
                    f"{minimum} and {n_samples}."
                )
            return result
        return numeric

    def _explicit_temporal_bandwidth(self) -> Optional[float]:
        value = self.temporal_bandwidth
        if value is None or self._is_auto(value):
            return None
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise TypeError("temporal_bandwidth must be numeric or 'aicc'.")
        result = float(value)
        if not np.isfinite(result) or result <= 0.0:
            raise ValueError("temporal_bandwidth must be positive and finite.")
        return result

    def _explicit_alpha(self) -> Optional[float]:
        value = self.alpha
        if value is None or self._is_auto(value):
            return None
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise TypeError("alpha must be numeric or 'aicc'.")
        result = float(value)
        if not np.isfinite(result) or not 0.0 <= result <= 1.0:
            raise ValueError("alpha must lie in [0, 1].")
        return result

    def _spatial_candidates(
        self,
        coords: np.ndarray,
        n_parameters: int,
    ) -> List[Union[int, float]]:
        explicit = self._explicit_spatial_bandwidth(
            coords.shape[0],
            n_parameters,
        )
        if explicit is not None:
            return [explicit]
        if self.spatial_bandwidth_candidates is not None:
            raw = list(self.spatial_bandwidth_candidates)
        elif self.adaptive:
            lower = min(coords.shape[0], n_parameters + 1)
            raw = (
                np.unique(
                    np.rint(
                        np.linspace(
                            lower,
                            coords.shape[0],
                            min(10, coords.shape[0] - lower + 1),
                        )
                    )
                )
                .astype(int)
                .tolist()
            )
        else:
            distances = compute_distance_matrix(
                coords,
                coords,
                metric=self.distance_metric,
            )
            positive = distances[distances > 0.0]
            if positive.size == 0:
                raise ValueError(
                    "Fixed spatial bandwidth selection requires distinct coordinates."
                )
            raw = np.quantile(positive, np.linspace(0.2, 0.9, 8)).tolist()
        candidates: List[Union[int, float]] = []
        original = self.spatial_bandwidth
        try:
            for value in raw:
                self.spatial_bandwidth = value
                candidate = self._explicit_spatial_bandwidth(
                    coords.shape[0],
                    n_parameters,
                )
                if candidate is not None and candidate not in candidates:
                    candidates.append(candidate)
        finally:
            self.spatial_bandwidth = original
        if not candidates:
            raise ValueError("No valid spatial bandwidth candidates were supplied.")
        return candidates

    def _temporal_candidates(self, times: np.ndarray) -> List[float]:
        explicit = self._explicit_temporal_bandwidth()
        if explicit is not None:
            return [explicit]
        if self.temporal_bandwidth_candidates is not None:
            raw = list(self.temporal_bandwidth_candidates)
        else:
            distances = np.abs(times[:, None] - times[None, :])
            positive = distances[distances > 0.0]
            raw = (
                [1.0]
                if positive.size == 0
                else np.quantile(
                    positive,
                    np.linspace(0.2, 1.0, 8),
                ).tolist()
            )
        candidates = sorted(
            {
                float(value)
                for value in raw
                if isinstance(value, Real)
                and not isinstance(value, (bool, np.bool_))
                and np.isfinite(float(value))
                and float(value) > 0.0
            }
        )
        if not candidates:
            raise ValueError("No valid temporal bandwidth candidates were supplied.")
        return candidates

    def _alpha_candidates(self) -> List[float]:
        explicit = self._explicit_alpha()
        if explicit is not None:
            return [explicit]
        raw = (
            self.alpha_candidates
            if self.alpha_candidates is not None
            else np.linspace(0.0, 1.0, 11)
        )
        candidates = sorted(
            {
                float(value)
                for value in raw
                if isinstance(value, Real)
                and not isinstance(value, (bool, np.bool_))
                and np.isfinite(float(value))
                and 0.0 <= float(value) <= 1.0
            }
        )
        if not candidates:
            raise ValueError("No valid alpha candidates were supplied.")
        return candidates

    def _spatiotemporal_weights(
        self,
        query_coords: np.ndarray,
        query_times: np.ndarray,
        source_coords: np.ndarray,
        source_times: np.ndarray,
        spatial_bandwidth: Union[int, float],
        temporal_bandwidth: float,
    ) -> np.ndarray:
        spatial_distance = np.asarray(
            compute_distance_matrix(
                query_coords,
                source_coords,
                metric=self.distance_metric,
            ),
            dtype=float,
        )
        temporal_distance = np.abs(query_times[:, None] - source_times[None, :])
        if self.adaptive:
            spatial_scales = np.asarray(
                [
                    adaptive_bandwidth_weights(row, int(spatial_bandwidth))
                    for row in spatial_distance
                ],
                dtype=float,
            )
        else:
            spatial_scales = np.full(
                query_coords.shape[0],
                float(spatial_bandwidth),
                dtype=float,
            )
        normalized_spatial = spatial_distance / spatial_scales[:, None]
        normalized_temporal = temporal_distance / float(temporal_bandwidth)
        weights = np.exp(-0.5 * (normalized_spatial**2 + normalized_temporal**2))
        if self.causal:
            future = source_times[None, :] > query_times[:, None]
            weights[future] = 0.0
        if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
            raise ValueError(
                "The SGTWR spatiotemporal kernel produced invalid weights."
            )
        return weights

    def _combine_weights(
        self,
        spatiotemporal: np.ndarray,
        similarity: np.ndarray,
        alpha: float,
        *,
        query_times: Optional[np.ndarray] = None,
        source_times: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        combined = alpha * spatiotemporal + (1.0 - alpha) * similarity
        if self.causal:
            if query_times is None or source_times is None:
                raise RuntimeError("Causal SGTWR requires query and source times.")
            future = source_times[None, :] > query_times[:, None]
            combined[future] = 0.0
        if not np.all(np.isfinite(combined)) or np.any(combined < 0.0):
            raise ValueError("SGTWR combined weights are invalid.")
        if np.any(np.max(combined, axis=1) <= 0.0):
            raise ValueError("An SGTWR location has no positive combined weights.")
        return combined

    def _system_matrix(
        self,
        X_design: np.ndarray,
        weights: np.ndarray,
    ) -> np.ndarray:
        system = X_design.T @ (X_design * weights[:, None])
        if self.ridge > 0.0:
            penalty = np.eye(system.shape[0], dtype=float) * self.ridge
            if self.fit_intercept:
                penalty[0, 0] = 0.0
            system += penalty
        return system

    def _fit_weight_matrix(
        self,
        X_design: np.ndarray,
        y: np.ndarray,
        weights: np.ndarray,
        query_design: np.ndarray,
        *,
        compute_covariance: bool,
    ) -> _SGTWRLocalFit:
        n_locations = query_design.shape[0]
        n_parameters = X_design.shape[1]
        parameters = np.empty((n_locations, n_parameters), dtype=float)
        fitted = np.empty(n_locations, dtype=float)
        hat = np.empty((n_locations, X_design.shape[0]), dtype=float)
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
                    "SGTWR local system is singular at calibration location "
                    f"{location}; increase either bandwidth, increase alpha, "
                    "remove collinear variables, or set a small ridge value."
                ) from exc
            beta = operator @ y
            parameters[location] = beta
            fitted[location] = float(query_design[location] @ beta)
            hat[location] = query_design[location] @ operator
            if covariance is not None:
                covariance[location] = np.sum(operator**2, axis=1)
        return _SGTWRLocalFit(parameters, fitted, hat, covariance)

    def _selection_score(
        self,
        X_design: np.ndarray,
        y: np.ndarray,
        coords: np.ndarray,
        times: np.ndarray,
        similarity: np.ndarray,
        spatial_bandwidth: Union[int, float],
        temporal_bandwidth: float,
        alpha: float,
    ) -> float:
        try:
            spatiotemporal = self._spatiotemporal_weights(
                coords,
                times,
                coords,
                times,
                spatial_bandwidth,
                temporal_bandwidth,
            )
            combined = self._combine_weights(
                spatiotemporal,
                similarity,
                alpha,
                query_times=times,
                source_times=times,
            )
            local_fit = self._fit_weight_matrix(
                X_design,
                y,
                combined,
                X_design,
                compute_covariance=False,
            )
            diagnostics = compute_diagnostics(
                y,
                local_fit.fitted_values,
                hat_matrix=local_fit.hat_matrix,
                compute_gwr_stats=True,
            )
            score = float(diagnostics["aicc"])
            if not np.isfinite(score):
                score = np.inf
        except (ValueError, np.linalg.LinAlgError):
            score = np.inf
        self.selection_history_.append(
            {
                "spatial_bandwidth": float(spatial_bandwidth),
                "temporal_bandwidth": float(temporal_bandwidth),
                "alpha": float(alpha),
                "aicc": score,
            }
        )
        return score

    def _select_parameters(
        self,
        X_design: np.ndarray,
        y: np.ndarray,
        coords: np.ndarray,
        times: np.ndarray,
        similarity: np.ndarray,
    ) -> Tuple[Union[int, float], float, float]:
        spatial_values = self._spatial_candidates(
            coords,
            X_design.shape[1],
        )
        temporal_values = self._temporal_candidates(times)
        alpha_values = self._alpha_candidates()
        if (
            len(spatial_values) == 1
            and len(temporal_values) == 1
            and len(alpha_values) == 1
        ):
            self.selection_history_ = []
            return spatial_values[0], temporal_values[0], alpha_values[0]
        self.selection_history_ = []
        best: Optional[Tuple[float, Union[int, float], float, float]] = None
        for spatial_bandwidth in spatial_values:
            for temporal_bandwidth in temporal_values:
                for alpha in alpha_values:
                    score = self._selection_score(
                        X_design,
                        y,
                        coords,
                        times,
                        similarity,
                        spatial_bandwidth,
                        temporal_bandwidth,
                        alpha,
                    )
                    if best is None or score < best[0]:
                        best = (
                            score,
                            spatial_bandwidth,
                            temporal_bandwidth,
                            alpha,
                        )
        if best is None or not np.isfinite(best[0]):
            raise RuntimeError("SGTWR selection found no estimable model.")
        return best[1], best[2], best[3]

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        coords: Union[np.ndarray, pd.DataFrame],
        times: object,
    ) -> "SGTWR":
        """Fit SGTWR at observed space-time locations."""
        self._reset_fit_state()
        try:
            X_arr, names = self._coerce_X(X)
            y_arr = self._coerce_y(y, X_arr.shape[0])
            coords_arr = validate_coords(coords)
            if coords_arr.shape[0] != X_arr.shape[0]:
                raise ValueError("X and coords must contain the same number of rows.")
            times_arr = self._fit_times(times)
            if times_arr.size != X_arr.shape[0]:
                raise ValueError("times must contain one value per row of X.")

            self.feature_names_ = names
            self.similarity_indices_ = self._resolve_similarity_indices(
                names,
                X_arr.shape[1],
            )
            self.similarity_feature_names_ = tuple(
                names[index] for index in self.similarity_indices_
            )
            similarity_X = self._fit_similarity_scaler(
                X_arr[:, self.similarity_indices_]
            )
            similarity = self._similarity_weights(similarity_X, similarity_X)
            X_design = add_intercept(X_arr) if self.fit_intercept else X_arr.copy()
            (
                spatial_bandwidth,
                temporal_bandwidth,
                alpha,
            ) = self._select_parameters(
                X_design,
                y_arr,
                coords_arr,
                times_arr,
                similarity,
            )
            spatiotemporal = self._spatiotemporal_weights(
                coords_arr,
                times_arr,
                coords_arr,
                times_arr,
                spatial_bandwidth,
                temporal_bandwidth,
            )
            combined = self._combine_weights(
                spatiotemporal,
                similarity,
                alpha,
                query_times=times_arr,
                source_times=times_arr,
            )
            local_fit = self._fit_weight_matrix(
                X_design,
                y_arr,
                combined,
                X_design,
                compute_covariance=True,
            )
            residuals = y_arr - local_fit.fitted_values
            diagnostics = compute_diagnostics(
                y_arr,
                local_fit.fitted_values,
                hat_matrix=local_fit.hat_matrix,
                compute_gwr_stats=True,
            )
            trace_s = float(diagnostics["trace_S"])
            trace_sts = float(diagnostics["trace_StS"])
            denominator = (
                y_arr.size - trace_s
                if self.sigma2_v1
                else y_arr.size - 2.0 * trace_s + trace_sts
            )
            if denominator <= 0.0:
                raise ValueError(
                    "SGTWR residual effective degrees of freedom are not positive; "
                    "increase either bandwidth or simplify the design."
                )
            sigma2 = float(np.dot(residuals, residuals) / denominator)
            if local_fit.covariance_diagonal is None:
                raise RuntimeError("SGTWR covariance factors were not computed.")
            parameter_se = np.sqrt(
                np.maximum(
                    local_fit.covariance_diagonal * sigma2,
                    0.0,
                )
            )
            parameter_t = np.divide(
                local_fit.parameters,
                parameter_se,
                out=np.full_like(local_fit.parameters, np.nan),
                where=parameter_se > 0.0,
            )

            self.X_train_ = X_arr.copy()
            self.X_design_ = X_design
            self.y_train_ = y_arr.copy()
            self.coords_train_ = coords_arr.copy()
            self.times_train_ = times_arr.copy()
            self.spatial_bandwidth_ = spatial_bandwidth
            self.temporal_bandwidth_ = temporal_bandwidth
            self.alpha_ = alpha
            self.parameters_ = local_fit.parameters
            if self.fit_intercept:
                self.intercept_ = local_fit.parameters[:, 0]
                self.coef_ = local_fit.parameters[:, 1:]
                self.intercept_se_ = parameter_se[:, 0]
                self.coef_se_ = parameter_se[:, 1:]
                self.intercept_t_ = parameter_t[:, 0]
                self.coef_t_ = parameter_t[:, 1:]
            else:
                self.intercept_ = np.zeros(y_arr.size, dtype=float)
                self.coef_ = local_fit.parameters
                self.intercept_se_ = np.zeros(y_arr.size, dtype=float)
                self.coef_se_ = parameter_se
                self.intercept_t_ = np.full(y_arr.size, np.nan)
                self.coef_t_ = parameter_t
            self.coefficients_ = self.coef_
            self.fitted_values_ = local_fit.fitted_values
            self.residuals_ = residuals
            self.hat_matrix_ = local_fit.hat_matrix
            self.influence_ = np.diag(local_fit.hat_matrix)
            self.parameter_covariance_diagonal_ = local_fit.covariance_diagonal
            self.parameter_standard_errors_ = parameter_se
            self.parameter_t_values_ = parameter_t
            self.sigma2_ = sigma2
            self.diagnostics_ = diagnostics
            if self.store_weights:
                self.spatiotemporal_weights_ = spatiotemporal
                self.similarity_weights_ = similarity
                self.combined_weights_ = combined
            self._is_fitted = True
        except Exception:
            self._reset_fit_state()
            raise
        if self.verbose:
            print(
                "SGTWR fitted: "
                f"spatial_bandwidth={self.spatial_bandwidth_}, "
                f"temporal_bandwidth={self.temporal_bandwidth_:.6g}, "
                f"alpha={self.alpha_:.4f}, "
                f"AICc={self.diagnostics_['aicc']:.6f}"
            )
        return self

    def _require_fitted(self) -> None:
        if not self._is_fitted:
            raise ValueError("SGTWR is not fitted. Call fit() first.")

    def predict_result(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        coords: Union[np.ndarray, pd.DataFrame],
        times: object,
    ) -> SGTWRPredictionResult:
        """Recalibrate SGTWR at new space-time locations."""
        self._require_fitted()
        if (
            self.X_train_ is None
            or self.X_design_ is None
            or self.y_train_ is None
            or self.coords_train_ is None
            or self.times_train_ is None
            or self.spatial_bandwidth_ is None
            or self.temporal_bandwidth_ is None
            or self.alpha_ is None
        ):
            raise RuntimeError("SGTWR training state is incomplete.")
        X_arr, _ = self._coerce_X(X, expected_names=self.feature_names_)
        coords_arr = validate_coords(coords)
        times_arr = self._predict_times(times)
        if X_arr.shape[0] != coords_arr.shape[0] or X_arr.shape[0] != times_arr.size:
            raise ValueError(
                "X, coords, and times must contain the same number of rows."
            )
        query_design = add_intercept(X_arr) if self.fit_intercept else X_arr.copy()
        training_similarity = self._transform_similarity(self.X_train_)
        query_similarity = self._transform_similarity(X_arr)
        spatiotemporal = self._spatiotemporal_weights(
            coords_arr,
            times_arr,
            self.coords_train_,
            self.times_train_,
            self.spatial_bandwidth_,
            self.temporal_bandwidth_,
        )
        similarity = self._similarity_weights(
            query_similarity,
            training_similarity,
        )
        combined = self._combine_weights(
            spatiotemporal,
            similarity,
            self.alpha_,
            query_times=times_arr,
            source_times=self.times_train_,
        )
        local_fit = self._fit_weight_matrix(
            self.X_design_,
            self.y_train_,
            combined,
            query_design,
            compute_covariance=False,
        )
        if self.fit_intercept:
            intercept = local_fit.parameters[:, 0]
            coef = local_fit.parameters[:, 1:]
        else:
            intercept = np.zeros(X_arr.shape[0], dtype=float)
            coef = local_fit.parameters
        return SGTWRPredictionResult(
            predictions=local_fit.fitted_values,
            coef=coef,
            intercept=intercept,
            coords=coords_arr.copy(),
            times=times_arr.copy(),
            feature_names=self.feature_names_,
        )

    def predict(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        coords: Union[np.ndarray, pd.DataFrame],
        times: object,
    ) -> np.ndarray:
        """Return SGTWR predictions at new space-time locations."""
        return self.predict_result(X, coords, times).predictions

    def get_results(self) -> pd.DataFrame:
        """Return fitted values and local coefficients as a DataFrame."""
        self._require_fitted()
        if (
            self.fitted_values_ is None
            or self.coords_train_ is None
            or self.times_train_ is None
            or self.coef_ is None
            or self.intercept_ is None
        ):
            raise RuntimeError("SGTWR fitted results are incomplete.")
        frame = SGTWRPredictionResult(
            predictions=self.fitted_values_,
            coef=self.coef_,
            intercept=self.intercept_,
            coords=self.coords_train_,
            times=self.times_train_,
            feature_names=self.feature_names_,
        ).to_frame()
        frame["residual"] = self.residuals_
        return frame
