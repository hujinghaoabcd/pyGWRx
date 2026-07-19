# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Similarity and geographically weighted regression.

This module implements the Gaussian SGWR model proposed by Lessani and Li
(2024). SGWR combines a conventional geographic kernel with an attribute-space
similarity kernel and estimates the mixing parameter from Gaussian GWR AICc.

Author:
    Jinghao Hu
"""

from __future__ import annotations

__author__ = "Jinghao Hu"
__license__ = "MIT"

from dataclasses import dataclass
from numbers import Integral, Real
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from pygwrx.core._summary import format_summary
from pygwrx.core.kernels import get_kernel_function
from pygwrx.core.metrics import compute_diagnostics
from pygwrx.core.solver import adaptive_bandwidth_weights
from pygwrx.core.utils import add_intercept, compute_distance_matrix, validate_coords
from pygwrx.models.gwr import GWR

Number = Union[int, float]
Bandwidth = Union[int, float, str, None]
Alpha = Union[float, str, None]


@dataclass(frozen=True)
class SGWRPredictionResult:
    """Detailed SGWR predictions at new calibration locations."""

    predictions: np.ndarray
    coef: np.ndarray
    intercept: np.ndarray
    coords: np.ndarray
    feature_names: Tuple[str, ...]

    def to_frame(self) -> pd.DataFrame:
        """Return predictions and local parameters as a DataFrame."""
        data: Dict[str, np.ndarray] = {
            "coord_0": self.coords[:, 0],
            "coord_1": self.coords[:, 1],
            "prediction": self.predictions,
            "intercept": self.intercept,
        }
        for index, name in enumerate(self.feature_names):
            data[f"coef_{name}"] = self.coef[:, index]
        return pd.DataFrame(data)


@dataclass
class _LocalFit:
    parameters: np.ndarray
    fitted_values: np.ndarray
    hat_matrix: np.ndarray
    covariance_diagonal: Optional[np.ndarray]


class SGWR:
    r"""Similarity and geographically weighted regression.

    For calibration location :math:`i`, SGWR constructs

    .. math::

        W_i^{GS} = \alpha W_i^G + (1 - \alpha) W_i^S,

    where :math:`W_i^G` is a geographic kernel and the published similarity
    kernel is

    .. math::

        w_{ij}^S = \exp\left[-\left(\frac{1}{m}
        \sum_{r=1}^{m}|z_{ir}-z_{jr}|\right)^2\right].

    The similarity variables ``z`` are standardized using the training-sample
    mean and population standard deviation before distances are calculated.
    ``alpha=1`` is ordinary GWR and ``alpha=0`` is similarity-only local
    regression.

    Args:
        bandwidth: Geographic bandwidth. Numeric values are used directly.
            ``None`` or ``"aicc"`` selects the bandwidth from a pure GWR using
            AICc before optimizing ``alpha``.
        adaptive: Interpret a numeric bandwidth as a one-based neighbour count.
        kernel: Geographic kernel used in the final SGWR fit.
        alpha: Geographic mixing proportion. Numeric values lie in ``[0, 1]``.
            ``None`` or ``"aicc"`` selects alpha by SGWR AICc.
        similarity_vars: Predictor names or zero-based column indices used to
            construct attribute similarity. ``None`` uses all predictors.
        standardize_similarity: Standardize similarity variables before the
            published mean-absolute-distance calculation.
        bandwidth_kernel: Optional kernel used only for automatic pure-GWR
            bandwidth selection. This permits the software-paper hybrid of an
            adaptive bi-square search followed by an adaptive Gaussian SGWR fit.
        bandwidth_range: Optional search bounds passed to the standard GWR
            bandwidth selector.
        alpha_range: Bounds used for automatic alpha selection.
        alpha_grid_size: Number of deterministic coarse alpha candidates before
            bounded local refinement.
        fit_intercept: Include a local intercept.
        distance_metric: Coordinate distance metric used by pyGWRx.
        sigma2_v1: Residual variance convention. ``True`` uses
            ``RSS / (n - trace(S))``; ``False`` uses
            ``RSS / (n - 2 trace(S) + trace(S'S))``.
        ridge: Optional non-negative numerical ridge added to slope diagonals.
            The intercept is not penalized.
        store_weights: Store the three ``n x n`` training weight matrices.
        verbose: Print selection and fit progress.

    References:
        Lessani, M. N., & Li, Z. (2024). SGWR: similarity and geographically
        weighted regression. *International Journal of Geographical Information
        Science, 38*(7), 1232-1255.

        Lessani, M. N., & Li, Z. (2025). Enhancing the computational efficiency
        of the SGWR model and introducing its software implementation.
        *Annals of GIS*.
    """

    _AUTO_NAMES = {"aicc", "auto", "optimize", "optimise"}

    def __init__(
        self,
        bandwidth: Bandwidth = "aicc",
        adaptive: bool = True,
        kernel: str = "bisquare",
        alpha: Alpha = "aicc",
        similarity_vars: Optional[Sequence[Union[int, str]]] = None,
        *,
        standardize_similarity: bool = True,
        bandwidth_kernel: Optional[str] = None,
        bandwidth_range: Optional[Tuple[float, float]] = None,
        alpha_range: Tuple[float, float] = (0.01, 1.0),
        alpha_grid_size: int = 21,
        fit_intercept: bool = True,
        distance_metric: str = "euclidean",
        sigma2_v1: bool = True,
        ridge: float = 0.0,
        store_weights: bool = True,
        verbose: bool = False,
    ) -> None:
        self.bandwidth = bandwidth
        self.adaptive = self._validate_boolean(adaptive, "adaptive")
        self.kernel = self._validate_kernel_name(kernel, "kernel")
        self.alpha = alpha
        self.similarity_vars = similarity_vars
        self.standardize_similarity = self._validate_boolean(
            standardize_similarity, "standardize_similarity"
        )
        self.bandwidth_kernel = (
            self.kernel
            if bandwidth_kernel is None
            else self._validate_kernel_name(bandwidth_kernel, "bandwidth_kernel")
        )
        self.bandwidth_range = bandwidth_range
        self.alpha_range = self._validate_alpha_range(alpha_range)
        self.alpha_grid_size = self._validate_alpha_grid_size(alpha_grid_size)
        self.fit_intercept = self._validate_boolean(fit_intercept, "fit_intercept")
        self.distance_metric = self._validate_distance_metric(distance_metric)
        self.sigma2_v1 = self._validate_boolean(sigma2_v1, "sigma2_v1")
        self.ridge = self._validate_nonnegative_float(ridge, "ridge")
        self.store_weights = self._validate_boolean(store_weights, "store_weights")
        self.verbose = self._validate_boolean(verbose, "verbose")

        self.kernel_func_ = get_kernel_function(self.kernel)
        self._reset_fit_state()

    @staticmethod
    def _validate_boolean(value: bool, name: str) -> bool:
        if not isinstance(value, (bool, np.bool_)):
            raise TypeError(f"{name} must be boolean.")
        return bool(value)

    @staticmethod
    def _validate_kernel_name(value: str, name: str) -> str:
        if not isinstance(value, str):
            raise TypeError(f"{name} must be a kernel name string.")
        normalized = value.strip().lower()
        get_kernel_function(normalized)
        return normalized

    @staticmethod
    def _validate_distance_metric(value: str) -> str:
        if not isinstance(value, str):
            raise TypeError("distance_metric must be a string.")
        normalized = value.strip().lower()
        if normalized not in {
            "euclidean",
            "manhattan",
            "cityblock",
            "chebyshev",
            "haversine",
        }:
            raise ValueError(
                "distance_metric must be one of 'euclidean', 'manhattan', "
                "'cityblock', 'chebyshev', or 'haversine'."
            )
        return normalized

    @staticmethod
    def _validate_nonnegative_float(value: Real, name: str) -> float:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise TypeError(f"{name} must be a non-negative real scalar.")
        value_float = float(value)
        if not np.isfinite(value_float) or value_float < 0.0:
            raise ValueError(f"{name} must be finite and non-negative.")
        return value_float

    @staticmethod
    def _validate_alpha_grid_size(value: int) -> int:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
            raise TypeError("alpha_grid_size must be an integer.")
        value_int = int(value)
        if value_int < 3:
            raise ValueError("alpha_grid_size must be at least 3.")
        return value_int

    @staticmethod
    def _validate_alpha_range(value: Tuple[float, float]) -> Tuple[float, float]:
        if not isinstance(value, (tuple, list)) or len(value) != 2:
            raise TypeError("alpha_range must be a two-value tuple.")
        lower, upper = float(value[0]), float(value[1])
        if not np.isfinite(lower) or not np.isfinite(upper):
            raise ValueError("alpha_range values must be finite.")
        if lower < 0.0 or upper > 1.0 or lower >= upper:
            raise ValueError("alpha_range must satisfy 0 <= lower < upper <= 1.")
        return lower, upper

    def _reset_fit_state(self) -> None:
        self._is_fitted = False
        self.bandwidth_: Optional[Union[int, float]] = None
        self.alpha_: Optional[float] = None
        self.feature_names_: Tuple[str, ...] = ()
        self.similarity_feature_names_: Tuple[str, ...] = ()
        self.similarity_indices_: Optional[np.ndarray] = None
        self.similarity_mean_: Optional[np.ndarray] = None
        self.similarity_scale_: Optional[np.ndarray] = None
        self.X_train_: Optional[np.ndarray] = None
        self.X_design_: Optional[np.ndarray] = None
        self.y_train_: Optional[np.ndarray] = None
        self.coords_train_: Optional[np.ndarray] = None
        self.parameters_: Optional[np.ndarray] = None
        self.intercept_: Optional[np.ndarray] = None
        self.coef_: Optional[np.ndarray] = None
        self.coefficients_: Optional[np.ndarray] = None
        self.fitted_values_: Optional[np.ndarray] = None
        self.residuals_: Optional[np.ndarray] = None
        self.hat_matrix_: Optional[np.ndarray] = None
        self.influence_: Optional[np.ndarray] = None
        self.parameter_covariance_diagonal_: Optional[np.ndarray] = None
        self.parameter_standard_errors_: Optional[np.ndarray] = None
        self.parameter_t_values_: Optional[np.ndarray] = None
        self.intercept_se_: Optional[np.ndarray] = None
        self.coef_se_: Optional[np.ndarray] = None
        self.intercept_t_: Optional[np.ndarray] = None
        self.coef_t_: Optional[np.ndarray] = None
        self.sigma2_: Optional[float] = None
        self.standardized_residuals_: Optional[np.ndarray] = None
        self.cooks_distance_: Optional[np.ndarray] = None
        self.local_r2_: Optional[np.ndarray] = None
        self.diagnostics_: Optional[Dict[str, float]] = None
        self.spatial_weights_: Optional[np.ndarray] = None
        self.similarity_weights_: Optional[np.ndarray] = None
        self.combined_weights_: Optional[np.ndarray] = None
        self.alpha_search_history_: List[Tuple[float, float]] = []
        self.alpha_score_: Optional[float] = None
        self.bandwidth_selector_: Optional[GWR] = None

    @staticmethod
    def _coerce_X(
        X: Union[np.ndarray, pd.DataFrame],
        *,
        expected_names: Optional[Tuple[str, ...]] = None,
    ) -> Tuple[np.ndarray, Tuple[str, ...]]:
        if isinstance(X, pd.DataFrame):
            if expected_names is not None and tuple(X.columns) != expected_names:
                raise ValueError(
                    "Prediction DataFrame columns must match the training columns "
                    "in the same order."
                )
            names = tuple(str(column) for column in X.columns)
            values = X.to_numpy(dtype=float)
        else:
            try:
                values = np.asarray(X, dtype=float)
            except (TypeError, ValueError) as exc:
                raise TypeError("X must contain numeric values.") from exc
            if values.ndim == 1:
                values = values.reshape(-1, 1)
            names = tuple(f"x{index}" for index in range(values.shape[1]))

        if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
            raise ValueError("X must be a non-empty two-dimensional matrix.")
        if not np.all(np.isfinite(values)):
            raise ValueError("X contains NaN or infinite values.")
        if expected_names is not None and values.shape[1] != len(expected_names):
            raise ValueError(
                f"X must have {len(expected_names)} columns; got {values.shape[1]}."
            )
        return values, names

    @staticmethod
    def _coerce_y(y: Union[np.ndarray, pd.Series], n_samples: int) -> np.ndarray:
        try:
            values = np.asarray(y, dtype=float)
        except (TypeError, ValueError) as exc:
            raise TypeError("y must contain numeric values.") from exc
        if values.ndim == 2 and values.shape[1] == 1:
            values = values[:, 0]
        elif values.ndim != 1:
            raise ValueError("y must be one-dimensional or a single-column array.")
        if values.shape[0] != n_samples:
            raise ValueError("X and y must contain the same number of samples.")
        if not np.all(np.isfinite(values)):
            raise ValueError("y contains NaN or infinite values.")
        return values

    def _resolve_similarity_indices(
        self,
        feature_names: Tuple[str, ...],
        n_features: int,
    ) -> np.ndarray:
        if self.similarity_vars is None:
            return np.arange(n_features, dtype=int)
        if isinstance(self.similarity_vars, (str, bytes)):
            requested: Iterable[Union[int, str]] = [self.similarity_vars]
        else:
            requested = self.similarity_vars

        indices: List[int] = []
        for item in requested:
            if isinstance(item, (bool, np.bool_)):
                raise TypeError("similarity_vars cannot contain boolean values.")
            if isinstance(item, Integral):
                index = int(item)
                if index < 0 or index >= n_features:
                    raise IndexError(
                        "similarity variable index "
                        f"{index} is outside [0, {n_features})."
                    )
            elif isinstance(item, str):
                if item not in feature_names:
                    raise ValueError(f"Unknown similarity variable name: {item!r}.")
                index = feature_names.index(item)
            else:
                raise TypeError(
                    "similarity_vars entries must be integer indices or column names."
                )
            if index not in indices:
                indices.append(index)
        if not indices:
            raise ValueError("similarity_vars must select at least one predictor.")
        return np.asarray(indices, dtype=int)

    def _fit_similarity_scaler(self, X_similarity: np.ndarray) -> np.ndarray:
        if self.standardize_similarity:
            mean = np.mean(X_similarity, axis=0)
            scale = np.std(X_similarity, axis=0, ddof=0)
            scale = np.where(scale <= np.finfo(float).eps, 1.0, scale)
        else:
            mean = np.zeros(X_similarity.shape[1], dtype=float)
            scale = np.ones(X_similarity.shape[1], dtype=float)
        self.similarity_mean_ = mean
        self.similarity_scale_ = scale
        return (X_similarity - mean) / scale

    def _transform_similarity(self, X: np.ndarray) -> np.ndarray:
        if (
            self.similarity_indices_ is None
            or self.similarity_mean_ is None
            or self.similarity_scale_ is None
        ):
            raise RuntimeError("The similarity transformation is unavailable.")
        selected = X[:, self.similarity_indices_]
        return (selected - self.similarity_mean_) / self.similarity_scale_

    @staticmethod
    def _similarity_distance(
        query_similarity: np.ndarray,
        training_similarity: np.ndarray,
    ) -> np.ndarray:
        differences = np.abs(
            query_similarity[:, np.newaxis, :] - training_similarity[np.newaxis, :, :]
        )
        return np.mean(differences, axis=2)

    @classmethod
    def _similarity_weights(
        cls,
        query_similarity: np.ndarray,
        training_similarity: np.ndarray,
    ) -> np.ndarray:
        distance = cls._similarity_distance(query_similarity, training_similarity)
        return np.exp(-(distance**2))

    @staticmethod
    def _normalize_weight_rows(weights: np.ndarray) -> np.ndarray:
        maxima = np.max(weights, axis=1)
        if np.any(~np.isfinite(maxima)) or np.any(maxima <= 0.0):
            raise ValueError("Every combined-weight row must contain a positive value.")
        return weights / maxima[:, np.newaxis]

    def _spatial_weights(
        self,
        query_coords: np.ndarray,
        training_coords: np.ndarray,
        bandwidth: Union[int, float],
        *,
        kernel_name: Optional[str] = None,
    ) -> np.ndarray:
        distances = compute_distance_matrix(
            query_coords,
            training_coords,
            metric=self.distance_metric,
        )
        kernel = get_kernel_function(kernel_name or self.kernel)
        weights = np.empty_like(distances, dtype=float)
        for index, distance_row in enumerate(distances):
            local_bandwidth = (
                adaptive_bandwidth_weights(distance_row, int(bandwidth))
                if self.adaptive
                else float(bandwidth)
            )
            weights[index] = kernel(distance_row, local_bandwidth)
        if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
            raise ValueError("The geographic kernel produced invalid weights.")
        return weights

    def _combine_weights(
        self,
        spatial_weights: np.ndarray,
        similarity_weights: np.ndarray,
        alpha: float,
    ) -> np.ndarray:
        combined = alpha * spatial_weights + (1.0 - alpha) * similarity_weights
        return self._normalize_weight_rows(combined)

    def _resolve_numeric_bandwidth(
        self,
        value: Number,
        n_samples: int,
        n_design_columns: int,
    ) -> Union[int, float]:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise TypeError("bandwidth must be numeric, None, or 'aicc'.")
        numeric = float(value)
        if not np.isfinite(numeric) or numeric <= 0.0:
            raise ValueError("bandwidth must be finite and greater than zero.")
        if self.adaptive:
            if not numeric.is_integer():
                raise ValueError(
                    "adaptive bandwidth must be an integer neighbour count."
                )
            count = int(numeric)
            minimum = n_design_columns + 1
            if count < minimum:
                raise ValueError(
                    f"adaptive bandwidth must be at least {minimum} for this design."
                )
            if count > n_samples:
                raise ValueError(
                    f"adaptive bandwidth cannot exceed n_samples={n_samples}."
                )
            return count
        return numeric

    def _select_bandwidth(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: np.ndarray,
        coords: np.ndarray,
        X_design: np.ndarray,
    ) -> Union[int, float]:
        value = self.bandwidth
        if isinstance(value, str):
            if value.strip().lower() not in self._AUTO_NAMES:
                raise ValueError("bandwidth string must be 'aicc'.")
            automatic = True
        else:
            automatic = value is None

        if not automatic:
            return self._resolve_numeric_bandwidth(
                value, X_design.shape[0], X_design.shape[1]
            )

        if self.verbose:
            print("Selecting a pure-GWR bandwidth by AICc...")
        selector = GWR(
            kernel=self.bandwidth_kernel,
            bandwidth="aicc",
            adaptive=self.adaptive,
            bandwidth_range=self.bandwidth_range,
            fit_intercept=self.fit_intercept,
            distance_metric=self.distance_metric,
            sigma2_v1=self.sigma2_v1,
            verbose=False,
        )
        selector.fit(X, y, coords)
        if selector.bandwidth_ is None:
            raise RuntimeError("The pure-GWR bandwidth selector returned no bandwidth.")
        self.bandwidth_selector_ = selector
        return int(selector.bandwidth_) if self.adaptive else float(selector.bandwidth_)

    @classmethod
    def _resolve_numeric_alpha(cls, value: Alpha) -> Optional[float]:
        if isinstance(value, str):
            if value.strip().lower() not in cls._AUTO_NAMES:
                raise ValueError("alpha string must be 'aicc'.")
            return None
        if value is None:
            return None
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise TypeError("alpha must be numeric, None, or 'aicc'.")
        numeric = float(value)
        if not np.isfinite(numeric) or numeric < 0.0 or numeric > 1.0:
            raise ValueError("alpha must lie within [0, 1].")
        return numeric

    def _system_matrix(self, X_design: np.ndarray, weights: np.ndarray) -> np.ndarray:
        system = X_design.T @ (X_design * weights[:, np.newaxis])
        if self.ridge > 0.0:
            penalty = np.eye(X_design.shape[1]) * self.ridge
            if self.fit_intercept:
                penalty[0, 0] = 0.0
            system = system + penalty
        return system

    def _fit_weight_matrix(
        self,
        X_design: np.ndarray,
        y: np.ndarray,
        weights: np.ndarray,
        query_design: np.ndarray,
        *,
        compute_covariance: bool,
    ) -> _LocalFit:
        n_locations = weights.shape[0]
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
                coefficient_operator = np.linalg.solve(system, xtw)
            except np.linalg.LinAlgError as exc:
                raise np.linalg.LinAlgError(
                    "SGWR local system is singular at calibration location "
                    f"{location}; increase the geographic bandwidth, choose a larger "
                    "alpha, remove collinear variables, or set a small ridge value."
                ) from exc
            beta = coefficient_operator @ y
            parameters[location] = beta
            fitted[location] = float(query_design[location] @ beta)
            hat[location] = query_design[location] @ coefficient_operator
            if covariance is not None:
                covariance[location] = np.sum(coefficient_operator**2, axis=1)

        return _LocalFit(parameters, fitted, hat, covariance)

    def _alpha_objective(
        self,
        alpha: float,
        X_design: np.ndarray,
        y: np.ndarray,
        spatial: np.ndarray,
        similarity: np.ndarray,
    ) -> float:
        try:
            combined = self._combine_weights(spatial, similarity, float(alpha))
            fitted = self._fit_weight_matrix(
                X_design,
                y,
                combined,
                X_design,
                compute_covariance=False,
            )
            diagnostics = compute_diagnostics(
                y,
                fitted.fitted_values,
                hat_matrix=fitted.hat_matrix,
                compute_gwr_stats=True,
            )
            score = float(diagnostics["aicc"])
        except (ValueError, np.linalg.LinAlgError):
            score = np.inf
        self.alpha_search_history_.append((float(alpha), score))
        return score

    def _select_alpha(
        self,
        X_design: np.ndarray,
        y: np.ndarray,
        spatial: np.ndarray,
        similarity: np.ndarray,
    ) -> float:
        explicit = self._resolve_numeric_alpha(self.alpha)
        if explicit is not None:
            self.alpha_search_history_ = []
            return explicit

        lower, upper = self.alpha_range
        grid = np.linspace(lower, upper, self.alpha_grid_size)
        self.alpha_search_history_ = []
        scores = np.asarray(
            [
                self._alpha_objective(
                    candidate,
                    X_design,
                    y,
                    spatial,
                    similarity,
                )
                for candidate in grid
            ],
            dtype=float,
        )
        if not np.any(np.isfinite(scores)):
            raise RuntimeError(
                "Automatic alpha selection found no estimable SGWR model."
            )
        best_index = int(np.nanargmin(scores))
        best_alpha = float(grid[best_index])
        best_score = float(scores[best_index])

        left = float(grid[max(best_index - 1, 0)])
        right = float(grid[min(best_index + 1, grid.size - 1)])
        if right > left:
            result = minimize_scalar(
                lambda candidate: self._alpha_objective(
                    candidate,
                    X_design,
                    y,
                    spatial,
                    similarity,
                ),
                method="bounded",
                bounds=(left, right),
                options={"xatol": 1e-5, "maxiter": 100},
            )
            if result.success and np.isfinite(result.fun) and result.fun < best_score:
                best_alpha = float(result.x)
                best_score = float(result.fun)

        self.alpha_score_ = best_score
        return best_alpha

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        coords: Union[np.ndarray, pd.DataFrame],
    ) -> "SGWR":
        """Fit Gaussian SGWR at the observed locations."""
        self._reset_fit_state()
        try:
            X_arr, feature_names = self._coerce_X(X)
            y_arr = self._coerce_y(y, X_arr.shape[0])
            coords_arr = validate_coords(coords)
            if coords_arr.shape[0] != X_arr.shape[0]:
                raise ValueError("X and coords must contain the same number of rows.")

            similarity_indices = self._resolve_similarity_indices(
                feature_names, X_arr.shape[1]
            )
            self.similarity_indices_ = similarity_indices
            self.feature_names_ = feature_names
            self.similarity_feature_names_ = tuple(
                feature_names[index] for index in similarity_indices
            )
            X_similarity = self._fit_similarity_scaler(X_arr[:, similarity_indices])
            X_design = add_intercept(X_arr) if self.fit_intercept else X_arr.copy()

            bandwidth = self._select_bandwidth(X, y_arr, coords_arr, X_design)
            spatial = self._spatial_weights(coords_arr, coords_arr, bandwidth)
            similarity = self._similarity_weights(X_similarity, X_similarity)
            alpha = self._select_alpha(X_design, y_arr, spatial, similarity)
            combined = self._combine_weights(spatial, similarity, alpha)

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
                    "Residual effective degrees of freedom are not positive; "
                    "increase the bandwidth or simplify the design."
                )
            sigma2 = float(np.dot(residuals, residuals) / denominator)
            if local_fit.covariance_diagonal is None:
                raise RuntimeError("SGWR covariance factors were not computed.")
            parameter_se = np.sqrt(
                np.maximum(local_fit.covariance_diagonal * sigma2, 0.0)
            )
            parameter_t = np.divide(
                local_fit.parameters,
                parameter_se,
                out=np.full_like(local_fit.parameters, np.nan),
                where=parameter_se > 0.0,
            )
            influence = np.diag(local_fit.hat_matrix)
            standardized_residuals = np.divide(
                residuals,
                np.sqrt(np.maximum(sigma2 * (1.0 - influence), 0.0)),
                out=np.full_like(residuals, np.nan),
                where=(1.0 - influence) > 0.0,
            )
            cooks = np.divide(
                standardized_residuals**2 * influence,
                max(trace_s, np.finfo(float).eps) * (1.0 - influence),
                out=np.full_like(residuals, np.nan),
                where=(1.0 - influence) > 0.0,
            )

            local_r2 = np.empty(y_arr.size, dtype=float)
            for index, weight_row in enumerate(combined):
                weight_sum = float(np.sum(weight_row))
                local_mean = float(np.dot(weight_row, y_arr) / weight_sum)
                tss = float(np.dot(weight_row, (y_arr - local_mean) ** 2))
                rss = float(np.dot(weight_row, residuals**2))
                local_r2[index] = np.nan if tss <= 0.0 else 1.0 - rss / tss

            self.bandwidth_ = bandwidth
            self.alpha_ = alpha
            self.X_train_ = X_arr.copy()
            self.X_design_ = X_design
            self.y_train_ = y_arr
            self.coords_train_ = coords_arr
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
            self.influence_ = influence
            self.parameter_covariance_diagonal_ = local_fit.covariance_diagonal
            self.parameter_standard_errors_ = parameter_se
            self.parameter_t_values_ = parameter_t
            self.sigma2_ = sigma2
            self.standardized_residuals_ = standardized_residuals
            self.cooks_distance_ = cooks
            self.local_r2_ = local_r2
            self.diagnostics_ = diagnostics
            if self.store_weights:
                self.spatial_weights_ = spatial
                self.similarity_weights_ = similarity
                self.combined_weights_ = combined
            self._is_fitted = True
        except Exception:
            self._reset_fit_state()
            raise

        if self.verbose:
            print(
                "SGWR fitted: "
                f"bandwidth={self.bandwidth_}, alpha={self.alpha_:.6f}, "
                f"AICc={self.diagnostics_['aicc']:.6f}"
            )
        return self

    def _require_fitted(self) -> None:
        if not self._is_fitted:
            raise ValueError("SGWR is not fitted. Call fit() first.")

    def _prediction_inputs(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        coords: Union[np.ndarray, pd.DataFrame],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        self._require_fitted()
        X_arr, _ = self._coerce_X(X, expected_names=self.feature_names_)
        coords_arr = validate_coords(coords)
        if coords_arr.shape[0] != X_arr.shape[0]:
            raise ValueError("X and coords must contain the same number of rows.")
        if (
            self.coords_train_ is None
            or coords_arr.shape[1] != self.coords_train_.shape[1]
        ):
            raise ValueError("Prediction coordinates have the wrong dimension.")
        X_design = add_intercept(X_arr) if self.fit_intercept else X_arr.copy()
        return X_arr, X_design, coords_arr

    def predict_result(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        coords: Union[np.ndarray, pd.DataFrame],
    ) -> SGWRPredictionResult:
        """Recalibrate SGWR at new locations and return local parameters."""
        X_arr, X_design, coords_arr = self._prediction_inputs(X, coords)
        if (
            self.X_design_ is None
            or self.y_train_ is None
            or self.coords_train_ is None
            or self.bandwidth_ is None
            or self.alpha_ is None
        ):
            raise RuntimeError("Training state is incomplete.")

        training_similarity = self._transform_similarity(self.X_train_)
        query_similarity = self._transform_similarity(X_arr)
        spatial = self._spatial_weights(coords_arr, self.coords_train_, self.bandwidth_)
        similarity = self._similarity_weights(query_similarity, training_similarity)
        combined = self._combine_weights(spatial, similarity, self.alpha_)
        local_fit = self._fit_weight_matrix(
            self.X_design_,
            self.y_train_,
            combined,
            X_design,
            compute_covariance=False,
        )
        if self.fit_intercept:
            intercept = local_fit.parameters[:, 0]
            coef = local_fit.parameters[:, 1:]
        else:
            intercept = np.zeros(X_arr.shape[0], dtype=float)
            coef = local_fit.parameters
        return SGWRPredictionResult(
            predictions=local_fit.fitted_values,
            coef=coef,
            intercept=intercept,
            coords=coords_arr.copy(),
            feature_names=self.feature_names_,
        )

    def predict(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        coords: Union[np.ndarray, pd.DataFrame],
    ) -> np.ndarray:
        """Predict at new locations using direct SGWR recalibration."""
        return self.predict_result(X, coords).predictions

    def results_frame(self) -> pd.DataFrame:
        """Return training-location parameters, inference, and fitted values."""
        self._require_fitted()
        if (
            self.coords_train_ is None
            or self.fitted_values_ is None
            or self.residuals_ is None
            or self.coef_ is None
            or self.coef_se_ is None
            or self.coef_t_ is None
            or self.intercept_ is None
            or self.intercept_se_ is None
            or self.intercept_t_ is None
            or self.local_r2_ is None
            or self.influence_ is None
            or self.cooks_distance_ is None
        ):
            raise RuntimeError("Training results are incomplete.")
        data: Dict[str, np.ndarray] = {
            "coord_0": self.coords_train_[:, 0],
            "coord_1": self.coords_train_[:, 1],
            "fitted": self.fitted_values_,
            "residual": self.residuals_,
            "intercept": self.intercept_,
            "intercept_se": self.intercept_se_,
            "intercept_t": self.intercept_t_,
            "local_r2": self.local_r2_,
            "influence": self.influence_,
            "cooks_distance": self.cooks_distance_,
        }
        for index, name in enumerate(self.feature_names_):
            data[f"coef_{name}"] = self.coef_[:, index]
            data[f"se_{name}"] = self.coef_se_[:, index]
            data[f"t_{name}"] = self.coef_t_[:, index]
        return pd.DataFrame(data)

    def summary(self) -> str:
        """Return a plain-text SGWR configuration and diagnostics table."""
        self._require_fitted()
        if self.diagnostics_ is None:
            raise RuntimeError("Diagnostics are unavailable.")
        summary: Dict[str, object] = dict(self.diagnostics_)
        summary.update(
            {
                "n_samples": int(self.y_train_.size),
                "n_features": len(self.feature_names_),
                "feature_names": self.feature_names_,
                "similarity_features": self.similarity_feature_names_,
                "bandwidth": self.bandwidth_,
                "adaptive": self.adaptive,
                "kernel": self.kernel,
                "bandwidth_kernel": self.bandwidth_kernel,
                "alpha": self.alpha_,
                "standardize_similarity": self.standardize_similarity,
                "sigma2": self.sigma2_,
            }
        )
        return format_summary("SGWR Summary", summary)
