# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Standard Gaussian geographically weighted regression.

This module provides the reference estimator interface, local inference, diagnostics, and prediction behavior used as the quality baseline for other pyGWRx regression models.

Author:
    Jinghao Hu
"""

from __future__ import annotations

__author__ = "Jinghao Hu"
__license__ = "MIT"

import warnings
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd

from pygwrx.core.bandwidth import get_bandwidth_selector
from pygwrx.core.base import BaseSpatialRegressor
from pygwrx.core.kernels import get_kernel_function
from pygwrx.core.metrics import compute_diagnostics
from pygwrx.core.solver import (
    _weighted_least_squares_details,
    adaptive_bandwidth_weights,
)
from pygwrx.core.utils import add_intercept, compute_distance_matrix, validate_coords


@dataclass(frozen=True)
class GWRPredictionResult:
    """Rich prediction result returned by :meth:`GWR.predict_result`."""

    predictions: np.ndarray
    coef: np.ndarray
    intercept: np.ndarray
    coords: np.ndarray
    feature_names: Tuple[str, ...]
    coef_standard_errors: Optional[np.ndarray] = None
    intercept_standard_errors: Optional[np.ndarray] = None
    coef_t_values: Optional[np.ndarray] = None
    intercept_t_values: Optional[np.ndarray] = None

    def to_frame(self) -> pd.DataFrame:
        data: Dict[str, np.ndarray] = {
            "coord_0": self.coords[:, 0],
            "coord_1": self.coords[:, 1],
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
class _LocalFitResult:
    params: np.ndarray
    fitted_values: np.ndarray
    distances: np.ndarray
    influence: np.ndarray
    trace_S: float
    trace_StS: float
    covariance_factors: Optional[np.ndarray]
    hat_matrix: Optional[np.ndarray]
    local_rank: np.ndarray
    local_condition_number: np.ndarray


class GWR(BaseSpatialRegressor):
    """Gaussian geographically weighted regression.

    At each target location :math:`s_i`, the model estimates

    .. math::

        \\hat\\beta(s_i) = (X^T W_i X)^{-1}X^T W_i y,

    where ``W_i`` is produced by a spatial kernel. A fixed bandwidth is a
    distance; an adaptive bandwidth is an integer neighbour count.

    Args:
        kernel: Kernel name or callable accepting ``(distances, bandwidth)``.
        bandwidth: Numeric bandwidth or automatic-selection criterion. If ``None``,
            ``bandwidth_method`` is used.
        bandwidth_method: Criterion used only when ``bandwidth=None``.
        adaptive: Interpret the fitted bandwidth as an integer neighbour count.
        bandwidth_range: User-specified search interval. Adaptive bounds must be integers.
        optimization_method: One-dimensional search method used by automatic bandwidth selection.
        fit_intercept: Include a local intercept.
        distance_metric: Metric forwarded to the core distance implementation.
        sigma2_v1: Residual-variance convention. ``True`` uses ``RSS / (n - trace(S))``;
            ``False`` uses ``RSS / (n - 2 trace(S) + trace(S'S))``.
        verbose: Print fit progress.
    """

    def __init__(
        self,
        kernel: Union[str, Callable[[np.ndarray, float], np.ndarray]] = "gaussian",
        bandwidth: Union[float, int, str, None] = "cv",
        bandwidth_method: str = "cv",
        adaptive: bool = False,
        bandwidth_range: Optional[Tuple[float, float]] = None,
        optimization_method: str = "golden_section",
        fit_intercept: bool = True,
        distance_metric: str = "euclidean",
        sigma2_v1: bool = True,
        verbose: bool = False,
    ) -> None:
        if not isinstance(sigma2_v1, (bool, np.bool_)):
            raise TypeError("sigma2_v1 must be boolean.")
        if isinstance(bandwidth, str) and bandwidth.strip().lower() == "adaptive":
            raise ValueError(
                "GWR uses adaptive=True to request a nearest-neighbour bandwidth; "
                "bandwidth must be numeric, None, or one of 'cv', 'aic', 'aicc', 'bic'."
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
        self.sigma2_v1 = bool(sigma2_v1)
        self.S_matrix_: Optional[np.ndarray] = None
        self.bandwidth_search_: Optional[Dict[str, object]] = None
        self._reset_inference_state()

    def _reset_inference_state(self) -> None:
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
        self.local_rank_: Optional[np.ndarray] = None
        self.local_condition_number_: Optional[np.ndarray] = None
        self.rank_deficient_: Optional[np.ndarray] = None
        self.inference_enabled_: bool = False

    def _reset_fit_state(self) -> None:
        self._mark_unfitted()
        self._reset_regression_state()
        self._reset_gwr_state()
        self._reset_inference_state()
        self.S_matrix_ = None
        self.bandwidth_search_ = None

    def _resolve_bandwidth(
        self,
        X_design: np.ndarray,
        y: np.ndarray,
        coords: np.ndarray,
    ) -> Union[int, float]:
        if isinstance(self.bandwidth, str) or self.bandwidth is None:
            method = (
                self.bandwidth.strip().lower()
                if isinstance(self.bandwidth, str)
                else self.bandwidth_method.strip().lower()
            )
            if method not in {"cv", "aic", "aicc", "bic"}:
                raise ValueError(
                    "GWR bandwidth selection method must be one of "
                    "'cv', 'aic', 'aicc', or 'bic'."
                )
            selector = get_bandwidth_selector(
                method,
                adaptive=self.adaptive,
                verbose=self.verbose,
                optimization_method=self.optimization_method,
            )
            selected = selector.select(
                X_design,
                y,
                coords,
                self.kernel_func_,
                bandwidth_range=self.bandwidth_range,
                distance_metric=self.distance_metric,
            )
            selected_value: Union[int, float] = (
                int(selected) if self.adaptive else float(selected)
            )
            raw_range = getattr(selector, "search_range_", None)
            if raw_range is None:
                search_range = None
                boundary_solution = False
            elif self.adaptive:
                search_range = (int(raw_range[0]), int(raw_range[1]))
                boundary_solution = int(selected_value) in search_range
            else:
                search_range = (float(raw_range[0]), float(raw_range[1]))
                selected_float = float(selected_value)
                scale = max(
                    1.0,
                    abs(selected_float),
                    abs(search_range[0]),
                    abs(search_range[1]),
                )
                atol = 32.0 * np.finfo(float).eps * scale
                boundary_solution = bool(
                    np.isclose(
                        selected_float,
                        search_range[0],
                        rtol=1e-10,
                        atol=atol,
                    )
                    or np.isclose(
                        selected_float,
                        search_range[1],
                        rtol=1e-10,
                        atol=atol,
                    )
                )

            trace = tuple(getattr(selector, "search_trace_", ()))
            best_score = getattr(selector, "best_score_", None)
            selector_method = getattr(
                selector, "optimization_method", self.optimization_method
            )
            self.bandwidth_search_ = {
                "criterion": method,
                "adaptive": bool(self.adaptive),
                "optimization_method": (
                    "exhaustive_integer" if self.adaptive else selector_method
                ),
                "search_range": search_range,
                "selected": selected_value,
                "best_score": None if best_score is None else float(best_score),
                "trace": trace,
                "boundary_solution": bool(boundary_solution),
            }
            return selected_value

        value = float(self.bandwidth)
        if self.adaptive:
            if not value.is_integer():
                raise ValueError(
                    "adaptive bandwidth must be an integer neighbour count."
                )
            k = int(value)
            minimum = X_design.shape[1] + 1
            if k < minimum:
                raise ValueError(
                    f"adaptive bandwidth must be at least {minimum} for a design "
                    f"matrix with {X_design.shape[1]} columns."
                )
            if k > X_design.shape[0]:
                raise ValueError(
                    f"adaptive bandwidth cannot exceed n_samples={X_design.shape[0]}."
                )
            return k
        return value

    def _weights_from_distances(self, distances: np.ndarray) -> np.ndarray:
        if self.bandwidth_ is None or self.kernel_func_ is None:
            raise RuntimeError("The fitted bandwidth and kernel are unavailable.")
        local_bandwidth = (
            adaptive_bandwidth_weights(distances, int(self.bandwidth_))
            if self.adaptive
            else float(self.bandwidth_)
        )
        weights = np.asarray(self.kernel_func_(distances, local_bandwidth), dtype=float)
        if weights.shape != distances.shape:
            raise ValueError("The kernel returned an unexpected weight shape.")
        if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
            raise ValueError("The kernel returned invalid weights.")
        if not np.any(weights > 0.0):
            raise ValueError("The local kernel contains no positive weights.")
        return weights

    @staticmethod
    def _warn_rank_deficiency(
        rank_deficient: np.ndarray,
        *,
        context: str,
        n_parameters: int,
    ) -> None:
        indices = np.flatnonzero(rank_deficient)
        if indices.size == 0:
            return
        preview = ", ".join(str(int(index)) for index in indices[:5])
        suffix = ", ..." if indices.size > 5 else ""
        warnings.warn(
            f"{indices.size} {context} weighted design(s) are rank deficient for "
            f"{n_parameters} parameters (locations: {preview}{suffix}). Coefficients "
            "use the Moore-Penrose minimum-norm WLS solution; coefficient standard "
            "errors and t values are unavailable at rank-deficient locations.",
            RuntimeWarning,
            stacklevel=3,
        )

    def _fit_training_locations(
        self,
        X_design: np.ndarray,
        *,
        store_hat_matrix: bool,
        compute_inference: bool,
    ) -> _LocalFitResult:
        if self.coords_train_ is None or self.y_train_ is None:
            raise RuntimeError("Training data are unavailable.")

        distances = np.asarray(
            compute_distance_matrix(
                self.coords_train_,
                self.coords_train_,
                metric=self.distance_metric,
            ),
            dtype=float,
        )
        n_samples, n_parameters = X_design.shape
        params = np.empty((n_samples, n_parameters), dtype=float)
        fitted = np.empty(n_samples, dtype=float)
        influence = np.empty(n_samples, dtype=float)
        local_rank = np.empty(n_samples, dtype=int)
        local_condition_number = np.empty(n_samples, dtype=float)
        covariance_factors = (
            np.empty((n_samples, n_parameters), dtype=float)
            if compute_inference
            else None
        )
        hat_matrix = (
            np.empty((n_samples, n_samples), dtype=float) if store_hat_matrix else None
        )
        trace_sts = 0.0

        for index, distance_row in enumerate(distances):
            weights = self._weights_from_distances(distance_row)
            solve = _weighted_least_squares_details(
                X_design,
                self.y_train_,
                weights,
            )
            beta = solve.beta
            inverse_normal = solve.inverse_normal
            local_rank[index] = solve.rank
            local_condition_number[index] = solve.condition_number
            inverse_xtx_xtw = inverse_normal @ (X_design.T * weights)
            hat_row = X_design[index] @ inverse_xtx_xtw

            params[index] = beta
            fitted[index] = float(X_design[index] @ beta)
            influence[index] = float(hat_row[index])
            trace_sts += float(np.dot(hat_row, hat_row))
            if hat_matrix is not None:
                hat_matrix[index] = hat_row
            if covariance_factors is not None:
                if solve.rank < n_parameters:
                    covariance_factors[index] = np.nan
                else:
                    covariance_factors[index] = np.sum(
                        inverse_xtx_xtw**2,
                        axis=1,
                    )

        rank_deficient = local_rank < n_parameters
        self._warn_rank_deficiency(
            rank_deficient,
            context="calibration-location",
            n_parameters=n_parameters,
        )

        return _LocalFitResult(
            params=params,
            fitted_values=fitted,
            distances=distances,
            influence=influence,
            trace_S=float(np.sum(influence)),
            trace_StS=float(trace_sts),
            covariance_factors=covariance_factors,
            hat_matrix=hat_matrix,
            local_rank=local_rank,
            local_condition_number=local_condition_number,
        )

    def _compute_local_r2_from_distances(self, distances: np.ndarray) -> np.ndarray:
        if self.y_train_ is None or self.residuals_ is None:
            raise RuntimeError("Fitted values and residuals are unavailable.")
        local_r2 = np.full(self.y_train_.shape[0], np.nan, dtype=float)
        residual_sq = self.residuals_**2
        for index, distance_row in enumerate(distances):
            weights = self._weights_from_distances(distance_row)
            weight_sum = float(np.sum(weights))
            if weight_sum <= np.finfo(float).eps:
                continue
            local_mean = float(np.dot(weights, self.y_train_) / weight_sum)
            local_tss = float(np.dot(weights, (self.y_train_ - local_mean) ** 2))
            local_rss = float(np.dot(weights, residual_sq))
            if np.isclose(local_tss, 0.0, rtol=0.0, atol=np.finfo(float).eps):
                local_r2[index] = (
                    1.0
                    if np.isclose(local_rss, 0.0, rtol=0.0, atol=np.finfo(float).eps)
                    else 0.0
                )
            else:
                local_r2[index] = 1.0 - local_rss / local_tss
        return local_r2

    def _set_inference_results(
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
        self.influence_ = np.asarray(self.influence_, dtype=float)

        leverage_term = 1.0 - self.influence_
        self.standardized_residuals_ = np.full(self.n_samples_, np.nan, dtype=float)
        valid_leverage = leverage_term > np.finfo(float).eps
        if np.isfinite(self.sigma2_) and self.sigma2_ >= 0.0:
            self.standardized_residuals_[valid_leverage] = self.residuals_[
                valid_leverage
            ] / np.sqrt(self.sigma2_ * leverage_term[valid_leverage])

        self.cooks_distance_ = np.full(self.n_samples_, np.nan, dtype=float)
        if trace_S > np.finfo(float).eps:
            valid = leverage_term > np.finfo(float).eps
            self.cooks_distance_[valid] = (
                self.standardized_residuals_[valid] ** 2
                * self.influence_[valid]
                / (trace_S * leverage_term[valid])
            )

        if covariance_factors is None or not np.isfinite(self.sigma2_):
            return

        covariance_diagonal = covariance_factors * self.sigma2_
        covariance_diagonal = np.maximum(covariance_diagonal, 0.0)
        standard_errors = np.sqrt(covariance_diagonal)
        full_params = (
            np.column_stack([self.intercept_, self.coef_])
            if self.fit_intercept
            else self.coef_
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
        *,
        compute_hat_matrix: bool = True,
        compute_local_r2: bool = True,
        compute_inference: bool = True,
        compute_hat_matrix_flag: Optional[bool] = None,
        verbose: Optional[bool] = None,
    ) -> "GWR":
        """Fit the Gaussian GWR model and return ``self``.

        The smoother traces and influence values are always computed. Setting
        ``compute_hat_matrix=False`` avoids storing the full ``n x n`` matrix while
        retaining valid AIC/AICc/BIC, effective-parameter, residual-variance, and
        influence diagnostics.

        ``compute_hat_matrix_flag`` is retained as a compatibility alias for older
        PyGWRx code. New code should use ``compute_hat_matrix``.
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
        if not isinstance(self.sigma2_v1, (bool, np.bool_)):
            raise TypeError("sigma2_v1 must be boolean.")
        self._reset_fit_state()

        try:
            X_arr, y_arr, coords_arr = self._validate_inputs(X, y, coords)
            feature_names = (
                None
                if self.feature_names_in_ is None
                else self.feature_names_in_.copy()
            )
            self._store_training_data(X_arr, y_arr, coords_arr, copy=True)
            self.feature_names_in_ = feature_names

            X_design = (
                add_intercept(self.X_train_) if self.fit_intercept else self.X_train_
            )
            self.kernel_func_ = get_kernel_function(self.kernel)
            self.bandwidth_ = self._resolve_bandwidth(
                X_design, self.y_train_, self.coords_train_
            )

            if self.verbose:
                kind = "adaptive k" if self.adaptive else "fixed distance"
                print(f"Fitting GWR with {kind} bandwidth={self.bandwidth_}...")

            self.inference_enabled_ = bool(compute_inference)
            local_fit = self._fit_training_locations(
                X_design,
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
            self.residuals_ = self.y_train_ - self.fitted_values_
            self.influence_ = local_fit.influence.copy()
            self.local_rank_ = local_fit.local_rank.copy()
            self.local_condition_number_ = local_fit.local_condition_number.copy()
            self.rank_deficient_ = self.local_rank_ < X_design.shape[1]
            self.hat_matrix_ = local_fit.hat_matrix
            self.S_matrix_ = self.hat_matrix_  # compatibility alias
            self.diagnostics_ = compute_diagnostics(
                self.y_train_,
                self.fitted_values_,
                compute_gwr_stats=True,
                trace_S=local_fit.trace_S,
                trace_StS=local_fit.trace_StS,
            )

            self.local_r2_ = (
                self._compute_local_r2_from_distances(local_fit.distances)
                if compute_local_r2
                else None
            )
            self._set_inference_results(
                local_fit.covariance_factors,
                trace_S=local_fit.trace_S,
                trace_StS=local_fit.trace_StS,
            )

            self._mark_fitted()
            return self
        except Exception:
            self._reset_fit_state()
            raise

    def predict(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        coords: Union[np.ndarray, pd.DataFrame],
    ) -> np.ndarray:
        return self.predict_result(X, coords).predictions

    def _prediction_parameters(
        self, coords: Union[np.ndarray, pd.DataFrame]
    ) -> Dict[str, Optional[np.ndarray]]:
        self._check_is_fitted()
        if (
            self.X_train_ is None
            or self.y_train_ is None
            or self.coords_train_ is None
            or self.bandwidth_ is None
            or self.kernel_func_ is None
        ):
            raise RuntimeError("Stored training state is incomplete.")

        coords_arr = validate_coords(coords)
        X_design = add_intercept(self.X_train_) if self.fit_intercept else self.X_train_
        distances = compute_distance_matrix(
            coords_arr, self.coords_train_, metric=self.distance_metric
        )
        full_params = np.empty((coords_arr.shape[0], X_design.shape[1]), dtype=float)
        local_rank = np.empty(coords_arr.shape[0], dtype=int)
        local_condition_number = np.empty(coords_arr.shape[0], dtype=float)
        covariance_factors = (
            np.empty_like(full_params) if self.inference_enabled_ else None
        )
        for index, distance_row in enumerate(distances):
            weights = self._weights_from_distances(distance_row)
            solve = _weighted_least_squares_details(
                X_design,
                self.y_train_,
                weights,
            )
            inverse_xtx_xtw = solve.inverse_normal @ (X_design.T * weights)
            full_params[index] = solve.beta
            local_rank[index] = solve.rank
            local_condition_number[index] = solve.condition_number
            if covariance_factors is not None:
                if solve.rank < X_design.shape[1]:
                    covariance_factors[index] = np.nan
                else:
                    covariance_factors[index] = np.sum(
                        inverse_xtx_xtw**2,
                        axis=1,
                    )

        rank_deficient = local_rank < X_design.shape[1]
        self._warn_rank_deficiency(
            rank_deficient,
            context="prediction-location",
            n_parameters=X_design.shape[1],
        )

        if self.fit_intercept:
            intercept = full_params[:, 0]
            coef = full_params[:, 1:]
        else:
            intercept = np.zeros(coords_arr.shape[0], dtype=float)
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
            "coords": coords_arr,
            "coef": coef,
            "intercept": intercept,
            "standard_errors": standard_errors,
            "t_values": t_values,
            "local_rank": local_rank,
            "local_condition_number": local_condition_number,
            "rank_deficient": rank_deficient,
        }

    def predict_result(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        coords: Union[np.ndarray, pd.DataFrame],
    ) -> GWRPredictionResult:
        X_arr, coords_arr = self._validate_prediction_inputs(X, coords)
        params = self._prediction_parameters(coords_arr)
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

        return GWRPredictionResult(
            predictions=np.asarray(predictions, dtype=float),
            coef=coef,
            intercept=intercept,
            coords=np.asarray(coords_arr, dtype=float),
            feature_names=names,
            coef_standard_errors=coef_se,
            intercept_standard_errors=intercept_se,
            coef_t_values=coef_t,
            intercept_t_values=intercept_t,
        )

    def get_local_parameters(
        self, coords: Union[np.ndarray, pd.DataFrame]
    ) -> Dict[str, np.ndarray]:
        """Return local intercepts and slopes at arbitrary coordinates."""
        params = self._prediction_parameters(coords)
        return {
            "intercept": np.asarray(params["intercept"], dtype=float).copy(),
            "coef": np.asarray(params["coef"], dtype=float).copy(),
            "coords": np.asarray(params["coords"], dtype=float).copy(),
            "local_rank": np.asarray(params["local_rank"], dtype=int).copy(),
            "local_condition_number": np.asarray(
                params["local_condition_number"],
                dtype=float,
            ).copy(),
            "rank_deficient": np.asarray(
                params["rank_deficient"],
                dtype=bool,
            ).copy(),
        }

    def get_local_coefficients(
        self, coords: Union[np.ndarray, pd.DataFrame]
    ) -> np.ndarray:
        """Compatibility helper returning slopes only."""
        return self.get_local_parameters(coords)["coef"]

    def to_frame(self) -> pd.DataFrame:
        frame = super().to_frame()
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
            ("local_rank", self.local_rank_),
            ("local_condition_number", self.local_condition_number_),
            ("rank_deficient", self.rank_deficient_),
        ):
            if values is not None:
                frame[name] = values
        return frame

    def summary(self) -> str:
        """Return a stable text summary of global and local model results."""
        self._check_is_fitted()
        if self.X_train_ is None or self.y_train_ is None:
            raise RuntimeError("Training data are unavailable.")

        X_global = add_intercept(self.X_train_) if self.fit_intercept else self.X_train_
        global_beta = np.linalg.lstsq(X_global, self.y_train_, rcond=None)[0]
        global_fitted = X_global @ global_beta
        global_residuals = self.y_train_ - global_fitted
        global_rss = float(np.dot(global_residuals, global_residuals))
        n, p = X_global.shape
        global_df = max(n - p, 1)
        global_sigma2 = global_rss / global_df
        covariance = global_sigma2 * np.linalg.pinv(X_global.T @ X_global)
        global_se = np.sqrt(np.maximum(np.diag(covariance), 0.0))

        feature_names = (
            [str(name) for name in self.feature_names_in_]
            if self.feature_names_in_ is not None
            else [f"x{index}" for index in range(self.X_train_.shape[1])]
        )
        global_names = (["intercept"] if self.fit_intercept else []) + feature_names
        local_matrix = (
            np.column_stack([self.intercept_, self.coef_])
            if self.fit_intercept
            else np.asarray(self.coef_)
        )

        bandwidth_lines = [
            f"Bandwidth: {self.bandwidth_} ({'adaptive neighbours' if self.adaptive else 'fixed distance'})"
        ]
        if self.bandwidth_search_ is not None:
            search_range = self.bandwidth_search_.get("search_range")
            bandwidth_lines.extend(
                [
                    f"Bandwidth criterion: {self.bandwidth_search_.get('criterion')}",
                    f"Bandwidth search method: {self.bandwidth_search_.get('optimization_method')}",
                    f"Bandwidth search range: {search_range}",
                    f"Bandwidth boundary solution: {self.bandwidth_search_.get('boundary_solution')}",
                ]
            )

        rank_deficient_count = (
            int(np.count_nonzero(self.rank_deficient_))
            if self.rank_deficient_ is not None
            else 0
        )

        lines = [
            "=" * 78,
            "Gaussian Geographically Weighted Regression (GWR)",
            "=" * 78,
            f"Samples: {n}",
            f"Predictors: {self.X_train_.shape[1]}",
            f"Kernel: {self.kernel}",
            *bandwidth_lines,
            f"Distance metric: {self.distance_metric}",
            f"Rank-deficient local fits: {rank_deficient_count}/{n}",
            f"Residual variance (sigma^2): {self.sigma2_:.6f}",
            "",
            "Global OLS reference",
            "-" * 78,
            f"{'Variable':<22}{'Estimate':>14}{'Std. Error':>16}",
        ]
        for name, estimate, standard_error in zip(global_names, global_beta, global_se):
            lines.append(f"{name:<22}{estimate:>14.6f}{standard_error:>16.6f}")

        lines.extend(
            [
                "",
                "Local coefficient distribution",
                "-" * 78,
                f"{'Variable':<22}{'Min':>12}{'Median':>12}{'Mean':>12}{'Max':>12}",
            ]
        )
        for index, name in enumerate(global_names):
            values = local_matrix[:, index]
            lines.append(
                f"{name:<22}{np.min(values):>12.6f}{np.median(values):>12.6f}"
                f"{np.mean(values):>12.6f}{np.max(values):>12.6f}"
            )

        lines.extend(["", "GWR diagnostics", "-" * 78])
        ordered = (
            ("R-squared", "r2"),
            ("Adjusted R-squared", "adj_r2"),
            ("RSS", "rss"),
            ("RMSE", "rmse"),
            ("MAE", "mae"),
            ("AIC", "aic"),
            ("AICc", "aicc"),
            ("BIC", "bic"),
            ("trace(S) / ENP v1", "trace_S"),
            ("trace(S'S)", "trace_StS"),
            ("ENP v2", "enp_v2"),
            ("EDF v2", "edf_v2"),
        )
        for label, key in ordered:
            value = self.diagnostics_.get(key, np.nan) if self.diagnostics_ else np.nan
            lines.append(f"{label:<30}{value:>14.6f}")
        lines.append("=" * 78)
        return "\n".join(lines)


__all__ = ["GWR", "GWRPredictionResult"]
