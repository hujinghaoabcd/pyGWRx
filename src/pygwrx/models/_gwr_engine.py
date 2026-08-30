# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Private execution engine for standard Gaussian GWR.

This module owns the model-specific orchestration extracted from
:mod:`pygwrx.models.gwr`.  It deliberately remains private: public estimator
contracts stay on :class:`pygwrx.models.gwr.GWR`, while generic numerical
primitives remain in :mod:`pygwrx.core`.

The engine does not define a new estimator base class.  Distance generation,
weight construction and rank policy are injected by the owning estimator so
that robust, temporal or other model-specific semantics are not silently
folded into standard GWR.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Iterator, Optional, Protocol

import numpy as np

from pygwrx.core.bandwidth import (
    Bandwidth,
    BandwidthRange,
    KernelFunction,
    _BaseSelector,
    _fit_local_model,
    _kernel_weights,
)
from pygwrx.core.distance import _iter_distance_rows
from pygwrx.core.gaussian_diagnostics import compute_aic, compute_aicc, compute_bic
from pygwrx.core.solver import (
    _solve_weighted_least_squares,
    adaptive_bandwidth_weights,
)


class _DistanceRowsProvider(Protocol):
    """Yield target-to-training distance rows under estimator-owned geometry."""

    def __call__(self, target_coords: np.ndarray) -> Iterator[np.ndarray]: ...


class _WeightRowProvider(Protocol):
    """Return one estimator-defined row of observation weights."""

    def __call__(self, distances: np.ndarray) -> np.ndarray: ...


class _RankPolicy(Protocol):
    """Apply estimator-owned behavior after local numerical-rank evaluation."""

    def __call__(
        self,
        rank_deficient: np.ndarray,
        *,
        context: str,
        n_parameters: int,
    ) -> None: ...


@dataclass(frozen=True)
class _GWRLocalFitResult:
    """Private calibration result produced by the Gaussian local-fit engine."""

    params: np.ndarray
    fitted_values: np.ndarray
    distances: Iterable[np.ndarray]
    influence: np.ndarray
    trace_S: float
    trace_StS: float
    covariance_factors: Optional[np.ndarray]
    hat_matrix: Optional[np.ndarray]
    local_rank: np.ndarray
    local_condition_number: np.ndarray


@dataclass(frozen=True)
class _GWRPredictionFitResult:
    """Private local-parameter result for arbitrary prediction locations."""

    full_params: np.ndarray
    covariance_factors: Optional[np.ndarray]
    local_rank: np.ndarray
    local_condition_number: np.ndarray
    rank_deficient: np.ndarray


@dataclass(frozen=True)
class _GWRInferenceResult:
    """Private inference arrays derived from one completed GWR calibration."""

    influence: np.ndarray
    sigma2: float
    standardized_residuals: np.ndarray
    cooks_distance: np.ndarray
    parameter_covariance_diagonal: Optional[np.ndarray]
    parameter_standard_errors: Optional[np.ndarray]
    parameter_t_values: Optional[np.ndarray]
    intercept_se: Optional[np.ndarray]
    coef_se: Optional[np.ndarray]
    intercept_t: Optional[np.ndarray]
    coef_t: Optional[np.ndarray]


class _GWRBandwidthSelector(_BaseSelector):
    """Model-owned standard-GWR bandwidth objective with legacy search semantics."""

    _VALID_CRITERIA = {"cv", "aic", "aicc", "bic"}

    def __init__(
        self,
        criterion: str,
        *,
        n_intervals: int = 20,
        optimization_method: str = "golden_section",
        adaptive: bool = False,
        verbose: bool = False,
    ) -> None:
        if not isinstance(criterion, str):
            raise TypeError("criterion must be a string.")
        normalized = criterion.strip().lower()
        if normalized not in self._VALID_CRITERIA:
            raise ValueError(
                "criterion must be one of 'cv', 'aic', 'aicc', or 'bic'."
            )
        super().__init__(
            n_intervals=n_intervals,
            optimization_method=optimization_method,
            adaptive=adaptive,
            verbose=verbose,
        )
        self.criterion = normalized

    def select(
        self,
        X: np.ndarray,
        y: np.ndarray,
        coords: np.ndarray,
        kernel_func: KernelFunction,
        bandwidth_range: BandwidthRange = None,
        distance_metric: str = "euclidean",
    ) -> Bandwidth:
        X_arr, y_arr, coords_arr, lower, upper = self._prepare(
            X,
            y,
            coords,
            kernel_func,
            bandwidth_range,
            distance_metric,
        )

        if self.criterion == "cv":
            self._print_header("Cross-Validation Bandwidth Selection", lower, upper)

            def objective(bandwidth: Bandwidth) -> float:
                squared_error = 0.0
                for i, dists in enumerate(
                    _iter_distance_rows(coords_arr, distance_metric=distance_metric)
                ):
                    weights = _kernel_weights(
                        dists,
                        bandwidth,
                        adaptive=self.adaptive,
                        kernel_func=kernel_func,
                    ).copy()
                    weights[i] = 0.0
                    beta, _ = _fit_local_model(X_arr, y_arr, weights)
                    residual = float(y_arr[i] - X_arr[i] @ beta)
                    squared_error += residual * residual
                return squared_error

            label = "CV"
        elif self.criterion in {"aic", "aicc"}:
            corrected = self.criterion == "aicc"
            label = "AICc" if corrected else "AIC"
            self._print_header(f"{label} Bandwidth Selection", lower, upper)

            def objective(bandwidth: Bandwidth) -> float:
                n_samples = y_arr.size
                fitted = np.empty(n_samples, dtype=float)
                trace_s = 0.0

                for i, dists in enumerate(
                    _iter_distance_rows(coords_arr, distance_metric=distance_metric)
                ):
                    weights = _kernel_weights(
                        dists,
                        bandwidth,
                        adaptive=self.adaptive,
                        kernel_func=kernel_func,
                    )
                    beta, hat_row = _fit_local_model(
                        X_arr,
                        y_arr,
                        weights,
                        target_row=X_arr[i],
                    )
                    assert hat_row is not None
                    fitted[i] = X_arr[i] @ beta
                    trace_s += float(hat_row[i])

                if corrected:
                    return float(compute_aicc(y_arr, fitted, trace_s))
                return float(compute_aic(y_arr, fitted, trace_s))

        else:
            label = "BIC"
            self._print_header("BIC Bandwidth Selection", lower, upper)

            def objective(bandwidth: Bandwidth) -> float:
                n_samples = y_arr.size
                fitted = np.empty(n_samples, dtype=float)
                trace_s = 0.0

                for i, dists in enumerate(
                    _iter_distance_rows(coords_arr, distance_metric=distance_metric)
                ):
                    weights = _kernel_weights(
                        dists,
                        bandwidth,
                        adaptive=self.adaptive,
                        kernel_func=kernel_func,
                    )
                    beta, hat_row = _fit_local_model(
                        X_arr,
                        y_arr,
                        weights,
                        target_row=X_arr[i],
                    )
                    assert hat_row is not None
                    fitted[i] = X_arr[i] @ beta
                    trace_s += float(hat_row[i])

                return float(compute_bic(y_arr, fitted, trace_s))

        best_bandwidth, best_score = self._search(objective, lower, upper)
        if self.verbose:
            bandwidth_label = "Optimal k" if self.adaptive else "Optimal bandwidth"
            print(f"\n{bandwidth_label}: {best_bandwidth}")
            print(f"{label} score: {best_score:.6f}")
        return best_bandwidth


def _get_gwr_bandwidth_selector(
    method: str,
    *,
    adaptive: bool,
    verbose: bool,
    optimization_method: str,
) -> _GWRBandwidthSelector:
    """Create the private model-owned selector used by standard GWR."""
    return _GWRBandwidthSelector(
        method,
        adaptive=adaptive,
        verbose=verbose,
        optimization_method=optimization_method,
    )


def _gwr_spatial_weights(
    distances: np.ndarray,
    *,
    bandwidth: Bandwidth,
    adaptive: bool,
    kernel_func: KernelFunction,
) -> np.ndarray:
    """Construct one standard-GWR spatial weight row without owning geometry."""
    local_bandwidth = (
        adaptive_bandwidth_weights(distances, int(bandwidth))
        if adaptive
        else float(bandwidth)
    )
    weights = np.asarray(kernel_func(distances, local_bandwidth), dtype=float)
    if weights.shape != distances.shape:
        raise ValueError("The kernel returned an unexpected weight shape.")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("The kernel returned invalid weights.")
    if not np.any(weights > 0.0):
        raise ValueError("The local kernel contains no positive weights.")
    return weights


def _fit_gwr_training_locations(
    X_design: np.ndarray,
    y_train: np.ndarray,
    coords_train: np.ndarray,
    *,
    distance_rows: _DistanceRowsProvider,
    weights_from_distances: _WeightRowProvider,
    rank_policy: _RankPolicy,
    store_hat_matrix: bool,
    compute_inference: bool,
) -> _GWRLocalFitResult:
    """Calibrate Gaussian local WLS at every training location."""
    n_samples, n_parameters = X_design.shape
    params = np.empty((n_samples, n_parameters), dtype=float)
    fitted = np.empty(n_samples, dtype=float)
    influence = np.empty(n_samples, dtype=float)
    local_rank = np.empty(n_samples, dtype=int)
    local_condition_number = np.empty(n_samples, dtype=float)
    covariance_factors = (
        np.empty((n_samples, n_parameters), dtype=float) if compute_inference else None
    )
    hat_matrix = (
        np.empty((n_samples, n_samples), dtype=float) if store_hat_matrix else None
    )
    trace_sts = 0.0

    for index, distance_row in enumerate(distance_rows(coords_train)):
        weights = weights_from_distances(distance_row)
        solve = _solve_weighted_least_squares(X_design, y_train, weights)
        beta = solve.params
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
    rank_policy(
        rank_deficient,
        context="calibration-location",
        n_parameters=n_parameters,
    )

    return _GWRLocalFitResult(
        params=params,
        fitted_values=fitted,
        distances=distance_rows(coords_train),
        influence=influence,
        trace_S=float(np.sum(influence)),
        trace_StS=float(trace_sts),
        covariance_factors=covariance_factors,
        hat_matrix=hat_matrix,
        local_rank=local_rank,
        local_condition_number=local_condition_number,
    )


def _compute_gwr_local_r2(
    y_train: np.ndarray,
    residuals: np.ndarray,
    distance_rows: Iterable[np.ndarray],
    *,
    weights_from_distances: _WeightRowProvider,
) -> np.ndarray:
    """Compute standard-GWR local R² from estimator-defined weight rows."""
    local_r2 = np.full(y_train.shape[0], np.nan, dtype=float)
    residual_sq = residuals**2
    for index, distance_row in enumerate(distance_rows):
        weights = weights_from_distances(distance_row)
        weight_sum = float(np.sum(weights))
        if weight_sum <= np.finfo(float).eps:
            continue
        local_mean = float(np.dot(weights, y_train) / weight_sum)
        local_tss = float(np.dot(weights, (y_train - local_mean) ** 2))
        local_rss = float(np.dot(weights, residual_sq))
        if np.isclose(local_tss, 0.0, rtol=0.0, atol=np.finfo(float).eps):
            local_r2[index] = (
                1.0
                if np.isclose(
                    local_rss,
                    0.0,
                    rtol=0.0,
                    atol=np.finfo(float).eps,
                )
                else 0.0
            )
        else:
            local_r2[index] = 1.0 - local_rss / local_tss
    return local_r2


def _collect_gwr_inference(
    residuals: np.ndarray,
    influence: np.ndarray,
    coef: np.ndarray,
    intercept: np.ndarray,
    covariance_factors: Optional[np.ndarray],
    *,
    n_samples: int,
    fit_intercept: bool,
    sigma2_v1: bool,
    trace_S: float,
    trace_StS: float,
) -> _GWRInferenceResult:
    """Collect standard Gaussian GWR residual and coefficient inference arrays."""
    rss = float(np.dot(residuals, residuals))
    denominator = (
        n_samples - trace_S
        if sigma2_v1
        else n_samples - 2.0 * trace_S + trace_StS
    )
    sigma2 = rss / denominator if denominator > 0.0 else np.nan
    influence_arr = np.asarray(influence, dtype=float)

    leverage_term = 1.0 - influence_arr
    standardized_residuals = np.full(n_samples, np.nan, dtype=float)
    valid_leverage = leverage_term > np.finfo(float).eps
    if np.isfinite(sigma2) and sigma2 > np.finfo(float).eps:
        standardized_residuals[valid_leverage] = residuals[valid_leverage] / np.sqrt(
            sigma2 * leverage_term[valid_leverage]
        )

    cooks_distance = np.full(n_samples, np.nan, dtype=float)
    if trace_S > np.finfo(float).eps:
        valid = leverage_term > np.finfo(float).eps
        cooks_distance[valid] = (
            standardized_residuals[valid] ** 2
            * influence_arr[valid]
            / (trace_S * leverage_term[valid])
        )

    if covariance_factors is None or not np.isfinite(sigma2):
        return _GWRInferenceResult(
            influence=influence_arr,
            sigma2=float(sigma2),
            standardized_residuals=standardized_residuals,
            cooks_distance=cooks_distance,
            parameter_covariance_diagonal=None,
            parameter_standard_errors=None,
            parameter_t_values=None,
            intercept_se=None,
            coef_se=None,
            intercept_t=None,
            coef_t=None,
        )

    covariance_diagonal = covariance_factors * sigma2
    covariance_diagonal = np.maximum(covariance_diagonal, 0.0)
    standard_errors = np.sqrt(covariance_diagonal)
    full_params = (
        np.column_stack([intercept, coef]) if fit_intercept else np.asarray(coef)
    )
    t_values = np.full_like(full_params, np.nan, dtype=float)
    np.divide(
        full_params,
        standard_errors,
        out=t_values,
        where=standard_errors > np.finfo(float).eps,
    )

    if fit_intercept:
        intercept_se = standard_errors[:, 0]
        coef_se = standard_errors[:, 1:]
        intercept_t = t_values[:, 0]
        coef_t = t_values[:, 1:]
    else:
        intercept_se = np.zeros(n_samples, dtype=float)
        coef_se = standard_errors
        intercept_t = np.full(n_samples, np.nan, dtype=float)
        coef_t = t_values

    return _GWRInferenceResult(
        influence=influence_arr,
        sigma2=float(sigma2),
        standardized_residuals=standardized_residuals,
        cooks_distance=cooks_distance,
        parameter_covariance_diagonal=covariance_diagonal,
        parameter_standard_errors=standard_errors,
        parameter_t_values=t_values,
        intercept_se=intercept_se,
        coef_se=coef_se,
        intercept_t=intercept_t,
        coef_t=coef_t,
    )


def _fit_gwr_prediction_locations(
    X_design: np.ndarray,
    y_train: np.ndarray,
    target_coords: np.ndarray,
    *,
    distance_rows: _DistanceRowsProvider,
    weights_from_distances: _WeightRowProvider,
    rank_policy: _RankPolicy,
    compute_inference: bool,
) -> _GWRPredictionFitResult:
    """Estimate local Gaussian parameters at arbitrary target locations."""
    full_params = np.empty((target_coords.shape[0], X_design.shape[1]), dtype=float)
    local_rank = np.empty(target_coords.shape[0], dtype=int)
    local_condition_number = np.empty(target_coords.shape[0], dtype=float)
    covariance_factors = np.empty_like(full_params) if compute_inference else None

    for index, distance_row in enumerate(distance_rows(target_coords)):
        weights = weights_from_distances(distance_row)
        solve = _solve_weighted_least_squares(X_design, y_train, weights)
        full_params[index] = solve.params
        local_rank[index] = solve.rank
        local_condition_number[index] = solve.condition_number
        if covariance_factors is not None:
            if solve.rank < X_design.shape[1]:
                covariance_factors[index] = np.nan
            else:
                inverse_xtx_xtw = solve.inverse_normal @ (X_design.T * weights)
                covariance_factors[index] = np.sum(
                    inverse_xtx_xtw**2,
                    axis=1,
                )

    rank_deficient = local_rank < X_design.shape[1]
    rank_policy(
        rank_deficient,
        context="prediction-location",
        n_parameters=X_design.shape[1],
    )
    return _GWRPredictionFitResult(
        full_params=full_params,
        covariance_factors=covariance_factors,
        local_rank=local_rank,
        local_condition_number=local_condition_number,
        rank_deficient=rank_deficient,
    )
