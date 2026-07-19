# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Multiscale geographically weighted regression.

This module implements Gaussian multiscale geographically weighted regression
(MGWR) with variable-specific bandwidths, iterative backfitting, exact smoother
trace diagnostics, and local parameter inference.

Author:
    Jinghao Hu
"""

from __future__ import annotations

__author__ = "Jinghao Hu"
__license__ = "MIT"

import warnings
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from pygwrx.core.bandwidth import get_bandwidth_selector
from pygwrx.core.base import BaseMultiscaleRegressor
from pygwrx.core.kernels import get_kernel_function
from pygwrx.core.metrics import (
    compute_aic,
    compute_aicc,
    compute_bic,
    compute_diagnostics,
)
from pygwrx.core.solver import adaptive_bandwidth_weights, weighted_least_squares
from pygwrx.core.utils import add_intercept, compute_distance_matrix

Bandwidth = Union[int, float]
BandwidthInput = Optional[Union[Bandwidth, Sequence[Bandwidth]]]
BandwidthRange = Optional[Tuple[float, float]]
BandwidthRanges = Optional[
    Union[BandwidthRange, Sequence[Optional[Tuple[float, float]]]]
]


@dataclass
class _BackfittingResult:
    """Internal result produced by the MGWR backfitting routine."""

    params: np.ndarray
    contributions: np.ndarray
    residuals: np.ndarray
    bandwidths: np.ndarray
    bandwidth_history: np.ndarray
    convergence_history: np.ndarray
    n_iter: int
    converged: bool
    initial_bandwidth: Bandwidth


@dataclass
class _InferenceResult:
    """Internal exact MGWR smoother and covariance diagnostics."""

    effective_params_by_variable: np.ndarray
    covariance_factors: Optional[np.ndarray]
    influence: np.ndarray
    trace_S: float
    trace_StS: float
    hat_matrix: Optional[np.ndarray]
    partial_hat_matrices: Optional[np.ndarray]


class MGWR(BaseMultiscaleRegressor):
    """Gaussian multiscale geographically weighted regression.

    MGWR represents the response as a sum of spatially varying additive terms,
    with one bandwidth for the intercept and each predictor when an intercept is
    fitted. The model is calibrated by iteratively updating one term at a time
    while holding the remaining additive terms fixed.

    Args:
        kernel: Kernel name or callable accepting ``(distances, bandwidth)``.
        bandwidths: Optional manual bandwidth or sequence of bandwidths. A
            scalar is applied to every parameter. A sequence must contain one
            value per fitted parameter, including the intercept when present.
            If ``None``, variable-specific bandwidths are selected by
            backfitting.
        bandwidth_method: Criterion used for the initial GWR bandwidth and each
            variable-specific bandwidth search. Supported values are ``"cv"``,
            ``"aic"``, ``"aicc"``, and ``"bic"``.
        adaptive: Interpret bandwidths as integer nearest-neighbour counts.
        bandwidth_range: Optional common search range for all parameters.
        bandwidth_ranges: Optional parameter-specific search ranges. Supply one
            range per fitted parameter, including the intercept when present.
        init_bandwidth: Optional bandwidth for the initial single-bandwidth GWR
            fit. If ``None``, it is selected automatically.
        optimization_method: One-dimensional bandwidth search method.
        search_tol: Convergence tolerance used by variable-specific bandwidth searches.
        search_max_iter: Maximum iterations for each variable-specific bandwidth search.
        max_iter: Maximum number of backfitting iterations.
        tol: Score-of-change convergence tolerance.
        rss_score: Use relative RSS change instead of smoothing-function change
            as the convergence score.
        bws_same_times: Stop repeating bandwidth searches after the complete
            bandwidth vector remains unchanged for this many iterations.
        fit_intercept: Include a spatially varying intercept.
        distance_metric: Distance metric used to form spatial neighbourhoods.
        sigma2_v1: Residual-variance convention. ``True`` uses
            ``RSS / (n - trace(S))``; ``False`` uses
            ``RSS / (n - 2 trace(S) + trace(S'S))``.
        verbose: Print backfitting progress.

    Attributes:
        bandwidths_: Final variable-specific bandwidth vector.
        bandwidth_history_: Bandwidth vector from every backfitting iteration.
        convergence_history_: Score of change from every iteration.
        initial_bandwidth_: Initial single-bandwidth GWR bandwidth.
        effective_params_by_variable_: Effective parameter count for each
            coefficient surface.
        coef_: Local slope estimates with shape
            ``(n_samples, n_features)``.
        intercept_: Local intercept estimates with shape ``(n_samples,)``.
        parameter_standard_errors_: Local standard errors for all fitted
            parameters.
        parameter_t_values_: Local t statistics for all fitted parameters.
        converged_: Whether the backfitting score reached ``tol``.
        n_iter_: Number of completed backfitting iterations.

    Notes:
        Out-of-sample MGWR prediction is intentionally not exposed because the
        widely used reference Python implementation does not provide a validated
        prediction algorithm for independently supplied target locations. Use
        ``fitted_values_`` for calibration-location estimates.

    References:
        Fotheringham, A. S., Yang, W., and Kang, W. (2017). Multiscale
        geographically weighted regression (MGWR). *Annals of the American
        Association of Geographers*, 107(6), 1247-1265.
    """

    def __init__(
        self,
        kernel: Union[str, Callable[[np.ndarray, float], np.ndarray]] = "bisquare",
        bandwidths: BandwidthInput = None,
        bandwidth_method: str = "aicc",
        adaptive: bool = True,
        bandwidth_range: BandwidthRange = None,
        bandwidth_ranges: BandwidthRanges = None,
        init_bandwidth: Optional[Bandwidth] = None,
        optimization_method: str = "golden_section",
        search_tol: float = 1e-6,
        search_max_iter: int = 200,
        max_iter: int = 200,
        tol: float = 1e-5,
        rss_score: bool = False,
        bws_same_times: int = 5,
        fit_intercept: bool = True,
        distance_metric: str = "euclidean",
        sigma2_v1: bool = True,
        verbose: bool = False,
    ) -> None:
        if not isinstance(
            bandwidth_method, str
        ) or bandwidth_method.strip().lower() not in {
            "cv",
            "aic",
            "aicc",
            "bic",
        }:
            raise ValueError(
                "bandwidth_method must be one of 'cv', 'aic', 'aicc', or 'bic'."
            )
        if not isinstance(
            search_tol, (int, float, np.integer, np.floating)
        ) or isinstance(search_tol, (bool, np.bool_)):
            raise TypeError("search_tol must be a positive finite number.")
        if not np.isfinite(float(search_tol)) or float(search_tol) <= 0.0:
            raise ValueError("search_tol must be a positive finite number.")
        if not isinstance(search_max_iter, (int, np.integer)) or isinstance(
            search_max_iter, (bool, np.bool_)
        ):
            raise TypeError("search_max_iter must be an integer.")
        if int(search_max_iter) < 1:
            raise ValueError("search_max_iter must be at least 1.")
        if not isinstance(max_iter, (int, np.integer)) or isinstance(
            max_iter, (bool, np.bool_)
        ):
            raise TypeError("max_iter must be an integer.")
        if int(max_iter) < 1:
            raise ValueError("max_iter must be at least 1.")
        if not isinstance(tol, (int, float, np.integer, np.floating)) or isinstance(
            tol, (bool, np.bool_)
        ):
            raise TypeError("tol must be a positive finite number.")
        if not np.isfinite(float(tol)) or float(tol) <= 0.0:
            raise ValueError("tol must be a positive finite number.")
        if not isinstance(rss_score, (bool, np.bool_)):
            raise TypeError("rss_score must be boolean.")
        if not isinstance(bws_same_times, (int, np.integer)) or isinstance(
            bws_same_times, (bool, np.bool_)
        ):
            raise TypeError("bws_same_times must be an integer.")
        if int(bws_same_times) < 0:
            raise ValueError("bws_same_times must be non-negative.")
        if not isinstance(sigma2_v1, (bool, np.bool_)):
            raise TypeError("sigma2_v1 must be boolean.")

        # BaseGWR's single bandwidth is used only to validate and store the
        # initial GWR setting. Final MGWR scales are exposed as bandwidths_.
        super().__init__(
            kernel=kernel,
            bandwidth=(
                init_bandwidth if init_bandwidth is not None else bandwidth_method
            ),
            bandwidth_method=bandwidth_method,
            adaptive=adaptive,
            bandwidth_range=bandwidth_range,
            optimization_method=optimization_method,
            fit_intercept=fit_intercept,
            distance_metric=distance_metric,
            verbose=verbose,
        )
        self.bandwidths = bandwidths
        self.bandwidth_ranges = bandwidth_ranges
        self.init_bandwidth = init_bandwidth
        self.search_tol = float(search_tol)
        self.search_max_iter = int(search_max_iter)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.rss_score = bool(rss_score)
        self.bws_same_times = int(bws_same_times)
        self.sigma2_v1 = bool(sigma2_v1)
        self._reset_mgwr_state()

    def _reset_mgwr_state(self) -> None:
        self.initial_bandwidth_: Optional[Bandwidth] = None
        self.parameter_contributions_: Optional[np.ndarray] = None
        self.effective_params_by_variable_: Optional[np.ndarray] = None
        self.ENP_j_: Optional[np.ndarray] = None
        self.n_iter_: int = 0
        self.converged_: bool = False
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
        self.adjusted_alpha_by_variable_: Optional[np.ndarray] = None
        self.critical_t_values_: Optional[np.ndarray] = None
        self.partial_hat_matrices_: Optional[np.ndarray] = None
        self.inference_enabled_: bool = False

    def _reset_fit_state(self) -> None:
        self._mark_unfitted()
        self._reset_regression_state()
        self._reset_gwr_state()
        self._reset_multiscale_state()
        self._reset_mgwr_state()

    def _validate_bandwidth_value(
        self,
        value: Bandwidth,
        *,
        n_samples: int,
        minimum_adaptive: int,
        name: str,
    ) -> Bandwidth:
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, float, np.integer, np.floating)
        ):
            raise TypeError(f"{name} must be numeric.")
        numeric = float(value)
        if not np.isfinite(numeric) or numeric <= 0.0:
            raise ValueError(f"{name} must be finite and greater than zero.")
        if self.adaptive:
            if not numeric.is_integer():
                raise ValueError(f"{name} must be an integer nearest-neighbour count.")
            neighbour_count = int(numeric)
            if neighbour_count < minimum_adaptive:
                raise ValueError(
                    f"{name} must be at least {minimum_adaptive}; "
                    f"received {neighbour_count}."
                )
            if neighbour_count > n_samples:
                raise ValueError(
                    f"{name} cannot exceed n_samples={n_samples}; "
                    f"received {neighbour_count}."
                )
            return neighbour_count
        return numeric

    def _resolve_manual_bandwidths(
        self,
        *,
        n_parameters: int,
        n_samples: int,
    ) -> Optional[np.ndarray]:
        if self.bandwidths is None:
            return None
        if isinstance(
            self.bandwidths, (int, float, np.integer, np.floating)
        ) and not isinstance(self.bandwidths, (bool, np.bool_)):
            values = [self.bandwidths] * n_parameters
        else:
            if isinstance(self.bandwidths, (str, bytes)):
                raise TypeError(
                    "bandwidths must be numeric, a numeric sequence, or None."
                )
            try:
                values = list(self.bandwidths)  # type: ignore[arg-type]
            except TypeError as error:
                raise TypeError(
                    "bandwidths must be numeric, a numeric sequence, or None."
                ) from error
            if len(values) != n_parameters:
                raise ValueError(
                    "bandwidths must contain one value per fitted parameter; "
                    f"expected {n_parameters}, received {len(values)}."
                )
        validated = [
            self._validate_bandwidth_value(
                value,
                n_samples=n_samples,
                minimum_adaptive=2,
                name=f"bandwidths[{index}]",
            )
            for index, value in enumerate(values)
        ]
        dtype = int if self.adaptive else float
        return np.asarray(validated, dtype=dtype)

    def _resolve_bandwidth_ranges(
        self,
        *,
        n_parameters: int,
        n_samples: int,
    ) -> List[BandwidthRange]:
        source = self.bandwidth_ranges
        if source is None:
            ranges: List[BandwidthRange] = [self.bandwidth_range] * n_parameters
        elif (
            isinstance(source, (tuple, list))
            and len(source) == 2
            and all(
                isinstance(value, (int, float, np.integer, np.floating))
                and not isinstance(value, (bool, np.bool_))
                for value in source
            )
        ):
            ranges = [tuple(source)] * n_parameters  # type: ignore[list-item]
        else:
            if isinstance(source, (str, bytes)):
                raise TypeError(
                    "bandwidth_ranges must be a range, a sequence of ranges, or None."
                )
            try:
                ranges = list(source)  # type: ignore[arg-type]
            except TypeError as error:
                raise TypeError(
                    "bandwidth_ranges must be a range, a sequence of ranges, or None."
                ) from error
            if len(ranges) != n_parameters:
                raise ValueError(
                    "bandwidth_ranges must contain one entry per fitted parameter; "
                    f"expected {n_parameters}, received {len(ranges)}."
                )

        normalized: List[BandwidthRange] = []
        for index, bandwidth_range in enumerate(ranges):
            if bandwidth_range is None:
                normalized.append(None)
                continue
            if (
                not isinstance(bandwidth_range, (tuple, list))
                or len(bandwidth_range) != 2
            ):
                raise TypeError(
                    f"bandwidth_ranges[{index}] must be a two-element range or None."
                )
            lower = self._validate_bandwidth_value(
                bandwidth_range[0],
                n_samples=n_samples,
                minimum_adaptive=2,
                name=f"bandwidth_ranges[{index}][0]",
            )
            upper = self._validate_bandwidth_value(
                bandwidth_range[1],
                n_samples=n_samples,
                minimum_adaptive=2,
                name=f"bandwidth_ranges[{index}][1]",
            )
            if float(lower) > float(upper):
                raise ValueError(
                    f"bandwidth_ranges[{index}] must satisfy lower <= upper."
                )
            normalized.append((float(lower), float(upper)))
        return normalized

    def _adaptive_weight_matrix(
        self,
        distances: np.ndarray,
        bandwidth: int,
    ) -> np.ndarray:
        local_bandwidths = np.partition(distances, int(bandwidth) - 1, axis=1)[
            :, int(bandwidth) - 1
        ]
        positive = local_bandwidths > 0.0
        if not np.all(positive):
            for index in np.where(~positive)[0]:
                positive_distances = distances[index][distances[index] > 0.0]
                if positive_distances.size == 0:
                    raise ValueError(
                        "Adaptive bandwidths cannot be constructed when all "
                        "calibration coordinates are identical."
                    )
                local_bandwidths[index] = np.min(positive_distances)
        local_bandwidths = np.nextafter(local_bandwidths, np.inf)

        if isinstance(self.kernel, str):
            name = self.kernel.strip().lower()
            normalized = distances / local_bandwidths[:, None]
            if name == "gaussian":
                return np.exp(-0.5 * normalized**2)
            if name == "exponential":
                return np.exp(-normalized)
            if name == "bisquare":
                weights = np.zeros_like(distances, dtype=float)
                mask = normalized < 1.0
                weights[mask] = (1.0 - normalized[mask] ** 2) ** 2
                return weights
            if name == "tricube":
                weights = np.zeros_like(distances, dtype=float)
                mask = normalized < 1.0
                weights[mask] = (1.0 - normalized[mask] ** 3) ** 3
                return weights
            if name == "boxcar":
                return (normalized <= 1.0).astype(float)

        if self.kernel_func_ is None:
            raise RuntimeError("Kernel function is unavailable.")
        return np.vstack(
            [
                self.kernel_func_(distance_row, local_bandwidths[index])
                for index, distance_row in enumerate(distances)
            ]
        )

    def _univariate_bandwidth_score(
        self,
        x: np.ndarray,
        y: np.ndarray,
        distances: np.ndarray,
        bandwidth: int,
    ) -> float:
        weights = self._adaptive_weight_matrix(distances, int(bandwidth))
        if self.bandwidth_method == "cv":
            weights = weights.copy()
            np.fill_diagonal(weights, 0.0)

        x_squared = x**2
        denominators = weights @ x_squared
        numerators = weights @ (x * y)
        params = np.empty_like(y, dtype=float)
        safe = denominators > np.finfo(float).eps * np.maximum(
            weights @ np.abs(x_squared), 1.0
        )
        params[safe] = numerators[safe] / denominators[safe]
        if not np.all(safe):
            design = x.reshape(-1, 1)
            for index in np.where(~safe)[0]:
                beta, _ = weighted_least_squares(design, y, weights[index])
                params[index] = beta[0]
        fitted = x * params

        if self.bandwidth_method == "cv":
            residuals = y - fitted
            return float(np.dot(residuals, residuals))

        diagonal_weights = np.diag(weights)
        trace_s = float(
            np.sum(
                np.divide(
                    x_squared * diagonal_weights,
                    denominators,
                    out=np.zeros_like(denominators),
                    where=np.abs(denominators) > np.finfo(float).eps,
                )
            )
        )
        if self.bandwidth_method == "aic":
            return float(compute_aic(y, fitted, trace_s))
        if self.bandwidth_method == "aicc":
            return float(compute_aicc(y, fitted, trace_s))
        return float(compute_bic(y, fitted, trace_s))

    def _select_adaptive_univariate_bandwidth(
        self,
        x: np.ndarray,
        y: np.ndarray,
        distances: np.ndarray,
        bandwidth_range: BandwidthRange,
    ) -> int:
        if bandwidth_range is None:
            lower = 2
            upper = x.size
        else:
            lower = int(np.ceil(bandwidth_range[0]))
            upper = int(np.floor(bandwidth_range[1]))
        if lower > upper:
            raise ValueError(
                "The adaptive bandwidth range contains no integer candidate."
            )

        cache: Dict[int, float] = {}

        def objective(candidate: int) -> float:
            candidate = int(np.clip(int(round(candidate)), lower, upper))
            if candidate not in cache:
                score = self._univariate_bandwidth_score(x, y, distances, candidate)
                cache[candidate] = score if np.isfinite(score) else np.inf
            return cache[candidate]

        if self.optimization_method == "grid":
            for candidate in range(lower, upper + 1):
                objective(candidate)
        else:
            # Match the reference MGWR implementation's discrete golden-section
            # search. Adaptive AICc surfaces are not perfectly smooth, so the
            # evaluated candidates are cached and the best finite score is used.
            delta = 0.38197
            a = float(lower)
            c = float(upper)
            b = round(a + delta * abs(c - a))
            d = round(c - delta * abs(c - a))
            difference = np.inf
            iterations = 0
            while (
                abs(difference) > self.search_tol and iterations < self.search_max_iter
            ):
                iterations += 1
                score_b = objective(int(round(b)))
                score_d = objective(int(round(d)))
                if score_b <= score_d:
                    c = d
                    d = b
                    b = round(a + delta * abs(c - a))
                else:
                    a = b
                    b = d
                    d = round(c - delta * abs(c - a))
                difference = score_b - score_d
                if int(round(b)) == int(round(d)):
                    objective(int(round(b)))
                    break

        finite = [
            (candidate, score)
            for candidate, score in cache.items()
            if np.isfinite(score)
        ]
        if not finite:
            raise RuntimeError("No valid adaptive bandwidth candidate was found.")
        return min(finite, key=lambda item: (item[1], item[0]))[0]

    def _select_bandwidth(
        self,
        X: np.ndarray,
        y: np.ndarray,
        bandwidth_range: BandwidthRange,
        distances: Optional[np.ndarray] = None,
    ) -> Bandwidth:
        if self.coords_train_ is None or self.kernel_func_ is None:
            raise RuntimeError("Training coordinates and kernel are unavailable.")
        if self.adaptive and X.shape[1] == 1 and distances is not None:
            return self._select_adaptive_univariate_bandwidth(
                X[:, 0], y, distances, bandwidth_range
            )
        selector = get_bandwidth_selector(
            self.bandwidth_method,
            adaptive=self.adaptive,
            optimization_method=self.optimization_method,
            verbose=False,
        )
        selected = selector.select(
            X,
            y,
            self.coords_train_,
            self.kernel_func_,
            bandwidth_range=bandwidth_range,
            distance_metric=self.distance_metric,
        )
        return int(selected) if self.adaptive else float(selected)

    def _resolve_initial_bandwidth(
        self,
        X_design: np.ndarray,
        y: np.ndarray,
    ) -> Bandwidth:
        if self.init_bandwidth is not None:
            return self._validate_bandwidth_value(
                self.init_bandwidth,
                n_samples=X_design.shape[0],
                minimum_adaptive=X_design.shape[1] + 1,
                name="init_bandwidth",
            )
        selected = self._select_bandwidth(X_design, y, self.bandwidth_range)
        return self._validate_bandwidth_value(
            selected,
            n_samples=X_design.shape[0],
            minimum_adaptive=X_design.shape[1] + 1,
            name="initial selected bandwidth",
        )

    def _weights_from_distances(
        self,
        distances: np.ndarray,
        bandwidth: Bandwidth,
    ) -> np.ndarray:
        if self.kernel_func_ is None:
            raise RuntimeError("Kernel function is unavailable.")
        local_bandwidth = (
            adaptive_bandwidth_weights(distances, int(bandwidth))
            if self.adaptive
            else float(bandwidth)
        )
        weights = np.asarray(self.kernel_func_(distances, local_bandwidth), dtype=float)
        if weights.shape != distances.shape:
            raise ValueError("The kernel returned an unexpected weight shape.")
        if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
            raise ValueError("The kernel returned invalid weights.")
        if not np.any(weights > 0.0):
            raise ValueError("The local kernel contains no positive weights.")
        return weights

    def _fit_initial_gwr(
        self,
        X_design: np.ndarray,
        y: np.ndarray,
        distances: np.ndarray,
        bandwidth: Bandwidth,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n_samples, n_parameters = X_design.shape
        params = np.empty((n_samples, n_parameters), dtype=float)
        for index, distance_row in enumerate(distances):
            weights = self._weights_from_distances(distance_row, bandwidth)
            beta, _ = weighted_least_squares(X_design, y, weights)
            params[index] = beta
        contributions = params * X_design
        residuals = y - np.sum(contributions, axis=1)
        return params, contributions, residuals

    def _fit_univariate_component(
        self,
        x: np.ndarray,
        response: np.ndarray,
        distances: np.ndarray,
        bandwidth: Bandwidth,
    ) -> Tuple[np.ndarray, np.ndarray]:
        n_samples = x.size
        params = np.empty(n_samples, dtype=float)
        design = x.reshape(-1, 1)
        for index, distance_row in enumerate(distances):
            weights = self._weights_from_distances(distance_row, bandwidth)
            beta, _ = weighted_least_squares(design, response, weights)
            params[index] = beta[0]
        return params, x * params

    def _backfit(
        self,
        X_design: np.ndarray,
        y: np.ndarray,
        distances: np.ndarray,
        initial_bandwidth: Bandwidth,
        manual_bandwidths: Optional[np.ndarray],
        bandwidth_ranges: Sequence[BandwidthRange],
    ) -> _BackfittingResult:
        params, contributions, residuals = self._fit_initial_gwr(
            X_design, y, distances, initial_bandwidth
        )
        n_samples, n_parameters = X_design.shape
        bandwidth_history: List[np.ndarray] = []
        convergence_history: List[float] = []
        stable_counter = 0
        previous_rss = float(np.dot(residuals, residuals))
        converged = False
        current_bandwidths = np.full(
            n_parameters,
            initial_bandwidth,
            dtype=int if self.adaptive else float,
        )

        for iteration in range(1, self.max_iter + 1):
            old_contributions = contributions.copy()
            new_contributions = np.zeros_like(contributions)
            new_params = np.zeros_like(params)
            iteration_bandwidths = np.empty_like(current_bandwidths)

            for parameter_index in range(n_parameters):
                # The current residual already contains updates from earlier
                # parameters in this iteration. Adding back the old term forms
                # the partial response required by GAM-style backfitting.
                partial_response = old_contributions[:, parameter_index] + residuals
                x_j = X_design[:, parameter_index]

                if manual_bandwidths is not None:
                    bandwidth = manual_bandwidths[parameter_index]
                elif stable_counter >= self.bws_same_times and bandwidth_history:
                    bandwidth = current_bandwidths[parameter_index]
                else:
                    bandwidth = self._select_bandwidth(
                        x_j.reshape(-1, 1),
                        partial_response,
                        bandwidth_ranges[parameter_index],
                        distances,
                    )
                bandwidth = self._validate_bandwidth_value(
                    bandwidth,
                    n_samples=n_samples,
                    minimum_adaptive=2,
                    name=f"bandwidth for parameter {parameter_index}",
                )

                param_j, contribution_j = self._fit_univariate_component(
                    x_j,
                    partial_response,
                    distances,
                    bandwidth,
                )
                new_params[:, parameter_index] = param_j
                new_contributions[:, parameter_index] = contribution_j
                iteration_bandwidths[parameter_index] = bandwidth
                residuals = partial_response - contribution_j

            if bandwidth_history and np.array_equal(
                bandwidth_history[-1], iteration_bandwidths
            ):
                stable_counter += 1
            else:
                stable_counter = 0

            fitted = np.sum(new_contributions, axis=1)
            if self.rss_score:
                new_rss = float(np.dot(y - fitted, y - fitted))
                denominator = max(new_rss, np.finfo(float).eps)
                score = abs(new_rss - previous_rss) / denominator
                previous_rss = new_rss
            else:
                numerator = float(
                    np.sum((new_contributions - old_contributions) ** 2) / n_samples
                )
                denominator = float(np.sum(fitted**2))
                score = np.sqrt(numerator / max(denominator, np.finfo(float).eps))

            params = new_params
            contributions = new_contributions
            residuals = y - fitted
            current_bandwidths = iteration_bandwidths.copy()
            bandwidth_history.append(iteration_bandwidths.copy())
            convergence_history.append(float(score))

            if self.verbose:
                bandwidth_text = ", ".join(
                    str(int(value)) if self.adaptive else f"{value:.6g}"
                    for value in iteration_bandwidths
                )
                print(
                    f"MGWR iteration {iteration}: SOC={score:.8g}; "
                    f"bandwidths=[{bandwidth_text}]"
                )

            if score < self.tol:
                converged = True
                break

        if not converged:
            warnings.warn(
                "MGWR reached max_iter before satisfying the convergence tolerance. "
                "Inspect convergence_history_ and consider increasing max_iter.",
                RuntimeWarning,
                stacklevel=2,
            )

        return _BackfittingResult(
            params=params,
            contributions=contributions,
            residuals=residuals,
            bandwidths=current_bandwidths,
            bandwidth_history=np.asarray(bandwidth_history),
            convergence_history=np.asarray(convergence_history, dtype=float),
            n_iter=len(convergence_history),
            converged=converged,
            initial_bandwidth=initial_bandwidth,
        )

    def _coefficient_smoother_row(
        self,
        x: np.ndarray,
        weights: np.ndarray,
    ) -> np.ndarray:
        denominator = float(np.dot(x * weights, x))
        scale = max(float(np.dot(np.abs(x) * weights, np.abs(x))), 1.0)
        if denominator <= np.finfo(float).eps * scale:
            # Use the same regularized weighted least-squares convention as the
            # fitted coefficients when the local univariate normal equation is
            # numerically singular.
            _, inverse_normal = weighted_least_squares(
                x.reshape(-1, 1), np.zeros_like(x), weights
            )
            return (inverse_normal @ (x.reshape(1, -1) * weights))[0]
        return x * weights / denominator

    def _compute_exact_inference(
        self,
        X_design: np.ndarray,
        distances: np.ndarray,
        *,
        n_chunks: int,
        store_hat_matrix: bool,
        store_partial_hat_matrices: bool,
        compute_covariance: bool,
    ) -> _InferenceResult:
        if self.initial_bandwidth_ is None or self.bandwidth_history_ is None:
            raise RuntimeError("Backfitting history is unavailable for inference.")
        n_samples, n_parameters = X_design.shape
        if not isinstance(n_chunks, (int, np.integer)) or isinstance(
            n_chunks, (bool, np.bool_)
        ):
            raise TypeError("n_chunks must be an integer.")
        if int(n_chunks) < 1:
            raise ValueError("n_chunks must be at least 1.")
        n_chunks = min(int(n_chunks), n_samples)

        effective_params = np.zeros(n_parameters, dtype=float)
        covariance_factors = (
            np.zeros((n_samples, n_parameters), dtype=float)
            if compute_covariance
            else None
        )
        influence = np.empty(n_samples, dtype=float)
        hat_matrix = (
            np.empty((n_samples, n_samples), dtype=float) if store_hat_matrix else None
        )
        partial_hat_matrices = (
            np.empty((n_samples, n_samples, n_parameters), dtype=float)
            if store_partial_hat_matrices
            else None
        )
        trace_sts = 0.0

        for chunk_indices in np.array_split(np.arange(n_samples), n_chunks):
            if chunk_indices.size == 0:
                continue
            chunk_size = int(chunk_indices.size)
            identity_chunk = np.zeros((n_samples, chunk_size), dtype=float)
            identity_chunk[chunk_indices, np.arange(chunk_size)] = 1.0
            partial_R = np.zeros((n_samples, chunk_size, n_parameters), dtype=float)
            partial_B = (
                np.zeros((n_samples, chunk_size, n_parameters), dtype=float)
                if compute_covariance
                else None
            )

            # Initialize the additive smoother decomposition from the same
            # single-bandwidth GWR used to initialize coefficient backfitting.
            for location, distance_row in enumerate(distances):
                weights = self._weights_from_distances(
                    distance_row, self.initial_bandwidth_
                )
                _, inverse_normal = weighted_least_squares(
                    X_design, np.zeros(n_samples, dtype=float), weights
                )
                coefficient_operator = inverse_normal @ (X_design.T * weights)
                projected = (coefficient_operator @ identity_chunk).T
                partial_R[location] = projected * X_design[location]
                if partial_B is not None:
                    partial_B[location] = projected

            error_operator = identity_chunk - np.sum(partial_R, axis=2)

            for iteration_bandwidths in self.bandwidth_history_:
                for parameter_index in range(n_parameters):
                    old_partial = partial_R[:, :, parameter_index] + error_operator
                    x_j = X_design[:, parameter_index]
                    bandwidth = iteration_bandwidths[parameter_index]
                    for location, distance_row in enumerate(distances):
                        weights = self._weights_from_distances(distance_row, bandwidth)
                        coefficient_row = self._coefficient_smoother_row(x_j, weights)
                        updated_B = coefficient_row @ old_partial
                        partial_R[location, :, parameter_index] = (
                            x_j[location] * updated_B
                        )
                        if partial_B is not None:
                            partial_B[location, :, parameter_index] = updated_B
                    error_operator = old_partial - partial_R[:, :, parameter_index]

            smoother_chunk = np.sum(partial_R, axis=2)
            trace_sts += float(np.sum(smoother_chunk**2))
            for local_column, global_index in enumerate(chunk_indices):
                influence[global_index] = smoother_chunk[global_index, local_column]
                effective_params += partial_R[global_index, local_column, :]

            if covariance_factors is not None and partial_B is not None:
                covariance_factors += np.sum(partial_B**2, axis=1)
            if hat_matrix is not None:
                hat_matrix[:, chunk_indices] = smoother_chunk
            if partial_hat_matrices is not None:
                partial_hat_matrices[:, chunk_indices, :] = partial_R

        return _InferenceResult(
            effective_params_by_variable=effective_params,
            covariance_factors=covariance_factors,
            influence=influence,
            trace_S=float(np.sum(effective_params)),
            trace_StS=float(trace_sts),
            hat_matrix=hat_matrix,
            partial_hat_matrices=partial_hat_matrices,
        )

    def _set_inference_results(
        self,
        inference: _InferenceResult,
        params: np.ndarray,
    ) -> None:
        if self.residuals_ is None:
            raise RuntimeError("Residuals are unavailable.")
        rss = float(np.dot(self.residuals_, self.residuals_))
        denominator = (
            self.n_samples_ - inference.trace_S
            if self.sigma2_v1
            else self.n_samples_ - 2.0 * inference.trace_S + inference.trace_StS
        )
        self.sigma2_ = rss / denominator if denominator > 0.0 else np.nan
        self.influence_ = inference.influence.copy()

        leverage_term = 1.0 - self.influence_
        self.standardized_residuals_ = np.full(self.n_samples_, np.nan, dtype=float)
        if (
            self.sigma2_ is not None
            and np.isfinite(self.sigma2_)
            and self.sigma2_ > 0.0
        ):
            valid = leverage_term > np.finfo(float).eps
            self.standardized_residuals_[valid] = self.residuals_[valid] / np.sqrt(
                self.sigma2_ * leverage_term[valid]
            )

        self.cooks_distance_ = np.full(self.n_samples_, np.nan, dtype=float)
        if inference.trace_S > np.finfo(float).eps:
            valid = leverage_term > np.finfo(float).eps
            self.cooks_distance_[valid] = (
                self.standardized_residuals_[valid] ** 2
                * self.influence_[valid]
                / (inference.trace_S * leverage_term[valid])
            )

        if (
            inference.covariance_factors is None
            or self.sigma2_ is None
            or not np.isfinite(self.sigma2_)
        ):
            return
        covariance_diagonal = np.maximum(
            inference.covariance_factors * self.sigma2_, 0.0
        )
        standard_errors = np.sqrt(covariance_diagonal)
        t_values = np.full_like(params, np.nan, dtype=float)
        np.divide(
            params,
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
        compute_hat_matrix: bool = False,
        store_partial_hat_matrices: bool = False,
        compute_inference: bool = True,
        n_chunks: int = 1,
        verbose: Optional[bool] = None,
    ) -> "MGWR":
        """Fit the MGWR model and return ``self``.

        Args:
            X: Predictor matrix with shape ``(n_samples, n_features)``.
            y: Response vector with shape ``(n_samples,)``.
            coords: Coordinates with shape ``(n_samples, 2)``.
            compute_hat_matrix: Retain the complete model smoother matrix.
            store_partial_hat_matrices: Retain one ``n x n`` smoother matrix
                per fitted parameter. This can require substantial memory.
            compute_inference: Compute local standard errors and t statistics.
                Exact smoother traces are computed regardless of this setting.
            n_chunks: Number of column chunks used during exact inference.
            verbose: Optional per-fit override of the estimator's verbosity.

        Returns:
            The fitted model instance.
        """
        for name, value in (
            ("compute_hat_matrix", compute_hat_matrix),
            ("store_partial_hat_matrices", store_partial_hat_matrices),
            ("compute_inference", compute_inference),
        ):
            if not isinstance(value, (bool, np.bool_)):
                raise TypeError(f"{name} must be boolean.")
        if verbose is not None:
            if not isinstance(verbose, (bool, np.bool_)):
                raise TypeError("verbose must be boolean or None.")
            self.verbose = bool(verbose)

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
            n_samples, n_parameters = X_design.shape
            self.kernel_func_ = get_kernel_function(self.kernel)
            distances = np.asarray(
                compute_distance_matrix(
                    self.coords_train_,
                    self.coords_train_,
                    metric=self.distance_metric,
                ),
                dtype=float,
            )

            manual_bandwidths = self._resolve_manual_bandwidths(
                n_parameters=n_parameters,
                n_samples=n_samples,
            )
            bandwidth_ranges = self._resolve_bandwidth_ranges(
                n_parameters=n_parameters,
                n_samples=n_samples,
            )
            initial_bandwidth = self._resolve_initial_bandwidth(X_design, self.y_train_)
            self.initial_bandwidth_ = initial_bandwidth
            self.bandwidth_ = initial_bandwidth  # compatibility: initial GWR scale

            if self.verbose:
                kind = "adaptive neighbours" if self.adaptive else "fixed distance"
                print(
                    f"Initializing MGWR with {kind} bandwidth="
                    f"{self.initial_bandwidth_}."
                )

            backfit = self._backfit(
                X_design,
                self.y_train_,
                distances,
                initial_bandwidth,
                manual_bandwidths,
                bandwidth_ranges,
            )
            self.bandwidths_ = backfit.bandwidths.copy()
            self.bandwidth_history_ = backfit.bandwidth_history.copy()
            self.convergence_history_ = backfit.convergence_history.copy()
            self.n_iter_ = backfit.n_iter
            self.converged_ = backfit.converged
            self.parameter_contributions_ = backfit.contributions.copy()

            params = backfit.params
            if self.fit_intercept:
                self.intercept_ = params[:, 0].copy()
                self.coef_ = params[:, 1:].copy()
            else:
                self.intercept_ = np.zeros(n_samples, dtype=float)
                self.coef_ = params.copy()
            self.fitted_values_ = np.sum(backfit.contributions, axis=1)
            self.residuals_ = self.y_train_ - self.fitted_values_
            self.local_r2_ = None

            self.inference_enabled_ = bool(compute_inference)
            inference = self._compute_exact_inference(
                X_design,
                distances,
                n_chunks=n_chunks,
                store_hat_matrix=bool(compute_hat_matrix),
                store_partial_hat_matrices=bool(store_partial_hat_matrices),
                compute_covariance=self.inference_enabled_,
            )
            self.effective_params_by_variable_ = (
                inference.effective_params_by_variable.copy()
            )
            self.ENP_j_ = self.effective_params_by_variable_
            self.hat_matrix_ = inference.hat_matrix
            self.partial_hat_matrices_ = inference.partial_hat_matrices
            self.diagnostics_ = compute_diagnostics(
                self.y_train_,
                self.fitted_values_,
                compute_gwr_stats=True,
                trace_S=inference.trace_S,
                trace_StS=inference.trace_StS,
            )
            self._set_inference_results(inference, params)

            alpha = np.asarray([0.10, 0.05, 0.01], dtype=float)
            safe_enp = np.maximum(
                self.effective_params_by_variable_, np.finfo(float).eps
            )
            self.adjusted_alpha_by_variable_ = alpha / safe_enp[:, None]
            self.critical_t_values_ = student_t.ppf(
                1.0 - self.adjusted_alpha_by_variable_[:, 1] / 2.0,
                max(self.n_samples_ - 1, 1),
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
        """Reject unvalidated out-of-sample MGWR prediction.

        Raises:
            NotImplementedError: Always. Use ``fitted_values_`` for calibration
                locations.
        """
        self._check_is_fitted()
        raise NotImplementedError(
            "Out-of-sample MGWR prediction is not implemented because a validated "
            "multiscale prediction operator is not yet part of the reference MGWR "
            "workflow. Use fitted_values_ for calibration locations."
        )

    def to_frame(self) -> pd.DataFrame:
        """Return calibration-location parameters and diagnostics."""
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
        ):
            if values is not None:
                frame[name] = values
        return frame

    def summary(self) -> str:
        """Return a stable text summary of MGWR calibration results."""
        self._check_is_fitted()
        if self.bandwidths_ is None or self.effective_params_by_variable_ is None:
            raise RuntimeError("MGWR bandwidth and inference results are unavailable.")
        feature_names = (
            [str(name) for name in self.feature_names_in_]
            if self.feature_names_in_ is not None
            else [f"x{index}" for index in range(self.n_features_in_ or 0)]
        )
        parameter_names = (["intercept"] if self.fit_intercept else []) + feature_names
        params = (
            np.column_stack([self.intercept_, self.coef_])
            if self.fit_intercept
            else np.asarray(self.coef_)
        )

        lines = [
            "=" * 88,
            "Multiscale Geographically Weighted Regression (MGWR)",
            "=" * 88,
            f"Samples: {self.n_samples_}",
            f"Predictors: {self.n_features_in_}",
            f"Kernel: {self.kernel}",
            f"Bandwidth type: {'adaptive neighbours' if self.adaptive else 'fixed distance'}",
            f"Bandwidth criterion: {self.bandwidth_method.upper()}",
            f"Initial GWR bandwidth: {self.initial_bandwidth_}",
            f"Backfitting iterations: {self.n_iter_}",
            f"Converged: {self.converged_}",
            f"Final SOC: {self.convergence_history_[-1]:.8g}",
            "",
            "Variable-specific scales and coefficient distributions",
            "-" * 88,
            f"{'Variable':<20}{'Bandwidth':>12}{'ENP_j':>12}{'Min':>11}{'Median':>11}{'Mean':>11}{'Max':>11}",
        ]
        for index, name in enumerate(parameter_names):
            values = params[:, index]
            bandwidth = self.bandwidths_[index]
            bandwidth_text = (
                str(int(bandwidth)) if self.adaptive else f"{float(bandwidth):.6g}"
            )
            lines.append(
                f"{name:<20}{bandwidth_text:>12}"
                f"{self.effective_params_by_variable_[index]:>12.4f}"
                f"{np.min(values):>11.5f}{np.median(values):>11.5f}"
                f"{np.mean(values):>11.5f}{np.max(values):>11.5f}"
            )

        lines.extend(["", "MGWR diagnostics", "-" * 88])
        for label, key in (
            ("R-squared", "r2"),
            ("Adjusted R-squared", "adj_r2"),
            ("RSS", "rss"),
            ("RMSE", "rmse"),
            ("MAE", "mae"),
            ("AIC", "aic"),
            ("AICc", "aicc"),
            ("BIC", "bic"),
            ("trace(S)", "trace_S"),
            ("trace(S'S)", "trace_StS"),
            ("ENP v2", "enp_v2"),
            ("EDF v2", "edf_v2"),
        ):
            value = self.diagnostics_.get(key, np.nan) if self.diagnostics_ else np.nan
            lines.append(f"{label:<32}{value:>16.6f}")
        sigma2 = np.nan if self.sigma2_ is None else self.sigma2_
        lines.append(f"{'Residual variance (sigma^2)':<32}{sigma2:>16.6f}")
        lines.append("=" * 88)
        return "\n".join(lines)


__all__ = ["MGWR"]
