# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Multiscale geographically and temporally weighted regression.

This module implements Gaussian MGTWR with variable-specific spatial
bandwidths, variable-specific temporal scale parameters, iterative additive
backfitting, and exact smoother diagnostics. The numerical implementation is
self-contained and uses the same NumPy/SciPy kernels, weighted least-squares
solver, diagnostics, and fitted-state conventions as the rest of pyGWRx.

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

from pygwrx.core._summary import format_summary
from pygwrx.core.kernels import get_kernel_function
from pygwrx.core.metrics import (
    compute_aic,
    compute_aicc,
    compute_bic,
    compute_diagnostics,
)
from pygwrx.core.solver import adaptive_bandwidth_weights, weighted_least_squares
from pygwrx.core.utils import add_intercept, compute_distance_matrix
from pygwrx.models.mgwr import MGWR

ArrayLike = Union[np.ndarray, pd.DataFrame]
VectorLike = Union[np.ndarray, pd.Series, pd.DataFrame]
Bandwidth = Union[int, float]
BandwidthInput = Optional[Union[Bandwidth, Sequence[Bandwidth]]]
TauInput = Optional[Union[float, Sequence[float]]]
BandwidthRange = Optional[Tuple[float, float]]


@dataclass
class _MGTWRBackfittingResult:
    """Internal result produced by MGTWR additive backfitting."""

    params: np.ndarray
    contributions: np.ndarray
    residuals: np.ndarray
    bandwidths: np.ndarray
    taus: np.ndarray
    bandwidth_history: np.ndarray
    tau_history: np.ndarray
    convergence_history: np.ndarray
    n_iter: int
    converged: bool
    initial_bandwidth: Bandwidth
    initial_tau: float


@dataclass
class _MGTWRInferenceResult:
    """Internal exact MGTWR smoother and covariance diagnostics."""

    effective_params_by_variable: np.ndarray
    covariance_factors: Optional[np.ndarray]
    influence: np.ndarray
    trace_S: float
    trace_StS: float
    hat_matrix: Optional[np.ndarray]
    partial_hat_matrices: Optional[np.ndarray]


class MGTWR(MGWR):
    """Gaussian multiscale geographically and temporally weighted regression.

    MGTWR represents the response as a sum of coefficient-specific
    spatiotemporal terms. Each fitted parameter receives an independent spatial
    bandwidth and temporal scale parameter. Calibration starts from a common
    GTWR fit and then updates one additive term at a time from its partial
    residual until the score of change converges.

    Args:
        bandwidths: Optional scalar or one spatial bandwidth per fitted
            parameter. The fitted parameter count includes the intercept when
            ``fit_intercept=True``. When omitted, scales are selected during
            backfitting.
        taus: Optional scalar or one non-negative temporal scale per fitted
            parameter. It must be supplied together with ``bandwidths``.
            Distances are combined as ``sqrt(ds**2 + tau * dt**2)``. A value of
            zero removes temporal distance for that coefficient.
        kernel: ``"gaussian"``, ``"bisquare"``, or ``"exponential"``.
        adaptive: Interpret spatial bandwidths as integer nearest-neighbour
            counts in combined spatiotemporal distance.
        fit_intercept: Include a spatiotemporally varying intercept.
        bandwidth_method: Scale-selection criterion: ``"aicc"``, ``"aic"``,
            ``"bic"``, or ``"cv"``.
        bandwidth_range: Common lower and upper spatial bandwidth bounds for
            automatic selection.
        tau_range: Common lower and upper temporal-scale bounds for automatic
            selection.
        init_bandwidth: Optional common spatial bandwidth for the initial GTWR
            fit.
        init_tau: Optional common temporal scale for the initial GTWR fit.
        tol: Resolution target used by the deterministic two-dimensional scale
            search.
        tol_multi: Backfitting score-of-change convergence tolerance.
        max_iter: Maximum number of backfitting iterations.
        rss_score: Use relative RSS change instead of smooth-function change as
            the convergence score.
        calculate_inference: Compute exact smoother traces, effective parameter
            counts, local standard errors, and information criteria.
        n_chunks: Number of column chunks used by exact smoother inference.
        verbose: Print scale-search and backfitting progress.

    Attributes:
        bandwidths_: Final variable-specific spatial bandwidths.
        taus_: Final variable-specific temporal scale parameters.
        temporal_bandwidths_: Equivalent temporal bandwidths, computed as
            ``bandwidth / sqrt(tau)`` and set to infinity when ``tau == 0``.
        bandwidth_history_: Spatial bandwidth vector from every iteration.
        tau_history_: Temporal-scale vector from every iteration.
        convergence_history_: Score of change from every iteration.
        params_: Local parameters including the intercept when fitted.
        effective_params_by_variable_: Exact effective parameter count for each
            coefficient surface when inference is enabled.
        parameter_standard_errors_: Local parameter standard errors when
            inference is enabled.
        parameter_t_values_: Local parameter t statistics when inference is
            enabled.

    Notes:
        Automatic scale selection uses a deterministic coarse-to-fine candidate
        search with explicit boundary evaluation; it is not an exhaustive proof
        of the global optimum. The numerical interpretation of ``tau`` depends
        on the coordinate and time units. Independent-target prediction is not
        exposed because a stable MGTWR prediction operator for independently
        supplied locations is not yet part of the package contract. Use
        ``fitted_values_`` for calibration-location estimates.

    References:
        Wu, C., Ren, F., Hu, W., and Du, Q. (2019). Multiscale geographically
        and temporally weighted regression: exploring the spatiotemporal
        determinants of housing prices. *International Journal of Geographical
        Information Science*, 33(3), 489-511.
    """

    _VALID_KERNELS = {"gaussian", "bisquare", "exponential"}
    _VALID_CRITERIA = {"aicc", "aic", "bic", "cv"}

    # These annotations mirror inherited fitted-state fields used directly by
    # this subclass and keep the standalone typed API check explicit.
    kernel_func_: Optional[Callable[[np.ndarray, float], np.ndarray]]
    feature_names_in_: Optional[np.ndarray]
    initial_bandwidth_: Optional[Bandwidth]
    bandwidth_history_: Optional[np.ndarray]
    temporal_bandwidths_: Optional[np.ndarray]

    def __init__(
        self,
        bandwidths: BandwidthInput = None,
        taus: TauInput = None,
        *,
        kernel: str = "bisquare",
        adaptive: bool = True,
        fit_intercept: bool = True,
        bandwidth_method: str = "aicc",
        bandwidth_range: BandwidthRange = None,
        tau_range: Tuple[float, float] = (0.0, 4.0),
        init_bandwidth: Optional[Bandwidth] = None,
        init_tau: Optional[float] = None,
        tol: float = 1e-6,
        tol_multi: float = 1e-5,
        max_iter: int = 200,
        rss_score: bool = False,
        calculate_inference: bool = True,
        n_chunks: int = 1,
        verbose: bool = False,
    ) -> None:
        kernel_name = str(kernel).strip().lower()
        if kernel_name not in self._VALID_KERNELS:
            raise ValueError(f"kernel must be one of {sorted(self._VALID_KERNELS)}.")
        method = str(bandwidth_method).strip().lower()
        if method not in self._VALID_CRITERIA:
            raise ValueError(
                "bandwidth_method must be one of 'aicc', 'aic', 'bic', or 'cv'."
            )
        if (bandwidths is None) != (taus is None):
            raise ValueError("bandwidths and taus must be supplied together.")
        if taus is not None:
            try:
                raw_taus = np.asarray(taus, dtype=float).reshape(-1)
            except (TypeError, ValueError) as error:
                raise TypeError("taus must contain numeric values.") from error
            if raw_taus.size == 0:
                raise ValueError("taus cannot be empty.")
            if not np.all(np.isfinite(raw_taus)) or np.any(raw_taus < 0.0):
                raise ValueError("All taus must be finite and non-negative.")
        if not isinstance(calculate_inference, (bool, np.bool_)):
            raise TypeError("calculate_inference must be boolean.")
        if not isinstance(n_chunks, (int, np.integer)) or isinstance(
            n_chunks, (bool, np.bool_)
        ):
            raise TypeError("n_chunks must be an integer.")
        if int(n_chunks) < 1:
            raise ValueError("n_chunks must be at least 1.")
        self._validate_tau_range(tau_range)
        if init_tau is not None:
            self._validate_tau_value(init_tau, name="init_tau")

        super().__init__(
            kernel=kernel_name,
            bandwidths=bandwidths,
            bandwidth_method=method,
            adaptive=adaptive,
            bandwidth_range=bandwidth_range,
            init_bandwidth=init_bandwidth,
            optimization_method="golden_section",
            search_tol=tol,
            search_max_iter=max(50, min(int(max_iter), 500)),
            max_iter=max_iter,
            tol=tol_multi,
            rss_score=rss_score,
            bws_same_times=5,
            fit_intercept=fit_intercept,
            distance_metric="euclidean",
            sigma2_v1=True,
            verbose=verbose,
        )
        self.taus = taus
        self.tau_range = (float(tau_range[0]), float(tau_range[1]))
        self.init_tau = None if init_tau is None else float(init_tau)
        self.calculate_inference = bool(calculate_inference)
        self.n_chunks = int(n_chunks)
        self._reset_mgtwr_state()

    @staticmethod
    def _validate_tau_value(value: float, *, name: str) -> float:
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, float, np.integer, np.floating)
        ):
            raise TypeError(f"{name} must be numeric.")
        numeric = float(value)
        if not np.isfinite(numeric) or numeric < 0.0:
            raise ValueError(f"{name} must be finite and non-negative.")
        return numeric

    @classmethod
    def _validate_tau_range(cls, value: Tuple[float, float]) -> None:
        if not isinstance(value, (tuple, list)) or len(value) != 2:
            raise TypeError("tau_range must be a two-element tuple/list.")
        lower = cls._validate_tau_value(value[0], name="tau_range[0]")
        upper = cls._validate_tau_value(value[1], name="tau_range[1]")
        if lower > upper:
            raise ValueError("tau_range must satisfy lower <= upper.")

    def _reset_mgtwr_state(self) -> None:
        self.initial_tau_: Optional[float] = None
        self.taus_: Optional[np.ndarray] = None
        self.temporal_bandwidths_: Optional[np.ndarray] = None
        self.tau_history_: Optional[np.ndarray] = None
        self.params_: Optional[np.ndarray] = None
        self.rss_: Optional[float] = None
        self.r2_: Optional[float] = None
        self.effective_params_: Optional[float] = None
        self.aic_: Optional[float] = None
        self.aicc_: Optional[float] = None
        self.bic_: Optional[float] = None

    def _reset_fit_state(self) -> None:
        super()._reset_fit_state()
        self._reset_mgtwr_state()

    def _resolve_manual_taus(self, *, n_parameters: int) -> Optional[np.ndarray]:
        if self.taus is None:
            return None
        if isinstance(
            self.taus, (int, float, np.integer, np.floating)
        ) and not isinstance(self.taus, (bool, np.bool_)):
            values = [self.taus] * n_parameters
        else:
            if isinstance(self.taus, (str, bytes)):
                raise TypeError("taus must be numeric, a numeric sequence, or None.")
            try:
                values = list(self.taus)  # type: ignore[arg-type]
            except TypeError as error:
                raise TypeError(
                    "taus must be numeric, a numeric sequence, or None."
                ) from error
            if len(values) != n_parameters:
                raise ValueError(
                    "taus must contain one value per fitted parameter; "
                    f"expected {n_parameters}, received {len(values)}."
                )
        return np.asarray(
            [
                self._validate_tau_value(value, name=f"taus[{index}]")
                for index, value in enumerate(values)
            ],
            dtype=float,
        )

    @staticmethod
    def _temporal_distances(times: np.ndarray) -> np.ndarray:
        return np.abs(times[:, None] - times[None, :])

    @staticmethod
    def _combine_distances(
        spatial_distances: np.ndarray,
        temporal_distances: np.ndarray,
        tau: float,
    ) -> np.ndarray:
        combined = np.sqrt(spatial_distances**2 + float(tau) * temporal_distances**2)
        if not np.all(np.isfinite(combined)) or np.any(combined < 0.0):
            raise ValueError(
                "The spatiotemporal distance calculation produced invalid values."
            )
        return combined

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
            raise ValueError("The local spatiotemporal kernel has no positive weights.")
        return weights

    def _scale_score(
        self,
        X: np.ndarray,
        y: np.ndarray,
        combined_distances: np.ndarray,
        bandwidth: Bandwidth,
    ) -> float:
        n_samples, n_parameters = X.shape
        fitted = np.empty(n_samples, dtype=float)
        trace_s = 0.0
        try:
            for index, distance_row in enumerate(combined_distances):
                weights = self._weights_from_distances(distance_row, bandwidth)
                if self.bandwidth_method == "cv":
                    weights = weights.copy()
                    weights[index] = 0.0
                if np.count_nonzero(weights > 0.0) < n_parameters:
                    return float("inf")
                beta, inverse_normal = weighted_least_squares(X, y, weights)
                fitted[index] = float(X[index] @ beta)
                if self.bandwidth_method != "cv":
                    coefficient_operator = inverse_normal @ (X.T * weights)
                    hat_row = X[index] @ coefficient_operator
                    trace_s += float(hat_row[index])
        except (ValueError, np.linalg.LinAlgError, FloatingPointError):
            return float("inf")

        if self.bandwidth_method == "cv":
            residuals = y - fitted
            return float(np.dot(residuals, residuals))
        if self.bandwidth_method == "aic":
            return float(compute_aic(y, fitted, trace_s))
        if self.bandwidth_method == "aicc":
            return float(compute_aicc(y, fitted, trace_s))
        return float(compute_bic(y, fitted, trace_s))

    def _bandwidth_candidates(
        self,
        combined_distances: np.ndarray,
        *,
        n_parameters: int,
        bandwidth_range: BandwidthRange,
    ) -> np.ndarray:
        n_samples = combined_distances.shape[0]
        if self.adaptive:
            lower = max(2, n_parameters + 1)
            upper = n_samples
            if bandwidth_range is not None:
                lower = max(lower, int(np.ceil(bandwidth_range[0])))
                upper = min(upper, int(np.floor(bandwidth_range[1])))
            if lower > upper:
                raise ValueError(
                    "The adaptive bandwidth range contains no valid candidate."
                )
            count = upper - lower + 1
            if count <= 25:
                return np.arange(lower, upper + 1, dtype=int)
            return np.unique(np.rint(np.linspace(lower, upper, 25)).astype(int))

        positive = combined_distances[combined_distances > np.finfo(float).eps]
        if positive.size == 0:
            raise ValueError("No positive spatiotemporal distances are available.")
        lower_fixed: float = float(np.min(positive))
        upper_fixed: float = float(np.max(positive))
        if bandwidth_range is not None:
            lower_fixed = max(lower_fixed, float(bandwidth_range[0]))
            upper_fixed = min(upper_fixed, float(bandwidth_range[1]))
        if lower_fixed > upper_fixed:
            raise ValueError("The fixed bandwidth range contains no valid candidate.")
        if np.isclose(lower_fixed, upper_fixed):
            return np.asarray([lower_fixed], dtype=float)
        return np.linspace(lower_fixed, upper_fixed, 17, dtype=float)

    def _select_bandwidth_for_tau(
        self,
        X: np.ndarray,
        y: np.ndarray,
        combined_distances: np.ndarray,
        bandwidth_range: BandwidthRange,
    ) -> Tuple[Bandwidth, float]:
        candidates = self._bandwidth_candidates(
            combined_distances,
            n_parameters=X.shape[1],
            bandwidth_range=bandwidth_range,
        )
        cache: Dict[float, float] = {}

        def evaluate(candidate: float) -> float:
            value: Bandwidth = (
                int(round(candidate)) if self.adaptive else float(candidate)
            )
            key = float(value)
            if key not in cache:
                score = self._scale_score(X, y, combined_distances, value)
                cache[key] = score if np.isfinite(score) else np.inf
            return cache[key]

        scores = np.asarray([evaluate(float(candidate)) for candidate in candidates])
        best_index = int(np.argmin(scores))
        if not np.isfinite(scores[best_index]):
            raise RuntimeError("No finite bandwidth candidate was found.")

        if not self.adaptive and candidates.size > 1:
            left = candidates[max(best_index - 1, 0)]
            right = candidates[min(best_index + 1, candidates.size - 1)]
            for _ in range(2):
                if right - left <= self.search_tol:
                    break
                refined = np.linspace(left, right, 9, dtype=float)
                refined_scores = np.asarray(
                    [evaluate(float(value)) for value in refined]
                )
                refined_best = int(np.argmin(refined_scores))
                left = refined[max(refined_best - 1, 0)]
                right = refined[min(refined_best + 1, refined.size - 1)]

        best_value, best_score = min(cache.items(), key=lambda item: (item[1], item[0]))
        return (int(best_value) if self.adaptive else float(best_value), best_score)

    def _tau_candidates(self, initial: Optional[float]) -> np.ndarray:
        lower, upper = self.tau_range
        if np.isclose(lower, upper):
            return np.asarray([lower], dtype=float)
        values = list(np.linspace(lower, upper, 9, dtype=float))
        if initial is not None and lower <= initial <= upper:
            values.append(float(initial))
        return np.unique(np.asarray(values, dtype=float))

    def _select_scale(
        self,
        X: np.ndarray,
        y: np.ndarray,
        spatial_distances: np.ndarray,
        temporal_distances: np.ndarray,
        bandwidth_range: BandwidthRange,
        *,
        initial_tau: Optional[float],
    ) -> Tuple[Bandwidth, float, float]:
        cache: Dict[float, Tuple[Bandwidth, float]] = {}

        def evaluate(tau: float) -> Tuple[Bandwidth, float]:
            key = float(tau)
            if key not in cache:
                combined = self._combine_distances(
                    spatial_distances, temporal_distances, key
                )
                cache[key] = self._select_bandwidth_for_tau(
                    X, y, combined, bandwidth_range
                )
            return cache[key]

        candidates = self._tau_candidates(initial_tau)
        scores = np.asarray([evaluate(float(tau))[1] for tau in candidates])
        best_index = int(np.argmin(scores))
        if not np.isfinite(scores[best_index]):
            raise RuntimeError("No finite spatiotemporal scale candidate was found.")

        left = candidates[max(best_index - 1, 0)]
        right = candidates[min(best_index + 1, candidates.size - 1)]
        for _ in range(2):
            if right - left <= self.search_tol:
                break
            refined = np.linspace(left, right, 7, dtype=float)
            refined_scores = np.asarray([evaluate(float(tau))[1] for tau in refined])
            refined_best = int(np.argmin(refined_scores))
            left = refined[max(refined_best - 1, 0)]
            right = refined[min(refined_best + 1, refined.size - 1)]

        tau, (bandwidth, score) = min(
            cache.items(), key=lambda item: (item[1][1], item[0])
        )
        if self.verbose:
            print(
                f"Selected MGTWR scale: bandwidth={bandwidth}, "
                f"tau={tau:.8g}, score={score:.8g}"
            )
        return bandwidth, float(tau), float(score)

    def _fit_initial_gtwr(
        self,
        X_design: np.ndarray,
        y: np.ndarray,
        combined_distances: np.ndarray,
        bandwidth: Bandwidth,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n_samples, n_parameters = X_design.shape
        params = np.empty((n_samples, n_parameters), dtype=float)
        for index, distance_row in enumerate(combined_distances):
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
        combined_distances: np.ndarray,
        bandwidth: Bandwidth,
    ) -> Tuple[np.ndarray, np.ndarray]:
        params = np.empty(x.size, dtype=float)
        design = x.reshape(-1, 1)
        for index, distance_row in enumerate(combined_distances):
            weights = self._weights_from_distances(distance_row, bandwidth)
            beta, _ = weighted_least_squares(design, response, weights)
            params[index] = beta[0]
        return params, x * params

    def _backfit(
        self,
        X_design: np.ndarray,
        y: np.ndarray,
        spatial_distances: np.ndarray,
        temporal_distances: np.ndarray,
        initial_bandwidth: Bandwidth,
        initial_tau: float,
        manual_bandwidths: Optional[np.ndarray],
        manual_taus: Optional[np.ndarray],
    ) -> _MGTWRBackfittingResult:
        initial_distances = self._combine_distances(
            spatial_distances, temporal_distances, initial_tau
        )
        params, contributions, residuals = self._fit_initial_gtwr(
            X_design, y, initial_distances, initial_bandwidth
        )
        n_samples, n_parameters = X_design.shape
        bandwidth_history: List[np.ndarray] = []
        tau_history: List[np.ndarray] = []
        convergence_history: List[float] = []
        previous_rss = float(np.dot(residuals, residuals))
        stable_counter = 0
        converged = False
        current_bandwidths = np.full(
            n_parameters,
            initial_bandwidth,
            dtype=int if self.adaptive else float,
        )
        current_taus = np.full(n_parameters, initial_tau, dtype=float)

        for iteration in range(1, self.max_iter + 1):
            old_contributions = contributions.copy()
            new_contributions = np.zeros_like(contributions)
            new_params = np.zeros_like(params)
            iteration_bandwidths = np.empty_like(current_bandwidths)
            iteration_taus = np.empty_like(current_taus)

            for parameter_index in range(n_parameters):
                partial_response = old_contributions[:, parameter_index] + residuals
                x_j = X_design[:, parameter_index]
                if manual_bandwidths is not None and manual_taus is not None:
                    bandwidth = manual_bandwidths[parameter_index]
                    tau = manual_taus[parameter_index]
                elif stable_counter >= self.bws_same_times and bandwidth_history:
                    bandwidth = current_bandwidths[parameter_index]
                    tau = current_taus[parameter_index]
                else:
                    bandwidth, tau, _ = self._select_scale(
                        x_j.reshape(-1, 1),
                        partial_response,
                        spatial_distances,
                        temporal_distances,
                        self.bandwidth_range,
                        initial_tau=current_taus[parameter_index],
                    )
                bandwidth = self._validate_bandwidth_value(
                    bandwidth,
                    n_samples=n_samples,
                    minimum_adaptive=2,
                    name=f"bandwidth for parameter {parameter_index}",
                )
                tau = self._validate_tau_value(
                    tau, name=f"tau for parameter {parameter_index}"
                )
                combined = self._combine_distances(
                    spatial_distances, temporal_distances, tau
                )
                param_j, contribution_j = self._fit_univariate_component(
                    x_j, partial_response, combined, bandwidth
                )
                new_params[:, parameter_index] = param_j
                new_contributions[:, parameter_index] = contribution_j
                iteration_bandwidths[parameter_index] = bandwidth
                iteration_taus[parameter_index] = tau
                residuals = partial_response - contribution_j

            if (
                bandwidth_history
                and np.array_equal(bandwidth_history[-1], iteration_bandwidths)
                and np.allclose(
                    tau_history[-1], iteration_taus, rtol=0.0, atol=self.search_tol
                )
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
            current_taus = iteration_taus.copy()
            bandwidth_history.append(iteration_bandwidths.copy())
            tau_history.append(iteration_taus.copy())
            convergence_history.append(float(score))

            if self.verbose:
                bandwidth_text = ", ".join(
                    str(int(value)) if self.adaptive else f"{value:.6g}"
                    for value in iteration_bandwidths
                )
                tau_text = ", ".join(f"{value:.6g}" for value in iteration_taus)
                print(
                    f"MGTWR iteration {iteration}: SOC={score:.8g}; "
                    f"bandwidths=[{bandwidth_text}]; taus=[{tau_text}]"
                )

            if score < self.tol:
                converged = True
                break

        if not converged:
            warnings.warn(
                "MGTWR reached max_iter before satisfying the convergence "
                "tolerance. Inspect convergence_history_ and consider increasing "
                "max_iter.",
                RuntimeWarning,
                stacklevel=2,
            )

        return _MGTWRBackfittingResult(
            params=params,
            contributions=contributions,
            residuals=residuals,
            bandwidths=current_bandwidths,
            taus=current_taus,
            bandwidth_history=np.asarray(bandwidth_history),
            tau_history=np.asarray(tau_history, dtype=float),
            convergence_history=np.asarray(convergence_history, dtype=float),
            n_iter=len(convergence_history),
            converged=converged,
            initial_bandwidth=initial_bandwidth,
            initial_tau=float(initial_tau),
        )

    def _compute_exact_inference(
        self,
        X_design: np.ndarray,
        spatial_distances: np.ndarray,
        temporal_distances: np.ndarray,
        *,
        n_chunks: int,
        compute_covariance: bool,
    ) -> _MGTWRInferenceResult:
        if (
            self.initial_bandwidth_ is None
            or self.initial_tau_ is None
            or self.bandwidth_history_ is None
            or self.tau_history_ is None
        ):
            raise RuntimeError("Backfitting history is unavailable for inference.")
        n_samples, n_parameters = X_design.shape
        n_chunks = min(int(n_chunks), n_samples)
        effective_params = np.zeros(n_parameters, dtype=float)
        covariance_factors = (
            np.zeros((n_samples, n_parameters), dtype=float)
            if compute_covariance
            else None
        )
        influence = np.empty(n_samples, dtype=float)
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

            initial_distances = self._combine_distances(
                spatial_distances, temporal_distances, self.initial_tau_
            )
            for location, distance_row in enumerate(initial_distances):
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
            for iteration_bandwidths, iteration_taus in zip(
                self.bandwidth_history_, self.tau_history_
            ):
                for parameter_index in range(n_parameters):
                    old_partial = partial_R[:, :, parameter_index] + error_operator
                    x_j = X_design[:, parameter_index]
                    combined = self._combine_distances(
                        spatial_distances,
                        temporal_distances,
                        float(iteration_taus[parameter_index]),
                    )
                    bandwidth = iteration_bandwidths[parameter_index]
                    for location, distance_row in enumerate(combined):
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

        return _MGTWRInferenceResult(
            effective_params_by_variable=effective_params,
            covariance_factors=covariance_factors,
            influence=influence,
            trace_S=float(np.sum(effective_params)),
            trace_StS=float(trace_sts),
            hat_matrix=None,
            partial_hat_matrices=None,
        )

    def fit(
        self,
        X: ArrayLike,
        y: VectorLike,
        coords: ArrayLike,
        times: VectorLike,
    ) -> "MGTWR":
        """Fit MGTWR and replace all prior fitted state atomically.

        Args:
            X: Predictor matrix with shape ``(n_samples, n_features)``.
            y: Response vector with shape ``(n_samples,)``.
            coords: Spatial coordinates with shape ``(n_samples, 2)``.
            times: Numeric time coordinate with one value per observation.

        Returns:
            The fitted model instance.
        """
        self._reset_fit_state()
        try:
            X_arr, y_arr, coords_arr = self._validate_inputs(X, y, coords)
            times_arr = np.asarray(times, dtype=float)
            if times_arr.ndim == 2 and 1 in times_arr.shape:
                times_arr = times_arr.reshape(-1)
            if times_arr.ndim != 1:
                raise ValueError("times must be one-dimensional.")
            if times_arr.shape[0] != X_arr.shape[0]:
                raise ValueError(
                    "X, y, coords, and times must contain the same number of rows."
                )
            if not np.all(np.isfinite(times_arr)):
                raise ValueError("times must contain only finite values.")

            feature_names = (
                None
                if self.feature_names_in_ is None
                else self.feature_names_in_.copy()
            )
            self._store_training_data(X_arr, y_arr, coords_arr, copy=True)
            self.feature_names_in_ = feature_names
            self.times_train_ = times_arr.copy()
            X_design = (
                add_intercept(self.X_train_) if self.fit_intercept else self.X_train_
            )
            n_samples, n_parameters = X_design.shape
            if n_samples <= n_parameters + 2:
                raise ValueError(
                    "MGTWR requires more observations than fitted parameters plus two."
                )

            self.kernel_func_ = get_kernel_function(self.kernel)
            spatial_distances = np.asarray(
                compute_distance_matrix(
                    self.coords_train_,
                    self.coords_train_,
                    metric=self.distance_metric,
                ),
                dtype=float,
            )
            temporal_distances = self._temporal_distances(self.times_train_)
            manual_bandwidths = self._resolve_manual_bandwidths(
                n_parameters=n_parameters,
                n_samples=n_samples,
            )
            manual_taus = self._resolve_manual_taus(n_parameters=n_parameters)

            if self.init_bandwidth is not None:
                initial_bandwidth = self._validate_bandwidth_value(
                    self.init_bandwidth,
                    n_samples=n_samples,
                    minimum_adaptive=n_parameters + 1,
                    name="init_bandwidth",
                )
            elif manual_bandwidths is not None:
                median_bandwidth = float(np.median(manual_bandwidths))
                initial_bandwidth = self._validate_bandwidth_value(
                    (
                        max(n_parameters + 1, int(round(median_bandwidth)))
                        if self.adaptive
                        else median_bandwidth
                    ),
                    n_samples=n_samples,
                    minimum_adaptive=n_parameters + 1,
                    name="initial bandwidth",
                )
            else:
                initial_bandwidth = None

            if self.init_tau is not None:
                initial_tau = self._validate_tau_value(self.init_tau, name="init_tau")
            elif manual_taus is not None:
                initial_tau = float(np.median(manual_taus))
            else:
                initial_tau = None

            if initial_bandwidth is None or initial_tau is None:
                selected_bandwidth, selected_tau, _ = self._select_scale(
                    X_design,
                    self.y_train_,
                    spatial_distances,
                    temporal_distances,
                    self.bandwidth_range,
                    initial_tau=initial_tau,
                )
                if initial_bandwidth is None:
                    initial_bandwidth = self._validate_bandwidth_value(
                        selected_bandwidth,
                        n_samples=n_samples,
                        minimum_adaptive=n_parameters + 1,
                        name="initial selected bandwidth",
                    )
                if initial_tau is None:
                    initial_tau = selected_tau

            self.initial_bandwidth_ = initial_bandwidth
            self.initial_tau_ = float(initial_tau)
            self.bandwidth_ = initial_bandwidth

            backfit = self._backfit(
                X_design,
                self.y_train_,
                spatial_distances,
                temporal_distances,
                initial_bandwidth,
                float(initial_tau),
                manual_bandwidths,
                manual_taus,
            )
            self.bandwidths_ = backfit.bandwidths.copy()
            self.taus_ = backfit.taus.copy()
            self.temporal_bandwidths_ = np.divide(
                self.bandwidths_.astype(float),
                np.sqrt(self.taus_),
                out=np.full(self.taus_.shape, np.inf, dtype=float),
                where=self.taus_ > 0.0,
            )
            self.bandwidth_history_ = backfit.bandwidth_history.copy()
            self.tau_history_ = backfit.tau_history.copy()
            self.convergence_history_ = backfit.convergence_history.copy()
            self.n_iter_ = backfit.n_iter
            self.converged_ = backfit.converged
            self.parameter_contributions_ = backfit.contributions.copy()
            self.params_ = backfit.params.copy()

            if self.fit_intercept:
                self.intercept_ = self.params_[:, 0].copy()
                self.coef_ = self.params_[:, 1:].copy()
            else:
                self.intercept_ = np.zeros(n_samples, dtype=float)
                self.coef_ = self.params_.copy()
            self.fitted_values_ = np.sum(backfit.contributions, axis=1)
            self.residuals_ = self.y_train_ - self.fitted_values_
            self.local_r2_ = None

            rss = float(np.dot(self.residuals_, self.residuals_))
            tss = float(
                np.dot(
                    self.y_train_ - np.mean(self.y_train_),
                    self.y_train_ - np.mean(self.y_train_),
                )
            )
            self.rss_ = rss
            self.r2_ = float(1.0 - rss / tss) if tss > 0.0 else np.nan
            self.inference_enabled_ = self.calculate_inference

            if self.calculate_inference:
                inference = self._compute_exact_inference(
                    X_design,
                    spatial_distances,
                    temporal_distances,
                    n_chunks=self.n_chunks,
                    compute_covariance=True,
                )
                self.effective_params_by_variable_ = (
                    inference.effective_params_by_variable.copy()
                )
                self.ENP_j_ = self.effective_params_by_variable_
                self.effective_params_ = float(
                    np.sum(self.effective_params_by_variable_)
                )
                self.hat_matrix_ = inference.hat_matrix
                self.partial_hat_matrices_ = inference.partial_hat_matrices
                self.diagnostics_ = compute_diagnostics(
                    self.y_train_,
                    self.fitted_values_,
                    compute_gwr_stats=True,
                    trace_S=inference.trace_S,
                    trace_StS=inference.trace_StS,
                )
                self._set_inference_results(inference, self.params_)
                self.aic_ = float(self.diagnostics_["aic"])
                self.aicc_ = float(self.diagnostics_["aicc"])
                self.bic_ = float(self.diagnostics_["bic"])
                alpha = np.asarray([0.10, 0.05, 0.01], dtype=float)
                safe_enp = np.maximum(
                    self.effective_params_by_variable_, np.finfo(float).eps
                )
                self.adjusted_alpha_by_variable_ = alpha / safe_enp[:, None]
                self.critical_t_values_ = student_t.ppf(
                    1.0 - self.adjusted_alpha_by_variable_[:, 1] / 2.0,
                    max(self.n_samples_ - 1, 1),
                )
            else:
                self.diagnostics_ = compute_diagnostics(
                    self.y_train_,
                    self.fitted_values_,
                    n_features=n_parameters,
                )

            self._mark_fitted()
            return self
        except Exception:
            self._reset_fit_state()
            raise

    def predict(self, X: ArrayLike, coords: ArrayLike, times: VectorLike) -> np.ndarray:
        """Reject unsupported independent-target prediction."""
        self._check_is_fitted()
        raise NotImplementedError(
            "Out-of-sample MGTWR prediction is not implemented. Use "
            "fitted_values_ for calibration locations."
        )

    def to_frame(self) -> pd.DataFrame:
        """Return calibration-location parameters and diagnostics."""
        frame = super().to_frame()
        if self.times_train_ is not None:
            frame.insert(self.coords_train_.shape[1], "time", self.times_train_)
        return frame

    def summary(self) -> str:
        """Return fitted model diagnostics as a plain-text table."""
        self._check_is_fitted()
        if (
            self.bandwidths_ is None
            or self.taus_ is None
            or self.temporal_bandwidths_ is None
        ):
            raise RuntimeError("MGTWR scale results are unavailable.")
        temporal_bandwidths = self.temporal_bandwidths_
        return format_summary(
            "MGTWR Summary",
            {
                "n_samples": int(self.n_samples_),
                "n_features": int(self.n_features_in_),
                "fit_intercept": bool(self.fit_intercept),
                "initial_bandwidth": self.initial_bandwidth_,
                "initial_tau": self.initial_tau_,
                "bandwidths": self.bandwidths_.tolist(),
                "taus": self.taus_.tolist(),
                "temporal_bandwidths": temporal_bandwidths.tolist(),
                "adaptive": bool(self.adaptive),
                "kernel": self.kernel,
                "iterations": int(self.n_iter_),
                "converged": bool(self.converged_),
                "rss": self.rss_,
                "r2": self.r2_,
                "sigma2": self.sigma2_,
                "effective_params": (
                    None
                    if self.effective_params_by_variable_ is None
                    else float(np.sum(self.effective_params_by_variable_))
                ),
                "aic": self.aic_,
                "aicc": self.aicc_,
                "bic": self.bic_,
            },
        )
