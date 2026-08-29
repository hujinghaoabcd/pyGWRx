# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Bandwidth selection for geographically weighted models.

The selectors in this module evaluate fixed-distance and adaptive-neighbor bandwidths using cross-validation and information criteria.

Author:
    Jinghao Hu
"""

from __future__ import annotations

__author__ = "Jinghao Hu"
__license__ = "MIT"

from abc import ABC, abstractmethod
from typing import Callable, Optional, Tuple, Union

import numpy as np

from pygwrx.core.solver import weighted_least_squares

Bandwidth = Union[int, float]
BandwidthRange = Optional[Tuple[float, float]]
KernelFunction = Callable[[np.ndarray, float], np.ndarray]


__all__ = [
    "BandwidthSelector",
    "CrossValidationSelector",
    "AICSelector",
    "BICSelector",
    "BANDWIDTH_SELECTORS",
    "get_bandwidth_selector",
]


class _InvalidCandidateError(RuntimeError):
    """Internal exception used when a candidate bandwidth is not estimable."""


# A single numerical regularization value is used for both coefficient fitting and
# hat-matrix calculations, so the fitted values and trace(S) refer to the same smoother.
_RIDGE = 0.0


def _validate_positive_int(value: int, name: str, minimum: int = 1) -> int:
    """Validate an integer configuration parameter."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer.")

    value = int(value)
    if value < minimum:
        raise ValueError(f"{name} must be greater than or equal to {minimum}.")
    return value


def _validate_bool(value: bool, name: str) -> bool:
    """Validate a boolean configuration parameter."""
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a boolean.")
    return bool(value)


def _validate_selector_inputs(
    X: np.ndarray,
    y: np.ndarray,
    coords: np.ndarray,
    kernel_func: KernelFunction,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate and normalize the common selector inputs."""
    try:
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        coords_arr = np.asarray(coords, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError("X, y, and coords must contain numeric values.") from exc

    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(-1, 1)
    elif X_arr.ndim != 2:
        raise ValueError("X must be a one- or two-dimensional array.")

    if y_arr.ndim == 2 and y_arr.shape[1] == 1:
        y_arr = y_arr[:, 0]
    elif y_arr.ndim != 1:
        raise ValueError("y must be one-dimensional or a single-column array.")

    if coords_arr.ndim == 1:
        coords_arr = coords_arr.reshape(1, -1)
    elif coords_arr.ndim != 2:
        raise ValueError("coords must be a one- or two-dimensional array.")

    n_samples = X_arr.shape[0]
    if n_samples == 0:
        raise ValueError("X, y, and coords must contain at least one sample.")
    if X_arr.shape[1] == 0:
        raise ValueError("X must contain at least one feature.")
    if y_arr.shape[0] != n_samples or coords_arr.shape[0] != n_samples:
        raise ValueError("X, y, and coords must contain the same number of samples.")
    if coords_arr.shape[1] == 0:
        raise ValueError("coords must contain at least one coordinate dimension.")

    if not np.all(np.isfinite(X_arr)):
        raise ValueError("X contains NaN or infinite values.")
    if not np.all(np.isfinite(y_arr)):
        raise ValueError("y contains NaN or infinite values.")
    if not np.all(np.isfinite(coords_arr)):
        raise ValueError("coords contains NaN or infinite values.")
    if not callable(kernel_func):
        raise TypeError("kernel_func must be callable.")

    return X_arr, y_arr, coords_arr


def _validate_bandwidth_range(
    bandwidth_range: Tuple[float, float],
    *,
    adaptive: bool,
    n_samples: int,
    n_features: int,
) -> tuple[Bandwidth, Bandwidth]:
    """Validate a user-supplied fixed or adaptive bandwidth interval."""
    if not isinstance(bandwidth_range, (tuple, list)) or len(bandwidth_range) != 2:
        raise TypeError("bandwidth_range must be a two-element tuple or list.")

    try:
        lower_raw = float(bandwidth_range[0])
        upper_raw = float(bandwidth_range[1])
    except (TypeError, ValueError) as exc:
        raise TypeError("bandwidth_range values must be real numbers.") from exc

    if not np.isfinite(lower_raw) or not np.isfinite(upper_raw):
        raise ValueError("bandwidth_range values must be finite.")

    if adaptive:
        # k is the neighbour-order bandwidth used by the rest of PyGWRx.  The range
        # includes the zero-distance observation at training locations, matching the
        # current local_regression/compute_hat_matrix semantics.
        min_allowed = max(n_features + 1, 2)
        max_allowed = n_samples

        if min_allowed > max_allowed:
            raise ValueError(
                "Adaptive bandwidth selection requires more samples than design-matrix "
                "columns so that leave-one-out local regressions are estimable."
            )

        if lower_raw > upper_raw:
            raise ValueError(
                "bandwidth_range lower bound must not exceed its upper bound."
            )

        lower = max(int(np.ceil(lower_raw)), min_allowed)
        upper = min(int(np.floor(upper_raw)), max_allowed)

        if lower > upper:
            raise ValueError(
                "The adaptive bandwidth range contains no valid integer k values after "
                f"enforcing {min_allowed} <= k <= {max_allowed}."
            )
        return lower, upper

    if lower_raw >= upper_raw:
        raise ValueError(
            "bandwidth_range lower bound must be smaller than its upper bound."
        )
    if lower_raw <= 0:
        raise ValueError("Fixed bandwidth bounds must be greater than zero.")
    return lower_raw, upper_raw


def _automatic_bandwidth_range(
    distances: np.ndarray,
    *,
    adaptive: bool,
    n_samples: int,
    n_features: int,
) -> tuple[Bandwidth, Bandwidth]:
    """Derive a valid search interval from a precomputed distance matrix."""
    if adaptive:
        lower = max(n_features + 1, 2, int(np.ceil(0.05 * n_samples)))
        upper = n_samples
        if lower > upper:
            raise ValueError(
                "Adaptive bandwidth selection is not possible: the sample size is too "
                "small for the number of design-matrix columns."
            )
        return lower, upper

    upper_triangle = distances[np.triu_indices_from(distances, k=1)]
    positive_distances = upper_triangle[
        np.isfinite(upper_triangle) & (upper_triangle > 0)
    ]

    if positive_distances.size == 0:
        raise ValueError(
            "Cannot select a fixed bandwidth because all pairwise coordinate distances "
            "are zero. Use distinct coordinates or an adaptive specification "
            "with valid non-zero neighbour distances."
        )

    lower = float(np.percentile(positive_distances, 5))
    upper = float(np.percentile(positive_distances, 95))

    # Percentiles can coincide for regular or heavily duplicated coordinates.  Build a
    # small but valid interval while retaining the observed spatial scale.
    if not np.isfinite(lower) or lower <= 0:
        lower = float(np.min(positive_distances))
    if not np.isfinite(upper) or upper <= 0:
        upper = float(np.max(positive_distances))
    if lower >= upper:
        scale = max(abs(lower), 1.0)
        lower = max(np.nextafter(0.0, 1.0), lower - 0.01 * scale)
        upper = upper + 0.01 * scale

    return lower, upper


def _normalize_candidate(
    bandwidth: float,
    *,
    adaptive: bool,
    lower: Bandwidth,
    upper: Bandwidth,
) -> Bandwidth:
    """Normalize an optimizer candidate to the valid fixed/adaptive domain."""
    if not np.isfinite(bandwidth):
        raise _InvalidCandidateError("Candidate bandwidth is not finite.")

    if adaptive:
        return int(np.clip(int(round(float(bandwidth))), int(lower), int(upper)))

    bandwidth = float(bandwidth)
    if bandwidth <= 0:
        raise _InvalidCandidateError("Fixed candidate bandwidth must be positive.")
    return float(np.clip(bandwidth, float(lower), float(upper)))


def _adaptive_distance_bandwidth(distances: np.ndarray, k: int) -> float:
    """Return a strictly positive distance scale for an adaptive k bandwidth."""
    if k < 1 or k > distances.size:
        raise _InvalidCandidateError(
            f"Adaptive bandwidth k must satisfy 1 <= k <= {distances.size}; got {k}."
        )

    bandwidth = float(np.partition(distances, k - 1)[k - 1])

    # Duplicate coordinates can place the k-th neighbour at distance zero.
    # In that case,
    # use the smallest positive distance.  If no positive distance exists, no distance-
    # based kernel can be evaluated meaningfully.
    if bandwidth <= 0:
        positive = distances[distances > 0]
        if positive.size == 0:
            raise _InvalidCandidateError(
                "All distances at this regression location are zero."
            )
        bandwidth = float(np.min(positive))

    # Compact kernels assign zero weight exactly at d == bandwidth.  Moving by one
    # representable float includes the k-th boundary neighbour without changing scale.
    return float(np.nextafter(bandwidth, np.inf))


def _kernel_weights(
    distances: np.ndarray,
    bandwidth: Bandwidth,
    *,
    adaptive: bool,
    kernel_func: KernelFunction,
) -> np.ndarray:
    """Compute and validate one row of spatial kernel weights."""
    if adaptive:
        distance_bandwidth = _adaptive_distance_bandwidth(distances, int(bandwidth))
    else:
        distance_bandwidth = float(bandwidth)

    weights = np.asarray(kernel_func(distances, distance_bandwidth), dtype=float)
    if weights.shape != distances.shape:
        raise ValueError(
            "kernel_func must return a weight array with the same shape as distances."
        )
    if not np.all(np.isfinite(weights)):
        raise ValueError("kernel_func returned NaN or infinite weights.")
    if np.any(weights < 0):
        raise ValueError("kernel_func returned negative weights.")
    return weights


def _fit_local_model(
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    *,
    target_row: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """Fit an unpenalized local weighted model for bandwidth scoring."""
    n_features = X.shape[1]
    positive = weights > 0.0
    if np.count_nonzero(positive) < n_features:
        raise _InvalidCandidateError(
            "Candidate bandwidth supplies fewer positive-weight observations than design columns."
        )
    Xw = X[positive] * np.sqrt(weights[positive])[:, None]
    if np.linalg.matrix_rank(Xw) < n_features:
        raise _InvalidCandidateError(
            "Candidate bandwidth produces a rank-deficient weighted design."
        )
    try:
        beta, inverse_normal = weighted_least_squares(X, y, weights, ridge=0.0)
    except np.linalg.LinAlgError as exc:
        raise _InvalidCandidateError("Local weighted solve failed.") from exc
    if target_row is None:
        return beta, None
    hat_row = target_row @ inverse_normal @ (X.T * weights)
    if not np.all(np.isfinite(hat_row)):
        raise _InvalidCandidateError("Hat-matrix row contains invalid values.")
    return beta, hat_row


def _integer_grid(lower: int, upper: int, n_intervals: int) -> np.ndarray:
    """Create unique integer candidates including both endpoints."""
    span = upper - lower + 1
    if n_intervals >= span:
        return np.arange(lower, upper + 1, dtype=int)

    candidates = np.rint(np.linspace(lower, upper, n_intervals)).astype(int)
    candidates = np.unique(np.concatenate(([lower], candidates, [upper])))
    return candidates


def _select_best_candidates(
    candidates: np.ndarray,
    objective: Callable[[Bandwidth], float],
) -> tuple[Bandwidth, float]:
    """Evaluate candidates and return the best finite result."""
    best_bandwidth: Optional[Bandwidth] = None
    best_score = np.inf

    for candidate in candidates:
        value: Bandwidth
        if np.issubdtype(np.asarray(candidate).dtype, np.integer):
            value = int(candidate)
        else:
            value = float(candidate)
        score = float(objective(value))
        if np.isfinite(score) and score < best_score:
            best_bandwidth = value
            best_score = score

    if best_bandwidth is None:
        raise RuntimeError(
            "Bandwidth selection failed for every candidate in the range."
        )
    return best_bandwidth, best_score


class BandwidthSelector(ABC):
    """Abstract base class for bandwidth selection methods."""

    @abstractmethod
    def select(
        self,
        X: np.ndarray,
        y: np.ndarray,
        coords: np.ndarray,
        kernel_func: KernelFunction,
        bandwidth_range: BandwidthRange = None,
        distance_metric: str = "euclidean",
    ) -> Bandwidth:
        """Select an optimal fixed-distance or adaptive integer bandwidth."""
        raise NotImplementedError


class _BaseSelector(BandwidthSelector):
    """Shared validation, range construction, and one-dimensional search logic."""

    def __init__(
        self,
        n_intervals: int = 20,
        optimization_method: str = "golden_section",
        adaptive: bool = False,
        verbose: bool = False,
    ) -> None:
        self.n_intervals = _validate_positive_int(n_intervals, "n_intervals", minimum=2)
        self.adaptive = _validate_bool(adaptive, "adaptive")
        self.verbose = _validate_bool(verbose, "verbose")

        if not isinstance(optimization_method, str):
            raise TypeError("optimization_method must be a string.")
        optimization_method = optimization_method.strip().lower()
        if optimization_method not in {"grid", "golden_section", "brent"}:
            raise ValueError(
                "optimization_method must be 'grid', 'golden_section', or 'brent'; "
                f"got {optimization_method!r}."
            )
        self.optimization_method = optimization_method

    def _prepare(
        self,
        X: np.ndarray,
        y: np.ndarray,
        coords: np.ndarray,
        kernel_func: KernelFunction,
        bandwidth_range: BandwidthRange,
        distance_metric: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Bandwidth, Bandwidth]:
        from pygwrx.core.utils import compute_distance_matrix

        if not isinstance(distance_metric, str):
            raise TypeError("distance_metric must be a string.")

        X_arr, y_arr, coords_arr = _validate_selector_inputs(X, y, coords, kernel_func)
        distances = compute_distance_matrix(
            coords_arr,
            coords_arr,
            metric=distance_metric,
        )
        distances = np.asarray(distances, dtype=float)
        if distances.shape != (X_arr.shape[0], X_arr.shape[0]):
            raise ValueError("The computed distance matrix has an invalid shape.")
        if not np.all(np.isfinite(distances)) or np.any(distances < 0):
            raise ValueError("The computed distance matrix contains invalid distances.")

        # Older PyGWRx model classes generate two invalid automatic ranges:
        #   adaptive small-n: (20, n) when n < 20
        #   fixed small-scale coordinates: (1, max_distance) when max_distance < 1
        # Detect only these legacy signatures and rebuild the range here.
        # Other reversed
        # user-supplied ranges still raise a clear error.
        legacy_auto_range = False
        if (
            bandwidth_range is not None
            and isinstance(bandwidth_range, (tuple, list))
            and len(bandwidth_range) == 2
        ):
            try:
                range_lower = float(bandwidth_range[0])
                range_upper = float(bandwidth_range[1])
            except (TypeError, ValueError):
                range_lower = np.nan
                range_upper = np.nan

            if self.adaptive:
                legacy_auto_range = (
                    X_arr.shape[0] < 20
                    and range_lower >= 20
                    and np.isclose(range_upper, X_arr.shape[0])
                )
            else:
                legacy_auto_range = (
                    np.isclose(range_lower, 1.0) and 0 < range_upper < 1.0
                )

        if bandwidth_range is None or legacy_auto_range:
            lower, upper = _automatic_bandwidth_range(
                distances,
                adaptive=self.adaptive,
                n_samples=X_arr.shape[0],
                n_features=X_arr.shape[1],
            )
        else:
            lower, upper = _validate_bandwidth_range(
                bandwidth_range,
                adaptive=self.adaptive,
                n_samples=X_arr.shape[0],
                n_features=X_arr.shape[1],
            )

        return X_arr, y_arr, coords_arr, distances, lower, upper

    def _search(
        self,
        objective_raw: Callable[[Bandwidth], float],
        lower: Bandwidth,
        upper: Bandwidth,
    ) -> tuple[Bandwidth, float]:
        cache: dict[Bandwidth, float] = {}

        def objective(candidate: float) -> float:
            try:
                normalized = _normalize_candidate(
                    candidate,
                    adaptive=self.adaptive,
                    lower=lower,
                    upper=upper,
                )
            except _InvalidCandidateError:
                return np.inf

            if normalized not in cache:
                try:
                    score = float(objective_raw(normalized))
                except _InvalidCandidateError:
                    score = np.inf
                cache[normalized] = score if np.isfinite(score) else np.inf
            return cache[normalized]

        if self.adaptive and int(lower) == int(upper):
            only_candidate = int(lower)
            only_score = objective(only_candidate)
            if not np.isfinite(only_score):
                raise RuntimeError(
                    "The only valid adaptive bandwidth candidate could not "
                    "be estimated."
                )
            return only_candidate, float(only_score)

        if self.optimization_method == "grid":
            if self.adaptive:
                candidates = _integer_grid(int(lower), int(upper), self.n_intervals)
            else:
                candidates = np.linspace(float(lower), float(upper), self.n_intervals)
            best_bandwidth, best_score = _select_best_candidates(candidates, objective)

        elif self.optimization_method == "golden_section":
            from pygwrx.core.optimization import GoldenSectionSearch

            optimizer = GoldenSectionSearch(
                tol=1e-4,
                max_iter=100,
                verbose=self.verbose,
            )
            result = optimizer.minimize(
                objective,
                float(lower),
                float(upper),
                adaptive=self.adaptive,
            )
            if not result.converged or not np.isfinite(result.score):
                raise RuntimeError("Golden-section bandwidth search did not converge.")

            candidate = _normalize_candidate(
                result.value,
                adaptive=self.adaptive,
                lower=lower,
                upper=upper,
            )
            if self.adaptive:
                neighborhood = np.arange(
                    max(int(lower), int(candidate) - 2),
                    min(int(upper), int(candidate) + 2) + 1,
                    dtype=int,
                )
                candidates = np.unique(
                    np.concatenate(([int(lower), int(upper)], neighborhood))
                )
                best_bandwidth, best_score = _select_best_candidates(
                    candidates,
                    objective,
                )
            else:
                best_bandwidth = float(candidate)
                best_score = objective(best_bandwidth)
                if not np.isfinite(best_score):
                    raise RuntimeError(
                        "Golden-section search returned an invalid bandwidth."
                    )

        else:
            from pygwrx.core.optimization import BrentSearch

            optimizer = BrentSearch(tol=1e-5, max_iter=100, verbose=self.verbose)
            result = optimizer.minimize(objective, float(lower), float(upper))
            if not result.converged or not np.isfinite(result.score):
                raise RuntimeError("Brent bandwidth search did not converge.")

            candidate = _normalize_candidate(
                result.value,
                adaptive=self.adaptive,
                lower=lower,
                upper=upper,
            )
            if self.adaptive:
                # Brent is continuous, while adaptive k is discrete.  The rounded result
                # and its immediate integer neighbourhood are evaluated explicitly.
                neighborhood = np.arange(
                    max(int(lower), int(candidate) - 2),
                    min(int(upper), int(candidate) + 2) + 1,
                    dtype=int,
                )
                candidates = np.unique(
                    np.concatenate(([int(lower), int(upper)], neighborhood))
                )
                best_bandwidth, best_score = _select_best_candidates(
                    candidates,
                    objective,
                )
            else:
                best_bandwidth = float(candidate)
                best_score = objective(best_bandwidth)
                if not np.isfinite(best_score):
                    raise RuntimeError("Brent search returned an invalid bandwidth.")

        if self.adaptive:
            return int(best_bandwidth), float(best_score)
        return float(best_bandwidth), float(best_score)

    def _print_header(self, title: str, lower: Bandwidth, upper: Bandwidth) -> None:
        if not self.verbose:
            return
        print(f"\n{title}")
        print(f"  Method: {self.optimization_method}")
        if self.adaptive:
            print(f"  Search range: [{int(lower)}, {int(upper)}]")
            print("  Type: Adaptive (integer neighbour-order bandwidth)")
        else:
            print(f"  Search range: [{float(lower):.6g}, {float(upper):.6g}]")
            print("  Type: Fixed (distance bandwidth)")


class CrossValidationSelector(_BaseSelector):
    """Select bandwidth by strict leave-one-out squared prediction error."""

    def select(
        self,
        X: np.ndarray,
        y: np.ndarray,
        coords: np.ndarray,
        kernel_func: KernelFunction,
        bandwidth_range: BandwidthRange = None,
        distance_metric: str = "euclidean",
    ) -> Bandwidth:
        X_arr, y_arr, _, distances, lower, upper = self._prepare(
            X,
            y,
            coords,
            kernel_func,
            bandwidth_range,
            distance_metric,
        )
        self._print_header("Cross-Validation Bandwidth Selection", lower, upper)

        def cv_objective(bandwidth: Bandwidth) -> float:
            squared_error = 0.0
            for i, dists in enumerate(distances):
                weights = _kernel_weights(
                    dists,
                    bandwidth,
                    adaptive=self.adaptive,
                    kernel_func=kernel_func,
                ).copy()
                # Strict leave-one-out: the focal observation has exactly zero weight.
                weights[i] = 0.0
                beta, _ = _fit_local_model(X_arr, y_arr, weights)
                residual = float(y_arr[i] - X_arr[i] @ beta)
                squared_error += residual * residual
            return squared_error

        best_bandwidth, best_score = self._search(cv_objective, lower, upper)
        if self.verbose:
            label = "Optimal k" if self.adaptive else "Optimal bandwidth"
            print(f"\n{label}: {best_bandwidth}")
            print(f"CV score: {best_score:.6f}")
        return best_bandwidth


class AICSelector(_BaseSelector):
    """Select bandwidth using Gaussian GWR AIC or AICc."""

    def __init__(
        self,
        n_intervals: int = 20,
        corrected: bool = True,
        adaptive: bool = False,
        optimization_method: str = "golden_section",
        verbose: bool = False,
    ) -> None:
        super().__init__(
            n_intervals=n_intervals,
            optimization_method=optimization_method,
            adaptive=adaptive,
            verbose=verbose,
        )
        self.corrected = _validate_bool(corrected, "corrected")

    def select(
        self,
        X: np.ndarray,
        y: np.ndarray,
        coords: np.ndarray,
        kernel_func: KernelFunction,
        bandwidth_range: BandwidthRange = None,
        distance_metric: str = "euclidean",
    ) -> Bandwidth:
        from pygwrx.core.metrics import compute_aic, compute_aicc

        X_arr, y_arr, _, distances, lower, upper = self._prepare(
            X,
            y,
            coords,
            kernel_func,
            bandwidth_range,
            distance_metric,
        )
        criterion = "AICc" if self.corrected else "AIC"
        self._print_header(f"{criterion} Bandwidth Selection", lower, upper)

        def information_objective(bandwidth: Bandwidth) -> float:
            n_samples = y_arr.size
            fitted = np.empty(n_samples, dtype=float)
            trace_s = 0.0

            for i, dists in enumerate(distances):
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

            if self.corrected:
                return float(compute_aicc(y_arr, fitted, trace_s))
            return float(compute_aic(y_arr, fitted, trace_s))

        best_bandwidth, best_score = self._search(information_objective, lower, upper)
        if self.verbose:
            label = "Optimal k" if self.adaptive else "Optimal bandwidth"
            print(f"\n{label}: {best_bandwidth}")
            print(f"{criterion} score: {best_score:.6f}")
        return best_bandwidth


class BICSelector(_BaseSelector):
    """Select bandwidth using Gaussian GWR BIC."""

    def select(
        self,
        X: np.ndarray,
        y: np.ndarray,
        coords: np.ndarray,
        kernel_func: KernelFunction,
        bandwidth_range: BandwidthRange = None,
        distance_metric: str = "euclidean",
    ) -> Bandwidth:
        from pygwrx.core.metrics import compute_bic

        X_arr, y_arr, _, distances, lower, upper = self._prepare(
            X,
            y,
            coords,
            kernel_func,
            bandwidth_range,
            distance_metric,
        )
        self._print_header("BIC Bandwidth Selection", lower, upper)

        def bic_objective(bandwidth: Bandwidth) -> float:
            n_samples = y_arr.size
            fitted = np.empty(n_samples, dtype=float)
            trace_s = 0.0

            for i, dists in enumerate(distances):
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

        best_bandwidth, best_score = self._search(bic_objective, lower, upper)
        if self.verbose:
            label = "Optimal k" if self.adaptive else "Optimal bandwidth"
            print(f"\n{label}: {best_bandwidth}")
            print(f"BIC score: {best_score:.6f}")
        return best_bandwidth


BANDWIDTH_SELECTORS = {
    "cv": CrossValidationSelector,
    "aic": AICSelector,
    "aicc": AICSelector,
    "bic": BICSelector,
}


def get_bandwidth_selector(method: str, **kwargs) -> BandwidthSelector:
    """Create a bandwidth selector by method name.

    Constructor parameters belong in ``kwargs``.  Search-time parameters such as
    ``bandwidth_range`` and ``distance_metric`` must be supplied to ``select()``.
    """
    if not isinstance(method, str):
        raise TypeError("method must be a string.")

    method_name = method.strip().lower()
    if method_name not in BANDWIDTH_SELECTORS:
        available = ", ".join(sorted(BANDWIDTH_SELECTORS))
        raise ValueError(
            f"Unknown bandwidth selection method: {method!r}. "
            f"Available methods: {available}."
        )

    if method_name in {"aic", "aicc"}:
        expected_corrected = method_name == "aicc"
        if "corrected" in kwargs:
            supplied = _validate_bool(kwargs.pop("corrected"), "corrected")
            if supplied != expected_corrected:
                raise ValueError(
                    f"method={method_name!r} conflicts with corrected={supplied}."
                )
        return AICSelector(corrected=expected_corrected, **kwargs)

    selector_class = BANDWIDTH_SELECTORS[method_name]
    return selector_class(**kwargs)
