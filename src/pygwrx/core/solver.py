# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Numerical solvers for local weighted regression.

This module provides weighted least-squares estimation, local regression, adaptive-bandwidth conversion, and hat-matrix calculations.

Author:
    Jinghao Hu
"""

from __future__ import annotations

__author__ = "Jinghao Hu"
__license__ = "MIT"

import warnings
from typing import Callable, NamedTuple, Optional, Tuple, Union

import numpy as np

KernelFunction = Callable[[np.ndarray, float], np.ndarray]

__all__ = [
    "weighted_least_squares",
    "local_regression",
    "compute_hat_matrix",
    "adaptive_bandwidth_weights",
]


_DEFAULT_RIDGE = 0.0


class _WeightedLeastSquaresResult(NamedTuple):
    """Private numerical details from one weighted SVD solve."""

    beta: np.ndarray
    inverse_normal: np.ndarray
    rank: int
    singular_values: np.ndarray
    condition_number: float


def _validate_nonnegative_scalar(value: float, name: str) -> float:
    """Validate a finite, non-negative scalar."""
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a real scalar, not bool.")

    value_arr = np.asarray(value)
    if value_arr.ndim != 0:
        raise TypeError(f"{name} must be a scalar value.")

    try:
        value_float = float(value_arr)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a real scalar.") from exc

    if not np.isfinite(value_float):
        raise ValueError(f"{name} must be finite.")
    if value_float < 0:
        raise ValueError(f"{name} must be non-negative.")

    return value_float


def _validate_design_matrix(X: np.ndarray) -> np.ndarray:
    """Validate and normalize a regression design matrix."""
    try:
        X_arr = np.asarray(X, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError("X must contain numeric values.") from exc

    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(-1, 1)
    elif X_arr.ndim != 2:
        raise ValueError("X must be a one- or two-dimensional array.")

    if X_arr.shape[0] == 0:
        raise ValueError("X must contain at least one sample.")
    if X_arr.shape[1] == 0:
        raise ValueError("X must contain at least one feature.")
    if not np.all(np.isfinite(X_arr)):
        raise ValueError("X contains NaN or infinite values.")

    return X_arr


def _validate_response(y: np.ndarray, n_samples: int) -> np.ndarray:
    """Validate a single-response target vector."""
    try:
        y_arr = np.asarray(y, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError("y must contain numeric values.") from exc

    if y_arr.ndim == 2 and y_arr.shape[1] == 1:
        y_arr = y_arr[:, 0]
    elif y_arr.ndim != 1:
        raise ValueError("y must be one-dimensional or a single-column array.")

    if y_arr.shape[0] != n_samples:
        raise ValueError(
            "X and y must contain the same number of samples; "
            f"got {n_samples} and {y_arr.shape[0]}."
        )
    if not np.all(np.isfinite(y_arr)):
        raise ValueError("y contains NaN or infinite values.")

    return y_arr


def _validate_weights(weights: np.ndarray, n_samples: int) -> np.ndarray:
    """Validate non-negative observation weights while preserving exact zeros."""
    try:
        weights_arr = np.asarray(weights, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError("weights must contain numeric values.") from exc

    if weights_arr.ndim == 2 and weights_arr.shape[1] == 1:
        weights_arr = weights_arr[:, 0]
    elif weights_arr.ndim != 1:
        raise ValueError("weights must be one-dimensional or a single-column array.")

    if weights_arr.shape[0] != n_samples:
        raise ValueError(
            "weights must have one value per observation; "
            f"expected {n_samples}, got {weights_arr.shape[0]}."
        )
    if not np.all(np.isfinite(weights_arr)):
        raise ValueError("weights contain NaN or infinite values.")
    if np.any(weights_arr < 0):
        raise ValueError("weights must be non-negative.")
    if not np.any(weights_arr > 0):
        raise ValueError("At least one observation weight must be positive.")

    return weights_arr


def _validate_coordinates(
    coords: np.ndarray,
    *,
    name: str,
    expected_rows: Optional[int] = None,
    expected_dimension: Optional[int] = None,
) -> np.ndarray:
    """Validate a coordinate matrix."""
    try:
        coords_arr = np.asarray(coords, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must contain numeric values.") from exc

    if coords_arr.ndim == 1:
        coords_arr = coords_arr.reshape(1, -1)
    elif coords_arr.ndim != 2:
        raise ValueError(f"{name} must be a one- or two-dimensional array.")

    if coords_arr.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one location.")
    if coords_arr.shape[1] == 0:
        raise ValueError(f"{name} must contain at least one coordinate dimension.")
    if expected_rows is not None and coords_arr.shape[0] != expected_rows:
        raise ValueError(
            f"{name} must contain {expected_rows} rows; got {coords_arr.shape[0]}."
        )
    if expected_dimension is not None and coords_arr.shape[1] != expected_dimension:
        raise ValueError(
            f"{name} must have {expected_dimension} coordinate columns; "
            f"got {coords_arr.shape[1]}."
        )
    if not np.all(np.isfinite(coords_arr)):
        raise ValueError(f"{name} contains NaN or infinite values.")

    return coords_arr


def _validate_kernel(kernel_func: KernelFunction) -> KernelFunction:
    """Validate a kernel callable."""
    if not callable(kernel_func):
        raise TypeError("kernel_func must be callable.")
    return kernel_func


def _validate_fixed_bandwidth(bandwidth: float) -> float:
    """Validate a strictly positive fixed-distance bandwidth."""
    bandwidth_value = _validate_nonnegative_scalar(bandwidth, "bandwidth")
    if bandwidth_value == 0:
        raise ValueError("bandwidth must be greater than zero.")
    return bandwidth_value


def _validate_adaptive_k(k_nearest: Union[int, float], n_samples: int) -> int:
    """Validate an adaptive neighbour-order bandwidth."""
    if isinstance(k_nearest, (bool, np.bool_)):
        raise TypeError("k_nearest must be an integer, not bool.")

    try:
        k_float = float(k_nearest)
    except (TypeError, ValueError) as exc:
        raise TypeError("k_nearest must be an integer.") from exc

    if not np.isfinite(k_float) or not k_float.is_integer():
        raise ValueError("k_nearest must be a finite integer value.")

    k_value = int(k_float)
    if k_value < 1 or k_value > n_samples:
        raise ValueError(
            f"k_nearest must satisfy 1 <= k_nearest <= {n_samples}; got {k_value}."
        )

    return k_value


def _compute_kernel_weights(
    distances: np.ndarray,
    kernel_func: KernelFunction,
    bandwidth: Union[float, int],
    *,
    adaptive: bool,
) -> np.ndarray:
    """Compute and validate one row of kernel weights."""
    if adaptive:
        distance_bandwidth = adaptive_bandwidth_weights(
            distances,
            k_nearest=int(bandwidth),
        )
    else:
        distance_bandwidth = _validate_fixed_bandwidth(bandwidth)

    weights = np.asarray(
        kernel_func(distances, distance_bandwidth),
        dtype=float,
    )

    if weights.shape != distances.shape:
        raise ValueError(
            "kernel_func must return an array with the same shape as distances; "
            f"expected {distances.shape}, got {weights.shape}."
        )
    if not np.all(np.isfinite(weights)):
        raise ValueError("kernel_func returned NaN or infinite weights.")
    if np.any(weights < 0):
        raise ValueError("kernel_func returned negative weights.")
    if not np.any(weights > 0):
        raise ValueError(
            "The selected bandwidth and kernel produced no positive observation weights."
        )

    return weights


def _weighted_least_squares_details(
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    *,
    ridge: float = _DEFAULT_RIDGE,
) -> _WeightedLeastSquaresResult:
    """Solve WLS once by SVD and retain private numerical-rank diagnostics."""
    X_arr = _validate_design_matrix(X)
    y_arr = _validate_response(y, X_arr.shape[0])
    weights_arr = _validate_weights(weights, X_arr.shape[0])
    ridge_value = _validate_nonnegative_scalar(ridge, "ridge")

    positive = weights_arr > 0.0
    sqrt_w = np.sqrt(weights_arr[positive])
    Xw = X_arr[positive] * sqrt_w[:, None]
    yw = y_arr[positive] * sqrt_w

    if ridge_value > 0.0:
        design = np.vstack(
            [Xw, np.sqrt(ridge_value) * np.eye(X_arr.shape[1], dtype=float)]
        )
        response = np.concatenate([yw, np.zeros(X_arr.shape[1], dtype=float)])
    else:
        design = Xw
        response = yw

    u, singular_values, vt = np.linalg.svd(design, full_matrices=False)
    if singular_values.size == 0 or singular_values[0] <= 0.0:
        raise np.linalg.LinAlgError("Weighted design has zero numerical rank.")

    cutoff = np.finfo(float).eps * max(design.shape) * singular_values[0]
    retained = singular_values > cutoff
    rank = int(np.count_nonzero(retained))
    if rank == 0:
        raise np.linalg.LinAlgError("Weighted design has zero numerical rank.")

    singular_retained = singular_values[retained]
    v = vt[retained].T
    u_retained = u[:, retained]
    beta = v @ ((u_retained.T @ response) / singular_retained)
    inverse_normal = (v * (1.0 / singular_retained**2)) @ v.T
    inverse_normal = 0.5 * (inverse_normal + inverse_normal.T)

    condition_number = (
        float(singular_values[0] / singular_retained[-1])
        if rank == design.shape[1]
        else np.inf
    )

    if not np.all(np.isfinite(beta)) or not np.all(np.isfinite(inverse_normal)):
        raise np.linalg.LinAlgError("Weighted SVD solve produced non-finite values.")

    return _WeightedLeastSquaresResult(
        beta=np.asarray(beta, dtype=float),
        inverse_normal=inverse_normal,
        rank=rank,
        singular_values=np.asarray(singular_values, dtype=float),
        condition_number=float(condition_number),
    )


def weighted_least_squares(
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    *,
    ridge: float = _DEFAULT_RIDGE,
) -> Tuple[np.ndarray, np.ndarray]:
    """Solve weighted least squares from ``sqrt(W) @ X``.

    The default ``ridge=0.0`` is standard unpenalized WLS. Positive values
    remain an explicit lower-level ridge option. The public two-value return
    contract is preserved while one internal SVD provides both coefficients and
    the inverse-normal operator.
    """
    result = _weighted_least_squares_details(X, y, weights, ridge=ridge)
    return result.beta, result.inverse_normal


def local_regression(
    X: np.ndarray,
    y: np.ndarray,
    coords: np.ndarray,
    target_coords: np.ndarray,
    kernel_func: KernelFunction,
    bandwidth: float,
    distance_metric: str = "euclidean",
    adaptive: bool = False,
    *,
    ridge: float = _DEFAULT_RIDGE,
) -> np.ndarray:
    """Perform local weighted regression at target locations.

    Args:
        X: Regression design matrix.
        y: Target vector.
        coords: Training coordinates.
        target_coords: Locations where local coefficients are required.
        kernel_func: Function with signature ``kernel_func(distances, bandwidth) -> weights``.
        bandwidth: Fixed distance when ``adaptive=False``; neighbour-order bandwidth ``k`` when
            ``adaptive=True``.
        distance_metric: Distance metric forwarded to ``compute_distance_matrix``.
        adaptive: Whether ``bandwidth`` is interpreted as a neighbour-order bandwidth.
        ridge: Optional non-negative ridge penalty. The default ``0.0`` is standard
            unpenalized WLS; positive values are explicit lower-level regularization.

    Returns:
        local_coefs: Local coefficient vectors.

    Notes:
        If a location has fewer positive-weight observations than design-matrix columns,
        the default path warns and returns the Moore-Penrose minimum-norm unpenalized WLS
        solution. It never copies coefficients from a preceding location and never falls
        back silently to global OLS, so results do not depend on target ordering.
    """
    from pygwrx.core.utils import compute_distance_matrix

    X_arr = _validate_design_matrix(X)
    y_arr = _validate_response(y, X_arr.shape[0])
    coords_arr = _validate_coordinates(
        coords,
        name="coords",
        expected_rows=X_arr.shape[0],
    )
    target_arr = _validate_coordinates(
        target_coords,
        name="target_coords",
        expected_dimension=coords_arr.shape[1],
    )
    kernel = _validate_kernel(kernel_func)
    ridge_value = _validate_nonnegative_scalar(ridge, "ridge")

    if not isinstance(adaptive, (bool, np.bool_)):
        raise TypeError("adaptive must be a boolean.")
    adaptive_value = bool(adaptive)

    if adaptive_value:
        bandwidth_value: Union[float, int] = _validate_adaptive_k(
            bandwidth,
            X_arr.shape[0],
        )
    else:
        bandwidth_value = _validate_fixed_bandwidth(bandwidth)

    distances = np.asarray(
        compute_distance_matrix(
            target_arr,
            coords_arr,
            metric=distance_metric,
        ),
        dtype=float,
    )

    expected_shape = (target_arr.shape[0], coords_arr.shape[0])
    if distances.shape != expected_shape:
        raise ValueError(
            "compute_distance_matrix returned an unexpected shape; "
            f"expected {expected_shape}, got {distances.shape}."
        )
    if not np.all(np.isfinite(distances)) or np.any(distances < 0):
        raise ValueError("Distance calculation produced invalid values.")

    local_coefs = np.empty(
        (target_arr.shape[0], X_arr.shape[1]),
        dtype=float,
    )

    for i, dists in enumerate(distances):
        try:
            weights = _compute_kernel_weights(
                dists,
                kernel,
                bandwidth_value,
                adaptive=adaptive_value,
            )

            n_positive = int(np.count_nonzero(weights > 0))
            if n_positive < X_arr.shape[1]:
                warnings.warn(
                    f"Location {i}: only {n_positive} positive-weight observations "
                    f"are available for {X_arr.shape[1]} design-matrix columns. "
                    "Returning a minimum-norm unpenalized local solution; consider increasing "
                    "the bandwidth.",
                    RuntimeWarning,
                    stacklevel=2,
                )

            beta, _ = weighted_least_squares(
                X_arr,
                y_arr,
                weights,
                ridge=ridge_value,
            )
        except (ValueError, TypeError, np.linalg.LinAlgError) as exc:
            raise RuntimeError(
                f"Local regression failed at target location {i}: {exc}"
            ) from exc

        local_coefs[i] = beta

    return local_coefs


def compute_hat_matrix(
    X: np.ndarray,
    coords: np.ndarray,
    kernel_func: KernelFunction,
    bandwidth: float,
    distance_metric: str = "euclidean",
    adaptive: bool = False,
    *,
    ridge: float = _DEFAULT_RIDGE,
) -> np.ndarray:
    """Compute the GWR hat matrix ``S`` such that ``y_hat = S @ y``.

    Args:
        X: Regression design matrix.
        coords: Training coordinates.
        kernel_func: Spatial kernel function.
        bandwidth: Fixed distance or adaptive neighbour-order bandwidth.
        distance_metric: Distance metric forwarded to ``compute_distance_matrix``.
        adaptive: Whether ``bandwidth`` is an adaptive neighbour-order bandwidth.
        ridge: Optional non-negative ridge penalty. The default ``0.0`` is unpenalized;
            positive values explicitly regularize the standalone smoother calculation.

    Returns:
        hat_matrix: Full smoother matrix.

    Notes:
        The full matrix requires ``8 * n_samples**2`` bytes for float64 storage, excluding
        the distance matrix and temporary arrays. Large-data models should eventually use
        trace-only or chunked diagnostics instead.
    """
    from pygwrx.core.utils import compute_distance_matrix

    X_arr = _validate_design_matrix(X)
    coords_arr = _validate_coordinates(
        coords,
        name="coords",
        expected_rows=X_arr.shape[0],
    )
    kernel = _validate_kernel(kernel_func)
    ridge_value = _validate_nonnegative_scalar(ridge, "ridge")

    if not isinstance(adaptive, (bool, np.bool_)):
        raise TypeError("adaptive must be a boolean.")
    adaptive_value = bool(adaptive)

    if adaptive_value:
        bandwidth_value: Union[float, int] = _validate_adaptive_k(
            bandwidth,
            X_arr.shape[0],
        )
    else:
        bandwidth_value = _validate_fixed_bandwidth(bandwidth)

    distances = np.asarray(
        compute_distance_matrix(
            coords_arr,
            coords_arr,
            metric=distance_metric,
        ),
        dtype=float,
    )

    expected_shape = (X_arr.shape[0], X_arr.shape[0])
    if distances.shape != expected_shape:
        raise ValueError(
            "compute_distance_matrix returned an unexpected shape; "
            f"expected {expected_shape}, got {distances.shape}."
        )
    if not np.all(np.isfinite(distances)) or np.any(distances < 0):
        raise ValueError("Distance calculation produced invalid values.")

    hat_matrix = np.empty(expected_shape, dtype=float)

    # The response is irrelevant for the smoother matrix. A zero vector lets us
    # reuse weighted_least_squares() so the hat matrix uses exactly the same
    # SVD-based inverse-normal operator as local coefficient estimation.
    dummy_y = np.zeros(X_arr.shape[0], dtype=float)

    for i, dists in enumerate(distances):
        try:
            weights = _compute_kernel_weights(
                dists,
                kernel,
                bandwidth_value,
                adaptive=adaptive_value,
            )

            n_positive = int(np.count_nonzero(weights > 0))
            if n_positive < X_arr.shape[1]:
                warnings.warn(
                    f"Location {i}: only {n_positive} positive-weight observations "
                    f"are available for {X_arr.shape[1]} design-matrix columns. "
                    "The corresponding hat-matrix row uses the unpenalized WLS operator.",
                    RuntimeWarning,
                    stacklevel=2,
                )

            _, inverse_normal = weighted_least_squares(
                X_arr,
                dummy_y,
                weights,
                ridge=ridge_value,
            )
            XtW = X_arr.T * weights
            hat_row = X_arr[i] @ inverse_normal @ XtW

            if not np.all(np.isfinite(hat_row)):
                raise np.linalg.LinAlgError(
                    "The hat-matrix row contains non-finite values."
                )
        except (ValueError, TypeError, np.linalg.LinAlgError) as exc:
            raise RuntimeError(
                f"Hat-matrix computation failed at location {i}: {exc}"
            ) from exc

        hat_matrix[i] = hat_row

    return hat_matrix


def adaptive_bandwidth_weights(
    distances: np.ndarray,
    k_nearest: int,
) -> float:
    """Convert an adaptive neighbour-order bandwidth into a distance scale.

    Args:
        distances: Non-negative distances from one regression location to all observations.
        k_nearest: One-based neighbour order used to determine the local distance scale. The
            current PyGWRx convention includes a zero-distance self observation when the
            regression location is one of the training locations.

    Returns:
        bandwidth: Strictly positive distance scale corresponding to the k-th ordered distance.

    Notes:
        ``np.partition`` is used for expected O(n) selection. If duplicate coordinates put
        the requested ordered distance at zero, the smallest positive distance is used. The
        result is advanced by one representable float with ``np.nextafter`` so compact
        kernels assign a positive (possibly tiny) weight to the boundary neighbour instead
        of excluding it exactly at ``d == bandwidth``.
    """
    try:
        distances_arr = np.asarray(distances, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError("distances must contain numeric values.") from exc

    if distances_arr.ndim != 1:
        raise ValueError("distances must be a one-dimensional array.")
    if distances_arr.size == 0:
        raise ValueError("distances must contain at least one value.")
    if not np.all(np.isfinite(distances_arr)):
        raise ValueError("distances contain NaN or infinite values.")
    if np.any(distances_arr < 0):
        raise ValueError("distances must be non-negative.")

    k_value = _validate_adaptive_k(k_nearest, distances_arr.size)
    distance_bandwidth = float(np.partition(distances_arr, k_value - 1)[k_value - 1])

    if distance_bandwidth <= 0:
        positive_distances = distances_arr[distances_arr > 0]
        if positive_distances.size == 0:
            raise ValueError(
                "Adaptive bandwidth is undefined because all distances are zero."
            )
        distance_bandwidth = float(np.min(positive_distances))

    return float(np.nextafter(distance_bandwidth, np.inf))
