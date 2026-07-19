# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Private numerical core for mixed geographically weighted regression.

The public estimator lives in :mod:`pygwrx.models.mixed_gwr`.  This module
implements the semiparametric partial-regression algorithm used by GWmodel's
``gwr.mixed`` implementation and is intentionally private.

Author:
    Jinghao Hu
"""

from __future__ import annotations

__author__ = "Jinghao Hu"
__license__ = "MIT"

from typing import Callable, Dict, Tuple, Union

import numpy as np

from pygwrx.core.solver import adaptive_bandwidth_weights, weighted_least_squares

KernelFunction = Callable[[np.ndarray, float], np.ndarray]
Bandwidth = Union[int, float]


def _validate_matrix(
    value: np.ndarray, name: str, *, n_rows: int | None = None
) -> np.ndarray:
    """Return a finite two-dimensional floating-point matrix."""
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must contain numeric values.") from exc
    if array.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional.")
    if n_rows is not None and array.shape[0] != n_rows:
        raise ValueError(f"{name} must contain {n_rows} rows; got {array.shape[0]}.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or infinite values.")
    return array


def _validate_vector(value: np.ndarray, name: str, *, n_rows: int) -> np.ndarray:
    """Return a finite one-dimensional floating-point vector."""
    try:
        array = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must contain numeric values.") from exc
    if array.shape[0] != n_rows:
        raise ValueError(f"{name} must contain {n_rows} values; got {array.shape[0]}.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or infinite values.")
    return array


def _validate_ridge(ridge: float) -> float:
    """Validate a non-negative regularization value."""
    if isinstance(ridge, (bool, np.bool_)):
        raise TypeError("ridge must be a real scalar, not bool.")
    try:
        value = float(ridge)
    except (TypeError, ValueError) as exc:
        raise TypeError("ridge must be a real scalar.") from exc
    if not np.isfinite(value) or value < 0:
        raise ValueError("ridge must be finite and non-negative.")
    return value


def _mixed_weights(
    kernel_func: KernelFunction,
    distances: np.ndarray,
    bandwidth: Bandwidth,
    adaptive: bool,
) -> np.ndarray:
    """Compute one kernel-weight row using fixed or adaptive bandwidth semantics."""
    distance_row = np.asarray(distances, dtype=float)
    if distance_row.ndim != 1:
        raise ValueError("distances must be one-dimensional.")
    if not np.all(np.isfinite(distance_row)) or np.any(distance_row < 0):
        raise ValueError("distances must be finite and non-negative.")

    if adaptive:
        local_bandwidth = adaptive_bandwidth_weights(distance_row, int(bandwidth))
    else:
        local_bandwidth = float(bandwidth)
        if not np.isfinite(local_bandwidth) or local_bandwidth <= 0:
            raise ValueError("Fixed bandwidth must be finite and greater than zero.")

    weights = np.asarray(kernel_func(distance_row, local_bandwidth), dtype=float)
    if weights.shape != distance_row.shape:
        raise ValueError("kernel_func returned weights with an invalid shape.")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("kernel_func returned invalid weights.")
    if not np.any(weights > 0):
        raise ValueError("The selected bandwidth produced no positive weights.")
    return weights


def _local_coefficients(
    X: np.ndarray,
    y: np.ndarray,
    distances: np.ndarray,
    bandwidth: Bandwidth,
    kernel_func: KernelFunction,
    *,
    adaptive: bool,
    ridge: float,
) -> np.ndarray:
    """Estimate local coefficient vectors at every target distance row."""
    X_arr = _validate_matrix(X, "X")
    if X_arr.shape[1] == 0:
        raise ValueError("X must contain at least one local design column.")
    y_arr = _validate_vector(y, "y", n_rows=X_arr.shape[0])
    distance_matrix = _validate_matrix(
        distances,
        "distances",
    )
    if distance_matrix.shape[1] != X_arr.shape[0]:
        raise ValueError(
            "distances must have one column per training observation; "
            f"expected {X_arr.shape[0]}, got {distance_matrix.shape[1]}."
        )
    ridge_value = _validate_ridge(ridge)

    coefficients = np.empty((distance_matrix.shape[0], X_arr.shape[1]), dtype=float)
    for index, distance_row in enumerate(distance_matrix):
        weights = _mixed_weights(kernel_func, distance_row, bandwidth, adaptive)
        coefficients[index], _ = weighted_least_squares(
            X_arr,
            y_arr,
            weights,
            ridge=ridge_value,
        )
    return coefficients


def _local_smoother_matrix(
    X_local: np.ndarray,
    distances: np.ndarray,
    bandwidth: Bandwidth,
    kernel_func: KernelFunction,
    *,
    adaptive: bool,
    ridge: float,
) -> np.ndarray:
    """Construct the local-regression smoother matrix at training locations."""
    X_arr = _validate_matrix(X_local, "X_local")
    distance_matrix = _validate_matrix(distances, "distances", n_rows=X_arr.shape[0])
    if distance_matrix.shape[1] != X_arr.shape[0]:
        raise ValueError("Training distance matrix must be square.")
    ridge_value = _validate_ridge(ridge)

    smoother = np.empty((X_arr.shape[0], X_arr.shape[0]), dtype=float)
    for index, distance_row in enumerate(distance_matrix):
        weights = _mixed_weights(kernel_func, distance_row, bandwidth, adaptive)
        _, inverse_normal = weighted_least_squares(
            X_arr,
            np.zeros(X_arr.shape[0], dtype=float),
            weights,
            ridge=ridge_value,
        )
        smoother[index] = X_arr[index] @ inverse_normal @ (X_arr.T * weights)
    return smoother


def _ridge_least_squares(X: np.ndarray, y: np.ndarray, ridge: float) -> np.ndarray:
    """Solve a global least-squares problem with optional diagonal regularization."""
    X_arr = _validate_matrix(X, "X")
    y_arr = _validate_vector(y, "y", n_rows=X_arr.shape[0])
    ridge_value = _validate_ridge(ridge)
    if X_arr.shape[1] == 0:
        return np.empty(0, dtype=float)
    if ridge_value == 0:
        return np.linalg.lstsq(X_arr, y_arr, rcond=None)[0]
    system = X_arr.T @ X_arr + ridge_value * np.eye(X_arr.shape[1])
    rhs = X_arr.T @ y_arr
    try:
        return np.linalg.solve(system, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(system) @ rhs


def fit_mixed_gwr_core(
    X_local: np.ndarray,
    X_global: np.ndarray,
    y: np.ndarray,
    bandwidth: Bandwidth,
    kernel_func: KernelFunction,
    training_distances: np.ndarray,
    *,
    target_distances: np.ndarray | None = None,
    adaptive: bool = False,
    ridge: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit the semiparametric partial-regression Mixed GWR core.

    Global variables and the response are first residualized against the local
    design using a GWR smoother.  The residualized global relationship is then
    estimated by global least squares.  Finally, local coefficients are fitted
    to the response after removing the global contribution.

    Returns:
        local_coefficients: Coefficients at target locations.
        global_coefficients: One constant coefficient vector.
    """
    X_local_arr = _validate_matrix(X_local, "X_local")
    if X_local_arr.shape[1] == 0:
        raise ValueError("Mixed GWR requires at least one local design column.")
    X_global_arr = _validate_matrix(
        X_global,
        "X_global",
        n_rows=X_local_arr.shape[0],
    )
    y_arr = _validate_vector(y, "y", n_rows=X_local_arr.shape[0])
    train_dist = _validate_matrix(
        training_distances,
        "training_distances",
        n_rows=X_local_arr.shape[0],
    )
    if train_dist.shape[1] != X_local_arr.shape[0]:
        raise ValueError("training_distances must be square.")
    target_dist = (
        train_dist
        if target_distances is None
        else _validate_matrix(
            target_distances,
            "target_distances",
        )
    )
    if target_dist.shape[1] != X_local_arr.shape[0]:
        raise ValueError(
            "target_distances must have one column per training observation."
        )

    smoother = _local_smoother_matrix(
        X_local_arr,
        train_dist,
        bandwidth,
        kernel_func,
        adaptive=adaptive,
        ridge=ridge,
    )
    residualizer = np.eye(X_local_arr.shape[0], dtype=float) - smoother

    if X_global_arr.shape[1] > 0:
        residual_global = residualizer @ X_global_arr
        residual_response = residualizer @ y_arr
        global_coefficients = _ridge_least_squares(
            residual_global,
            residual_response,
            ridge,
        )
        response_for_local = y_arr - X_global_arr @ global_coefficients
    else:
        global_coefficients = np.empty(0, dtype=float)
        response_for_local = y_arr

    local_coefficients = _local_coefficients(
        X_local_arr,
        response_for_local,
        target_dist,
        bandwidth,
        kernel_func,
        adaptive=adaptive,
        ridge=ridge,
    )
    return local_coefficients, global_coefficients


def compute_mixed_gwr_hat_matrix(
    X_local: np.ndarray,
    X_global: np.ndarray,
    bandwidth: Bandwidth,
    kernel_func: KernelFunction,
    distances: np.ndarray,
    *,
    adaptive: bool = False,
    ridge: float = 0.0,
) -> np.ndarray:
    """Return the exact linear smoother matrix for the fitted Mixed GWR model."""
    X_local_arr = _validate_matrix(X_local, "X_local")
    X_global_arr = _validate_matrix(
        X_global,
        "X_global",
        n_rows=X_local_arr.shape[0],
    )
    smoother = _local_smoother_matrix(
        X_local_arr,
        distances,
        bandwidth,
        kernel_func,
        adaptive=adaptive,
        ridge=ridge,
    )
    if X_global_arr.shape[1] == 0:
        return smoother

    identity = np.eye(X_local_arr.shape[0], dtype=float)
    residualizer = identity - smoother
    residual_global = residualizer @ X_global_arr
    ridge_value = _validate_ridge(ridge)
    system = residual_global.T @ residual_global
    if ridge_value > 0:
        system = system + ridge_value * np.eye(system.shape[0])
    try:
        inverse = np.linalg.solve(system, np.eye(system.shape[0]))
    except np.linalg.LinAlgError:
        inverse = np.linalg.pinv(system)
    global_operator = inverse @ residual_global.T @ residualizer
    global_smoother = X_global_arr @ global_operator
    return global_smoother + smoother @ (identity - global_smoother)


def compute_model_criteria(
    y: np.ndarray,
    fitted: np.ndarray,
    trace_s: float,
) -> Dict[str, float]:
    """Compute project-standard Gaussian AIC, AICc, and BIC diagnostics."""
    from pygwrx.core.metrics import compute_aic, compute_aicc, compute_bic

    y_arr = np.asarray(y, dtype=float).reshape(-1)
    fitted_arr = np.asarray(fitted, dtype=float).reshape(-1)
    if y_arr.shape != fitted_arr.shape:
        raise ValueError("y and fitted must have matching shapes.")
    residuals = y_arr - fitted_arr
    rss = float(residuals @ residuals)
    return {
        "aic": compute_aic(y_arr, fitted_arr, n_params=trace_s),
        "aicc": compute_aicc(y_arr, fitted_arr, n_params=trace_s),
        "bic": compute_bic(y_arr, fitted_arr, trace_S=trace_s),
        "rss": rss,
        "sigma2": rss / y_arr.size,
        "enp": float(trace_s),
    }


__all__ = [
    "fit_mixed_gwr_core",
    "compute_mixed_gwr_hat_matrix",
    "compute_model_criteria",
]
