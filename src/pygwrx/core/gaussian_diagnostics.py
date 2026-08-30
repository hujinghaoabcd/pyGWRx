# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Diagnostics for Gaussian geographically weighted smoothers.

This module owns Gaussian information criteria, local coefficients of
determination, and smoother/hat-matrix trace statistics. Distribution-neutral
metrics remain in :mod:`pygwrx.core.metrics`.

Author:
    Jinghao Hu
"""

from __future__ import annotations

__author__ = "Jinghao Hu"
__license__ = "MIT"

from typing import Dict, Optional, Tuple

import numpy as np

__all__ = [
    "compute_adjusted_r_squared",
    "compute_aic",
    "compute_aicc",
    "compute_bic",
    "compute_local_r_squared",
    "compute_effective_parameters",
    "compute_diagnostics",
    "compute_trace_statistics",
    "compute_edf",
    "compute_enp",
]


def _validate_targets(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Use the distribution-neutral target validator without changing its contract."""
    from pygwrx.core.metrics import _validate_targets as validate_targets

    return validate_targets(y_true, y_pred)


def _compute_r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Use the distribution-neutral R² implementation."""
    from pygwrx.core.metrics import compute_r_squared

    return compute_r_squared(y_true, y_pred)


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


def _validate_hat_matrix(
    hat_matrix: np.ndarray,
    n_samples: Optional[int] = None,
) -> np.ndarray:
    """Validate a square, finite hat matrix."""
    try:
        matrix = np.asarray(hat_matrix, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError("hat_matrix must be numeric array-like data.") from exc

    if matrix.ndim != 2:
        raise ValueError("hat_matrix must be a two-dimensional array.")

    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("hat_matrix must be square; " f"got shape {matrix.shape}.")

    if n_samples is not None and matrix.shape != (n_samples, n_samples):
        raise ValueError(
            "hat_matrix shape must match the number of observations; "
            f"expected {(n_samples, n_samples)}, got {matrix.shape}."
        )

    if not np.all(np.isfinite(matrix)):
        raise ValueError("hat_matrix must contain only finite values.")

    return matrix


def _residual_sum_of_squares(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """Return RSS after target validation."""
    y_true_arr, y_pred_arr = _validate_targets(y_true, y_pred)
    residuals = y_true_arr - y_pred_arr
    return float(np.dot(residuals, residuals))


def _safe_rss_for_log(rss: float) -> float:
    """Return a strictly positive RSS for logarithmic criteria."""
    # Only replace exact numerical zero. A fixed epsilon such as 1e-10 is
    # scale-dependent and can distort valid low-magnitude data.
    return max(float(rss), np.finfo(float).tiny)


def compute_adjusted_r_squared(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    edf: float,
) -> float:
    """Compute GWR adjusted R² from residual effective degrees of freedom.

    Formula
    -------
    Adj R² = 1 - (1 - R²) * (n - 1) / (EDF - 1)

    Here EDF is normally:
        n - 2 * trace(S) + trace(S.T @ S)
    """
    y_true_arr, y_pred_arr = _validate_targets(y_true, y_pred)
    edf_value = _validate_nonnegative_scalar(edf, "edf")

    if edf_value <= 1.0:
        return np.nan

    n = y_true_arr.size
    r2 = _compute_r_squared(y_true_arr, y_pred_arr)

    return float(1.0 - (1.0 - r2) * (n - 1.0) / (edf_value - 1.0))


def compute_aic(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_params: float,
) -> float:
    """Compute Gaussian GWR AIC using trace(S) as the complexity term.

    Formula
    -------
    AIC = n*log(RSS/n) + n*log(2π) + n + 2*(trace(S) + 1)

    Notes:
        This is a Gaussian RSS-based criterion. It must not be used for Poisson,
        Binomial, Gamma, or other non-Gaussian GWGLM families.
    """
    y_true_arr, y_pred_arr = _validate_targets(y_true, y_pred)
    n_params_value = _validate_nonnegative_scalar(n_params, "n_params")

    n = y_true_arr.size
    rss = _safe_rss_for_log(_residual_sum_of_squares(y_true_arr, y_pred_arr))

    return float(
        n * np.log(rss / n) + n * np.log(2.0 * np.pi) + n + 2.0 * (n_params_value + 1.0)
    )


def compute_aicc(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_params: float,
) -> float:
    """Compute Gaussian GWR corrected AIC (AICc). Compute the corrected Akaike information
    criterion for Gaussian GWR.

    Formula
    -------
    AICc = n*log(RSS/n) + n*log(2π)
           + n*(n + trace(S)) / (n - 2 - trace(S))

    Returns infinity when the correction denominator is not positive.
    """
    y_true_arr, y_pred_arr = _validate_targets(y_true, y_pred)
    n_params_value = _validate_nonnegative_scalar(n_params, "n_params")

    n = y_true_arr.size
    rss = _safe_rss_for_log(_residual_sum_of_squares(y_true_arr, y_pred_arr))
    denominator = n - 2.0 - n_params_value

    if denominator <= 0.0:
        return np.inf

    return float(
        n * np.log(rss / n)
        + n * np.log(2.0 * np.pi)
        + n * (n + n_params_value) / denominator
    )


def compute_bic(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    trace_S: float,
) -> float:
    """Compute Gaussian GWR BIC using trace(S).

    Formula
    -------
    BIC = n*log(RSS/n) + n*log(2π) + n
          + log(n)*(trace(S) + 1)

    Notes:
        This is a Gaussian RSS-based criterion and is not suitable for
        non-Gaussian GWGLM families.
    """
    y_true_arr, y_pred_arr = _validate_targets(y_true, y_pred)
    trace_s_value = _validate_nonnegative_scalar(trace_S, "trace_S")

    n = y_true_arr.size
    rss = _safe_rss_for_log(_residual_sum_of_squares(y_true_arr, y_pred_arr))

    return float(
        n * np.log(rss / n)
        + n * np.log(2.0 * np.pi)
        + n
        + np.log(n) * (trace_s_value + 1.0)
    )


def compute_local_r_squared(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Compute local weighted R² values. Compute a locally weighted coefficient of
    determination.

    Local R² is bounded above by 1 but may be negative when local predictions
    are worse than the local weighted-mean baseline.
    """
    y_true_arr, y_pred_arr = _validate_targets(y_true, y_pred)
    n_samples = y_true_arr.size

    try:
        weights_arr = np.asarray(weights, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError("weights must be numeric array-like data.") from exc

    expected_shape = (n_samples, n_samples)
    if weights_arr.shape != expected_shape:
        raise ValueError(
            f"weights must have shape {expected_shape}; got {weights_arr.shape}."
        )

    if not np.all(np.isfinite(weights_arr)):
        raise ValueError("weights must contain only finite values.")

    if np.any(weights_arr < 0.0):
        raise ValueError("weights must be non-negative.")

    residuals_sq = (y_true_arr - y_pred_arr) ** 2
    local_r2 = np.full(n_samples, np.nan, dtype=float)

    for i in range(n_samples):
        w = weights_arr[i]
        sum_w = float(np.sum(w))

        if sum_w <= np.finfo(float).eps:
            # No local information is available at this location.
            continue

        y_mean_weighted = float(np.dot(w, y_true_arr) / sum_w)
        tss_weighted = float(np.dot(w, (y_true_arr - y_mean_weighted) ** 2))
        rss_weighted = float(np.dot(w, residuals_sq))

        if np.isclose(
            tss_weighted,
            0.0,
            rtol=0.0,
            atol=np.finfo(float).eps,
        ):
            local_r2[i] = (
                1.0
                if np.isclose(
                    rss_weighted,
                    0.0,
                    rtol=0.0,
                    atol=np.finfo(float).eps,
                )
                else 0.0
            )
        else:
            local_r2[i] = 1.0 - rss_weighted / tss_weighted

    return local_r2


def compute_effective_parameters(hat_matrix: np.ndarray) -> float:
    """Return trace(S), the first common effective-parameter convention.

    Notes:
        A second convention is 2*trace(S) - trace(S.T @ S), returned by
        :func:`compute_enp`. Keeping these definitions explicit avoids mixing
        incompatible EDF/ENP conventions.
    """
    matrix = _validate_hat_matrix(hat_matrix)
    return float(np.trace(matrix))


def compute_diagnostics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    hat_matrix: Optional[np.ndarray] = None,
    n_features: Optional[int] = None,
    compute_gwr_stats: bool = False,
    *,
    trace_S: Optional[float] = None,
    trace_StS: Optional[float] = None,
) -> Dict[str, float]:
    """Compute diagnostic statistics for a Gaussian GWR-style model.

    Args:
        y_true: Observed values.
        y_pred: Fitted values.
        hat_matrix: GWR hat matrix. When supplied, trace(S), trace(S'S), EDF, and ENP are
            derived from this matrix.
        n_features: Backward-compatible parameter-count fallback used when no hat matrix
            is available. In the current PyGWRx model implementations this value
            is the number of columns in the fitted design matrix, so an intercept
            already present in that matrix must NOT be added again.
        compute_gwr_stats: Include trace_S, trace_StS, edf, enp_v1, enp_v2, and compatibility key
            enp in the returned dictionary.
        trace_S: Precomputed trace of the smoother matrix. Supply together with
            ``trace_StS`` when the full hat matrix is intentionally not stored.
        trace_StS: Precomputed trace of ``S.T @ S``. Supply together with
            ``trace_S`` to retain GWR complexity diagnostics without an ``n x n`` matrix.

    Notes:
        - Information criteria here are Gaussian RSS-based criteria.
        - Models without a reliable hat matrix receive only a parameter-count
          approximation of complexity.
        - For Poisson/Binomial/Gamma GWGLM, use family-specific log-likelihood and
          deviance diagnostics instead of this function.
    """
    y_true_arr, y_pred_arr = _validate_targets(y_true, y_pred)
    n = y_true_arr.size

    residuals = y_true_arr - y_pred_arr
    rss = float(np.dot(residuals, residuals))

    diagnostics: Dict[str, float] = {
        "r2": _compute_r_squared(y_true_arr, y_pred_arr),
        "rss": rss,
        "rmse": float(np.sqrt(np.mean(residuals**2))),
        "mae": float(np.mean(np.abs(residuals))),
    }

    if (trace_S is None) != (trace_StS is None):
        raise ValueError("trace_S and trace_StS must be supplied together.")
    if hat_matrix is not None and trace_S is not None:
        raise ValueError("Supply either hat_matrix or trace_S/trace_StS, not both.")

    if hat_matrix is not None:
        matrix = _validate_hat_matrix(hat_matrix, n_samples=n)
        trace_stats = compute_trace_statistics(matrix)
        trace_s = trace_stats["trace_S"]
        trace_sts = trace_stats["trace_StS"]

    elif trace_S is not None:
        trace_s = _validate_nonnegative_scalar(trace_S, "trace_S")
        trace_sts = _validate_nonnegative_scalar(trace_StS, "trace_StS")

    else:
        trace_s = None
        trace_sts = None

    if trace_s is not None and trace_sts is not None:

        edf_v2 = compute_edf(n, trace_s, trace_sts)
        enp_v1 = trace_s
        enp_v2 = compute_enp(trace_s, trace_sts)

        diagnostics.update(
            {
                # Backward compatibility: historically this key meant trace(S).
                "effective_params": enp_v1,
                "adj_r2": compute_adjusted_r_squared(
                    y_true_arr,
                    y_pred_arr,
                    edf=edf_v2,
                ),
                "aic": compute_aic(y_true_arr, y_pred_arr, n_params=trace_s),
                "aicc": compute_aicc(y_true_arr, y_pred_arr, n_params=trace_s),
                "bic": compute_bic(y_true_arr, y_pred_arr, trace_S=trace_s),
            }
        )

        if compute_gwr_stats:
            diagnostics.update(
                {
                    "trace_S": trace_s,
                    "trace_StS": trace_sts,
                    # Explicitly expose both conventions.
                    "enp_v1": enp_v1,
                    "edf_v1": float(n - enp_v1),
                    "enp_v2": enp_v2,
                    "edf_v2": edf_v2,
                    # Compatibility keys preserve the original GWmodel-style
                    # v2 convention used by existing PyGWRx code.
                    "enp": enp_v2,
                    "edf": edf_v2,
                }
            )

    elif n_features is not None:
        # IMPORTANT: Existing PyGWRx callers pass the fitted design-matrix
        # column count here; several already include an intercept. Therefore
        # do not add another +1 in this diagnostics layer.
        n_params = _validate_nonnegative_scalar(n_features, "n_features")
        residual_df = n - n_params

        diagnostics["effective_params"] = n_params
        diagnostics["adj_r2"] = (
            float(1.0 - (1.0 - diagnostics["r2"]) * (n - 1.0) / (residual_df - 1.0))
            if residual_df > 1.0
            else np.nan
        )
        diagnostics["aic"] = compute_aic(
            y_true_arr,
            y_pred_arr,
            n_params=n_params,
        )
        diagnostics["aicc"] = compute_aicc(
            y_true_arr,
            y_pred_arr,
            n_params=n_params,
        )
        diagnostics["bic"] = compute_bic(
            y_true_arr,
            y_pred_arr,
            trace_S=n_params,
        )

    else:
        # Complexity is unknown; do not invent a one-parameter model.
        diagnostics.update(
            {
                "effective_params": np.nan,
                "adj_r2": np.nan,
                "aic": np.nan,
                "aicc": np.nan,
                "bic": np.nan,
            }
        )

    return diagnostics


def compute_trace_statistics(hat_matrix: np.ndarray) -> Dict[str, float]:
    """Compute trace(S) and trace(S'S) from a validated hat matrix."""
    matrix = _validate_hat_matrix(hat_matrix)

    trace_s = float(np.trace(matrix))
    trace_sts = float(np.sum(matrix * matrix))

    return {
        "trace_S": trace_s,
        "trace_StS": trace_sts,
    }


def compute_edf(n: int, trace_S: float, trace_StS: float) -> float:
    """Compute residual effective degrees of freedom using the GWmodel convention.

    EDF_v2 = n - 2*trace(S) + trace(S'S)
    """
    if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)):
        raise TypeError("n must be a positive integer.")
    if n <= 0:
        raise ValueError("n must be greater than zero.")

    trace_s_value = _validate_nonnegative_scalar(trace_S, "trace_S")
    trace_sts_value = _validate_nonnegative_scalar(trace_StS, "trace_StS")
    edf = float(n - 2.0 * trace_s_value + trace_sts_value)

    # A saturated smoother can have theoretical EDF == 0 while independent
    # floating-point trace calculations leave a tiny negative residue. Clamp only
    # machine-roundoff-scale negative zero; materially negative EDF values remain
    # negative so downstream validation still rejects an invalid diagnostic state.
    roundoff_tolerance = (
        16.0
        * np.finfo(float).eps
        * max(
            1.0,
            float(n),
            2.0 * abs(trace_s_value),
            abs(trace_sts_value),
        )
    )
    if -roundoff_tolerance <= edf < 0.0:
        return 0.0
    return edf


def compute_enp(trace_S: float, trace_StS: float) -> float:
    """Compute the GWmodel-style effective number of parameters. Compute the effective
    parameter count using the GWmodel convention.

    ENP_v2 = 2*trace(S) - trace(S'S)
    """
    trace_s_value = _validate_nonnegative_scalar(trace_S, "trace_S")
    trace_sts_value = _validate_nonnegative_scalar(trace_StS, "trace_StS")

    return float(2.0 * trace_s_value - trace_sts_value)
