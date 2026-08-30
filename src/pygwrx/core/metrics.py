# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Distribution-neutral regression metrics.

Gaussian smoother diagnostics remain available from this module as compatibility
re-exports; their canonical implementation lives in
:mod:`pygwrx.core.gaussian_diagnostics`.

Author:
    Jinghao Hu
"""

from __future__ import annotations

__author__ = "Jinghao Hu"
__license__ = "MIT"

from typing import Tuple

import numpy as np

from pygwrx.core.gaussian_diagnostics import (
    compute_adjusted_r_squared,
    compute_aic,
    compute_aicc,
    compute_bic,
    compute_diagnostics,
    compute_edf,
    compute_effective_parameters,
    compute_enp,
    compute_local_r_squared,
    compute_trace_statistics,
)

__all__ = [
    "compute_r_squared",
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
    """Validate target arrays and return one-dimensional float arrays."""
    try:
        y_true_arr = np.asarray(y_true, dtype=float)
        y_pred_arr = np.asarray(y_pred, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError("y_true and y_pred must be numeric array-like data.") from exc

    # Accept (n,) and single-column/single-row vectors, but reject genuine matrices.
    if y_true_arr.ndim == 2 and 1 in y_true_arr.shape:
        y_true_arr = y_true_arr.reshape(-1)
    if y_pred_arr.ndim == 2 and 1 in y_pred_arr.shape:
        y_pred_arr = y_pred_arr.reshape(-1)

    if y_true_arr.ndim != 1 or y_pred_arr.ndim != 1:
        raise ValueError("y_true and y_pred must be one-dimensional arrays.")

    if y_true_arr.size == 0:
        raise ValueError("y_true and y_pred cannot be empty.")

    if y_true_arr.shape != y_pred_arr.shape:
        raise ValueError(
            "y_true and y_pred must have the same shape; "
            f"got {y_true_arr.shape} and {y_pred_arr.shape}."
        )

    if not np.all(np.isfinite(y_true_arr)):
        raise ValueError("y_true must contain only finite values.")

    if not np.all(np.isfinite(y_pred_arr)):
        raise ValueError("y_pred must contain only finite values.")

    return y_true_arr, y_pred_arr


def compute_r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute the coefficient of determination, R².

    R² is bounded above by 1 but may be negative. For a constant response,
    this function returns 1 for exact prediction and 0 otherwise, matching
    the finite convention commonly used by machine-learning libraries.
    """
    y_true_arr, y_pred_arr = _validate_targets(y_true, y_pred)

    residuals = y_true_arr - y_pred_arr
    centered = y_true_arr - np.mean(y_true_arr)

    ss_res = float(np.dot(residuals, residuals))
    ss_tot = float(np.dot(centered, centered))

    if np.isclose(ss_tot, 0.0, rtol=0.0, atol=np.finfo(float).eps):
        return (
            1.0
            if np.isclose(
                ss_res,
                0.0,
                rtol=0.0,
                atol=np.finfo(float).eps,
            )
            else 0.0
        )

    return float(1.0 - ss_res / ss_tot)
