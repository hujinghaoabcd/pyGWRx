# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Historical normal-equation WLS implementation retained for reference only.

This module preserves the earlier *unpenalized* weighted normal-equation approach so
that the numerical evolution of PyGWRx remains inspectable. It is intentionally not
imported or called by the production GWR solver.

The current production path in :mod:`pygwrx.core.solver` was upgraded to solve
``sqrt(W) @ X`` with ``numpy.linalg.lstsq``/SVD. That formulation targets the same
weighted least-squares objective for well-conditioned full-rank problems while being
more stable for ill-conditioned or rank-deficient local designs.

No ridge term is preserved here. The former hidden ``1e-8`` ridge variant is
intentionally omitted because standard GWR is now unpenalized by default.
"""

from __future__ import annotations

__author__ = "Jinghao Hu"
__license__ = "MIT"

from typing import Tuple

import numpy as np


def _weighted_least_squares_normal_equations_legacy(
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Solve unpenalized WLS through the historical normal equations.

    Notes:
        This function is retained only for implementation-history comparisons and
        regression diagnostics. Production GWR code must use
        :func:`pygwrx.core.solver.weighted_least_squares` instead.

        The system solved here is exactly ``(X.T @ W @ X) beta = X.T @ W @ y``.
        There is deliberately no ridge regularization.
    """
    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y, dtype=float).reshape(-1)
    weights_arr = np.asarray(weights, dtype=float).reshape(-1)

    if X_arr.ndim != 2:
        raise ValueError("X must be a two-dimensional design matrix.")
    if X_arr.shape[0] != y_arr.shape[0] or X_arr.shape[0] != weights_arr.shape[0]:
        raise ValueError("X, y, and weights must contain the same number of rows.")
    if not np.all(np.isfinite(X_arr)) or not np.all(np.isfinite(y_arr)):
        raise ValueError("X and y must contain only finite values.")
    if not np.all(np.isfinite(weights_arr)) or np.any(weights_arr < 0.0):
        raise ValueError("weights must be finite and non-negative.")
    if not np.any(weights_arr > 0.0):
        raise ValueError("At least one observation weight must be positive.")

    XtW = X_arr.T * weights_arr
    XtWX = XtW @ X_arr
    XtWy = XtW @ y_arr

    try:
        beta = np.linalg.solve(XtWX, XtWy)
        inverse_normal = np.linalg.solve(
            XtWX,
            np.eye(X_arr.shape[1], dtype=float),
        )
    except np.linalg.LinAlgError:
        inverse_normal = np.linalg.pinv(XtWX)
        beta = inverse_normal @ XtWy

    inverse_normal = 0.5 * (inverse_normal + inverse_normal.T)
    return np.asarray(beta, dtype=float), np.asarray(inverse_normal, dtype=float)
