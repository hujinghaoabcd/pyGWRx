# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Shared validation helpers for diagnostic modules.

Author:
    Jinghao Hu
"""

from __future__ import annotations

__author__ = "Jinghao Hu"
__license__ = "MIT"

from typing import Any, Optional

import numpy as np


def require_fitted(model: Any) -> Any:
    """Return *model* after verifying that it exposes fitted results."""
    fitted_flag = getattr(model, "_is_fitted", None)
    has_results = any(
        getattr(model, name, None) is not None
        for name in ("diagnostics_", "coef_", "loadings_", "local_mean_", "classes_")
    )
    if fitted_flag is False or (fitted_flag is None and not has_results):
        raise ValueError(
            f"{model.__class__.__name__} is not fitted. Fit the model before running diagnostics."
        )
    return model


def numeric_vector(value: Any, name: str, *, allow_nan: bool = False) -> np.ndarray:
    """Convert a vector-like value to a one-dimensional float array."""
    if value is None:
        raise ValueError(f"{name} is not available on the fitted model.")
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be numeric array-like data.") from exc
    if array.ndim == 2 and 1 in array.shape:
        array = array.reshape(-1)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array.")
    valid = np.isfinite(array) | (allow_nan & np.isnan(array))
    if not np.all(valid):
        raise ValueError(f"{name} contains unsupported non-finite values.")
    return array


def numeric_matrix(value: Any, name: str, *, allow_nan: bool = False) -> np.ndarray:
    """Convert a matrix-like value to a two-dimensional float array."""
    if value is None:
        raise ValueError(f"{name} is not available on the fitted model.")
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be numeric array-like data.") from exc
    if array.ndim != 2 or 0 in array.shape:
        raise ValueError(f"{name} must be a non-empty two-dimensional array.")
    valid = np.isfinite(array) | (allow_nan & np.isnan(array))
    if not np.all(valid):
        raise ValueError(f"{name} contains unsupported non-finite values.")
    return array


def first_available(model: Any, *names: str) -> Optional[Any]:
    """Return the first non-``None`` fitted attribute among *names*."""
    for name in names:
        value = getattr(model, name, None)
        if value is not None:
            return value
    return None


def training_coords(model: Any) -> np.ndarray:
    """Return calibration/evaluation coordinates using model-specific aliases."""
    require_fitted(model)
    value = first_available(
        model,
        "coords_train_",
        "coords_",
        "coords_summary_",
        "eval_coords_",
        "coords_data_",
    )
    if value is None and getattr(model, "coords_stages_", None):
        value = model.coords_stages_[-1]
    matrix = numeric_matrix(value, "training coordinates")
    if matrix.shape[1] < 2:
        raise ValueError("Training coordinates must contain at least two columns.")
    return matrix[:, :2]


def training_response(model: Any) -> Optional[np.ndarray]:
    """Return the fitted response vector when the model is supervised."""
    value = first_available(model, "y_train_", "y_", "y_train")
    if value is None and getattr(model, "y_stages_", None):
        value = model.y_stages_[-1]
    return None if value is None else numeric_vector(value, "training response")


def fitted_values(model: Any) -> Optional[np.ndarray]:
    """Return fitted values when available."""
    value = first_available(model, "fitted_values_", "mu_")
    return None if value is None else numeric_vector(value, "fitted values")
