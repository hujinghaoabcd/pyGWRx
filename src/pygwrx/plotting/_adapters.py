# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Adapters from fitted pyGWRx estimators to plotting-ready arrays.

Author:
    Jinghao Hu
"""

from __future__ import annotations

__author__ = "Jinghao Hu"
__license__ = "MIT"

from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple, Union

import numpy as np

from pygwrx.diagnostics import feature_names as diagnostic_feature_names
from pygwrx.diagnostics import parameter_inference, parameter_significance
from pygwrx.plotting._validation import (
    as_1d_finite,
    require_fitted_model,
    validate_coords,
)

FeatureLike = Union[str, int]


@dataclass(frozen=True)
class ParameterView:
    """A local parameter surface and its optional inference values."""

    values: np.ndarray
    t_values: Optional[np.ndarray]
    label: str
    parameter_index: int


def feature_names(model: Any) -> Tuple[str, ...]:
    """Return predictor names through the unified diagnostic adapter."""
    return diagnostic_feature_names(model)


def model_coords(model: Any) -> np.ndarray:
    require_fitted_model(model)
    coords = getattr(model, "coords_train_", None)
    if coords is None:
        raise ValueError(f"{model.__class__.__name__} does not expose coords_train_.")
    return validate_coords(coords, len(np.asarray(model.coef_)))


def parameter_view(model: Any, feature: FeatureLike) -> ParameterView:
    """Return a plotting view through unified parameter inference."""
    result = parameter_inference(model, feature)
    return ParameterView(
        values=result.values,
        t_values=result.statistic,
        label=result.label,
        parameter_index=result.parameter_index,
    )


def adjusted_alpha(model: Any, parameter_index: int, alpha: float) -> float:
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError("alpha must lie strictly between 0 and 1.")
    adjusted = getattr(model, "adjusted_alpha_by_variable_", None)
    if adjusted is not None:
        array = np.asarray(adjusted, dtype=float)
        if array.ndim == 2 and parameter_index < array.shape[0]:
            standard = np.asarray([0.10, 0.05, 0.01])
            nearest = int(np.argmin(np.abs(standard - float(alpha))))
            if np.isclose(standard[nearest], float(alpha)):
                return float(array[parameter_index, nearest])

    diagnostics = getattr(model, "diagnostics_", None) or {}
    trace_s = float(diagnostics.get("trace_S", np.nan))
    n_parameters = int(np.asarray(model.coef_).shape[1]) + int(
        bool(getattr(model, "fit_intercept", True))
    )
    if np.isfinite(trace_s) and trace_s > 0.0:
        return min(float(alpha) * n_parameters / trace_s, float(alpha))
    return float(alpha)


def significance_mask(
    model: Any,
    feature: FeatureLike,
    *,
    alpha: float = 0.05,
    correction: str = "adjusted",
) -> Tuple[np.ndarray, ParameterView, float]:
    """Return a significance mask through the unified inference layer."""
    frame = parameter_significance(
        model,
        feature,
        alpha=alpha,
        correction=correction,
    )
    view = parameter_view(model, feature)
    threshold = float(frame["threshold"].iloc[0])
    return frame["significant"].to_numpy(bool), view, threshold


def diagnostic_values(model: Any, metric: str) -> Tuple[np.ndarray, str, bool]:
    require_fitted_model(model)
    key = str(metric).strip().lower().replace("-", "_").replace(" ", "_")
    candidates = {
        "local_r2": ("local_r2_", "Local R²", False),
        "residual": ("residuals_", "Residual", True),
        "residuals": ("residuals_", "Residual", True),
        "standardized_residual": (
            "standardized_residuals_",
            "Standardized residual",
            True,
        ),
        "standardized_residuals": (
            "standardized_residuals_",
            "Standardized residual",
            True,
        ),
        "influence": ("influence_", "Influence (leverage)", False),
        "leverage": ("influence_", "Influence (leverage)", False),
        "cooks_distance": ("cooks_distance_", "Cook's distance", False),
        "cooks_d": ("cooks_distance_", "Cook's distance", False),
        "cv_contribution": ("cv_contributions_", "CV squared error", False),
    }
    if key not in candidates:
        raise ValueError(
            f"Unknown diagnostic {metric!r}. Choose from {sorted(candidates)}."
        )
    attribute, label, center_zero = candidates[key]
    values = getattr(model, attribute, None)
    if values is None:
        raise ValueError(
            f"{model.__class__.__name__} does not expose {attribute}; refit with the required diagnostics enabled."
        )
    return as_1d_finite(values, attribute, allow_nan=True), label, center_zero


def _compute_gwr_condition_numbers(model: Any) -> Optional[np.ndarray]:
    """Compute GWR-scale local design condition numbers on demand."""
    if getattr(model, "bandwidths_", None) is not None:
        return None
    X = getattr(model, "X_train_", None)
    coords = getattr(model, "coords_train_", None)
    if X is None or coords is None or not hasattr(model, "_weights_from_distances"):
        return None
    from pygwrx.core.utils import add_intercept, compute_distance_matrix

    X_array = np.asarray(X, dtype=float)
    design = (
        add_intercept(X_array)
        if bool(getattr(model, "fit_intercept", True))
        else X_array
    )
    distance_matrix = compute_distance_matrix(
        np.asarray(coords, dtype=float),
        np.asarray(coords, dtype=float),
        metric=getattr(model, "distance_metric", "euclidean"),
    )
    output = np.full(design.shape[0], np.nan, dtype=float)
    for index, distances in enumerate(distance_matrix):
        weights = np.asarray(model._weights_from_distances(distances), dtype=float)
        weighted = design * np.sqrt(np.maximum(weights, 0.0))[:, None]
        norms = np.sqrt(np.sum(weighted**2, axis=0))
        valid = norms > np.finfo(float).eps
        if np.sum(valid) < 2:
            continue
        standardized = weighted[:, valid] / norms[valid]
        singular = np.linalg.svd(standardized, compute_uv=False)
        if singular.size and singular[-1] > np.finfo(float).eps:
            output[index] = singular[0] / singular[-1]
    return output


def collinearity_values(
    model: Any, metric: str
) -> Tuple[np.ndarray, str, Optional[float]]:
    require_fitted_model(model)
    key = str(metric).strip().lower().replace("-", "_").replace(" ", "_")
    candidates = {
        "condition_number": (
            "condition_numbers_",
            "Local condition number",
            "cn_thresh",
        ),
        "local_condition_number": (
            "local_condition_numbers_",
            "Local condition number",
            "cn_thresh",
        ),
        "compensated_condition_number": (
            "compensated_condition_numbers_",
            "Compensated condition number",
            "cn_thresh",
        ),
        "penalized_system_condition_number": (
            "penalized_system_condition_numbers_",
            "Penalized-system condition number",
            "cn_thresh",
        ),
        "local_lambda": ("local_lambda_", "Local ridge parameter", None),
    }
    if key not in candidates:
        raise ValueError(f"Unknown collinearity metric {metric!r}.")
    attribute, label, threshold_attribute = candidates[key]
    values = getattr(model, attribute, None)
    if values is None and key in {"condition_number", "local_condition_number"}:
        values = _compute_gwr_condition_numbers(model)
        label = "Local condition number (computed)"
    if values is None:
        raise ValueError(f"{model.__class__.__name__} does not expose {attribute}.")
    threshold = (
        None
        if threshold_attribute is None
        else float(getattr(model, threshold_attribute, np.nan))
    )
    if threshold is not None and not np.isfinite(threshold):
        threshold = None
    return as_1d_finite(values, attribute, allow_nan=True), label, threshold


def parameter_names(model: Any, *, include_intercept: bool = True) -> Sequence[str]:
    names = list(feature_names(model))
    if include_intercept and bool(getattr(model, "fit_intercept", True)):
        return ["intercept", *names]
    return names
