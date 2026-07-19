# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Local residual, influence, and outlier diagnostics.

Author:
    Jinghao Hu
"""

from __future__ import annotations

__author__ = "Jinghao Hu"
__license__ = "MIT"

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from ._utils import (
    first_available,
    fitted_values,
    require_fitted,
    training_coords,
    training_response,
)


@dataclass(frozen=True)
class InfluenceThresholds:
    """Common reference thresholds for local influence diagnostics."""

    leverage: Optional[float]
    cooks_distance: Optional[float]
    standardized_residual: float = 2.0


def influence_thresholds(model: Any) -> InfluenceThresholds:
    """Return transparent rule-of-thumb thresholds for a fitted model."""
    require_fitted(model)
    y = training_response(model)
    n = None if y is None else int(y.size)
    p = getattr(model, "n_features_in_", None)
    if p is not None:
        p = int(p) + int(bool(getattr(model, "fit_intercept", True)))
    leverage = None if not n or p is None else min(1.0, 2.0 * p / n)
    cooks = None if not n else 4.0 / n
    return InfluenceThresholds(leverage=leverage, cooks_distance=cooks)


def local_diagnostic_frame(model: Any) -> pd.DataFrame:
    """Collect available row-wise diagnostics without mutating the model."""
    require_fitted(model)
    coords = training_coords(model)
    n = coords.shape[0]
    frame = pd.DataFrame({"coord_0": coords[:, 0], "coord_1": coords[:, 1]})

    y = training_response(model)
    fitted = fitted_values(model)
    if y is not None and y.size == n:
        frame["observed"] = y
    if fitted is not None and fitted.size == n:
        frame["fitted"] = fitted
    residual = first_available(model, "residuals_", "deviance_residuals_")
    if residual is None and y is not None and fitted is not None:
        residual = y - fitted
    if residual is not None:
        residual = np.asarray(residual, dtype=float).reshape(-1)
        if residual.size == n:
            frame["residual"] = residual

    aliases: Dict[str, tuple[str, ...]] = {
        "standardized_residual": ("standardized_residuals_", "pearson_residuals_"),
        "deviance_residual": ("deviance_residuals_",),
        "pearson_residual": ("pearson_residuals_",),
        "influence": ("influence_",),
        "cooks_distance": ("cooks_distance_",),
        "local_r2": ("local_r2_",),
        "cv_contribution": ("cv_contributions_",),
        "robust_weight": ("robust_weights_",),
        "robust_score": ("robust_residual_scores_",),
        "local_alpha": ("alpha_",),
        "local_lambda": ("local_lambda_",),
        "condition_number": ("condition_numbers_", "local_condition_numbers_"),
        "regime": ("regimes_",),
    }
    for output, names in aliases.items():
        value = first_available(model, *names)
        if value is None:
            continue
        array = np.asarray(value)
        if array.ndim == 0:
            continue
        array = array.reshape(-1)
        if array.size == n:
            frame[output] = array

    thresholds = influence_thresholds(model)
    if "standardized_residual" in frame:
        frame["large_residual"] = (
            np.abs(frame["standardized_residual"]) > thresholds.standardized_residual
        )
    if "influence" in frame and thresholds.leverage is not None:
        frame["high_leverage"] = frame["influence"] > thresholds.leverage
    if "cooks_distance" in frame and thresholds.cooks_distance is not None:
        frame["influential"] = frame["cooks_distance"] > thresholds.cooks_distance
    return frame
