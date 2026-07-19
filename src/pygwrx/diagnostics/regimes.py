# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Diagnostics for fitted spatial-regime models.

Author:
    Jinghao Hu
"""

from __future__ import annotations

__author__ = "Jinghao Hu"
__license__ = "MIT"

from typing import Any

import numpy as np
import pandas as pd

from ._utils import require_fitted, training_coords


def regime_frame(model: Any) -> pd.DataFrame:
    """Return coordinates, regime labels, residuals, and connectivity metadata."""
    require_fitted(model)
    regimes = getattr(model, "regimes_", None)
    if regimes is None:
        raise ValueError(f"{model.__class__.__name__} does not expose regimes_.")
    labels = np.asarray(regimes, dtype=int).reshape(-1)
    coords = training_coords(model)
    if labels.size != coords.shape[0]:
        raise ValueError("regimes_ length does not match coordinates.")
    frame = pd.DataFrame(
        {"coord_0": coords[:, 0], "coord_1": coords[:, 1], "regime": labels}
    )
    residuals = getattr(model, "residuals_", None)
    if residuals is not None:
        frame["residual"] = np.asarray(residuals, dtype=float).reshape(-1)
    return frame


def regime_summary(model: Any) -> pd.DataFrame:
    """Summarize regime sizes, residual error, and component counts."""
    frame = regime_frame(model)
    grouped = frame.groupby("regime", sort=True)
    summary = grouped.size().rename("n_samples").to_frame()
    if "residual" in frame:
        summary["rmse"] = grouped["residual"].apply(
            lambda values: float(np.sqrt(np.mean(np.asarray(values, dtype=float) ** 2)))
        )
        summary["mae"] = grouped["residual"].apply(
            lambda values: float(np.mean(np.abs(np.asarray(values, dtype=float))))
        )
    counts = getattr(model, "regime_component_counts_", None)
    if counts is not None:
        array = np.asarray(counts, dtype=int).reshape(-1)
        if array.size == summary.shape[0]:
            summary["connected_components"] = array
    return summary


def boundary_frame(model: Any) -> pd.DataFrame:
    """Return unique regime-boundary edges and their endpoints."""
    require_fitted(model)
    coords = training_coords(model)
    boundaries = getattr(model, "regime_boundaries_", None)
    if boundaries is None:
        raise ValueError(
            f"{model.__class__.__name__} does not expose regime_boundaries_."
        )
    records = []
    for left, right in boundaries:
        i, j = int(left), int(right)
        records.append(
            {
                "left": i,
                "right": j,
                "x0": coords[i, 0],
                "y0": coords[i, 1],
                "x1": coords[j, 0],
                "y1": coords[j, 1],
            }
        )
    return pd.DataFrame(records)
