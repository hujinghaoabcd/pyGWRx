# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Shared helpers for model-specific plotting modules.

Author:
    Jinghao Hu
"""

from __future__ import annotations

__author__ = "Jinghao Hu"
__license__ = "MIT"

from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np

from pygwrx.diagnostics._utils import require_fitted, training_coords
from pygwrx.plotting._style import default_figure_size


def figure_axis(ax, figsize, theme, *, wide: bool = False):
    """Return an existing axis or create a themed figure and axis."""
    if ax is not None:
        return ax.figure, ax
    return plt.subplots(figsize=figsize or default_figure_size(theme, wide=wide))


def coords_for_model(model: Any) -> np.ndarray:
    """Return validated plotting coordinates for a fitted model."""
    return training_coords(require_fitted(model))


def finite_array(value: Any, name: str, *, ndim: Optional[int] = None) -> np.ndarray:
    """Return a finite float array with an optional dimensionality check."""
    if value is None:
        raise ValueError(f"{name} is not available on the fitted model.")
    array = np.asarray(value, dtype=float)
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-dimensional.")
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite values.")
    return array


def add_reference_line(
    axis, value: Optional[float], *, label: str, vertical: bool = False
):
    """Add a dashed reference line when *value* is finite."""
    if value is None or not np.isfinite(value):
        return
    if vertical:
        axis.axvline(value, color="0.25", linestyle="--", linewidth=1.0, label=label)
    else:
        axis.axhline(value, color="0.25", linestyle="--", linewidth=1.0, label=label)
