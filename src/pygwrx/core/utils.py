# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Compatibility utilities and model-independent data helpers.

Ordinary distance metrics, distance-cache advice, and bounded distance
streaming are owned by :mod:`pygwrx.core.distance`. Validation is owned by
:mod:`pygwrx.core.validation`. This module keeps legacy imports working while
retaining the small data helpers that have not moved to another owner.

Author:
    Jinghao Hu
"""

from __future__ import annotations

__author__ = "Jinghao Hu"
__license__ = "MIT"

from typing import TYPE_CHECKING

import numpy as np

from pygwrx.core.distance import (  # noqa: F401
    _DEFAULT_DISTANCE_BLOCK_ROWS,
    _iter_distance_blocks,
    _iter_distance_rows,
    DistanceCache,
    chebyshev_distance,
    chunked_computation,
    compute_distance_matrix,
    euclidean_distance,
    haversine_distance,
    manhattan_distance,
    minkowski_distance,
)
from pygwrx.core.validation import (  # noqa: F401
    _extract_geopandas_coords,
    validate_coords,
    validate_data,
)

if TYPE_CHECKING:
    import geopandas as gpd


def add_intercept(X: np.ndarray) -> np.ndarray:
    """Add a leading intercept column of ones to a feature matrix."""
    try:
        X_array = np.asarray(X, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError("X must contain numeric values.") from exc

    if X_array.ndim == 0:
        raise ValueError("X must be one- or two-dimensional.")
    if X_array.ndim == 1:
        X_array = X_array.reshape(-1, 1)
    elif X_array.ndim != 2:
        raise ValueError(
            f"X must be one- or two-dimensional; got {X_array.ndim} dimensions."
        )

    if X_array.shape[0] == 0:
        raise ValueError("X must contain at least one sample.")
    if not np.all(np.isfinite(X_array)):
        raise ValueError("X contains NaN or infinite values.")

    intercept = np.ones((X_array.shape[0], 1), dtype=float)
    return np.hstack((intercept, X_array))


def extract_geopandas_coords(gdf: "gpd.GeoDataFrame") -> np.ndarray:
    """Extract [x, y] coordinates from the active Point geometry column."""
    return _extract_geopandas_coords(gdf)
