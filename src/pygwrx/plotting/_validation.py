# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Input validation utilities for plotting helpers.

Author:
    Jinghao Hu
"""

from __future__ import annotations

__author__ = "Jinghao Hu"
__license__ = "MIT"

from typing import Any, Optional

import numpy as np


def as_1d_finite(values: Any, name: str, *, allow_nan: bool = False) -> np.ndarray:
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.size == 0:
        raise ValueError(f"{name} cannot be empty.")
    if allow_nan:
        if not np.any(np.isfinite(array)):
            raise ValueError(f"{name} contains no finite values.")
    elif not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or infinite values.")
    return array


def validate_coords(coords: Any, n_samples: Optional[int] = None) -> np.ndarray:
    array = np.asarray(coords, dtype=float)
    if array.ndim != 2 or array.shape[1] < 2:
        raise ValueError(
            "coords must be a two-dimensional array with at least two columns."
        )
    if not np.all(np.isfinite(array[:, :2])):
        raise ValueError("coords contains NaN or infinite values.")
    if n_samples is not None and array.shape[0] != int(n_samples):
        raise ValueError(
            f"coords contains {array.shape[0]} rows, but {n_samples} values are required."
        )
    return array[:, :2]


def validate_geometry(geometry: Any, n_samples: int) -> Any:
    if geometry is None:
        return None
    try:
        import geopandas as gpd
    except (
        ImportError
    ) as error:  # pragma: no cover - project currently depends on geopandas
        raise ImportError("geopandas is required when geometry is supplied.") from error

    if isinstance(geometry, gpd.GeoDataFrame):
        geo = geometry.geometry
    elif isinstance(geometry, gpd.GeoSeries):
        geo = geometry
    else:
        raise TypeError("geometry must be a GeoDataFrame, GeoSeries, or None.")
    if len(geo) != int(n_samples):
        raise ValueError(
            f"geometry contains {len(geo)} features, but {n_samples} values are required."
        )
    if geo.isna().any() or geo.is_empty.any():
        raise ValueError("geometry contains missing or empty features.")
    return geo.reset_index(drop=True)


def require_fitted_model(model: Any) -> None:
    if model is None:
        raise TypeError("model cannot be None.")
    if hasattr(model, "is_fitted_") and not bool(model.is_fitted_):
        raise ValueError(f"{model.__class__.__name__} is not fitted.")
    if getattr(model, "coef_", None) is None:
        raise ValueError(
            f"{model.__class__.__name__} does not expose fitted local coefficients."
        )
