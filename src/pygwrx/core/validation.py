# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Model-independent array, feature, and coordinate validation helpers.

This module owns the canonical validation routines used by the core numerical
infrastructure. Legacy imports from :mod:`pygwrx.core.utils` remain available as
compatibility re-exports during the 0.1.x to 0.2.0 transition.

Author:
    Jinghao Hu
"""

from __future__ import annotations

__author__ = "Jinghao Hu"
__license__ = "MIT"

from numbers import Integral
from typing import TYPE_CHECKING, Tuple, Union

import numpy as np
import pandas as pd

from pygwrx._optional import import_required_dependency

if TYPE_CHECKING:
    import geopandas as gpd


CoordinateInput = Union[np.ndarray, pd.DataFrame]

__all__ = ["validate_coords", "validate_data"]


def _as_float_coordinate_array(
    coords: CoordinateInput,
    *,
    name: str,
    require_two_columns: bool = False,
) -> np.ndarray:
    """Convert coordinate-like input to a validated floating-point 2D array."""
    if isinstance(coords, pd.DataFrame):
        values = coords.to_numpy()
    else:
        values = coords

    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must contain numeric coordinate values.") from exc

    if array.size == 0:
        raise ValueError(f"{name} must contain at least one coordinate.")

    if array.ndim == 1:
        array = array.reshape(1, -1)
    elif array.ndim != 2:
        raise ValueError(
            f"{name} must be a one- or two-dimensional coordinate array; "
            f"got {array.ndim} dimensions."
        )

    if array.shape[1] == 0:
        raise ValueError(f"{name} must contain at least one coordinate dimension.")

    if require_two_columns and array.shape[1] != 2:
        raise ValueError(
            f"{name} must have exactly 2 columns (x, y); got {array.shape[1]}."
        )

    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or infinite values.")

    return array


def _validate_coordinate_pair(
    coords1: CoordinateInput,
    coords2: CoordinateInput,
    *,
    require_two_columns: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Validate two coordinate arrays and ensure compatible dimensions."""
    coords1_arr = _as_float_coordinate_array(
        coords1,
        name="coords1",
        require_two_columns=require_two_columns,
    )
    coords2_arr = _as_float_coordinate_array(
        coords2,
        name="coords2",
        require_two_columns=require_two_columns,
    )

    if coords1_arr.shape[1] != coords2_arr.shape[1]:
        raise ValueError(
            "coords1 and coords2 must have the same coordinate dimension; "
            f"got {coords1_arr.shape[1]} and {coords2_arr.shape[1]}."
        )

    return coords1_arr, coords2_arr


def _validate_positive_scalar(value: float, *, name: str) -> float:
    """Validate a positive finite scalar."""
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a positive real scalar, not bool.")

    value_array = np.asarray(value)
    if value_array.ndim != 0:
        raise TypeError(f"{name} must be a scalar value.")

    try:
        value_float = float(value_array)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a positive real scalar.") from exc

    if not np.isfinite(value_float):
        raise ValueError(f"{name} must be finite.")
    if value_float <= 0:
        raise ValueError(f"{name} must be greater than zero.")

    return value_float


def _validate_count(value: int, *, name: str) -> int:
    """Validate a non-negative integer count."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a non-negative integer.")

    value_int = int(value)
    if value_int < 0:
        raise ValueError(f"{name} must be non-negative.")

    return value_int


def _extract_geopandas_coords(gdf: "gpd.GeoDataFrame") -> np.ndarray:
    """Validate a Point GeoDataFrame and extract its active geometry coordinates."""
    geopandas = import_required_dependency(
        "geopandas", purpose="GeoDataFrame coordinate extraction"
    )
    if not isinstance(gdf, geopandas.GeoDataFrame):
        raise TypeError("gdf must be a GeoDataFrame.")

    try:
        geometry = gdf.geometry
    except (AttributeError, ValueError) as exc:
        raise ValueError("GeoDataFrame must have an active geometry column.") from exc

    if geometry.isna().any():
        raise ValueError("GeoDataFrame geometry contains missing values.")
    if geometry.is_empty.any():
        raise ValueError("GeoDataFrame geometry contains empty geometries.")

    geometry_types = geometry.geom_type
    if not geometry_types.eq("Point").all():
        found = sorted(set(geometry_types.astype(str)))
        raise ValueError(
            "GeoDataFrame geometries must all be Point type; "
            f"found: {found}. Use centroids explicitly for non-point geometries."
        )

    coords = np.column_stack(
        (
            geometry.x.to_numpy(dtype=float),
            geometry.y.to_numpy(dtype=float),
        )
    )

    if not np.all(np.isfinite(coords)):
        raise ValueError("Extracted coordinates contain NaN or infinite values.")

    return coords


def validate_coords(
    coords: Union[np.ndarray, pd.DataFrame, "gpd.GeoDataFrame"],
) -> np.ndarray:
    """Validate coordinate data and return a floating-point array of shape (n, 2)."""
    try:
        geopandas = import_required_dependency(
            "geopandas", purpose="GeoDataFrame coordinate input"
        )
    except ImportError:
        geopandas = None
    if geopandas is not None and isinstance(coords, geopandas.GeoDataFrame):
        coords = _extract_geopandas_coords(coords)

    return _as_float_coordinate_array(
        coords,
        name="coords",
        require_two_columns=True,
    )


def validate_data(
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series],
) -> Tuple[np.ndarray, np.ndarray]:
    """Validate a single-response feature matrix and target vector."""
    if isinstance(X, pd.DataFrame):
        X_values = X.to_numpy()
    else:
        X_values = X

    if isinstance(y, pd.DataFrame):
        if y.shape[1] != 1:
            raise ValueError(
                "y must contain exactly one response variable; "
                f"got {y.shape[1]} columns."
            )
        y_values = y.iloc[:, 0].to_numpy()
    elif isinstance(y, pd.Series):
        y_values = y.to_numpy()
    else:
        y_values = y

    try:
        X_array = np.asarray(X_values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError("Feature matrix X must contain numeric values.") from exc

    try:
        y_array = np.asarray(y_values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError("Target vector y must contain numeric values.") from exc

    if X_array.ndim == 0:
        raise ValueError("X must be a one- or two-dimensional array.")
    if X_array.ndim == 1:
        X_array = X_array.reshape(-1, 1)
    elif X_array.ndim != 2:
        raise ValueError(
            f"X must be one- or two-dimensional; got {X_array.ndim} dimensions."
        )

    if y_array.ndim == 0:
        raise ValueError("y must be a one-dimensional response vector.")
    if y_array.ndim == 2:
        if y_array.shape[1] != 1:
            raise ValueError(
                "y must contain exactly one response variable; "
                f"got shape {y_array.shape}."
            )
        y_array = y_array[:, 0]
    elif y_array.ndim != 1:
        raise ValueError(f"y must be one-dimensional; got {y_array.ndim} dimensions.")

    if X_array.shape[0] == 0:
        raise ValueError("X and y must contain at least one sample.")
    if X_array.shape[1] == 0:
        raise ValueError("X must contain at least one feature.")

    if X_array.shape[0] != y_array.shape[0]:
        raise ValueError(
            f"X and y have incompatible shapes: X has {X_array.shape[0]} samples, "
            f"y has {y_array.shape[0]} samples."
        )

    if not np.all(np.isfinite(X_array)):
        raise ValueError("Feature matrix X contains NaN or infinite values.")
    if not np.all(np.isfinite(y_array)):
        raise ValueError("Target vector y contains NaN or infinite values.")

    return X_array, y_array
