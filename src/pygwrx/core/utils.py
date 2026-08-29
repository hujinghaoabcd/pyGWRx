# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Utility functions for spatial data and numerical validation.

The helpers in this module cover distance calculation, GeoPandas extraction,
intercept construction, caching, chunked execution, and compatibility imports
for validation helpers now owned by :mod:`pygwrx.core.validation`.

Author:
    Jinghao Hu
"""

from __future__ import annotations

__author__ = "Jinghao Hu"
__license__ = "MIT"

from numbers import Integral
from typing import TYPE_CHECKING, Iterator, Optional, Tuple

import numpy as np
from scipy.spatial.distance import cdist

from pygwrx.core.validation import (  # noqa: F401
    CoordinateInput,
    _extract_geopandas_coords,
    _validate_coordinate_pair,
    _validate_count,
    _validate_positive_scalar,
    validate_coords,
    validate_data,
)

if TYPE_CHECKING:
    import geopandas as gpd


def _normalize_task(task: str) -> str:
    """Validate and normalize a distance-cache task name."""
    if not isinstance(task, str):
        raise TypeError("task must be a string.")

    task_name = task.strip().lower()
    if task_name not in {"gwr", "bandwidth"}:
        raise ValueError("task must be either 'gwr' or 'bandwidth'.")

    return task_name


def euclidean_distance(coords1: np.ndarray, coords2: np.ndarray) -> np.ndarray:
    """Compute pairwise Euclidean distances.

    Integer inputs are converted to float before arithmetic to prevent overflow.
    """
    coords1_arr, coords2_arr = _validate_coordinate_pair(coords1, coords2)
    return cdist(coords1_arr, coords2_arr, metric="euclidean")


def manhattan_distance(coords1: np.ndarray, coords2: np.ndarray) -> np.ndarray:
    """Compute pairwise Manhattan (L1/city-block) distances."""
    coords1_arr, coords2_arr = _validate_coordinate_pair(coords1, coords2)
    return cdist(coords1_arr, coords2_arr, metric="cityblock")


def chebyshev_distance(coords1: np.ndarray, coords2: np.ndarray) -> np.ndarray:
    """Compute pairwise Chebyshev (L-infinity) distances."""
    coords1_arr, coords2_arr = _validate_coordinate_pair(coords1, coords2)
    return cdist(coords1_arr, coords2_arr, metric="chebyshev")


def minkowski_distance(
    coords1: np.ndarray,
    coords2: np.ndarray,
    p: float = 2.0,
) -> np.ndarray:
    """Compute pairwise Minkowski (Lp) distances.

    Args:
        p: Norm order. It must be finite and >= 1, or positive infinity.
    """
    if isinstance(p, (bool, np.bool_)):
        raise TypeError("p must be a real scalar, not bool.")

    p_array = np.asarray(p)
    if p_array.ndim != 0:
        raise TypeError("p must be a scalar value.")

    try:
        p_value = float(p_array)
    except (TypeError, ValueError) as exc:
        raise TypeError("p must be a real scalar.") from exc

    if np.isnan(p_value):
        raise ValueError("p cannot be NaN.")
    if p_value < 1:
        raise ValueError("p must be greater than or equal to 1.")

    coords1_arr, coords2_arr = _validate_coordinate_pair(coords1, coords2)

    if p_value == 1:
        return cdist(coords1_arr, coords2_arr, metric="cityblock")
    if p_value == 2:
        return cdist(coords1_arr, coords2_arr, metric="euclidean")
    if np.isposinf(p_value):
        return cdist(coords1_arr, coords2_arr, metric="chebyshev")
    if not np.isfinite(p_value):
        raise ValueError("p must be finite or positive infinity.")

    return cdist(coords1_arr, coords2_arr, metric="minkowski", p=p_value)


def haversine_distance(
    coords1: np.ndarray,
    coords2: np.ndarray,
    radius: float = 6371.0,
) -> np.ndarray:
    """Compute great-circle distances using the Haversine formula.

    Coordinates must be ordered as [longitude, latitude] in degrees.
    The output unit is the same as the unit used for ``radius``.
    """
    coords1_arr, coords2_arr = _validate_coordinate_pair(
        coords1,
        coords2,
        require_two_columns=True,
    )
    radius_value = _validate_positive_scalar(radius, name="radius")

    for name, coords_arr in (("coords1", coords1_arr), ("coords2", coords2_arr)):
        latitudes = coords_arr[:, 1]
        if np.any((latitudes < -90.0) | (latitudes > 90.0)):
            raise ValueError(
                f"{name} latitude values must lie within [-90, 90] degrees."
            )

    lon1 = np.radians(coords1_arr[:, 0])
    lat1 = np.radians(coords1_arr[:, 1])
    lon2 = np.radians(coords2_arr[:, 0])
    lat2 = np.radians(coords2_arr[:, 1])

    dlon = lon2[np.newaxis, :] - lon1[:, np.newaxis]
    dlat = lat2[np.newaxis, :] - lat1[:, np.newaxis]

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1[:, np.newaxis])
        * np.cos(lat2[np.newaxis, :])
        * np.sin(dlon / 2.0) ** 2
    )
    a = np.clip(a, 0.0, 1.0)
    central_angle = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))

    return radius_value * central_angle


def compute_distance_matrix(
    coords1: np.ndarray,
    coords2: Optional[np.ndarray] = None,
    metric: str = "euclidean",
    **kwargs,
) -> np.ndarray:
    """Compute a pairwise distance matrix.

    Supported metrics are ``euclidean``, ``manhattan``, ``chebyshev``,
    ``minkowski``, and ``haversine``.
    """
    if coords2 is None:
        coords2 = coords1

    if not isinstance(metric, str):
        raise TypeError("metric must be a string.")

    metric_name = metric.strip().lower()
    aliases = {
        "l1": "manhattan",
        "cityblock": "manhattan",
        "l2": "euclidean",
        "great_circle": "haversine",
    }
    metric_name = aliases.get(metric_name, metric_name)

    supported = {"euclidean", "manhattan", "chebyshev", "minkowski", "haversine"}
    if metric_name not in supported:
        available = ", ".join(sorted(supported))
        raise ValueError(
            f"Unknown distance metric: {metric!r}. Available metrics: {available}."
        )

    allowed_kwargs = {
        "euclidean": set(),
        "manhattan": set(),
        "chebyshev": set(),
        "minkowski": {"p"},
        "haversine": {"radius"},
    }[metric_name]
    unexpected = set(kwargs) - allowed_kwargs
    if unexpected:
        unexpected_text = ", ".join(sorted(unexpected))
        raise TypeError(
            f"Unexpected parameter(s) for metric {metric_name!r}: {unexpected_text}."
        )

    if metric_name == "euclidean":
        return euclidean_distance(coords1, coords2)
    if metric_name == "manhattan":
        return manhattan_distance(coords1, coords2)
    if metric_name == "chebyshev":
        return chebyshev_distance(coords1, coords2)
    if metric_name == "minkowski":
        return minkowski_distance(coords1, coords2, p=kwargs.get("p", 2.0))

    return haversine_distance(coords1, coords2, radius=kwargs.get("radius", 6371.0))


class DistanceCache:
    """Distance-matrix cache policy based on actual matrix memory.

    The class retains the original public name for compatibility. It is a policy/advisor;
    it does not itself store matrices.
    """

    THRESHOLD_GWR = 5000
    THRESHOLD_BW = 10000

    MAX_CACHE_MEMORY_GWR_BYTES = THRESHOLD_GWR * THRESHOLD_GWR * 8
    MAX_CACHE_MEMORY_BW_BYTES = THRESHOLD_BW * THRESHOLD_BW * 8

    @staticmethod
    def should_cache(n_data: int, n_pred: int, task: str = "gwr") -> bool:
        """Return whether the required distance matrix fits the default cache budget."""
        n_data_int = _validate_count(n_data, name="n_data")
        n_pred_int = _validate_count(n_pred, name="n_pred")
        task_name = _normalize_task(task)

        if task_name == "bandwidth":
            required_bytes = n_data_int * n_data_int * 8
            limit_bytes = DistanceCache.MAX_CACHE_MEMORY_BW_BYTES
        else:
            required_bytes = n_data_int * n_pred_int * 8
            limit_bytes = DistanceCache.MAX_CACHE_MEMORY_GWR_BYTES

        return required_bytes <= limit_bytes

    @staticmethod
    def estimate_memory(n_data: int, n_pred: int) -> Tuple[int, str]:
        """Estimate memory occupied by a float64 distance matrix."""
        n_data_int = _validate_count(n_data, name="n_data")
        n_pred_int = _validate_count(n_pred, name="n_pred")
        size_bytes = n_data_int * n_pred_int * 8

        if size_bytes < 1024**2:
            size_str = f"{size_bytes / 1024:.1f} KB"
        elif size_bytes < 1024**3:
            size_str = f"{size_bytes / 1024**2:.1f} MB"
        else:
            size_str = f"{size_bytes / 1024**3:.2f} GB"

        return size_bytes, size_str

    @staticmethod
    def get_strategy(n_data: int, n_pred: int, task: str = "gwr") -> str:
        """Return ``'cache'`` or ``'on-the-fly'``."""
        return (
            "cache"
            if DistanceCache.should_cache(n_data, n_pred, task)
            else "on-the-fly"
        )

    @staticmethod
    def print_recommendation(n_data: int, n_pred: int, task: str = "gwr") -> None:
        """Print a detailed distance-matrix caching recommendation."""
        n_data_int = _validate_count(n_data, name="n_data")
        n_pred_int = _validate_count(n_pred, name="n_pred")
        task_name = _normalize_task(task)
        strategy = DistanceCache.get_strategy(n_data_int, n_pred_int, task_name)

        if task_name == "bandwidth":
            matrix_rows = n_data_int
            matrix_cols = n_data_int
            limit_bytes = DistanceCache.MAX_CACHE_MEMORY_BW_BYTES
            task_label = "Bandwidth Selection"
        else:
            matrix_rows = n_data_int
            matrix_cols = n_pred_int
            limit_bytes = DistanceCache.MAX_CACHE_MEMORY_GWR_BYTES
            task_label = "GWR Fitting"

        _, size_str = DistanceCache.estimate_memory(matrix_rows, matrix_cols)
        _, limit_str = DistanceCache.estimate_memory(1, limit_bytes // 8)

        print("Distance Matrix Caching Recommendation")
        print("=" * 50)
        print(f"Task: {task_label}")
        print(f"Data points: {n_data_int}")
        print(f"Prediction points: {n_pred_int}")
        print(f"Matrix shape: {matrix_rows} x {matrix_cols}")
        print(f"Memory required: {size_str}")
        print(f"Default cache budget: {limit_str}")
        print(f"\nRecommendation: {strategy.upper()}")

        if strategy == "cache":
            print("Reason: The distance matrix fits within the default cache budget.")
            print("Action: Precompute and cache the distance matrix.")
        else:
            print("Reason: The distance matrix exceeds the default cache budget.")
            print("Action: Compute distances on-the-fly or in chunks.")


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


def chunked_computation(
    n_items: int,
    chunk_size: int = 1000,
) -> Iterator[Tuple[int, int]]:
    """Yield half-open ``(start, end)`` index ranges for chunked processing."""
    n_items_int = _validate_count(n_items, name="n_items")

    if isinstance(chunk_size, (bool, np.bool_)) or not isinstance(
        chunk_size,
        Integral,
    ):
        raise TypeError("chunk_size must be a positive integer.")

    chunk_size_int = int(chunk_size)
    if chunk_size_int <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    for start in range(0, n_items_int, chunk_size_int):
        end = min(start + chunk_size_int, n_items_int)
        yield start, end


_DEFAULT_DISTANCE_BLOCK_ROWS = 128


def _iter_distance_blocks(
    target_coords: CoordinateInput,
    source_coords: Optional[CoordinateInput] = None,
    *,
    distance_metric: str = "euclidean",
    block_rows: int = _DEFAULT_DISTANCE_BLOCK_ROWS,
) -> Iterator[np.ndarray]:
    """Yield bounded target-to-source pairwise distance blocks."""
    if source_coords is None:
        source_coords = target_coords
    targets, sources = _validate_coordinate_pair(target_coords, source_coords)
    if isinstance(block_rows, (bool, np.bool_)) or not isinstance(block_rows, Integral):
        raise TypeError("block_rows must be a positive integer.")
    block_rows_int = int(block_rows)
    if block_rows_int <= 0:
        raise ValueError("block_rows must be greater than zero.")

    n_sources = sources.shape[0]
    for start, stop in chunked_computation(targets.shape[0], block_rows_int):
        block = np.asarray(
            compute_distance_matrix(
                targets[start:stop],
                sources,
                metric=distance_metric,
            ),
            dtype=float,
        )
        expected_shape = (stop - start, n_sources)
        if block.shape != expected_shape:
            raise ValueError(
                "The distance implementation returned an unexpected block shape."
            )
        if not np.all(np.isfinite(block)) or np.any(block < 0.0):
            raise ValueError("The distance implementation returned invalid distances.")
        yield block


def _iter_distance_rows(
    target_coords: CoordinateInput,
    source_coords: Optional[CoordinateInput] = None,
    *,
    distance_metric: str = "euclidean",
    block_rows: int = _DEFAULT_DISTANCE_BLOCK_ROWS,
) -> Iterator[np.ndarray]:
    """Yield target-to-source distance rows from bounded-size blocks."""
    for block in _iter_distance_blocks(
        target_coords,
        source_coords,
        distance_metric=distance_metric,
        block_rows=block_rows,
    ):
        yield from block
