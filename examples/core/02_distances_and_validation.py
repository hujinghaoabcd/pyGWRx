# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Use all public distance, validation, caching, and chunk helpers."""

# Allow this script to run directly from any working directory.
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_EXAMPLES_ROOT = _PROJECT_ROOT / "examples"
_SRC_ROOT = _PROJECT_ROOT / "src"
for _path in (_SRC_ROOT, _EXAMPLES_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import numpy as np
import pandas as pd

from pygwrx.core import (
    DistanceCache,
    add_intercept,
    chebyshev_distance,
    chunked_computation,
    compute_distance_matrix,
    euclidean_distance,
    haversine_distance,
    manhattan_distance,
    minkowski_distance,
    validate_coords,
    validate_data,
)

a = np.array([[0.0, 0.0], [1.0, 2.0]])
b = np.array([[2.0, 1.0], [3.0, 4.0]])
print("euclidean=", euclidean_distance(a, b))
print("manhattan=", manhattan_distance(a, b))
print("chebyshev=", chebyshev_distance(a, b))
print("minkowski_p3=", minkowski_distance(a, b, p=3.0))
print(
    "haversine_km=",
    haversine_distance(np.array([[116.4, 39.9]]), np.array([[121.5, 31.2]])),
)
print("matrix=", compute_distance_matrix(a, metric="euclidean"))

X, y = validate_data(pd.DataFrame({"x": [1, 2]}), pd.Series([3, 4]))
coords = validate_coords(pd.DataFrame(a, columns=["x", "y"]))
print("validated_shapes=", X.shape, y.shape, coords.shape)
print("with_intercept=", add_intercept(X))
print("chunks=", list(chunked_computation(10, chunk_size=4)))
print("cache_memory=", DistanceCache.estimate_memory(100, 50))
print("cache_strategy=", DistanceCache.get_strategy(100, 50, task="gwr"))
print("should_cache=", DistanceCache.should_cache(100, 50))
DistanceCache.print_recommendation(100, 50)
