# Distances and validation

This page documents **12** public symbols. Each entry includes its purpose, import path, full API docstring, and the maintained example that exercises it.

[Conceptual guide](../../guides/core-numerics.md){ .md-button }

## `euclidean_distance`

Compute pairwise Euclidean distances. Compute Euclidean distances between coordinate arrays.

| Property | Value |
|---|---|
| Type | `function` |
| Import | `from pygwrx.core import euclidean_distance` |
| Signature | `euclidean_distance(coords1: 'np.ndarray', coords2: 'np.ndarray') -> 'np.ndarray'` |
| Maintained example | [`examples/core/02_distances_and_validation.py`](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/core/02_distances_and_validation.py) |

::: pygwrx.core.euclidean_distance


## `manhattan_distance`

Compute pairwise Manhattan (L1/city-block) distances.

| Property | Value |
|---|---|
| Type | `function` |
| Import | `from pygwrx.core import manhattan_distance` |
| Signature | `manhattan_distance(coords1: 'np.ndarray', coords2: 'np.ndarray') -> 'np.ndarray'` |
| Maintained example | [`examples/core/02_distances_and_validation.py`](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/core/02_distances_and_validation.py) |

::: pygwrx.core.manhattan_distance


## `chebyshev_distance`

Compute pairwise Chebyshev (L-infinity) distances.

| Property | Value |
|---|---|
| Type | `function` |
| Import | `from pygwrx.core import chebyshev_distance` |
| Signature | `chebyshev_distance(coords1: 'np.ndarray', coords2: 'np.ndarray') -> 'np.ndarray'` |
| Maintained example | [`examples/core/02_distances_and_validation.py`](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/core/02_distances_and_validation.py) |

::: pygwrx.core.chebyshev_distance


## `minkowski_distance`

Compute pairwise Minkowski (Lp) distances. Compute Minkowski distances between coordinate arrays.

| Property | Value |
|---|---|
| Type | `function` |
| Import | `from pygwrx.core import minkowski_distance` |
| Signature | `minkowski_distance(coords1: 'np.ndarray', coords2: 'np.ndarray', p: 'float' = 2.0) -> 'np.ndarray'` |
| Maintained example | [`examples/core/02_distances_and_validation.py`](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/core/02_distances_and_validation.py) |

::: pygwrx.core.minkowski_distance


## `haversine_distance`

Compute great-circle distances using the Haversine formula.

| Property | Value |
|---|---|
| Type | `function` |
| Import | `from pygwrx.core import haversine_distance` |
| Signature | `haversine_distance(coords1: 'np.ndarray', coords2: 'np.ndarray', radius: 'float' = 6371.0) -> 'np.ndarray'` |
| Maintained example | [`examples/core/02_distances_and_validation.py`](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/core/02_distances_and_validation.py) |

::: pygwrx.core.haversine_distance


## `compute_distance_matrix`

Compute a pairwise distance matrix.

| Property | Value |
|---|---|
| Type | `function` |
| Import | `from pygwrx.core import compute_distance_matrix` |
| Signature | `compute_distance_matrix(coords1: 'np.ndarray', coords2: 'Optional[np.ndarray]' = None, metric: 'str' = 'euclidean', **kwargs) -> 'np.ndarray'` |
| Maintained example | [`examples/core/02_distances_and_validation.py`](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/core/02_distances_and_validation.py) |

::: pygwrx.core.compute_distance_matrix


## `DistanceCache`

Distance-matrix cache policy based on actual matrix memory. Decide whether a distance matrix is small enough to cache.

| Property | Value |
|---|---|
| Type | `class` |
| Import | `from pygwrx.core import DistanceCache` |
| Signature | `DistanceCache()` |
| Maintained example | [`examples/core/02_distances_and_validation.py`](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/core/02_distances_and_validation.py) |

::: pygwrx.core.DistanceCache


## `validate_coords`

Validate coordinate data and return a floating-point array of shape (n, 2).

| Property | Value |
|---|---|
| Type | `function` |
| Import | `from pygwrx.core import validate_coords` |
| Signature | `validate_coords(coords: "Union[np.ndarray, pd.DataFrame, 'gpd.GeoDataFrame']") -> 'np.ndarray'` |
| Maintained example | [`examples/core/02_distances_and_validation.py`](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/core/02_distances_and_validation.py) |

::: pygwrx.core.validate_coords


## `validate_data`

Validate a single-response feature matrix and target vector.

| Property | Value |
|---|---|
| Type | `function` |
| Import | `from pygwrx.core import validate_data` |
| Signature | `validate_data(X: 'Union[np.ndarray, pd.DataFrame]', y: 'Union[np.ndarray, pd.Series]') -> 'Tuple[np.ndarray, np.ndarray]'` |
| Maintained example | [`examples/core/02_distances_and_validation.py`](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/core/02_distances_and_validation.py) |

::: pygwrx.core.validate_data


## `add_intercept`

Add a leading intercept column of ones to a feature matrix.

| Property | Value |
|---|---|
| Type | `function` |
| Import | `from pygwrx.core import add_intercept` |
| Signature | `add_intercept(X: 'np.ndarray') -> 'np.ndarray'` |
| Maintained example | [`examples/core/02_distances_and_validation.py`](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/core/02_distances_and_validation.py) |

::: pygwrx.core.add_intercept


## `extract_geopandas_coords`

Extract [x, y] coordinates from the active Point geometry column.

| Property | Value |
|---|---|
| Type | `function` |
| Import | `from pygwrx.core import extract_geopandas_coords` |
| Signature | `extract_geopandas_coords(gdf: "'gpd.GeoDataFrame'") -> 'np.ndarray'` |
| Maintained example | [`examples/core/03_geopandas_coordinates.py`](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/core/03_geopandas_coordinates.py) |

::: pygwrx.core.extract_geopandas_coords


## `chunked_computation`

Yield half-open ``(start, end)`` index ranges for chunked processing.

| Property | Value |
|---|---|
| Type | `function` |
| Import | `from pygwrx.core import chunked_computation` |
| Signature | `chunked_computation(n_items: 'int', chunk_size: 'int' = 1000) -> 'Iterator[Tuple[int, int]]'` |
| Maintained example | [`examples/core/02_distances_and_validation.py`](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/core/02_distances_and_validation.py) |

::: pygwrx.core.chunked_computation


## Runnable examples used on this page

??? example "`examples/core/02_distances_and_validation.py`"

    ```python
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
    ```

??? example "`examples/core/03_geopandas_coordinates.py`"

    ```python
    # SPDX-FileCopyrightText: 2026 Jinghao Hu
    # SPDX-License-Identifier: MIT
    
    """Extract coordinates from a GeoDataFrame using the base installation."""
    
    # Allow this script to run directly from any working directory.
    import sys
    from pathlib import Path
    
    _PROJECT_ROOT = Path(__file__).resolve().parents[2]
    _EXAMPLES_ROOT = _PROJECT_ROOT / "examples"
    _SRC_ROOT = _PROJECT_ROOT / "src"
    for _path in (_SRC_ROOT, _EXAMPLES_ROOT):
        if str(_path) not in sys.path:
            sys.path.insert(0, str(_path))
    
    import geopandas as gpd
    from shapely.geometry import Point
    
    from pygwrx.core import extract_geopandas_coords
    
    gdf = gpd.GeoDataFrame(
        {"name": ["a", "b"]},
        geometry=[Point(0.0, 1.0), Point(2.0, 3.0)],
        crs="EPSG:3857",
    )
    print(extract_geopandas_coords(gdf))
    ```
