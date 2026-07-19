# Data conversion and persistence

This page documents **4** public symbols. Each entry includes its purpose, import path, full API docstring, and the maintained example that exercises it.

[Conceptual guide](../../guides/geospatial-io.md){ .md-button }

## `load_data`

Load a user data file and extract model features, target, and coordinates. Extract predictors, an optional response, and coordinates from a user data file.

| Property | Value |
|---|---|
| Type | `function` |
| Import | `from pygwrx.io import load_data` |
| Signature | `load_data(filepath: 'PathLike', x_cols: 'Optional[Sequence[str]]' = None, y_col: 'Optional[str]' = None, coord_cols: 'Optional[Tuple[str, str]]' = None, *, dropna: 'bool' = True) -> 'Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]'` |
| Maintained example | [`examples/io/02_tabular_roundtrip.py`](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/io/02_tabular_roundtrip.py) |

::: pygwrx.io.load_data


## `to_geodataframe`

Convert aligned arrays into a point GeoDataFrame.

| Property | Value |
|---|---|
| Type | `function` |
| Import | `from pygwrx.io import to_geodataframe` |
| Signature | `to_geodataframe(X: 'np.ndarray', y: 'Optional[np.ndarray]', coords: 'np.ndarray', feature_names: 'Optional[Sequence[str]]' = None, target_name: 'str' = 'target', crs: 'Optional[Union[str, int]]' = None) -> 'gpd.GeoDataFrame'` |
| Maintained example | [`examples/io/03_geodataframe_roundtrip.py`](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/io/03_geodataframe_roundtrip.py) |

::: pygwrx.io.to_geodataframe


## `from_geodataframe`

Extract aligned arrays from a point GeoDataFrame.

| Property | Value |
|---|---|
| Type | `function` |
| Import | `from pygwrx.io import from_geodataframe` |
| Signature | `from_geodataframe(gdf: "'gpd.GeoDataFrame'", x_cols: 'Optional[Sequence[str]]' = None, y_col: 'Optional[str]' = None, *, dropna: 'bool' = True) -> 'Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]'` |
| Maintained example | [`examples/io/03_geodataframe_roundtrip.py`](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/io/03_geodataframe_roundtrip.py) |

::: pygwrx.io.from_geodataframe


## `save_results`

Save model results to CSV, Parquet, Shapefile, GeoJSON, or GeoPackage.

| Property | Value |
|---|---|
| Type | `function` |
| Import | `from pygwrx.io import save_results` |
| Signature | `save_results(results: "Union[np.ndarray, pd.DataFrame, 'gpd.GeoDataFrame']", filepath: 'PathLike', format: 'Optional[str]' = None) -> 'Path'` |
| Maintained example | [`examples/io/02_tabular_roundtrip.py`](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/io/02_tabular_roundtrip.py) |

::: pygwrx.io.save_results


## Runnable examples used on this page

??? example "`examples/io/02_tabular_roundtrip.py`"

    ```python
    # SPDX-FileCopyrightText: 2026 Jinghao Hu
    # SPDX-License-Identifier: MIT
    
    """Load a CSV and save NumPy/DataFrame results in tabular formats."""
    
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
    from _common import OUTPUT_DIR
    
    from pygwrx.io import load_data, save_results
    
    source = OUTPUT_DIR / "io_input.csv"
    pd.DataFrame(
        {
            "east": [0.0, 1.0, 2.0],
            "north": [1.0, 1.5, 2.0],
            "x1": [2.0, 3.0, 4.0],
            "x2": [1.0, 0.0, 1.0],
            "target": [5.0, 6.0, 8.0],
        }
    ).to_csv(source, index=False)
    X, y, coords = load_data(
        source, x_cols=["x1", "x2"], y_col="target", coord_cols=("east", "north")
    )
    print("loaded=", X.shape, y.shape, coords.shape)
    print("csv=", save_results(np.column_stack((y, X)), OUTPUT_DIR / "array_results.csv"))
    try:
        print(
            "parquet=",
            save_results(
                pd.DataFrame(X, columns=["x1", "x2"]),
                OUTPUT_DIR / "frame_results",
                format="parquet",
            ),
        )
    except ImportError as exc:
        print("Parquet is optional; install pyGWRx[parquet]:", exc)
    ```

??? example "`examples/io/03_geodataframe_roundtrip.py`"

    ```python
    # SPDX-FileCopyrightText: 2026 Jinghao Hu
    # SPDX-License-Identifier: MIT
    
    """Convert arrays to/from GeoDataFrame and save a GeoJSON result."""
    
    # Allow this script to run directly from any working directory.
    import sys
    from pathlib import Path
    
    _PROJECT_ROOT = Path(__file__).resolve().parents[2]
    _EXAMPLES_ROOT = _PROJECT_ROOT / "examples"
    _SRC_ROOT = _PROJECT_ROOT / "src"
    for _path in (_SRC_ROOT, _EXAMPLES_ROOT):
        if str(_path) not in sys.path:
            sys.path.insert(0, str(_path))
    
    from _common import OUTPUT_DIR, spatial_regression
    
    from pygwrx.io import from_geodataframe, save_results, to_geodataframe
    
    X, y, coords = spatial_regression(n=8, p=2)
    gdf = to_geodataframe(
        X.to_numpy(),
        y,
        coords.to_numpy(),
        feature_names=list(X.columns),
        target_name="response",
        crs="EPSG:3857",
    )
    print(gdf.head())
    X2, y2, coords2 = from_geodataframe(gdf, x_cols=list(X.columns), y_col="response")
    print("roundtrip=", X2.shape, y2.shape, coords2.shape)
    print("geojson=", save_results(gdf, OUTPUT_DIR / "spatial_results.geojson"))
    ```
