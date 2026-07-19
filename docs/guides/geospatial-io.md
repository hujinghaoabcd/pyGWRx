# Data conversion, datasets, and persistence

The standard installation accepts NumPy arrays, pandas objects, and GeoPandas objects. PyArrow remains optional because it is only needed for Parquet and GeoParquet persistence.

## Data contract

- `X`: two-dimensional numeric matrix or DataFrame.
- `y`: one-dimensional response/class array where required.
- `coords`: `(n, 2)` numeric coordinates or a DataFrame with two coordinate columns.
- `times`: one value per row for GTWR/SGTWR/MGTWR, or ordered stage intervals for STWR.
- Geometry: convert to projected numeric coordinates before Euclidean modelling.

## Installation

GeoPandas and Shapely are installed with pyGWRx. Add PyArrow only when Parquet or GeoParquet persistence is required:

```bash
pip install -e ".[parquet]"  # PyArrow
```

## Maintained examples

### Bundled dataset registry

```python
# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""List, describe, and load every bundled dataset and compatibility alias."""

from pygwrx.io import (
    get_dataset_info,
    get_dublin_voter,
    get_dubvoter,
    list_datasets,
    load_columbus,
    load_crime,
    load_dataset,
    load_dublin_voter,
    load_dubvoter,
    load_ewhp,
    load_georgia,
    load_hiv,
    load_housing,
)

loaders = {
    "dublin_voter": load_dublin_voter,
    "hiv": load_hiv,
    "crime": load_crime,
    "housing": load_housing,
    "columbus": load_columbus,
    "ewhp": load_ewhp,
    "georgia": load_georgia,
}
print("datasets=", list_datasets(verbose=False))
for name, loader in loaders.items():
    info = get_dataset_info(name)
    frame = loader(return_type="frame")
    generic = load_dataset(name, return_type="frame")
    print(name, info["n_samples"], frame.shape, generic.shape)
print(
    "alias_shapes=",
    [
        fn(return_type="frame").shape
        for fn in (get_dublin_voter, load_dubvoter, get_dubvoter)
    ],
)
```

### Tabular round trip

```python
# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Load a CSV and save NumPy/DataFrame results in tabular formats."""

import numpy as np
import pandas as pd
from pygwrx.io import load_data, save_results
from _common import OUTPUT_DIR

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

### GeoDataFrame round trip

```python
# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Convert arrays to/from GeoDataFrame and save a GeoJSON result."""

from pygwrx.io import from_geodataframe, save_results, to_geodataframe
from _common import OUTPUT_DIR, spatial_regression

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

### Dataset return types

```python
# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Use supported dataset return types for arrays and GeoDataFrames."""

from pygwrx.io import load_dataset

X, y, coords = load_dataset("columbus", return_type="arrays")
print("columbus_arrays=", X.shape, y.shape, coords.shape)
gdf = load_dataset("georgia", return_type="geodataframe")
print("georgia_geodataframe=", gdf.shape, gdf.crs)
```

## Redistribution warning

The development tree contains third-party example data. Citation, academic-use permission, and permission to redistribute a dataset inside a wheel are separate legal questions. Confirm each dataset licence before public distribution.

See the [I/O API](../api/io/index.md) and [Data and inputs](../getting-started/data-and-inputs.md).
