# I/O examples

Bundled datasets, DataFrame/array conversion, GeoDataFrame round trips, and supported persistence helpers.

This page embeds **4** maintained scripts. The code shown here is read directly from `examples/io/`, so the documentation and executable source cannot silently diverge.

!!! tip "How to use this catalog"
    Read the purpose and APIs first, run the exact command, then use the inspection note to decide which output matters. For conceptual interpretation, follow the linked model or function guide rather than reading code alone.

## `01_bundled_datasets.py`

**Purpose.** List, describe, and load every bundled dataset and compatibility alias.

**Public APIs exercised.** `get_dataset_info`, `get_dublin_voter`, `get_dubvoter`, `list_datasets`, `load_columbus`, `load_crime`, `load_dataset`, `load_dublin_voter`, `load_dubvoter`, `load_ewhp`, `load_georgia`, `load_hiv`, `load_housing`

**Environment.** base installation.

**Run.** `python examples/io/01_bundled_datasets.py`

**What to inspect.** Verify column names, dtypes, coordinate order, CRS preservation, index alignment, and round-trip equality.

[Open source](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/io/01_bundled_datasets.py){ .md-button }

```python
# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""List, describe, and load every bundled dataset and compatibility alias."""

# Allow this script to run directly from any working directory.
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_EXAMPLES_ROOT = _PROJECT_ROOT / "examples"
_SRC_ROOT = _PROJECT_ROOT / "src"
for _path in (_SRC_ROOT, _EXAMPLES_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

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

## `02_tabular_roundtrip.py`

**Purpose.** Load a CSV and save NumPy/DataFrame results in tabular formats.

**Public APIs exercised.** `load_data`, `save_results`

**Environment.** base installation.

**Run.** `python examples/io/02_tabular_roundtrip.py`

**What to inspect.** Verify column names, dtypes, coordinate order, CRS preservation, index alignment, and round-trip equality.

[Open source](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/io/02_tabular_roundtrip.py){ .md-button }

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

## `03_geodataframe_roundtrip.py`

**Purpose.** Convert arrays to/from GeoDataFrame and save a GeoJSON result.

**Public APIs exercised.** `from_geodataframe`, `save_results`, `to_geodataframe`

**Environment.** base installation.

**Run.** `python examples/io/03_geodataframe_roundtrip.py`

**What to inspect.** Verify column names, dtypes, coordinate order, CRS preservation, index alignment, and round-trip equality.

[Open source](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/io/03_geodataframe_roundtrip.py){ .md-button }

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

## `04_dataset_return_types.py`

**Purpose.** Use supported dataset return types for arrays and GeoDataFrames.

**Public APIs exercised.** `load_dataset`

**Environment.** base installation.

**Run.** `python examples/io/04_dataset_return_types.py`

**What to inspect.** Verify column names, dtypes, coordinate order, CRS preservation, index alignment, and round-trip equality.

[Open source](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/io/04_dataset_return_types.py){ .md-button }

```python
# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Use supported dataset return types for arrays and GeoDataFrames."""

# Allow this script to run directly from any working directory.
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_EXAMPLES_ROOT = _PROJECT_ROOT / "examples"
_SRC_ROOT = _PROJECT_ROOT / "src"
for _path in (_SRC_ROOT, _EXAMPLES_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from pygwrx.io import load_dataset

X, y, coords = load_dataset("columbus", return_type="arrays")
print("columbus_arrays=", X.shape, y.shape, coords.shape)
gdf = load_dataset("georgia", return_type="geodataframe")
print("georgia_geodataframe=", gdf.shape, gdf.crs)
```
