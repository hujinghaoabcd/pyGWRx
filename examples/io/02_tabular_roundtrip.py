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
