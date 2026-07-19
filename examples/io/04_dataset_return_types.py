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
