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
