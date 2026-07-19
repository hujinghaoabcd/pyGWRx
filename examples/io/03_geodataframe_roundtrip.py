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
