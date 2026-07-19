# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Model-aware coefficient maps plus all historical array-based maps."""

# Allow this script to run directly from any working directory.
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_EXAMPLES_ROOT = _PROJECT_ROOT / "examples"
_SRC_ROOT = _PROJECT_ROOT / "src"
for _path in (_SRC_ROOT, _EXAMPLES_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import matplotlib

matplotlib.use("Agg", force=True)
import geopandas as gpd
import numpy as np
from _common import save_plot
from _models import surface_models
from shapely.geometry import Point

from pygwrx.plotting import (
    create_choropleth,
    plot_array_significance_map,
    plot_bandwidth,
    plot_coefficient_map,
    plot_coefficient_surface,
    plot_local_coefficients,
    plot_local_diagnostic_map,
    plot_local_r2,
    plot_model_significance_map,
    plot_multiple_coefficients,
    plot_significance_map,
)

X, y, coords, gwr, _, _ = surface_models()
coords_array = coords.to_numpy()
p_values = np.full_like(gwr.coef_, 0.02)
plots = {
    "coefficient_map.png": plot_coefficient_map(gwr, "x1", theme="paper"),
    "model_significance.png": plot_model_significance_map(gwr, "x1", correction="raw"),
    "dispatch_model_significance.png": plot_significance_map(gwr, "x1"),
    "local_diagnostic.png": plot_local_diagnostic_map(gwr, "local_r2"),
    "array_significance.png": plot_array_significance_map(
        coords_array, p_values, feature_idx=0, coefficients=gwr.coef_
    ),
    "dispatch_array_significance.png": plot_significance_map(
        coords_array, p_values, feature_idx=0, coefficients=gwr.coef_
    ),
    "local_coefficients.png": plot_local_coefficients(coords_array, gwr.coef_, 0, "x1"),
    "coefficient_surface.png": plot_coefficient_surface(
        coords_array, gwr.coef_, 0, interpolation="nearest"
    ),
    "array_local_r2.png": plot_local_r2(coords_array, gwr.local_r2_),
    "bandwidth_map.png": plot_bandwidth(
        coords_array, 2.0, sample_locations=coords_array[:3]
    ),
    "multiple_coefficients.png": plot_multiple_coefficients(
        coords_array, gwr.coef_, feature_names=["x1", "x2"], shared_scale=True
    ),
}
for name, result in plots.items():
    print(save_plot(result, name))
gdf = gpd.GeoDataFrame(
    {"value": gwr.coef_[:, 0]},
    geometry=[Point(x, y) for x, y in coords_array],
    crs="EPSG:3857",
)
print(save_plot(create_choropleth(gdf, "value"), "choropleth.png"))
