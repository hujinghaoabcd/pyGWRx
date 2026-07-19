# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Run a real-data GWR workflow from bundled data to geospatial output."""

from __future__ import annotations

# Allow this script to run directly from any working directory.
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_EXAMPLES_ROOT = _PROJECT_ROOT / "examples"
_SRC_ROOT = _PROJECT_ROOT / "src"
for _path in (_SRC_ROOT, _EXAMPLES_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from _common import OUTPUT_DIR

from pygwrx import GWR
from pygwrx.diagnostics import diagnostics_frame, parameter_inference
from pygwrx.io import load_georgia, save_results

bundle = load_georgia(return_type="dict")
X = bundle["frame"][bundle["feature_names"]]
y = bundle["frame"][bundle["target_name"]]
coords = bundle["coords"]

model = GWR(kernel="bisquare", bandwidth=48, adaptive=True).fit(X, y, coords)
print(model.summary())
print(diagnostics_frame([model], labels=["GWR"]))
print(parameter_inference(model, bundle["feature_names"][0]))
print(model.predict_result(X[:5], coords[:5]).to_frame())

result_frame = model.to_frame()
save_results(result_frame, OUTPUT_DIR / "georgia_gwr_results.csv")

geo = bundle["frame"].copy()
for column in result_frame.columns:
    if column not in geo.columns and len(result_frame[column]) == len(geo):
        geo[column] = result_frame[column].to_numpy()
save_results(geo, OUTPUT_DIR / "georgia_gwr_results.geojson")
