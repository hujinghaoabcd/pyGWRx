# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Fit STWR from multiple observation snapshots."""

# Allow this script to run directly from any working directory.
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_EXAMPLES_ROOT = _PROJECT_ROOT / "examples"
_SRC_ROOT = _PROJECT_ROOT / "src"
for _path in (_SRC_ROOT, _EXAMPLES_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from _common import print_model_result, stwr_stages

from pygwrx import STWR, STWRPredictionResult

X_list, y_list, coords_list, intervals = stwr_stages()
model = STWR(
    spatial_bandwidth=10,
    adaptive=True,
    alpha=0.3,
    theta=0.0,
    tick_nums=2,
    store_weights=True,
).fit(X_list, y_list, coords_list, intervals)
print_model_result(model)
result = model.predict_result(
    X_list[-1].iloc[:3],
    coords_list[-1].iloc[:3],
    reference_y=y_list[-1][:3],
)
assert isinstance(result, STWRPredictionResult)
print(result.to_frame())
