# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Fit similarity and geographically-temporally weighted regression."""

# Allow this script to run directly from any working directory.
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_EXAMPLES_ROOT = _PROJECT_ROOT / "examples"
_SRC_ROOT = _PROJECT_ROOT / "src"
for _path in (_SRC_ROOT, _EXAMPLES_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from _common import print_model_result, temporal_regression

from pygwrx import SGTWR, SGTWRPredictionResult

X, y, coords, times = temporal_regression(n=48, p=3)
model = SGTWR(
    spatial_bandwidth=24,
    temporal_bandwidth=2.0,
    adaptive=True,
    alpha=0.5,
    similarity_vars=["x1", "x2"],
    store_weights=True,
).fit(X, y, coords, times)
print_model_result(model)
print("combined_weights_shape=", model.combined_weights_.shape)
result = model.predict_result(X.iloc[:3], coords.iloc[:3], times[:3])
assert isinstance(result, SGTWRPredictionResult)
print(result.to_frame())
