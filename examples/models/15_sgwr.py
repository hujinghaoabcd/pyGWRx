# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Fit similarity and geographically weighted regression."""

# Allow this script to run directly from any working directory.
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_EXAMPLES_ROOT = _PROJECT_ROOT / "examples"
_SRC_ROOT = _PROJECT_ROOT / "src"
for _path in (_SRC_ROOT, _EXAMPLES_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from _common import print_model_result, spatial_regression

from pygwrx import SGWR

X, y, coords = spatial_regression(n=48, p=3)
model = SGWR(
    bandwidth=24,
    adaptive=True,
    alpha=0.45,
    similarity_vars=["x1", "x2"],
    store_weights=True,
).fit(X, y, coords)
print_model_result(model)
print("combined_weights_shape=", model.combined_weights_.shape)
print("predictions=", model.predict(X.iloc[:3], coords.iloc[:3]))
