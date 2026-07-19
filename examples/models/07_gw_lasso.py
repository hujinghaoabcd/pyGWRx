# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Fit geographically weighted Lasso with a fixed local penalty."""

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

from pygwrx import GWLasso

X, y, coords = spatial_regression(n=48, p=3)
model = GWLasso(
    bandwidth=24, adaptive=True, alpha=0.06, max_iter=1000, random_state=0
).fit(X, y, coords)
print_model_result(model)
print("selection_frequency=", model.selection_frequency_)
print("predictions=", model.predict(X.iloc[:3], coords.iloc[:3]))
