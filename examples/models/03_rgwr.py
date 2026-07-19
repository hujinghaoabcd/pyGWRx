# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Fit robust GWR in automatic down-weighting mode."""

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
from _common import print_model_result, spatial_regression

from pygwrx import RGWR

X, y, coords = spatial_regression()
y = y.copy()
y[[2, 20]] += np.array([5.0, -4.0])
model = RGWR(bandwidth=24, adaptive=True, max_iter=8).fit(X, y, coords)
print_model_result(model)
print("robust_weights=", model.robust_weights_[:8])
print("predictions=", model.predict(X.iloc[:3], coords.iloc[:3]))
