# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Fit MGWR with fixed variable-specific bandwidths."""

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

from pygwrx import MGWR

X, y, coords = spatial_regression(n=48, p=2)
model = MGWR(bandwidths=[24, 26, 28], adaptive=True, max_iter=8, tol=0.5).fit(
    X, y, coords, compute_inference=True
)
print_model_result(model)
try:
    model.predict(X.iloc[:2], coords.iloc[:2])
except NotImplementedError as exc:
    print("Expected MGWR prediction limitation:", exc)
