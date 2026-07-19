# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Fit the self-contained multiscale geographically and temporally weighted regression."""

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

from pygwrx import MGTWR

X, y, coords, times = temporal_regression(n=20, p=2)
model = MGTWR(
    bandwidths=[12, 12, 12],
    taus=[1.0, 1.0, 1.0],
    adaptive=True,
    calculate_inference=False,
).fit(X, y, coords, times)
print_model_result(model)
print("spatial_bandwidths=", model.bandwidths_)
print("temporal_scales=", model.taus_)
try:
    model.predict(X.iloc[:2], coords.iloc[:2], times[:2])
except NotImplementedError as exc:
    print("Expected MGTWR prediction limitation:", exc)
