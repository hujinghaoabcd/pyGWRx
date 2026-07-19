# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT
"""Compare standard, robust, and ridge-compensated local regressions."""

# Allow this script to run directly from any working directory.
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_EXAMPLES_ROOT = _PROJECT_ROOT / "examples"
_SRC_ROOT = _PROJECT_ROOT / "src"
for _path in (_SRC_ROOT, _EXAMPLES_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from _common import spatial_regression

from pygwrx import GWR, LCRGWR, RGWR
from pygwrx.diagnostics import model_diagnostic_summary

X, y, coords = spatial_regression()
models = {
    "GWR": GWR(bandwidth=24, adaptive=True),
    "RGWR": RGWR(bandwidth=24, adaptive=True, max_iter=5),
    "LCRGWR": LCRGWR(bandwidth=24, adaptive=True),
}
for name, model in models.items():
    model.fit(X, y, coords)
    summary = model_diagnostic_summary(model)
    print(name, summary)
