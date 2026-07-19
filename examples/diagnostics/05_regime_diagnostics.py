# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Export observation, regime, and boundary summaries for GR-GWR."""

# Allow this script to run directly from any working directory.
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_EXAMPLES_ROOT = _PROJECT_ROOT / "examples"
_SRC_ROOT = _PROJECT_ROOT / "src"
for _path in (_SRC_ROOT, _EXAMPLES_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from _common import regime_regression

from pygwrx import GRGWR
from pygwrx.diagnostics import boundary_frame, regime_frame, regime_summary

X, y, coords, _ = regime_regression(n=54)
model = GRGWR(n_regimes=2, bandwidth=18, max_iter=2, random_state=0).fit(X, y, coords)
print(regime_frame(model).head())
print(regime_summary(model))
print(boundary_frame(model).head())
