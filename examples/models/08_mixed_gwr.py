# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Fit a semiparametric Mixed GWR with global and local predictors."""

# Allow this script to run directly from any working directory.
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_EXAMPLES_ROOT = _PROJECT_ROOT / "examples"
_SRC_ROOT = _PROJECT_ROOT / "src"
for _path in (_SRC_ROOT, _EXAMPLES_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from _common import mixed_regression, print_model_result

from pygwrx import MixedGWR

X, y, coords = mixed_regression()
model = MixedGWR(
    bandwidth=28,
    adaptive=True,
    global_vars=["global_x"],
    local_vars=["local_x"],
    intercept_fixed=True,
).fit(X, y, coords, compute_enp=False)
print_model_result(model)
print("global_coefficients=", model.coef_global_)
print("predictions=", model.predict(X.iloc[:3], coords.iloc[:3]))
