# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Fit locally compensated ridge GWR for collinear predictors."""

# Allow this script to run directly from any working directory.
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_EXAMPLES_ROOT = _PROJECT_ROOT / "examples"
_SRC_ROOT = _PROJECT_ROOT / "src"
for _path in (_SRC_ROOT, _EXAMPLES_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from _common import collinear_regression, print_model_result

from pygwrx import LCRGWR

X, y, coords = collinear_regression()
model = LCRGWR(bandwidth=28, adaptive=True, cn_thresh=15.0, lambda_adjust=True).fit(
    X, y, coords
)
print_model_result(model)
print("local_condition_numbers=", model.local_condition_numbers_[:5])
print("local_lambdas=", model.local_lambdas_[:5])
