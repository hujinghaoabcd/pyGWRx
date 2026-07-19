# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Fit and predict with geographically and temporally weighted regression."""

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

from pygwrx import GTWR, GTWRPredictionResult

X, y, coords, times = temporal_regression()
model = GTWR(kernel="bisquare", bandwidth=24, adaptive=True, lambda_st=0.3).fit(
    X, y, coords, times
)
print_model_result(model)
print("score=", model.score(X, y, coords, times=times))
result = model.predict_result(X.iloc[:3], coords.iloc[:3], times[:3])
assert isinstance(result, GTWRPredictionResult)
print(result.to_frame())
