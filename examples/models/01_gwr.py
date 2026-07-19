# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Load a bundled real dataset, fit GWR, inspect it, and predict."""

from __future__ import annotations

# Allow this script to run directly from any working directory.
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_EXAMPLES_ROOT = _PROJECT_ROOT / "examples"
_SRC_ROOT = _PROJECT_ROOT / "src"
for _path in (_SRC_ROOT, _EXAMPLES_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from pygwrx import GWR, GWRPredictionResult
from pygwrx.io import load_columbus

bundle = load_columbus(return_type="dict")
X = bundle["data"]
y = bundle["target"]
coords = bundle["coords"]

print("dataset=", bundle["description"])
print("features=", bundle["feature_names"])
print("license=", bundle["license"])

model = GWR(kernel="bisquare", bandwidth=24, adaptive=True).fit(X, y, coords)
print(model.summary())
print("score=", model.score(X, y, coords))

result = model.predict_result(X[:3], coords[:3])
assert isinstance(result, GWRPredictionResult)
print(result.to_frame())
