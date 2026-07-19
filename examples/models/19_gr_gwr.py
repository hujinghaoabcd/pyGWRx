# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Fit geo-regime GWR and inspect connected spatial regimes."""

# Allow this script to run directly from any working directory.
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_EXAMPLES_ROOT = _PROJECT_ROOT / "examples"
_SRC_ROOT = _PROJECT_ROOT / "src"
for _path in (_SRC_ROOT, _EXAMPLES_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from _common import print_model_result, regime_regression

from pygwrx import GRGWR, GRGWRPredictionResult

X, y, coords, truth = regime_regression(n=56)
model = GRGWR(n_regimes=2, bandwidth=18, max_iter=2, random_state=0).fit(X, y, coords)
print_model_result(model)
print("regime_sizes=", model.regime_sizes_)
print(
    "truth_agreement_or_label_swap=",
    max((model.regimes_ == truth).mean(), (model.regimes_ != truth).mean()),
)
result = model.predict_result(X.iloc[:3], coords.iloc[:3])
assert isinstance(result, GRGWRPredictionResult)
print(result.to_frame())
