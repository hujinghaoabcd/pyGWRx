# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Fit latent-geometry GWR with auxiliary contextual attributes."""

# Allow this script to run directly from any working directory.
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_EXAMPLES_ROOT = _PROJECT_ROOT / "examples"
_SRC_ROOT = _PROJECT_ROOT / "src"
for _path in (_SRC_ROOT, _EXAMPLES_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from _common import latent_regression, print_model_result

from pygwrx import LGGWR, LGGWRPredictionResult

X, y, coords, attributes = latent_regression()
model = LGGWR(
    latent_dim=2, bandwidth=2.5, select_bandwidth=False, max_iter=8, random_state=0
).fit(X, y, coords, attributes)
print_model_result(model)
print("latent_coordinates_shape=", model.latent_coords_.shape)
result = model.predict_result(X.iloc[:3], coords.iloc[:3], attributes.iloc[:3])
assert isinstance(result, LGGWRPredictionResult)
print(result.to_frame())
