# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Run coefficient-wise bootstrap tests for spatial variability."""

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

from pygwrx import BootstrapGWR

X, y, coords = spatial_regression(n=42, p=2)
model = BootstrapGWR(
    bandwidth=22,
    adaptive=True,
    n_bootstrap=9,
    reselect_bandwidth=False,
    store_local_bootstrap=True,
    random_state=0,
).fit(X, y, coords)
print_model_result(model)
print("modified_pvalues=", model.modified_p_values_)
print("localized_p_values_shape=", model.localized_p_values_.shape)
