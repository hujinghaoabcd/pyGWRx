# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Compute geographically weighted summary statistics."""

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

from pygwrx import GWSS

X, _, coords = spatial_regression(n=48, p=3)
model = GWSS(bandwidth=24, adaptive=True, quantile=True).fit(X, coords)
print(model.summary())
print("local_means_shape=", model.local_mean_.shape)
print("local_correlation_pairs=", sorted(model.local_corr_))
print("first_correlation_shape=", next(iter(model.local_corr_.values())).shape)
