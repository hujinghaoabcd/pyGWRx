# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT
"""Compare GTWR and SGTWR on one synthetic space-time dataset."""

# Allow this script to run directly from any working directory.
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_EXAMPLES_ROOT = _PROJECT_ROOT / "examples"
_SRC_ROOT = _PROJECT_ROOT / "src"
for _path in (_SRC_ROOT, _EXAMPLES_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from _common import temporal_regression

from pygwrx import GTWR, SGTWR
from pygwrx.diagnostics import parameter_trajectory, temporal_groups

X, y, coords, times = temporal_regression(n=48, p=2)
gtwr = GTWR(bandwidth=24, adaptive=True, lambda_st=0.3).fit(X, y, coords, times)
sgtwr = SGTWR(
    spatial_bandwidth=24,
    temporal_bandwidth=2.0,
    adaptive=True,
    similarity_vars=["x1"],
).fit(X, y, coords, times)
for model in (gtwr, sgtwr):
    groups = temporal_groups(model)
    print(type(model).__name__, groups.values, [len(index) for index in groups.indices])
    print(parameter_trajectory(model, feature=0).head())
