# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Group time values and summarize temporal coefficient trajectories."""

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

from pygwrx import GTWR
from pygwrx.diagnostics import (
    TemporalGroups,
    model_times,
    parameter_trajectory,
    temporal_groups,
    temporal_parameter_frame,
)

X, y, coords, times = temporal_regression(n=48, p=2)
model = GTWR(bandwidth=24, adaptive=True, lambda_st=0.3).fit(X, y, coords, times)
groups = temporal_groups(model)
assert isinstance(groups, TemporalGroups)
print("times=", model_times(model)[:8])
print("group_values=", groups.values)
print(temporal_parameter_frame(model, "x1").head())
print(parameter_trajectory(model, "x1", reducer="mean"))
print(parameter_trajectory(model, "x1", location=3))
