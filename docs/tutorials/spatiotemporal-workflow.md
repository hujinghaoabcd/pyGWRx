# Spatiotemporal workflow

This workflow compares GTWR and SGTWR on one row-wise space-time dataset. GTWR uses geographic and temporal proximity; SGTWR additionally uses attribute similarity.

## Maintained workflow

```python
# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT
"""Compare GTWR and SGTWR on one synthetic space-time dataset."""

from pygwrx import GTWR, SGTWR
from pygwrx.diagnostics import parameter_trajectory, temporal_groups

from _common import temporal_regression

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
```

## Required validation upgrade

The compact example demonstrates the API, not a complete forecasting benchmark. For a scientific application:

1. sort or group observations by explicit time;
2. train on earlier periods;
3. predict later periods;
4. set causal weighting when the model supports it;
5. ensure similarity variables are known at prediction time;
6. compare against spatial-only GWR and a global temporal baseline.

## Diagnostics

`temporal_groups()` verifies the time groups available in the fitted model. `parameter_trajectory()` summarizes coefficient evolution. Also inspect temporal residuals, selected time scales, boundary solutions, and performance by horizon.
