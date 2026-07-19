# End-to-end GWR workflow

This tutorial follows a defensible sequence: establish aligned data, fit a GWR baseline, inspect model and local diagnostics, evaluate parameter inference, predict at target coordinates, and save an auditable result table.

## Maintained workflow

```python
# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT
"""Run an end-to-end GWR fit, diagnostics, prediction, and export workflow."""

from pygwrx import GWR
from pygwrx.diagnostics import diagnostics_frame, parameter_inference
from pygwrx.io import save_results

from _common import OUTPUT_DIR, spatial_regression

X, y, coords = spatial_regression()
model = GWR(kernel="bisquare", bandwidth=24, adaptive=True).fit(X, y, coords)
print(diagnostics_frame([model], labels=["GWR"]))
print(parameter_inference(model, "x1"))
print(model.predict_result(X.iloc[:5], coords.iloc[:5]).to_frame())
save_results(model.to_frame(), OUTPUT_DIR / "workflow_gwr_results.csv")
```

## Why each step matters

1. **Stable data contract:** predictors, response, and coordinates must share row identity.
2. **Explicit neighbourhood:** the example uses 24 adaptive neighbours with a bisquare kernel.
3. **Model comparison table:** `diagnostics_frame()` makes complexity and fit comparable across candidate models.
4. **Parameter inference:** inspect local uncertainty rather than mapping coefficients alone.
5. **Typed prediction:** `predict_result()` preserves target-local information.
6. **Persistence:** save a table that can be joined to GIS data through a stable ID.

## Extend the workflow

- add a global least-squares baseline;
- select bandwidth inside a training fold;
- use spatial blocks for validation;
- calculate local collinearity and influence;
- compare coefficient surfaces with significance masks;
- export vector or GeoPackage results after a verified join.

## Report

Record the coordinate system, kernel, fixed/adaptive setting, bandwidth, selection method, ENP, fit diagnostics, local inference method, validation design, and any residual spatial structure.
