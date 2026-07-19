# Quick start: from data to interpreted GWR

This walkthrough creates a reproducible spatial dataset, fits a standard GWR, inspects global and local diagnostics, predicts at target coordinates, and exports a result table.

## Load a bundled dataset and fit GWR

```python
from pygwrx import GWR
from pygwrx.io import load_columbus

data = load_columbus(return_type="dict")
model = GWR(kernel="bisquare", bandwidth=24, adaptive=True).fit(
    data["data"], data["target"], data["coords"]
)
print(model.summary())
print(data["license"], data["source_url"])
```


## 1. Prepare aligned data

Every row of `X`, `y`, and `coords` must refer to the same observation.

```python
import numpy as np
import pandas as pd

rng = np.random.default_rng(7)
n = 72
coords = pd.DataFrame(
    rng.uniform(0, 20, size=(n, 2)),
    columns=["east", "north"],
)
X = pd.DataFrame(
    rng.normal(size=(n, 2)),
    columns=["income", "access"],
)

# The income effect varies with easting; access has a stable negative effect.
local_income = 0.8 + 0.04 * coords["east"].to_numpy()
y = 3.0 + local_income * X["income"] - 0.6 * X["access"]
y += rng.normal(scale=0.4, size=n)
```

Before fitting real data, check:

```python
assert len(X) == len(y) == len(coords)
assert np.isfinite(X.to_numpy()).all()
assert np.isfinite(y).all()
assert np.isfinite(coords.to_numpy()).all()
```

## 2. Establish a global baseline

A simple global least-squares fit provides a reference for coefficient magnitude and residual behaviour.

```python
X_global = np.column_stack([np.ones(n), X.to_numpy()])
beta_global, *_ = np.linalg.lstsq(X_global, y, rcond=None)
y_global = X_global @ beta_global
print("global coefficients:", beta_global)
print("global RMSE:", np.sqrt(np.mean((y - y_global) ** 2)))
```

## 3. Fit GWR

```python
from pygwrx import GWR

model = GWR(
    kernel="bisquare",
    bandwidth=28,   # 28 nearest neighbours
    adaptive=True,
    fit_intercept=True,
)
model.fit(X, y, coords)
```

For a real study, do not choose 28 merely because it runs. Use a documented criterion and search range where appropriate:

```python
candidate = GWR(
    kernel="bisquare",
    bandwidth="aicc",
    bandwidth_range=(20, 50),
    adaptive=True,
)
```

## 4. Inspect fitted outputs

```python
print(model.summary())
print("selected bandwidth:", model.bandwidth_)
print("model diagnostics:", model.diagnostics_)

calibration = model.to_frame()
print(calibration.head())
print(calibration.filter(like="coef").describe())
```

The calibration table is location-indexed. Preserve a stable observation ID before joining it back to GIS data.

## 5. Run model-aware diagnostics

```python
from pygwrx.diagnostics import (
    diagnostics_frame,
    local_diagnostic_frame,
    parameter_inference,
    parameter_significance,
)

print(diagnostics_frame([model], labels=["GWR"]))
print(local_diagnostic_frame(model).head())
print(parameter_inference(model, feature="income").head())
print(
    parameter_significance(
        model,
        feature="income",
        alpha=0.05,
        correction="fdr_bh",
    ).head()
)
```

At minimum, inspect residuals, influence, coefficient uncertainty, local condition numbers, and the sensitivity of conclusions to the bandwidth.

## 6. Predict at target locations

```python
X_new = X.iloc[:4].copy()
coords_new = coords.iloc[:4].copy()

values = model.predict(X_new, coords_new)
result = model.predict_result(X_new, coords_new)

print(values)
print(result.to_frame())
```

`predict_result()` preserves target-local coefficients and metadata where the model supports them. This is different from copying or interpolating calibration coefficients.

## 7. Visualize

Matplotlib is included in the standard installation:

```python
from pygwrx.plotting import plot_coefficient_map, plot_diagnostic_panel

fig, ax = plot_coefficient_map(model, feature="income", theme="paper")
fig.savefig("income_coefficient.png", dpi=200, bbox_inches="tight")

fig, axes = plot_diagnostic_panel(model, theme="paper")
fig.savefig("gwr_diagnostics.png", dpi=200, bbox_inches="tight")
```

Plotting functions return Matplotlib objects and do not call `plt.show()` automatically.

## 8. Export

```python
from pygwrx.io import save_results

save_results(calibration, "gwr_calibration.csv")
save_results(result.to_frame(), "gwr_predictions.csv")
```

## 9. Interpretation checklist

- Is the local coefficient variation larger than its uncertainty?
- Are high or sign-changing coefficients concentrated where local condition numbers are poor?
- Does the local model reduce residual structure relative to the global baseline?
- Are influential observations driving a surface?
- Does the result survive a plausible bandwidth range and a spatial validation split?

## Continue

- [Detailed GWR handbook](../models/gwr.md)
- [Kernels and bandwidths](../guides/kernels-and-bandwidths.md)
- [Diagnostics and inference](../guides/diagnostics.md)
- [End-to-end maintained workflow](../tutorials/end-to-end-gwr.md)
- [GWR API](../api/models/gwr.md)
