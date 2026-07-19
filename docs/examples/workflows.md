# Workflow examples

Multi-step analyses that combine data preparation, fitting, diagnostics, comparison, prediction, and export.

This page embeds **3** maintained scripts. The code shown here is read directly from `examples/workflows/`, so the documentation and executable source cannot silently diverge.

!!! tip "How to use this catalog"
    Read the purpose and APIs first, run the exact command, then use the inspection note to decide which output matters. For conceptual interpretation, follow the linked model or function guide rather than reading code alone.

## `01_end_to_end_gwr.py`

**Purpose.** Run a real-data GWR workflow from bundled data to geospatial output.

**Public APIs exercised.** `GWR`, `diagnostics_frame`, `parameter_inference`, `load_georgia`, `save_results`

**Environment.** base installation.

**Run.** `python examples/workflows/01_end_to_end_gwr.py`

**What to inspect.** Follow the order of operations and note where validation, interpretation, and capability checks occur before export.

[Open source](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/workflows/01_end_to_end_gwr.py){ .md-button }

```python
# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Run a real-data GWR workflow from bundled data to geospatial output."""

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

from _common import OUTPUT_DIR

from pygwrx import GWR
from pygwrx.diagnostics import diagnostics_frame, parameter_inference
from pygwrx.io import load_georgia, save_results

bundle = load_georgia(return_type="dict")
X = bundle["frame"][bundle["feature_names"]]
y = bundle["frame"][bundle["target_name"]]
coords = bundle["coords"]

model = GWR(kernel="bisquare", bandwidth=48, adaptive=True).fit(X, y, coords)
print(model.summary())
print(diagnostics_frame([model], labels=["GWR"]))
print(parameter_inference(model, bundle["feature_names"][0]))
print(model.predict_result(X[:5], coords[:5]).to_frame())

result_frame = model.to_frame()
save_results(result_frame, OUTPUT_DIR / "georgia_gwr_results.csv")

geo = bundle["frame"].copy()
for column in result_frame.columns:
    if column not in geo.columns and len(result_frame[column]) == len(geo):
        geo[column] = result_frame[column].to_numpy()
save_results(geo, OUTPUT_DIR / "georgia_gwr_results.geojson")
```

## `02_model_comparison.py`

**Purpose.** Compare standard, robust, and ridge-compensated local regressions.

**Public APIs exercised.** `GWR`, `LCRGWR`, `RGWR`, `model_diagnostic_summary`

**Environment.** base installation.

**Run.** `python examples/workflows/02_model_comparison.py`

**What to inspect.** Follow the order of operations and note where validation, interpretation, and capability checks occur before export.

[Open source](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/workflows/02_model_comparison.py){ .md-button }

```python
# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT
"""Compare standard, robust, and ridge-compensated local regressions."""

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

from pygwrx import GWR, LCRGWR, RGWR
from pygwrx.diagnostics import model_diagnostic_summary

X, y, coords = spatial_regression()
models = {
    "GWR": GWR(bandwidth=24, adaptive=True),
    "RGWR": RGWR(bandwidth=24, adaptive=True, max_iter=5),
    "LCRGWR": LCRGWR(bandwidth=24, adaptive=True),
}
for name, model in models.items():
    model.fit(X, y, coords)
    summary = model_diagnostic_summary(model)
    print(name, summary)
```

## `03_spatiotemporal_workflow.py`

**Purpose.** Compare GTWR and SGTWR on one synthetic space-time dataset.

**Public APIs exercised.** `GTWR`, `SGTWR`, `parameter_trajectory`, `temporal_groups`

**Environment.** base installation.

**Run.** `python examples/workflows/03_spatiotemporal_workflow.py`

**What to inspect.** Follow the order of operations and note where validation, interpretation, and capability checks occur before export.

[Open source](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/workflows/03_spatiotemporal_workflow.py){ .md-button }

```python
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
```
