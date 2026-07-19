# Diagnostics examples

Global summaries, local inference, collinearity, influence, residual, temporal, weight, and regime diagnostics.

This page embeds **5** maintained scripts. The code shown here is read directly from `examples/diagnostics/`, so the documentation and executable source cannot silently diverge.

!!! tip "How to use this catalog"
    Read the purpose and APIs first, run the exact command, then use the inspection note to decide which output matters. For conceptual interpretation, follow the linked model or function guide rather than reading code alone.

## `01_model_and_residual_diagnostics.py`

**Purpose.** Extract global and local diagnostics from fitted models.

**Public APIs exercised.** `GWR`, `DiagnosticSummary`, `InfluenceThresholds`, `diagnostics_frame`, `influence_thresholds`, `local_diagnostic_frame`, `model_diagnostic_summary`

**Environment.** base installation.

**Run.** `python examples/diagnostics/01_model_and_residual_diagnostics.py`

**What to inspect.** Check that the reported statistic matches the fitted model contract and interpret thresholds together with spatial context.

[Open source](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/diagnostics/01_model_and_residual_diagnostics.py){ .md-button }

```python
# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Extract global and local diagnostics from fitted models."""

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

from pygwrx import GWR
from pygwrx.diagnostics import (
    DiagnosticSummary,
    InfluenceThresholds,
    diagnostics_frame,
    influence_thresholds,
    local_diagnostic_frame,
    model_diagnostic_summary,
)

X, y, coords = spatial_regression(n=42, p=2)
first = GWR(bandwidth=22, adaptive=True, kernel="bisquare").fit(X, y, coords)
second = GWR(bandwidth=24, adaptive=True, kernel="gaussian").fit(X, y, coords)
summary = model_diagnostic_summary(first)
thresholds = influence_thresholds(first)
assert isinstance(summary, DiagnosticSummary)
assert isinstance(thresholds, InfluenceThresholds)
print(summary.to_series())
print(thresholds)
print(local_diagnostic_frame(first).head())
print(diagnostics_frame([first, second], labels=["bisquare", "gaussian"]))
```

## `02_inference_and_collinearity.py`

**Purpose.** Use coefficient inference, multiple-testing correction, and collinearity tools.

**Public APIs exercised.** `GWR`, `LocalCollinearityDiagnostics`, `ParameterInference`, `adjust_pvalues`, `feature_names`, `parameter_inference`, `parameter_significance`

**Environment.** base installation.

**Run.** `python examples/diagnostics/02_inference_and_collinearity.py`

**What to inspect.** Check that the reported statistic matches the fitted model contract and interpret thresholds together with spatial context.

[Open source](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/diagnostics/02_inference_and_collinearity.py){ .md-button }

```python
# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Use coefficient inference, multiple-testing correction, and collinearity tools."""

# Allow this script to run directly from any working directory.
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_EXAMPLES_ROOT = _PROJECT_ROOT / "examples"
_SRC_ROOT = _PROJECT_ROOT / "src"
for _path in (_SRC_ROOT, _EXAMPLES_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import numpy as np
from _common import collinear_regression

from pygwrx import GWR
from pygwrx.diagnostics import (
    LocalCollinearityDiagnostics,
    ParameterInference,
    adjust_pvalues,
    feature_names,
    parameter_inference,
    parameter_significance,
)

X, y, coords = collinear_regression(n=44)
model = GWR(bandwidth=24, adaptive=True).fit(X, y, coords)
view = parameter_inference(model, "x1")
assert isinstance(view, ParameterInference)
print("feature_names=", feature_names(model))
print("inference_label=", view.label)
print(parameter_significance(model, "x1", correction="bh").head())
print("adjusted=", adjust_pvalues(np.array([0.01, 0.04, 0.2, 0.8]), method="bh"))
collinearity = LocalCollinearityDiagnostics(model)
print(collinearity.summary_frame().head())
print("vif_shape=", collinearity.compute_vif().shape)
print("vdp_shape=", collinearity.compute_vdp().shape)
print("correlation_shape=", collinearity.compute_local_correlations().shape)
print("condition_numbers=", collinearity.compute_condition_number()[:5])
print("diagnosis_keys=", sorted(collinearity.diagnose(verbose=False)))
```

## `03_temporal_diagnostics.py`

**Purpose.** Group time values and summarize temporal coefficient trajectories.

**Public APIs exercised.** `GTWR`, `TemporalGroups`, `model_times`, `parameter_trajectory`, `temporal_groups`, `temporal_parameter_frame`

**Environment.** base installation.

**Run.** `python examples/diagnostics/03_temporal_diagnostics.py`

**What to inspect.** Check that the reported statistic matches the fitted model contract and interpret thresholds together with spatial context.

[Open source](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/diagnostics/03_temporal_diagnostics.py){ .md-button }

```python
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
```

## `04_weight_diagnostics.py`

**Purpose.** Inspect spatial, similarity, temporal, and combined weight components.

**Public APIs exercised.** `SGTWR`, `SGWR`, `WeightComponents`, `focus_weight_components`, `weight_components`

**Environment.** base installation.

**Run.** `python examples/diagnostics/04_weight_diagnostics.py`

**What to inspect.** Check that the reported statistic matches the fitted model contract and interpret thresholds together with spatial context.

[Open source](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/diagnostics/04_weight_diagnostics.py){ .md-button }

```python
# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Inspect spatial, similarity, temporal, and combined weight components."""

# Allow this script to run directly from any working directory.
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_EXAMPLES_ROOT = _PROJECT_ROOT / "examples"
_SRC_ROOT = _PROJECT_ROOT / "src"
for _path in (_SRC_ROOT, _EXAMPLES_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from _common import spatial_regression, temporal_regression

from pygwrx import SGTWR, SGWR
from pygwrx.diagnostics import (
    WeightComponents,
    focus_weight_components,
    weight_components,
)

X, y, coords = spatial_regression(n=40, p=2)
sgwr = SGWR(bandwidth=20, adaptive=True, alpha=0.45, store_weights=True).fit(
    X, y, coords
)
components = weight_components(sgwr)
assert isinstance(components, WeightComponents)
print("sgwr_components=", sorted(components.components))
print("sgwr_focus=", {k: v[:5] for k, v in focus_weight_components(sgwr, 3).items()})

Xt, yt, coordst, times = temporal_regression(n=40, p=2)
sgtwr = SGTWR(
    spatial_bandwidth=20,
    temporal_bandwidth=2.0,
    adaptive=True,
    alpha=0.5,
    store_weights=True,
).fit(Xt, yt, coordst, times)
print("sgtwr_components=", sorted(weight_components(sgtwr).components))
```

## `05_regime_diagnostics.py`

**Purpose.** Export observation, regime, and boundary summaries for GR-GWR.

**Public APIs exercised.** `GRGWR`, `boundary_frame`, `regime_frame`, `regime_summary`

**Environment.** `pip install -e ".[ml]"`.

**Run.** `python examples/diagnostics/05_regime_diagnostics.py`

**What to inspect.** Check that the reported statistic matches the fitted model contract and interpret thresholds together with spatial context.

[Open source](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/diagnostics/05_regime_diagnostics.py){ .md-button }

```python
# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Export observation, regime, and boundary summaries for GR-GWR."""

# Allow this script to run directly from any working directory.
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_EXAMPLES_ROOT = _PROJECT_ROOT / "examples"
_SRC_ROOT = _PROJECT_ROOT / "src"
for _path in (_SRC_ROOT, _EXAMPLES_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from _common import regime_regression

from pygwrx import GRGWR
from pygwrx.diagnostics import boundary_frame, regime_frame, regime_summary

X, y, coords, _ = regime_regression(n=54)
model = GRGWR(n_regimes=2, bandwidth=18, max_iter=2, random_state=0).fit(X, y, coords)
print(regime_frame(model).head())
print(regime_summary(model))
print(boundary_frame(model).head())
```
