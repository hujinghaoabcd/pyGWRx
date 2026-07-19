# Diagnostics and inference

Local spatial models require more than a coefficient map. The same observation contributes to many overlapping local fits, bandwidth choice changes the effective sample size, and local design matrices can become ill-conditioned even when the global design appears acceptable.

## Recommended diagnostic sequence

1. **Global fit and complexity:** R²/deviance, AICc, ENP/EDF, trace statistics.
2. **Residual behaviour:** fitted-versus-observed, standardized residuals, spatial residual map, QQ/histogram where appropriate.
3. **Influence:** leverage and Cook’s distance relative to model-aware thresholds.
4. **Parameter uncertainty:** standard errors, test statistics, adjusted p-values, and significance masks.
5. **Local collinearity:** condition number, VIF, and variance-decomposition information where available.
6. **Neighbourhoods:** effective counts, weight concentration, spatial/temporal/similarity components.
7. **Model-specific structure:** temporal trajectories, regimes, robust weights, sparsity, bootstrap distributions, or latent geometry.

!!! danger "Do not diagnose only the winning model"
    Compare candidate models using the same validation design and inspect whether added flexibility merely absorbs noise or leakage.

## Common entry points

```python
from pygwrx.diagnostics import (
    diagnostics_frame,
    local_diagnostic_frame,
    parameter_inference,
    parameter_significance,
)

comparison = diagnostics_frame([model_a, model_b], labels=["A", "B"])
local = local_diagnostic_frame(model_a)
inference = parameter_inference(model_a)
significant = parameter_significance(model_a, alpha=0.05, correction="fdr_bh")
```

## Model and residual diagnostics

```python
# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Extract global and local diagnostics from fitted models."""

from pygwrx import GWR
from pygwrx.diagnostics import (
    DiagnosticSummary,
    InfluenceThresholds,
    diagnostics_frame,
    influence_thresholds,
    local_diagnostic_frame,
    model_diagnostic_summary,
)
from _common import spatial_regression

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

## Inference and local collinearity

```python
# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Use coefficient inference, multiple-testing correction, and collinearity tools."""

import numpy as np
from pygwrx import GWR
from pygwrx.diagnostics import (
    LocalCollinearityDiagnostics,
    ParameterInference,
    adjust_pvalues,
    feature_names,
    parameter_inference,
    parameter_significance,
)
from _common import collinear_regression

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

## Temporal diagnostics

```python
# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Group time values and summarize temporal coefficient trajectories."""

from pygwrx import GTWR
from pygwrx.diagnostics import (
    TemporalGroups,
    model_times,
    parameter_trajectory,
    temporal_groups,
    temporal_parameter_frame,
)
from _common import temporal_regression

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

## Weight diagnostics

```python
# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Inspect spatial, similarity, temporal, and combined weight components."""

from pygwrx import SGWR, SGTWR
from pygwrx.diagnostics import (
    WeightComponents,
    focus_weight_components,
    weight_components,
)
from _common import spatial_regression, temporal_regression

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

## Regime diagnostics

```python
# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Export observation, regime, and boundary summaries for GR-GWR."""

from pygwrx import GRGWR
from pygwrx.diagnostics import boundary_frame, regime_frame, regime_summary
from _common import regime_regression

X, y, coords, _ = regime_regression(n=54)
model = GRGWR(n_regimes=2, bandwidth=18, max_iter=2, random_state=0).fit(X, y, coords)
print(regime_frame(model).head())
print(regime_summary(model))
print(boundary_frame(model).head())
```

## Interpretation rules

- Correct p-values when many local hypotheses are examined.
- A high local condition number can make sign and magnitude unstable.
- High leverage is not automatically an error; investigate why the location is influential.
- Residual spatial structure indicates omitted process, misspecified neighbourhoods, or both.
- For non-Gaussian and classification models, use family-appropriate residuals and validation metrics.

See the [Diagnostics API](../api/diagnostics/index.md).
