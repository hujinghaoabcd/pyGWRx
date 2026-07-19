# Residuals and influence

This page documents **3** public symbols. Each entry includes its purpose, import path, full API docstring, and the maintained example that exercises it.

[Conceptual guide](../../guides/diagnostics.md){ .md-button }

## `InfluenceThresholds`

Common reference thresholds for local influence diagnostics.

| Property | Value |
|---|---|
| Type | `class` |
| Import | `from pygwrx.diagnostics import InfluenceThresholds` |
| Signature | `InfluenceThresholds(leverage: 'Optional[float]', cooks_distance: 'Optional[float]', standardized_residual: 'float' = 2.0) -> None` |
| Maintained example | [`examples/diagnostics/01_model_and_residual_diagnostics.py`](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/diagnostics/01_model_and_residual_diagnostics.py) |

::: pygwrx.diagnostics.InfluenceThresholds


## `influence_thresholds`

Return transparent rule-of-thumb thresholds for a fitted model.

| Property | Value |
|---|---|
| Type | `function` |
| Import | `from pygwrx.diagnostics import influence_thresholds` |
| Signature | `influence_thresholds(model: 'Any') -> 'InfluenceThresholds'` |
| Maintained example | [`examples/diagnostics/01_model_and_residual_diagnostics.py`](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/diagnostics/01_model_and_residual_diagnostics.py) |

::: pygwrx.diagnostics.influence_thresholds


## `local_diagnostic_frame`

Collect available row-wise diagnostics without mutating the model.

| Property | Value |
|---|---|
| Type | `function` |
| Import | `from pygwrx.diagnostics import local_diagnostic_frame` |
| Signature | `local_diagnostic_frame(model: 'Any') -> 'pd.DataFrame'` |
| Maintained example | [`examples/diagnostics/01_model_and_residual_diagnostics.py`](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/diagnostics/01_model_and_residual_diagnostics.py) |

::: pygwrx.diagnostics.local_diagnostic_frame


## Runnable examples used on this page

??? example "`examples/diagnostics/01_model_and_residual_diagnostics.py`"

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
