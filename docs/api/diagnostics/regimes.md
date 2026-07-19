# Regime diagnostics

This page documents **3** public symbols. Each entry includes its purpose, import path, full API docstring, and the maintained example that exercises it.

[Conceptual guide](../../guides/diagnostics.md){ .md-button }

## `boundary_frame`

Return unique regime-boundary edges and their endpoints.

| Property | Value |
|---|---|
| Type | `function` |
| Import | `from pygwrx.diagnostics import boundary_frame` |
| Signature | `boundary_frame(model: 'Any') -> 'pd.DataFrame'` |
| Maintained example | [`examples/diagnostics/05_regime_diagnostics.py`](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/diagnostics/05_regime_diagnostics.py) |

::: pygwrx.diagnostics.boundary_frame


## `regime_frame`

Return coordinates, regime labels, residuals, and connectivity metadata.

| Property | Value |
|---|---|
| Type | `function` |
| Import | `from pygwrx.diagnostics import regime_frame` |
| Signature | `regime_frame(model: 'Any') -> 'pd.DataFrame'` |
| Maintained example | [`examples/diagnostics/05_regime_diagnostics.py`](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/diagnostics/05_regime_diagnostics.py) |

::: pygwrx.diagnostics.regime_frame


## `regime_summary`

Summarize regime sizes, residual error, and component counts.

| Property | Value |
|---|---|
| Type | `function` |
| Import | `from pygwrx.diagnostics import regime_summary` |
| Signature | `regime_summary(model: 'Any') -> 'pd.DataFrame'` |
| Maintained example | [`examples/diagnostics/05_regime_diagnostics.py`](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/diagnostics/05_regime_diagnostics.py) |

::: pygwrx.diagnostics.regime_summary


## Runnable examples used on this page

??? example "`examples/diagnostics/05_regime_diagnostics.py`"

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
