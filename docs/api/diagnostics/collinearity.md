# Local collinearity

This page documents **1** public symbols. Each entry includes its purpose, import path, full API docstring, and the maintained example that exercises it.

[Conceptual guide](../../guides/diagnostics.md){ .md-button }

## `LocalCollinearityDiagnostics`

Diagnose spatially varying multicollinearity in a fitted GWR model.

| Property | Value |
|---|---|
| Type | `class` |
| Import | `from pygwrx.diagnostics import LocalCollinearityDiagnostics` |
| Signature | `LocalCollinearityDiagnostics(gwr_model: Any, tolerance: float = 1e-10) -> None` |
| Maintained example | [`examples/diagnostics/02_inference_and_collinearity.py`](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/diagnostics/02_inference_and_collinearity.py) |

::: pygwrx.diagnostics.LocalCollinearityDiagnostics


## Runnable examples used on this page

??? example "`examples/diagnostics/02_inference_and_collinearity.py`"

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
