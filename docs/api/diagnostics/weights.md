# Weight diagnostics

This page documents **3** public symbols. Each entry includes its purpose, import path, full API docstring, and the maintained example that exercises it.

[Conceptual guide](../../guides/diagnostics.md){ .md-button }

## `WeightComponents`

Named weight matrices exposed by a fitted model.

| Property | Value |
|---|---|
| Type | `class` |
| Import | `from pygwrx.diagnostics import WeightComponents` |
| Signature | `WeightComponents(components: 'Mapping[str, np.ndarray]', combined_name: 'Optional[str]') -> None` |
| Maintained example | [`examples/diagnostics/04_weight_diagnostics.py`](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/diagnostics/04_weight_diagnostics.py) |

::: pygwrx.diagnostics.WeightComponents


## `focus_weight_components`

Return one row from every stored weight component.

| Property | Value |
|---|---|
| Type | `function` |
| Import | `from pygwrx.diagnostics import focus_weight_components` |
| Signature | `focus_weight_components(model: 'Any', focus: 'int') -> 'Dict[str, np.ndarray]'` |
| Maintained example | [`examples/diagnostics/04_weight_diagnostics.py`](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/diagnostics/04_weight_diagnostics.py) |

::: pygwrx.diagnostics.focus_weight_components


## `weight_components`

Collect stored weight matrices using stable semantic names.

| Property | Value |
|---|---|
| Type | `function` |
| Import | `from pygwrx.diagnostics import weight_components` |
| Signature | `weight_components(model: 'Any') -> 'WeightComponents'` |
| Maintained example | [`examples/diagnostics/04_weight_diagnostics.py`](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/diagnostics/04_weight_diagnostics.py) |

::: pygwrx.diagnostics.weight_components


## Runnable examples used on this page

??? example "`examples/diagnostics/04_weight_diagnostics.py`"

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
