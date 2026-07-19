# Temporal diagnostics

This page documents **5** public symbols. Each entry includes its purpose, import path, full API docstring, and the maintained example that exercises it.

[Conceptual guide](../../guides/diagnostics.md){ .md-button }

## `TemporalGroups`

Unique time values and row indices for a fitted spatiotemporal model.

| Property | Value |
|---|---|
| Type | `class` |
| Import | `from pygwrx.diagnostics import TemporalGroups` |
| Signature | `TemporalGroups(values: 'np.ndarray', indices: 'Tuple[np.ndarray, ...]') -> None` |
| Maintained example | [`examples/diagnostics/03_temporal_diagnostics.py`](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/diagnostics/03_temporal_diagnostics.py) |

::: pygwrx.diagnostics.TemporalGroups


## `model_times`

Return one time value per plotted row.

| Property | Value |
|---|---|
| Type | `function` |
| Import | `from pygwrx.diagnostics import model_times` |
| Signature | `model_times(model: 'Any') -> 'np.ndarray'` |
| Maintained example | [`examples/diagnostics/03_temporal_diagnostics.py`](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/diagnostics/03_temporal_diagnostics.py) |

::: pygwrx.diagnostics.model_times


## `parameter_trajectory`

Aggregate a parameter surface over time or follow the nearest location.

| Property | Value |
|---|---|
| Type | `function` |
| Import | `from pygwrx.diagnostics import parameter_trajectory` |
| Signature | `parameter_trajectory(model: 'Any', feature: 'FeatureLike', *, location: 'Optional[Union[int, Sequence[float]]]' = None, reducer: 'str' = 'mean') -> 'pd.DataFrame'` |
| Maintained example | [`examples/diagnostics/03_temporal_diagnostics.py`](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/diagnostics/03_temporal_diagnostics.py) |

::: pygwrx.diagnostics.parameter_trajectory


## `temporal_groups`

Group fitted rows by exact time value while preserving chronological order.

| Property | Value |
|---|---|
| Type | `function` |
| Import | `from pygwrx.diagnostics import temporal_groups` |
| Signature | `temporal_groups(model: 'Any') -> 'TemporalGroups'` |
| Maintained example | [`examples/diagnostics/03_temporal_diagnostics.py`](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/diagnostics/03_temporal_diagnostics.py) |

::: pygwrx.diagnostics.temporal_groups


## `temporal_parameter_frame`

Return local parameters with coordinates and times in tidy form.

| Property | Value |
|---|---|
| Type | `function` |
| Import | `from pygwrx.diagnostics import temporal_parameter_frame` |
| Signature | `temporal_parameter_frame(model: 'Any', feature: 'FeatureLike') -> 'pd.DataFrame'` |
| Maintained example | [`examples/diagnostics/03_temporal_diagnostics.py`](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/diagnostics/03_temporal_diagnostics.py) |

::: pygwrx.diagnostics.temporal_parameter_frame


## Runnable examples used on this page

??? example "`examples/diagnostics/03_temporal_diagnostics.py`"

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
