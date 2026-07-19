# Dataset registry

This page documents **13** public symbols. Each entry includes its purpose, import path, full API docstring, and the maintained example that exercises it.

[Conceptual guide](../../guides/geospatial-io.md){ .md-button }

## `load_dataset`

Load a bundled example dataset by name.

| Property | Value |
|---|---|
| Type | `function` |
| Import | `from pygwrx.io import load_dataset` |
| Signature | `load_dataset(name: str, return_type: str = 'frame', data_dir: Union[str, os.PathLike[str], NoneType] = None, dropna: bool = True) -> Any` |
| Maintained example | [`examples/io/01_bundled_datasets.py`](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/io/01_bundled_datasets.py) |

::: pygwrx.io.load_dataset


## `load_dublin_voter`

Load the Dublin voter turnout dataset. See :func:`load_dataset`.

| Property | Value |
|---|---|
| Type | `function` |
| Import | `from pygwrx.io import load_dublin_voter` |
| Signature | `load_dublin_voter(return_type: str = 'frame', **kwargs: Any) -> Any` |
| Maintained example | [`examples/io/01_bundled_datasets.py`](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/io/01_bundled_datasets.py) |

::: pygwrx.io.load_dublin_voter


## `load_hiv`

Load the county-level HIV prevalence dataset. See :func:`load_dataset`.

| Property | Value |
|---|---|
| Type | `function` |
| Import | `from pygwrx.io import load_hiv` |
| Signature | `load_hiv(return_type: str = 'frame', **kwargs: Any) -> Any` |
| Maintained example | [`examples/io/01_bundled_datasets.py`](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/io/01_bundled_datasets.py) |

::: pygwrx.io.load_hiv


## `load_crime`

Load the county-level crime dataset. See :func:`load_dataset`.

| Property | Value |
|---|---|
| Type | `function` |
| Import | `from pygwrx.io import load_crime` |
| Signature | `load_crime(return_type: str = 'frame', **kwargs: Any) -> Any` |
| Maintained example | [`examples/io/01_bundled_datasets.py`](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/io/01_bundled_datasets.py) |

::: pygwrx.io.load_crime


## `load_housing`

Load the neighborhood house-price dataset. See :func:`load_dataset`.

| Property | Value |
|---|---|
| Type | `function` |
| Import | `from pygwrx.io import load_housing` |
| Signature | `load_housing(return_type: str = 'frame', **kwargs: Any) -> Any` |
| Maintained example | [`examples/io/01_bundled_datasets.py`](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/io/01_bundled_datasets.py) |

::: pygwrx.io.load_housing


## `load_columbus`

Load the Columbus (OH) crime dataset. See :func:`load_dataset`.

| Property | Value |
|---|---|
| Type | `function` |
| Import | `from pygwrx.io import load_columbus` |
| Signature | `load_columbus(return_type: str = 'frame', **kwargs: Any) -> Any` |
| Maintained example | [`examples/io/01_bundled_datasets.py`](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/io/01_bundled_datasets.py) |

::: pygwrx.io.load_columbus


## `load_ewhp`

Load the England & Wales house-price dataset. See :func:`load_dataset`.

| Property | Value |
|---|---|
| Type | `function` |
| Import | `from pygwrx.io import load_ewhp` |
| Signature | `load_ewhp(return_type: str = 'frame', **kwargs: Any) -> Any` |
| Maintained example | [`examples/io/01_bundled_datasets.py`](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/io/01_bundled_datasets.py) |

::: pygwrx.io.load_ewhp


## `load_georgia`

Load the Georgia educational-attainment dataset. See :func:`load_dataset`.

| Property | Value |
|---|---|
| Type | `function` |
| Import | `from pygwrx.io import load_georgia` |
| Signature | `load_georgia(return_type: str = 'frame', **kwargs: Any) -> Any` |
| Maintained example | [`examples/io/01_bundled_datasets.py`](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/io/01_bundled_datasets.py) |

::: pygwrx.io.load_georgia


## `get_dublin_voter`

Load the Dublin voter turnout dataset. See :func:`load_dataset`.

| Property | Value |
|---|---|
| Type | `function` |
| Import | `from pygwrx.io import get_dublin_voter` |
| Signature | `get_dublin_voter(return_type: str = 'frame', **kwargs: Any) -> Any` |
| Maintained example | [`examples/io/01_bundled_datasets.py`](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/io/01_bundled_datasets.py) |

::: pygwrx.io.get_dublin_voter


## `load_dubvoter`

Load the Dublin voter turnout dataset. See :func:`load_dataset`.

| Property | Value |
|---|---|
| Type | `function` |
| Import | `from pygwrx.io import load_dubvoter` |
| Signature | `load_dubvoter(return_type: str = 'frame', **kwargs: Any) -> Any` |
| Maintained example | [`examples/io/01_bundled_datasets.py`](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/io/01_bundled_datasets.py) |

::: pygwrx.io.load_dubvoter


## `get_dubvoter`

Load the Dublin voter turnout dataset. See :func:`load_dataset`.

| Property | Value |
|---|---|
| Type | `function` |
| Import | `from pygwrx.io import get_dubvoter` |
| Signature | `get_dubvoter(return_type: str = 'frame', **kwargs: Any) -> Any` |
| Maintained example | [`examples/io/01_bundled_datasets.py`](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/io/01_bundled_datasets.py) |

::: pygwrx.io.get_dubvoter


## `get_dataset_info`

Return registry metadata for a dataset without loading its data file.

| Property | Value |
|---|---|
| Type | `function` |
| Import | `from pygwrx.io import get_dataset_info` |
| Signature | `get_dataset_info(dataset_name: str = 'dublin_voter') -> Dict[str, Any]` |
| Maintained example | [`examples/io/01_bundled_datasets.py`](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/io/01_bundled_datasets.py) |

::: pygwrx.io.get_dataset_info


## `list_datasets`

List available built-in datasets and optionally print their metadata.

| Property | Value |
|---|---|
| Type | `function` |
| Import | `from pygwrx.io import list_datasets` |
| Signature | `list_datasets(verbose: bool = True) -> List[str]` |
| Maintained example | [`examples/io/01_bundled_datasets.py`](https://github.com/hujinghaoabcd/pyGWRx/blob/main/examples/io/01_bundled_datasets.py) |

::: pygwrx.io.list_datasets


## Runnable examples used on this page

??? example "`examples/io/01_bundled_datasets.py`"

    ```python
    # SPDX-FileCopyrightText: 2026 Jinghao Hu
    # SPDX-License-Identifier: MIT
    
    """List, describe, and load every bundled dataset and compatibility alias."""
    
    # Allow this script to run directly from any working directory.
    import sys
    from pathlib import Path
    
    _PROJECT_ROOT = Path(__file__).resolve().parents[2]
    _EXAMPLES_ROOT = _PROJECT_ROOT / "examples"
    _SRC_ROOT = _PROJECT_ROOT / "src"
    for _path in (_SRC_ROOT, _EXAMPLES_ROOT):
        if str(_path) not in sys.path:
            sys.path.insert(0, str(_path))
    
    from pygwrx.io import (
        get_dataset_info,
        get_dublin_voter,
        get_dubvoter,
        list_datasets,
        load_columbus,
        load_crime,
        load_dataset,
        load_dublin_voter,
        load_dubvoter,
        load_ewhp,
        load_georgia,
        load_hiv,
        load_housing,
    )
    
    loaders = {
        "dublin_voter": load_dublin_voter,
        "hiv": load_hiv,
        "crime": load_crime,
        "housing": load_housing,
        "columbus": load_columbus,
        "ewhp": load_ewhp,
        "georgia": load_georgia,
    }
    print("datasets=", list_datasets(verbose=False))
    for name, loader in loaders.items():
        info = get_dataset_info(name)
        frame = loader(return_type="frame")
        generic = load_dataset(name, return_type="frame")
        print(name, info["n_samples"], frame.shape, generic.shape)
    print(
        "alias_shapes=",
        [
            fn(return_type="frame").shape
            for fn in (get_dublin_voter, load_dubvoter, get_dubvoter)
        ],
    )
    ```
