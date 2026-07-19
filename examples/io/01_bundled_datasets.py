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
