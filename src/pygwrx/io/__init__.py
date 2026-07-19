# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Public data input and output interface for pyGWRx.

This module exposes built-in dataset loaders, user-data conversion helpers, and result-saving utilities.

Author:
    Jinghao Hu
"""

__author__ = "Jinghao Hu"
__license__ = "MIT"

from pygwrx.io.data import (
    from_geodataframe,
    load_data,
    save_results,
    to_geodataframe,
)
from pygwrx.io.datasets import (
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

__all__ = [
    "load_data",
    "to_geodataframe",
    "from_geodataframe",
    "save_results",
    "load_dataset",
    "load_dublin_voter",
    "load_hiv",
    "load_crime",
    "load_housing",
    "load_columbus",
    "load_ewhp",
    "load_georgia",
    "get_dublin_voter",
    "load_dubvoter",
    "get_dubvoter",
    "get_dataset_info",
    "list_datasets",
]
