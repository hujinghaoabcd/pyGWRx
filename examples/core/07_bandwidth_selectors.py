# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Select bandwidths with CV, AIC/AICc, and BIC selectors."""

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
from _common import spatial_regression

from pygwrx.core import (
    AICSelector,
    BandwidthSelector,
    BICSelector,
    CrossValidationSelector,
    gaussian_kernel,
    get_bandwidth_selector,
)

X, y, coords = spatial_regression(n=28, p=2)
Xa, ya, ca = X.to_numpy(), np.asarray(y), coords.to_numpy()
selectors = [
    CrossValidationSelector(n_intervals=5, adaptive=True, verbose=False),
    AICSelector(n_intervals=5, corrected=False, adaptive=True, verbose=False),
    AICSelector(n_intervals=5, corrected=True, adaptive=True, verbose=False),
    BICSelector(n_intervals=5, adaptive=True, verbose=False),
]
for selector in selectors:
    selected = selector.select(
        Xa,
        ya,
        ca,
        gaussian_kernel,
        bandwidth_range=(10, 18),
    )
    print(
        type(selector).__name__,
        selected,
        "evaluated=",
        len(selector.search_trace_),
    )
print("factory=", type(get_bandwidth_selector("aicc", adaptive=True)).__name__)
print("abstract_base=", BandwidthSelector)
