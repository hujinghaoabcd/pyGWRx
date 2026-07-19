# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Use both public scalar optimizers and the OptimizationResult container."""

# Allow this script to run directly from any working directory.
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_EXAMPLES_ROOT = _PROJECT_ROOT / "examples"
_SRC_ROOT = _PROJECT_ROOT / "src"
for _path in (_SRC_ROOT, _EXAMPLES_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from pygwrx.core import BrentSearch, GoldenSectionSearch, OptimizationResult


def objective(x):
    """Simple convex objective with a known minimum."""
    return (x - 2.5) ** 2 + 1.0


golden = GoldenSectionSearch(tol=1e-7, max_iter=100, verbose=False)
brent = BrentSearch(tol=1e-7, max_iter=100, verbose=False)
print("golden=", golden.minimize(objective, 0.0, 5.0))
print("brent=", brent.minimize(objective, 0.0, 5.0))
print("manual_result=", OptimizationResult(2.5, 1.0, 10, True, evaluations=12))
