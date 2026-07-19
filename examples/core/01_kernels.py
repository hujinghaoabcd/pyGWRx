# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Evaluate every public kernel and resolve kernels by name or callable."""

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

from pygwrx.core import (
    bisquare_kernel,
    boxcar_kernel,
    exponential_kernel,
    gaussian_kernel,
    get_kernel_function,
    tricube_kernel,
)

distances = np.array([0.0, 0.5, 1.0, 2.0])
for kernel in (
    gaussian_kernel,
    bisquare_kernel,
    exponential_kernel,
    tricube_kernel,
    boxcar_kernel,
):
    print(kernel.__name__, kernel(distances, bandwidth=1.5))
print("resolved=", get_kernel_function("bisquare").__name__)
print("callable_passthrough=", get_kernel_function(gaussian_kernel) is gaussian_kernel)
