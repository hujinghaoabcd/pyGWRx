# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Run all public local-regression solver utilities."""

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
    adaptive_bandwidth_weights,
    compute_hat_matrix,
    gaussian_kernel,
    local_regression,
    weighted_least_squares,
)

rng = np.random.default_rng(0)
coords = rng.uniform(0.0, 5.0, size=(20, 2))
x = rng.normal(size=20)
X = np.column_stack((np.ones(20), x))
y = 1.0 + 2.0 * x + rng.normal(0.0, 0.05, 20)
distances = np.linalg.norm(coords - coords[0], axis=1)
weights = gaussian_kernel(distances, bandwidth=2.0)
beta, covariance = weighted_least_squares(X, y, weights)
print("beta=", beta)
print("covariance_shape=", covariance.shape)
print("adaptive_scale=", adaptive_bandwidth_weights(distances, 8))
print(
    "local_parameters=",
    local_regression(X, y, coords, coords[:3], gaussian_kernel, 2.0),
)
hat = compute_hat_matrix(X, coords, gaussian_kernel, 2.0)
print("hat_shape_trace=", hat.shape, np.trace(hat))
