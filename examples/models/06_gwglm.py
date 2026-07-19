# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Fit Gaussian, binomial, and Poisson GWGLM families."""

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
from _common import count_regression, print_model_result, spatial_regression

from pygwrx import GWGLM, GWGLMPredictionResult

X, y, coords = spatial_regression(p=2)
gaussian = GWGLM(family="gaussian", bandwidth=24, adaptive=True).fit(X, y, coords)
print_model_result(gaussian)

binary = (y > np.median(y)).astype(int)
binomial = GWGLM(family="binomial", bandwidth=24, adaptive=True).fit(X, binary, coords)
binomial_result = binomial.predict_result(X.iloc[:3], coords.iloc[:3])
assert isinstance(binomial_result, GWGLMPredictionResult)
print(binomial_result.to_frame())

Xc, counts, coordsc, exposure = count_regression()
poisson = GWGLM(family="poisson", bandwidth=24, adaptive=True).fit(
    Xc, counts, coordsc, exposure=exposure
)
print(
    "poisson means=",
    poisson.predict(Xc.iloc[:3], coordsc.iloc[:3], exposure=exposure[:3]),
)
