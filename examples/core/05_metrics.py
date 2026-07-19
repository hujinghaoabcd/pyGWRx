# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Calculate every public model-fit and effective-parameter metric."""

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
    compute_adjusted_r_squared,
    compute_aic,
    compute_aicc,
    compute_bic,
    compute_diagnostics,
    compute_edf,
    compute_effective_parameters,
    compute_enp,
    compute_local_r_squared,
    compute_r_squared,
    compute_trace_statistics,
)

y = np.array([1.0, 2.0, 2.8, 4.2, 5.0])
yhat = np.array([1.1, 1.9, 3.0, 4.0, 4.9])
hat = np.eye(5) * 0.4
weights = np.vstack([np.linspace(1.0, 0.2, 5)] * 5)
trace = compute_trace_statistics(hat)
print("r2=", compute_r_squared(y, yhat))
print("adjusted_r2=", compute_adjusted_r_squared(y, yhat, edf=3.0))
print("aic=", compute_aic(y, yhat, n_params=2.0))
print("aicc=", compute_aicc(y, yhat, n_params=2.0))
print("bic=", compute_bic(y, yhat, trace_S=2.0))
print("local_r2=", compute_local_r_squared(y, yhat, weights))
print("effective_parameters=", compute_effective_parameters(hat))
print("trace_statistics=", trace)
print("edf=", compute_edf(5, trace["trace_S"], trace["trace_StS"]))
print("enp=", compute_enp(trace["trace_S"], trace["trace_StS"]))
print(
    "diagnostics=",
    compute_diagnostics(y, yhat, hat, n_features=1, compute_gwr_stats=True),
)
