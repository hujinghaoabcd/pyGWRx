# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Fit geographically weighted discriminant analysis."""

# Allow this script to run directly from any working directory.
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_EXAMPLES_ROOT = _PROJECT_ROOT / "examples"
_SRC_ROOT = _PROJECT_ROOT / "src"
for _path in (_SRC_ROOT, _EXAMPLES_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from _common import classification_data

from pygwrx import GWDA

X, y, coords = classification_data()
model = GWDA(bandwidth=28, adaptive=True, quadratic=False).fit(X, y, coords)
print(model.summary())
print("classes=", model.classes_)
print("predictions=", model.predict(X.iloc[:5], coords.iloc[:5]))
print("probabilities=", model.predict_proba(X.iloc[:5], coords.iloc[:5]))
