# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""All GWSS, GWPCA, and GWDA visualization functions."""

# Allow this script to run directly from any working directory.
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_EXAMPLES_ROOT = _PROJECT_ROOT / "examples"
_SRC_ROOT = _PROJECT_ROOT / "src"
for _path in (_SRC_ROOT, _EXAMPLES_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import matplotlib

matplotlib.use("Agg", force=True)
from _common import save_plot
from _models import multivariate_models

from pygwrx.plotting import (
    plot_gwda_classification,
    plot_gwda_confusion_matrix,
    plot_gwpca_explained_variance,
    plot_gwpca_loading,
    plot_gwss_statistic,
)

X, coords, gwss, gwpca, Xc, yc, cc, gwda = multivariate_models()
plots = {
    "gwss_mean.png": plot_gwss_statistic(gwss, "mean", "x1"),
    "gwss_correlation.png": plot_gwss_statistic(
        gwss, "correlation", "x1", second_feature="x2"
    ),
    "gwpca_variance.png": plot_gwpca_explained_variance(gwpca, 0),
    "gwpca_cumulative.png": plot_gwpca_explained_variance(gwpca, 0, cumulative=True),
    "gwpca_loading.png": plot_gwpca_loading(gwpca, "x1", 0),
    "gwda_classification.png": plot_gwda_classification(gwda),
    "gwda_confidence.png": plot_gwda_classification(gwda, confidence=True),
    "gwda_confusion.png": plot_gwda_confusion_matrix(gwda, normalize=True),
}
for name, result in plots.items():
    print(save_plot(result, name))
