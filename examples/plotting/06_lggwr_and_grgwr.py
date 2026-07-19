# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""All visualization functions for the two original research models."""

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
from _models import original_models

from pygwrx.plotting import (
    plot_grgwr_coefficient_surface,
    plot_grgwr_convergence,
    plot_grgwr_regime_sizes,
    plot_grgwr_regimes,
    plot_lggwr_latent_geometry,
    plot_lggwr_metric_matrix,
    plot_lggwr_neighbourhood_comparison,
    plot_lggwr_training,
)

lggwr, grgwr = original_models()
plots = {
    "lggwr_geometry.png": plot_lggwr_latent_geometry(lggwr),
    "lggwr_metric.png": plot_lggwr_metric_matrix(lggwr),
    "lggwr_training.png": plot_lggwr_training(lggwr),
    "lggwr_neighbours.png": plot_lggwr_neighbourhood_comparison(lggwr, 0),
    "grgwr_regimes.png": plot_grgwr_regimes(grgwr),
    "grgwr_convergence.png": plot_grgwr_convergence(grgwr),
    "grgwr_sizes.png": plot_grgwr_regime_sizes(grgwr),
    "grgwr_surface.png": plot_grgwr_coefficient_surface(grgwr, "x1"),
}
for name, result in plots.items():
    print(save_plot(result, name))
