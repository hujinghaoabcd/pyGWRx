# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""All temporal, multiscale, weight decomposition, and selection-history plots."""

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
from _models import temporal_models

from pygwrx.plotting import (
    plot_mgtwr_scales,
    plot_selection_history,
    plot_temporal_bandwidths,
    plot_temporal_coefficient_slices,
    plot_temporal_residuals,
    plot_temporal_trajectory,
    plot_weight_decomposition,
    plot_weight_profiles,
)

X, y, coords, times, gtwr, mgtwr, sgtwr, sgwr, stwr, search_model = temporal_models()
plots = {
    "temporal_slices.png": plot_temporal_coefficient_slices(gtwr, "x1"),
    "temporal_trajectory.png": plot_temporal_trajectory(gtwr, "x1"),
    "temporal_residuals.png": plot_temporal_residuals(gtwr),
    "temporal_bandwidths.png": plot_temporal_bandwidths(sgtwr),
    "mgtwr_scales.png": plot_mgtwr_scales(mgtwr),
    "sgwr_decomposition.png": plot_weight_decomposition(sgwr, 0),
    "sgwr_profiles.png": plot_weight_profiles(sgwr, 0, sort_by="combined"),
    "stwr_decomposition.png": plot_weight_decomposition(stwr, 0),
    "sgtwr_decomposition.png": plot_weight_decomposition(sgtwr, 0),
    "selection_history.png": plot_selection_history(search_model),
}
for name, result in plots.items():
    print(save_plot(result, name))
