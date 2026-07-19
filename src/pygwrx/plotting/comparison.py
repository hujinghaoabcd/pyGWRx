# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Comparative plots for fitted geographically weighted models.

Author:
    Jinghao Hu
"""

from __future__ import annotations

__author__ = "Jinghao Hu"
__license__ = "MIT"

from typing import Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from pygwrx.plotting._adapters import parameter_view
from pygwrx.plotting._style import plotting_theme
from pygwrx.plotting.surfaces import plot_coefficient_map


def compare_coefficient_surfaces(
    models: Sequence[object],
    feature,
    *,
    geometry=None,
    labels: Optional[Sequence[str]] = None,
    significance: Optional[str] = None,
    alpha: float = 0.05,
    shared_scale: bool = True,
    theme: str = "default",
    figsize: Optional[Tuple[float, float]] = None,
):
    """Compare the same local coefficient across two or more fitted models."""
    model_list = list(models)
    if len(model_list) < 2:
        raise ValueError("models must contain at least two fitted estimators.")
    if labels is None:
        model_labels = [model.__class__.__name__ for model in model_list]
    else:
        model_labels = [str(label) for label in labels]
        if len(model_labels) != len(model_list):
            raise ValueError("labels must contain one entry per model.")

    values = [parameter_view(model, feature).values for model in model_list]
    if len({array.size for array in values}) != 1:
        raise ValueError("All models must contain the same number of calibration rows.")
    combined = np.concatenate(values)
    finite = combined[np.isfinite(combined)]
    if finite.size == 0:
        raise ValueError("Coefficient surfaces contain no finite values.")
    lower = float(np.min(finite)) if shared_scale else None
    upper = float(np.max(finite)) if shared_scale else None
    if shared_scale and lower < 0.0 < upper:
        extent = max(abs(lower), abs(upper))
        lower, upper = -extent, extent

    with plotting_theme(theme):
        n = len(model_list)
        fig, axes = plt.subplots(
            1,
            n,
            figsize=figsize or (4.8 * n, 4.4),
            squeeze=False,
        )
        for index, (model, label) in enumerate(zip(model_list, model_labels)):
            plot_coefficient_map(
                model,
                feature,
                geometry=geometry,
                significance=significance,
                alpha=alpha,
                theme=theme,
                ax=axes[0, index],
                vmin=lower,
                vmax=upper,
                title=label,
            )
        fig.tight_layout()
        return fig, axes.reshape(-1)


def compare_model_diagnostics(
    models: Sequence[object],
    *,
    metrics: Sequence[str] = ("r2", "rmse", "aicc", "enp"),
    labels: Optional[Sequence[str]] = None,
    theme: str = "default",
    figsize: Optional[Tuple[float, float]] = None,
):
    """Compare normalized global diagnostics across fitted models."""
    from pygwrx.diagnostics import diagnostics_frame

    frame = diagnostics_frame(models, labels=labels)
    selected = [str(metric) for metric in metrics if str(metric) in frame.columns]
    if not selected:
        raise ValueError(
            "None of the requested metrics are available on the fitted models."
        )
    with plotting_theme(theme):
        fig, axes = plt.subplots(
            1,
            len(selected),
            figsize=figsize or (4.2 * len(selected), 4.0),
            squeeze=False,
        )
        for axis, metric in zip(axes.flat, selected):
            values = frame[metric].astype(float)
            axis.bar(frame.index.astype(str), values)
            axis.set_title(metric.replace("_", " ").upper())
            axis.tick_params(axis="x", rotation=30)
            axis.grid(True, axis="y", alpha=0.22)
        fig.tight_layout()
        return fig, axes.reshape(-1)
