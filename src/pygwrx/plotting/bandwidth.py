# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Bandwidth, kernel, and multiscale visualizations.

Author:
    Jinghao Hu
"""

from __future__ import annotations

__author__ = "Jinghao Hu"
__license__ = "MIT"

from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from pygwrx.core.utils import compute_distance_matrix
from pygwrx.plotting._adapters import model_coords, parameter_names
from pygwrx.plotting._style import default_figure_size, plotting_theme
from pygwrx.plotting._validation import as_1d_finite


def plot_bandwidth_selection(
    bandwidths,
    scores,
    selected_bandwidth,
    *,
    criterion: str = "CV",
    theme: str = "default",
    ax: Optional[plt.Axes] = None,
    figsize: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
):
    """Plot an externally recorded bandwidth-selection curve."""
    bandwidth_array = as_1d_finite(bandwidths, "bandwidths")
    score_array = as_1d_finite(scores, "scores")
    if bandwidth_array.size != score_array.size:
        raise ValueError("bandwidths and scores must have the same length.")
    order = np.argsort(bandwidth_array)
    with plotting_theme(theme):
        if ax is None:
            fig, axis = plt.subplots(
                figsize=figsize or default_figure_size(theme, wide=True)
            )
        else:
            fig, axis = ax.figure, ax
        axis.plot(bandwidth_array[order], score_array[order], marker="o", linewidth=1.5)
        axis.axvline(float(selected_bandwidth), linestyle="--", color="0.25")
        axis.set_xlabel("Bandwidth")
        axis.set_ylabel(str(criterion).upper())
        axis.set_title(title or f"Bandwidth selection ({str(criterion).upper()})")
        axis.grid(True, alpha=0.25)
        fig.tight_layout()
        return fig, axis


def plot_mgwr_bandwidths(
    model,
    *,
    theme: str = "default",
    ax: Optional[plt.Axes] = None,
    figsize: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
):
    """Plot variable-specific MGWR bandwidths."""
    bandwidths = getattr(model, "bandwidths_", None)
    if bandwidths is None:
        raise ValueError("The fitted model does not expose bandwidths_.")
    bandwidth_array = as_1d_finite(bandwidths, "bandwidths_")
    names = list(parameter_names(model, include_intercept=True))
    if len(names) != bandwidth_array.size:
        names = [f"parameter {index}" for index in range(bandwidth_array.size)]
    with plotting_theme(theme):
        if ax is None:
            fig, axis = plt.subplots(
                figsize=figsize or default_figure_size(theme, wide=True)
            )
        else:
            fig, axis = ax.figure, ax
        positions = np.arange(bandwidth_array.size)
        axis.barh(positions, bandwidth_array)
        axis.set_yticks(positions, labels=names)
        axis.invert_yaxis()
        axis.set_xlabel(
            "Adaptive neighbours"
            if getattr(model, "adaptive", False)
            else "Distance bandwidth"
        )
        axis.set_title(
            title or f"{model.__class__.__name__}: variable-specific bandwidths"
        )
        for position, value in zip(positions, bandwidth_array):
            text = (
                str(int(round(value)))
                if getattr(model, "adaptive", False)
                else f"{value:.4g}"
            )
            axis.text(value, position, f"  {text}", va="center")
        fig.tight_layout()
        return fig, axis


def plot_kernel_weights(
    model,
    focus: int = 0,
    *,
    theme: str = "default",
    figsize: Optional[Tuple[float, float]] = None,
    marker_size: float = 55.0,
):
    """Show the spatial neighbourhood and weight-decay curve at one calibration point."""
    coords = model_coords(model)
    if not isinstance(focus, (int, np.integer)) or isinstance(focus, (bool, np.bool_)):
        raise TypeError("focus must be an integer calibration-row index.")
    index = int(focus)
    if index < 0 or index >= coords.shape[0]:
        raise IndexError(f"focus must lie in [0, {coords.shape[0] - 1}].")
    if not hasattr(model, "_weights_from_distances"):
        raise TypeError(
            "model does not expose the standard pyGWRx kernel-weight interface."
        )
    distances = compute_distance_matrix(
        coords[index : index + 1],
        coords,
        metric=getattr(model, "distance_metric", "euclidean"),
    ).reshape(-1)
    weights = np.asarray(model._weights_from_distances(distances), dtype=float).reshape(
        -1
    )

    with plotting_theme(theme):
        fig, axes = plt.subplots(1, 2, figsize=figsize or (10.0, 4.4))
        scatter = axes[0].scatter(
            coords[:, 0],
            coords[:, 1],
            c=weights,
            cmap="viridis",
            s=marker_size,
            edgecolors="0.2",
            linewidths=0.35,
        )
        axes[0].scatter(
            coords[index, 0],
            coords[index, 1],
            marker="*",
            s=marker_size * 3.0,
            color="none",
            edgecolors="black",
            linewidths=1.3,
            label=f"Focus {index}",
        )
        axes[0].set_aspect("equal", adjustable="datalim")
        axes[0].set_xlabel("X coordinate")
        axes[0].set_ylabel("Y coordinate")
        axes[0].set_title("Kernel neighbourhood")
        axes[0].legend(loc="best")
        fig.colorbar(scatter, ax=axes[0], fraction=0.046, pad=0.04, label="Weight")

        order = np.argsort(distances)
        axes[1].plot(distances[order], weights[order], marker="o", markersize=3)
        axes[1].set_xlabel("Distance from focus")
        axes[1].set_ylabel("Kernel weight")
        axes[1].set_title(
            f"{getattr(model, 'kernel', 'kernel')} decay; bandwidth={getattr(model, 'bandwidth_', np.nan)}"
        )
        axes[1].grid(True, alpha=0.25)
        fig.tight_layout()
        return fig, axes
