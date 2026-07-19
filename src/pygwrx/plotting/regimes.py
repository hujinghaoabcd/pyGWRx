# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Spatial-regime visualization for GR-GWR.

Author:
    Jinghao Hu
"""

from __future__ import annotations

__author__ = "Jinghao Hu"
__license__ = "MIT"

from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap

from pygwrx.diagnostics import boundary_frame, regime_frame, regime_summary
from pygwrx.plotting._model_helpers import figure_axis
from pygwrx.plotting._spatial import render_spatial_values
from pygwrx.plotting._style import plotting_theme, resolve_color_scale


def plot_grgwr_regimes(
    model,
    *,
    geometry=None,
    show_boundaries: bool = True,
    theme: str = "default",
    ax: Optional[plt.Axes] = None,
    figsize: Optional[Tuple[float, float]] = None,
    title: str = "GR-GWR spatial regimes",
):
    """Map final mechanism regimes and optional boundary graph edges."""
    frame = regime_frame(model)
    labels = frame["regime"].to_numpy(int)
    n = int(np.max(labels)) + 1
    colors = plt.get_cmap("tab20").colors[: max(n, 1)]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-0.5, n + 0.5), cmap.N)
    coords = frame[["coord_0", "coord_1"]].to_numpy(float)
    with plotting_theme(theme):
        fig, axis = figure_axis(ax, figsize, theme)
        render_spatial_values(
            axis,
            labels.astype(float),
            coords=coords,
            geometry=geometry,
            cmap=cmap,
            norm=norm,
        )
        if show_boundaries and geometry is None:
            edges = boundary_frame(model)
            for row in edges.itertuples(index=False):
                axis.plot(
                    [row.x0, row.x1],
                    [row.y0, row.y1],
                    color="black",
                    linewidth=0.8,
                    alpha=0.65,
                )
        handles = [
            plt.Line2D(
                [0], [0], marker="o", linestyle="", color=cmap(i), label=f"Regime {i}"
            )
            for i in range(n)
        ]
        axis.legend(handles=handles, loc="best", ncol=2)
        axis.set_title(title)
        fig.tight_layout()
        return fig, axis


def plot_grgwr_convergence(
    model,
    *,
    theme: str = "default",
    ax: Optional[plt.Axes] = None,
    figsize: Optional[Tuple[float, float]] = None,
    title: str = "GR-GWR ICM convergence",
):
    """Plot the accepted penalized objective sequence."""
    history = np.asarray(
        getattr(model, "objective_history_", None), dtype=float
    ).reshape(-1)
    if history.size == 0:
        raise ValueError("The fitted model does not expose objective_history_.")
    with plotting_theme(theme):
        fig, axis = figure_axis(ax, figsize, theme, wide=True)
        axis.plot(np.arange(history.size), history, marker="o")
        axis.set_xlabel("Accepted iteration")
        axis.set_ylabel("Penalized objective")
        axis.set_title(title)
        axis.grid(True, alpha=0.22)
        fig.tight_layout()
        return fig, axis


def plot_grgwr_regime_sizes(
    model,
    *,
    theme: str = "default",
    ax: Optional[plt.Axes] = None,
    figsize: Optional[Tuple[float, float]] = None,
    title: str = "GR-GWR regime sizes",
):
    """Plot sample counts and optional RMSE by final regime."""
    summary = regime_summary(model)
    with plotting_theme(theme):
        fig, axis = figure_axis(ax, figsize, theme, wide=True)
        labels = [f"Regime {index}" for index in summary.index]
        axis.bar(labels, summary["n_samples"].to_numpy(float), label="Samples")
        axis.set_ylabel("Sample count")
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=30)
        if "rmse" in summary:
            second = axis.twinx()
            second.plot(
                labels,
                summary["rmse"].to_numpy(float),
                marker="o",
                linestyle="--",
                label="RMSE",
            )
            second.set_ylabel("Regime RMSE")
            lines = (
                axis.get_legend_handles_labels()[0]
                + second.get_legend_handles_labels()[0]
            )
            labels_legend = (
                axis.get_legend_handles_labels()[1]
                + second.get_legend_handles_labels()[1]
            )
            axis.legend(lines, labels_legend, loc="best")
        axis.grid(True, axis="y", alpha=0.22)
        fig.tight_layout()
        return fig, axis


def plot_grgwr_coefficient_surface(
    model,
    feature,
    *,
    show_boundaries: bool = True,
    theme: str = "default",
    ax: Optional[plt.Axes] = None,
    figsize: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
):
    """Map a GR-GWR coefficient while overlaying discovered regime boundaries."""
    coef = np.asarray(getattr(model, "coef_", None), dtype=float)
    names = list(
        getattr(model, "feature_names_", [f"x{i}" for i in range(coef.shape[1])])
    )
    if isinstance(feature, str):
        if feature not in names:
            raise ValueError(f"Unknown feature {feature!r}.")
        index = names.index(feature)
    else:
        index = int(feature)
    values = coef[:, index]
    frame = regime_frame(model)
    coords = frame[["coord_0", "coord_1"]].to_numpy(float)
    with plotting_theme(theme):
        fig, axis = figure_axis(ax, figsize, theme)
        cmap, norm = resolve_color_scale(values, center_zero=None)
        artist = render_spatial_values(
            axis, values, coords=coords, cmap=cmap, norm=norm
        )
        fig.colorbar(
            artist,
            ax=axis,
            fraction=0.046,
            pad=0.04,
            label=f"Coefficient: {names[index]}",
        )
        if show_boundaries:
            for row in boundary_frame(model).itertuples(index=False):
                axis.plot(
                    [row.x0, row.x1],
                    [row.y0, row.y1],
                    color="black",
                    linewidth=0.8,
                    alpha=0.65,
                )
        axis.set_title(
            title or f"GR-GWR coefficient and regime boundaries: {names[index]}"
        )
        fig.tight_layout()
        return fig, axis
