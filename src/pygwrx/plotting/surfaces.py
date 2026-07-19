# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Model-aware coefficient, significance, diagnostic, and collinearity maps.

Author:
    Jinghao Hu
"""

from __future__ import annotations

__author__ = "Jinghao Hu"
__license__ = "MIT"

from typing import Any, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch

from pygwrx.plotting._adapters import (
    collinearity_values,
    diagnostic_values,
    model_coords,
    parameter_view,
    significance_mask,
)
from pygwrx.plotting._spatial import add_colorbar, render_spatial_values
from pygwrx.plotting._style import (
    default_figure_size,
    plotting_theme,
    resolve_color_scale,
)


def _figure_ax(ax, figsize, theme):
    if ax is not None:
        return ax.figure, ax
    return plt.subplots(figsize=figsize or default_figure_size(theme))


def plot_coefficient_map(
    model: Any,
    feature: Any,
    *,
    geometry: Any = None,
    significance: Optional[str] = None,
    alpha: float = 0.05,
    theme: str = "default",
    ax: Optional[plt.Axes] = None,
    figsize: Optional[Tuple[float, float]] = None,
    cmap: Optional[str] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    marker_size: float = 45.0,
    title: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot one fitted local coefficient surface.

    Args:
        model: Fitted pyGWRx regression model.
        feature: Feature name, zero-based coefficient index, or ``"intercept"``.
        geometry: Optional GeoDataFrame or GeoSeries aligned to calibration rows.
        significance: Optional correction method: ``adjusted``, ``raw``,
            ``bonferroni``, ``bh``, or ``by``. Non-significant locations are grey.
        alpha: Significance level.
        theme: Plotting theme.
        ax: Existing axis.
        figsize: Figure size when ``ax`` is not supplied.
        cmap: Matplotlib colormap.
        vmin: Shared lower colour limit.
        vmax: Shared upper colour limit.
        marker_size: Point size for coordinate maps.
        title: Optional title.

    Returns:
        Matplotlib ``(figure, axis)``.
    """
    with plotting_theme(theme):
        fig, axis = _figure_ax(ax, figsize, theme)
        view = parameter_view(model, feature)
        mask = None
        suffix = ""
        if significance is not None:
            mask, _, threshold = significance_mask(
                model, feature, alpha=alpha, correction=significance
            )
            suffix = f"; {significance} significance, α={alpha:g}"
        cmap_name, norm = resolve_color_scale(
            view.values, center_zero=None, vmin=vmin, vmax=vmax, cmap=cmap
        )
        artist = render_spatial_values(
            axis,
            view.values,
            coords=model_coords(model),
            geometry=geometry,
            cmap=cmap_name,
            norm=norm,
            mask=mask,
            marker_size=marker_size,
        )
        add_colorbar(fig, axis, artist, f"Coefficient: {view.label}")
        axis.set_title(title or f"{model.__class__.__name__}: {view.label}{suffix}")
        fig.tight_layout()
        return fig, axis


def plot_significance_map(
    model,
    feature,
    *,
    geometry=None,
    correction: str = "adjusted",
    alpha: float = 0.05,
    theme: str = "default",
    ax: Optional[plt.Axes] = None,
    figsize: Optional[Tuple[float, float]] = None,
    marker_size: float = 45.0,
    title: Optional[str] = None,
):
    """Map negative-significant, non-significant, and positive-significant areas."""
    with plotting_theme(theme):
        fig, axis = _figure_ax(ax, figsize, theme)
        mask, view, _ = significance_mask(
            model, feature, alpha=alpha, correction=correction
        )
        categories = np.zeros(view.values.size, dtype=float)
        categories[mask & (view.values < 0.0)] = -1.0
        categories[mask & (view.values > 0.0)] = 1.0
        cmap = ListedColormap(["#4575b4", "#d9d9d9", "#d73027"])
        norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], cmap.N)
        render_spatial_values(
            axis,
            categories,
            coords=model_coords(model),
            geometry=geometry,
            cmap=cmap,
            norm=norm,
            marker_size=marker_size,
        )
        axis.legend(
            handles=[
                Patch(facecolor="#4575b4", label="Significant negative"),
                Patch(facecolor="#d9d9d9", label="Not significant"),
                Patch(facecolor="#d73027", label="Significant positive"),
            ],
            loc="best",
        )
        axis.set_title(
            title
            or f"{model.__class__.__name__}: {view.label} significance ({correction}, α={alpha:g})"
        )
        fig.tight_layout()
        return fig, axis


def plot_local_diagnostic_map(
    model,
    metric: str,
    *,
    geometry=None,
    theme: str = "default",
    ax: Optional[plt.Axes] = None,
    figsize: Optional[Tuple[float, float]] = None,
    cmap: Optional[str] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    marker_size: float = 45.0,
    title: Optional[str] = None,
):
    """Plot a fitted local diagnostic such as Local R² or Cook's distance."""
    with plotting_theme(theme):
        fig, axis = _figure_ax(ax, figsize, theme)
        values, label, center_zero = diagnostic_values(model, metric)
        cmap_name, norm = resolve_color_scale(
            values,
            center_zero=center_zero,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
        )
        artist = render_spatial_values(
            axis,
            values,
            coords=model_coords(model),
            geometry=geometry,
            cmap=cmap_name,
            norm=norm,
            marker_size=marker_size,
        )
        add_colorbar(fig, axis, artist, label)
        axis.set_title(title or f"{model.__class__.__name__}: {label}")
        fig.tight_layout()
        return fig, axis


def plot_local_collinearity(
    model,
    metric: str = "condition_number",
    *,
    geometry=None,
    theme: str = "default",
    ax: Optional[plt.Axes] = None,
    figsize: Optional[Tuple[float, float]] = None,
    cmap: str = "magma",
    marker_size: float = 45.0,
    show_threshold: bool = True,
    title: Optional[str] = None,
):
    """Plot local condition numbers or LCR-GWR ridge compensation."""
    with plotting_theme(theme):
        fig, axis = _figure_ax(ax, figsize, theme)
        values, label, threshold = collinearity_values(model, metric)
        cmap_name, norm = resolve_color_scale(values, center_zero=False, cmap=cmap)
        artist = render_spatial_values(
            axis,
            values,
            coords=model_coords(model),
            geometry=geometry,
            cmap=cmap_name,
            norm=norm,
            marker_size=marker_size,
        )
        add_colorbar(fig, axis, artist, label)
        if show_threshold and threshold is not None:
            count = int(np.sum(values > threshold))
            axis.text(
                0.02,
                0.02,
                f"> {threshold:g}: {count}/{values.size}",
                transform=axis.transAxes,
                ha="left",
                va="bottom",
                bbox={"facecolor": "white", "edgecolor": "0.5", "alpha": 0.85},
            )
        axis.set_title(title or f"{model.__class__.__name__}: {label}")
        fig.tight_layout()
        return fig, axis
