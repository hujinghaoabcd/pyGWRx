# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Visual diagnostics for robust and generalized local regression models.

Author:
    Jinghao Hu
"""

from __future__ import annotations

__author__ = "Jinghao Hu"
__license__ = "MIT"

from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from pygwrx.diagnostics import local_diagnostic_frame
from pygwrx.plotting._model_helpers import coords_for_model, figure_axis
from pygwrx.plotting._spatial import add_colorbar, render_spatial_values
from pygwrx.plotting._style import plotting_theme, resolve_color_scale


def plot_rgwr_weights(
    model,
    *,
    geometry=None,
    theme: str = "default",
    ax: Optional[plt.Axes] = None,
    figsize: Optional[Tuple[float, float]] = None,
    cmap: str = "viridis",
    title: Optional[str] = None,
):
    """Map final robust weights and outline completely rejected observations."""
    frame = local_diagnostic_frame(model)
    if "robust_weight" not in frame:
        raise ValueError("The fitted model does not expose robust_weights_.")
    values = frame["robust_weight"].to_numpy(float)
    with plotting_theme(theme):
        fig, axis = figure_axis(ax, figsize, theme)
        cmap_name, norm = resolve_color_scale(values, center_zero=False, cmap=cmap)
        artist = render_spatial_values(
            axis,
            values,
            coords=coords_for_model(model),
            geometry=geometry,
            cmap=cmap_name,
            norm=norm,
        )
        add_colorbar(fig, axis, artist, "Robust weight")
        rejected = values <= np.finfo(float).eps
        if geometry is None and np.any(rejected):
            coords = coords_for_model(model)
            axis.scatter(
                coords[rejected, 0],
                coords[rejected, 1],
                facecolors="none",
                edgecolors="black",
                s=95,
                linewidths=1.2,
                label="Rejected",
            )
            axis.legend(loc="best")
        axis.set_title(title or f"{model.__class__.__name__}: robust weights")
        fig.tight_layout()
        return fig, axis


def plot_rgwr_convergence(
    model,
    *,
    theme: str = "default",
    ax: Optional[plt.Axes] = None,
    figsize: Optional[Tuple[float, float]] = None,
    title: str = "Robust GWR convergence",
):
    """Plot iteration MSE and the number of downweighted observations."""
    mse = np.asarray(getattr(model, "mse_history_", None), dtype=float).reshape(-1)
    history = getattr(model, "weight_history_", None)
    if mse.size == 0 or not np.all(np.isfinite(mse)):
        raise ValueError("The fitted model does not expose a finite mse_history_.")
    with plotting_theme(theme):
        fig, axis = figure_axis(ax, figsize, theme, wide=True)
        iterations = np.arange(mse.size)
        axis.plot(iterations, mse, marker="o", label="MSE")
        axis.set_xlabel("Iteration")
        axis.set_ylabel("Mean squared error")
        axis.set_title(title)
        axis.grid(True, alpha=0.22)
        if history is not None:
            down = np.asarray(
                [
                    np.sum(np.asarray(weights, dtype=float) < 1.0 - 1.0e-12)
                    for weights in history
                ],
                dtype=float,
            )
            if down.size == mse.size:
                second = axis.twinx()
                second.plot(
                    iterations, down, linestyle="--", marker="s", label="Downweighted"
                )
                second.set_ylabel("Downweighted observations")
                lines = axis.get_lines() + second.get_lines()
                axis.legend(lines, [line.get_label() for line in lines], loc="best")
        fig.tight_layout()
        return fig, axis


def plot_gwglm_residuals(
    model,
    *,
    residual: str = "deviance",
    geometry=None,
    theme: str = "default",
    ax: Optional[plt.Axes] = None,
    figsize: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
):
    """Map Pearson, deviance, or raw residuals from a fitted GWGLM."""
    token = str(residual).strip().lower()
    attributes = {
        "deviance": ("deviance_residuals_", "Deviance residual"),
        "pearson": ("pearson_residuals_", "Pearson residual"),
        "raw": ("residuals_", "Raw residual"),
    }
    if token not in attributes:
        raise ValueError("residual must be 'deviance', 'pearson', or 'raw'.")
    attribute, label = attributes[token]
    values = getattr(model, attribute, None)
    if values is None:
        raise ValueError(f"The fitted model does not expose {attribute}.")
    array = np.asarray(values, dtype=float).reshape(-1)
    with plotting_theme(theme):
        fig, axis = figure_axis(ax, figsize, theme)
        cmap, norm = resolve_color_scale(array, center_zero=True)
        artist = render_spatial_values(
            axis,
            array,
            coords=coords_for_model(model),
            geometry=geometry,
            cmap=cmap,
            norm=norm,
        )
        add_colorbar(fig, axis, artist, label)
        family = getattr(model, "family", "GWGLM")
        axis.set_title(title or f"{model.__class__.__name__} ({family}): {label}")
        fig.tight_layout()
        return fig, axis
