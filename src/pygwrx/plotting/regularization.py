# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Plots for geographically weighted regularization and mixed models.

Author:
    Jinghao Hu
"""

from __future__ import annotations

__author__ = "Jinghao Hu"
__license__ = "MIT"

from typing import Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from pygwrx.diagnostics import feature_names
from pygwrx.plotting._model_helpers import coords_for_model, figure_axis
from pygwrx.plotting._spatial import render_spatial_values
from pygwrx.plotting._style import plotting_theme, resolve_color_scale


def plot_gwlasso_selection_frequency(
    model,
    *,
    theme: str = "default",
    ax: Optional[plt.Axes] = None,
    figsize: Optional[Tuple[float, float]] = None,
    title: str = "GWLasso local variable selection",
):
    """Plot the percentage of locations with a non-zero coefficient."""
    coef = np.asarray(getattr(model, "coef_", None), dtype=float)
    if coef.ndim != 2:
        raise ValueError("The fitted model does not expose a local coef_ matrix.")
    active_tol = float(getattr(model, "active_tol", 1.0e-8))
    frequency = np.mean(np.abs(coef) > active_tol, axis=0) * 100.0
    names = list(feature_names(model))
    with plotting_theme(theme):
        fig, axis = figure_axis(ax, figsize, theme, wide=True)
        axis.bar(names, frequency)
        axis.set_ylim(0.0, 100.0)
        axis.set_ylabel("Selected locations (%)")
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=30)
        axis.grid(True, axis="y", alpha=0.22)
        fig.tight_layout()
        return fig, axis


def plot_gwlasso_active_map(
    model,
    feature,
    *,
    geometry=None,
    theme: str = "default",
    ax: Optional[plt.Axes] = None,
    figsize: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
):
    """Map locations where a GWLasso coefficient is active."""
    names = list(feature_names(model))
    if isinstance(feature, str):
        if feature not in names:
            raise ValueError(f"Unknown feature {feature!r}.")
        index = names.index(feature)
    else:
        index = int(feature)
    coef = np.asarray(getattr(model, "coef_", None), dtype=float)
    if coef.ndim != 2 or index < 0 or index >= coef.shape[1]:
        raise ValueError("feature is outside the fitted coefficient range.")
    active = (
        np.abs(coef[:, index]) > float(getattr(model, "active_tol", 1.0e-8))
    ).astype(float)
    with plotting_theme(theme):
        fig, axis = figure_axis(ax, figsize, theme)
        cmap, norm = resolve_color_scale(
            active, center_zero=False, vmin=-0.01, vmax=1.01, cmap="Greys"
        )
        render_spatial_values(
            axis,
            active,
            coords=coords_for_model(model),
            geometry=geometry,
            cmap=cmap,
            norm=norm,
        )
        axis.set_title(title or f"GWLasso active coefficient: {names[index]}")
        fig.tight_layout()
        return fig, axis


def plot_gwlasso_alpha(
    model,
    *,
    geometry=None,
    theme: str = "default",
    ax: Optional[plt.Axes] = None,
    figsize: Optional[Tuple[float, float]] = None,
    title: str = "GWLasso local regularization",
):
    """Map the locally selected Lasso penalty."""
    alpha = np.asarray(getattr(model, "alpha_", None), dtype=float)
    if alpha.ndim == 0:
        alpha = np.full(coords_for_model(model).shape[0], float(alpha))
    alpha = alpha.reshape(-1)
    with plotting_theme(theme):
        fig, axis = figure_axis(ax, figsize, theme)
        cmap, norm = resolve_color_scale(alpha, center_zero=False, cmap="magma")
        artist = render_spatial_values(
            axis,
            alpha,
            coords=coords_for_model(model),
            geometry=geometry,
            cmap=cmap,
            norm=norm,
        )
        fig.colorbar(artist, ax=axis, fraction=0.046, pad=0.04, label="Alpha")
        axis.set_title(title)
        fig.tight_layout()
        return fig, axis


def plot_mixed_gwr_coefficients(
    model,
    *,
    theme: str = "default",
    figsize: Optional[Tuple[float, float]] = None,
    title: str = "Mixed GWR global and local coefficients",
):
    """Compare global coefficients with distributions of local coefficients."""
    global_coef = np.asarray(getattr(model, "coef_global_", None), dtype=float).reshape(
        -1
    )
    local_coef = np.asarray(getattr(model, "coef_local_", None), dtype=float)
    if local_coef.ndim != 2:
        raise ValueError("The fitted model does not expose coef_local_.")
    names = list(feature_names(model))
    global_indices = np.asarray(getattr(model, "global_var_indices_", []), dtype=int)
    local_indices = np.asarray(getattr(model, "local_var_indices_", []), dtype=int)
    with plotting_theme(theme):
        fig, axes = plt.subplots(1, 2, figsize=figsize or (11.0, 4.5))
        if global_coef.size:
            axes[0].bar([names[i] for i in global_indices], global_coef)
            axes[0].axhline(0.0, color="0.4", linestyle="--", linewidth=0.8)
        else:
            axes[0].text(0.5, 0.5, "No global variables", ha="center", va="center")
        axes[0].set_title("Global coefficients")
        if local_coef.shape[1]:
            labels = [names[i] for i in local_indices]
            version_parts = matplotlib.__version__.split(".")
            version = (int(version_parts[0]), int(version_parts[1]))
            label_keyword = "tick_labels" if version >= (3, 9) else "labels"
            boxplot_kwargs = {label_keyword: labels}
            axes[1].boxplot(
                [local_coef[:, j] for j in range(local_coef.shape[1])],
                showfliers=False,
                **boxplot_kwargs,
            )
            axes[1].axhline(0.0, color="0.4", linestyle="--", linewidth=0.8)
        axes[1].set_title("Local coefficient distributions")
        for axis in axes:
            axis.tick_params(axis="x", rotation=30)
            axis.grid(True, axis="y", alpha=0.22)
        fig.suptitle(title)
        fig.tight_layout()
        return fig, axes
