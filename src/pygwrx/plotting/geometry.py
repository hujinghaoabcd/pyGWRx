# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Visual interpretation of LG-GWR latent geometry.

Author:
    Jinghao Hu
"""

from __future__ import annotations

__author__ = "Jinghao Hu"
__license__ = "MIT"

from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from pygwrx.plotting._model_helpers import coords_for_model, figure_axis
from pygwrx.plotting._style import plotting_theme


def plot_lggwr_latent_geometry(
    model,
    *,
    values=None,
    theme: str = "default",
    figsize: Optional[Tuple[float, float]] = None,
    title: str = "LG-GWR geographical and latent geometry",
):
    """Compare physical coordinates with the first two latent dimensions."""
    coords = coords_for_model(model)
    latent = np.asarray(getattr(model, "latent_coords_", None), dtype=float)
    if latent.ndim != 2 or latent.shape[0] != coords.shape[0]:
        raise ValueError("The fitted model does not expose row-aligned latent_coords_.")
    if latent.shape[1] == 1:
        latent = np.column_stack([latent[:, 0], np.zeros(latent.shape[0])])
    color = (
        np.arange(coords.shape[0])
        if values is None
        else np.asarray(values, dtype=float).reshape(-1)
    )
    if color.size != coords.shape[0]:
        raise ValueError("values must contain one value per fitted observation.")
    with plotting_theme(theme):
        fig, axes = plt.subplots(
            1, 2, figsize=figsize or (10.0, 4.4), constrained_layout=True
        )
        artists = []
        artists.append(
            axes[0].scatter(
                coords[:, 0],
                coords[:, 1],
                c=color,
                cmap="viridis",
                edgecolors="0.2",
                linewidths=0.3,
            )
        )
        artists.append(
            axes[1].scatter(
                latent[:, 0],
                latent[:, 1],
                c=color,
                cmap="viridis",
                edgecolors="0.2",
                linewidths=0.3,
            )
        )
        axes[0].set_title("Geographical space")
        axes[1].set_title("Learned latent space")
        for axis in axes:
            axis.set_aspect("equal", adjustable="datalim")
            axis.set_xlabel("Dimension 1")
            axis.set_ylabel("Dimension 2")
        fig.colorbar(
            artists[-1],
            ax=axes,
            fraction=0.025,
            pad=0.03,
            label="Observation index" if values is None else "Value",
        )
        fig.suptitle(title)
        return fig, axes


def plot_lggwr_metric_matrix(
    model,
    *,
    theme: str = "default",
    ax: Optional[plt.Axes] = None,
    figsize: Optional[Tuple[float, float]] = None,
    title: str = "LG-GWR learned metric matrix",
):
    """Plot the rotation-invariant metric matrix ``A.T @ A`` or ``B.T @ B``."""
    matrix = np.asarray(getattr(model, "metric_matrix_", None), dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("The fitted model does not expose a square metric_matrix_.")
    names = list(getattr(model, "geometry_feature_names_", []))
    if len(names) != matrix.shape[0]:
        names = [f"g{i}" for i in range(matrix.shape[0])]
    extent = max(float(np.max(np.abs(matrix))), np.finfo(float).eps)
    with plotting_theme(theme):
        fig, axis = figure_axis(ax, figsize, theme)
        image = axis.imshow(matrix, cmap="RdBu_r", vmin=-extent, vmax=extent)
        fig.colorbar(
            image, ax=axis, fraction=0.046, pad=0.04, label="Metric coefficient"
        )
        axis.set_xticks(np.arange(len(names)), labels=names, rotation=45, ha="right")
        axis.set_yticks(np.arange(len(names)), labels=names)
        axis.set_title(title)
        fig.tight_layout()
        return fig, axis


def plot_lggwr_training(
    model,
    *,
    theme: str = "default",
    figsize: Optional[Tuple[float, float]] = None,
    title: str = "LG-GWR optimization history",
):
    """Plot LOO loss and bandwidth updates from latent-geometry learning."""
    loss = np.asarray(getattr(model, "loss_history_", None), dtype=float).reshape(-1)
    bandwidth = np.asarray(
        getattr(model, "bandwidth_history_", None), dtype=float
    ).reshape(-1)
    if loss.size == 0:
        raise ValueError("The fitted model does not expose loss_history_.")
    with plotting_theme(theme):
        fig, axes = plt.subplots(1, 2, figsize=figsize or (10.0, 4.2))
        axes[0].plot(np.arange(loss.size), loss, marker="o", markersize=3)
        axes[0].set_xlabel("Optimization step")
        axes[0].set_ylabel("LOO loss")
        axes[0].set_title("Geometry learning")
        if bandwidth.size:
            axes[1].plot(np.arange(bandwidth.size), bandwidth, marker="s")
            axes[1].set_xlabel("Bandwidth update")
            axes[1].set_ylabel("Bandwidth")
        else:
            axes[1].text(0.5, 0.5, "No bandwidth history", ha="center", va="center")
        axes[1].set_title("Scale coordination")
        for axis in axes:
            axis.grid(True, alpha=0.22)
        fig.suptitle(title)
        fig.tight_layout()
        return fig, axes


def plot_lggwr_neighbourhood_comparison(
    model,
    focus: int,
    *,
    n_neighbors: int = 12,
    theme: str = "default",
    figsize: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
):
    """Compare geographical and learned nearest neighbours around one observation."""
    coords = coords_for_model(model)
    latent = np.asarray(getattr(model, "latent_coords_", None), dtype=float)
    index = int(focus)
    if index < 0 or index >= coords.shape[0]:
        raise IndexError("focus is outside the fitted sample range.")
    k = min(max(int(n_neighbors), 1), coords.shape[0])
    geo_dist = np.linalg.norm(coords - coords[index], axis=1)
    latent_dist = np.linalg.norm(latent - latent[index], axis=1)
    geo_neighbors = np.argsort(geo_dist)[:k]
    latent_neighbors = np.argsort(latent_dist)[:k]
    with plotting_theme(theme):
        fig, axes = plt.subplots(
            1, 2, figsize=figsize or (10.0, 4.4), constrained_layout=True
        )
        for axis, neighbors, panel in (
            (axes[0], geo_neighbors, "Geographical neighbours"),
            (axes[1], latent_neighbors, "Latent neighbours"),
        ):
            axis.scatter(coords[:, 0], coords[:, 1], c="0.82", s=32)
            axis.scatter(
                coords[neighbors, 0],
                coords[neighbors, 1],
                c=np.arange(k),
                cmap="viridis",
                s=55,
            )
            axis.scatter(
                coords[index, 0],
                coords[index, 1],
                marker="*",
                s=180,
                c="black",
                label="Focus",
            )
            axis.set_aspect("equal", adjustable="datalim")
            axis.set_title(panel)
            axis.legend(loc="best")
        fig.suptitle(title or f"LG-GWR neighbourhood comparison: focus {index}")
        return fig, axes
