# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Visualization for GWSS, GWPCA, and GWDA results.

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

from pygwrx.plotting._model_helpers import coords_for_model, figure_axis
from pygwrx.plotting._spatial import add_colorbar, render_spatial_values
from pygwrx.plotting._style import plotting_theme, resolve_color_scale


def _feature_index(model, feature) -> tuple[int, str]:
    names = getattr(model, "feature_names_in_", None)
    if names is None:
        names = getattr(model, "feature_names_", None)
    if names is None:
        names = getattr(model, "var_names_", None)
    if names is None:
        matrix = getattr(model, "local_mean_", None)
        if matrix is None:
            matrix = getattr(model, "X_train_", None)
        n = 0 if matrix is None else int(np.asarray(matrix).shape[1])
        names = [f"x{i}" for i in range(n)]
    else:
        names = [str(name) for name in names]
    if isinstance(feature, str):
        if feature not in names:
            raise ValueError(f"Unknown feature {feature!r}.")
        return names.index(feature), feature
    index = int(feature)
    if index < 0 or index >= len(names):
        raise IndexError("feature index is outside the fitted range.")
    return index, names[index]


def plot_gwss_statistic(
    model,
    statistic: str = "mean",
    feature=0,
    *,
    second_feature=None,
    geometry=None,
    theme: str = "default",
    ax: Optional[plt.Axes] = None,
    figsize: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
):
    """Map a local GWSS univariate or pairwise summary statistic."""
    token = str(statistic).strip().lower().replace("-", "_")
    univariate = {
        "mean": ("local_mean_", "Local mean", False),
        "variance": ("local_var_", "Local variance", False),
        "std": ("local_std_", "Local standard deviation", False),
        "skewness": ("local_skewness_", "Local skewness", True),
        "cv": ("local_cv_", "Local coefficient of variation", True),
        "median": ("local_median_", "Local median", False),
        "iqr": ("local_iqr_", "Local IQR", False),
        "qi": ("local_qi_", "Local quantile imbalance", True),
    }
    pairwise = {
        "covariance": ("local_cov_", "Local covariance", True),
        "correlation": ("local_corr_", "Local Pearson correlation", True),
        "spearman": ("local_corr_spearman_", "Local Spearman correlation", True),
    }
    first, first_label = _feature_index(model, feature)
    if token in univariate:
        attribute, label, center = univariate[token]
        matrix = np.asarray(getattr(model, attribute, None), dtype=float)
        if matrix.ndim != 2:
            raise ValueError(f"The fitted model does not expose {attribute}.")
        values = matrix[:, first]
        display = f"{label}: {first_label}"
    elif token in pairwise:
        if second_feature is None:
            raise ValueError("second_feature is required for pairwise statistics.")
        second, second_label = _feature_index(model, second_feature)
        attribute, label, center = pairwise[token]
        raw = getattr(model, attribute, None)
        if isinstance(raw, dict):
            pair = tuple(sorted((first, second)))
            if pair[0] == pair[1] or pair not in raw:
                raise ValueError(
                    "Pairwise GWSS requires two different fitted features."
                )
            values = np.asarray(raw[pair], dtype=float).reshape(-1)
        else:
            array = np.asarray(raw, dtype=float)
            if array.ndim == 3:
                values = array[:, first, second]
            elif array.ndim == 2:
                p = len(getattr(model, "var_names_", [])) or int(
                    getattr(model, "n_features_in_", 0) or 0
                )
                pairs = [(i, j) for i in range(p) for j in range(i + 1, p)]
                try:
                    pair_index = pairs.index(tuple(sorted((first, second))))
                except ValueError as exc:
                    raise ValueError(
                        "Pairwise GWSS requires two different features."
                    ) from exc
                values = array[:, pair_index]
            else:
                raise ValueError(
                    f"The fitted model does not expose usable {attribute}."
                )
        display = f"{label}: {first_label} × {second_label}"
    else:
        raise ValueError(f"Unknown GWSS statistic {statistic!r}.")
    coords = np.asarray(getattr(model, "coords_summary_", None), dtype=float)
    with plotting_theme(theme):
        fig, axis = figure_axis(ax, figsize, theme)
        cmap, norm = resolve_color_scale(values, center_zero=center)
        artist = render_spatial_values(
            axis,
            values,
            coords=coords,
            geometry=geometry,
            cmap=cmap,
            norm=norm,
        )
        add_colorbar(fig, axis, artist, display)
        axis.set_title(title or display)
        fig.tight_layout()
        return fig, axis


def plot_gwpca_explained_variance(
    model,
    component: int = 0,
    *,
    cumulative: bool = False,
    geometry=None,
    theme: str = "default",
    ax: Optional[plt.Axes] = None,
    figsize: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
):
    """Map local explained variance for one component or cumulatively."""
    local_pv = np.asarray(getattr(model, "local_pv_", None), dtype=float)
    if local_pv.ndim != 2:
        raise ValueError("The fitted model does not expose local_pv_.")
    index = int(component)
    if index < 0 or index >= local_pv.shape[1]:
        raise IndexError("component is outside the fitted component range.")
    values = (
        np.sum(local_pv[:, : index + 1], axis=1) if cumulative else local_pv[:, index]
    )
    label = (
        f"Cumulative explained variance through PC{index + 1}"
        if cumulative
        else f"PC{index + 1} explained variance"
    )
    with plotting_theme(theme):
        fig, axis = figure_axis(ax, figsize, theme)
        cmap, norm = resolve_color_scale(
            values,
            center_zero=False,
            vmin=0.0,
            vmax=(
                max(100.0, float(np.nanmax(values))) if np.nanmax(values) > 1.0 else 1.0
            ),
        )
        artist = render_spatial_values(
            axis,
            values,
            coords=np.asarray(model.eval_coords_, dtype=float),
            geometry=geometry,
            cmap=cmap,
            norm=norm,
        )
        add_colorbar(fig, axis, artist, label)
        axis.set_title(title or label)
        fig.tight_layout()
        return fig, axis


def plot_gwpca_loading(
    model,
    feature=0,
    component: int = 0,
    *,
    geometry=None,
    theme: str = "default",
    ax: Optional[plt.Axes] = None,
    figsize: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
):
    """Map a local principal-component loading surface."""
    index, name = _feature_index(model, feature)
    loadings = np.asarray(getattr(model, "loadings_", None), dtype=float)
    pc = int(component)
    if loadings.ndim != 3 or pc < 0 or pc >= loadings.shape[2]:
        raise ValueError(
            "The fitted model does not expose the requested loading component."
        )
    values = loadings[:, index, pc]
    with plotting_theme(theme):
        fig, axis = figure_axis(ax, figsize, theme)
        cmap, norm = resolve_color_scale(values, center_zero=True)
        artist = render_spatial_values(
            axis,
            values,
            coords=np.asarray(model.eval_coords_, dtype=float),
            geometry=geometry,
            cmap=cmap,
            norm=norm,
        )
        add_colorbar(fig, axis, artist, f"Loading: {name}")
        axis.set_title(title or f"PC{pc + 1} loading: {name}")
        fig.tight_layout()
        return fig, axis


def plot_gwda_classification(
    model,
    *,
    geometry=None,
    confidence: bool = False,
    theme: str = "default",
    ax: Optional[plt.Axes] = None,
    figsize: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
):
    """Map predicted classes or maximum class probability for GWDA."""
    probabilities = np.asarray(getattr(model, "probabilities_", None), dtype=float)
    classes = np.asarray(getattr(model, "classes_", None))
    if probabilities.ndim != 2 or classes.ndim != 1:
        raise ValueError(
            "The fitted model does not expose probabilities_ and classes_."
        )
    coords = coords_for_model(model)
    if confidence:
        values = np.max(probabilities, axis=1)
        with plotting_theme(theme):
            fig, axis = figure_axis(ax, figsize, theme)
            cmap, norm = resolve_color_scale(
                values, center_zero=False, vmin=0.0, vmax=1.0
            )
            artist = render_spatial_values(
                axis, values, coords=coords, geometry=geometry, cmap=cmap, norm=norm
            )
            add_colorbar(fig, axis, artist, "Maximum class probability")
            axis.set_title(title or "GWDA classification confidence")
            fig.tight_layout()
            return fig, axis
    predicted = classes[np.argmax(probabilities, axis=1)]
    _, encoded = np.unique(predicted, return_inverse=True)
    cmap = ListedColormap(plt.get_cmap("tab10").colors[: max(len(classes), 1)])
    norm = BoundaryNorm(np.arange(-0.5, len(classes) + 0.5), cmap.N)
    with plotting_theme(theme):
        fig, axis = figure_axis(ax, figsize, theme)
        render_spatial_values(
            axis,
            encoded.astype(float),
            coords=coords,
            geometry=geometry,
            cmap=cmap,
            norm=norm,
        )
        handles = [
            plt.Line2D(
                [0], [0], marker="o", linestyle="", color=cmap(i), label=str(label)
            )
            for i, label in enumerate(classes)
        ]
        axis.legend(handles=handles, title="Class", loc="best")
        axis.set_title(title or "GWDA predicted class")
        fig.tight_layout()
        return fig, axis


def plot_gwda_confusion_matrix(
    model,
    *,
    normalize: bool = False,
    theme: str = "default",
    ax: Optional[plt.Axes] = None,
    figsize: Optional[Tuple[float, float]] = None,
    title: str = "GWDA confusion matrix",
):
    """Plot a calibration/validation confusion matrix when labels are available."""
    true = np.asarray(getattr(model, "y_train_", None))
    probabilities = np.asarray(getattr(model, "probabilities_", None), dtype=float)
    classes = np.asarray(getattr(model, "classes_", None))
    if true.ndim != 1 or probabilities.ndim != 2:
        raise ValueError(
            "The fitted model does not expose row-aligned labels and probabilities."
        )
    predicted = classes[np.argmax(probabilities, axis=1)]
    matrix = np.zeros((classes.size, classes.size), dtype=float)
    for i, actual in enumerate(classes):
        for j, guess in enumerate(classes):
            matrix[i, j] = np.sum((true == actual) & (predicted == guess))
    if normalize:
        row_sum = matrix.sum(axis=1, keepdims=True)
        matrix = np.divide(
            matrix, row_sum, out=np.zeros_like(matrix), where=row_sum > 0
        )
    with plotting_theme(theme):
        fig, axis = figure_axis(ax, figsize, theme)
        image = axis.imshow(matrix, cmap="Blues", aspect="equal")
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        axis.set_xticks(
            np.arange(classes.size), labels=[str(value) for value in classes]
        )
        axis.set_yticks(
            np.arange(classes.size), labels=[str(value) for value in classes]
        )
        axis.set_xlabel("Predicted")
        axis.set_ylabel("Observed")
        axis.set_title(title)
        for i in range(classes.size):
            for j in range(classes.size):
                axis.text(
                    j,
                    i,
                    f"{matrix[i, j]:.2f}" if normalize else f"{int(matrix[i, j])}",
                    ha="center",
                    va="center",
                )
        fig.tight_layout()
        return fig, axis
