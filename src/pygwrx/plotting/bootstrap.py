# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Visualization for bootstrap tests of geographical variability.

Author:
    Jinghao Hu
"""

from __future__ import annotations

__author__ = "Jinghao Hu"
__license__ = "MIT"

from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from pygwrx.diagnostics import feature_names
from pygwrx.plotting._model_helpers import coords_for_model, figure_axis
from pygwrx.plotting._spatial import add_colorbar, render_spatial_values
from pygwrx.plotting._style import plotting_theme, resolve_color_scale


def _parameter_index(model, feature) -> tuple[int, str]:
    names = list(feature_names(model))
    fit_intercept = bool(getattr(model, "fit_intercept", True))
    if isinstance(feature, str) and feature.lower() in {"intercept", "constant"}:
        if not fit_intercept:
            raise ValueError("The fitted model does not include an intercept.")
        return 0, "Intercept"
    if isinstance(feature, str):
        if feature not in names:
            raise ValueError(f"Unknown feature {feature!r}.")
        index = names.index(feature)
    else:
        index = int(feature)
    if index < 0 or index >= len(names):
        raise IndexError("feature index is outside the fitted range.")
    return index + int(fit_intercept), names[index]


def plot_bootstrap_pvalues(
    model,
    feature,
    *,
    test: str = "localized",
    geometry=None,
    alpha: float = 0.05,
    theme: str = "default",
    ax: Optional[plt.Axes] = None,
    figsize: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
):
    """Map localized bootstrap p values or show a global modified-test p value."""
    index, label = _parameter_index(model, feature)
    token = str(test).strip().lower()
    if token == "localized":
        values = np.asarray(getattr(model, "localized_p_values_", None), dtype=float)
        if values.ndim != 2:
            raise ValueError("The fitted model does not expose localized_p_values_.")
        p = values[:, index]
    elif token == "modified":
        values = np.asarray(
            getattr(model, "modified_p_values_", None), dtype=float
        ).reshape(-1)
        if index >= values.size:
            raise ValueError(
                "modified_p_values_ does not contain the requested parameter."
            )
        p = np.full(coords_for_model(model).shape[0], values[index])
    else:
        raise ValueError("test must be 'localized' or 'modified'.")
    with plotting_theme(theme):
        fig, axis = figure_axis(ax, figsize, theme)
        cmap, norm = resolve_color_scale(
            p, center_zero=False, vmin=0.0, vmax=1.0, cmap="viridis_r"
        )
        artist = render_spatial_values(
            axis,
            p,
            coords=coords_for_model(model),
            geometry=geometry,
            cmap=cmap,
            norm=norm,
        )
        add_colorbar(fig, axis, artist, "Bootstrap p value")
        significant = int(np.sum(p <= alpha))
        axis.text(
            0.02,
            0.02,
            f"p ≤ {alpha:g}: {significant}/{p.size}",
            transform=axis.transAxes,
            bbox={"facecolor": "white", "edgecolor": "0.5", "alpha": 0.85},
        )
        axis.set_title(title or f"Bootstrap {token} test: {label}")
        fig.tight_layout()
        return fig, axis


def plot_bootstrap_bandwidths(
    model,
    *,
    theme: str = "default",
    ax: Optional[plt.Axes] = None,
    figsize: Optional[Tuple[float, float]] = None,
    title: str = "Bootstrap bandwidth distribution",
):
    """Plot bandwidth variability across bootstrap replications."""
    values = np.asarray(
        getattr(model, "bootstrap_bandwidths_", None), dtype=float
    ).reshape(-1)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError(
            "The fitted model does not expose finite bootstrap_bandwidths_."
        )
    with plotting_theme(theme):
        fig, axis = figure_axis(ax, figsize, theme)
        axis.hist(values, bins="auto", edgecolor="0.25", linewidth=0.5)
        observed = getattr(model, "bandwidth_", None)
        if observed is not None:
            axis.axvline(float(observed), color="0.2", linestyle="--", label="Observed")
            axis.legend(loc="best")
        axis.set_xlabel("Bandwidth")
        axis.set_ylabel("Bootstrap count")
        axis.set_title(title)
        axis.grid(True, axis="y", alpha=0.22)
        fig.tight_layout()
        return fig, axis
