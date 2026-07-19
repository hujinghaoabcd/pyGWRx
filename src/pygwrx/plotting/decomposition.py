# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Weight-decomposition plots for SGWR, STWR, and SGTWR.

Author:
    Jinghao Hu
"""

from __future__ import annotations

__author__ = "Jinghao Hu"
__license__ = "MIT"

import math
from typing import Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pygwrx.diagnostics import focus_weight_components
from pygwrx.plotting._model_helpers import coords_for_model
from pygwrx.plotting._spatial import add_colorbar, render_spatial_values
from pygwrx.plotting._style import plotting_theme, resolve_color_scale


def _source_coords(model, n_columns: int) -> np.ndarray:
    coords = getattr(model, "coords_train_", None)
    if coords is not None and len(coords) == n_columns:
        return np.asarray(coords, dtype=float)
    stages = getattr(model, "coords_stages_", None)
    tick_nums = getattr(model, "tick_nums_", None)
    if stages:
        count = len(stages) if tick_nums is None else int(tick_nums)
        stacked = np.vstack(list(reversed(stages[-count:])))
        if stacked.shape[0] == n_columns:
            return stacked
    coords = coords_for_model(model)
    if coords.shape[0] == n_columns:
        return coords
    raise ValueError("Stored weight columns cannot be aligned to model coordinates.")


def plot_weight_decomposition(
    model,
    focus: int,
    *,
    components: Optional[Sequence[str]] = None,
    theme: str = "default",
    figsize: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
):
    """Map stored spatial, temporal, similarity, and combined weights."""
    rows = focus_weight_components(model, focus)
    selected = list(rows) if components is None else [str(name) for name in components]
    unknown = [name for name in selected if name not in rows]
    if unknown:
        raise ValueError(f"Unknown weight components: {unknown}.")
    cols = min(3, len(selected))
    n_rows = int(math.ceil(len(selected) / cols))
    n_columns = len(rows[selected[0]])
    source_coords = _source_coords(model, n_columns)
    with plotting_theme(theme):
        fig, axes = plt.subplots(
            n_rows, cols, figsize=figsize or (4.2 * cols, 3.8 * n_rows), squeeze=False
        )
        for axis, name in zip(axes.flat, selected):
            values = rows[name]
            cmap, norm = resolve_color_scale(values, center_zero=False, vmin=0.0)
            artist = render_spatial_values(
                axis, values, coords=source_coords, cmap=cmap, norm=norm
            )
            add_colorbar(fig, axis, artist, f"{name} weight")
            axis.set_title(name.replace("_", " ").title())
        for axis in axes.flat[len(selected) :]:
            axis.axis("off")
        fig.suptitle(title or f"{model.__class__.__name__}: weights for focus {focus}")
        fig.tight_layout()
        return fig, axes


def plot_weight_profiles(
    model,
    focus: int,
    *,
    components: Optional[Sequence[str]] = None,
    sort_by: Optional[str] = None,
    theme: str = "default",
    ax: Optional[plt.Axes] = None,
    figsize: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
):
    """Compare sorted one-dimensional profiles of stored weight components."""
    rows = focus_weight_components(model, focus)
    selected = list(rows) if components is None else list(components)
    unknown = [name for name in selected if name not in rows]
    if unknown:
        raise ValueError(f"Unknown weight components: {unknown}.")
    key = sort_by or ("combined" if "combined" in rows else selected[0])
    if key not in rows:
        raise ValueError(f"sort_by={key!r} is not a stored component.")
    order = np.argsort(rows[key])[::-1]
    with plotting_theme(theme):
        if ax is None:
            fig, axis = plt.subplots(figsize=figsize or (8.0, 4.5))
        else:
            fig, axis = ax.figure, ax
        for name in selected:
            axis.plot(
                np.arange(order.size), rows[name][order], label=name.replace("_", " ")
            )
        axis.set_xlabel(f"Source observations sorted by {key} weight")
        axis.set_ylabel("Weight")
        axis.set_title(title or f"{model.__class__.__name__}: weight profiles")
        axis.legend(loc="best")
        axis.grid(True, alpha=0.22)
        fig.tight_layout()
        return fig, axis


def plot_selection_history(
    model,
    *,
    criterion: str = "aicc",
    theme: str = "default",
    ax: Optional[plt.Axes] = None,
    figsize: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
):
    """Plot AICc/CV values from an SGWR/STWR/SGTWR parameter search."""
    history = getattr(model, "selection_history_", None)
    if not history:
        history = getattr(model, "alpha_search_history_", None)
    if not history:
        history = getattr(model, "lambda_selection_history_", None)
    if not history:
        raise ValueError(
            "The fitted model does not expose a non-empty selection history."
        )
    frame = pd.DataFrame(history)
    token = str(criterion).strip().lower()
    candidate = next((name for name in frame.columns if name.lower() == token), None)
    if candidate is None:
        candidate = next(
            (name for name in frame.columns if name.lower() in {"score", "aicc", "cv"}),
            None,
        )
    if candidate is None:
        raise ValueError("Selection history does not contain a score column.")
    score = pd.to_numeric(frame[candidate], errors="coerce")
    with plotting_theme(theme):
        if ax is None:
            fig, axis = plt.subplots(figsize=figsize or (8.0, 4.5))
        else:
            fig, axis = ax.figure, ax
        axis.plot(np.arange(len(frame)), score, marker="o")
        if np.isfinite(score).any():
            best = int(np.nanargmin(score.to_numpy(float)))
            axis.scatter(
                [best], [score.iloc[best]], marker="*", s=140, label="Selected"
            )
            axis.legend(loc="best")
        axis.set_xlabel("Candidate index")
        axis.set_ylabel(candidate)
        axis.set_title(title or f"{model.__class__.__name__}: parameter search")
        axis.grid(True, alpha=0.22)
        fig.tight_layout()
        return fig, axis
