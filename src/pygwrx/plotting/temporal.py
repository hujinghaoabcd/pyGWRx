# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Publication-ready temporal plots for GTWR-family models.

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

from pygwrx.diagnostics import (
    model_times,
    parameter_trajectory,
    temporal_groups,
    temporal_parameter_frame,
)
from pygwrx.plotting._model_helpers import figure_axis
from pygwrx.plotting._spatial import render_spatial_values
from pygwrx.plotting._style import plotting_theme, resolve_color_scale


def plot_temporal_coefficient_slices(
    model,
    feature,
    *,
    times: Optional[Sequence[object]] = None,
    max_cols: int = 3,
    shared_scale: bool = True,
    theme: str = "default",
    figsize: Optional[Tuple[float, float]] = None,
    cmap: Optional[str] = None,
    title: Optional[str] = None,
):
    """Plot coefficient maps at selected observed time slices."""
    frame = temporal_parameter_frame(model, feature)
    available = list(temporal_groups(model).values)
    selected = available if times is None else list(times)
    missing = [value for value in selected if value not in available]
    if missing:
        raise ValueError(f"Unknown time slices: {missing}.")
    cols = min(max(int(max_cols), 1), len(selected))
    rows = int(math.ceil(len(selected) / cols))
    values = frame.loc[frame["time"].isin(selected), "coefficient"].to_numpy(float)
    cmap_name, norm = resolve_color_scale(values, center_zero=None, cmap=cmap)
    with plotting_theme(theme):
        fig, axes = plt.subplots(
            rows,
            cols,
            figsize=figsize or (4.2 * cols, 3.8 * rows),
            squeeze=False,
            constrained_layout=True,
        )
        artist = None
        for axis, time_value in zip(axes.flat, selected):
            subset = frame[frame["time"] == time_value]
            local_cmap, local_norm = (
                (cmap_name, norm)
                if shared_scale
                else resolve_color_scale(
                    subset["coefficient"].to_numpy(float), center_zero=None, cmap=cmap
                )
            )
            artist = render_spatial_values(
                axis,
                subset["coefficient"].to_numpy(float),
                coords=subset[["coord_0", "coord_1"]].to_numpy(float),
                cmap=local_cmap,
                norm=local_norm,
            )
            axis.set_title(str(time_value))
        for axis in axes.flat[len(selected) :]:
            axis.axis("off")
        if artist is not None and shared_scale:
            fig.colorbar(
                artist,
                ax=list(axes.flat[: len(selected)]),
                fraction=0.025,
                pad=0.02,
                label="Coefficient",
            )
        fig.suptitle(title or f"{model.__class__.__name__}: {feature} through time")
        return fig, axes


def plot_temporal_trajectory(
    model,
    feature,
    *,
    location=None,
    reducer: str = "mean",
    theme: str = "default",
    ax: Optional[plt.Axes] = None,
    figsize: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
):
    """Plot a coefficient trajectory aggregated by time or followed by location."""
    frame = parameter_trajectory(model, feature, location=location, reducer=reducer)
    with plotting_theme(theme):
        fig, axis = figure_axis(ax, figsize, theme, wide=True)
        axis.plot(frame["time"], frame["coefficient"], marker="o")
        axis.axhline(0.0, color="0.45", linestyle="--", linewidth=0.8)
        axis.set_xlabel("Time")
        axis.set_ylabel("Local coefficient")
        mode = "nearest location" if location is not None else reducer
        axis.set_title(
            title or f"{model.__class__.__name__}: {feature} trajectory ({mode})"
        )
        axis.grid(True, alpha=0.22)
        fig.autofmt_xdate()
        fig.tight_layout()
        return fig, axis


def plot_temporal_residuals(
    model,
    *,
    theme: str = "default",
    ax: Optional[plt.Axes] = None,
    figsize: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
):
    """Plot fitted residuals against time with a zero reference line."""
    residuals = np.asarray(getattr(model, "residuals_", None), dtype=float).reshape(-1)
    times = model_times(model)
    if residuals.size != times.size:
        raise ValueError("residuals_ length does not match time values.")
    with plotting_theme(theme):
        fig, axis = figure_axis(ax, figsize, theme, wide=True)
        axis.scatter(times, residuals, alpha=0.7)
        axis.axhline(0.0, color="0.25", linestyle="--", linewidth=1.0)
        axis.set_xlabel("Time")
        axis.set_ylabel("Residual")
        axis.set_title(title or f"{model.__class__.__name__}: residuals through time")
        axis.grid(True, alpha=0.22)
        fig.autofmt_xdate()
        fig.tight_layout()
        return fig, axis


def plot_temporal_bandwidths(
    model,
    *,
    theme: str = "default",
    ax: Optional[plt.Axes] = None,
    figsize: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
):
    """Plot spatial/temporal scales for GTWR, MGTWR, STWR, or SGTWR."""
    names = []
    values = []
    for label, attribute in (
        ("Spatial bandwidth", "spatial_bandwidth_"),
        ("Temporal bandwidth", "temporal_bandwidth_"),
        ("GTWR bandwidth", "bandwidth_"),
        ("Lambda", "lambda_st_"),
        ("Alpha", "alpha_"),
        ("Theta", "theta_"),
    ):
        value = getattr(model, attribute, None)
        if (
            value is not None
            and np.asarray(value).ndim == 0
            and np.isfinite(float(value))
        ):
            names.append(label)
            values.append(float(value))
    bandwidths = getattr(model, "bandwidths_", None)
    taus = getattr(model, "taus_", None)
    if bandwidths is not None:
        for index, value in enumerate(np.asarray(bandwidths, dtype=float).reshape(-1)):
            names.append(f"Bandwidth {index}")
            values.append(float(value))
    if taus is not None:
        for index, value in enumerate(np.asarray(taus, dtype=float).reshape(-1)):
            names.append(f"Tau {index}")
            values.append(float(value))
    if not values:
        raise ValueError("The fitted model does not expose temporal scale parameters.")
    with plotting_theme(theme):
        fig, axis = figure_axis(ax, figsize, theme, wide=True)
        axis.bar(names, values)
        axis.set_ylabel("Fitted value")
        axis.set_title(title or f"{model.__class__.__name__}: spatiotemporal scales")
        axis.tick_params(axis="x", rotation=35)
        axis.grid(True, axis="y", alpha=0.22)
        fig.tight_layout()
        return fig, axis


def plot_mgtwr_scales(
    model,
    *,
    theme: str = "default",
    figsize: Optional[Tuple[float, float]] = None,
    title: str = "MGTWR variable-specific spatiotemporal scales",
):
    """Plot variable-specific spatial bandwidths and temporal scale parameters."""
    bandwidths = np.asarray(getattr(model, "bandwidths_", None), dtype=float).reshape(
        -1
    )
    taus = np.asarray(getattr(model, "taus_", None), dtype=float).reshape(-1)
    temporal = np.asarray(
        getattr(model, "temporal_bandwidths_", None), dtype=float
    ).reshape(-1)
    if bandwidths.size == 0 or taus.size != bandwidths.size:
        raise ValueError("The fitted model does not expose aligned MGTWR scales.")
    raw_names = getattr(model, "feature_names_in_", None)
    predictor_names = [] if raw_names is None else [str(name) for name in raw_names]
    if len(predictor_names) != int(getattr(model, "n_features_in_", 0) or 0):
        predictor_names = [f"x{i}" for i in range(max(bandwidths.size - 1, 0))]
    names = (
        ["Intercept"] if bool(getattr(model, "fit_intercept", True)) else []
    ) + predictor_names
    if len(names) != bandwidths.size:
        names = [f"Parameter {i}" for i in range(bandwidths.size)]
    panels = [("Spatial bandwidth", bandwidths), ("Tau", taus)]
    if temporal.size == bandwidths.size and np.all(np.isfinite(temporal)):
        panels.append(("Temporal bandwidth", temporal))
    with plotting_theme(theme):
        fig, axes = plt.subplots(
            1,
            len(panels),
            figsize=figsize or (4.2 * len(panels), 4.2),
            squeeze=False,
            constrained_layout=True,
        )
        for axis, (label, values) in zip(axes.flat, panels):
            axis.bar(names, values)
            axis.set_ylabel(label)
            axis.set_title(label)
            axis.tick_params(axis="x", rotation=35)
            axis.grid(True, axis="y", alpha=0.22)
        fig.suptitle(title)
        return fig, axes.reshape(-1)
