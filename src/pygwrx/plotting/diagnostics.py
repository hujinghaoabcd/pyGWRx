# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Reusable regression-diagnostic plots.

The functions in this module remain array-based so they can visualize external
predictions as well as fitted pyGWRx models. Model-aware spatial maps are in
:mod:`pygwrx.plotting.surfaces`.

Author:
    Jinghao Hu
"""

from __future__ import annotations

__author__ = "Jinghao Hu"
__license__ = "MIT"

from typing import Mapping, Optional, Sequence, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from pygwrx.core.metrics import compute_r_squared
from pygwrx.plotting._style import default_figure_size, plotting_theme
from pygwrx.plotting._validation import as_1d_finite, validate_coords


def _new_axis(ax, figsize, theme, *, wide=False):
    if ax is not None:
        return ax.figure, ax
    return plt.subplots(figsize=figsize or default_figure_size(theme, wide=wide))


def plot_residuals(
    fitted_values,
    residuals,
    figsize: Optional[Tuple[float, float]] = None,
    *,
    theme: str = "default",
    ax: Optional[plt.Axes] = None,
    title: str = "Residuals vs fitted values",
    **kwargs,
):
    """Plot residuals against fitted values with a binned mean trend."""
    fitted = as_1d_finite(fitted_values, "fitted_values")
    resid = as_1d_finite(residuals, "residuals")
    if fitted.size != resid.size:
        raise ValueError("fitted_values and residuals must have the same length.")
    with plotting_theme(theme):
        fig, axis = _new_axis(ax, figsize, theme, wide=True)
        axis.scatter(fitted, resid, alpha=kwargs.pop("alpha", 0.65), **kwargs)
        axis.axhline(0.0, color="0.25", linestyle="--", linewidth=1.0)
        if fitted.size >= 10:
            edges = np.quantile(fitted, np.linspace(0.0, 1.0, min(11, fitted.size + 1)))
            edges = np.unique(edges)
            centres, means = [], []
            for lower, upper in zip(edges[:-1], edges[1:]):
                mask = (fitted >= lower) & (
                    fitted <= upper if upper == edges[-1] else fitted < upper
                )
                if np.any(mask):
                    centres.append(float(np.mean(fitted[mask])))
                    means.append(float(np.mean(resid[mask])))
            if len(centres) > 1:
                axis.plot(
                    centres, means, color="0.15", linewidth=1.6, label="Binned mean"
                )
                axis.legend(loc="best")
        axis.set_xlabel("Fitted value")
        axis.set_ylabel("Residual")
        axis.set_title(title)
        axis.grid(True, alpha=0.22)
        fig.tight_layout()
        return fig, axis


def plot_residual_histogram(
    residuals,
    bins: int = 30,
    density: bool = True,
    figsize: Optional[Tuple[float, float]] = None,
    *,
    theme: str = "default",
    ax: Optional[plt.Axes] = None,
    title: str = "Residual distribution",
    **kwargs,
):
    """Plot a residual histogram and optional fitted normal density."""
    resid = as_1d_finite(residuals, "residuals")
    if isinstance(bins, (bool, np.bool_)) or int(bins) < 1:
        raise ValueError("bins must be a positive integer.")
    with plotting_theme(theme):
        fig, axis = _new_axis(ax, figsize, theme, wide=True)
        axis.hist(
            resid,
            bins=int(bins),
            density=bool(density),
            alpha=kwargs.pop("alpha", 0.72),
            edgecolor=kwargs.pop("edgecolor", "0.25"),
            linewidth=kwargs.pop("linewidth", 0.5),
            **kwargs,
        )
        sigma = float(np.std(resid, ddof=1)) if resid.size > 1 else 0.0
        if density and sigma > np.finfo(float).eps:
            mean = float(np.mean(resid))
            x_values = np.linspace(float(np.min(resid)), float(np.max(resid)), 200)
            axis.plot(
                x_values,
                stats.norm.pdf(x_values, mean, sigma),
                color="0.15",
                linewidth=1.6,
                label=f"Normal fit (μ={mean:.3g}, σ={sigma:.3g})",
            )
            axis.legend(loc="best")
        axis.set_xlabel("Residual")
        axis.set_ylabel("Density" if density else "Count")
        axis.set_title(title)
        axis.grid(True, axis="y", alpha=0.22)
        fig.tight_layout()
        return fig, axis


def plot_qq(
    residuals,
    figsize: Optional[Tuple[float, float]] = None,
    *,
    theme: str = "default",
    ax: Optional[plt.Axes] = None,
    title: str = "Normal Q–Q plot",
    **kwargs,
):
    """Create a normal Q–Q plot for residuals."""
    del kwargs
    resid = as_1d_finite(residuals, "residuals")
    with plotting_theme(theme):
        fig, axis = _new_axis(ax, figsize, theme)
        stats.probplot(resid, dist="norm", plot=axis)
        axis.set_title(title)
        axis.grid(True, alpha=0.22)
        fig.tight_layout()
        return fig, axis


def plot_spatial_residuals(
    coords,
    residuals,
    cmap: str = "RdBu_r",
    figsize: Optional[Tuple[float, float]] = None,
    *,
    theme: str = "default",
    ax: Optional[plt.Axes] = None,
    title: str = "Spatial residual pattern",
    **kwargs,
):
    """Map residuals using calibration point coordinates."""
    resid = as_1d_finite(residuals, "residuals")
    coords_arr = validate_coords(coords, resid.size)
    extent = max(float(np.max(np.abs(resid))), np.finfo(float).eps)
    with plotting_theme(theme):
        fig, axis = _new_axis(ax, figsize, theme)
        scatter = axis.scatter(
            coords_arr[:, 0],
            coords_arr[:, 1],
            c=resid,
            cmap=cmap,
            vmin=-extent,
            vmax=extent,
            s=kwargs.pop("s", 45.0),
            alpha=kwargs.pop("alpha", 0.9),
            edgecolors=kwargs.pop("edgecolors", "0.2"),
            linewidths=kwargs.pop("linewidths", 0.35),
            **kwargs,
        )
        fig.colorbar(scatter, ax=axis, fraction=0.046, pad=0.04, label="Residual")
        axis.set_xlabel("X coordinate")
        axis.set_ylabel("Y coordinate")
        axis.set_title(title)
        axis.set_aspect("equal", adjustable="datalim")
        fig.tight_layout()
        return fig, axis


def plot_observed_vs_predicted(
    y_true,
    y_pred,
    figsize: Optional[Tuple[float, float]] = None,
    *,
    theme: str = "default",
    ax: Optional[plt.Axes] = None,
    title: str = "Observed vs predicted",
    **kwargs,
):
    """Plot observed and predicted values with a one-to-one reference."""
    observed = as_1d_finite(y_true, "y_true")
    predicted = as_1d_finite(y_pred, "y_pred")
    if observed.size != predicted.size:
        raise ValueError("y_true and y_pred must have the same length.")
    lower = float(min(np.min(observed), np.min(predicted)))
    upper = float(max(np.max(observed), np.max(predicted)))
    if np.isclose(lower, upper):
        lower, upper = lower - 0.5, upper + 0.5
    with plotting_theme(theme):
        fig, axis = _new_axis(ax, figsize, theme)
        axis.scatter(observed, predicted, alpha=kwargs.pop("alpha", 0.65), **kwargs)
        axis.plot(
            [lower, upper], [lower, upper], color="0.25", linestyle="--", linewidth=1.2
        )
        r2 = compute_r_squared(observed, predicted)
        axis.text(
            0.04,
            0.96,
            f"R² = {r2:.4f}",
            transform=axis.transAxes,
            va="top",
            bbox={"facecolor": "white", "edgecolor": "0.5", "alpha": 0.85},
        )
        axis.set_xlabel("Observed")
        axis.set_ylabel("Predicted")
        axis.set_title(title)
        axis.set_aspect("equal", adjustable="box")
        axis.grid(True, alpha=0.22)
        fig.tight_layout()
        return fig, axis


def plot_bandwidth_selection(
    bandwidths,
    scores,
    selected_bandwidth,
    criterion: str = "CV",
    figsize: Optional[Tuple[float, float]] = None,
    *,
    theme: str = "default",
    ax: Optional[plt.Axes] = None,
    **kwargs,
):
    """Compatibility wrapper for :func:`pygwrx.plotting.bandwidth.plot_bandwidth_selection`."""
    from pygwrx.plotting.bandwidth import plot_bandwidth_selection as implementation

    return implementation(
        bandwidths,
        scores,
        selected_bandwidth,
        criterion=criterion,
        figsize=figsize,
        theme=theme,
        ax=ax,
        title=kwargs.pop("title", None),
    )


def plot_coefficient_variability(
    coefficients,
    feature_names: Optional[Sequence[str]] = None,
    figsize: Optional[Tuple[float, float]] = None,
    *,
    theme: str = "default",
    ax: Optional[plt.Axes] = None,
    kind: str = "box",
    global_coefficients: Optional[Sequence[float]] = None,
    **kwargs,
):
    """Compare distributions of local coefficients across variables."""
    array = np.asarray(coefficients, dtype=float)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError("coefficients must be a non-empty two-dimensional array.")
    if not np.all(np.isfinite(array)):
        raise ValueError("coefficients contains NaN or infinite values.")
    names = (
        [f"x{index}" for index in range(array.shape[1])]
        if feature_names is None
        else [str(name) for name in feature_names]
    )
    if len(names) != array.shape[1]:
        raise ValueError("feature_names must contain one label per coefficient column.")
    mode = str(kind).strip().lower()
    if mode not in {"box", "violin"}:
        raise ValueError("kind must be 'box' or 'violin'.")
    with plotting_theme(theme):
        fig, axis = _new_axis(ax, figsize, theme, wide=True)
        data = [array[:, index] for index in range(array.shape[1])]
        if mode == "box":
            version_parts = matplotlib.__version__.split(".")
            version = (int(version_parts[0]), int(version_parts[1]))
            label_keyword = "tick_labels" if version >= (3, 9) else "labels"
            boxplot_kwargs = {label_keyword: names}
            boxplot_kwargs.update(kwargs)
            axis.boxplot(
                data,
                showfliers=boxplot_kwargs.pop("showfliers", False),
                **boxplot_kwargs,
            )
        else:
            axis.violinplot(data, showmeans=True, showextrema=True, **kwargs)
            axis.set_xticks(np.arange(1, len(names) + 1), labels=names)
        if global_coefficients is not None:
            global_array = as_1d_finite(global_coefficients, "global_coefficients")
            if global_array.size != array.shape[1]:
                raise ValueError(
                    "global_coefficients must contain one value per column."
                )
            axis.scatter(
                np.arange(1, len(names) + 1),
                global_array,
                marker="D",
                color="0.15",
                label="Global coefficient",
            )
            axis.legend(loc="best")
        axis.axhline(0.0, color="0.45", linestyle="--", linewidth=0.8)
        axis.set_ylabel("Local coefficient")
        axis.set_title("Local coefficient variability")
        axis.tick_params(axis="x", rotation=30)
        axis.grid(True, axis="y", alpha=0.22)
        fig.tight_layout()
        return fig, axis


def plot_diagnostic_panel(
    y_true,
    y_pred=None,
    residuals=None,
    coords=None,
    figsize: Tuple[float, float] = (14, 9),
    *,
    theme: str = "default",
):
    """Create a complete calibration-diagnostic panel without displaying it."""
    if hasattr(y_true, "coef_") and y_pred is None and residuals is None:
        model = y_true
        observed_source = getattr(model, "y_train_", None)
        predicted_source = getattr(model, "fitted_values_", None)
        residual_source = getattr(model, "residuals_", None)
        if (
            observed_source is None
            or predicted_source is None
            or residual_source is None
        ):
            raise ValueError(
                "The fitted model does not expose calibration diagnostics."
            )
        if coords is None:
            coords = getattr(model, "coords_train_", None)
        y_true, y_pred, residuals = observed_source, predicted_source, residual_source
    if y_pred is None or residuals is None:
        raise TypeError(
            "Provide a fitted model or y_true, y_pred, and residuals arrays."
        )
    observed = as_1d_finite(y_true, "y_true")
    predicted = as_1d_finite(y_pred, "y_pred")
    resid = as_1d_finite(residuals, "residuals")
    if not (observed.size == predicted.size == resid.size):
        raise ValueError("y_true, y_pred, and residuals must have the same length.")
    coords_arr = None if coords is None else validate_coords(coords, observed.size)
    with plotting_theme(theme):
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        plot_observed_vs_predicted(observed, predicted, theme=theme, ax=axes[0, 0])
        plot_residuals(predicted, resid, theme=theme, ax=axes[0, 1])
        plot_qq(resid, theme=theme, ax=axes[0, 2])
        plot_residual_histogram(resid, theme=theme, ax=axes[1, 0])
        scale = np.sqrt(np.abs(resid - np.mean(resid)))
        axes[1, 1].scatter(predicted, scale, alpha=0.65)
        axes[1, 1].set_xlabel("Fitted value")
        axes[1, 1].set_ylabel("√|centred residual|")
        axes[1, 1].set_title("Scale–location")
        axes[1, 1].grid(True, alpha=0.22)
        if coords_arr is None:
            axes[1, 2].axis("off")
            axes[1, 2].text(
                0.5, 0.5, "Coordinates not supplied", ha="center", va="center"
            )
        else:
            plot_spatial_residuals(coords_arr, resid, theme=theme, ax=axes[1, 2])
        fig.tight_layout()
        return fig, axes


def plot_local_diagnostics(
    coords,
    diagnostics: Mapping[str, Sequence[float]],
    figsize: Optional[Tuple[float, float]] = None,
    *,
    theme: str = "default",
    ncols: int = 3,
):
    """Plot several local diagnostic arrays on a common coordinate set."""
    if not isinstance(diagnostics, Mapping) or not diagnostics:
        raise ValueError("diagnostics must be a non-empty mapping.")
    first = as_1d_finite(next(iter(diagnostics.values())), "diagnostic", allow_nan=True)
    coords_arr = validate_coords(coords, first.size)
    if isinstance(ncols, (bool, np.bool_)) or int(ncols) < 1:
        raise ValueError("ncols must be a positive integer.")
    columns = int(ncols)
    rows = int(np.ceil(len(diagnostics) / columns))
    with plotting_theme(theme):
        fig, axes = plt.subplots(
            rows,
            columns,
            figsize=figsize or (4.6 * columns, 4.0 * rows),
            squeeze=False,
        )
        for axis, (name, values) in zip(axes.flat, diagnostics.items()):
            array = as_1d_finite(values, str(name), allow_nan=True)
            if array.size != first.size:
                raise ValueError("Every diagnostic array must match coords length.")
            finite = np.isfinite(array)
            scatter = axis.scatter(
                coords_arr[finite, 0],
                coords_arr[finite, 1],
                c=array[finite],
                cmap="viridis",
                s=40,
                edgecolors="0.2",
                linewidths=0.3,
            )
            fig.colorbar(scatter, ax=axis, fraction=0.046, pad=0.04)
            axis.set_title(str(name).replace("_", " ").title())
            axis.set_aspect("equal", adjustable="datalim")
            axis.set_xlabel("X coordinate")
            axis.set_ylabel("Y coordinate")
        for axis in list(axes.flat)[len(diagnostics) :]:
            axis.set_visible(False)
        fig.tight_layout()
        return fig, axes
