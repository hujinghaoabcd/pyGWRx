# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Array-based spatial maps retained for backward compatibility.

New code should prefer the model-aware functions in
:mod:`pygwrx.plotting.surfaces`.

Author:
    Jinghao Hu
"""

from __future__ import annotations

__author__ = "Jinghao Hu"
__license__ = "MIT"

from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Circle, Patch
from scipy.interpolate import griddata

from pygwrx._optional import import_required_dependency

if TYPE_CHECKING:
    import geopandas as gpd

from pygwrx.plotting._spatial import add_colorbar, render_spatial_values
from pygwrx.plotting._style import (
    default_figure_size,
    plotting_theme,
    resolve_color_scale,
)
from pygwrx.plotting._validation import as_1d_finite, validate_coords


def plot_local_coefficients(
    coords,
    coefficients,
    feature_idx: int = 0,
    feature_name: Optional[str] = None,
    cmap: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
    basemap: Optional[gpd.GeoDataFrame] = None,
    *,
    theme: str = "default",
    ax: Optional[plt.Axes] = None,
    **kwargs,
):
    """Plot one column of a local coefficient array."""
    array = np.asarray(coefficients, dtype=float)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if array.ndim != 2:
        raise ValueError("coefficients must be one- or two-dimensional.")
    index = int(feature_idx)
    if index < 0 or index >= array.shape[1]:
        raise IndexError(f"feature_idx must lie in [0, {array.shape[1] - 1}].")
    coords_arr = validate_coords(coords, array.shape[0])
    values = as_1d_finite(array[:, index], "coefficients")
    label = feature_name or f"Feature {index}"
    with plotting_theme(theme):
        if ax is None:
            fig, axis = plt.subplots(figsize=figsize or default_figure_size(theme))
        else:
            fig, axis = ax.figure, ax
        if basemap is not None:
            basemap.plot(ax=axis, color="0.92", edgecolor="0.55", linewidth=0.4)
        cmap_name, norm = resolve_color_scale(values, center_zero=None, cmap=cmap)
        artist = render_spatial_values(
            axis,
            values,
            coords=coords_arr,
            cmap=cmap_name,
            norm=norm,
            marker_size=kwargs.pop("s", 45.0),
            alpha=kwargs.pop("alpha", 0.9),
            edgecolor=kwargs.pop("edgecolors", "0.2"),
            linewidth=kwargs.pop("linewidths", 0.35),
        )
        add_colorbar(fig, axis, artist, f"Coefficient: {label}")
        axis.set_title(title or f"Local coefficients: {label}")
        fig.tight_layout()
        return fig, axis


def plot_coefficient_surface(
    coords,
    coefficients,
    feature_idx: int = 0,
    method: str = "contourf",
    n_levels: int = 20,
    figsize: Optional[Tuple[float, float]] = None,
    *,
    theme: str = "default",
    ax: Optional[plt.Axes] = None,
    grid_size: int = 100,
    interpolation: str = "linear",
    cmap: Optional[str] = None,
    **kwargs,
):
    """Interpolate a local coefficient array to a regular plotting grid."""
    array = np.asarray(coefficients, dtype=float)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if array.ndim != 2:
        raise ValueError("coefficients must be one- or two-dimensional.")
    index = int(feature_idx)
    if index < 0 or index >= array.shape[1]:
        raise IndexError(f"feature_idx must lie in [0, {array.shape[1] - 1}].")
    coords_arr = validate_coords(coords, array.shape[0])
    values = as_1d_finite(array[:, index], "coefficients")
    mode = str(method).strip().lower()
    if mode not in {"contour", "contourf", "surface"}:
        raise ValueError("method must be 'contour', 'contourf', or 'surface'.")
    if int(grid_size) < 10:
        raise ValueError("grid_size must be at least 10.")
    x_grid = np.linspace(coords_arr[:, 0].min(), coords_arr[:, 0].max(), int(grid_size))
    y_grid = np.linspace(coords_arr[:, 1].min(), coords_arr[:, 1].max(), int(grid_size))
    grid_x, grid_y = np.meshgrid(x_grid, y_grid)
    grid_values = griddata(coords_arr, values, (grid_x, grid_y), method=interpolation)
    if not np.any(np.isfinite(grid_values)):
        raise ValueError(
            "Interpolation produced no finite surface; try interpolation='nearest'."
        )
    cmap_name, norm = resolve_color_scale(values, center_zero=None, cmap=cmap)
    with plotting_theme(theme):
        if ax is None:
            if mode == "surface":
                fig = plt.figure(figsize=figsize or default_figure_size(theme))
                axis = fig.add_subplot(111, projection="3d")
            else:
                fig, axis = plt.subplots(figsize=figsize or default_figure_size(theme))
        else:
            fig, axis = ax.figure, ax
        if mode == "surface":
            artist = axis.plot_surface(
                grid_x,
                grid_y,
                grid_values,
                cmap=cmap_name,
                norm=norm,
                linewidth=0.0,
                **kwargs,
            )
            axis.set_zlabel("Coefficient")
        elif mode == "contour":
            artist = axis.contour(
                grid_x,
                grid_y,
                grid_values,
                levels=int(n_levels),
                cmap=cmap_name,
                norm=norm,
                **kwargs,
            )
        else:
            artist = axis.contourf(
                grid_x,
                grid_y,
                grid_values,
                levels=int(n_levels),
                cmap=cmap_name,
                norm=norm,
                **kwargs,
            )
        fig.colorbar(artist, ax=axis, fraction=0.046, pad=0.04, label="Coefficient")
        axis.scatter(coords_arr[:, 0], coords_arr[:, 1], s=8, c="0.2", alpha=0.35)
        axis.set_xlabel("X coordinate")
        axis.set_ylabel("Y coordinate")
        axis.set_title(f"Interpolated coefficient surface: feature {index}")
        fig.tight_layout()
        return fig, axis


def plot_significance_map(
    coords,
    p_values,
    alpha: float = 0.05,
    feature_idx: int = 0,
    figsize: Optional[Tuple[float, float]] = None,
    *,
    coefficients=None,
    theme: str = "default",
    ax: Optional[plt.Axes] = None,
    **kwargs,
):
    """Plot significant and non-significant locations from local p-values."""
    p_array = np.asarray(p_values, dtype=float)
    if p_array.ndim == 1:
        p_array = p_array.reshape(-1, 1)
    if p_array.ndim != 2:
        raise ValueError("p_values must be one- or two-dimensional.")
    index = int(feature_idx)
    if index < 0 or index >= p_array.shape[1]:
        raise IndexError(f"feature_idx must lie in [0, {p_array.shape[1] - 1}].")
    values = as_1d_finite(p_array[:, index], "p_values")
    coords_arr = validate_coords(coords, values.size)
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError("alpha must lie strictly between 0 and 1.")
    significant = values <= float(alpha)
    categories = significant.astype(float)
    labels = ["Not significant", "Significant"]
    colors = ["#d9d9d9", "#2c7fb8"]
    if coefficients is not None:
        coef_array = np.asarray(coefficients, dtype=float)
        if coef_array.ndim == 2:
            coef_array = coef_array[:, index]
        coef_array = as_1d_finite(coef_array, "coefficients")
        if coef_array.size != values.size:
            raise ValueError("coefficients must match p_values length.")
        categories = np.zeros(values.size, dtype=float)
        categories[significant & (coef_array < 0.0)] = -1.0
        categories[significant & (coef_array > 0.0)] = 1.0
        labels = ["Significant negative", "Not significant", "Significant positive"]
        colors = ["#4575b4", "#d9d9d9", "#d73027"]
        boundaries = [-1.5, -0.5, 0.5, 1.5]
    else:
        boundaries = [-0.5, 0.5, 1.5]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(boundaries, cmap.N)
    with plotting_theme(theme):
        if ax is None:
            fig, axis = plt.subplots(figsize=figsize or default_figure_size(theme))
        else:
            fig, axis = ax.figure, ax
        render_spatial_values(
            axis,
            categories,
            coords=coords_arr,
            cmap=cmap,
            norm=norm,
            marker_size=kwargs.pop("s", 45.0),
        )
        axis.legend(
            handles=[
                Patch(facecolor=color, label=label)
                for color, label in zip(colors, labels)
            ],
            loc="best",
        )
        axis.set_title(f"Local coefficient significance (α={alpha:g})")
        fig.tight_layout()
        return fig, axis


def plot_local_r2(
    coords,
    local_r2,
    cmap: str = "YlOrRd",
    figsize: Optional[Tuple[float, float]] = None,
    *,
    theme: str = "default",
    ax: Optional[plt.Axes] = None,
    **kwargs,
):
    """Plot spatial local R² values."""
    values = as_1d_finite(local_r2, "local_r2", allow_nan=True)
    coords_arr = validate_coords(coords, values.size)
    with plotting_theme(theme):
        if ax is None:
            fig, axis = plt.subplots(figsize=figsize or default_figure_size(theme))
        else:
            fig, axis = ax.figure, ax
        cmap_name, norm = resolve_color_scale(
            values,
            center_zero=False,
            vmin=kwargs.pop("vmin", None),
            vmax=kwargs.pop("vmax", None),
            cmap=cmap,
        )
        artist = render_spatial_values(
            axis,
            values,
            coords=coords_arr,
            cmap=cmap_name,
            norm=norm,
            marker_size=kwargs.pop("s", 45.0),
        )
        add_colorbar(fig, axis, artist, "Local R²")
        axis.set_title("Local R²")
        fig.tight_layout()
        return fig, axis


def plot_bandwidth(
    coords,
    bandwidth: Union[float, np.ndarray],
    kernel: str = "gaussian",
    sample_locations=None,
    figsize: Optional[Tuple[float, float]] = None,
    *,
    theme: str = "default",
    ax: Optional[plt.Axes] = None,
    **kwargs,
):
    """Visualize fixed-distance bandwidth footprints at selected locations."""
    coords_arr = validate_coords(coords)
    locations = (
        coords_arr[: min(5, coords_arr.shape[0])]
        if sample_locations is None
        else validate_coords(sample_locations)
    )
    bw = np.asarray(bandwidth, dtype=float).reshape(-1)
    if bw.size == 1:
        bw = np.repeat(bw, locations.shape[0])
    if bw.size != locations.shape[0]:
        raise ValueError(
            "bandwidth must be scalar or contain one value per sample location."
        )
    if not np.all(np.isfinite(bw)) or np.any(bw <= 0.0):
        raise ValueError("bandwidth values must be finite and positive.")
    with plotting_theme(theme):
        if ax is None:
            fig, axis = plt.subplots(figsize=figsize or default_figure_size(theme))
        else:
            fig, axis = ax.figure, ax
        axis.scatter(
            coords_arr[:, 0],
            coords_arr[:, 1],
            s=kwargs.pop("s", 24.0),
            color="0.55",
            alpha=0.65,
        )
        for location, radius in zip(locations, bw):
            axis.add_patch(
                Circle(location, float(radius), fill=False, linewidth=1.0, alpha=0.75)
            )
            axis.scatter(location[0], location[1], marker="x", color="0.15")
        axis.set_aspect("equal", adjustable="datalim")
        axis.set_xlabel("X coordinate")
        axis.set_ylabel("Y coordinate")
        axis.set_title(f"{kernel} bandwidth footprints")
        fig.tight_layout()
        return fig, axis


def create_choropleth(
    gdf: "gpd.GeoDataFrame",
    column: str,
    cmap: str = "viridis",
    legend: bool = True,
    figsize: Optional[Tuple[float, float]] = None,
    *,
    theme: str = "default",
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    **kwargs,
):
    """Create a validated GeoDataFrame choropleth."""
    geopandas = import_required_dependency(
        "geopandas", purpose="GeoDataFrame choropleths"
    )
    if not isinstance(gdf, geopandas.GeoDataFrame):
        raise TypeError("gdf must be a GeoDataFrame.")
    if column not in gdf.columns:
        raise ValueError(f"column {column!r} is not present in gdf.")
    with plotting_theme(theme):
        if ax is None:
            fig, axis = plt.subplots(figsize=figsize or default_figure_size(theme))
        else:
            fig, axis = ax.figure, ax
        gdf.plot(column=column, cmap=cmap, legend=legend, ax=axis, **kwargs)
        axis.set_title(title or str(column).replace("_", " ").title())
        axis.set_axis_off()
        fig.tight_layout()
        return fig, axis


def plot_multiple_coefficients(
    coords,
    coefficients,
    feature_names: Optional[List[str]] = None,
    ncols: int = 2,
    figsize: Optional[Tuple[float, float]] = None,
    *,
    theme: str = "default",
    shared_scale: bool = False,
    **kwargs,
):
    """Create a panel containing every coefficient column."""
    array = np.asarray(coefficients, dtype=float)
    if array.ndim != 2 or array.shape[1] == 0:
        raise ValueError("coefficients must be a non-empty two-dimensional array.")
    coords_arr = validate_coords(coords, array.shape[0])
    names = (
        [f"x{index}" for index in range(array.shape[1])]
        if feature_names is None
        else [str(name) for name in feature_names]
    )
    if len(names) != array.shape[1]:
        raise ValueError("feature_names must contain one name per coefficient column.")
    columns = int(ncols)
    if columns < 1:
        raise ValueError("ncols must be positive.")
    rows = int(np.ceil(array.shape[1] / columns))
    combined = array[np.isfinite(array)]
    vmin = float(np.min(combined)) if shared_scale else None
    vmax = float(np.max(combined)) if shared_scale else None
    if shared_scale and vmin < 0.0 < vmax:
        extent = max(abs(vmin), abs(vmax))
        vmin, vmax = -extent, extent
    with plotting_theme(theme):
        fig, axes = plt.subplots(
            rows, columns, figsize=figsize or (4.6 * columns, 4.0 * rows), squeeze=False
        )
        for index, axis in enumerate(axes.flat[: array.shape[1]]):
            values = array[:, index]
            cmap_name, norm = resolve_color_scale(
                values, center_zero=None, vmin=vmin, vmax=vmax, cmap=kwargs.get("cmap")
            )
            artist = render_spatial_values(
                axis,
                values,
                coords=coords_arr,
                cmap=cmap_name,
                norm=norm,
                marker_size=kwargs.get("s", 40.0),
            )
            add_colorbar(fig, axis, artist, names[index])
            axis.set_title(names[index])
        for axis in list(axes.flat)[array.shape[1] :]:
            axis.set_visible(False)
        fig.tight_layout()
        return fig, axes
