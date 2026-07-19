# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Low-level point and polygon rendering shared by public map functions.

Author:
    Jinghao Hu
"""

from __future__ import annotations

__author__ = "Jinghao Hu"
__license__ = "MIT"

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from pygwrx.plotting._validation import validate_coords, validate_geometry


def render_spatial_values(
    ax: plt.Axes,
    values: np.ndarray,
    *,
    coords: np.ndarray,
    geometry=None,
    cmap: str,
    norm: Normalize,
    mask: Optional[np.ndarray] = None,
    nonsignificant_color: str = "0.82",
    marker_size: float = 45.0,
    edgecolor: str = "0.2",
    linewidth: float = 0.35,
    alpha: float = 0.9,
):
    values_arr = np.asarray(values, dtype=float).reshape(-1)
    coords_arr = validate_coords(coords, values_arr.size)
    geo = validate_geometry(geometry, values_arr.size)
    finite = np.isfinite(values_arr)
    active = (
        finite if mask is None else finite & np.asarray(mask, dtype=bool).reshape(-1)
    )
    if active.size != values_arr.size:
        raise ValueError("mask must contain one value per observation.")

    if geo is None:
        if mask is not None:
            ax.scatter(
                coords_arr[~active, 0],
                coords_arr[~active, 1],
                c=nonsignificant_color,
                s=marker_size,
                edgecolors=edgecolor,
                linewidths=linewidth,
                alpha=0.75,
                zorder=1,
            )
        artist = ax.scatter(
            coords_arr[active, 0],
            coords_arr[active, 1],
            c=values_arr[active],
            cmap=cmap,
            norm=norm,
            s=marker_size,
            edgecolors=edgecolor,
            linewidths=linewidth,
            alpha=alpha,
            zorder=2,
        )
    else:
        if mask is not None and np.any(~active):
            geo[~active].plot(
                ax=ax,
                color=nonsignificant_color,
                edgecolor=edgecolor,
                linewidth=linewidth,
                alpha=0.8,
            )
        if np.any(active):
            geo[active].plot(
                ax=ax,
                color=plt.get_cmap(cmap)(norm(values_arr[active])),
                edgecolor=edgecolor,
                linewidth=linewidth,
                alpha=alpha,
            )
        artist = ScalarMappable(norm=norm, cmap=cmap)
        artist.set_array(values_arr[finite])

    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")
    return artist


def add_colorbar(fig, ax, artist, label: str):
    colorbar = fig.colorbar(artist, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label(label)
    return colorbar
