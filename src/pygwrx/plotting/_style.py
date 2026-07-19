# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Shared plotting styles and colour normalization.

Author:
    Jinghao Hu
"""

from __future__ import annotations

__author__ = "Jinghao Hu"
__license__ = "MIT"

from contextlib import contextmanager
from typing import Dict, Iterator, Mapping, Optional, Tuple

import matplotlib as mpl
import numpy as np
from matplotlib.colors import Normalize, TwoSlopeNorm

_THEMES: Dict[str, Mapping[str, object]] = {
    "default": {
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "semibold",
        "axes.labelsize": 10,
        "axes.titlesize": 12,
        "legend.frameon": False,
    },
    "paper": {
        "figure.dpi": 120,
        "savefig.dpi": 600,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "normal",
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    },
    "presentation": {
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "font.size": 12,
        "axes.labelsize": 12,
        "axes.titlesize": 14,
        "axes.titleweight": "semibold",
        "legend.fontsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
    },
}


@contextmanager
def plotting_theme(theme: str = "default") -> Iterator[None]:
    """Apply a temporary pyGWRx plotting theme.

    Args:
        theme: One of ``"default"``, ``"paper"``, or ``"presentation"``.

    Yields:
        A temporary Matplotlib style context.
    """
    key = str(theme).strip().lower()
    if key not in _THEMES:
        raise ValueError(f"Unknown theme {theme!r}. Choose from {sorted(_THEMES)}.")
    with mpl.rc_context(_THEMES[key]):
        yield


def finite_range(values: np.ndarray) -> Tuple[float, float]:
    """Return a finite plotting range, including for constant arrays."""
    array = np.asarray(values, dtype=float).reshape(-1)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        raise ValueError("Plot values contain no finite observations.")
    lower = float(np.min(finite))
    upper = float(np.max(finite))
    if np.isclose(lower, upper):
        pad = max(abs(lower) * 0.05, 1.0e-9)
        lower -= pad
        upper += pad
    return lower, upper


def resolve_color_scale(
    values: np.ndarray,
    *,
    center_zero: Optional[bool] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap: Optional[str] = None,
) -> Tuple[str, Normalize]:
    """Choose a semantically appropriate colormap and normalization."""
    lower, upper = finite_range(values)
    lower = lower if vmin is None else float(vmin)
    upper = upper if vmax is None else float(vmax)
    if lower >= upper:
        raise ValueError("vmin must be smaller than vmax.")

    use_zero = lower < 0.0 < upper if center_zero is None else bool(center_zero)
    if use_zero:
        extent = max(abs(lower), abs(upper))
        lower = -extent if vmin is None else lower
        upper = extent if vmax is None else upper
        if not lower < 0.0 < upper:
            raise ValueError("A zero-centred scale requires vmin < 0 < vmax.")
        return cmap or "RdBu_r", TwoSlopeNorm(vmin=lower, vcenter=0.0, vmax=upper)
    return cmap or "viridis", Normalize(vmin=lower, vmax=upper)


def default_figure_size(theme: str, *, wide: bool = False) -> Tuple[float, float]:
    key = str(theme).strip().lower()
    if key == "paper":
        return (7.2, 4.2) if wide else (4.6, 4.0)
    if key == "presentation":
        return (12.0, 7.0) if wide else (8.0, 6.0)
    return (10.0, 5.8) if wide else (7.5, 6.0)
