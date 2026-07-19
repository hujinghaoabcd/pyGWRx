# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Interpretive plots for the published scalable GWR estimator.

Author:
    Jinghao Hu
"""

from __future__ import annotations

__author__ = "Jinghao Hu"
__license__ = "MIT"

from typing import Any, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from pygwrx.plotting._style import plotting_theme


def plot_scalable_gwr_kernel(
    model: Any,
    *,
    max_distance: Optional[float] = None,
    n_points: int = 200,
    theme: str = "default",
    figsize: Optional[Tuple[float, float]] = None,
    title: str = "Scalable GWR multiscale kernel approximation",
) -> Tuple[plt.Figure, np.ndarray]:
    """Plot fitted polynomial-kernel components and their mixture.

    Args:
        model: Fitted :class:`~pygwrx.models.ScalableGWR` estimator.
        max_distance: Largest displayed distance. By default, three fitted base
            bandwidths are shown.
        n_points: Number of distance samples.
        theme: Plotting theme.
        figsize: Optional figure size.
        title: Figure title.

    Returns:
        Matplotlib ``(figure, axes)``.
    """
    if not bool(getattr(model, "_is_fitted", False)):
        raise ValueError("ScalableGWR must be fitted before plotting its kernel.")
    base_bandwidth = float(getattr(model, "base_bandwidth_", np.nan))
    scale = float(getattr(model, "scale_", np.nan))
    penalty = float(getattr(model, "penalty_", np.nan))
    degree = int(getattr(model, "polynomial", 0))
    if (
        not np.isfinite(base_bandwidth)
        or base_bandwidth <= 0.0
        or not np.isfinite(scale)
        or scale <= 0.0
        or degree < 1
    ):
        raise ValueError(
            "The fitted model does not expose valid ScaGWR kernel parameters."
        )
    limit = 3.0 * base_bandwidth if max_distance is None else float(max_distance)
    if not np.isfinite(limit) or limit <= 0.0:
        raise ValueError("max_distance must be finite and positive.")
    count = int(n_points)
    if count < 20:
        raise ValueError("n_points must be at least 20.")

    distances = np.linspace(0.0, limit, count)
    if str(getattr(model, "kernel", "gaussian")) == "gaussian":
        base = np.exp(-np.square(distances / base_bandwidth))
    else:
        base = np.exp(-distances / base_bandwidth)
    basis = np.ones((count, degree + 1), dtype=float)
    numerator = 2.0 ** (degree / 2.0)
    for index in range(1, degree + 1):
        exponent = numerator / (2.0**index)
        basis[:, index] = np.power(base, exponent)
    powers = np.arange(1, degree + 2, dtype=float)
    logits = powers * np.log(scale)
    logits -= np.max(logits)
    coefficients = np.exp(logits)
    coefficients /= np.sum(coefficients)
    mixture = basis @ coefficients

    with plotting_theme(theme):
        fig, axes = plt.subplots(
            1,
            2,
            figsize=figsize or (10.5, 4.3),
            constrained_layout=True,
        )
        for index in range(basis.shape[1]):
            axes[0].plot(
                distances,
                basis[:, index],
                linewidth=0.9,
                alpha=0.45,
                label=f"Basis {index}",
            )
        axes[0].plot(distances, mixture, linewidth=2.2, label="Fitted mixture")
        axes[0].set_xlabel("Distance")
        axes[0].set_ylabel("Kernel value")
        axes[0].set_title("Effective kernel")
        axes[0].grid(True, alpha=0.22)
        axes[0].legend(loc="best", ncol=2)

        labels = [f"Basis {index}" for index in range(coefficients.size)]
        axes[1].bar(labels, coefficients)
        axes[1].set_ylim(0.0, max(1.0, float(np.max(coefficients)) * 1.12))
        axes[1].set_ylabel("Mixture coefficient")
        axes[1].set_title("Polynomial mixture")
        axes[1].tick_params(axis="x", rotation=30)
        axes[1].grid(True, axis="y", alpha=0.22)
        diagnostics = getattr(model, "diagnostics_", None) or {}
        cv_rmse = diagnostics.get("cv_rmse", np.nan)
        text = (
            f"Q={int(getattr(model, 'bandwidth_', getattr(model, 'bandwidth', 0)))}\n"
            f"scale={scale:.4g}\n"
            f"penalty={penalty:.4g}\n"
            f"CV RMSE={float(cv_rmse):.4g}"
        )
        axes[1].text(
            0.98,
            0.98,
            text,
            transform=axes[1].transAxes,
            ha="right",
            va="top",
            bbox={"facecolor": "white", "edgecolor": "0.6", "alpha": 0.88},
        )
        fig.suptitle(title)
        return fig, axes
