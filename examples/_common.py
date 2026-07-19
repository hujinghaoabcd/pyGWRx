# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Shared deterministic datasets and helpers for the pyGWRx example suite."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

EXAMPLES_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = EXAMPLES_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


def spatial_regression(n: int = 48, p: int = 3, seed: int = 42):
    """Return a small spatially non-stationary regression dataset."""
    rng = np.random.default_rng(seed)
    coords = rng.uniform(0.0, 10.0, size=(n, 2))
    X = rng.normal(size=(n, p))
    beta = np.tile(np.linspace(1.2, -0.4, p), (n, 1))
    beta[:, 0] += 0.08 * coords[:, 0]
    y = 1.5 + np.sum(X * beta, axis=1) + rng.normal(0.0, 0.12, size=n)
    frame = pd.DataFrame(X, columns=[f"x{i + 1}" for i in range(p)])
    coord_frame = pd.DataFrame(coords, columns=["east", "north"])
    return frame, y, coord_frame


def temporal_regression(n: int = 48, p: int = 2, seed: int = 7):
    """Return regression data with repeated time groups."""
    X, y, coords = spatial_regression(n=n, p=p, seed=seed)
    groups = 4
    times = np.repeat(np.arange(groups, dtype=float), n // groups)
    if times.size < n:
        times = np.pad(times, (0, n - times.size), mode="edge")
    y = y + 0.15 * times
    return X, y, coords, times


def stwr_stages(n_per_stage: int = 16, seed: int = 11):
    """Return three snapshots for STWR."""
    rng = np.random.default_rng(seed)
    x_axis = np.linspace(0.0, 6.0, n_per_stage)
    base_coords = np.column_stack((x_axis, 0.25 * np.sin(x_axis)))
    X_list, y_list, coords_list = [], [], []
    for stage in range(3):
        x1 = np.linspace(-1.0, 1.0, n_per_stage)
        x2 = rng.normal(scale=0.7, size=n_per_stage)
        X = pd.DataFrame({"trend": x1, "noise": x2})
        y = 2.0 + 0.2 * stage + (1.1 + 0.1 * stage) * x1 - 0.45 * x2
        y += rng.normal(scale=0.025, size=n_per_stage)
        X_list.append(X)
        y_list.append(y)
        coords_list.append(pd.DataFrame(base_coords, columns=["east", "north"]))
    return X_list, y_list, coords_list, [0.0, 1.0, 1.5]


def count_regression(n: int = 48, seed: int = 13):
    """Return a Poisson count dataset with exposure."""
    rng = np.random.default_rng(seed)
    coords = rng.uniform(0.0, 8.0, size=(n, 2))
    X = pd.DataFrame(rng.normal(size=(n, 2)), columns=["income", "access"])
    exposure = rng.uniform(0.8, 2.0, size=n)
    eta = 0.2 + 0.35 * X["income"].to_numpy() - 0.25 * X["access"].to_numpy()
    eta += 0.04 * coords[:, 0]
    y = rng.poisson(exposure * np.exp(eta))
    return X, y, pd.DataFrame(coords, columns=["east", "north"]), exposure


def classification_data(n: int = 60, seed: int = 19):
    """Return a two-class spatial classification dataset."""
    rng = np.random.default_rng(seed)
    coords = rng.uniform(0.0, 10.0, size=(n, 2))
    X = pd.DataFrame(rng.normal(size=(n, 3)), columns=["a", "b", "c"])
    score = X["a"].to_numpy() - 0.7 * X["b"].to_numpy() + 0.1 * coords[:, 0]
    y = np.where(score >= np.median(score), "high", "low")
    return X, y, pd.DataFrame(coords, columns=["east", "north"])


def collinear_regression(n: int = 54, seed: int = 23):
    """Return data with strong local predictor collinearity."""
    rng = np.random.default_rng(seed)
    coords = rng.uniform(0.0, 10.0, size=(n, 2))
    x1 = rng.normal(size=n)
    x2 = x1 + rng.normal(0.0, 0.025, size=n)
    x3 = rng.normal(size=n)
    X = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3})
    y = 1.0 + 1.8 * x1 - 1.4 * x2 + 0.4 * x3 + rng.normal(0.0, 0.08, n)
    return X, y, pd.DataFrame(coords, columns=["east", "north"])


def mixed_regression(n: int = 72, seed: int = 29):
    """Return data with one global and one local predictor."""
    rng = np.random.default_rng(seed)
    coords = rng.uniform(0.0, 10.0, size=(n, 2))
    global_x = rng.normal(size=n)
    local_x = rng.normal(size=n)
    local_beta = 0.8 + 0.25 * coords[:, 0]
    y = 2.0 + 2.5 * global_x + local_beta * local_x + rng.normal(0.0, 0.1, n)
    X = pd.DataFrame({"global_x": global_x, "local_x": local_x})
    return X, y, pd.DataFrame(coords, columns=["east", "north"])


def latent_regression(n: int = 54, seed: int = 31):
    """Return data whose coefficient variation is driven by latent attributes."""
    rng = np.random.default_rng(seed)
    coords = rng.uniform(0.0, 10.0, size=(n, 2))
    attributes = pd.DataFrame(rng.uniform(size=(n, 2)), columns=["land_use", "density"])
    X = pd.DataFrame(rng.normal(size=(n, 2)), columns=["x1", "x2"])
    beta1 = 1.0 + 1.6 * attributes["land_use"].to_numpy()
    y = 0.5 + beta1 * X["x1"].to_numpy() - 1.2 * X["x2"].to_numpy()
    y += rng.normal(0.0, 0.06, n)
    return X, y, pd.DataFrame(coords, columns=["east", "north"]), attributes


def regime_regression(n: int = 90, seed: int = 37):
    """Return a two-regime process with a sharp left/right boundary."""
    rng = np.random.default_rng(seed)
    coords = rng.uniform(0.0, 10.0, size=(n, 2))
    X = pd.DataFrame(rng.normal(size=(n, 2)), columns=["x1", "x2"])
    regime = coords[:, 0] >= 5.0
    slope = np.where(regime, -2.0, 2.0)
    y = slope * X["x1"].to_numpy() + X["x2"].to_numpy()
    y += rng.normal(0.0, 0.1, n)
    return X, y, pd.DataFrame(coords, columns=["east", "north"]), regime.astype(int)


def print_model_result(model: Any, *, rows: int = 3) -> None:
    """Print a compact model result without assuming a single result protocol."""
    print(f"model={model.__class__.__name__}")
    for name in ("r2_", "aic_", "aicc_", "bic_", "bandwidth_", "bandwidths_"):
        if hasattr(model, name) and getattr(model, name) is not None:
            print(f"{name}={getattr(model, name)}")
    if hasattr(model, "to_frame"):
        try:
            print(model.to_frame().head(rows))
        except (AttributeError, NotImplementedError):
            pass
    if hasattr(model, "summary"):
        summary = model.summary()
        if isinstance(summary, str):
            print("\n".join(summary.splitlines()[:8]))
        else:
            print(summary)


def save_plot(result: Any, filename: str) -> Path:
    """Save a `(figure, axes)` result or a bare Matplotlib figure."""
    import matplotlib.pyplot as plt

    figure = result[0] if isinstance(result, tuple) else result
    if figure is None:
        figure = plt.gcf()
    path = OUTPUT_DIR / filename
    figure.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(figure)
    return path
