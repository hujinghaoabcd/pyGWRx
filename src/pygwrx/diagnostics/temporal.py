# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Time-indexed views for GTWR-family estimators.

Author:
    Jinghao Hu
"""

from __future__ import annotations

__author__ = "Jinghao Hu"
__license__ = "MIT"

from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from ._utils import require_fitted, training_coords
from .inference import FeatureLike, parameter_inference


@dataclass(frozen=True)
class TemporalGroups:
    """Unique time values and row indices for a fitted spatiotemporal model."""

    values: np.ndarray
    indices: Tuple[np.ndarray, ...]


def model_times(model: Any) -> np.ndarray:
    """Return one time value per plotted row."""
    require_fitted(model)
    value = getattr(model, "times_train_", None)
    if value is not None:
        times = np.asarray(value).reshape(-1)
        if times.size != training_coords(model).shape[0]:
            raise ValueError(
                "times_train_ length does not match calibration coordinates."
            )
        return times
    # STWR estimates only the latest stage. Represent it by cumulative time.
    intervals = getattr(model, "time_intervals_", None)
    coords_stages = getattr(model, "coords_stages_", None)
    if intervals is not None and coords_stages:
        current = float(np.sum(np.asarray(intervals, dtype=float)))
        return np.full(len(coords_stages[-1]), current, dtype=float)
    raise ValueError(
        f"{model.__class__.__name__} does not expose row-aligned time values."
    )


def temporal_groups(model: Any) -> TemporalGroups:
    """Group fitted rows by exact time value while preserving chronological order."""
    times = model_times(model)
    values = pd.unique(times)
    try:
        values = np.asarray(sorted(values))
    except TypeError:
        values = np.asarray(values)
    groups = tuple(np.flatnonzero(times == value) for value in values)
    return TemporalGroups(values=values, indices=groups)


def temporal_parameter_frame(model: Any, feature: FeatureLike) -> pd.DataFrame:
    """Return local parameters with coordinates and times in tidy form."""
    view = parameter_inference(model, feature)
    coords = training_coords(model)
    times = model_times(model)
    if view.values.size != coords.shape[0]:
        raise ValueError(
            "Parameter surface length does not match temporal coordinates."
        )
    return pd.DataFrame(
        {
            "coord_0": coords[:, 0],
            "coord_1": coords[:, 1],
            "time": times,
            "coefficient": view.values,
        }
    )


def parameter_trajectory(
    model: Any,
    feature: FeatureLike,
    *,
    location: Optional[Union[int, Sequence[float]]] = None,
    reducer: str = "mean",
) -> pd.DataFrame:
    """Aggregate a parameter surface over time or follow the nearest location."""
    frame = temporal_parameter_frame(model, feature)
    if location is not None:
        if isinstance(location, (int, np.integer)) and not isinstance(
            location, (bool, np.bool_)
        ):
            reference = frame.loc[int(location), ["coord_0", "coord_1"]].to_numpy(float)
        else:
            reference = np.asarray(location, dtype=float).reshape(-1)
            if reference.size != 2:
                raise ValueError("location must be a row index or an (x, y) pair.")
        records: List[dict] = []
        for time, group in frame.groupby("time", sort=True):
            distances = np.sqrt(
                (group["coord_0"] - reference[0]) ** 2
                + (group["coord_1"] - reference[1]) ** 2
            )
            row = group.loc[distances.idxmin()]
            records.append(
                {
                    "time": time,
                    "coefficient": row["coefficient"],
                    "coord_0": row["coord_0"],
                    "coord_1": row["coord_1"],
                }
            )
        return pd.DataFrame(records)
    token = str(reducer).strip().lower()
    if token not in {"mean", "median", "min", "max"}:
        raise ValueError("reducer must be 'mean', 'median', 'min', or 'max'.")
    series = getattr(frame.groupby("time", sort=True)["coefficient"], token)()
    return series.rename("coefficient").reset_index()
