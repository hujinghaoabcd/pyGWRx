# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Canonical model-independent time-axis normalization for pyGWRx.

This module owns numeric/datetime time parsing, resolved datetime units,
training origins, and prediction-time compatibility checks. Spatiotemporal
distance formulas and model-specific temporal effects do not belong here.

Author:
    Jinghao Hu
"""

from __future__ import annotations

__author__ = "Jinghao Hu"
__license__ = "MIT"

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

__all__: tuple[str, ...] = ()

_TIME_FACTORS_SECONDS = {
    "seconds": 1.0,
    "minutes": 60.0,
    "hours": 3_600.0,
    "days": 86_400.0,
    "weeks": 604_800.0,
}
_TIME_UNIT_ALIASES = {
    "s": "seconds",
    "sec": "seconds",
    "secs": "seconds",
    "second": "seconds",
    "seconds": "seconds",
    "m": "minutes",
    "min": "minutes",
    "mins": "minutes",
    "minute": "minutes",
    "minutes": "minutes",
    "h": "hours",
    "hr": "hours",
    "hrs": "hours",
    "hour": "hours",
    "hours": "hours",
    "d": "days",
    "day": "days",
    "days": "days",
    "w": "weeks",
    "week": "weeks",
    "weeks": "weeks",
}


@dataclass(frozen=True)
class TimeAxis:
    """Internal normalized representation of one model-independent time axis."""

    values: np.ndarray
    unit: str
    origin: Optional[pd.Timestamp]
    datetime_like: bool


def looks_datetime_like(times: object) -> bool:
    """Return whether values follow the existing datetime-like input convention."""
    if isinstance(times, (pd.DatetimeIndex, pd.PeriodIndex)):
        return True
    if isinstance(times, pd.Series):
        return bool(
            pd.api.types.is_datetime64_any_dtype(times.dtype)
            or pd.api.types.is_period_dtype(times.dtype)
        )
    array = np.asarray(times)
    return bool(
        np.issubdtype(array.dtype, np.datetime64) or array.dtype.kind in {"O", "U", "S"}
    )


def auto_time_unit(span_seconds: float) -> str:
    """Choose the historical GTWR datetime unit from a training time span."""
    if span_seconds < 120.0:
        return "seconds"
    if span_seconds < 7_200.0:
        return "minutes"
    if span_seconds < 172_800.0:
        return "hours"
    if span_seconds < 1_209_600.0:
        return "days"
    return "weeks"


def _coerce_time_vector(times: object) -> np.ndarray:
    """Return the historical one-dimensional raw time representation."""
    raw = np.asarray(times)
    if raw.ndim > 2 or (raw.ndim == 2 and 1 not in raw.shape):
        raise ValueError("times must be one-dimensional or a single-column vector.")
    raw = raw.reshape(-1)
    if raw.size == 0:
        raise ValueError("times cannot be empty.")
    return raw


def _parse_datetime_values(raw: np.ndarray) -> pd.DatetimeIndex:
    """Parse datetime-like values using the existing GTWR validation contract."""
    try:
        values = pd.to_datetime(raw, errors="raise")
    except Exception as exc:
        raise ValueError("times could not be parsed as datetime values.") from exc
    if values.isna().any():
        raise ValueError("times contains missing datetime values.")
    return pd.DatetimeIndex(values)


def normalize_training_times(times: object, *, time_unit: str = "auto") -> TimeAxis:
    """Normalize training times without defining any spatiotemporal distance."""
    raw = _coerce_time_vector(times)

    if looks_datetime_like(times):
        values = _parse_datetime_values(raw)
        origin = pd.Timestamp(values.min())
        span_seconds = float((values.max() - values.min()).total_seconds())
        if time_unit == "auto":
            unit = auto_time_unit(span_seconds)
        else:
            try:
                unit = _TIME_UNIT_ALIASES[time_unit]
            except KeyError as exc:
                raise ValueError(
                    "time_unit must be 'auto', seconds, minutes, hours, days, or weeks."
                ) from exc
        elapsed_seconds = np.asarray((values - origin).total_seconds(), dtype=float)
        converted = elapsed_seconds / _TIME_FACTORS_SECONDS[unit]
        datetime_like = True
    else:
        try:
            converted = np.asarray(raw, dtype=float)
        except (TypeError, ValueError) as exc:
            raise TypeError("Numeric times must contain real scalar values.") from exc
        origin = None
        unit = "numeric"
        datetime_like = False

    converted = np.asarray(converted, dtype=float).reshape(-1)
    if not np.all(np.isfinite(converted)):
        raise ValueError("times contains NaN or infinite values.")
    return TimeAxis(
        values=converted,
        unit=unit,
        origin=origin,
        datetime_like=datetime_like,
    )


def normalize_prediction_times(times: object, *, axis: TimeAxis) -> np.ndarray:
    """Normalize prediction times against an already resolved training axis."""
    raw = _coerce_time_vector(times)
    datetime_like = looks_datetime_like(times)

    if datetime_like:
        if not axis.datetime_like or axis.origin is None:
            raise ValueError(
                "Prediction times must be datetime-like because the model was fitted "
                "with datetime-like times."
            )
        values = _parse_datetime_values(raw)
        elapsed_seconds = np.asarray(
            (values - axis.origin).total_seconds(),
            dtype=float,
        )
        converted = elapsed_seconds / _TIME_FACTORS_SECONDS[axis.unit]
    else:
        if axis.datetime_like:
            raise ValueError(
                "Prediction times must be datetime-like because the model was fitted "
                "with datetime-like times."
            )
        try:
            converted = np.asarray(raw, dtype=float)
        except (TypeError, ValueError) as exc:
            raise TypeError("Numeric times must contain real scalar values.") from exc

    converted = np.asarray(converted, dtype=float).reshape(-1)
    if not np.all(np.isfinite(converted)):
        raise ValueError("times contains NaN or infinite values.")
    return converted
