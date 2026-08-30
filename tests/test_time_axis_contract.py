# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Architecture contracts for the B4 canonical time-axis boundary.

Author:
    Jinghao Hu
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import pygwrx
import pygwrx.core as core
from pygwrx import GTWR
from pygwrx.core import time as time_module
from pygwrx.core.time import TimeAxis, normalize_prediction_times, normalize_training_times


def test_time_axis_is_private_first_and_formula_free() -> None:
    """Keep the B4 value object internal and model-independent."""
    assert time_module.__all__ == ()
    assert not hasattr(core, "TimeAxis")
    assert not hasattr(pygwrx, "TimeAxis")
    for model_specific_name in (
        "temporal_distances",
        "combine_distances",
        "spatiotemporal_distance",
        "temporal_effect",
    ):
        assert not hasattr(time_module, model_specific_name)


def test_numeric_time_axis_is_not_rescaled() -> None:
    """Numeric times retain their existing values and numeric convention."""
    raw = np.array([2.5, 4.0, 8.25])
    axis = normalize_training_times(raw, time_unit="auto")
    assert isinstance(axis, TimeAxis)
    assert axis.unit == "numeric"
    assert axis.origin is None
    assert axis.datetime_like is False
    np.testing.assert_array_equal(axis.values, raw)
    np.testing.assert_array_equal(
        normalize_prediction_times(np.array([3.0, 9.0]), axis=axis),
        np.array([3.0, 9.0]),
    )


def test_datetime_axis_preserves_gtwr_auto_unit_and_origin() -> None:
    """Datetime normalization keeps the historical GTWR auto-unit thresholds."""
    raw = pd.Timestamp("2026-01-01") + pd.to_timedelta([0.0, 1.0, 3.0], unit="h")
    axis = normalize_training_times(raw, time_unit="auto")
    assert axis.unit == "hours"
    assert axis.origin == pd.Timestamp("2026-01-01")
    assert axis.datetime_like is True
    np.testing.assert_allclose(axis.values, np.array([0.0, 1.0, 3.0]))
    future = raw + pd.to_timedelta(30.0, unit="m")
    np.testing.assert_allclose(
        normalize_prediction_times(future, axis=axis),
        np.array([0.5, 1.5, 3.5]),
    )


def test_prediction_time_kind_compatibility_is_preserved() -> None:
    """Prediction input kind must remain compatible with the fitted time axis."""
    datetimes = pd.date_range("2026-01-01", periods=3, freq="h")
    datetime_axis = normalize_training_times(datetimes, time_unit="hours")
    with pytest.raises(ValueError, match="datetime-like"):
        normalize_prediction_times(np.arange(3, dtype=float), axis=datetime_axis)

    numeric_axis = normalize_training_times(np.arange(3, dtype=float))
    with pytest.raises(ValueError, match="datetime-like"):
        normalize_prediction_times(datetimes, axis=numeric_axis)


def test_time_axis_input_shape_and_finite_value_contracts() -> None:
    """Retain historical GTWR shape, emptiness, and finite-value validation."""
    with pytest.raises(ValueError, match="one-dimensional"):
        normalize_training_times(np.ones((2, 2)))
    with pytest.raises(ValueError, match="cannot be empty"):
        normalize_training_times(np.array([], dtype=float))
    with pytest.raises(ValueError, match="NaN or infinite"):
        normalize_training_times(np.array([0.0, np.nan]))


def test_gtwr_mirrors_canonical_datetime_axis_without_public_state_drift() -> None:
    """GTWR keeps its frozen fitted-state names while using the canonical axis."""
    rng = np.random.default_rng(42)
    coords = np.column_stack([np.arange(8, dtype=float), np.zeros(8)])
    X = rng.normal(size=(8, 1))
    y = 1.0 + 0.5 * X[:, 0]
    datetimes = pd.Timestamp("2026-01-01") + pd.to_timedelta(
        np.arange(8, dtype=float), unit="h"
    )
    model = GTWR(
        kernel="gaussian",
        bandwidth=4.0,
        lambda_st=0.5,
        causal=False,
        time_unit="auto",
    ).fit(X, y, coords, datetimes, compute_local_r2=False, compute_inference=False)

    assert isinstance(model._time_axis, TimeAxis)
    assert model.time_unit_ == model._time_axis.unit == "hours"
    assert model.time_origin_ == model._time_axis.origin == pd.Timestamp("2026-01-01")
    assert model.time_input_kind_ == "datetime"
    np.testing.assert_allclose(model.times_train_, model._time_axis.values)
