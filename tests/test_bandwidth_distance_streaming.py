# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Regression tests for bounded-memory bandwidth distance evaluation."""

import importlib

import numpy as np
import pytest

from pygwrx.core.bandwidth import AICSelector, BICSelector, CrossValidationSelector
from pygwrx.core.kernels import gaussian_kernel

bandwidth_module = importlib.import_module("pygwrx.core.bandwidth")


def _make_data(n_samples: int = 132):
    rng = np.random.default_rng(20260829)
    coords = rng.uniform(0.0, 1.0, size=(n_samples, 2))
    X = np.column_stack([np.ones(n_samples), rng.normal(size=n_samples)])
    y = 1.5 + 0.8 * X[:, 1] + rng.normal(0.0, 0.08, size=n_samples)
    return X, y, coords


def _track_distance_blocks(monkeypatch, n_samples: int):
    original = bandwidth_module.compute_distance_matrix
    calls = []

    def tracked(left, right, metric="euclidean"):
        left_arr = np.asarray(left)
        right_arr = np.asarray(right)
        calls.append((left_arr.shape[0], right_arr.shape[0]))
        if left_arr.shape[0] == n_samples and right_arr.shape[0] == n_samples:
            raise AssertionError(
                "bandwidth selection requested a full n x n distance matrix"
            )
        return original(left, right, metric=metric)

    monkeypatch.setattr(bandwidth_module, "compute_distance_matrix", tracked)
    return calls


@pytest.mark.parametrize(
    "selector",
    [
        CrossValidationSelector(n_intervals=2, optimization_method="grid"),
        AICSelector(n_intervals=2, optimization_method="grid"),
        BICSelector(n_intervals=2, optimization_method="grid"),
    ],
)
def test_fixed_bandwidth_objectives_use_bounded_distance_blocks(monkeypatch, selector):
    X, y, coords = _make_data()
    calls = _track_distance_blocks(monkeypatch, coords.shape[0])
    selected = selector.select(
        X,
        y,
        coords,
        gaussian_kernel,
        bandwidth_range=(0.25, 1.25),
    )
    assert np.isfinite(float(selected))
    assert calls
    assert max(left_rows for left_rows, _ in calls) <= 128
    assert {right_rows for _, right_rows in calls} == {coords.shape[0]}


def test_fixed_automatic_range_uses_bounded_distance_scan(monkeypatch):
    X, y, coords = _make_data()
    calls = _track_distance_blocks(monkeypatch, coords.shape[0])
    selector = CrossValidationSelector(n_intervals=2, optimization_method="grid")
    selected = selector.select(X, y, coords, gaussian_kernel)
    assert np.isfinite(float(selected))
    assert selector.search_range_ is not None
    assert calls
    assert max(left_rows for left_rows, _ in calls) <= 128


def test_adaptive_bandwidth_objective_uses_bounded_distance_blocks(monkeypatch):
    X, y, coords = _make_data()
    calls = _track_distance_blocks(monkeypatch, coords.shape[0])
    selector = CrossValidationSelector(adaptive=True)
    selected = selector.select(
        X,
        y,
        coords,
        gaussian_kernel,
        bandwidth_range=(8, 10),
    )
    assert int(selected) in {8, 9, 10}
    assert calls
    assert max(left_rows for left_rows, _ in calls) <= 128
