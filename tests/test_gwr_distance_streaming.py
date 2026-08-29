# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Regression tests for bounded-memory GWR distance evaluation."""

from __future__ import annotations

import importlib

import numpy as np

from pygwrx import GWR

gwr_module = importlib.import_module("pygwrx.models.gwr")


def _make_data(n_samples: int = 300):
    rng = np.random.default_rng(20260829)
    coords = rng.uniform(0.0, 1.0, size=(n_samples, 2))
    X = rng.normal(size=(n_samples, 2))
    y = 2.5 + 1.2 * X[:, 0] - 0.7 * X[:, 1] + rng.normal(0.0, 0.1, n_samples)
    return X, y, coords


def _track_distance_blocks(monkeypatch):
    original = gwr_module.compute_distance_matrix
    calls: list[tuple[int, int]] = []

    def tracked(coords1, coords2=None, metric="euclidean", **kwargs):
        first = np.asarray(coords1)
        second = first if coords2 is None else np.asarray(coords2)
        calls.append((first.shape[0], second.shape[0]))
        if first.shape[0] > gwr_module._DISTANCE_BLOCK_ROWS:
            raise AssertionError(
                "GWR requested more target rows than the bounded distance block size."
            )
        return original(coords1, coords2, metric=metric, **kwargs)

    monkeypatch.setattr(gwr_module, "compute_distance_matrix", tracked)
    return calls


def test_numeric_bandwidth_fit_streams_calibration_and_local_r2_distances(
    monkeypatch,
):
    X, y, coords = _make_data()
    calls = _track_distance_blocks(monkeypatch)

    model = GWR(kernel="gaussian", bandwidth=0.45).fit(
        X,
        y,
        coords,
        compute_hat_matrix=False,
        compute_local_r2=True,
        compute_inference=True,
    )

    assert model.hat_matrix_ is None
    assert model.local_r2_ is not None
    assert np.all(np.isfinite(model.fitted_values_))
    assert np.all(np.isfinite(model.local_r2_))
    assert calls
    assert max(rows for rows, _ in calls) <= gwr_module._DISTANCE_BLOCK_ROWS
    assert not any(
        rows == coords.shape[0] and cols == coords.shape[0] for rows, cols in calls
    )


def test_prediction_streams_target_to_training_distances(monkeypatch):
    X, y, coords = _make_data()
    model = GWR(kernel="gaussian", bandwidth=0.45).fit(
        X,
        y,
        coords,
        compute_hat_matrix=False,
        compute_local_r2=False,
        compute_inference=False,
    )

    calls = _track_distance_blocks(monkeypatch)
    rng = np.random.default_rng(7)
    n_targets = 300
    X_new = rng.normal(size=(n_targets, X.shape[1]))
    coords_new = rng.uniform(0.0, 1.0, size=(n_targets, 2))

    predictions = model.predict(X_new, coords_new)

    assert predictions.shape == (n_targets,)
    assert np.all(np.isfinite(predictions))
    assert calls
    assert max(rows for rows, _ in calls) <= gwr_module._DISTANCE_BLOCK_ROWS
    assert not any(
        rows == n_targets and cols == coords.shape[0] for rows, cols in calls
    )
