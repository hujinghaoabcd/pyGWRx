# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Tests for the standardized GWSS implementation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import rankdata

from pygwrx.models.gwss import GWSS


def _gwmodel_reference(X, coords, bandwidth, *, quantile=True):
    """Independent NumPy translation of GWmodel::gwss fixed bisquare formulas."""
    distances = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(axis=2))
    raw = np.where(distances < bandwidth, (1 - (distances / bandwidth) ** 2) ** 2, 0.0)
    W = raw / raw.sum(axis=1, keepdims=True)
    means = W @ X
    var = np.empty_like(means)
    skew = np.empty_like(means)
    median = np.empty_like(means)
    iqr = np.empty_like(means)
    qi = np.empty_like(means)
    for i, w in enumerate(W):
        for j in range(X.shape[1]):
            centered = X[:, j] - means[i, j]
            var[i, j] = w @ centered**2
            sd = np.sqrt(var[i, j])
            skew[i, j] = (w @ centered**3) / sd**3
            order = np.argsort(X[:, j], kind="stable")
            cumulative = np.cumsum(w[order])
            q = []
            for p in (0.25, 0.5, 0.75):
                eligible = np.flatnonzero(cumulative <= p)
                index = eligible[-1] if eligible.size else 0
                q.append(X[order[index], j])
            median[i, j] = q[1]
            iqr[i, j] = q[2] - q[0]
            qi[i, j] = (2 * q[1] - q[2] - q[0]) / iqr[i, j] if iqr[i, j] else np.nan
    covariance = np.empty(X.shape[0])
    correlation = np.empty(X.shape[0])
    spearman = np.empty(X.shape[0])
    rx, ry = rankdata(X[:, 0]), rankdata(X[:, 1])
    for i, w in enumerate(W):
        correction = 1 - w @ w

        def cov(a, b):
            return w @ ((a - w @ a) * (b - w @ b)) / correction

        covariance[i] = cov(X[:, 0], X[:, 1])
        correlation[i] = covariance[i] / np.sqrt(
            cov(X[:, 0], X[:, 0]) * cov(X[:, 1], X[:, 1])
        )
        spearman[i] = cov(rx, ry) / np.sqrt(cov(rx, rx) * cov(ry, ry))
    return means, var, skew, median, iqr, qi, covariance, correlation, spearman


@pytest.fixture
def sample_data():
    coords = np.array(
        [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0]]
    )
    X = np.array(
        [[1.0, 8.0], [2.0, 5.0], [4.0, 4.0], [3.0, 7.0], [7.0, 2.0], [9.0, 1.0]]
    )
    return X, coords


def test_matches_gwmodel_source_formulas(sample_data):
    X, coords = sample_data
    expected = _gwmodel_reference(X, coords, 2.5)
    model = GWSS(kernel="bisquare", bandwidth=2.5, quantile=True).fit(X, coords)
    np.testing.assert_allclose(model.local_mean_, expected[0], atol=1e-12)
    np.testing.assert_allclose(model.local_var_, expected[1], atol=1e-12)
    np.testing.assert_allclose(model.local_skewness_, expected[2], atol=1e-12)
    np.testing.assert_allclose(model.local_median_, expected[3], atol=1e-12)
    np.testing.assert_allclose(model.local_iqr_, expected[4], atol=1e-12)
    np.testing.assert_allclose(model.local_qi_, expected[5], atol=1e-12, equal_nan=True)
    np.testing.assert_allclose(model.local_cov_[(0, 1)], expected[6], atol=1e-12)
    np.testing.assert_allclose(model.local_corr_[(0, 1)], expected[7], atol=1e-12)
    np.testing.assert_allclose(
        model.local_corr_spearman_[(0, 1)], expected[8], atol=1e-12
    )


def test_adaptive_bandwidth_is_neighbour_count(sample_data):
    X, coords = sample_data
    model = GWSS(kernel="boxcar", bandwidth=3, adaptive=True).fit(X, coords)
    np.testing.assert_allclose(model.weights_.sum(axis=1), 1.0)
    assert np.all((model.weights_ > 0).sum(axis=1) == 3)


def test_global_boxcar_degenerates_to_global_statistics(sample_data):
    X, coords = sample_data
    model = GWSS(kernel="boxcar", bandwidth=100.0).fit(X, coords)
    expected_mean = X.mean(axis=0)
    expected_var = ((X - expected_mean) ** 2).mean(axis=0)
    np.testing.assert_allclose(model.local_mean_, np.tile(expected_mean, (len(X), 1)))
    np.testing.assert_allclose(model.local_var_, np.tile(expected_var, (len(X), 1)))


def test_dataframe_and_independent_summary_locations(sample_data):
    X, coords = sample_data
    frame = pd.DataFrame(X, columns=["income", "price"])
    summary_coords = pd.DataFrame([[0.5, 0.5], [1.5, 0.5]], columns=["x", "y"])
    model = GWSS(kernel="gaussian", bandwidth=1.0).fit(
        frame, pd.DataFrame(coords), summary_coords
    )
    assert model.var_names_ == ["income", "price"]
    assert model.to_dataframe().shape[0] == 2
    assert "Cov_income.price" in model.to_dataframe().columns


def test_known_spatial_gradient(sample_data):
    _, coords = sample_data
    X = coords[:, [0]] * 10.0
    model = GWSS(kernel="gaussian", bandwidth=0.6).fit(X, coords)
    assert model.local_mean_[0, 0] < model.local_mean_[2, 0]
    assert model.local_mean_[3, 0] < model.local_mean_[5, 0]


def test_bandwidth_selection_returns_valid_value(sample_data):
    X, coords = sample_data
    fixed = GWSS(kernel="gaussian").select_bandwidth(X, coords)
    adaptive = GWSS(kernel="boxcar", adaptive=True).select_bandwidth(X, coords)
    assert fixed > 0
    assert 2 <= adaptive <= len(X)


@pytest.mark.parametrize("bad_bandwidth", [0, -1, np.inf, np.nan])
def test_rejects_invalid_fixed_bandwidth(bad_bandwidth):
    with pytest.raises((TypeError, ValueError)):
        GWSS(bandwidth=bad_bandwidth)


def test_failed_refit_clears_previous_state(sample_data):
    X, coords = sample_data
    model = GWSS(bandwidth=2.0).fit(X, coords)
    assert model._is_fitted
    bad = X.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError):
        model.fit(bad, coords)
    assert not model._is_fitted
    assert model.local_mean_ is None


def test_constant_variable_reports_undefined_shape_statistics(sample_data):
    _, coords = sample_data
    X = np.ones((len(coords), 1))
    model = GWSS(kernel="boxcar", bandwidth=100.0).fit(X, coords)
    np.testing.assert_allclose(model.local_std_, 0.0)
    assert np.isnan(model.local_skewness_).all()
