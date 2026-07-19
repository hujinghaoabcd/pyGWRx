# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Validation tests for the published-source SGTWR implementation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pygwrx import SGTWR, SGWR
from pygwrx.core.solver import adaptive_bandwidth_weights
from pygwrx.core.utils import add_intercept, compute_distance_matrix


def _data(n: int = 28, seed: int = 77):
    rng = np.random.default_rng(seed)
    grid_x = np.arange(n) % 7
    grid_y = np.arange(n) // 7
    coords = np.column_stack((grid_x, grid_y)).astype(float)
    times = np.repeat(np.arange(4), 7)[:n].astype(float)
    x1 = rng.normal(size=n)
    x2 = 0.4 * x1 + rng.normal(scale=0.8, size=n)
    X = np.column_stack((x1, x2))
    local_slope = 0.8 + 0.12 * coords[:, 0] + 0.08 * times
    y = 2.0 + local_slope * x1 - 0.5 * x2 + rng.normal(scale=0.05, size=n)
    return X, y, coords, times


def test_similarity_weights_match_published_sgwr_formula():
    X, y, coords, times = _data()
    model = SGTWR(
        spatial_bandwidth=12,
        temporal_bandwidth=2.0,
        adaptive=True,
        alpha=0.5,
        similarity_vars=[0, 1],
        ridge=1e-8,
    ).fit(X, y, coords, times)
    standardized = (X - X.mean(axis=0)) / X.std(axis=0, ddof=0)
    distance = np.mean(
        np.abs(standardized[:, None, :] - standardized[None, :, :]),
        axis=2,
    )
    expected = np.exp(-(distance**2))
    np.testing.assert_allclose(model.similarity_weights_, expected)


def test_spatiotemporal_weights_match_equation_18():
    X, y, coords, times = _data()
    spatial_bandwidth = 12
    temporal_bandwidth = 1.7
    model = SGTWR(
        spatial_bandwidth=spatial_bandwidth,
        temporal_bandwidth=temporal_bandwidth,
        adaptive=True,
        alpha=1.0,
        ridge=1e-8,
    ).fit(X, y, coords, times)
    spatial_distance = compute_distance_matrix(coords, coords)
    temporal_distance = np.abs(times[:, None] - times[None, :])
    scales = np.array(
        [adaptive_bandwidth_weights(row, spatial_bandwidth) for row in spatial_distance]
    )
    expected = np.exp(
        -0.5
        * (
            (spatial_distance / scales[:, None]) ** 2
            + (temporal_distance / temporal_bandwidth) ** 2
        )
    )
    np.testing.assert_allclose(model.spatiotemporal_weights_, expected)


def test_combined_weights_are_published_convex_combination():
    X, y, coords, times = _data()
    alpha = 0.65
    model = SGTWR(
        spatial_bandwidth=12,
        temporal_bandwidth=2.0,
        adaptive=True,
        alpha=alpha,
        ridge=1e-8,
    ).fit(X, y, coords, times)
    expected = (
        alpha * model.spatiotemporal_weights_
        + (1.0 - alpha) * model.similarity_weights_
    )
    np.testing.assert_allclose(model.combined_weights_, expected)


def test_large_temporal_bandwidth_degenerates_to_sgwr():
    X, y, coords, times = _data(n=28)
    alpha = 0.7
    sgtwr = SGTWR(
        spatial_bandwidth=12,
        temporal_bandwidth=1.0e12,
        adaptive=True,
        alpha=alpha,
        similarity_vars=[0, 1],
        ridge=1e-8,
    ).fit(X, y, coords, times)
    sgwr = SGWR(
        bandwidth=12,
        adaptive=True,
        kernel="gaussian",
        alpha=alpha,
        similarity_vars=[0, 1],
        ridge=1e-8,
    ).fit(X, y, coords)
    np.testing.assert_allclose(
        sgtwr.spatiotemporal_weights_,
        sgwr.spatial_weights_,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        sgtwr.combined_weights_,
        sgwr.combined_weights_,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        sgtwr.fitted_values_,
        sgwr.fitted_values_,
        atol=1e-10,
    )


def test_local_coefficients_match_manual_wls():
    X, y, coords, times = _data(n=28)
    ridge = 1e-7
    model = SGTWR(
        spatial_bandwidth=14,
        temporal_bandwidth=2.0,
        adaptive=True,
        alpha=0.55,
        ridge=ridge,
    ).fit(X, y, coords, times)
    X_design = add_intercept(X)
    location = 6
    weights = model.combined_weights_[location]
    system = X_design.T @ (X_design * weights[:, None])
    penalty = np.eye(X_design.shape[1]) * ridge
    penalty[0, 0] = 0.0
    expected = np.linalg.solve(
        system + penalty,
        X_design.T @ (weights * y),
    )
    np.testing.assert_allclose(
        model.parameters_[location],
        expected,
        atol=1e-11,
    )


def test_similarity_only_alpha_zero_is_independent_of_space_time():
    X, y, coords, times = _data(n=28)
    first = SGTWR(
        spatial_bandwidth=12,
        temporal_bandwidth=1.0,
        adaptive=True,
        alpha=0.0,
        ridge=1e-8,
    ).fit(X, y, coords, times)
    second = SGTWR(
        spatial_bandwidth=18,
        temporal_bandwidth=10.0,
        adaptive=True,
        alpha=0.0,
        ridge=1e-8,
    ).fit(X, y, coords * 10.0, times * 5.0)
    np.testing.assert_allclose(
        first.combined_weights_,
        second.combined_weights_,
    )
    np.testing.assert_allclose(
        first.fitted_values_,
        second.fitted_values_,
    )


def test_deterministic_aicc_selection_uses_supplied_candidates():
    X, y, coords, times = _data(n=28)
    model = SGTWR(
        spatial_bandwidth="aicc",
        temporal_bandwidth="aicc",
        adaptive=True,
        alpha="aicc",
        spatial_bandwidth_candidates=[12, 16],
        temporal_bandwidth_candidates=[1.0, 3.0],
        alpha_candidates=[0.25, 0.75],
        ridge=1e-8,
    ).fit(X, y, coords, times)
    assert model.spatial_bandwidth_ in {12, 16}
    assert model.temporal_bandwidth_ in {1.0, 3.0}
    assert model.alpha_ in {0.25, 0.75}
    assert len(model.selection_history_) == 8
    assert any(np.isfinite(item["aicc"]) for item in model.selection_history_)


def test_prediction_is_direct_recalibration_and_nonmutating():
    X, y, coords, times = _data(n=28)
    model = SGTWR(
        spatial_bandwidth=14,
        temporal_bandwidth=2.0,
        adaptive=True,
        alpha=0.5,
        ridge=1e-8,
    ).fit(X, y, coords, times)
    original = model.parameters_.copy()
    result = model.predict_result(
        X[:3],
        coords[:3] + 0.1,
        times[:3] + 0.2,
    )
    assert result.predictions.shape == (3,)
    assert result.coef.shape == (3, 2)
    np.testing.assert_allclose(model.parameters_, original)


def test_datetime_times_use_validated_gtwr_unit_conversion():
    X, y, coords, times = _data(n=28)
    datetimes = pd.Timestamp("2022-01-01") + pd.to_timedelta(times, unit="D")
    model = SGTWR(
        spatial_bandwidth=14,
        temporal_bandwidth=2.0,
        adaptive=True,
        alpha=0.6,
        time_unit="days",
        ridge=1e-8,
    ).fit(X, y, coords, datetimes)
    np.testing.assert_allclose(model.times_train_, times)
    prediction = model.predict(X[:2], coords[:2], datetimes[:2])
    assert np.all(np.isfinite(prediction))


def test_dataframe_feature_selection_and_order():
    X, y, coords, times = _data(n=28)
    frame = pd.DataFrame(X, columns=["income", "density"])
    model = SGTWR(
        spatial_bandwidth=14,
        temporal_bandwidth=2.0,
        adaptive=True,
        alpha=0.5,
        similarity_vars=["income"],
        ridge=1e-8,
    ).fit(frame, y, coords, times)
    assert model.similarity_feature_names_ == ("income",)
    with pytest.raises(ValueError):
        model.predict(frame[["density", "income"]], coords, times)


def test_causal_mode_excludes_future_from_all_combined_weights():
    X, y, coords, times = _data(n=28)
    model = SGTWR(
        spatial_bandwidth=20.0,
        temporal_bandwidth=3.0,
        adaptive=False,
        alpha=0.5,
        causal=True,
        ridge=1e-8,
    ).fit(X, y, coords, times)
    query = 0
    future = times > times[query]
    assert np.all(model.spatiotemporal_weights_[query, future] == 0.0)
    assert np.all(model.combined_weights_[query, future] == 0.0)


def test_failed_refit_clears_state():
    X, y, coords, times = _data(n=28)
    model = SGTWR(
        spatial_bandwidth=14,
        temporal_bandwidth=2.0,
        adaptive=True,
        alpha=0.5,
        ridge=1e-8,
    ).fit(X, y, coords, times)
    with pytest.raises(ValueError):
        model.fit(X[:-1], y, coords[:-1], times[:-1])
    with pytest.raises(ValueError, match="not fitted"):
        model.get_results()


def test_result_and_diagnostic_shapes():
    X, y, coords, times = _data(n=28)
    model = SGTWR(
        spatial_bandwidth=14,
        temporal_bandwidth=2.0,
        adaptive=True,
        alpha=0.5,
        ridge=1e-8,
    ).fit(X, y, coords, times)
    assert model.hat_matrix_.shape == (28, 28)
    assert model.coef_se_.shape == (28, 2)
    assert model.get_results().shape[0] == 28
    assert np.isfinite(model.diagnostics_["aicc"])
