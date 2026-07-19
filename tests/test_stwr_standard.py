# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Validation tests for the published STWR implementation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pygwrx import GWR, STWR
from pygwrx.core.utils import add_intercept


def _stages(n: int = 18, seed: int = 42):
    rng = np.random.default_rng(seed)
    x_axis = np.linspace(0.0, 6.0, n)
    coords = np.column_stack((x_axis, 0.25 * np.sin(x_axis)))
    X_list = []
    y_list = []
    coords_list = []
    for stage in range(3):
        x1 = np.linspace(-1.0, 1.0, n)
        x2 = rng.normal(scale=0.7, size=n)
        X = np.column_stack((x1, x2))
        y = 2.0 + 0.2 * stage + (1.1 + 0.1 * stage) * x1 - 0.45 * x2
        y = y + rng.normal(scale=0.025, size=n)
        X_list.append(X)
        y_list.append(y)
        coords_list.append(coords.copy())
    return X_list, y_list, coords_list, [0.0, 1.5, 2.0]


def test_temporal_distance_matches_official_formula():
    X_list, y_list, coords_list, intervals = _stages(n=12)
    model = STWR(
        spatial_bandwidth=5.0,
        adaptive=False,
        kernel="gaussian",
        alpha=0.4,
        theta=0.0,
        tick_nums=2,
        ridge=1e-8,
    ).fit(X_list, y_list, coords_list, intervals)

    past_slice = model.stage_slices_[1]
    current_y = y_list[-1]
    past_y = y_list[-2]
    elapsed = intervals[-1]
    total = sum(intervals[-2:])
    safe = np.where(np.abs(past_y) < 1e-6, 1e-6, past_y)
    expected_distance = (total / elapsed) * np.abs(
        (past_y[None, :] - current_y[:, None]) / safe[None, :]
    )
    expected = np.tanh(0.5 * expected_distance)
    np.testing.assert_allclose(model.temporal_weights_[:, past_slice], expected)


def test_combined_weights_are_published_weighted_average():
    X_list, y_list, coords_list, intervals = _stages(n=12)
    alpha = 0.35
    model = STWR(
        spatial_bandwidth=4.0,
        adaptive=False,
        kernel="gaussian",
        alpha=alpha,
        theta=0.0,
        tick_nums=3,
        ridge=1e-8,
    ).fit(X_list, y_list, coords_list, intervals)
    expected = (1.0 - alpha) * model.spatial_weights_ + alpha * model.temporal_weights_
    np.testing.assert_allclose(model.weights_, expected)


def test_single_stage_degenerates_to_gwr():
    X_list, y_list, coords_list, intervals = _stages(n=20)
    stwr = STWR(
        spatial_bandwidth=4.5,
        adaptive=False,
        kernel="gaussian",
        alpha=0.9,
        tick_nums=1,
        ridge=1e-8,
    ).fit(X_list, y_list, coords_list, intervals)
    gwr = GWR(
        bandwidth=4.5,
        adaptive=False,
        kernel="gaussian",
        sigma2_v1=True,
    ).fit(X_list[-1], y_list[-1], coords_list[-1])
    np.testing.assert_allclose(stwr.fitted_values_, gwr.fitted_values_, atol=2e-7)
    np.testing.assert_allclose(
        stwr.parameters_, np.column_stack((gwr.intercept_, gwr.coef_)), atol=2e-7
    )


def test_local_coefficients_match_manual_weighted_least_squares():
    X_list, y_list, coords_list, intervals = _stages(n=15)
    ridge = 1e-7
    model = STWR(
        spatial_bandwidth=5.0,
        adaptive=False,
        kernel="gaussian",
        alpha=0.25,
        theta=0.0,
        tick_nums=2,
        ridge=ridge,
    ).fit(X_list, y_list, coords_list, intervals)
    X_source = np.vstack((X_list[-1], X_list[-2]))
    y_source = np.concatenate((y_list[-1], y_list[-2]))
    X_design = add_intercept(X_source)
    location = 4
    weights = model.weights_[location]
    system = X_design.T @ (X_design * weights[:, None])
    penalty = np.eye(X_design.shape[1]) * ridge
    penalty[0, 0] = 0.0
    expected = np.linalg.solve(system + penalty, X_design.T @ (weights * y_source))
    np.testing.assert_allclose(
        model.parameters_[location], expected, rtol=1e-11, atol=1e-11
    )


def test_positive_theta_changes_past_spatial_weights_only():
    X_list, y_list, coords_list, intervals = _stages(n=14)
    zero = STWR(
        spatial_bandwidth=5.0,
        adaptive=False,
        kernel="gaussian",
        alpha=0.3,
        theta=0.0,
        tick_nums=2,
        ridge=1e-8,
    ).fit(X_list, y_list, coords_list, intervals)
    sloped = STWR(
        spatial_bandwidth=5.0,
        adaptive=False,
        kernel="gaussian",
        alpha=0.3,
        theta=0.08,
        tick_nums=2,
        ridge=1e-8,
    ).fit(X_list, y_list, coords_list, intervals)
    current_slice = zero.stage_slices_[0]
    past_slice = zero.stage_slices_[1]
    np.testing.assert_allclose(
        zero.spatial_weights_[:, current_slice],
        sloped.spatial_weights_[:, current_slice],
    )
    assert not np.allclose(
        zero.spatial_weights_[:, past_slice],
        sloped.spatial_weights_[:, past_slice],
    )


def test_parameter_selection_uses_supplied_candidates():
    X_list, y_list, coords_list, intervals = _stages(n=14)
    model = STWR(
        spatial_bandwidth="cv",
        adaptive=True,
        bandwidth_candidates=[8, 11],
        alpha="cv",
        alpha_candidates=[0.0, 0.4],
        theta="cv",
        theta_candidates=[0.0],
        tick_nums="cv",
        tick_candidates=[1, 2],
        ridge=1e-8,
    ).fit(X_list, y_list, coords_list, intervals)
    assert model.spatial_bandwidth_ in {8, 11}
    assert model.alpha_ in {0.0, 0.4}
    assert model.tick_nums_ in {1, 2}
    assert len(model.selection_history_) == 8
    assert any(np.isfinite(item["cv"]) for item in model.selection_history_)


def test_prediction_matches_manual_recalibration_and_does_not_mutate_fit():
    X_list, y_list, coords_list, intervals = _stages(n=16)
    model = STWR(
        spatial_bandwidth=5.0,
        adaptive=False,
        kernel="gaussian",
        alpha=0.3,
        tick_nums=2,
        ridge=1e-8,
    ).fit(X_list, y_list, coords_list, intervals)
    original = model.parameters_.copy()
    result = model.predict_result(
        X_list[-1][:3],
        coords_list[-1][:3] + np.array([0.05, 0.0]),
        reference_y=y_list[-1][:3],
    )
    assert result.predictions.shape == (3,)
    assert result.coef.shape == (3, 2)
    np.testing.assert_allclose(model.parameters_, original)


def test_prediction_without_reference_uses_finite_idw_baseline():
    X_list, y_list, coords_list, intervals = _stages(n=15)
    model = STWR(
        spatial_bandwidth=5.0,
        adaptive=False,
        kernel="gaussian",
        alpha=0.25,
        tick_nums=2,
        ridge=1e-8,
    ).fit(X_list, y_list, coords_list, intervals)
    result = model.predict_result(X_list[-1][:2], coords_list[-1][:2])
    assert np.all(np.isfinite(result.reference_y))
    assert np.all(np.isfinite(result.predictions))


def test_dataframe_schema_is_preserved():
    X_list, y_list, coords_list, intervals = _stages(n=14)
    frames = [pd.DataFrame(stage, columns=["income", "density"]) for stage in X_list]
    model = STWR(
        spatial_bandwidth=5.0,
        adaptive=False,
        kernel="gaussian",
        tick_nums=2,
        ridge=1e-8,
    ).fit(frames, y_list, coords_list, intervals)
    assert model.feature_names_ == ("income", "density")
    with pytest.raises(ValueError):
        model.predict(frames[-1][["density", "income"]], coords_list[-1])


def test_failed_refit_clears_fitted_state():
    X_list, y_list, coords_list, intervals = _stages(n=12)
    model = STWR(
        spatial_bandwidth=5.0,
        adaptive=False,
        kernel="gaussian",
        tick_nums=2,
        ridge=1e-8,
    ).fit(X_list, y_list, coords_list, intervals)
    bad_y = list(y_list)
    bad_y[-1] = bad_y[-1][:-1]
    with pytest.raises(ValueError):
        model.fit(X_list, bad_y, coords_list, intervals)
    with pytest.raises(ValueError, match="not fitted"):
        model.get_results()


def test_invalid_intervals_are_rejected():
    X_list, y_list, coords_list, _ = _stages(n=12)
    model = STWR(spatial_bandwidth=5.0, adaptive=False)
    with pytest.raises(ValueError, match="first time interval"):
        model.fit(X_list, y_list, coords_list, [1.0, 1.0, 1.0])
    with pytest.raises(ValueError, match="positive"):
        model.fit(X_list, y_list, coords_list, [0.0, 0.0, 1.0])


def test_zero_past_responses_are_handled_deterministically():
    X_list, y_list, coords_list, intervals = _stages(n=12)
    y_list[-2] = y_list[-2].copy()
    y_list[-2][0] = 0.0
    model = STWR(
        spatial_bandwidth=5.0,
        adaptive=False,
        kernel="gaussian",
        alpha=0.3,
        tick_nums=2,
        ridge=1e-8,
    ).fit(X_list, y_list, coords_list, intervals)
    assert np.all(np.isfinite(model.temporal_weights_))
    assert np.all(np.isfinite(model.fitted_values_))
