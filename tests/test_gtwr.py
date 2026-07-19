# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Reference and regression tests for standard GTWR.

Author:
    Jinghao Hu
"""

from __future__ import annotations

__author__ = "Jinghao Hu"
__license__ = "MIT"

import numpy as np
import pandas as pd
import pytest
from scipy.spatial.distance import cdist

from pygwrx import GTWR, GWR, GTWRPredictionResult

FUTURE_DISTANCE = 1.0e50


def _panel_data(seed: int = 2026, n_space: int = 8, n_time: int = 5):
    rng = np.random.default_rng(seed)
    angles = np.linspace(0.0, 2.0 * np.pi, n_space, endpoint=False)
    base_coords = np.column_stack([np.cos(angles), np.sin(angles)])
    coords = np.tile(base_coords, (n_time, 1))
    times = np.repeat(np.arange(n_time, dtype=float), n_space)
    x1 = rng.normal(size=coords.shape[0])
    x2 = rng.normal(size=coords.shape[0])
    X = np.column_stack([x1, x2])
    beta1 = 0.8 + 0.20 * coords[:, 0] + 0.12 * times
    beta2 = -0.5 + 0.15 * coords[:, 1] - 0.08 * times
    y = 1.2 + beta1 * x1 + beta2 * x2 + rng.normal(scale=0.08, size=X.shape[0])
    return X, y, coords, times


def _gwmodel_distances(
    coords: np.ndarray,
    times: np.ndarray,
    *,
    lambda_st: float,
    ksi: float,
    causal: bool,
) -> np.ndarray:
    spatial = cdist(coords, coords)
    delta = times[:, None] - times[None, :]
    temporal = (
        np.where(delta >= 0.0, delta, FUTURE_DISTANCE) if causal else np.abs(delta)
    )
    cross = (
        2.0
        * np.sqrt(np.maximum(lambda_st * (1.0 - lambda_st) * spatial * temporal, 0.0))
        * np.cos(ksi)
    )
    return np.maximum(
        lambda_st * spatial + (1.0 - lambda_st) * temporal + cross,
        0.0,
    )


def _euclidean_distances(
    coords: np.ndarray,
    times: np.ndarray,
    *,
    tau: float,
) -> np.ndarray:
    augmented = np.column_stack([coords, np.sqrt(tau) * times])
    return cdist(augmented, augmented)


def _reference_fit(
    X: np.ndarray,
    y: np.ndarray,
    distances: np.ndarray,
    *,
    bandwidth: float,
):
    design = np.column_stack([np.ones(X.shape[0]), X])
    params = np.empty_like(design)
    fitted = np.empty(X.shape[0], dtype=float)
    influence = np.empty(X.shape[0], dtype=float)
    covariance = np.empty_like(design)
    hat = np.empty((X.shape[0], X.shape[0]), dtype=float)
    for index, distance_row in enumerate(distances):
        weights = np.exp(-0.5 * (distance_row / bandwidth) ** 2)
        xtw = design.T * weights
        normal = xtw @ design
        inverse_xtx_xtw = np.linalg.solve(normal, xtw)
        beta = inverse_xtx_xtw @ y
        hat_row = design[index] @ inverse_xtx_xtw
        params[index] = beta
        fitted[index] = design[index] @ beta
        influence[index] = hat_row[index]
        covariance[index] = np.sum(inverse_xtx_xtw**2, axis=1)
        hat[index] = hat_row
    return params, fitted, influence, covariance, hat


def _reference_cv_score(
    X: np.ndarray,
    y: np.ndarray,
    distances: np.ndarray,
    *,
    bandwidth: int,
) -> float:
    design = np.column_stack([np.ones(X.shape[0]), X])
    fitted = np.empty(X.shape[0], dtype=float)
    for index, distance_row in enumerate(distances):
        local_bandwidth = float(
            np.nextafter(
                np.partition(distance_row, bandwidth - 1)[bandwidth - 1],
                np.inf,
            )
        )
        weights = np.zeros_like(distance_row)
        mask = distance_row < local_bandwidth
        weights[mask] = (1.0 - (distance_row[mask] / local_bandwidth) ** 2) ** 2
        weights[index] = 0.0
        xtw = design.T * weights
        beta = np.linalg.solve(xtw @ design, xtw @ y)
        fitted[index] = design[index] @ beta
    residuals = y - fitted
    return float(residuals @ residuals)


def test_default_gwmodel_distance_uses_absolute_time_differences():
    X, y, coords, times = _panel_data(n_space=6, n_time=3)
    model = GTWR(
        bandwidth=4.0,
        kernel="gaussian",
        lambda_st=0.35,
        ksi=0.4,
    ).fit(X, y, coords, times, compute_local_r2=False, compute_inference=False)
    expected = _gwmodel_distances(
        coords,
        times,
        lambda_st=0.35,
        ksi=0.4,
        causal=False,
    )
    np.testing.assert_allclose(model.spatiotemporal_distance_matrix_, expected)
    np.testing.assert_allclose(
        model.temporal_distance_matrix_,
        np.abs(times[:, None] - times[None, :]),
    )
    assert model.causal is False


def test_gwmodel_distance_matches_independent_formula_and_causality():
    X, y, coords, times = _panel_data(n_space=6, n_time=3)
    model = GTWR(
        bandwidth=4.0,
        kernel="gaussian",
        lambda_st=0.35,
        ksi=0.4,
        causal=True,
    ).fit(X, y, coords, times, compute_local_r2=False, compute_inference=False)
    expected = _gwmodel_distances(
        coords,
        times,
        lambda_st=0.35,
        ksi=0.4,
        causal=True,
    )
    np.testing.assert_allclose(model.spatiotemporal_distance_matrix_, expected)
    future = times[None, :] > times[:, None]
    assert np.all(model.spatiotemporal_distance_matrix_[future] > 1.0e40)


def test_fixed_gwmodel_fit_matches_independent_local_wls():
    X, y, coords, times = _panel_data()
    distances = _gwmodel_distances(
        coords,
        times,
        lambda_st=0.42,
        ksi=0.2,
        causal=False,
    )
    expected = _reference_fit(X, y, distances, bandwidth=1.8)
    model = GTWR(
        kernel="gaussian",
        bandwidth=1.8,
        lambda_st=0.42,
        ksi=0.2,
        causal=False,
        sigma2_v1=False,
    ).fit(X, y, coords, times, compute_local_r2=False)
    actual_params = np.column_stack([model.intercept_, model.coef_])
    np.testing.assert_allclose(actual_params, expected[0], atol=5e-9, rtol=0.0)
    np.testing.assert_allclose(model.fitted_values_, expected[1], atol=5e-9, rtol=0.0)
    np.testing.assert_allclose(model.influence_, expected[2], atol=5e-9, rtol=0.0)
    np.testing.assert_allclose(model.hat_matrix_, expected[4], atol=5e-9, rtol=0.0)
    assert model.diagnostics_["trace_S"] == pytest.approx(
        np.trace(expected[4]), abs=1e-8
    )
    assert model.diagnostics_["trace_StS"] == pytest.approx(
        np.sum(expected[4] ** 2), abs=1e-8
    )


def test_euclidean_mode_matches_public_python_gtwr_metric():
    X, y, coords, times = _panel_data(seed=11)
    distances = _euclidean_distances(coords, times, tau=1.7)
    expected = _reference_fit(X, y, distances, bandwidth=1.55)
    model = GTWR(
        kernel="gaussian",
        bandwidth=1.55,
        distance_combination="euclidean",
        tau=1.7,
        causal=False,
    ).fit(X, y, coords, times, compute_local_r2=False)
    np.testing.assert_allclose(
        np.column_stack([model.intercept_, model.coef_]),
        expected[0],
        atol=5e-9,
        rtol=0.0,
    )
    np.testing.assert_allclose(model.fitted_values_, expected[1], atol=5e-9, rtol=0.0)
    np.testing.assert_allclose(model.influence_, expected[2], atol=5e-9, rtol=0.0)
    np.testing.assert_allclose(
        model.parameter_covariance_diagonal_,
        expected[3] * model.sigma2_,
        atol=5e-9,
        rtol=0.0,
    )


def test_lambda_one_reduces_to_standard_gwr():
    X, y, coords, times = _panel_data(seed=31)
    gwr = GWR(
        kernel="gaussian",
        bandwidth=1.4,
        sigma2_v1=False,
    ).fit(X, y, coords, compute_local_r2=False)
    gtwr = GTWR(
        kernel="gaussian",
        bandwidth=1.4,
        lambda_st=1.0,
        causal=True,
        sigma2_v1=False,
    ).fit(X, y, coords, times, compute_local_r2=False)
    np.testing.assert_allclose(gtwr.intercept_, gwr.intercept_, atol=2e-10)
    np.testing.assert_allclose(gtwr.coef_, gwr.coef_, atol=2e-10)
    np.testing.assert_allclose(gtwr.fitted_values_, gwr.fitted_values_, atol=2e-10)
    np.testing.assert_allclose(gtwr.hat_matrix_, gwr.hat_matrix_, atol=2e-10)


def test_adaptive_cv_grid_matches_independent_leave_one_out_search():
    X, y, coords, times = _panel_data(seed=52, n_space=7, n_time=4)
    distances = _gwmodel_distances(
        coords,
        times,
        lambda_st=0.5,
        ksi=0.0,
        causal=False,
    )
    scores = {
        bandwidth: _reference_cv_score(
            X,
            y,
            distances,
            bandwidth=bandwidth,
        )
        for bandwidth in range(10, 14)
    }
    model = GTWR(
        kernel="bisquare",
        bandwidth="cv",
        adaptive=True,
        bandwidth_range=(10, 13),
        optimization_method="grid",
        lambda_st=0.5,
        causal=False,
    ).fit(X, y, coords, times, compute_local_r2=False, compute_inference=False)
    expected = min(scores, key=lambda value: (scores[value], value))
    assert model.bandwidth_ == expected
    assert model.bandwidth_score_ == pytest.approx(scores[expected], abs=1e-7)


def test_lambda_grid_selection_matches_recorded_finite_minimum():
    X, y, coords, times = _panel_data(seed=81, n_space=7, n_time=4)
    model = GTWR(
        kernel="gaussian",
        bandwidth=1.6,
        bandwidth_method="cv",
        lambda_st="auto",
        lambda_range=(0.2, 0.8),
        lambda_grid_size=4,
        causal=False,
    ).fit(X, y, coords, times, compute_local_r2=False, compute_inference=False)
    finite = [
        row for row in model.lambda_selection_history_ if np.isfinite(row["score"])
    ]
    expected = min(finite, key=lambda row: (row["score"], row["lambda_st"]))
    assert model.lambda_st_ == pytest.approx(expected["lambda_st"])
    assert model.bandwidth_ == pytest.approx(expected["bandwidth"])
    assert model.bandwidth_score_ == pytest.approx(expected["score"])


def test_no_hat_matrix_retains_exact_trace_diagnostics():
    X, y, coords, times = _panel_data(seed=90)
    full = GTWR(
        kernel="gaussian",
        bandwidth=1.7,
        lambda_st=0.4,
        causal=False,
    ).fit(X, y, coords, times, compute_local_r2=False, compute_hat_matrix=True)
    compact = GTWR(
        kernel="gaussian",
        bandwidth=1.7,
        lambda_st=0.4,
        causal=False,
    ).fit(X, y, coords, times, compute_local_r2=False, compute_hat_matrix=False)
    assert compact.hat_matrix_ is None
    np.testing.assert_allclose(compact.fitted_values_, full.fitted_values_)
    assert compact.diagnostics_["trace_S"] == pytest.approx(
        full.diagnostics_["trace_S"]
    )
    assert compact.diagnostics_["trace_StS"] == pytest.approx(
        full.diagnostics_["trace_StS"]
    )
    assert compact.diagnostics_["aicc"] == pytest.approx(full.diagnostics_["aicc"])


def test_datetime_conversion_prediction_and_result_export():
    X, y, coords, times = _panel_data(seed=101, n_space=6, n_time=4)
    date_times = pd.Timestamp("2026-01-01") + pd.to_timedelta(times, unit="h")
    frame = pd.DataFrame(X, columns=["mobility", "density"])
    model = GTWR(
        bandwidth=1.7,
        kernel="gaussian",
        lambda_st=0.5,
        causal=False,
        time_unit="auto",
    ).fit(frame, y, coords, date_times, compute_local_r2=False)
    assert model.time_unit_ == "hours"
    np.testing.assert_allclose(model.times_train_, times)
    result = model.predict_result(frame.iloc[:3], coords[:3], date_times[:3])
    assert isinstance(result, GTWRPredictionResult)
    assert result.predictions.shape == (3,)
    assert {"time", "prediction", "coef_mobility", "coef_density"} <= set(
        result.to_frame().columns
    )
    exported = model.to_frame()
    assert {"time", "coef_mobility", "coef_density", "influence"} <= set(
        exported.columns
    )
    with pytest.raises(ValueError, match="datetime-like"):
        model.predict(frame.iloc[:2], coords[:2], times[:2])


def test_prediction_matches_manual_new_location_local_regression():
    X, y, coords, times = _panel_data(seed=141)
    model = GTWR(
        kernel="gaussian",
        bandwidth=1.8,
        lambda_st=0.45,
        causal=False,
    ).fit(X, y, coords, times, compute_local_r2=False)
    X_new = X[:2] + np.array([[0.1, -0.1], [-0.2, 0.05]])
    coords_new = coords[:2] + np.array([[0.05, 0.03], [-0.04, 0.02]])
    times_new = times[:2] + 0.4
    spatial = cdist(coords_new, coords)
    temporal = np.abs(times_new[:, None] - times[None, :])
    combined = (
        0.45 * spatial
        + 0.55 * temporal
        + 2.0 * np.sqrt(0.45 * 0.55 * spatial * temporal)
    )
    design = np.column_stack([np.ones(X.shape[0]), X])
    expected = []
    for row, x_new in zip(combined, X_new):
        weights = np.exp(-0.5 * (row / 1.8) ** 2)
        xtw = design.T * weights
        beta = np.linalg.solve(xtw @ design, xtw @ y)
        expected.append(np.r_[1.0, x_new] @ beta)
    np.testing.assert_allclose(
        model.predict(X_new, coords_new, times_new),
        expected,
        atol=5e-9,
        rtol=0.0,
    )


def test_dataframe_schema_copy_safety_and_failed_refit_reset():
    X, y, coords, times = _panel_data(seed=171)
    frame = pd.DataFrame(X, columns=["x_a", "x_b"])
    model = GTWR(
        bandwidth=1.7,
        kernel="gaussian",
        lambda_st=0.4,
        causal=False,
    ).fit(frame, y, coords, times, compute_local_r2=False)
    old_coef = model.coef_.copy()
    X[:] = 999.0
    np.testing.assert_allclose(model.coef_, old_coef)
    with pytest.raises(ValueError, match="same order"):
        model.predict(frame[["x_b", "x_a"]].iloc[:2], coords[:2], times[:2])
    with pytest.raises(ValueError, match="same number of samples"):
        model.fit(frame, y[:-1], coords, times)
    assert model.is_fitted_ is False
    assert model.coef_ is None


def test_validation_and_manual_small_sample_fit():
    X, y, coords, times = _panel_data(seed=200, n_space=5, n_time=2)
    with pytest.raises(ValueError, match="lambda_st"):
        GTWR(lambda_st=1.2)
    with pytest.raises(ValueError, match="only available"):
        GTWR(distance_combination="euclidean", lambda_st="auto")
    with pytest.raises(ValueError, match="integer neighbour"):
        GTWR(bandwidth=5.5, adaptive=True)
    with pytest.raises(ValueError, match="minimum number"):
        GTWR(
            bandwidth=7,
            adaptive=True,
            lambda_st=0.4,
            causal=True,
        ).fit(X, y, coords, times)

    # A manual bandwidth must not be rejected merely because an optional LOO
    # score is undefined for a small causal sample.
    small = GTWR(
        kernel="gaussian",
        bandwidth=2.0,
        lambda_st=0.5,
        causal=True,
    ).fit(X, y, coords, times, compute_local_r2=False, compute_inference=False)
    assert small.is_fitted_ is True
    assert small.bandwidth_score_ is None or np.isnan(small.bandwidth_score_)


def test_spatiotemporal_simulation_improves_coefficient_recovery_over_gwr():
    rng = np.random.default_rng(404)
    n_space = 12
    n_time = 7
    angles = np.linspace(0.0, 2.0 * np.pi, n_space, endpoint=False)
    base = np.column_stack([np.cos(angles), np.sin(angles)])
    coords = np.tile(base, (n_time, 1))
    times = np.repeat(np.linspace(0.0, 3.0, n_time), n_space)
    x = rng.normal(size=coords.shape[0])
    X = x[:, None]
    beta = 0.5 + 0.35 * coords[:, 0] + 0.85 * np.sin(times * 1.4)
    y = 1.0 + beta * x + rng.normal(scale=0.06, size=x.size)

    gwr = GWR(kernel="gaussian", bandwidth=1.0).fit(
        X, y, coords, compute_local_r2=False, compute_inference=False
    )
    gtwr = GTWR(
        kernel="gaussian",
        bandwidth=0.9,
        distance_combination="euclidean",
        tau=1.8,
        causal=False,
    ).fit(X, y, coords, times, compute_local_r2=False, compute_inference=False)
    gwr_rmse = float(np.sqrt(np.mean((gwr.coef_[:, 0] - beta) ** 2)))
    gtwr_rmse = float(np.sqrt(np.mean((gtwr.coef_[:, 0] - beta) ** 2)))
    assert gtwr_rmse < 0.65 * gwr_rmse
