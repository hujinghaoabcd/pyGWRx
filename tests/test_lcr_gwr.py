# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Reference and regression tests for standard LCR-GWR.

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

from pygwrx import GWR, LCRGWR
from pygwrx.io.datasets import load_dublin_voter


def _adaptive_weights(distances: np.ndarray, k: int) -> np.ndarray:
    bandwidth = float(np.partition(distances, k - 1)[k - 1])
    if bandwidth <= 0.0:
        bandwidth = float(np.min(distances[distances > 0.0]))
    bandwidth = float(np.nextafter(bandwidth, np.inf))
    result = np.zeros_like(distances)
    mask = distances < bandwidth
    result[mask] = (1.0 - (distances[mask] / bandwidth) ** 2) ** 2
    return result


def _gwmodel_reference_fit(
    X: np.ndarray,
    y: np.ndarray,
    coords: np.ndarray,
    *,
    bandwidth: int,
    cn_thresh: float = 30.0,
    lambda_ridge: float = 0.0,
    lambda_adjust: bool = True,
):
    """Independent translation of GWmodel R/collinearity.r."""
    X_design = np.column_stack([np.ones(X.shape[0]), X])
    design_scales = np.r_[1.0, np.std(X, axis=0, ddof=1)]
    y_scale = float(np.std(y, ddof=1))
    distances = cdist(coords, coords)
    params = np.empty_like(X_design)
    fitted = np.empty(y.shape[0], dtype=float)
    conditions = np.empty(y.shape[0], dtype=float)
    lambdas = np.empty(y.shape[0], dtype=float)

    for index, distance_row in enumerate(distances):
        weights = _adaptive_weights(distance_row, bandwidth)
        weighted_for_cn = weights[:, None] * X_design
        normalized = weighted_for_cn / np.sqrt(np.sum(weighted_for_cn**2, axis=0))
        singular_values = np.linalg.svd(normalized, compute_uv=False)
        condition = singular_values[0] / singular_values[-1]
        local_lambda = float(lambda_ridge)
        if lambda_adjust and condition > cn_thresh:
            local_lambda = (singular_values[0] - cn_thresh * singular_values[-1]) / (
                cn_thresh - 1.0
            )

        X_weighted_scaled = np.sqrt(weights)[:, None] * X_design / design_scales
        y_weighted_scaled = np.sqrt(weights) * y / y_scale
        beta = np.linalg.solve(
            X_weighted_scaled.T @ X_weighted_scaled
            + local_lambda * np.eye(X_design.shape[1]),
            X_weighted_scaled.T @ y_weighted_scaled,
        )
        beta = beta * y_scale / design_scales
        params[index] = beta
        fitted[index] = X_design[index] @ beta
        conditions[index] = condition
        lambdas[index] = local_lambda
    return params, fitted, conditions, lambdas


@pytest.fixture
def collinear_data():
    rng = np.random.default_rng(2026)
    n_samples = 48
    coords = rng.uniform(0.0, 1.0, size=(n_samples, 2))
    x1 = rng.normal(size=n_samples)
    x2 = 0.985 * x1 + rng.normal(scale=0.035, size=n_samples)
    X = np.column_stack([x1, x2])
    y = 1.1 + 2.0 * x1 - 1.4 * x2 + rng.normal(scale=0.18, size=n_samples)
    return X, y, coords


def test_matches_independent_gwmodel_translation(collinear_data):
    X, y, coords = collinear_data
    expected_params, expected_fitted, expected_cn, expected_lambda = (
        _gwmodel_reference_fit(X, y, coords, bandwidth=24)
    )
    model = LCRGWR(
        kernel="bisquare",
        bandwidth=24,
        adaptive=True,
        lambda_ridge=0.0,
        lambda_adjust=True,
        cn_thresh=30.0,
    ).fit(
        X,
        y,
        coords,
        compute_local_r2=False,
        compute_inference=False,
        compute_cv=False,
    )

    actual_params = np.column_stack([model.intercept_, model.coef_])
    np.testing.assert_allclose(actual_params, expected_params, atol=2e-10, rtol=0.0)
    np.testing.assert_allclose(
        model.fitted_values_, expected_fitted, atol=2e-10, rtol=0.0
    )
    np.testing.assert_allclose(
        model.condition_numbers_, expected_cn, atol=2e-10, rtol=0.0
    )
    np.testing.assert_allclose(
        model.local_lambda_, expected_lambda, atol=2e-12, rtol=0.0
    )


def test_zero_lambda_without_adjustment_reduces_to_gwr():
    rng = np.random.default_rng(12)
    coords = rng.uniform(size=(42, 2))
    X = rng.normal(size=(42, 2))
    y = 0.8 + 1.4 * X[:, 0] - 0.6 * X[:, 1] + rng.normal(scale=0.1, size=42)

    gwr = GWR(kernel="bisquare", bandwidth=22, adaptive=True).fit(
        X, y, coords, compute_local_r2=False
    )
    lcr = LCRGWR(
        kernel="bisquare",
        bandwidth=22,
        adaptive=True,
        lambda_ridge=0.0,
        lambda_adjust=False,
    ).fit(X, y, coords, compute_local_r2=False, compute_cv=False)

    np.testing.assert_allclose(lcr.intercept_, gwr.intercept_, atol=1e-7, rtol=0.0)
    np.testing.assert_allclose(lcr.coef_, gwr.coef_, atol=1e-7, rtol=0.0)
    np.testing.assert_allclose(
        lcr.fitted_values_, gwr.fitted_values_, atol=1e-7, rtol=0.0
    )
    np.testing.assert_allclose(lcr.hat_matrix_, gwr.hat_matrix_, atol=1e-7, rtol=0.0)


def test_local_compensation_hits_requested_legacy_condition_threshold(collinear_data):
    X, y, coords = collinear_data
    model = LCRGWR(
        bandwidth=20,
        adaptive=True,
        lambda_adjust=True,
        cn_thresh=25.0,
    ).fit(X, y, coords, compute_local_r2=False, compute_cv=False)

    affected = model.locally_compensated_mask_
    assert np.any(affected)
    assert np.all(model.local_lambda_[affected] > 0.0)
    np.testing.assert_allclose(
        model.compensated_condition_numbers_[affected],
        25.0,
        atol=2e-12,
        rtol=0.0,
    )
    assert np.all(np.isfinite(model.penalized_system_condition_numbers_))


def test_penalized_hat_matrix_is_internally_consistent(collinear_data):
    X, y, coords = collinear_data
    model = LCRGWR(
        bandwidth=21,
        adaptive=True,
        lambda_adjust=True,
        cn_thresh=20.0,
    ).fit(X, y, coords, compute_local_r2=False, compute_cv=False)

    np.testing.assert_allclose(model.hat_matrix_ @ y, model.fitted_values_, atol=2e-10)
    assert model.diagnostics_["trace_S"] == pytest.approx(
        float(np.trace(model.hat_matrix_)), abs=1e-12
    )
    assert model.diagnostics_["trace_StS"] == pytest.approx(
        float(np.sum(model.hat_matrix_**2)), abs=1e-12
    )


def test_training_location_prediction_reproduces_fitted_values(collinear_data):
    X, y, coords = collinear_data
    model = LCRGWR(
        bandwidth=24,
        adaptive=True,
        lambda_adjust=True,
        cn_thresh=30.0,
    ).fit(X, y, coords, compute_local_r2=False, compute_cv=False)

    predictions = model.predict(X, coords)
    np.testing.assert_allclose(predictions, model.fitted_values_, atol=2e-10)
    diagnostics = model.get_local_diagnostics(coords[:4])
    assert list(diagnostics.columns) == [
        "coord_0",
        "coord_1",
        "condition_number",
        "local_lambda",
        "compensated_condition_number",
        "penalized_system_condition_number",
    ]


def test_automatic_cv_bandwidth_matches_exhaustive_search(collinear_data):
    X, y, coords = collinear_data
    scores = {}
    for bandwidth in range(18, 24):
        candidate = LCRGWR(
            bandwidth=bandwidth,
            adaptive=True,
            lambda_adjust=True,
            cn_thresh=30.0,
        ).fit(
            X,
            y,
            coords,
            compute_hat_matrix=False,
            compute_local_r2=False,
            compute_inference=False,
            compute_cv=True,
        )
        scores[bandwidth] = candidate.bandwidth_cv_score_

    model = LCRGWR(
        bandwidth="cv",
        adaptive=True,
        bandwidth_range=(18, 23),
        optimization_method="grid",
        lambda_adjust=True,
        cn_thresh=30.0,
    ).fit(
        X,
        y,
        coords,
        compute_hat_matrix=False,
        compute_local_r2=False,
        compute_inference=False,
        compute_cv=True,
    )
    expected = min(scores, key=scores.get)
    assert model.bandwidth_ == expected
    assert model.bandwidth_cv_score_ == pytest.approx(scores[expected], abs=1e-10)


def test_dataframe_schema_and_result_columns_are_preserved(collinear_data):
    X, y, coords = collinear_data
    frame = pd.DataFrame(X, columns=["income", "housing"])
    model = LCRGWR(bandwidth=24, adaptive=True).fit(
        frame, y, coords, compute_local_r2=False, compute_cv=True
    )
    result = model.to_frame()

    np.testing.assert_array_equal(model.feature_names_in_, ["income", "housing"])
    for column in (
        "intercept",
        "coef_income",
        "coef_housing",
        "condition_number",
        "local_lambda",
        "compensated_condition_number",
        "penalized_system_condition_number",
        "locally_compensated",
        "cv_residual",
        "cv_score",
    ):
        assert column in result.columns

    with pytest.raises(ValueError, match="Prediction DataFrame columns"):
        model.predict(frame[["housing", "income"]], coords)


def test_failed_refit_clears_previous_state(collinear_data):
    X, y, coords = collinear_data
    model = LCRGWR(bandwidth=24, adaptive=True).fit(
        X, y, coords, compute_local_r2=False, compute_cv=False
    )
    assert model.is_fitted_

    X_bad = X.copy()
    X_bad[:, 1] = 1.0
    with pytest.raises(ValueError, match="constant"):
        model.fit(X_bad, y, coords)
    assert not model.is_fitted_
    assert model.coef_ is None
    assert model.condition_numbers_ is None
    assert model.local_lambda_ is None


def test_dublin_voter_reference_statistics_are_reproduced():
    data = load_dublin_voter()
    feature_names = [
        "DiffAdd",
        "LARent",
        "SC1",
        "Unempl",
        "LowEduc",
        "Age18_24",
        "Age25_44",
        "Age45_64",
    ]
    model = LCRGWR(
        kernel="bisquare",
        bandwidth=157,
        adaptive=True,
        lambda_adjust=True,
        cn_thresh=30.0,
    ).fit(
        data[feature_names],
        data["GenEl2004"],
        data[["X", "Y"]],
        compute_hat_matrix=False,
        compute_local_r2=False,
        compute_inference=False,
        compute_cv=False,
    )

    # The bundled shapefile version is extremely close to the published
    # GWmodel summary (CN 34.34--73.72; lambda 0.01108--0.05374).
    assert np.min(model.condition_numbers_) == pytest.approx(34.48, abs=0.2)
    assert np.median(model.condition_numbers_) == pytest.approx(53.98, abs=0.3)
    assert np.max(model.condition_numbers_) == pytest.approx(75.29, abs=2.0)
    assert np.min(model.local_lambda_) == pytest.approx(0.01146, abs=6e-4)
    assert np.median(model.local_lambda_) == pytest.approx(0.04055, abs=8e-4)
    assert np.max(model.local_lambda_) == pytest.approx(0.05452, abs=1.5e-3)
    assert np.count_nonzero(model.locally_compensated_mask_) == 322


def test_local_zero_variance_predictor_is_stabilized():
    rng = np.random.default_rng(8)
    first_cluster = rng.normal(loc=(0.0, 0.0), scale=0.03, size=(20, 2))
    second_cluster = rng.normal(loc=(2.0, 2.0), scale=0.03, size=(20, 2))
    coords = np.vstack([first_cluster, second_cluster])
    x1 = rng.normal(size=40)
    x2 = np.r_[np.zeros(20), np.ones(20)]
    X = np.column_stack([x1, x2])
    y = 1.0 + 2.0 * x1 + 0.5 * x2 + rng.normal(scale=0.1, size=40)

    model = LCRGWR(
        bandwidth=12,
        adaptive=True,
        lambda_adjust=True,
        cn_thresh=30.0,
    ).fit(
        X,
        y,
        coords,
        compute_local_r2=False,
        compute_inference=False,
        compute_cv=False,
    )

    assert np.any(np.isinf(model.condition_numbers_))
    assert np.all(model.local_lambda_ > 0.0)
    assert np.all(np.isfinite(model.coef_))
    assert np.all(np.isfinite(model.fitted_values_))


def test_lcr_reduces_coefficient_error_under_severe_local_collinearity():
    rng = np.random.default_rng(2026)
    side = 10
    grid_x, grid_y = np.meshgrid(
        np.linspace(0.0, 1.0, side),
        np.linspace(0.0, 1.0, side),
    )
    coords = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    x1 = rng.normal(size=coords.shape[0])
    x2 = x1 + rng.normal(scale=0.02, size=coords.shape[0])
    X = np.column_stack([x1, x2])

    intercept = 1.0 + 0.3 * coords[:, 1]
    beta_1 = 2.0 + 0.8 * np.sin(2.0 * np.pi * coords[:, 0])
    beta_2 = -1.5 + 0.6 * np.cos(2.0 * np.pi * coords[:, 1])
    signal = intercept + beta_1 * x1 + beta_2 * x2
    y = signal + rng.normal(scale=0.3, size=coords.shape[0])
    truth = np.column_stack([intercept, beta_1, beta_2])

    gwr = GWR(kernel="bisquare", bandwidth=30, adaptive=True).fit(
        X,
        y,
        coords,
        compute_local_r2=False,
        compute_inference=False,
    )
    lcr = LCRGWR(
        kernel="bisquare",
        bandwidth=30,
        adaptive=True,
        lambda_adjust=True,
        cn_thresh=30.0,
    ).fit(
        X,
        y,
        coords,
        compute_local_r2=False,
        compute_inference=False,
        compute_cv=False,
    )

    gwr_params = np.column_stack([gwr.intercept_, gwr.coef_])
    lcr_params = np.column_stack([lcr.intercept_, lcr.coef_])
    gwr_rmse = float(np.sqrt(np.mean((gwr_params - truth) ** 2)))
    lcr_rmse = float(np.sqrt(np.mean((lcr_params - truth) ** 2)))

    assert np.count_nonzero(lcr.locally_compensated_mask_) == coords.shape[0]
    assert lcr_rmse < 0.25 * gwr_rmse


def test_summary_identifies_lcr_model(collinear_data):
    X, y, coords = collinear_data
    model = LCRGWR(bandwidth=24, adaptive=True).fit(
        X, y, coords, compute_local_r2=False, compute_cv=False
    )
    summary = model.summary()
    assert "Locally Compensated Ridge" in summary
    assert "Condition-number threshold" in summary
    assert "Locally compensated locations" in summary
    assert "trace(S)" in summary
    summary.encode("cp936")
