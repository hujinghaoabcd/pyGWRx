"""Numerical and API tests for Geo-Regime GWR.

The suite checks the model's defining properties rather than only array shapes:
connected contiguous regimes, direct local-WLS candidate costs, monotone accepted
objectives, conditional hat-matrix diagnostics, and direct recalibration at query
locations.
"""

from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import pytest

from pygwrx import GRGWR, GRGWRPredictionResult


@pytest.fixture(scope="module")
def offset_data():
    """Data with a large offset so the intercept has an observable effect."""
    rng = np.random.default_rng(3)
    n = 80
    coords = rng.random((n, 2)) * 10.0
    X = rng.random((n, 2))
    y = 50.0 + X @ np.array([2.0, -1.0]) + 0.2 * coords[:, 0]
    y += rng.normal(0.0, 0.1, n)
    return X, y, coords


@pytest.fixture(scope="module")
def regime_data():
    """Two-region process with a sharp left/right slope discontinuity."""
    rng = np.random.default_rng(0)
    n = 200
    coords = rng.random((n, 2)) * 10.0
    X = rng.random((n, 2))
    truth = (coords[:, 0] >= 5.0).astype(int)
    slope = np.where(truth == 0, 2.0, -2.0)
    y = slope * X[:, 0] + X[:, 1] + rng.normal(0.0, 0.1, n)
    return X, y, coords, truth


def _fit_regime_model(regime_data, **kwargs):
    X, y, coords, _ = regime_data
    settings = dict(
        n_regimes=2,
        bandwidth=30,
        fit_intercept=True,
        lambda_boundary=1.0,
        max_iter=10,
        random_state=42,
    )
    settings.update(kwargs)
    return GRGWR(**settings).fit(X, y, coords)


def test_fit_intercept_changes_shape_and_improves_fit(offset_data):
    X, y, coords = offset_data
    without = GRGWR(n_regimes=2, bandwidth=25, fit_intercept=False).fit(X, y, coords)
    with_intercept = GRGWR(n_regimes=2, bandwidth=25, fit_intercept=True).fit(
        X, y, coords
    )

    assert with_intercept.local_parameters_.shape[1] == (
        without.local_parameters_.shape[1] + 1
    )
    assert with_intercept.diagnostics_["r2"] > 0.9
    assert with_intercept.diagnostics_["r2"] > without.diagnostics_["r2"]


def test_recovers_sharp_spatial_regimes(regime_data):
    _, _, _, truth = regime_data
    model = _fit_regime_model(regime_data)
    agreement = max(np.mean(model.regimes_ == truth), np.mean(model.regimes_ != truth))
    assert agreement > 0.95


def test_labels_are_contiguous_nonempty_and_large_enough(regime_data):
    model = _fit_regime_model(regime_data)
    assert np.array_equal(np.unique(model.regimes_), np.arange(model.n_regimes_actual_))
    assert np.all(model.regime_sizes_ >= model._min_regime_size_)
    assert sum(model.regime_sizes_) == len(model.regimes_)
    assert all(block.shape[0] > 0 for block in model.coefficients_)


def test_all_reported_regimes_are_connected(regime_data):
    model = _fit_regime_model(regime_data)
    assert model.enforce_connectivity
    assert np.array_equal(
        model.regime_component_counts_, np.ones(model.n_regimes_actual_, dtype=int)
    )


def test_graph_is_symmetric_and_edges_are_unique(regime_data):
    model = _fit_regime_model(regime_data)
    graph = model.adjacency_matrix_.toarray().astype(bool)
    assert np.array_equal(graph, graph.T)
    assert not np.any(np.diag(graph))
    assert len(model.edges_) == len(set(model.edges_))
    assert all(i < j and graph[i, j] for i, j in model.edges_)
    assert len(model.regime_boundaries_) == model.diagnostics_["n_boundaries"]


def test_accepted_objective_history_is_monotone(regime_data):
    model = _fit_regime_model(regime_data)
    history = np.asarray(model.objective_history_)
    assert history.size >= 1
    assert np.all(np.diff(history) <= model.tol + 1e-12)
    assert model.diagnostics_["objective"] == pytest.approx(history[-1])


def test_candidate_error_matches_direct_local_wls(regime_data):
    model = _fit_regime_model(regime_data, max_iter=0)
    node = 0
    regime = int(model.regimes_[node])
    indices = np.flatnonzero(model.regimes_ == regime)
    indices = indices[indices != node]
    weights = model._weights(model.distance_matrix_[node, indices])
    beta, _ = model._solve_local(model._Xd[indices], model.y_[indices], weights)
    expected = float((model.y_[node] - model._Xd[node] @ beta) ** 2)
    assert model._candidate_error(node, regime, model.regimes_) == pytest.approx(
        expected, abs=1e-12
    )


def test_gamma_endpoints_have_exact_semantics(regime_data):
    X, y, coords, _ = regime_data
    coefficient_only = GRGWR(
        n_regimes=2,
        bandwidth=30,
        spatial_constraint_weight=0.0,
        max_iter=0,
    ).fit(X, y, coords)
    coordinate_only = GRGWR(
        n_regimes=2,
        bandwidth=30,
        spatial_constraint_weight=1.0,
        max_iter=0,
    ).fit(X, y, coords)
    mixed = GRGWR(
        n_regimes=2,
        bandwidth=30,
        spatial_constraint_weight=0.4,
        max_iter=0,
    ).fit(X, y, coords)

    assert coefficient_only.clustering_features_.shape[1] == X.shape[1]
    assert coordinate_only.clustering_features_.shape[1] == 2
    assert mixed.clustering_features_.shape[1] == X.shape[1] + 2
    expected_coords = (coords - coords.min(axis=0)) / np.ptp(coords, axis=0)
    np.testing.assert_allclose(
        coordinate_only.clustering_features_, expected_coords, atol=1e-12
    )


def test_conditional_diagnostics_match_hat_matrix(regime_data):
    model = _fit_regime_model(regime_data)
    diagnostics = model.diagnostics_
    for key in (
        "conditional_aic",
        "conditional_aicc",
        "conditional_bic",
        "conditional_enp",
    ):
        assert np.isfinite(diagnostics[key])
    assert diagnostics["conditional_enp"] == pytest.approx(np.trace(model.hat_matrix_))
    assert diagnostics["conditional_aicc"] == diagnostics["aicc"]
    assert diagnostics["conditional_enp_v2"] == diagnostics["enp"]


def test_training_predictions_equal_final_local_recalibration(regime_data):
    model = _fit_regime_model(regime_data)
    X, _, coords, _ = regime_data
    result = model.predict_result(X, coords)
    assert isinstance(result, GRGWRPredictionResult)
    np.testing.assert_allclose(result.predictions, model.fitted_values_, atol=1e-10)
    np.testing.assert_allclose(result.coefficients, model.coef_, atol=1e-10)
    np.testing.assert_allclose(result.intercepts, model.intercept_, atol=1e-10)
    np.testing.assert_array_equal(result.regimes, model.regimes_)


def test_prediction_result_frame_and_new_locations(regime_data):
    model = _fit_regime_model(regime_data)
    X, _, coords, _ = regime_data
    result = model.predict_result(X[:4], coords[:4] + 0.01)
    frame = result.to_frame()
    assert result.predictions.shape == (4,)
    assert np.all(np.isfinite(result.predictions))
    assert list(frame.columns) == [
        "coord_0",
        "coord_1",
        "prediction",
        "regime",
        "intercept",
        "coef_x0",
        "coef_x1",
    ]


def test_dataframe_names_and_prediction_order_validation(regime_data):
    X, y, coords, _ = regime_data
    frame = pd.DataFrame(X, columns=["income", "housing"])
    model = GRGWR(n_regimes=2, bandwidth=30).fit(frame, y, coords)
    assert model.feature_names_ == ("income", "housing")
    assert "coef_income" in model.to_frame().columns
    model.predict(frame.iloc[:3], coords[:3])
    with pytest.raises(ValueError, match="columns must match"):
        model.predict(frame[["housing", "income"]].iloc[:3], coords[:3])


def test_legacy_intercept_column_is_removed(regime_data):
    X, y, coords, _ = regime_data
    X_legacy = np.column_stack([np.ones(len(X)), X])
    with pytest.warns(UserWarning, match="all-ones"):
        model = GRGWR(n_regimes=2, bandwidth=30, fit_intercept=True).fit(
            X_legacy, y, coords
        )
    assert model.n_features_in_ == X.shape[1]
    assert model.local_parameters_.shape[1] == X.shape[1] + 1


def test_requested_regime_count_is_reduced_safely():
    rng = np.random.default_rng(15)
    X = rng.normal(size=(30, 2))
    coords = rng.uniform(size=(30, 2))
    y = 1.0 + X[:, 0] - X[:, 1] + rng.normal(scale=0.1, size=30)
    with pytest.warns(UserWarning, match="n_regimes reduced"):
        model = GRGWR(
            n_regimes=10,
            bandwidth=np.int64(8),
            min_regime_size=6,
            max_iter=0,
        ).fit(X, y, coords)
    assert model.n_regimes_actual_ <= 5
    assert np.all(model.regime_sizes_ >= 6)


def test_random_state_is_reproducible(regime_data):
    first = _fit_regime_model(regime_data, random_state=123)
    second = _fit_regime_model(regime_data, random_state=123)
    np.testing.assert_array_equal(first.regimes_, second.regimes_)
    np.testing.assert_allclose(first.local_parameters_, second.local_parameters_)
    np.testing.assert_allclose(first.objective_history_, second.objective_history_)


def test_pickle_round_trip_preserves_predictions(regime_data):
    model = _fit_regime_model(regime_data)
    X, _, coords, _ = regime_data
    restored = pickle.loads(pickle.dumps(model))
    np.testing.assert_allclose(
        restored.predict(X[:10], coords[:10]),
        model.predict(X[:10], coords[:10]),
        atol=1e-12,
    )


def test_failed_refit_clears_previous_state(regime_data):
    model = _fit_regime_model(regime_data)
    X, y, coords, _ = regime_data
    with pytest.raises(ValueError, match="same number of rows"):
        model.fit(X[:-1], y, coords)
    assert not model._is_fitted
    assert model.regimes_ is None
    with pytest.raises(ValueError, match="not fitted"):
        model.predict(X[:2], coords[:2])


@pytest.mark.parametrize(
    "kwargs, exception",
    [
        ({"n_regimes": 0}, ValueError),
        ({"bandwidth": 0}, ValueError),
        ({"kernel": "unknown"}, ValueError),
        ({"lambda_boundary": -1.0}, ValueError),
        ({"spatial_constraint_weight": 1.1}, ValueError),
        ({"n_neighbors": 0}, ValueError),
    ],
)
def test_constructor_validation(kwargs, exception):
    with pytest.raises(exception):
        GRGWR(**kwargs)


def test_conditional_parameter_selection_returns_best_fitted_model(regime_data):
    X, y, coords, _ = regime_data
    best, table = GRGWR.select_parameters(
        X,
        y,
        coords,
        n_regimes_grid=(2,),
        bandwidth_grid=(25, 30),
        lambda_boundary_grid=(1.0,),
        spatial_constraint_grid=(0.5,),
        criterion="conditional_aicc",
        max_iter=3,
        random_state=42,
    )
    assert best._is_fitted
    assert best.selection_criterion_ == "conditional_aicc"
    assert len(table) == 2
    assert np.all(np.diff(table["score"].to_numpy()) >= 0.0)
    assert best.diagnostics_["conditional_aicc"] == pytest.approx(
        table.iloc[0]["score"]
    )


def test_spatial_cv_parameter_selection_is_finite():
    rng = np.random.default_rng(91)
    n = 60
    coords = rng.uniform(0.0, 10.0, size=(n, 2))
    X = rng.normal(size=(n, 2))
    y = 1.0 + np.where(coords[:, 0] < 5.0, 1.5, -1.5) * X[:, 0]
    y += 0.5 * X[:, 1] + rng.normal(scale=0.2, size=n)
    best, table = GRGWR.select_parameters(
        X,
        y,
        coords,
        n_regimes_grid=(2,),
        bandwidth_grid=(12,),
        lambda_boundary_grid=(0.5,),
        spatial_constraint_grid=(0.5,),
        criterion="spatial_cv",
        cv_folds=3,
        max_iter=2,
        n_neighbors=5,
        random_state=4,
    )
    assert best._is_fitted
    assert best.selection_criterion_ == "spatial_cv"
    assert np.isfinite(table.iloc[0]["score"])
    assert len(table.iloc[0]["fold_scores"]) == 3
