"""Numerical and API tests for similarity-geographically weighted regression."""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from pygwrx import GWR, SGWR


@pytest.fixture(scope="module")
def spatial_data():
    """Generate a stable spatially varying Gaussian regression problem."""
    rng = np.random.default_rng(20260717)
    n = 64
    coords = rng.uniform(0.0, 10.0, size=(n, 2))
    context = rng.normal(size=n)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    X = np.column_stack([x1, x2, context])
    local_slope = 1.4 + 0.9 * context + 0.04 * coords[:, 0]
    y = 0.8 + local_slope * x1 - 1.1 * x2 + rng.normal(0.0, 0.16, n)
    return X, y, coords


@pytest.fixture(scope="module")
def small_data():
    """Generate a compact dataset for exact local-matrix comparisons."""
    rng = np.random.default_rng(314159)
    n = 36
    coords = rng.uniform(-2.0, 2.0, size=(n, 2))
    X = rng.normal(size=(n, 2))
    y = 1.2 + 1.8 * X[:, 0] - 0.7 * X[:, 1] + rng.normal(0.0, 0.08, n)
    return X, y, coords


def _manual_similarity(X: np.ndarray, *, standardize: bool = True) -> np.ndarray:
    if standardize:
        scale = np.std(X, axis=0, ddof=0)
        scale = np.where(scale <= np.finfo(float).eps, 1.0, scale)
        Z = (X - np.mean(X, axis=0)) / scale
    else:
        Z = X
    distance = np.mean(np.abs(Z[:, None, :] - Z[None, :, :]), axis=2)
    return np.exp(-(distance**2))


def _manual_wls(
    X_design: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    system = X_design.T @ (X_design * weights[:, None])
    rhs = X_design.T @ (weights * y)
    return np.linalg.solve(system, rhs)


def test_published_similarity_formula_exact(small_data):
    X, y, coords = small_data
    model = SGWR(
        bandwidth=3.0,
        adaptive=False,
        kernel="gaussian",
        alpha=0.35,
        store_weights=True,
    ).fit(X, y, coords)

    expected = _manual_similarity(X)
    np.testing.assert_allclose(
        model.similarity_weights_, expected, rtol=0.0, atol=1e-14
    )
    np.testing.assert_allclose(np.diag(model.similarity_weights_), 1.0, atol=0.0)


def test_similarity_formula_without_standardization(small_data):
    X, y, coords = small_data
    model = SGWR(
        bandwidth=3.0,
        adaptive=False,
        kernel="gaussian",
        alpha=0.4,
        standardize_similarity=False,
    ).fit(X, y, coords)

    np.testing.assert_allclose(
        model.similarity_weights_,
        _manual_similarity(X, standardize=False),
        rtol=0.0,
        atol=1e-14,
    )


def test_similarity_standardization_is_shift_scale_invariant(spatial_data):
    X, y, coords = spatial_data
    transformed = X.copy()
    transformed[:, 2] = 17.0 + 8.5 * transformed[:, 2]

    base = SGWR(
        bandwidth=28,
        adaptive=True,
        alpha=0.45,
        similarity_vars=[2],
    ).fit(X, y, coords)
    shifted = SGWR(
        bandwidth=28,
        adaptive=True,
        alpha=0.45,
        similarity_vars=[2],
    ).fit(transformed, y, coords)

    np.testing.assert_allclose(
        base.similarity_weights_, shifted.similarity_weights_, rtol=0.0, atol=2e-14
    )


def test_alpha_one_matches_standard_gwr(spatial_data):
    X, y, coords = spatial_data
    sgwr = SGWR(
        bandwidth=28,
        adaptive=True,
        kernel="bisquare",
        alpha=1.0,
    ).fit(X, y, coords)
    gwr = GWR(
        bandwidth=28,
        adaptive=True,
        kernel="bisquare",
        fit_intercept=True,
        verbose=False,
    ).fit(X, y, coords)

    np.testing.assert_allclose(sgwr.intercept_, gwr.intercept_, rtol=0.0, atol=2e-8)
    np.testing.assert_allclose(sgwr.coef_, gwr.coef_, rtol=0.0, atol=2e-8)
    np.testing.assert_allclose(
        sgwr.fitted_values_, gwr.fitted_values_, rtol=0.0, atol=2e-8
    )


def test_local_fit_matches_explicit_weighted_least_squares(small_data):
    X, y, coords = small_data
    alpha = 0.37
    bandwidth = 2.4
    model = SGWR(
        bandwidth=bandwidth,
        adaptive=False,
        kernel="gaussian",
        alpha=alpha,
    ).fit(X, y, coords)

    X_design = np.column_stack([np.ones(X.shape[0]), X])
    for location in (0, 9, 21):
        beta = _manual_wls(X_design, y, model.combined_weights_[location])
        np.testing.assert_allclose(
            model.parameters_[location], beta, rtol=0.0, atol=2e-12
        )
        assert model.fitted_values_[location] == pytest.approx(
            X_design[location] @ beta, abs=2e-12
        )


def test_alpha_zero_is_similarity_only_wls(small_data):
    X, y, coords = small_data
    model = SGWR(
        bandwidth=3.0,
        adaptive=False,
        kernel="gaussian",
        alpha=0.0,
    ).fit(X, y, coords)
    X_design = np.column_stack([np.ones(X.shape[0]), X])
    expected = _manual_wls(X_design, y, model.similarity_weights_[5])
    np.testing.assert_allclose(model.parameters_[5], expected, atol=2e-12, rtol=0.0)


def test_automatic_alpha_minimizes_aicc_on_search_interval(spatial_data):
    X, y, coords = spatial_data
    common = dict(
        bandwidth=28,
        adaptive=True,
        kernel="bisquare",
        alpha_range=(0.05, 0.95),
    )
    selected = SGWR(alpha="aicc", alpha_grid_size=7, **common).fit(X, y, coords)
    lower = SGWR(alpha=0.05, **common).fit(X, y, coords)
    upper = SGWR(alpha=0.95, **common).fit(X, y, coords)

    assert 0.05 <= selected.alpha_ <= 0.95
    assert selected.alpha_score_ == pytest.approx(selected.diagnostics_["aicc"])
    assert len(selected.alpha_search_history_) >= 7
    assert (
        selected.diagnostics_["aicc"]
        <= min(lower.diagnostics_["aicc"], upper.diagnostics_["aicc"]) + 1e-7
    )


def test_automatic_bandwidth_reuses_standard_gwr_selector(spatial_data):
    X, y, coords = spatial_data
    model = SGWR(
        bandwidth="aicc",
        bandwidth_range=(20, 38),
        adaptive=True,
        bandwidth_kernel="bisquare",
        kernel="gaussian",
        alpha=0.4,
    ).fit(X, y, coords)

    assert model.bandwidth_selector_ is not None
    assert model.bandwidth_ == model.bandwidth_selector_.bandwidth_
    assert isinstance(model.bandwidth_, int)


def test_prediction_is_direct_local_recalibration(small_data):
    X, y, coords = small_data
    alpha = 0.42
    bandwidth = 2.8
    model = SGWR(
        bandwidth=bandwidth,
        adaptive=False,
        kernel="gaussian",
        alpha=alpha,
    ).fit(X, y, coords)

    X_new = np.array([[0.35, -0.2], [-0.8, 0.6]])
    coords_new = np.array([[0.4, -0.5], [-1.3, 1.1]])
    result = model.predict_result(X_new, coords_new)

    train_z = (X - model.similarity_mean_) / model.similarity_scale_
    query_z = (X_new - model.similarity_mean_) / model.similarity_scale_
    similarity_distance = np.mean(
        np.abs(query_z[:, None, :] - train_z[None, :, :]), axis=2
    )
    similarity = np.exp(-(similarity_distance**2))
    geographic_distance = np.sqrt(
        np.sum((coords_new[:, None, :] - coords[None, :, :]) ** 2, axis=2)
    )
    geographic = np.exp(-0.5 * (geographic_distance / bandwidth) ** 2)
    combined = alpha * geographic + (1.0 - alpha) * similarity
    combined /= np.max(combined, axis=1, keepdims=True)

    train_design = np.column_stack([np.ones(X.shape[0]), X])
    query_design = np.column_stack([np.ones(X_new.shape[0]), X_new])
    for location in range(X_new.shape[0]):
        beta = _manual_wls(train_design, y, combined[location])
        np.testing.assert_allclose(
            np.r_[result.intercept[location], result.coef[location]],
            beta,
            rtol=0.0,
            atol=2e-12,
        )
        assert result.predictions[location] == pytest.approx(
            query_design[location] @ beta, abs=2e-12
        )


def test_prediction_does_not_mutate_training_results(small_data):
    X, y, coords = small_data
    model = SGWR(bandwidth=2.8, adaptive=False, alpha=0.5).fit(X, y, coords)
    parameters = model.parameters_.copy()
    fitted = model.fitted_values_.copy()
    diagnostics = dict(model.diagnostics_)

    prediction = model.predict(X[:3], coords[:3] + 0.1)

    assert prediction.shape == (3,)
    np.testing.assert_array_equal(model.parameters_, parameters)
    np.testing.assert_array_equal(model.fitted_values_, fitted)
    assert model.diagnostics_ == diagnostics


def test_dataframe_names_results_and_prediction_frame(spatial_data):
    X, y, coords = spatial_data
    frame = pd.DataFrame(X, columns=["income", "age", "context"])
    model = SGWR(
        bandwidth=28,
        adaptive=True,
        alpha=0.4,
        similarity_vars=["context"],
    ).fit(frame, pd.Series(y), coords)

    assert model.feature_names_ == ("income", "age", "context")
    assert model.similarity_feature_names_ == ("context",)
    result_frame = model.results_frame()
    assert {"coef_income", "se_age", "t_context", "local_r2"}.issubset(
        result_frame.columns
    )
    prediction_frame = model.predict_result(frame.iloc[:2], coords[:2]).to_frame()
    assert {"prediction", "intercept", "coef_context"}.issubset(
        prediction_frame.columns
    )


def test_inference_and_diagnostics_are_finite(spatial_data):
    X, y, coords = spatial_data
    model = SGWR(bandwidth=30, adaptive=True, alpha=0.5).fit(X, y, coords)

    assert model.hat_matrix_.shape == (X.shape[0], X.shape[0])
    assert model.parameter_standard_errors_.shape == model.parameters_.shape
    assert np.all(np.isfinite(model.parameter_standard_errors_))
    assert np.all(np.isfinite(model.influence_))
    for key in ("r2", "adj_r2", "aicc", "trace_S", "trace_StS", "enp"):
        assert np.isfinite(model.diagnostics_[key])
    summary = model.summary()
    assert "bandwidth" in summary
    assert "alpha" in summary


def test_store_weights_false_avoids_training_weight_matrices(small_data):
    X, y, coords = small_data
    model = SGWR(
        bandwidth=2.8,
        adaptive=False,
        alpha=0.5,
        store_weights=False,
    ).fit(X, y, coords)

    assert model.spatial_weights_ is None
    assert model.similarity_weights_ is None
    assert model.combined_weights_ is None
    assert model.predict(X[:2], coords[:2]).shape == (2,)


def test_no_intercept_mode(small_data):
    X, y, coords = small_data
    model = SGWR(
        bandwidth=3.0,
        adaptive=False,
        alpha=0.5,
        fit_intercept=False,
    ).fit(X, y, coords)

    assert model.parameters_.shape == X.shape
    np.testing.assert_array_equal(model.intercept_, np.zeros(X.shape[0]))
    assert model.coef_.shape == X.shape


@pytest.mark.parametrize(
    ("kwargs", "error_type"),
    [
        ({"alpha_range": (0.7, 0.2)}, ValueError),
        ({"alpha_grid_size": 2}, ValueError),
        ({"ridge": -1.0}, ValueError),
    ],
)
def test_constructor_validation(kwargs, error_type):
    with pytest.raises(error_type):
        SGWR(**kwargs)


def test_similarity_metric_is_fixed_by_the_model_definition():
    assert "similarity_metric" not in inspect.signature(SGWR).parameters


def test_fit_validation_and_prediction_column_order(small_data):
    X, y, coords = small_data
    with pytest.raises(ValueError, match="alpha"):
        SGWR(bandwidth=20, alpha=1.2).fit(X, y, coords)
    with pytest.raises(ValueError, match="Unknown similarity variable"):
        SGWR(bandwidth=20, alpha=0.5, similarity_vars=["missing"]).fit(
            pd.DataFrame(X, columns=["a", "b"]), y, coords
        )
    with pytest.raises(ValueError, match="at least"):
        SGWR(bandwidth=2, adaptive=True, alpha=0.5).fit(X, y, coords)

    frame = pd.DataFrame(X, columns=["a", "b"])
    fitted = SGWR(bandwidth=20, adaptive=True, alpha=0.5).fit(frame, y, coords)
    with pytest.raises(ValueError, match="same order"):
        fitted.predict(frame[["b", "a"]].iloc[:2], coords[:2])


def test_failed_refit_clears_previous_state(small_data):
    X, y, coords = small_data
    model = SGWR(bandwidth=20, adaptive=True, alpha=0.5).fit(X, y, coords)
    invalid_y = y.copy()
    invalid_y[0] = np.nan

    with pytest.raises(ValueError, match="NaN"):
        model.fit(X, invalid_y, coords)

    assert not model._is_fitted
    assert model.parameters_ is None
    assert model.fitted_values_ is None
    with pytest.raises(ValueError, match="not fitted"):
        model.predict(X[:1], coords[:1])
