"""Standard geographically weighted Lasso tests."""

import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Lasso

from pygwrx.core.kernels import get_kernel_function
from pygwrx.core.solver import adaptive_bandwidth_weights
from pygwrx.core.utils import compute_distance_matrix
from pygwrx.models.gw_lasso import GWLasso


def _data(seed=41, n=42):
    rng = np.random.default_rng(seed)
    coords = rng.uniform(-1.5, 1.5, size=(n, 2))
    X = rng.normal(size=(n, 3))
    beta0 = 1.1 + 0.25 * coords[:, 0]
    beta1 = 1.8 + 0.35 * coords[:, 1]
    beta2 = np.where(coords[:, 0] < 0.0, -1.2, 0.0)
    beta3 = np.zeros(n)
    y = (
        beta0
        + beta1 * X[:, 0]
        + beta2 * X[:, 1]
        + beta3 * X[:, 2]
        + rng.normal(0.0, 0.08, n)
    )
    return X, y, coords


def _weights(model, coords, location=0):
    distances = compute_distance_matrix(
        coords[[location]], coords, metric=model.distance_metric
    )[0]
    if model.adaptive:
        bw = adaptive_bandwidth_weights(distances, int(model.bandwidth_))
    else:
        bw = float(model.bandwidth_)
    return np.asarray(get_kernel_function(model.kernel)(distances, bw), dtype=float)


def _independent_lasso(X, y, weights, alpha):
    positive = weights > 0
    X_local = X[positive]
    y_local = y[positive]
    w_local = weights[positive]
    x_mean = np.average(X_local, axis=0, weights=w_local)
    y_mean = np.average(y_local, weights=w_local)
    X_centered = X_local - x_mean
    x_scale = np.sqrt(
        np.sum(w_local[:, None] * X_centered**2, axis=0) / np.sum(w_local)
    )
    x_scale[x_scale <= np.sqrt(np.finfo(float).eps)] = 1.0
    X_scaled = X_centered / x_scale
    model = Lasso(
        alpha=alpha,
        fit_intercept=False,
        max_iter=10000,
        tol=1e-10,
        selection="cyclic",
    ).fit(X_scaled, y_local - y_mean, sample_weight=w_local)
    coef = model.coef_ / x_scale
    intercept = y_mean - x_mean @ coef
    return coef, intercept


def test_fixed_alpha_matches_independent_weighted_lasso():
    X, y, coords = _data()
    model = GWLasso(
        kernel="bisquare",
        bandwidth=18,
        adaptive=True,
        alpha=0.08,
        tol=1e-10,
        max_iter=10000,
    ).fit(X, y, coords)
    weights = _weights(model, coords)
    expected_coef, expected_intercept = _independent_lasso(X, y, weights, alpha=0.08)
    np.testing.assert_allclose(model.coef_[0], expected_coef, rtol=1e-9, atol=1e-9)
    assert model.intercept_[0] == pytest.approx(expected_intercept, rel=1e-9, abs=1e-9)


def test_zero_penalty_matches_explicit_weighted_least_squares():
    X, y, coords = _data(seed=42)
    model = GWLasso(
        kernel="gaussian",
        bandwidth=0.9,
        adaptive=False,
        alpha=0.0,
    ).fit(X, y, coords)
    for location in (0, 7, 19):
        weights = _weights(model, coords, location)
        design = np.column_stack((np.ones(len(X)), X))
        sqrt_w = np.sqrt(weights)
        expected = np.linalg.lstsq(design * sqrt_w[:, None], y * sqrt_w, rcond=None)[0]
        assert model.intercept_[location] == pytest.approx(expected[0], rel=1e-9)
        np.testing.assert_allclose(model.coef_[location], expected[1:], rtol=1e-9)


def test_larger_penalty_increases_sparsity():
    X, y, coords = _data(seed=43)
    weak = GWLasso(bandwidth=20, adaptive=True, alpha=0.01).fit(X, y, coords)
    strong = GWLasso(bandwidth=20, adaptive=True, alpha=0.4).fit(X, y, coords)
    weak_active = np.count_nonzero(np.abs(weak.coef_) > weak.active_tol)
    strong_active = np.count_nonzero(np.abs(strong.coef_) > strong.active_tol)
    assert strong_active < weak_active


def test_local_alpha_cv_is_deterministic_and_uses_grid():
    X, y, coords = _data(seed=44, n=36)
    kwargs = dict(
        bandwidth=16,
        adaptive=True,
        alpha="cv",
        alpha_grid=[0.4, 0.15, 0.04],
        cv_folds=3,
        random_state=9,
    )
    first = GWLasso(**kwargs).fit(X, y, coords)
    second = GWLasso(**kwargs).fit(X, y, coords)
    np.testing.assert_array_equal(first.alpha_, second.alpha_)
    np.testing.assert_allclose(first.coef_, second.coef_)
    assert set(np.unique(first.alpha_)).issubset({0.4, 0.15, 0.04})
    assert np.all(np.isfinite(first.local_alpha_cv_score_))


def test_adaptive_bandwidth_is_neighbor_count():
    X, y, coords = _data(seed=45)
    model = GWLasso(kernel="bisquare", bandwidth=14, adaptive=True, alpha=0.05).fit(
        X, y, coords
    )
    assert model.bandwidth_ == 14
    distances = compute_distance_matrix(coords[[0]], coords)[0]
    distance_bw = adaptive_bandwidth_weights(distances, 14)
    weights = get_kernel_function("bisquare")(distances, distance_bw)
    assert np.count_nonzero(weights > 0) <= 14
    assert np.all(np.isfinite(model.coef_))


def test_adaptive_bandwidth_cv_records_candidates():
    X, y, coords = _data(seed=46, n=30)
    model = GWLasso(
        bandwidth="adaptive",
        alpha=0.08,
        n_bandwidths=4,
        bandwidth_range=(10, 20),
    ).fit(X, y, coords)
    assert isinstance(model.bandwidth_, int)
    assert 10 <= model.bandwidth_ <= 20
    assert list(model.bandwidth_scores_.columns) == [
        "bandwidth",
        "rmse",
        "failed_locations",
    ]
    assert np.isfinite(model.bandwidth_scores_["rmse"]).any()


def test_fixed_bandwidth_cv_records_candidates():
    X, y, coords = _data(seed=47, n=28)
    model = GWLasso(
        kernel="gaussian",
        bandwidth="cv",
        adaptive=False,
        alpha=0.06,
        n_bandwidths=4,
        bandwidth_range=(0.6, 1.8),
    ).fit(X, y, coords)
    assert isinstance(model.bandwidth_, float)
    assert 0.6 <= model.bandwidth_ <= 1.8
    assert len(model.bandwidth_scores_) == 4


def test_prediction_recalibrates_and_preserves_training_state():
    X, y, coords = _data(seed=48)
    model = GWLasso(bandwidth=18, adaptive=True, alpha=0.07).fit(X, y, coords)
    old_coef = model.coef_.copy()
    old_fitted = model.fitted_values_.copy()
    coords0 = coords[:4] + np.array([0.04, -0.03])
    X0 = X[:4] + 0.05
    result = model.predict_parameters(coords0)
    prediction = model.predict(X0, coords0)
    expected = result.intercepts + np.einsum("ij,ij->i", X0, result.coefficients)
    np.testing.assert_allclose(prediction, expected)
    assert result.coefficients.shape == (4, X.shape[1])
    np.testing.assert_array_equal(model.coef_, old_coef)
    np.testing.assert_array_equal(model.fitted_values_, old_fitted)


def test_training_location_prediction_matches_fixed_alpha_fit():
    X, y, coords = _data(seed=49)
    model = GWLasso(bandwidth=18, adaptive=True, alpha=0.06).fit(X, y, coords)
    prediction = model.predict(X, coords)
    np.testing.assert_allclose(prediction, model.fitted_values_, rtol=1e-11, atol=1e-11)


def test_dataframe_metadata_and_column_order_validation():
    X, y, coords = _data(seed=50)
    frame = pd.DataFrame(X, columns=["income", "housing", "noise"])
    model = GWLasso(bandwidth=18, adaptive=True, alpha=0.05).fit(
        frame,
        pd.Series(y),
        pd.DataFrame(coords, columns=["x", "y"]),
    )
    assert tuple(model.feature_names_in_) == ("income", "housing", "noise")
    with pytest.raises(ValueError, match="columns"):
        model.predict(frame[["noise", "housing", "income"]], coords)


def test_selection_frequency_and_active_variables_are_consistent():
    X, y, coords = _data(seed=51)
    model = GWLasso(bandwidth=17, adaptive=True, alpha=0.12).fit(X, y, coords)
    expected = np.mean(np.abs(model.coef_) > model.active_tol, axis=0)
    np.testing.assert_allclose(model.get_variable_importance(), expected)
    for index, active in enumerate(model.active_vars_):
        np.testing.assert_array_equal(
            active, np.flatnonzero(np.abs(model.coef_[index]) > model.active_tol)
        )


def test_to_frame_contains_coefficients_and_diagnostics():
    X, y, coords = _data(seed=52)
    frame = pd.DataFrame(X, columns=["x1", "x2", "x3"])
    model = GWLasso(bandwidth=18, adaptive=True, alpha=0.08).fit(frame, y, coords)
    output = model.to_frame()
    assert output.shape[0] == len(y)
    for name in (
        "coord_0",
        "coord_1",
        "intercept",
        "alpha",
        "prediction",
        "coef_x1",
        "selected_x1",
        "residual",
        "objective",
        "converged",
    ):
        assert name in output.columns


def test_no_intercept_mode_matches_weighted_origin_regression():
    X, y, coords = _data(seed=53)
    model = GWLasso(
        bandwidth=0.8,
        kernel="gaussian",
        alpha=0.0,
        fit_intercept=False,
    ).fit(X, y, coords)
    weights = _weights(model, coords)
    sqrt_w = np.sqrt(weights)
    expected = np.linalg.lstsq(X * sqrt_w[:, None], y * sqrt_w, rcond=None)[0]
    assert model.intercept_[0] == 0.0
    np.testing.assert_allclose(model.coef_[0], expected, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize(
    "kwargs, error",
    [
        ({"alpha": -0.1}, ValueError),
        ({"alpha": "aicc"}, ValueError),
        ({"alpha_grid": [0.1, 0.0]}, ValueError),
        ({"cv_folds": 1}, ValueError),
        ({"n_bandwidths": 1}, ValueError),
    ],
)
def test_invalid_parameters_are_rejected(kwargs, error):
    with pytest.raises(error):
        GWLasso(**kwargs)


def test_bandwidth_method_is_not_a_single_choice_public_parameter():
    assert "bandwidth_method" not in inspect.signature(GWLasso).parameters


def test_fit_failure_clears_previous_state():
    X, y, coords = _data(seed=54)
    model = GWLasso(bandwidth=18, adaptive=True, alpha=0.05).fit(X, y, coords)
    assert model._is_fitted
    bad = X.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        model.fit(bad, y, coords)
    assert not model._is_fitted
    assert model.coef_ is None
    assert model.alpha_ is None


def test_known_spatial_selection_pattern_is_recovered():
    rng = np.random.default_rng(55)
    n = 80
    coords = np.column_stack((np.linspace(-2.0, 2.0, n), np.zeros(n)))
    X = rng.normal(size=(n, 3))
    beta2 = np.where(coords[:, 0] < 0.0, 1.8, 0.0)
    y = 1.0 + 2.2 * X[:, 0] + beta2 * X[:, 1] + rng.normal(0.0, 0.05, n)
    model = GWLasso(
        kernel="gaussian",
        bandwidth=0.55,
        alpha=0.13,
    ).fit(X, y, coords)
    left = np.mean(np.abs(model.coef_[coords[:, 0] < -0.5, 1]) > model.active_tol)
    right = np.mean(np.abs(model.coef_[coords[:, 0] > 0.5, 1]) > model.active_tol)
    assert left > 0.8
    assert right < 0.5
    assert model.selection_frequency_[2] < 0.5


def test_public_columbus_example_runs_reproducibly():
    package_root = Path(__file__).resolve().parents[1]
    path = package_root / "src" / "pygwrx" / "data" / "Columbus" / "columbus.csv"
    data = pd.read_csv(path, encoding="utf-8-sig")
    X = data[["INC", "HOVAL"]]
    y = data["CRIME"]
    coords = data[["X", "Y"]]
    kwargs = dict(
        kernel="exponential",
        bandwidth=24,
        adaptive=True,
        alpha="cv",
        alpha_grid=[0.5, 0.15, 0.04],
        cv_folds=3,
        random_state=4,
    )
    first = GWLasso(**kwargs).fit(X, y, coords)
    second = GWLasso(**kwargs).fit(X, y, coords)
    np.testing.assert_allclose(first.coef_, second.coef_)
    np.testing.assert_array_equal(first.alpha_, second.alpha_)
    assert first.coef_.shape == (49, 2)
    assert np.all(np.isfinite(first.fitted_values_))
    assert 0.0 <= first.diagnostics_["r2"] <= 1.0
