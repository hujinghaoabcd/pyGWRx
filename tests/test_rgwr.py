"""Reference and regression tests for classical robust GWR."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from scipy.spatial.distance import cdist

from pygwrx import GWR, RGWR

RIDGE = 1.0e-8


def _kernel_weights(distances, bandwidth, *, kernel, adaptive):
    distances = np.asarray(distances, dtype=float)
    if adaptive:
        k = int(bandwidth)
        local_bandwidth = float(np.partition(distances, k - 1)[k - 1])
        if local_bandwidth <= 0.0:
            local_bandwidth = float(np.min(distances[distances > 0.0]))
        local_bandwidth = float(np.nextafter(local_bandwidth, np.inf))
    else:
        local_bandwidth = float(bandwidth)

    if kernel == "gaussian":
        return np.exp(-0.5 * (distances / local_bandwidth) ** 2)
    if kernel == "bisquare":
        weights = np.zeros_like(distances)
        mask = distances < local_bandwidth
        weights[mask] = (1.0 - (distances[mask] / local_bandwidth) ** 2) ** 2
        return weights
    raise ValueError(kernel)


def _reference_local_fit(
    X_design, y, coords, robust_weights, *, bandwidth, kernel, adaptive
):
    distances = cdist(coords, coords)
    n_samples, n_parameters = X_design.shape
    params = np.empty((n_samples, n_parameters), dtype=float)
    fitted = np.empty(n_samples, dtype=float)
    hat = np.empty((n_samples, n_samples), dtype=float)

    for index, distance_row in enumerate(distances):
        spatial = _kernel_weights(
            distance_row,
            bandwidth,
            kernel=kernel,
            adaptive=adaptive,
        )
        weights = spatial * robust_weights
        system = X_design.T @ (weights[:, None] * X_design)
        system = system + RIDGE * np.eye(n_parameters)
        inverse = np.linalg.pinv(system)
        beta = inverse @ (X_design.T @ (weights * y))
        transform = inverse @ (X_design.T * weights)
        params[index] = beta
        fitted[index] = X_design[index] @ beta
        hat[index] = X_design[index] @ transform
    return params, fitted, hat


def _reference_automatic(
    X, y, coords, *, bandwidth, kernel, adaptive, cut1, cut2, tol, max_iter
):
    X_design = np.column_stack([np.ones(X.shape[0]), X])
    ones = np.ones(X.shape[0], dtype=float)
    params, fitted, hat = _reference_local_fit(
        X_design,
        y,
        coords,
        ones,
        bandwidth=bandwidth,
        kernel=kernel,
        adaptive=adaptive,
    )
    residuals = y - fitted
    mse = float(np.mean(residuals**2))

    def residual_weights(resid, value):
        scores = (
            np.zeros_like(resid)
            if value <= np.finfo(float).eps
            else np.abs(resid / np.sqrt(value))
        )
        result = np.ones_like(scores)
        transition = scores > cut1
        rejected = scores > cut2
        span = cut2 - cut1
        result[transition] = (1.0 - ((scores[transition] - cut1) / span) ** 2) ** 2
        result[rejected] = 0.0
        result[~np.isfinite(result)] = 0.0
        return result

    candidate = residual_weights(residuals, mse)
    used = ones
    converged = False
    n_iter = 0
    for iteration in range(max_iter):
        used = candidate.copy()
        params, fitted, hat = _reference_local_fit(
            X_design,
            y,
            coords,
            used,
            bandwidth=bandwidth,
            kernel=kernel,
            adaptive=adaptive,
        )
        residuals = y - fitted
        new_mse = float(np.mean(residuals**2))
        relative = (
            0.0 if new_mse <= np.finfo(float).eps else abs(mse - new_mse) / new_mse
        )
        n_iter = iteration + 1
        candidate = residual_weights(residuals, new_mse)
        if relative <= tol:
            converged = True
            break
        mse = new_mse
    return params, fitted, hat, used, n_iter, converged


def _reference_filtered(X, y, coords, *, bandwidth, kernel, adaptive, cut_filter):
    X_design = np.column_stack([np.ones(X.shape[0]), X])
    ones = np.ones(X.shape[0], dtype=float)
    initial_params, initial_fitted, initial_hat = _reference_local_fit(
        X_design,
        y,
        coords,
        ones,
        bandwidth=bandwidth,
        kernel=kernel,
        adaptive=adaptive,
    )
    residuals = y - initial_fitted
    trace_s = float(np.trace(initial_hat))
    trace_sts = float(np.sum(initial_hat**2))
    sigma2 = float(np.dot(residuals, residuals)) / (
        X.shape[0] - 2.0 * trace_s + trace_sts
    )
    q_diag = np.sum((np.eye(X.shape[0]) - initial_hat) ** 2, axis=0)
    studentized = residuals / np.sqrt(sigma2 * q_diag)
    weights = (np.isfinite(studentized) & (np.abs(studentized) < cut_filter)).astype(
        float
    )
    params, fitted, hat = _reference_local_fit(
        X_design,
        y,
        coords,
        weights,
        bandwidth=bandwidth,
        kernel=kernel,
        adaptive=adaptive,
    )
    return params, fitted, hat, weights, studentized


@pytest.fixture
def contaminated_data():
    rng = np.random.default_rng(417)
    n = 42
    coords = rng.uniform(0.0, 1.0, size=(n, 2))
    X = rng.normal(size=(n, 2))
    y = 1.2 + 1.8 * X[:, 0] - 0.9 * X[:, 1] + rng.normal(scale=0.18, size=n)
    y[[4, 19, 34]] += np.array([6.5, -5.5, 7.0])
    return X, y, coords


def test_automatic_matches_independent_gwmodel_translation(contaminated_data):
    X, y, coords = contaminated_data
    expected = _reference_automatic(
        X,
        y,
        coords,
        bandwidth=21,
        kernel="bisquare",
        adaptive=True,
        cut1=2.0,
        cut2=3.0,
        tol=1.0e-7,
        max_iter=20,
    )
    model = RGWR(
        kernel="bisquare",
        bandwidth=21,
        adaptive=True,
        method="automatic",
        cut1=2.0,
        cut2=3.0,
        tol=1.0e-7,
        max_iter=20,
    ).fit(X, y, coords, compute_local_r2=False)

    (
        expected_params,
        expected_fitted,
        expected_hat,
        expected_weights,
        n_iter,
        converged,
    ) = expected
    actual_params = np.column_stack([model.intercept_, model.coef_])
    np.testing.assert_allclose(actual_params, expected_params, rtol=2e-7, atol=2e-7)
    np.testing.assert_allclose(
        model.fitted_values_, expected_fitted, rtol=2e-7, atol=2e-7
    )
    np.testing.assert_allclose(model.hat_matrix_, expected_hat, rtol=2e-7, atol=2e-7)
    np.testing.assert_allclose(
        model.robust_weights_, expected_weights, rtol=1e-12, atol=1e-12
    )
    assert model.n_iter_ == n_iter
    assert model.converged_ is converged


def test_filtered_matches_independent_gwmodel_translation(contaminated_data):
    X, y, coords = contaminated_data
    expected = _reference_filtered(
        X,
        y,
        coords,
        bandwidth=0.62,
        kernel="gaussian",
        adaptive=False,
        cut_filter=3.0,
    )
    model = RGWR(
        kernel="gaussian",
        bandwidth=0.62,
        adaptive=False,
        method="filtered",
        cut_filter=3.0,
    ).fit(X, y, coords, compute_local_r2=False)

    (
        expected_params,
        expected_fitted,
        expected_hat,
        expected_weights,
        expected_studentized,
    ) = expected
    actual_params = np.column_stack([model.intercept_, model.coef_])
    np.testing.assert_allclose(actual_params, expected_params, rtol=2e-7, atol=2e-7)
    np.testing.assert_allclose(
        model.fitted_values_, expected_fitted, rtol=2e-7, atol=2e-7
    )
    np.testing.assert_allclose(model.hat_matrix_, expected_hat, rtol=2e-7, atol=2e-7)
    np.testing.assert_array_equal(model.robust_weights_, expected_weights)
    np.testing.assert_allclose(
        model.initial_studentized_residuals_,
        expected_studentized,
        rtol=2e-7,
        atol=2e-7,
    )
    assert set(np.unique(model.robust_weights_)).issubset({0.0, 1.0})
    assert model.n_iter_ == 1
    assert model.converged_


def test_clean_high_threshold_rgwr_reduces_to_standard_gwr():
    rng = np.random.default_rng(9)
    coords = rng.uniform(size=(36, 2))
    X = rng.normal(size=(36, 2))
    y = 0.7 + 1.1 * X[:, 0] - 0.4 * X[:, 1] + rng.normal(scale=0.05, size=36)
    gwr = GWR(kernel="bisquare", bandwidth=20, adaptive=True).fit(
        X, y, coords, compute_local_r2=False
    )
    rgwr = RGWR(
        kernel="bisquare",
        bandwidth=20,
        adaptive=True,
        method="automatic",
        cut1=20.0,
        cut2=30.0,
        tol=1.0e-10,
    ).fit(X, y, coords, compute_local_r2=False)
    np.testing.assert_allclose(rgwr.robust_weights_, 1.0, atol=0.0)
    np.testing.assert_allclose(rgwr.coef_, gwr.coef_, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(rgwr.intercept_, gwr.intercept_, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(
        rgwr.fitted_values_, gwr.fitted_values_, rtol=1e-10, atol=1e-10
    )


def test_rgwr_improves_known_surface_recovery_under_outliers():
    rng = np.random.default_rng(123)
    grid_x, grid_y = np.meshgrid(np.linspace(0.0, 1.0, 8), np.linspace(0.0, 1.0, 8))
    coords = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    X = rng.normal(size=(coords.shape[0], 2))
    intercept = 1.0 + 0.8 * coords[:, 0]
    beta_1 = 1.5 + 0.6 * np.sin(np.pi * coords[:, 1])
    beta_2 = -1.0 + 0.5 * coords[:, 0]
    mean = intercept + beta_1 * X[:, 0] + beta_2 * X[:, 1]
    y = mean + rng.normal(scale=0.12, size=coords.shape[0])
    outliers = np.array([5, 12, 31, 50])
    y[outliers] += np.array([8.0, -7.0, 9.0, -8.0])

    gwr = GWR(kernel="bisquare", bandwidth=28, adaptive=True).fit(
        X, y, coords, compute_local_r2=False
    )
    rgwr = RGWR(
        kernel="bisquare",
        bandwidth=28,
        adaptive=True,
        method="automatic",
        tol=1.0e-6,
    ).fit(X, y, coords, compute_local_r2=False)

    true_coef = np.column_stack([beta_1, beta_2])
    gwr_rmse = float(np.sqrt(np.mean((gwr.coef_ - true_coef) ** 2)))
    rgwr_rmse = float(np.sqrt(np.mean((rgwr.coef_ - true_coef) ** 2)))
    assert rgwr_rmse < 0.5 * gwr_rmse
    assert np.all(rgwr.robust_weights_[outliers] == 0.0)


def test_prediction_reuses_final_robust_weights(contaminated_data):
    X, y, coords = contaminated_data
    model = RGWR(
        kernel="bisquare",
        bandwidth=22,
        adaptive=True,
        method="automatic",
    ).fit(X, y, coords, compute_local_r2=False)
    predicted = model.predict(X, coords)
    np.testing.assert_allclose(predicted, model.fitted_values_, rtol=2e-7, atol=2e-7)


def test_dataframe_names_and_robust_columns(contaminated_data):
    X, y, coords = contaminated_data
    frame = pd.DataFrame(X, columns=["income", "population"])
    model = RGWR(
        bandwidth=22,
        adaptive=True,
        method="filtered",
    ).fit(frame, y, coords, compute_local_r2=False)
    assert list(model.feature_names_in_) == ["income", "population"]
    result = model.to_frame()
    for column in (
        "coef_income",
        "coef_population",
        "robust_weight",
        "downweighted",
        "robust_outlier",
        "initial_studentized_residual",
    ):
        assert column in result.columns


def test_filtered_can_discard_initial_hat_matrix(contaminated_data):
    X, y, coords = contaminated_data
    model = RGWR(
        bandwidth=0.8,
        method="filtered",
    ).fit(X, y, coords, compute_hat_matrix=False, compute_local_r2=False)
    assert model.hat_matrix_ is None
    assert model.initial_studentized_residuals_ is not None
    assert np.isfinite(model.diagnostics_["aicc"])


@pytest.mark.parametrize(
    "kwargs, error",
    [
        ({"method": "huber"}, ValueError),
        ({"max_iter": 0}, ValueError),
        ({"tol": 0.0}, ValueError),
        ({"cut1": -1.0}, ValueError),
        ({"cut1": 3.0, "cut2": 3.0}, ValueError),
        ({"cut_filter": 0.0}, ValueError),
    ],
)
def test_invalid_robust_parameters(kwargs, error):
    with pytest.raises(error):
        RGWR(**kwargs)


def test_failed_refit_clears_model_state(contaminated_data):
    X, y, coords = contaminated_data
    model = RGWR(bandwidth=22, adaptive=True).fit(X, y, coords, compute_local_r2=False)
    model.cut1 = 0.0
    model.cut2 = 1.0e-12
    with pytest.raises(RuntimeError):
        model.fit(X, y, coords, compute_local_r2=False)
    assert not model.is_fitted_
    assert model.coef_ is None
    assert model.robust_weights_ is None


def test_max_iter_warning_and_nonconvergence(contaminated_data):
    X, y, coords = contaminated_data
    model = RGWR(
        bandwidth=22,
        adaptive=True,
        method="automatic",
        max_iter=1,
        tol=1.0e-16,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(X, y, coords, compute_local_r2=False)
    assert model.n_iter_ == 1
    assert not model.converged_
    assert any("max_iter" in str(item.message) for item in caught)
