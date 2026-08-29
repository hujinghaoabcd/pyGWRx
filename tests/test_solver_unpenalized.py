"""Tests for the unpenalized weighted least-squares core."""

import numpy as np

from pygwrx.core import compute_hat_matrix, local_regression, weighted_least_squares
from pygwrx.core.kernels import gaussian_kernel


def test_default_matches_weighted_lstsq():
    rng = np.random.default_rng(123)
    X = np.column_stack([np.ones(12), rng.normal(size=(12, 2))])
    y = rng.normal(size=12)
    w = np.array([1.0, 0.8, 0.0, 0.4, 1.2, 0.6, 0.2, 0.0, 1.5, 0.7, 0.9, 0.3])
    beta, inverse_normal = weighted_least_squares(X, y, w)
    sw = np.sqrt(w)
    Xw = X * sw[:, None]
    yw = y * sw
    np.testing.assert_allclose(
        beta, np.linalg.lstsq(Xw, yw, rcond=None)[0], rtol=1e-11, atol=1e-12
    )
    np.testing.assert_allclose(
        inverse_normal, np.linalg.pinv(Xw.T @ Xw, hermitian=True), rtol=1e-9, atol=1e-11
    )


def test_rank_deficient_default_is_minimum_norm():
    x = np.linspace(-2.0, 2.0, 9)
    X = np.column_stack([np.ones_like(x), x, 2 * x])
    y = 3.0 + 4 * x
    w = np.linspace(0.3, 1.1, x.size)
    beta, _ = weighted_least_squares(X, y, w)
    sw = np.sqrt(w)
    expected = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)[0]
    np.testing.assert_allclose(beta, expected, rtol=1e-11, atol=1e-12)


def test_explicit_ridge_still_matches_closed_form():
    rng = np.random.default_rng(7)
    X = np.column_stack([np.ones(10), rng.normal(size=(10, 2))])
    y = rng.normal(size=10)
    w = np.linspace(0.2, 1.3, 10)
    ridge = 0.25
    beta, inverse_normal = weighted_least_squares(X, y, w, ridge=ridge)
    system = X.T @ (w[:, None] * X) + ridge * np.eye(X.shape[1])
    np.testing.assert_allclose(
        beta, np.linalg.solve(system, X.T @ (w * y)), rtol=1e-11, atol=1e-12
    )
    np.testing.assert_allclose(
        inverse_normal, np.linalg.inv(system), rtol=1e-10, atol=1e-11
    )


def test_feature_rescaling_preserves_fitted_values():
    rng = np.random.default_rng(77)
    X = np.column_stack([np.ones(30), rng.normal(size=30), rng.normal(size=30)])
    y = 1.5 + 2 * X[:, 1] - 0.4 * X[:, 2] + rng.normal(0.0, 0.01, 30)
    w = np.exp(-np.linspace(0.0, 2.0, 30))
    beta, _ = weighted_least_squares(X, y, w)
    X2 = X.copy()
    X2[:, 1] *= 1e4
    X2[:, 2] *= 1e-4
    beta2, _ = weighted_least_squares(X2, y, w)
    np.testing.assert_allclose(X2 @ beta2, X @ beta, rtol=1e-8, atol=1e-9)


def test_hat_matrix_matches_same_svd_wls_operator():
    rng = np.random.default_rng(314159)
    X = np.column_stack([np.ones(9), rng.normal(size=(9, 2))])
    coords = np.column_stack([np.linspace(0.0, 4.0, 9), rng.normal(scale=0.2, size=9)])
    bandwidth = 1.7

    hat = compute_hat_matrix(X, coords, gaussian_kernel, bandwidth)
    distances = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(axis=2))
    dummy_y = np.zeros(X.shape[0], dtype=float)

    expected = np.empty_like(hat)
    for i, dists in enumerate(distances):
        weights = gaussian_kernel(dists, bandwidth)
        _, inverse_normal = weighted_least_squares(X, dummy_y, weights)
        expected[i] = X[i] @ inverse_normal @ (X.T * weights)

    np.testing.assert_allclose(hat, expected, rtol=1e-11, atol=1e-12)


def test_rank_deficient_hat_matrix_matches_minimum_norm_local_predictions():
    x = np.linspace(-2.0, 2.0, 11)
    X = np.column_stack([np.ones_like(x), x, 2.0 * x])
    y = 2.5 + 3.2 * x
    coords = np.column_stack([x, 0.15 * x**2])
    bandwidth = 1.6

    hat = compute_hat_matrix(X, coords, gaussian_kernel, bandwidth)
    local_beta = local_regression(
        X,
        y,
        coords,
        coords,
        gaussian_kernel,
        bandwidth,
    )
    fitted_from_beta = np.einsum("ij,ij->i", X, local_beta)

    assert np.all(np.isfinite(hat))
    np.testing.assert_allclose(hat @ y, fitted_from_beta, rtol=1e-10, atol=1e-11)


def test_ridge_hat_matrix_matches_explicit_ridge_local_predictions():
    rng = np.random.default_rng(2718)
    X = np.column_stack([np.ones(10), rng.normal(size=(10, 2))])
    y = rng.normal(size=10)
    coords = np.column_stack(
        [np.linspace(0.0, 3.0, 10), rng.normal(scale=0.1, size=10)]
    )
    bandwidth = 1.2
    ridge = 0.15

    hat = compute_hat_matrix(
        X,
        coords,
        gaussian_kernel,
        bandwidth,
        ridge=ridge,
    )
    local_beta = local_regression(
        X,
        y,
        coords,
        coords,
        gaussian_kernel,
        bandwidth,
        ridge=ridge,
    )
    fitted_from_beta = np.einsum("ij,ij->i", X, local_beta)

    np.testing.assert_allclose(hat @ y, fitted_from_beta, rtol=1e-10, atol=1e-11)
