"""Tests for the unpenalized weighted least-squares core."""

import numpy as np

from pygwrx.core import weighted_least_squares


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
