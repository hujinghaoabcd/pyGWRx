"""Regression tests for rank-aware single-SVD weighted least squares."""

from __future__ import annotations

import numpy as np
import pytest

from pygwrx.core.solver import (
    _weighted_least_squares_details,
    weighted_least_squares,
)


def test_weighted_least_squares_uses_one_svd(monkeypatch: pytest.MonkeyPatch) -> None:
    rng = np.random.default_rng(20260829)
    X = np.column_stack([np.ones(24), rng.normal(size=(24, 3))])
    y = rng.normal(size=24)
    weights = np.linspace(0.2, 1.3, 24)

    original_svd = np.linalg.svd
    calls = 0

    def counting_svd(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_svd(*args, **kwargs)

    monkeypatch.setattr(np.linalg, "svd", counting_svd)
    beta, inverse_normal = weighted_least_squares(X, y, weights)

    assert calls == 1
    assert beta.shape == (X.shape[1],)
    assert inverse_normal.shape == (X.shape[1], X.shape[1])


def test_private_details_match_lstsq_and_report_full_rank() -> None:
    rng = np.random.default_rng(42)
    X = np.column_stack([np.ones(18), rng.normal(size=(18, 2))])
    y = rng.normal(size=18)
    weights = np.linspace(0.1, 1.0, 18)

    result = _weighted_least_squares_details(X, y, weights)
    sqrt_w = np.sqrt(weights)
    Xw = X * sqrt_w[:, None]
    yw = y * sqrt_w
    expected_beta = np.linalg.lstsq(Xw, yw, rcond=None)[0]

    np.testing.assert_allclose(result.beta, expected_beta, rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(
        result.inverse_normal,
        np.linalg.pinv(Xw.T @ Xw, hermitian=True),
        rtol=1e-9,
        atol=1e-11,
    )
    assert result.rank == X.shape[1]
    assert result.singular_values.shape == (X.shape[1],)
    assert np.all(np.diff(result.singular_values) <= 0.0)
    assert np.isfinite(result.condition_number)
    assert result.condition_number >= 1.0


def test_private_details_report_rank_deficiency_without_changing_minimum_norm() -> None:
    x = np.linspace(-2.0, 2.0, 15)
    X = np.column_stack([np.ones_like(x), x, 2.0 * x])
    y = 2.0 + 3.0 * x
    weights = np.linspace(0.25, 1.25, x.size)

    result = _weighted_least_squares_details(X, y, weights)
    sqrt_w = np.sqrt(weights)
    expected_beta = np.linalg.lstsq(
        X * sqrt_w[:, None],
        y * sqrt_w,
        rcond=None,
    )[0]

    assert result.rank == 2
    assert np.isinf(result.condition_number)
    np.testing.assert_allclose(result.beta, expected_beta, rtol=1e-11, atol=1e-12)

    public_beta, public_inverse = weighted_least_squares(X, y, weights)
    np.testing.assert_allclose(public_beta, result.beta, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        public_inverse,
        result.inverse_normal,
        rtol=0.0,
        atol=0.0,
    )
