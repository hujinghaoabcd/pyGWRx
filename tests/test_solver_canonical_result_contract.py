"""Architecture contract for the B6 canonical WLS result."""

from __future__ import annotations

import inspect
from dataclasses import FrozenInstanceError, is_dataclass

import numpy as np
import pytest

import pygwrx
import pygwrx.core as core
import pygwrx.core.solver as solver
from pygwrx.core.solver import (
    _solve_weighted_least_squares,
    _weighted_least_squares_details,
    _WeightedLeastSquaresResult,
    weighted_least_squares,
)


def _example_problem() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(20260830)
    X = np.column_stack([np.ones(20), rng.normal(size=(20, 2))])
    y = rng.normal(size=20)
    weights = np.linspace(0.15, 1.25, 20)
    return X, y, weights


def test_canonical_result_is_private_first_structured_and_frozen() -> None:
    X, y, weights = _example_problem()
    result = _solve_weighted_least_squares(X, y, weights)

    assert isinstance(result, _WeightedLeastSquaresResult)
    assert is_dataclass(result)
    assert result.params.shape == (X.shape[1],)
    assert result.inverse_normal.shape == (X.shape[1], X.shape[1])
    assert result.beta is result.params
    with pytest.raises(FrozenInstanceError):
        result.rank = 0

    assert "_WeightedLeastSquaresResult" not in solver.__all__
    assert "_solve_weighted_least_squares" not in solver.__all__
    assert not hasattr(core, "_WeightedLeastSquaresResult")
    assert not hasattr(core, "_solve_weighted_least_squares")
    assert not hasattr(pygwrx, "_WeightedLeastSquaresResult")
    assert not hasattr(pygwrx, "_solve_weighted_least_squares")


def test_pre_b6_private_details_name_is_exact_canonical_alias() -> None:
    assert _weighted_least_squares_details is _solve_weighted_least_squares


def test_public_wls_wrapper_keeps_signature_and_exact_numerical_bridge() -> None:
    signature = inspect.signature(weighted_least_squares)
    assert tuple(signature.parameters) == ("X", "y", "weights", "ridge")
    assert signature.parameters["ridge"].kind is inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["ridge"].default == 0.0

    X, y, weights = _example_problem()
    canonical = _solve_weighted_least_squares(X, y, weights)
    public_params, public_inverse = weighted_least_squares(X, y, weights)

    np.testing.assert_allclose(public_params, canonical.params, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        public_inverse, canonical.inverse_normal, rtol=0.0, atol=0.0
    )


def test_canonical_rank_deficient_result_preserves_minimum_norm_policy() -> None:
    x = np.linspace(-2.0, 2.0, 15)
    X = np.column_stack([np.ones_like(x), x, 2.0 * x])
    y = 2.0 + 3.0 * x
    weights = np.linspace(0.25, 1.25, x.size)

    result = _solve_weighted_least_squares(X, y, weights)
    sqrt_w = np.sqrt(weights)
    expected = np.linalg.lstsq(
        X * sqrt_w[:, None],
        y * sqrt_w,
        rcond=None,
    )[0]

    assert result.rank == 2
    assert np.isinf(result.condition_number)
    np.testing.assert_allclose(result.params, expected, rtol=1e-11, atol=1e-12)
