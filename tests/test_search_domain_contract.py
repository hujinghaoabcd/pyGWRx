# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Architecture contracts for B7 generic optimization search domains."""

from __future__ import annotations

import dataclasses
import inspect

import numpy as np
import pytest

import pygwrx
import pygwrx.core as core
from pygwrx import GTWR
from pygwrx.core.optimization import (
    GoldenSectionSearch,
    _ContinuousSearchDomain,
    _IntegerSearchDomain,
)


def _result_tuple(result):
    return (
        result.value,
        result.score,
        result.iterations,
        result.converged,
        result.evaluations,
        result.message,
    )


def test_search_domains_are_private_frozen_value_objects() -> None:
    continuous = _ContinuousSearchDomain.from_bounds(1, 4)
    integer = _IntegerSearchDomain.from_bounds(1.2, 5.8)

    assert dataclasses.is_dataclass(continuous)
    assert dataclasses.is_dataclass(integer)
    with pytest.raises(dataclasses.FrozenInstanceError):
        continuous.lower = 0.0  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        integer.upper = 9.0  # type: ignore[misc]

    assert continuous == _ContinuousSearchDomain(1.0, 4.0)
    assert integer == _IntegerSearchDomain(1.2, 5.8)
    assert "_ContinuousSearchDomain" not in core.__all__
    assert "_IntegerSearchDomain" not in core.__all__
    assert not hasattr(core, "_ContinuousSearchDomain")
    assert not hasattr(core, "_IntegerSearchDomain")
    assert not hasattr(pygwrx, "_ContinuousSearchDomain")
    assert not hasattr(pygwrx, "_IntegerSearchDomain")


def test_public_golden_signature_preserves_adaptive_compatibility_wrapper() -> None:
    signature = inspect.signature(GoldenSectionSearch.minimize)
    assert tuple(signature.parameters) == (
        "self",
        "func",
        "lower",
        "upper",
        "adaptive",
    )
    assert signature.parameters["adaptive"].default is False


def test_integer_compatibility_wrapper_matches_explicit_domain_and_trajectory() -> None:
    wrapper_trace = []
    domain_trace = []

    def wrapper_objective(value):
        wrapper_trace.append(int(value))
        return float((int(value) - 7) ** 2)

    def domain_objective(value):
        domain_trace.append(int(value))
        return float((int(value) - 7) ** 2)

    wrapper_optimizer = GoldenSectionSearch(tol=1e-5, max_iter=100, verbose=False)
    domain_optimizer = GoldenSectionSearch(tol=1e-5, max_iter=100, verbose=False)
    wrapper_result = wrapper_optimizer.minimize(
        wrapper_objective, 3.2, 12.8, adaptive=True
    )
    domain_result = domain_optimizer._minimize_on_domain(
        domain_objective, _IntegerSearchDomain.from_bounds(3.2, 12.8)
    )

    assert _result_tuple(wrapper_result) == _result_tuple(domain_result)
    assert wrapper_trace == domain_trace


def test_continuous_compatibility_wrapper_matches_explicit_domain_and_trajectory() -> (
    None
):
    wrapper_trace = []
    domain_trace = []

    def wrapper_objective(value):
        wrapper_trace.append(float(value))
        return float((float(value) - 2.25) ** 2)

    def domain_objective(value):
        domain_trace.append(float(value))
        return float((float(value) - 2.25) ** 2)

    wrapper_optimizer = GoldenSectionSearch(tol=1e-5, max_iter=100, verbose=False)
    domain_optimizer = GoldenSectionSearch(tol=1e-5, max_iter=100, verbose=False)
    wrapper_result = wrapper_optimizer.minimize(
        wrapper_objective, 0.5, 5.0, adaptive=False
    )
    domain_result = domain_optimizer._minimize_on_domain(
        domain_objective, _ContinuousSearchDomain.from_bounds(0.5, 5.0)
    )

    assert _result_tuple(wrapper_result) == _result_tuple(domain_result)
    np.testing.assert_array_equal(wrapper_trace, domain_trace)


def test_adaptive_gtwr_does_not_use_public_adaptive_compatibility_wrapper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_wrapper(*args, **kwargs):
        raise AssertionError("GTWR must use an explicit integer search domain")

    monkeypatch.setattr(GoldenSectionSearch, "minimize", forbidden_wrapper)

    rng = np.random.default_rng(20260830)
    n_space = 7
    n_time = 4
    angles = np.linspace(0.0, 2.0 * np.pi, n_space, endpoint=False)
    base_coords = np.column_stack([np.cos(angles), np.sin(angles)])
    coords = np.tile(base_coords, (n_time, 1))
    times = np.repeat(np.arange(n_time, dtype=float), n_space)
    X = rng.normal(size=(coords.shape[0], 2))
    y = (
        1.0
        + 0.7 * X[:, 0]
        - 0.4 * X[:, 1]
        + rng.normal(scale=0.05, size=coords.shape[0])
    )

    model = GTWR(
        kernel="bisquare",
        bandwidth="cv",
        adaptive=True,
        bandwidth_range=(10, 13),
        optimization_method="golden_section",
        lambda_st=0.5,
        causal=False,
        verbose=False,
    ).fit(
        X,
        y,
        coords,
        times,
        compute_local_r2=False,
        compute_inference=False,
    )

    assert isinstance(model.bandwidth_, int)
    assert 10 <= model.bandwidth_ <= 13
