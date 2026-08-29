"""Regression checks for the preserved normal-equation WLS reference."""

from __future__ import annotations

import inspect

import numpy as np

from pygwrx.core._legacy_solver import (
    _weighted_least_squares_normal_equations_legacy,
)
from pygwrx.core.solver import weighted_least_squares


def test_legacy_normal_equations_match_current_solver_on_well_conditioned_data():
    rng = np.random.default_rng(20260829)
    X = np.column_stack(
        [
            np.ones(30),
            rng.normal(size=30),
            rng.normal(size=30),
        ]
    )
    beta_true = np.array([1.5, -0.7, 2.1])
    y = X @ beta_true + rng.normal(scale=0.05, size=30)
    weights = np.linspace(0.2, 1.0, 30)

    beta_legacy, inverse_legacy = _weighted_least_squares_normal_equations_legacy(
        X, y, weights
    )
    beta_current, inverse_current = weighted_least_squares(X, y, weights)

    np.testing.assert_allclose(beta_legacy, beta_current, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(
        inverse_legacy,
        inverse_current,
        rtol=1e-10,
        atol=1e-10,
    )


def test_legacy_reference_has_no_regularization_parameter_or_penalty_term():
    signature = inspect.signature(_weighted_least_squares_normal_equations_legacy)
    assert "ridge" not in signature.parameters

    source = inspect.getsource(_weighted_least_squares_normal_equations_legacy)
    assert "XtWX +" not in source


def test_production_solver_no_longer_contains_normal_equation_helpers():
    import pygwrx.core.solver as solver_module

    source = inspect.getsource(solver_module)
    assert "def _normal_equations(" not in source
    assert "def _solve_linear_system(" not in source
    assert "_legacy_solver" not in source
