# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Reference tests for Gaussian GWR diagnostic formulas."""

from __future__ import annotations

import numpy as np
import pytest

from pygwrx.core import compute_aic, compute_aicc, compute_bic


def _gaussian_log_likelihood_term(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    residuals = y_true - y_pred
    rss = float(np.dot(residuals, residuals))
    n = y_true.size
    return float(n * np.log(rss / n) + n * np.log(2.0 * np.pi) + n)


def test_information_criteria_follow_gaussian_gwr_definitions():
    y_true = np.asarray([1.1, 2.4, 2.8, 4.2, 5.1, 5.9])
    y_pred = np.asarray([1.0, 2.2, 3.0, 4.1, 5.3, 5.7])
    trace_s = 2.35
    n = y_true.size
    likelihood_term = _gaussian_log_likelihood_term(y_true, y_pred)

    expected_aic = likelihood_term + 2.0 * (trace_s + 1.0)
    expected_aicc = likelihood_term - n + n * (n + trace_s) / (n - trace_s - 2.0)
    expected_bic = likelihood_term + (trace_s + 1.0) * np.log(n)

    assert compute_aic(y_true, y_pred, trace_s) == pytest.approx(expected_aic)
    assert compute_aicc(y_true, y_pred, trace_s) == pytest.approx(expected_aicc)
    assert compute_bic(y_true, y_pred, trace_s) == pytest.approx(expected_bic)


def test_aicc_returns_infinity_when_small_sample_correction_is_undefined():
    y_true = np.arange(5.0)
    y_pred = y_true + 0.1
    assert np.isinf(compute_aicc(y_true, y_pred, n_params=3.0))
