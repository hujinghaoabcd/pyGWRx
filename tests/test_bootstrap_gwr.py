# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Numerical and engineering tests for BootstrapGWR."""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from pygwrx.models.bootstrap_gwr import BootstrapGWR


def make_stationary_data(seed: int = 42, n: int = 48):
    rng = np.random.default_rng(seed)
    coords = rng.uniform(0.0, 10.0, size=(n, 2))
    X = rng.normal(size=(n, 2))
    y = 1.2 + 1.8 * X[:, 0] - 0.7 * X[:, 1] + rng.normal(scale=0.45, size=n)
    return X, y, coords


def make_nonstationary_data(seed: int = 7, n: int = 64):
    rng = np.random.default_rng(seed)
    x_coord = np.linspace(0.0, 12.0, n)
    coords = np.column_stack([x_coord, np.zeros(n)])
    X = rng.normal(size=(n, 1))
    slope = np.where(x_coord < 6.0, 3.5, -3.5)
    y = 0.8 + slope * X[:, 0] + rng.normal(scale=0.25, size=n)
    return X, y, coords


def make_model(**kwargs):
    defaults = dict(
        bandwidth=24,
        adaptive=True,
        kernel="gaussian",
        n_bootstrap=7,
        reselect_bandwidth=False,
        random_state=123,
        verbose=False,
    )
    defaults.update(kwargs)
    return BootstrapGWR(**defaults)


def test_modified_statistic_matches_gwmodel_definition():
    X, y, coords = make_stationary_data()
    model = make_model(n_bootstrap=3).fit(X, y, coords)
    expected = np.std(
        model.coefficients_gwr_ / model.local_standard_errors_, axis=0, ddof=1
    )
    np.testing.assert_allclose(model.modified_statistics_, expected, atol=1e-12)
    np.testing.assert_allclose(model.test_statistic_, expected, atol=1e-12)


def test_localized_statistic_matches_mlr_pseudo_t_formula():
    X, y, coords = make_stationary_data()
    model = make_model(n_bootstrap=3).fit(X, y, coords)
    expected = (
        model.coefficients_gwr_ - model.coefficients_global_[np.newaxis, :]
    ) / model.local_standard_errors_
    np.testing.assert_allclose(model.localized_statistics_, expected, atol=1e-12)


def test_parametric_bootstrap_is_reproducible():
    X, y, coords = make_stationary_data()
    first = make_model().fit(X, y, coords)
    second = make_model().fit(X, y, coords)
    np.testing.assert_allclose(
        first.bootstrap_modified_statistics_, second.bootstrap_modified_statistics_
    )
    np.testing.assert_allclose(first.localized_p_values_, second.localized_p_values_)


def test_plus_one_and_gwmodel_pvalue_conventions_are_explicit():
    X, y, coords = make_stationary_data()
    corrected = make_model(pvalue_method="plus_one").fit(X, y, coords)
    compatible = make_model(pvalue_method="gwmodel").fit(X, y, coords)
    increment = 1.0 / (corrected.n_bootstrap + 1)
    np.testing.assert_allclose(
        corrected.modified_p_values_, compatible.modified_p_values_ + increment
    )
    np.testing.assert_allclose(
        corrected.localized_p_values_, compatible.localized_p_values_ + increment
    )


def test_fixed_observed_bandwidth_is_reused_when_requested():
    X, y, coords = make_stationary_data()
    model = make_model(bandwidth=22, n_bootstrap=5).fit(X, y, coords)
    assert model.bandwidth_ == 22
    np.testing.assert_allclose(model.bootstrap_bandwidths_, 22.0)


def test_automatic_bandwidth_can_be_conditioned_after_observed_fit():
    X, y, coords = make_stationary_data(n=34)
    model = BootstrapGWR(
        bandwidth="cv",
        adaptive=True,
        bandwidth_range=(12, 20),
        kernel="gaussian",
        n_bootstrap=2,
        reselect_bandwidth=False,
        random_state=4,
    ).fit(X, y, coords)
    assert 12 <= model.bandwidth_ <= 20
    np.testing.assert_allclose(model.bootstrap_bandwidths_, float(model.bandwidth_))


def test_full_local_bootstrap_storage_and_critical_values():
    X, y, coords = make_stationary_data(n=36)
    model = make_model(n_bootstrap=5, store_local_bootstrap=True).fit(X, y, coords)
    assert model.bootstrap_localized_statistics_.shape == (5, 36, 3)
    assert model.localized_lower_critical_.shape == (36, 3)
    assert model.localized_upper_critical_.shape == (36, 3)
    expected_lower = np.quantile(model.bootstrap_localized_statistics_, 0.025, axis=0)
    np.testing.assert_allclose(model.localized_lower_critical_, expected_lower)


def test_default_path_avoids_large_local_bootstrap_array():
    X, y, coords = make_stationary_data(n=36)
    model = make_model(n_bootstrap=3).fit(X, y, coords)
    assert model.bootstrap_localized_statistics_ is None
    assert model.localized_p_values_.shape == (36, 3)


def test_strong_spatially_varying_slope_is_detected():
    X, y, coords = make_nonstationary_data()
    model = BootstrapGWR(
        bandwidth=16,
        adaptive=True,
        kernel="gaussian",
        n_bootstrap=39,
        reselect_bandwidth=False,
        random_state=11,
    ).fit(X, y, coords)
    assert model.modified_statistics_[1] > model.modified_critical_values_[1]
    assert model.modified_p_values_[1] <= 0.05


def test_dataframe_names_and_local_result_frame():
    X, y, coords = make_stationary_data(n=38)
    frame = pd.DataFrame(X, columns=["income", "education"])
    coord_frame = pd.DataFrame(coords, columns=["x", "y"])
    model = make_model(n_bootstrap=3).fit(frame, pd.Series(y), coord_frame)
    assert tuple(model.parameter_names_) == ("intercept", "income", "education")
    result = model.to_frame()
    assert result.shape[0] == X.shape[0]
    assert "coef_income" in result
    assert "localized_p_education" in result


def test_no_intercept_path_has_predictor_only_statistics():
    X, y, coords = make_stationary_data(n=40)
    model = make_model(fit_intercept=False, n_bootstrap=3).fit(X, y, coords)
    assert model.coefficients_gwr_.shape == (40, 2)
    assert model.modified_statistics_.shape == (2,)
    assert model.parameter_names_ == ("x0", "x1")


def test_summary_contains_coefficientwise_results():
    X, y, coords = make_stationary_data(n=38)
    model = make_model(n_bootstrap=3).fit(X, y, coords)
    result = model.summary()
    assert isinstance(result, str)
    assert "parameter_names" in result
    assert "modified_statistics" in result
    assert "null_model" in result


def test_inactive_compatibility_parameters_are_not_public():
    signature = inspect.signature(BootstrapGWR)
    assert "test_type" not in signature.parameters
    assert "null_model" not in signature.parameters


def test_invalid_pvalue_and_tail_options_are_rejected():
    X, y, coords = make_stationary_data(n=30)
    with pytest.raises(ValueError, match="pvalue_method"):
        make_model(pvalue_method="legacy", n_bootstrap=2).fit(X, y, coords)
    with pytest.raises(ValueError, match="localized_tail"):
        make_model(localized_tail="left", n_bootstrap=2).fit(X, y, coords)


def test_failed_refit_clears_previous_results():
    X, y, coords = make_stationary_data(n=36)
    model = make_model(n_bootstrap=2).fit(X, y, coords)
    assert model._is_fitted
    with pytest.raises(ValueError):
        model.fit(X, y[:-1], coords)
    assert not model._is_fitted
    assert model.modified_statistics_ is None
    assert model.localized_p_values_ is None


def test_rank_deficient_global_null_fails_loudly():
    X, y, coords = make_stationary_data(n=36)
    X_bad = np.column_stack([X[:, 0], X[:, 0]])
    with pytest.raises(np.linalg.LinAlgError, match="rank deficient"):
        make_model(n_bootstrap=2).fit(X_bad, y, coords)


def test_bundled_georgia_case_runs_with_named_coefficients():
    from pygwrx.io import load_georgia

    frame = load_georgia(return_type="frame")
    feature_names = ["PctRural", "PctPov", "PctBlack", "PctEld", "PctFB"]
    X = frame[feature_names]
    y = frame["PctBach"]
    coords = frame[["X", "Y"]]
    model = BootstrapGWR(
        bandwidth=80,
        adaptive=True,
        kernel="bisquare",
        n_bootstrap=2,
        reselect_bandwidth=False,
        random_state=9,
    ).fit(X, y, coords)
    assert model.parameter_names_ == ("intercept", *feature_names)
    assert model.modified_statistics_.shape == (6,)
    assert np.all(np.isfinite(model.modified_p_values_))
