# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Freeze-contract regression tests for standard GWR."""

import numpy as np
import pytest

from pygwrx import GWR


def _data(n_samples: int = 36):
    rng = np.random.default_rng(20260829)
    coords = rng.uniform(0.0, 1.0, size=(n_samples, 2))
    X = rng.normal(size=(n_samples, 2))
    y = 1.25 + 0.9 * X[:, 0] - 0.4 * X[:, 1] + rng.normal(0.0, 0.05, n_samples)
    return X, y, coords


def test_gwr_default_does_not_store_full_hat_matrix():
    X, y, coords = _data()
    model = GWR(kernel="gaussian", bandwidth=0.8).fit(X, y, coords)

    assert model.hat_matrix_ is None
    assert model.S_matrix_ is None
    assert model.diagnostics_ is not None
    assert np.isfinite(model.diagnostics_["trace_S"])
    assert np.isfinite(model.diagnostics_["trace_StS"])
    assert model.influence_ is not None


def test_explicit_hat_matrix_storage_does_not_change_gwr_numerics():
    X, y, coords = _data()
    default = GWR(kernel="gaussian", bandwidth=0.8).fit(X, y, coords)
    stored = GWR(kernel="gaussian", bandwidth=0.8).fit(
        X, y, coords, compute_hat_matrix=True
    )

    assert stored.hat_matrix_ is not None
    np.testing.assert_allclose(default.coef_, stored.coef_, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        default.intercept_, stored.intercept_, rtol=0.0, atol=0.0
    )
    np.testing.assert_allclose(
        default.fitted_values_, stored.fitted_values_, rtol=0.0, atol=0.0
    )
    assert default.diagnostics_["trace_S"] == stored.diagnostics_["trace_S"]
    assert default.diagnostics_["trace_StS"] == stored.diagnostics_["trace_StS"]


def test_legacy_hat_matrix_flag_can_still_request_storage():
    X, y, coords = _data()
    model = GWR(kernel="gaussian", bandwidth=0.8).fit(
        X,
        y,
        coords,
        compute_hat_matrix_flag=True,
        compute_local_r2=False,
    )

    assert model.hat_matrix_ is not None
    assert model.S_matrix_ is model.hat_matrix_
    assert model.hat_matrix_.shape == (X.shape[0], X.shape[0])


def test_fixed_equal_bandwidth_range_is_rejected_early():
    with pytest.raises(ValueError, match=r"fixed bandwidth_range.*lower < upper"):
        GWR(bandwidth="cv", adaptive=False, bandwidth_range=(1.0, 1.0))


def test_adaptive_equal_bandwidth_range_remains_valid_single_candidate():
    X, y, coords = _data(24)
    model = GWR(
        kernel="gaussian",
        bandwidth="cv",
        adaptive=True,
        bandwidth_range=(8, 8),
    ).fit(X, y, coords, compute_local_r2=False)

    assert model.bandwidth_ == 8
    assert model.bandwidth_search_ is not None
    assert model.bandwidth_search_["search_range"] == (8, 8)


def test_near_perfect_fit_leaves_undefined_residual_diagnostics_as_nan():
    x = np.linspace(-1.0, 1.0, 30)
    X = x[:, None]
    y = 2.0 + 3.0 * x
    coords = np.column_stack([x, np.zeros_like(x)])

    model = GWR(kernel="gaussian", bandwidth=100.0).fit(
        X, y, coords, compute_local_r2=False
    )

    assert model.sigma2_ is not None
    assert model.sigma2_ <= np.finfo(float).eps
    assert np.all(np.isnan(model.standardized_residuals_))
    assert np.all(np.isnan(model.cooks_distance_))


def test_summary_uses_shared_rank_aware_solver_not_normal_equation_pinv(monkeypatch):
    X, y, coords = _data()
    model = GWR(kernel="gaussian", bandwidth=0.8).fit(X, y, coords)

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "summary must not use a separate normal-equation pseudoinverse"
        )

    monkeypatch.setattr(np.linalg, "pinv", forbidden)
    text = model.summary()
    assert "Global OLS reference" in text
    assert "GWR diagnostics" in text


def test_failed_refit_clears_previous_fitted_state():
    X, y, coords = _data()
    model = GWR(kernel="gaussian", bandwidth=0.8).fit(X, y, coords)
    assert model.is_fitted_

    with pytest.raises(TypeError, match="compute_hat_matrix must be boolean"):
        model.fit(X, y, coords, compute_hat_matrix="yes")

    assert not model.is_fitted_
    assert model.n_samples_ is None
    assert model.n_features_in_ is None
    assert model.feature_names_in_ is None
    assert model.X_train_ is None
    assert model.y_train_ is None
    assert model.coords_train_ is None
    assert model.coef_ is None
    assert model.intercept_ is None
    assert model.bandwidth_ is None
    assert model.bandwidth_search_ is None


def test_failed_parameter_validation_refit_also_clears_previous_state():
    X, y, coords = _data()
    model = GWR(kernel="gaussian", bandwidth=0.8).fit(X, y, coords)
    model.bandwidth = -1.0

    with pytest.raises(ValueError, match="numeric bandwidth"):
        model.fit(X, y, coords)

    assert not model.is_fitted_
    assert model.n_samples_ is None
    assert model.coef_ is None
    assert model.bandwidth_ is None


def test_prediction_result_exposes_rank_diagnostics_even_without_inference():
    X, y, coords = _data()
    model = GWR(kernel="gaussian", bandwidth=0.8).fit(
        X, y, coords, compute_inference=False
    )
    result = model.predict_result(X[:4], coords[:4])

    assert result.local_rank is not None
    assert result.local_condition_number is not None
    assert result.rank_deficient is not None
    assert result.local_rank.shape == (4,)
    assert result.local_condition_number.shape == (4,)
    assert result.rank_deficient.shape == (4,)
    assert result.coef_standard_errors is None
    frame = result.to_frame()
    assert "local_rank" in frame
    assert "local_condition_number" in frame
    assert "rank_deficient" in frame
