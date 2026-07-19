# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Regression tests for the standard Gaussian MGWR implementation.

Author:
    Jinghao Hu
"""

from __future__ import annotations

__author__ = "Jinghao Hu"
__license__ = "MIT"

import numpy as np
import pandas as pd
import pytest

from pygwrx.models import MGWR


def _reference_data():
    """Return the deterministic dataset used for the mgwr 2.2.1 comparison."""
    rng = np.random.default_rng(42)
    n_samples = 35
    coords = rng.uniform(0.0, 10.0, size=(n_samples, 2))
    X = rng.normal(size=(n_samples, 2))
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    y = (
        1.0
        + 1.5 * X[:, 0]
        - 0.8 * X[:, 1]
        + 0.15 * coords[:, 0]
        + rng.normal(0.0, 0.3, size=n_samples)
    )
    y = (y - y.mean()) / y.std()
    return X, y, coords


def _reference_model(**kwargs):
    defaults = dict(
        kernel="bisquare",
        adaptive=True,
        bandwidth_method="aicc",
        bandwidth_ranges=[(5, 35)] * 3,
        init_bandwidth=20,
        max_iter=10,
        tol=1e-5,
        bws_same_times=3,
        fit_intercept=True,
    )
    defaults.update(kwargs)
    return MGWR(**defaults)


def test_reference_bandwidths_and_inference_match_mgwr_221():
    X, y, coords = _reference_data()
    model = _reference_model().fit(
        X,
        y,
        coords,
        compute_hat_matrix=True,
        store_partial_hat_matrices=True,
        n_chunks=1,
    )

    np.testing.assert_array_equal(model.bandwidths_, [20, 32, 34])
    assert model.converged_
    assert model.n_iter_ == 6
    assert model.hat_matrix_.shape == (35, 35)
    assert model.partial_hat_matrices_.shape == (35, 35, 3)

    expected_params = np.array(
        [
            [0.2562114177, 0.8446184181, -0.4558558709],
            [0.2271250072, 0.8043151298, -0.4683823724],
            [-0.2235541879, 0.8801266101, -0.4569427693],
        ]
    )
    params = np.column_stack([model.intercept_, model.coef_])
    np.testing.assert_allclose(params[:3], expected_params, atol=2e-7, rtol=0.0)
    np.testing.assert_allclose(
        model.effective_params_by_variable_,
        [4.1990793563, 2.4796732551, 1.9982104580],
        atol=2e-6,
        rtol=0.0,
    )
    assert model.diagnostics_["aicc"] == pytest.approx(11.1020174780, abs=3e-6)
    assert model.diagnostics_["trace_S"] == pytest.approx(
        np.sum(model.effective_params_by_variable_), abs=1e-10
    )
    assert model.parameter_standard_errors_.shape == (35, 3)
    assert np.all(np.isfinite(model.parameter_standard_errors_))


def test_chunked_inference_matches_single_chunk():
    X, y, coords = _reference_data()
    single = _reference_model().fit(X, y, coords, n_chunks=1)
    chunked = _reference_model().fit(X, y, coords, n_chunks=4)

    np.testing.assert_allclose(
        chunked.effective_params_by_variable_,
        single.effective_params_by_variable_,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        chunked.parameter_standard_errors_,
        single.parameter_standard_errors_,
        atol=1e-10,
    )
    np.testing.assert_allclose(chunked.influence_, single.influence_, atol=1e-10)
    assert chunked.diagnostics_["trace_StS"] == pytest.approx(
        single.diagnostics_["trace_StS"], abs=1e-10
    )


def test_manual_bandwidths_are_respected():
    X, y, coords = _reference_data()
    model = MGWR(
        kernel="bisquare",
        bandwidths=[18, 24, 30],
        adaptive=True,
        init_bandwidth=20,
        max_iter=30,
        tol=1e-6,
    ).fit(X, y, coords, compute_inference=False)

    np.testing.assert_array_equal(model.bandwidths_, [18, 24, 30])
    assert np.all(model.bandwidth_history_ == np.array([18, 24, 30]))
    assert model.parameter_standard_errors_ is None
    assert model.diagnostics_["trace_S"] > 0.0


def test_dataframe_names_and_result_frame_are_preserved():
    X, y, coords = _reference_data()
    X_frame = pd.DataFrame(X, columns=["income", "rurality"])
    model = _reference_model().fit(X_frame, y, coords)
    result = model.to_frame()

    np.testing.assert_array_equal(model.feature_names_in_, ["income", "rurality"])
    for column in (
        "intercept",
        "coef_income",
        "coef_rurality",
        "se_income",
        "se_rurality",
        "t_income",
        "t_rurality",
        "fitted",
        "residual",
    ):
        assert column in result.columns


def test_training_arrays_are_copied():
    X, y, coords = _reference_data()
    model = _reference_model().fit(X, y, coords)
    stored_X = model.X_train_.copy()
    stored_y = model.y_train_.copy()
    stored_coords = model.coords_train_.copy()

    X[:] = 999.0
    y[:] = -999.0
    coords[:] = 0.0

    np.testing.assert_allclose(model.X_train_, stored_X)
    np.testing.assert_allclose(model.y_train_, stored_y)
    np.testing.assert_allclose(model.coords_train_, stored_coords)


def test_out_of_sample_prediction_is_explicitly_unsupported():
    X, y, coords = _reference_data()
    model = _reference_model().fit(X, y, coords)
    with pytest.raises(NotImplementedError, match="Out-of-sample MGWR prediction"):
        model.predict(X[:2], coords[:2])


def test_invalid_bandwidth_vector_length_raises_and_resets_state():
    X, y, coords = _reference_data()
    model = MGWR(
        bandwidths=[20, 25],
        adaptive=True,
        init_bandwidth=20,
    )
    with pytest.raises(ValueError, match="one value per fitted parameter"):
        model.fit(X, y, coords)
    assert not model.is_fitted_
    assert model.coef_ is None
    assert model.bandwidths_ is None


def test_nonconvergence_is_reported_without_corrupting_results():
    X, y, coords = _reference_data()
    model = _reference_model(max_iter=1, tol=1e-20)
    with pytest.warns(RuntimeWarning, match="reached max_iter"):
        model.fit(X, y, coords)
    assert model.is_fitted_
    assert not model.converged_
    assert model.n_iter_ == 1
    assert np.all(np.isfinite(model.fitted_values_))


def test_fixed_gaussian_reference_case():
    rng = np.random.default_rng(7)
    n_samples = 30
    coords = rng.uniform(0.0, 10.0, size=(n_samples, 2))
    X = rng.normal(size=(n_samples, 2))
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    y = 0.5 + 1.2 * X[:, 0] - 0.7 * X[:, 1] + rng.normal(0.0, 0.2, n_samples)
    y = (y - y.mean()) / y.std()

    model = MGWR(
        kernel="gaussian",
        adaptive=False,
        bandwidth_method="aicc",
        bandwidth_ranges=[(2.0, 12.0)] * 3,
        init_bandwidth=6.0,
        max_iter=10,
        tol=1e-5,
        bws_same_times=3,
    ).fit(X, y, coords)

    np.testing.assert_allclose(model.bandwidths_, [12.0, 12.0, 12.0])
    assert model.diagnostics_["aicc"] == pytest.approx(-31.7366090084, abs=3e-6)
    np.testing.assert_allclose(
        model.effective_params_by_variable_,
        [1.10979259, 1.08179022, 1.09782817],
        atol=3e-6,
    )


def test_summary_reports_variable_specific_scales():
    X, y, coords = _reference_data()
    model = _reference_model().fit(X, y, coords)
    summary = model.summary()

    assert "Multiscale Geographically Weighted Regression" in summary
    assert "Variable-specific scales" in summary
    assert "Initial GWR bandwidth: 20" in summary
    assert "Converged: True" in summary
    assert "trace(S)" in summary
