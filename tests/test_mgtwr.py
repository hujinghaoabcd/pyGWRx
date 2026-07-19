# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Validation tests for the self-contained MGTWR implementation."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pygwrx.models import MGTWR, MGWR


def make_data(n=36, seed=42):
    rng = np.random.default_rng(seed)
    coords = rng.uniform(0.0, 5.0, size=(n, 2))
    times = rng.uniform(0.0, 3.0, size=n)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    beta1 = 1.5 + 0.15 * coords[:, 0] + 0.10 * times
    beta2 = -0.8 + 0.12 * coords[:, 1]
    y = 2.0 + beta1 * x1 + beta2 * x2 + rng.normal(0.0, 0.05, size=n)
    return np.column_stack((x1, x2)), y, coords, times


def test_zero_temporal_scale_matches_mgwr_exactly():
    X, y, coords, times = make_data(n=30, seed=2)
    bandwidths = [3.5, 3.0, 4.0]
    initial_bandwidth = float(np.median(bandwidths))
    mgtwr = MGTWR(
        bandwidths=bandwidths,
        taus=0.0,
        kernel="gaussian",
        adaptive=False,
        init_bandwidth=initial_bandwidth,
        calculate_inference=False,
        tol_multi=1e-7,
        max_iter=100,
    ).fit(X, y, coords, times)
    mgwr = MGWR(
        bandwidths=bandwidths,
        kernel="gaussian",
        adaptive=False,
        init_bandwidth=initial_bandwidth,
        tol=1e-7,
        max_iter=100,
    ).fit(X, y, coords, compute_inference=False)

    mgwr_params = np.column_stack((mgwr.intercept_, mgwr.coef_))
    np.testing.assert_allclose(mgtwr.params_, mgwr_params, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(
        mgtwr.fitted_values_, mgwr.fitted_values_, rtol=0.0, atol=1e-12
    )


def test_dataframe_inputs_and_parameter_shapes():
    X, y, coords, times = make_data()
    model = MGTWR(
        bandwidths=4.0,
        taus=1.0,
        kernel="bisquare",
        adaptive=False,
        calculate_inference=False,
    ).fit(
        pd.DataFrame(X, columns=["x1", "x2"]),
        pd.Series(y, name="y"),
        pd.DataFrame(coords, columns=["x", "y"]),
        pd.Series(times, name="time"),
    )
    assert model.params_.shape == (len(y), 3)
    assert model.coef_.shape == (len(y), 2)
    assert model.intercept_.shape == (len(y),)
    assert model.feature_names_in_.tolist() == ["x1", "x2"]
    assert np.isfinite(model.r2_)
    assert "time" in model.to_frame().columns


def test_known_truth_simulation_recovers_signal():
    X, y, coords, times = make_data(n=48, seed=7)
    model = MGTWR(
        bandwidths=[4.5, 3.5, 4.0],
        taus=[0.8, 1.2, 0.7],
        kernel="gaussian",
        adaptive=False,
        calculate_inference=False,
    ).fit(X, y, coords, times)
    assert model.r2_ > 0.95
    assert np.sqrt(np.mean(model.residuals_**2)) < 0.25


def test_global_scale_degenerates_toward_ols():
    X, y, coords, times = make_data(n=40, seed=9)
    model = MGTWR(
        bandwidths=1e6,
        taus=1.0,
        kernel="gaussian",
        adaptive=False,
        calculate_inference=False,
    ).fit(X, y, coords, times)
    Xd = np.column_stack((np.ones(len(y)), X))
    ols = np.linalg.lstsq(Xd, y, rcond=None)[0]
    np.testing.assert_allclose(model.params_.mean(axis=0), ols, atol=2e-4)


def test_automatic_scale_selection_stays_inside_bounds():
    X, y, coords, times = make_data(n=20, seed=3)
    model = MGTWR(
        kernel="gaussian",
        adaptive=True,
        bandwidth_range=(6, 16),
        tau_range=(0.0, 2.0),
        bandwidth_method="aicc",
        calculate_inference=False,
        max_iter=5,
        tol_multi=1e-3,
    ).fit(X[:, :1], y, coords, times)
    assert np.all((model.bandwidths_ >= 6) & (model.bandwidths_ <= 16))
    assert np.all((model.taus_ >= 0.0) & (model.taus_ <= 2.0))
    assert np.all(np.isfinite(model.convergence_history_))


def test_temporal_scale_zero_makes_time_order_irrelevant():
    X, y, coords, times = make_data(n=32, seed=11)
    kwargs = dict(
        bandwidths=[4.0, 3.5, 4.5],
        taus=0.0,
        kernel="gaussian",
        adaptive=False,
        calculate_inference=False,
    )
    first = MGTWR(**kwargs).fit(X, y, coords, times)
    second = MGTWR(**kwargs).fit(X, y, coords, times[::-1])
    np.testing.assert_allclose(first.params_, second.params_, rtol=0.0, atol=1e-12)


def test_prediction_is_explicitly_unsupported():
    X, y, coords, times = make_data()
    model = MGTWR(
        bandwidths=4.0, taus=1.0, adaptive=False, calculate_inference=False
    ).fit(X, y, coords, times)
    with pytest.raises(NotImplementedError, match="Out-of-sample MGTWR"):
        model.predict(X[:2], coords[:2], times[:2])


def test_invalid_inputs_and_boundary_checks():
    with pytest.raises(ValueError, match="supplied together"):
        MGTWR(bandwidths=[3.0], taus=None)
    with pytest.raises(ValueError, match="non-negative"):
        MGTWR(bandwidths=[3.0], taus=[-0.1])
    X, y, coords, times = make_data()
    with pytest.raises(ValueError, match="incompatible shapes|same number of rows"):
        MGTWR(bandwidths=4.0, taus=1.0).fit(X[:-1], y, coords, times)


def test_failed_refit_clears_previous_state():
    X, y, coords, times = make_data()
    model = MGTWR(
        bandwidths=4.0, taus=1.0, adaptive=False, calculate_inference=False
    ).fit(X, y, coords, times)
    assert model._is_fitted
    with pytest.raises(ValueError):
        model.fit(X[:-1], y, coords, times)
    assert not model._is_fitted
    assert model.params_ is None
    assert model.fitted_values_ is None
    assert model.rss_ is None


def test_inference_attributes_are_exposed():
    X, y, coords, times = make_data(n=24, seed=21)
    model = MGTWR(
        bandwidths=4.5,
        taus=1.0,
        kernel="gaussian",
        adaptive=False,
        calculate_inference=True,
    ).fit(X, y, coords, times)
    assert model.effective_params_by_variable_.shape == (3,)
    assert model.parameter_standard_errors_.shape == (24, 3)
    assert model.parameter_t_values_.shape == (24, 3)
    assert np.isfinite(model.aic_)
    assert np.isfinite(model.aicc_)
    assert np.isfinite(model.bic_)
    assert np.isfinite(model.sigma2_)
    assert np.isclose(
        model.effective_params_, np.sum(model.effective_params_by_variable_)
    )


@pytest.mark.reference
def test_fixed_scale_matches_frozen_independent_reference():
    fixture_path = (
        Path(__file__).parent / "reference_data" / "mgtwr_fixed_gaussian_reference.json"
    )
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    inputs = fixture["inputs"]
    config = fixture["configuration"]
    expected = fixture["expected"]

    model = MGTWR(
        bandwidths=config["bandwidths"],
        taus=config["taus"],
        kernel="gaussian",
        adaptive=False,
        init_bandwidth=config["init_bandwidth"],
        init_tau=config["init_tau"],
        calculate_inference=True,
        tol_multi=1e-8,
        max_iter=200,
    ).fit(
        np.asarray(inputs["X"], dtype=float),
        np.asarray(inputs["y"], dtype=float),
        np.asarray(inputs["coords"], dtype=float),
        np.asarray(inputs["times"], dtype=float),
    )

    np.testing.assert_allclose(model.params_, expected["params"], rtol=0.0, atol=5e-8)
    np.testing.assert_allclose(
        model.fitted_values_, expected["fitted_values"], rtol=0.0, atol=8e-8
    )
    np.testing.assert_allclose(
        model.residuals_, expected["residuals"], rtol=0.0, atol=8e-8
    )
    np.testing.assert_allclose(
        model.effective_params_by_variable_,
        expected["effective_params_by_variable"],
        rtol=0.0,
        atol=2e-6,
    )
    np.testing.assert_allclose(
        model.parameter_standard_errors_,
        expected["parameter_standard_errors"],
        rtol=0.0,
        atol=5e-9,
    )
    np.testing.assert_allclose(
        model.parameter_t_values_,
        expected["parameter_t_values"],
        rtol=0.0,
        atol=8e-6,
    )
    assert model.sigma2_ == pytest.approx(expected["sigma2"], abs=5e-10)
    assert model.rss_ == pytest.approx(expected["rss"], abs=2e-8)
    assert model.r2_ == pytest.approx(expected["r2"], abs=2e-9)
    assert model.aic_ == pytest.approx(expected["aic_standard"], abs=2e-7)
    assert model.aicc_ == pytest.approx(expected["aicc_standard"], abs=2e-7)
    assert model.bic_ == pytest.approx(expected["bic_standard"], abs=2e-7)
    assert model.n_iter_ == expected["n_iter"]
