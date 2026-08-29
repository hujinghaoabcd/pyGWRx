"""Tests for rank-aware GWR calibration and inference."""

from __future__ import annotations

import numpy as np
import pytest

from pygwrx import GWR


def _rank_deficient_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.linspace(-2.0, 2.0, 24)
    X = np.column_stack([x, 2.0 * x])
    coords = np.column_stack([np.linspace(0.0, 10.0, x.size), 0.1 * x**2])
    y = 2.0 + 3.0 * x + 0.05 * np.sin(np.arange(x.size))
    return X, y, coords


def test_rank_deficient_calibration_keeps_coefficients_but_masks_inference() -> None:
    X, y, coords = _rank_deficient_data()

    with pytest.warns(RuntimeWarning, match="rank deficient"):
        model = GWR(
            kernel="gaussian",
            bandwidth=20.0,
            adaptive=False,
        ).fit(
            X,
            y,
            coords,
            compute_hat_matrix=False,
            compute_local_r2=False,
            compute_inference=True,
        )

    assert model.local_rank_ is not None
    assert model.local_condition_number_ is not None
    assert model.rank_deficient_ is not None
    np.testing.assert_array_equal(model.local_rank_, np.full(X.shape[0], 2))
    assert np.all(model.rank_deficient_)
    assert np.all(np.isinf(model.local_condition_number_))

    assert np.all(np.isfinite(model.coef_))
    assert np.all(np.isfinite(model.intercept_))
    assert np.all(np.isfinite(model.fitted_values_))

    assert model.parameter_covariance_diagonal_ is not None
    assert model.parameter_standard_errors_ is not None
    assert model.parameter_t_values_ is not None
    assert np.all(np.isnan(model.parameter_covariance_diagonal_))
    assert np.all(np.isnan(model.parameter_standard_errors_))
    assert np.all(np.isnan(model.parameter_t_values_))
    assert np.all(np.isnan(model.intercept_se_))
    assert np.all(np.isnan(model.coef_se_))
    assert np.all(np.isnan(model.intercept_t_))
    assert np.all(np.isnan(model.coef_t_))


def test_full_rank_gwr_retains_finite_inference(synthetic) -> None:
    model = GWR(
        kernel="gaussian",
        bandwidth=6.0,
        adaptive=False,
    ).fit(
        synthetic["X"],
        synthetic["y"],
        synthetic["coords"],
        compute_hat_matrix=False,
        compute_local_r2=False,
        compute_inference=True,
    )

    n_parameters = synthetic["X"].shape[1] + 1
    assert model.local_rank_ is not None
    assert model.local_condition_number_ is not None
    assert model.rank_deficient_ is not None
    np.testing.assert_array_equal(
        model.local_rank_,
        np.full(synthetic["n"], n_parameters),
    )
    assert not np.any(model.rank_deficient_)
    assert np.all(np.isfinite(model.local_condition_number_))
    assert np.all(np.isfinite(model.parameter_standard_errors_))
    assert np.all(np.isfinite(model.parameter_t_values_))


def test_rank_diagnostics_are_retained_when_inference_is_disabled() -> None:
    X, y, coords = _rank_deficient_data()

    with pytest.warns(RuntimeWarning, match="rank deficient"):
        model = GWR(kernel="gaussian", bandwidth=20.0).fit(
            X,
            y,
            coords,
            compute_hat_matrix=False,
            compute_local_r2=False,
            compute_inference=False,
        )

    assert model.local_rank_ is not None
    assert model.rank_deficient_ is not None
    assert np.all(model.rank_deficient_)
    assert model.parameter_standard_errors_ is None
    assert model.parameter_t_values_ is None


def test_prediction_recalibration_reports_rank_deficiency() -> None:
    X, y, coords = _rank_deficient_data()
    with pytest.warns(RuntimeWarning, match="rank deficient"):
        model = GWR(kernel="gaussian", bandwidth=20.0).fit(
            X,
            y,
            coords,
            compute_hat_matrix=False,
            compute_local_r2=False,
            compute_inference=True,
        )

    X_new = np.array([[0.25, 0.5], [-0.5, -1.0]])
    coords_new = np.array([[2.5, 0.1], [7.5, 0.2]])

    with pytest.warns(RuntimeWarning, match="rank deficient"):
        result = model.predict_result(X_new, coords_new)
    assert np.all(np.isfinite(result.predictions))
    assert np.all(np.isnan(result.intercept_standard_errors))
    assert np.all(np.isnan(result.coef_standard_errors))
    assert np.all(np.isnan(result.intercept_t_values))
    assert np.all(np.isnan(result.coef_t_values))

    with pytest.warns(RuntimeWarning, match="rank deficient"):
        params = model.get_local_parameters(coords_new)
    np.testing.assert_array_equal(params["local_rank"], np.array([2, 2]))
    assert np.all(params["rank_deficient"])
    assert np.all(np.isinf(params["local_condition_number"]))


def test_rank_diagnostics_are_exported_and_summarized() -> None:
    X, y, coords = _rank_deficient_data()
    with pytest.warns(RuntimeWarning, match="rank deficient"):
        model = GWR(kernel="gaussian", bandwidth=20.0).fit(
            X,
            y,
            coords,
            compute_hat_matrix=False,
            compute_local_r2=False,
            compute_inference=False,
        )

    frame = model.to_frame()
    assert "local_rank" in frame
    assert "local_condition_number" in frame
    assert "rank_deficient" in frame
    assert frame["rank_deficient"].all()

    summary = model.summary()
    assert f"Rank-deficient local fits: {X.shape[0]}/{X.shape[0]}" in summary
