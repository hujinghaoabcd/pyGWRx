"""Regression tests for the standardized flagship GWR API."""

import inspect

import numpy as np
import pandas as pd
import pytest

import pygwrx
from pygwrx.models import GWR, GWRPredictionResult


def test_constructor_has_no_backend_parameter():
    assert "backend" not in inspect.signature(GWR).parameters
    model = GWR()
    assert not hasattr(model, "backend")
    assert not hasattr(model, "n_jobs")
    assert not hasattr(model, "get_params")


def test_dataframe_schema_is_recorded_and_prediction_order_is_checked(synthetic):
    columns = ["income", "population", "accessibility"]
    X = pd.DataFrame(synthetic["X"], columns=columns)
    model = GWR(bandwidth=30, adaptive=True)
    model.fit(X, synthetic["y"], synthetic["coords"], compute_hat_matrix=False)

    assert model.n_features_in_ == 3
    assert model.feature_names_in_.tolist() == columns
    with pytest.raises(ValueError, match="same order"):
        model.predict(X[columns[::-1]], synthetic["coords"])


def test_training_arrays_are_copied(synthetic):
    X = synthetic["X"].copy()
    y = synthetic["y"].copy()
    coords = synthetic["coords"].copy()
    model = GWR(bandwidth=30, adaptive=True)
    model.fit(X, y, coords, compute_hat_matrix=False)
    baseline = model.predict(X[:4], coords[:4])

    X[:] = 999.0
    y[:] = -999.0
    coords[:] = -999.0
    repeated = model.predict(model.X_train_[:4], model.coords_train_[:4])
    np.testing.assert_allclose(repeated, baseline)


def test_failed_refit_leaves_clean_unfitted_state(synthetic):
    model = GWR(bandwidth=30, adaptive=True)
    model.fit(
        synthetic["X"], synthetic["y"], synthetic["coords"], compute_hat_matrix=False
    )
    assert model.is_fitted_

    with pytest.raises(ValueError):
        model.fit(synthetic["X"], synthetic["y"][:-1], synthetic["coords"])

    assert not model.is_fitted_
    assert model.coef_ is None
    assert model.X_train_ is None
    assert model.bandwidth_ is None


def test_adaptive_manual_bandwidth_is_integer_and_validated(synthetic):
    model = GWR(bandwidth=30, adaptive=True)
    model.fit(
        synthetic["X"], synthetic["y"], synthetic["coords"], compute_hat_matrix=False
    )
    assert isinstance(model.bandwidth_, int)

    with pytest.raises(ValueError, match="integer"):
        GWR(bandwidth=30.5, adaptive=True)

    with pytest.raises(ValueError, match="cannot exceed"):
        GWR(bandwidth=1000, adaptive=True).fit(
            synthetic["X"], synthetic["y"], synthetic["coords"]
        )


def test_trace_only_diagnostics_match_full_hat_matrix(synthetic):
    full = GWR(bandwidth=30, adaptive=True)
    full.fit(
        synthetic["X"],
        synthetic["y"],
        synthetic["coords"],
        compute_hat_matrix=True,
    )
    compact = GWR(bandwidth=30, adaptive=True)
    compact.fit(
        synthetic["X"],
        synthetic["y"],
        synthetic["coords"],
        compute_hat_matrix=False,
    )

    assert compact.hat_matrix_ is None
    assert compact.S_matrix_ is None
    for key in (
        "effective_params",
        "trace_S",
        "trace_StS",
        "enp_v2",
        "edf_v2",
        "adj_r2",
        "aic",
        "aicc",
        "bic",
    ):
        assert np.isfinite(compact.diagnostics_[key])
        np.testing.assert_allclose(
            compact.diagnostics_[key], full.diagnostics_[key], rtol=1e-10, atol=1e-10
        )
    np.testing.assert_allclose(compact.influence_, np.diag(full.hat_matrix_))


def test_hat_matrix_alias_and_smoother_identity(synthetic):
    model = GWR(bandwidth=30, adaptive=True)
    model.fit(
        synthetic["X"],
        synthetic["y"],
        synthetic["coords"],
        compute_hat_matrix=True,
    )
    assert model.S_matrix_ is model.hat_matrix_
    np.testing.assert_allclose(
        model.hat_matrix_ @ model.y_train_, model.fitted_values_, atol=1e-8
    )


def test_prediction_result_contains_full_local_equation(synthetic):
    model = GWR(bandwidth=30, adaptive=True)
    model.fit(
        synthetic["X"], synthetic["y"], synthetic["coords"], compute_hat_matrix=False
    )
    result = model.predict_result(synthetic["X"][:5], synthetic["coords"][:5])

    assert isinstance(result, GWRPredictionResult)
    assert result.predictions.shape == (5,)
    assert result.coef.shape == (5, 3)
    assert result.intercept.shape == (5,)
    np.testing.assert_allclose(
        result.predictions,
        np.einsum("ij,ij->i", synthetic["X"][:5], result.coef) + result.intercept,
    )
    assert "prediction" in result.to_frame().columns


def test_summary_survives_singular_global_reference():
    x = np.linspace(0.0, 1.0, 30)
    X = np.column_stack([x, 2.0 * x])
    coords = np.column_stack([x, np.zeros_like(x)])
    y = 1.0 + 3.0 * x
    model = GWR(bandwidth=20, adaptive=True)
    model.fit(X, y, coords, compute_hat_matrix=False)
    text = model.summary()
    assert "Global OLS reference" in text
    assert "Local coefficient distribution" in text
    text.encode("cp936")


def test_public_models_no_longer_expose_backend_parameter():
    model_names = [
        "GWR",
        "MGWR",
        "RGWR",
        "GTWR",
        "GWGLM",
        "GWLasso",
        "MixedGWR",
        "GWPCA",
        "GWDA",
        "GWSS",
    ]
    for name in model_names:
        cls = getattr(pygwrx, name)
        assert "backend" not in inspect.signature(cls).parameters, name


def test_experimental_backend_package_is_removed():
    with pytest.raises(ModuleNotFoundError):
        __import__("pygwrx.experimental")


def test_inference_outputs_have_standard_shapes(synthetic):
    model = GWR(bandwidth=30, adaptive=True)
    model.fit(
        synthetic["X"],
        synthetic["y"],
        synthetic["coords"],
        compute_hat_matrix=False,
        compute_inference=True,
    )

    n, p = synthetic["X"].shape
    assert np.isfinite(model.sigma2_)
    assert model.influence_.shape == (n,)
    assert model.standardized_residuals_.shape == (n,)
    assert model.cooks_distance_.shape == (n,)
    assert model.parameter_standard_errors_.shape == (n, p + 1)
    assert model.parameter_t_values_.shape == (n, p + 1)
    assert model.intercept_se_.shape == (n,)
    assert model.coef_se_.shape == (n, p)
    assert model.intercept_t_.shape == (n,)
    assert model.coef_t_.shape == (n, p)
    assert np.all(model.parameter_standard_errors_ >= 0.0)


def test_sigma2_conventions_use_documented_denominators(synthetic):
    v1 = GWR(bandwidth=30, adaptive=True, sigma2_v1=True)
    v1.fit(
        synthetic["X"], synthetic["y"], synthetic["coords"], compute_hat_matrix=False
    )
    v2 = GWR(bandwidth=30, adaptive=True, sigma2_v1=False)
    v2.fit(
        synthetic["X"], synthetic["y"], synthetic["coords"], compute_hat_matrix=False
    )

    rss = float(np.dot(v1.residuals_, v1.residuals_))
    trace_s = v1.diagnostics_["trace_S"]
    trace_sts = v1.diagnostics_["trace_StS"]
    np.testing.assert_allclose(v1.sigma2_, rss / (synthetic["n"] - trace_s))
    np.testing.assert_allclose(
        v2.sigma2_, rss / (synthetic["n"] - 2.0 * trace_s + trace_sts)
    )


def test_compute_inference_false_disables_parameter_uncertainty(synthetic):
    model = GWR(bandwidth=30, adaptive=True)
    model.fit(
        synthetic["X"],
        synthetic["y"],
        synthetic["coords"],
        compute_hat_matrix=False,
        compute_inference=False,
    )
    assert model.parameter_standard_errors_ is None
    assert model.parameter_t_values_ is None
    assert model.coef_se_ is None
    result = model.predict_result(synthetic["X"][:3], synthetic["coords"][:3])
    assert result.coef_standard_errors is None
    assert result.intercept_standard_errors is None


def test_prediction_result_and_frame_include_inference(synthetic):
    model = GWR(bandwidth=30, adaptive=True)
    model.fit(
        synthetic["X"], synthetic["y"], synthetic["coords"], compute_hat_matrix=False
    )
    result = model.predict_result(synthetic["X"][:4], synthetic["coords"][:4])
    assert result.coef_standard_errors.shape == (4, synthetic["X"].shape[1])
    assert result.intercept_standard_errors.shape == (4,)
    assert result.coef_t_values.shape == (4, synthetic["X"].shape[1])
    assert result.intercept_t_values.shape == (4,)

    frame = model.to_frame()
    for column in (
        "intercept_se",
        "intercept_t",
        "influence",
        "standardized_residual",
        "cooks_distance",
    ):
        assert column in frame.columns


def test_string_adaptive_bandwidth_alias_is_rejected():
    with pytest.raises(ValueError, match="adaptive=True"):
        GWR(bandwidth="adaptive")
