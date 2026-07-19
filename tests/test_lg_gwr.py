"""Tests for the standardised LG-GWR implementation."""

from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import pytest
from scipy.spatial.distance import cdist

from pygwrx import LGGWR, LGGWRPredictionResult


@pytest.fixture(scope="module")
def latent_data():
    """Attribute-driven synthetic non-stationarity."""
    rng = np.random.default_rng(7)
    n = 70
    coords = rng.random((n, 2)) * 10.0
    attrs = rng.random((n, 2))
    X = rng.random((n, 2))
    beta0 = 1.0 + 2.0 * attrs[:, 0]
    y = 0.5 + beta0 * X[:, 0] - 1.5 * X[:, 1] + rng.normal(0, 0.05, n)
    return {"X": X, "y": y, "coords": coords, "attrs": attrs, "n": n}


@pytest.mark.parametrize("kernel", ["gaussian", "bisquare", "exponential"])
def test_analytical_gradient_matches_finite_difference(kernel):
    """The joint analytical gradient must match central finite differences."""
    rng = np.random.default_rng(0)
    n, p = 25, 2
    coords = rng.random((n, 2)) * 5.0
    attrs = rng.random((n, 2))
    X = rng.random((n, p))
    y = rng.random(n)
    u = np.hstack([coords, attrs])

    model = LGGWR(
        latent_dim=2,
        kernel=kernel,
        lambda_reg=0.01,
        scale_constraint="none",
        fit_intercept=False,
        random_state=1,
        verbose=False,
    )
    model.A_ = model._initialize_A(u.shape[1], np.random.default_rng(1), mode="random")
    h = float(np.median(np.linalg.norm(u @ model.A_.T, axis=1))) + 1.0

    def loss_at(A):
        model.A_ = A
        cache = model._forward_loo(X, y, u @ A.T, h)
        return model._compute_loss(y, cache["yhat"])

    A0 = model.A_.copy()
    z0 = u @ A0.T
    cache0 = model._forward_loo(X, y, z0, h)
    g_analytic = model._compute_gradient(X, y, u, z0, h, cache0)

    eps = 1e-6
    g_num = np.zeros_like(A0)
    for i in range(A0.shape[0]):
        for j in range(A0.shape[1]):
            Ap = A0.copy()
            Ap[i, j] += eps
            Am = A0.copy()
            Am[i, j] -= eps
            g_num[i, j] = (loss_at(Ap) - loss_at(Am)) / (2 * eps)

    assert np.allclose(g_analytic, g_num, atol=1e-4, rtol=1e-3)


@pytest.mark.parametrize("kernel", ["gaussian", "bisquare"])
def test_separable_gradient_matches_finite_difference(kernel):
    """The separable analytical gradient must match finite differences."""
    rng = np.random.default_rng(0)
    n, p, q = 22, 2, 2
    coords = rng.random((n, 2)) * 5.0
    attrs = rng.random((n, q))
    X = rng.random((n, p))
    y = rng.random(n)

    model = LGGWR(
        latent_dim=2,
        geometry="separable",
        kernel=kernel,
        lambda_reg=0.01,
        scale_constraint="none",
        fit_intercept=False,
        random_state=1,
        verbose=False,
    )
    model.B_ = model._initialize_B(q, np.random.default_rng(1), mode="random")
    dg = cdist(coords, coords)
    z0 = attrs @ model.B_.T
    h_g = 2.0 * float(dg.max())
    h_a = 2.0 * float(cdist(z0, z0).max())
    Kg = model._kernel_weights(dg, h_g)

    def loss_at(B):
        model.B_ = B
        cache = model._forward_loo_sep(X, y, Kg, attrs @ B.T, h_a)
        return float(np.mean((y - cache["yhat"]) ** 2) + 0.01 * np.sum(B**2))

    B0 = model.B_.copy()
    cache0 = model._forward_loo_sep(X, y, Kg, attrs @ B0.T, h_a)
    g_analytic = model._compute_gradient_sep(X, y, attrs, attrs @ B0.T, h_a, cache0)

    eps = 1e-6
    g_num = np.zeros_like(B0)
    for i in range(B0.shape[0]):
        for j in range(B0.shape[1]):
            Bp = B0.copy()
            Bp[i, j] += eps
            Bm = B0.copy()
            Bm[i, j] -= eps
            g_num[i, j] = (loss_at(Bp) - loss_at(Bm)) / (2 * eps)

    assert np.allclose(g_analytic, g_num, atol=1e-4, rtol=1e-3)


def test_fixed_norm_and_l2_cannot_be_combined():
    with pytest.raises(ValueError, match="lambda_reg must be 0"):
        LGGWR(lambda_reg=0.01)


def test_training_learns_attribute_geometry(latent_data):
    data = latent_data
    model = LGGWR(max_iter=50, random_state=0, verbose=False)
    model.fit(data["X"], data["y"], data["coords"], data["attrs"])
    assert model.best_loss_ <= model.loss_history_[0]
    metric = model.metric_frame()
    contribution = metric.loc[
        metric["geometry_feature"] == "attr_0", "metric_contribution"
    ].iloc[0]
    assert contribution > 0.5
    assert model.diagnostics_["r2"] > 0.95


def test_learning_helps_versus_no_learning(latent_data):
    data = latent_data
    learned = LGGWR(max_iter=50, learning_rate=0.05, random_state=0, verbose=False)
    learned.fit(data["X"], data["y"], data["coords"], data["attrs"])
    baseline = LGGWR(
        max_iter=1,
        learning_rate=0.0,
        select_bandwidth=False,
        bandwidth_updates=0,
        random_state=0,
        verbose=False,
    )
    baseline.fit(data["X"], data["y"], data["coords"], data["attrs"])
    assert np.sum(learned.residuals_**2) < np.sum(baseline.residuals_**2)


def test_reproducible_restarts(latent_data):
    data = latent_data
    kwargs = dict(max_iter=20, n_restarts=2, random_state=42, verbose=False)
    first = LGGWR(**kwargs).fit(data["X"], data["y"], data["coords"], data["attrs"])
    second = LGGWR(**kwargs).fit(data["X"], data["y"], data["coords"], data["attrs"])
    assert np.allclose(first.A_, second.A_)
    assert np.allclose(first.coef_, second.coef_)
    assert first.restart_scores_ == second.restart_scores_


def test_geometry_standardisation_is_unit_invariant(latent_data):
    data = latent_data
    base = LGGWR(max_iter=25, random_state=3, verbose=False).fit(
        data["X"], data["y"], data["coords"], data["attrs"]
    )
    transformed_coords = data["coords"] * 1000.0 + np.array([2e6, -4e6])
    transformed_attrs = data["attrs"] * np.array([100.0, 0.01]) + np.array([9.0, -7.0])
    scaled = LGGWR(max_iter=25, random_state=3, verbose=False).fit(
        data["X"], data["y"], transformed_coords, transformed_attrs
    )
    assert np.allclose(base.fitted_values_, scaled.fitted_values_, atol=1e-7)
    assert np.allclose(
        base.metric_contributions_, scaled.metric_contributions_, atol=1e-7
    )


def test_dataframe_names_prediction_result_and_frames(latent_data):
    data = latent_data
    X = pd.DataFrame(data["X"], columns=["income", "housing"])
    coords = pd.DataFrame(data["coords"], columns=["east", "north"])
    attrs = pd.DataFrame(data["attrs"], columns=["context", "noise"])
    model = LGGWR(max_iter=15, random_state=0, verbose=False).fit(
        X, data["y"], coords, attrs
    )
    result = model.predict_result(X, coords, attrs)
    assert isinstance(result, LGGWRPredictionResult)
    assert result.feature_names == ("income", "housing")
    assert model.to_frame().shape[0] == data["n"]
    assert set(["coef_income", "coef_housing", "latent_0"]).issubset(model.to_frame())
    assert result.to_frame().shape[0] == data["n"]
    assert np.allclose(result.predictions, model.fitted_values_)
    with pytest.raises(ValueError, match="columns must match"):
        model.predict(X[["housing", "income"]], coords, attrs)


def test_legacy_intercept_is_detected_without_duplication(latent_data):
    data = latent_data
    X_legacy = np.column_stack([np.ones(data["n"]), data["X"]])
    with pytest.warns(UserWarning, match="all-ones"):
        model = LGGWR(max_iter=10, random_state=0, verbose=False).fit(
            X_legacy, data["y"], data["coords"], data["attrs"]
        )
    assert model.X_design_.shape[1] == 3
    assert model.coef_.shape == (data["n"], 2)
    assert model.intercept_.shape == (data["n"],)


def test_no_intercept_path(latent_data):
    data = latent_data
    model = LGGWR(fit_intercept=False, max_iter=12, random_state=0, verbose=False).fit(
        data["X"], data["y"], data["coords"], data["attrs"]
    )
    assert model.coefficients_.shape == (data["n"], 2)
    assert np.all(model.intercept_ == 0.0)


def test_constant_attribute_is_safe(latent_data):
    data = latent_data
    attrs = np.column_stack([data["attrs"], np.ones(data["n"])])
    model = LGGWR(max_iter=12, random_state=0, verbose=False).fit(
        data["X"], data["y"], data["coords"], attrs
    )
    assert model.constant_attribute_mask_.tolist() == [False, False, True]
    assert np.all(np.isfinite(model.A_))


def test_final_loss_and_bandwidth_history_match_final_state(latent_data):
    data = latent_data
    model = LGGWR(max_iter=20, random_state=0, verbose=False).fit(
        data["X"], data["y"], data["coords"], data["attrs"]
    )
    cache = model._forward_loo(
        model.X_design_, model.y_train_, model.latent_coords_, model.bandwidth_
    )
    assert model.final_loo_loss_ == pytest.approx(
        model._compute_loss(model.y_train_, cache["yhat"])
    )
    assert len(model.bandwidth_history_) == 3
    assert model.n_iter_ == len(model.loss_history_)
    assert model.stop_reason_ in {
        "max_iter",
        "patience",
        "tolerance",
        "nonfinite_loss",
        "nonfinite_gradient",
    }


def test_bandwidth_selection_never_worsens_same_geometry_aicc(latent_data):
    data = latent_data
    fixed = LGGWR(
        max_iter=20,
        select_bandwidth=False,
        bandwidth_updates=0,
        random_state=0,
        verbose=False,
    ).fit(data["X"], data["y"], data["coords"], data["attrs"])
    selected = LGGWR(
        max_iter=20,
        select_bandwidth=True,
        bandwidth_updates=0,
        random_state=0,
        verbose=False,
    ).fit(data["X"], data["y"], data["coords"], data["attrs"])
    assert np.allclose(fixed.A_, selected.A_)
    assert selected.diagnostics_["aicc"] <= fixed.diagnostics_["aicc"] + 1e-8


def test_separable_recovers_geographic_gwr_when_attributes_off(latent_data):
    data = latent_data
    model = LGGWR(
        geometry="separable",
        bandwidth_updates=0,
        select_bandwidth=False,
        max_iter=5,
        random_state=0,
        verbose=False,
    ).fit(data["X"], data["y"], data["coords"], data["attrs"])
    h_g, _ = model.bandwidth_
    dg = cdist(model.coords_geometry_, model.coords_geometry_)
    zeta = model.attrs_geometry_ @ model.B_.T
    betas_off, _ = model._local_fit_with_hat_sep(
        model.X_design_, model.y_train_, dg, zeta, h_g, np.inf
    )
    weights = model._kernel_weights(dg, h_g)
    yhat_geo = np.zeros(data["n"])
    for i in range(data["n"]):
        beta, _ = model._hat_solution(
            model.X_design_, model.y_train_, weights[i], model.X_design_[i]
        )
        yhat_geo[i] = model.X_design_[i] @ beta
    assert np.allclose(
        np.einsum("ij,ij->i", model.X_design_, betas_off), yhat_geo, atol=1e-8
    )


def test_separable_fit_predict_and_metric(latent_data):
    data = latent_data
    model = LGGWR(geometry="separable", max_iter=20, random_state=0, verbose=False).fit(
        data["X"], data["y"], data["coords"], data["attrs"]
    )
    assert model.metric_matrix_.shape == (2, 2)
    assert model.metric_frame().shape[0] == 2
    assert np.sum(model.metric_contributions_) == pytest.approx(1.0)
    assert np.allclose(
        model.predict(data["X"], data["coords"], data["attrs"]),
        model.fitted_values_,
    )


def test_no_divergence_on_heavy_tailed_data():
    rng = np.random.default_rng(0)
    n = 90
    coords = rng.random((n, 2)) * 10.0
    attrs = rng.random((n, 2))
    X = rng.random((n, 2))
    y = 0.5 + X[:, 0] * 2.0 - X[:, 1] + rng.normal(0, 0.1, n)
    y[0] += 60.0
    model = LGGWR(max_iter=60, random_state=0, verbose=False).fit(X, y, coords, attrs)
    assert np.all(np.isfinite(model.A_))
    assert np.all(np.isfinite(model.fitted_values_))
    assert np.isfinite(model.final_loo_loss_)
    assert np.all(np.isfinite(list(model.diagnostics_.values())))


def test_failed_refit_clears_state(latent_data):
    data = latent_data
    model = LGGWR(max_iter=8, random_state=0, verbose=False).fit(
        data["X"], data["y"], data["coords"], data["attrs"]
    )
    with pytest.raises(ValueError, match="same rows"):
        model.fit(data["X"][:-1], data["y"], data["coords"], data["attrs"])
    assert not model._is_fitted
    assert model.A_ is None
    assert model.fitted_values_ is None


def test_pickle_round_trip_preserves_predictions(latent_data):
    data = latent_data
    model = LGGWR(max_iter=10, random_state=0, verbose=False).fit(
        data["X"], data["y"], data["coords"], data["attrs"]
    )
    restored = pickle.loads(pickle.dumps(model))
    assert np.allclose(
        restored.predict(data["X"], data["coords"], data["attrs"]),
        model.fitted_values_,
    )
