# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Validation tests for the standard GWGLM implementation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pygwrx import GWGLM, GWR


def _data(seed: int = 123, n: int = 30):
    rng = np.random.default_rng(seed)
    coords = rng.uniform(0.0, 8.0, size=(n, 2))
    X = rng.normal(size=(n, 2))
    X_design = np.column_stack([np.ones(n), X])
    exposure = rng.uniform(0.6, 2.0, size=n)
    poisson_mu = exposure * np.exp(X_design @ np.array([0.15, 0.35, -0.25]))
    y_poisson = rng.poisson(poisson_mu).astype(float)
    probability = 1.0 / (1.0 + np.exp(-(X_design @ np.array([-0.10, 0.65, -0.45]))))
    y_binomial = rng.binomial(1, probability).astype(float)
    return X, coords, exposure, y_poisson, y_binomial


def test_gaussian_family_degenerates_to_standard_gwr():
    X, coords, _, _, _ = _data()
    y = 2.0 + 0.7 * X[:, 0] - 0.4 * X[:, 1]
    gwr = GWR(kernel="bisquare", bandwidth=5.0).fit(X, y, coords)
    model = GWGLM(family="gaussian", kernel="bisquare", bandwidth=5.0).fit(X, y, coords)
    np.testing.assert_allclose(model.intercept_, gwr.intercept_)
    np.testing.assert_allclose(model.coef_, gwr.coef_)
    np.testing.assert_allclose(model.fitted_values_, gwr.fitted_values_)
    assert model.family_ == "gaussian"
    assert model.converged_ is True


@pytest.mark.reference
@pytest.mark.parametrize("family", ["poisson", "binomial"])
def test_fixed_bandwidth_matches_mgwr_reference(family):
    mgwr_gwr = pytest.importorskip("mgwr.gwr")
    spglm_family = pytest.importorskip("spglm.family")
    X, coords, exposure, y_poisson, y_binomial = _data()
    y = y_poisson if family == "poisson" else y_binomial
    kwargs = {"exposure": exposure} if family == "poisson" else {}
    model = GWGLM(
        family=family,
        kernel="bisquare",
        bandwidth=5.5,
        max_iter=100,
        tol=1.0e-8,
    ).fit(X, y, coords, **kwargs)
    reference_family = (
        spglm_family.Poisson() if family == "poisson" else spglm_family.Binomial()
    )
    reference = mgwr_gwr.GWR(
        coords,
        y.reshape(-1, 1),
        X,
        5.5,
        family=reference_family,
        offset=(exposure.reshape(-1, 1) if family == "poisson" else None),
        kernel="bisquare",
        fixed=True,
        constant=True,
        n_jobs=1,
    ).fit(tol=1.0e-8, max_iter=100)
    np.testing.assert_allclose(
        np.column_stack([model.intercept_, model.coef_]),
        reference.params,
        atol=1.0e-6,
        rtol=1.0e-6,
    )
    np.testing.assert_allclose(
        model.fitted_values_, reference.predy.ravel(), atol=1.0e-6, rtol=1.0e-6
    )
    np.testing.assert_allclose(
        model.influence_, reference.influ.ravel(), atol=1.0e-6, rtol=1.0e-6
    )
    np.testing.assert_allclose(
        model.parameter_standard_errors_, reference.bse, atol=1.0e-6, rtol=1.0e-6
    )
    assert model.diagnostics_["aicc"] == pytest.approx(reference.aicc, abs=1.0e-5)


@pytest.mark.reference
def test_adaptive_poisson_matches_mgwr_reference():
    mgwr_gwr = pytest.importorskip("mgwr.gwr")
    spglm_family = pytest.importorskip("spglm.family")
    X, coords, exposure, y, _ = _data(seed=8, n=36)
    model = GWGLM(
        family="poisson",
        kernel="bisquare",
        bandwidth=18,
        adaptive=True,
        max_iter=100,
        tol=1.0e-8,
    ).fit(X, y, coords, exposure=exposure)
    reference = mgwr_gwr.GWR(
        coords,
        y.reshape(-1, 1),
        X,
        18,
        family=spglm_family.Poisson(),
        offset=exposure.reshape(-1, 1),
        kernel="bisquare",
        fixed=False,
        constant=True,
        n_jobs=1,
    ).fit(tol=1.0e-8, max_iter=100)
    np.testing.assert_allclose(
        np.column_stack([model.intercept_, model.coef_]),
        reference.params,
        atol=2.0e-6,
        rtol=2.0e-6,
    )
    np.testing.assert_allclose(
        model.fitted_values_, reference.predy.ravel(), atol=2.0e-6, rtol=2.0e-6
    )
    assert model.diagnostics_["aicc"] == pytest.approx(reference.aicc, abs=1.0e-4)


def test_poisson_exposure_and_offset_are_equivalent():
    X, coords, exposure, y, _ = _data()
    by_exposure = GWGLM(family="poisson", kernel="bisquare", bandwidth=5.5).fit(
        X, y, coords, exposure=exposure
    )
    by_offset = GWGLM(family="poisson", kernel="bisquare", bandwidth=5.5).fit(
        X, y, coords, offset=np.log(exposure)
    )
    np.testing.assert_allclose(by_exposure.coef_, by_offset.coef_)
    np.testing.assert_allclose(by_exposure.fitted_values_, by_offset.fitted_values_)
    prediction_exposure = by_exposure.predict(X[:4], coords[:4], exposure=exposure[:4])
    prediction_offset = by_exposure.predict(
        X[:4], coords[:4], offset=np.log(exposure[:4])
    )
    np.testing.assert_allclose(prediction_exposure, prediction_offset)


def test_aicc_grid_bandwidth_matches_explicit_candidate_scores():
    X, coords, exposure, y, _ = _data(seed=19, n=18)
    candidates = range(8, 11)
    scores = {}
    for bandwidth in candidates:
        fitted = GWGLM(
            family="poisson",
            kernel="bisquare",
            bandwidth=bandwidth,
            adaptive=True,
            max_iter=60,
        ).fit(X, y, coords, exposure=exposure)
        scores[bandwidth] = fitted.diagnostics_["aicc"]
    selected = GWGLM(
        family="poisson",
        kernel="bisquare",
        bandwidth="aicc",
        adaptive=True,
        bandwidth_range=(8, 10),
        optimization_method="grid",
        max_iter=60,
    ).fit(X, y, coords, exposure=exposure)
    expected = min(scores, key=lambda value: (scores[value], value))
    assert selected.bandwidth_ == expected
    assert selected.bandwidth_selection_score_ == pytest.approx(scores[expected])


def test_cv_bandwidth_records_leave_one_out_residuals():
    X, coords, _, _, y = _data(seed=77, n=18)
    model = GWGLM(
        family="binomial",
        kernel="bisquare",
        bandwidth="cv",
        adaptive=True,
        bandwidth_range=(8, 10),
        optimization_method="grid",
        max_iter=60,
    ).fit(X, y, coords)
    assert model.bandwidth_ in {8, 9, 10}
    assert model.cv_residuals_.shape == (18,)
    np.testing.assert_allclose(model.cv_contributions_, model.cv_residuals_**2)
    assert model.bandwidth_selection_score_ == pytest.approx(
        float(np.sum(model.cv_contributions_))
    )


def test_validation_rejects_unsupported_responses_and_exposure():
    X, coords, exposure, y_poisson, y_binomial = _data()
    with pytest.raises(ValueError, match="Gamma"):
        GWGLM(family="gamma")
    with pytest.raises(ValueError, match="non-negative"):
        GWGLM(family="poisson", bandwidth=5.0).fit(X, -np.ones_like(y_poisson), coords)
    with pytest.raises(ValueError, match="Bernoulli"):
        GWGLM(family="binomial", bandwidth=5.0).fit(
            X, np.full_like(y_binomial, 0.5), coords
        )
    with pytest.raises(ValueError, match="strictly positive"):
        GWGLM(family="poisson", bandwidth=5.0).fit(
            X, y_poisson, coords, exposure=np.zeros_like(exposure)
        )
    with pytest.raises(ValueError, match="not both"):
        GWGLM(family="poisson", bandwidth=5.0).fit(
            X,
            y_poisson,
            coords,
            exposure=exposure,
            offset=np.log(exposure),
        )


def test_dataframe_schema_state_safety_and_export():
    X, coords, exposure, y, _ = _data()
    frame = pd.DataFrame(X, columns=["income", "density"])
    model = GWGLM(family="poisson", bandwidth=5.5).fit(
        frame, y, coords, exposure=exposure
    )
    exported = model.to_frame()
    assert {"coef_income", "coef_density", "deviance_residual", "exposure"} <= set(
        exported.columns
    )
    with pytest.raises(ValueError, match="same order"):
        model.predict(
            frame[["density", "income"]].iloc[:2],
            coords[:2],
            exposure=exposure[:2],
        )
    old_coef = model.coef_.copy()
    X[:] = 999.0
    np.testing.assert_allclose(model.coef_, old_coef)
    with pytest.raises(ValueError):
        model.fit(frame, -np.ones_like(y), coords)
    assert model.is_fitted_ is False
    assert model.coef_ is None


def test_nonconvergence_is_exposed_without_hiding_fit_results():
    X, coords, exposure, y, _ = _data(seed=15)
    model = GWGLM(
        family="poisson",
        bandwidth=5.5,
        max_iter=1,
        tol=1.0e-14,
    ).fit(X, y, coords, exposure=exposure)
    assert model.is_fitted_ is True
    assert model.converged_ is False
    assert np.any(~model.local_converged_)
    assert np.all(np.isfinite(model.fitted_values_))


def test_local_iwls_honours_explicit_initial_parameters():
    """An explicit starting vector must initialize eta and mu, not only stopping."""
    X, coords, exposure, y, _ = _data(seed=91, n=20)
    model = GWGLM(family="poisson", bandwidth=5.5, max_iter=1, tol=1.0e-14)
    X_design = np.column_stack([np.ones(X.shape[0]), X])
    spatial_weights = np.ones(X.shape[0])
    initial = np.array([0.2, -0.1, 0.05])
    result = model._iwls(X_design, y, spatial_weights, exposure, initial_params=initial)
    # Reproduce one IWLS update from the supplied starting vector.
    eta0 = X_design @ initial
    mu0 = np.exp(eta0) * exposure
    sqrt_w = np.sqrt(mu0)
    z = eta0 + (y - mu0) / mu0
    wx = X_design * sqrt_w[:, None]
    wz = z * sqrt_w
    xtw = wx.T * spatial_weights
    system = xtw @ wx + 1.0e-8 * np.eye(X_design.shape[1])
    expected = np.linalg.solve(system, xtw @ wz)
    np.testing.assert_allclose(result.params, expected, rtol=1.0e-10, atol=1.0e-10)
