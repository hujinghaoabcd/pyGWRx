"""End-to-end checks for the flagship GWR model."""

import numpy as np
import pytest

from pygwrx import GWR


@pytest.fixture(scope="module")
def fitted_gwr(synthetic):
    model = GWR(kernel="gaussian", bandwidth="cv", adaptive=True)
    model.fit(synthetic["X"], synthetic["y"], synthetic["coords"])
    return model, synthetic


def test_fit_populates_attributes(fitted_gwr):
    model, data = fitted_gwr
    n, k = data["X"].shape
    assert np.asarray(model.coef_).shape == (n, k)
    assert np.asarray(model.intercept_).shape == (n,)
    assert np.asarray(model.fitted_values_).shape == (n,)
    assert np.asarray(model.residuals_).shape == (n,)
    assert model.bandwidth_ is not None and model.bandwidth_ > 0


def test_diagnostics_are_sane(fitted_gwr):
    model, _ = fitted_gwr
    diag = model.diagnostics_
    for key in ("r2", "adj_r2", "aic", "aicc", "bic", "rmse"):
        assert key in diag
    # the model has real signal, so R^2 should be clearly positive
    assert 0.0 < diag["r2"] <= 1.0 + 1e-9
    assert diag["rmse"] >= 0.0


def test_predict_shape_and_finiteness(fitted_gwr):
    model, data = fitted_gwr
    preds = model.predict(data["X"], data["coords"])
    assert np.asarray(preds).shape == (data["n"],)
    assert np.all(np.isfinite(preds))


def test_fixed_bandwidth_numeric_value(synthetic):
    model = GWR(kernel="bisquare", bandwidth=5.0, adaptive=False)
    model.fit(synthetic["X"], synthetic["y"], synthetic["coords"])
    assert model.bandwidth_ == pytest.approx(5.0)


def test_adaptive_predict_matches_fitted_on_training_points(synthetic):
    """Regression guard: predicting on the training coordinates of an ADAPTIVE model
    must reproduce its fitted values.

    Previously _predict_basic hardcoded adaptive=False, so bandwidth_ (a neighbour
    count k) was treated as a distance and adaptive predictions collapsed toward
    global OLS — predict() on the training points diverged wildly from fitted_values_.
    """
    model = GWR(kernel="gaussian", bandwidth="aicc", adaptive=True)
    model.fit(synthetic["X"], synthetic["y"], synthetic["coords"])
    preds = model.predict(synthetic["X"], synthetic["coords"])
    # predictions on the training points should closely track the fitted values
    assert np.corrcoef(preds, model.fitted_values_)[0, 1] > 0.99
