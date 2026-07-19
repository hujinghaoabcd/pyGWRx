"""Smoke tests: each model fits on synthetic data without raising.

These are fast, dependency-light sanity checks — not accuracy tests. They
guard against regressions that break a model's ``fit`` entirely.
"""

import warnings

import numpy as np
import pytest

import pygwrx


def _fit_regression_model(factory, data):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = factory()
        model.fit(data["X"], data["y"], data["coords"])
    return model


# (id, factory) for regression models sharing the fit(X, y, coords) signature
REGRESSION_MODELS = [
    ("GWR", lambda: pygwrx.GWR(bandwidth="cv", adaptive=True)),
    ("RGWR", lambda: pygwrx.RGWR()),
    ("LCRGWR", lambda: pygwrx.LCRGWR(bandwidth=30, verbose=False)),
    ("ScalableGWR", lambda: pygwrx.ScalableGWR(bandwidth=30, verbose=False)),
]


def _local_coefficients(model):
    """Return per-location coefficients regardless of the attribute name.

    Models are inconsistent: GWR/RGWR expose ``coef_`` while LCRGWR/ScalableGWR
    expose ``coefficients_``. Accept either.
    """
    for attr in ("coef_", "coefficients_"):
        if hasattr(model, attr):
            return np.asarray(getattr(model, attr))
    raise AttributeError("model exposes neither coef_ nor coefficients_")


@pytest.mark.parametrize(
    "name,factory", REGRESSION_MODELS, ids=[m[0] for m in REGRESSION_MODELS]
)
def test_regression_model_fits(name, factory, synthetic):
    model = _fit_regression_model(factory, synthetic)
    coef = _local_coefficients(model)
    assert coef.shape[0] == synthetic["n"]
    assert np.all(np.isfinite(coef))
    # every regression model exposes fitted values of the right length
    fitted = np.asarray(model.fitted_values_)
    assert fitted.shape[0] == synthetic["n"]
    assert np.all(np.isfinite(fitted))


def test_gtwr_fits(synthetic):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = pygwrx.GTWR(adaptive=True, causal=False)
        model.fit(
            synthetic["X"], synthetic["y"], synthetic["coords"], synthetic["times"]
        )
    assert np.asarray(model.coef_).shape[0] == synthetic["n"]


def test_gwss_fits(synthetic):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = pygwrx.GWSS(bandwidth=30, adaptive=True, verbose=False)
        model.fit(synthetic["X"], synthetic["coords"])
    # GWSS is a summary-statistics model; just require it fit without error
    assert model is not None


def test_gwpca_fits(synthetic):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = pygwrx.GWPCA(verbose=False)
        model.fit(synthetic["X"], synthetic["coords"])
    assert model is not None
