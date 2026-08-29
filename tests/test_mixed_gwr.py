"""Tests for the standard MixedGWR public estimator and private numerical core."""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pygwrx import MixedGWR
from pygwrx.core.kernels import get_kernel_function
from pygwrx.core.utils import add_intercept, compute_distance_matrix
from pygwrx.models._mixed_gwr_core import (
    compute_mixed_gwr_hat_matrix,
    compute_model_criteria,
    fit_mixed_gwr_core,
)


@pytest.fixture(scope="module")
def mixed_data():
    """Synthetic model with one global and one spatially varying slope."""
    rng = np.random.default_rng(0)
    n = 120
    coords = rng.random((n, 2)) * 10.0
    xg = rng.random(n)
    xl = rng.random(n)
    b_local = 1.0 + 0.4 * coords[:, 0]
    y = 2.0 + 3.0 * xg + b_local * xl + rng.normal(0, 0.1, n)
    X = np.column_stack([xg, xl])
    return {"X": X, "y": y, "coords": coords, "n": n}


def make_model(**kwargs):
    params = dict(
        kernel="bisquare",
        bandwidth=40,
        adaptive=True,
        global_vars=[0],
        local_vars=[1],
        intercept_fixed=True,
        fit_intercept=True,
        verbose=False,
    )
    params.update(kwargs)
    return MixedGWR(**params)


def test_only_one_public_model_and_private_core_file():
    import pygwrx.models as models

    assert models.MixedGWR is MixedGWR
    signature = inspect.signature(MixedGWR)
    assert "max_iter" not in signature.parameters
    assert "tol" not in signature.parameters
    assert "auto_select" not in signature.parameters
    assert "selection_criterion" not in signature.parameters
    assert "selection_method" not in signature.parameters
    root = Path(__file__).resolve().parents[1]
    assert (root / "src/pygwrx/models/_mixed_gwr_core.py").exists()
    assert not (root / "src/pygwrx/models/mixed_gwr_improved.py").exists()


def test_adaptive_recovers_global_coefficients(mixed_data):
    model = make_model().fit(
        mixed_data["X"], mixed_data["y"], mixed_data["coords"], compute_enp=False
    )
    assert abs(float(model.intercept_) - 2.0) < 0.6
    assert abs(float(model.coef_global_[0]) - 3.0) < 0.6


def test_adaptive_differs_from_fixed(mixed_data):
    adaptive = make_model().fit(
        mixed_data["X"], mixed_data["y"], mixed_data["coords"], compute_enp=False
    )
    fixed = make_model(adaptive=False).fit(
        mixed_data["X"], mixed_data["y"], mixed_data["coords"], compute_enp=False
    )
    assert not np.allclose(adaptive.coef_global_, fixed.coef_global_)


def test_training_prediction_uses_same_adaptive_semantics_and_preserves_state(
    mixed_data,
):
    model = make_model().fit(
        mixed_data["X"], mixed_data["y"], mixed_data["coords"], compute_enp=False
    )
    state = (
        model.coef_local_.copy(),
        model.coef_global_.copy(),
        model.fitted_values_.copy(),
        model.residuals_.copy(),
    )
    predicted = model.predict(mixed_data["X"], mixed_data["coords"])
    np.testing.assert_allclose(predicted, model.fitted_values_, atol=1e-10, rtol=1e-10)
    for before, after in zip(
        state,
        (model.coef_local_, model.coef_global_, model.fitted_values_, model.residuals_),
    ):
        np.testing.assert_array_equal(before, after)


def test_dataframe_names_and_prediction_column_order(mixed_data):
    frame = pd.DataFrame(mixed_data["X"], columns=["global", "local"])
    model = MixedGWR(
        bandwidth=40,
        adaptive=True,
        global_vars=["global"],
        local_vars=["local"],
        verbose=False,
    ).fit(frame, mixed_data["y"], mixed_data["coords"], compute_enp=False)
    assert model.global_var_indices_.tolist() == [0]
    assert model.local_var_indices_.tolist() == [1]
    with pytest.raises(ValueError, match="same order"):
        model.predict(frame[["local", "global"]], mixed_data["coords"])


def test_partition_validation_rejects_overlap_and_missing_features(mixed_data):
    with pytest.raises(ValueError, match="overlap"):
        MixedGWR(local_vars=[0], global_vars=[0], bandwidth=40, adaptive=True).fit(
            mixed_data["X"], mixed_data["y"], mixed_data["coords"], compute_enp=False
        )
    with pytest.raises(ValueError, match="missing indices"):
        MixedGWR(local_vars=[0], global_vars=[], bandwidth=40, adaptive=True).fit(
            mixed_data["X"], mixed_data["y"], mixed_data["coords"], compute_enp=False
        )


def test_all_features_local_is_valid_with_global_intercept(mixed_data):
    model = MixedGWR(
        bandwidth=40,
        adaptive=True,
        local_vars=None,
        global_vars=None,
        intercept_fixed=True,
        verbose=False,
    ).fit(mixed_data["X"], mixed_data["y"], mixed_data["coords"], compute_enp=False)
    assert model.coef_global_.shape == (0,)
    assert model.coef_local_.shape == mixed_data["X"].shape
    assert np.isscalar(model.intercept_)


def test_local_intercept_is_returned_as_surface(mixed_data):
    model = make_model(intercept_fixed=False).fit(
        mixed_data["X"], mixed_data["y"], mixed_data["coords"], compute_enp=False
    )
    assert np.asarray(model.intercept_).shape == (mixed_data["n"],)
    assert model.coef_local_.shape == (mixed_data["n"], 1)


def test_collinear_local_design_uses_deterministic_minimum_norm_solution():
    coords = np.column_stack([np.arange(20.0), np.zeros(20)])
    x = np.linspace(-1.0, 1.0, 20)
    X = np.column_stack([x, x])
    y = 1.0 + 2.0 * x
    model = MixedGWR(
        bandwidth=8,
        adaptive=True,
        local_vars=[0, 1],
        global_vars=None,
        ridge=0.0,
        verbose=False,
    ).fit(X, y, coords, compute_enp=False)
    assert np.all(np.isfinite(model.coef_local_))
    # With two identical columns, the Moore-Penrose minimum-norm solution splits
    # the slope equally. This is deterministic across LAPACK/BLAS implementations.
    np.testing.assert_allclose(model.coef_local_, 1.0, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(model.fitted_values_, y, atol=1e-12, rtol=1e-12)


def test_ridge_is_explicit_and_changes_collinear_fit():
    coords = np.column_stack([np.arange(25.0), np.zeros(25)])
    x = np.linspace(-2.0, 2.0, 25)
    X = np.column_stack([x, x + 1e-10 * np.arange(25)])
    y = 3.0 + 2.0 * x
    unregularized = MixedGWR(
        bandwidth=10, adaptive=True, local_vars=[0, 1], ridge=0.0
    ).fit(X, y, coords, compute_enp=False)
    regularized = MixedGWR(
        bandwidth=10, adaptive=True, local_vars=[0, 1], ridge=1e-2
    ).fit(X, y, coords, compute_enp=False)
    assert not np.allclose(unregularized.coef_local_, regularized.coef_local_)


def test_exact_hat_matrix_matches_unit_response_definition():
    rng = np.random.default_rng(12)
    n = 14
    coords = rng.normal(size=(n, 2))
    X_local = add_intercept(rng.normal(size=(n, 1)))
    X_global = rng.normal(size=(n, 1))
    distances = compute_distance_matrix(coords, coords)
    kernel = get_kernel_function("gaussian")
    direct = compute_mixed_gwr_hat_matrix(
        X_local,
        X_global,
        8,
        kernel,
        distances,
        adaptive=True,
    )
    brute = np.empty((n, n))
    for column in range(n):
        response = np.zeros(n)
        response[column] = 1.0
        local, global_ = fit_mixed_gwr_core(
            X_local,
            X_global,
            response,
            8,
            kernel,
            distances,
            adaptive=True,
        )
        brute[:, column] = np.einsum("ij,ij->i", X_local, local) + X_global @ global_
    np.testing.assert_allclose(direct, brute, atol=1e-9, rtol=1e-9)


def test_diagnostics_use_project_standard_information_criteria(mixed_data):
    model = make_model(bandwidth=55).fit(
        mixed_data["X"], mixed_data["y"], mixed_data["coords"], compute_enp=True
    )
    criteria = compute_model_criteria(
        mixed_data["y"], model.fitted_values_, float(model.enp_)
    )
    assert model.aic_ == pytest.approx(criteria["aic"])
    assert model.aicc_ == pytest.approx(criteria["aicc"])
    assert model.bic_ == pytest.approx(criteria["bic"])
    assert model.hat_matrix_.shape == (mixed_data["n"], mixed_data["n"])


def test_failed_refit_clears_previous_state(mixed_data):
    model = make_model().fit(
        mixed_data["X"], mixed_data["y"], mixed_data["coords"], compute_enp=False
    )
    bad = mixed_data["X"].copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError):
        model.fit(bad, mixed_data["y"], mixed_data["coords"], compute_enp=False)
    assert not model.is_fitted_
    assert model.fitted_values_ is None
    assert model.coef_local_ is None


def test_dublin_voter_public_reference_global_coefficients():
    gpd = pytest.importorskip("geopandas")
    data_path = (
        Path(__file__).resolve().parents[1] / "src/pygwrx/data/DubVoter/Dub.voter.shp"
    )
    data = gpd.read_file(data_path)
    feature_names = [
        "DiffAdd",
        "LARent",
        "SC1",
        "Unempl",
        "LowEduc",
        "Age18_24",
        "Age25_44",
        "Age45_64",
    ]
    centroids = data.geometry.centroid
    coords = np.column_stack([centroids.x, centroids.y])
    model = MixedGWR(
        kernel="bisquare",
        bandwidth=109,
        adaptive=True,
        local_vars=["SC1", "Unempl", "Age18_24"],
        global_vars=["DiffAdd", "LARent", "LowEduc", "Age25_44", "Age45_64"],
        intercept_fixed=True,
        verbose=False,
    ).fit(data[feature_names], data["GenEl2004"], coords, compute_enp=False)

    published = np.array([86.31399, -0.15299, -0.11481, 0.12894, -0.53151, -0.25790])
    actual = np.concatenate([[float(model.intercept_)], model.coef_global_])
    np.testing.assert_allclose(actual, published, atol=0.007, rtol=0.0)

    expected_quantiles = np.array(
        [
            [0.01948, -1.03400, -0.40890],
            [0.10360, -0.77250, -0.20600],
            [0.19820, -0.65630, -0.12650],
            [0.42970, -0.52440, -0.06580],
            [0.71120, -0.06680, 0.11390],
        ]
    )
    actual_quantiles = np.quantile(model.coef_local_, [0, 0.25, 0.5, 0.75, 1], axis=0)
    np.testing.assert_allclose(
        actual_quantiles, expected_quantiles, atol=0.015, rtol=0.0
    )
