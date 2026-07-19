from types import SimpleNamespace

import numpy as np
import pytest

from pygwrx.diagnostics import LocalCollinearityDiagnostics


def make_model(
    X,
    coords,
    *,
    bandwidth=3.0,
    adaptive=False,
    kernel="gaussian",
    distance_metric="euclidean",
    fit_intercept=True,
):
    return SimpleNamespace(
        _is_fitted=True,
        X_train_=np.asarray(X, dtype=float),
        coords_train_=np.asarray(coords, dtype=float),
        bandwidth_=bandwidth,
        adaptive=adaptive,
        kernel=kernel,
        distance_metric=distance_metric,
        fit_intercept=fit_intercept,
    )


@pytest.fixture
def regular_data():
    x1 = np.linspace(0.0, 5.0, 8)
    x2 = np.array([0.0, 1.0, 0.5, 2.0, 1.5, 3.0, 2.2, 4.0])
    X = np.column_stack([x1, x2])
    coords = np.column_stack([x1, np.zeros_like(x1)])
    return X, coords


def test_requires_fitted_model(regular_data):
    X, coords = regular_data
    model = make_model(X, coords)
    model._is_fitted = False
    with pytest.raises(ValueError, match="must be fitted"):
        LocalCollinearityDiagnostics(model)


def test_requires_at_least_two_predictors():
    model = make_model(
        np.arange(5.0)[:, None], np.column_stack([np.arange(5), np.zeros(5)])
    )
    with pytest.raises(ValueError, match="at least two"):
        LocalCollinearityDiagnostics(model)


def test_fixed_bandwidth_uses_model_distance_metric():
    X = np.column_stack([np.arange(3.0), [0.0, 1.0, 4.0]])
    coords = np.array([[0.0, 0.0], [1.0, 1.0], [3.0, 0.0]])
    model = make_model(
        X,
        coords,
        bandwidth=1.5,
        kernel="boxcar",
        distance_metric="manhattan",
    )
    diag = LocalCollinearityDiagnostics(model)
    weights = diag._get_local_weights(0)

    # Manhattan distance to [1, 1] is 2, so only the focal point is included.
    assert weights[0] == pytest.approx(1.0)
    assert weights[1] == 0.0
    assert weights[2] == 0.0


def test_adaptive_bandwidth_uses_kth_neighbour_distance():
    X = np.column_stack([np.arange(4.0), [0.0, 1.0, 4.0, 9.0]])
    coords = np.column_stack([np.array([0.0, 1.0, 2.0, 10.0]), np.zeros(4)])
    model = make_model(
        X,
        coords,
        bandwidth=3,
        adaptive=True,
        kernel="bisquare",
    )
    diag = LocalCollinearityDiagnostics(model)
    weights = diag._get_local_weights(0)

    # k=3 includes distances 0, 1, and 2; the far point at 10 is excluded.
    assert np.count_nonzero(weights > 0.0) == 3
    assert weights[2] > 0.0
    assert weights[3] == 0.0
    assert weights.sum() == pytest.approx(1.0)


def test_complete_diagnostics_shapes_and_vdp_normalization(regular_data):
    X, coords = regular_data
    model = make_model(X, coords, bandwidth=4.0, fit_intercept=True)
    diag = LocalCollinearityDiagnostics(model)
    result = diag.diagnose(verbose=False)

    assert result["local_correlations"].shape == (8, 1)
    assert result["vif"].shape == (8, 2)
    assert result["condition_number"].shape == (8,)
    assert result["vdp"].shape == (8, 3, 3)
    assert result["design_names"] == ["intercept", "x0", "x1"]
    np.testing.assert_allclose(
        np.nansum(result["vdp"], axis=1),
        np.ones((8, 3)),
        atol=1e-8,
    )


def test_without_intercept_vdp_uses_predictor_dimension(regular_data):
    X, coords = regular_data
    model = make_model(X, coords, bandwidth=4.0, fit_intercept=False)
    result = LocalCollinearityDiagnostics(model).diagnose(verbose=False)
    assert result["vdp"].shape == (8, 2, 2)
    assert result["design_names"] == ["x0", "x1"]


def test_exact_collinearity_is_reported_as_infinite():
    x = np.linspace(0.0, 5.0, 10)
    X = np.column_stack([x, 2.0 * x])
    coords = np.column_stack([x, np.zeros_like(x)])
    model = make_model(X, coords, bandwidth=10.0)
    result = LocalCollinearityDiagnostics(model).diagnose(verbose=False)

    assert np.isinf(result["vif"]).all()
    assert np.isinf(result["condition_number"]).all()
    assert result["summary"]["max_vif"] == np.inf
    assert result["summary"]["max_cn"] == np.inf
    assert result["summary"]["pct_severe_vif_locations"] == 100.0


def test_locally_constant_predictor_is_not_hidden():
    x = np.linspace(0.0, 5.0, 8)
    X = np.column_stack([x, np.ones_like(x)])
    coords = np.column_stack([x, np.zeros_like(x)])
    model = make_model(X, coords, bandwidth=10.0)
    result = LocalCollinearityDiagnostics(model).diagnose(verbose=False)

    assert np.isinf(result["vif"][:, 1]).all()
    assert np.isnan(result["local_correlations"]).all()


def test_summary_counts_locations_not_cells(regular_data, monkeypatch):
    X, coords = regular_data
    diag = LocalCollinearityDiagnostics(make_model(X, coords))
    computed = diag._compute_all()
    computed["vif"] = np.array(
        [
            [11.0, 1.0],
            [12.0, 1.0],
            [13.0, 1.0],
            [14.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
        ]
    )
    computed["condition_number"] = np.ones(8)
    monkeypatch.setattr(diag, "_compute_all", lambda: computed)

    summary = diag.diagnose(verbose=False)["summary"]
    assert summary["pct_severe_vif_locations"] == 50.0
    assert summary["pct_severe_vif_cells"] == 25.0
    assert summary["pct_severe_vif"] == 50.0


def test_rejects_multiscale_and_spatiotemporal_models(regular_data):
    X, coords = regular_data

    multiscale = make_model(X, coords)
    multiscale.bandwidths_ = np.array([2.0, 3.0])
    with pytest.raises(NotImplementedError, match="multiscale"):
        LocalCollinearityDiagnostics(multiscale)

    spatiotemporal = make_model(X, coords)
    spatiotemporal.times_train_ = np.arange(X.shape[0], dtype=float)
    with pytest.raises(NotImplementedError, match="Spatiotemporal"):
        LocalCollinearityDiagnostics(spatiotemporal)


def test_public_compute_methods_share_cached_computation(regular_data, monkeypatch):
    X, coords = regular_data
    diag = LocalCollinearityDiagnostics(make_model(X, coords, bandwidth=4.0))
    calls = {"count": 0}
    original = diag._get_local_weights

    def wrapped(index):
        calls["count"] += 1
        return original(index)

    monkeypatch.setattr(diag, "_get_local_weights", wrapped)
    diag.compute_vif()
    diag.compute_condition_number()
    diag.compute_vdp()
    diag.compute_local_correlations()

    assert calls["count"] == X.shape[0]


def test_integration_with_fitted_gwr(regular_data):
    from pygwrx.models import GWR

    X, coords = regular_data
    y = 2.0 + 1.5 * X[:, 0] - 0.4 * X[:, 1]
    model = GWR(bandwidth=4.0, adaptive=False)
    model.fit(X, y, coords, compute_hat_matrix_flag=False)

    result = LocalCollinearityDiagnostics(model).diagnose(verbose=False)
    assert result["vif"].shape == X.shape
    assert np.isfinite(result["condition_number"]).all()
