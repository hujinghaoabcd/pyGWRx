"""Numerical and API tests for the standardized GWPCA implementation."""

import numpy as np
import pandas as pd
import pytest
from sklearn.decomposition import PCA

from pygwrx.models import GWPCA


def _kernel(distances, bandwidth, name):
    z = distances / bandwidth
    if name == "gaussian":
        return np.exp(-0.5 * z**2)
    if name == "bisquare":
        return np.where(distances <= bandwidth, (1.0 - z**2) ** 2, 0.0)
    if name == "boxcar":
        return (distances <= bandwidth).astype(float)
    raise AssertionError(name)


def _weights(distances, bandwidth, kernel, adaptive):
    if not adaptive:
        return _kernel(distances, float(bandwidth), kernel)
    order = np.argsort(distances, kind="stable")
    k = int(bandwidth)
    if kernel == "boxcar":
        result = np.zeros_like(distances, dtype=float)
        result[order[:k]] = 1.0
        return result
    local_bandwidth = distances[order[k - 1]]
    if local_bandwidth == 0:
        result = np.zeros_like(distances, dtype=float)
        result[order[:k]] = 1.0
        return result
    return _kernel(distances, local_bandwidth, kernel)


def _canonicalize(loadings):
    result = loadings.copy()
    for component in range(result.shape[1]):
        pivot = np.argmax(np.abs(result[:, component]))
        if result[pivot, component] < 0:
            result[:, component] *= -1
    return result


def _reference_gwpca(
    X,
    coords,
    eval_coords,
    *,
    bandwidth,
    kernel,
    adaptive,
    n_components,
    scaling,
):
    mean = X.mean(axis=0)
    scale = X.std(axis=0, ddof=1) if scaling else np.ones(X.shape[1])
    processed = (X - mean) / scale
    distances = np.sqrt(
        ((eval_coords[:, None, :] - coords[None, :, :]) ** 2).sum(axis=2)
    )
    loadings = np.empty((len(eval_coords), X.shape[1], n_components))
    variances = np.empty((len(eval_coords), X.shape[1]))
    means = np.empty((len(eval_coords), X.shape[1]))
    for i, distance in enumerate(distances):
        weight = _weights(distance, bandwidth, kernel, adaptive)
        use = weight > 0
        w = weight[use]
        local = processed[use]
        local_mean = (local * w[:, None]).sum(axis=0) / w.sum()
        weighted = (local - local_mean) * np.sqrt(w)[:, None]
        _, singular, vt = np.linalg.svd(weighted, full_matrices=False)
        loadings[i] = _canonicalize(vt.T[:, :n_components])
        variances[i] = 0.0
        variances[i, : len(singular)] = singular**2 / w.sum()
        means[i] = local_mean
    pv = variances[:, :n_components] / variances.sum(axis=1, keepdims=True) * 100
    return loadings, variances, means, pv


@pytest.fixture
def data():
    rng = np.random.default_rng(324)
    coords = np.column_stack([np.linspace(0, 8, 24), np.sin(np.linspace(0, 3, 24))])
    latent = rng.normal(size=(24, 2))
    X = np.column_stack(
        [
            latent[:, 0] + 0.1 * rng.normal(size=24),
            0.8 * latent[:, 0] + 0.3 * latent[:, 1],
            latent[:, 1] + 0.1 * rng.normal(size=24),
            0.4 * latent[:, 0] - 0.6 * latent[:, 1] + 0.1 * rng.normal(size=24),
        ]
    )
    return X, coords


def test_matches_independent_gwmodel_translation(data):
    X, coords = data
    eval_coords = coords[[1, 7, 15, 22]]
    model = GWPCA(
        n_components=2,
        kernel="gaussian",
        bandwidth=2.5,
        adaptive=False,
        scaling=True,
    ).fit(X, coords, eval_coords=eval_coords)
    reference = _reference_gwpca(
        X,
        coords,
        eval_coords,
        bandwidth=2.5,
        kernel="gaussian",
        adaptive=False,
        n_components=2,
        scaling=True,
    )
    np.testing.assert_allclose(model.loadings_, reference[0], atol=1e-12)
    np.testing.assert_allclose(model.var_, reference[1], atol=1e-12)
    np.testing.assert_allclose(model.local_means_, reference[2], atol=1e-12)
    np.testing.assert_allclose(model.local_pv_, reference[3], atol=1e-12)


def test_adaptive_bisquare_matches_reference(data):
    X, coords = data
    model = GWPCA(
        n_components=2,
        kernel="bisquare",
        bandwidth=12,
        adaptive=True,
        scaling=False,
    ).fit(X, coords)
    reference = _reference_gwpca(
        X,
        coords,
        coords,
        bandwidth=12,
        kernel="bisquare",
        adaptive=True,
        n_components=2,
        scaling=False,
    )
    np.testing.assert_allclose(model.loadings_, reference[0], atol=1e-12)
    np.testing.assert_allclose(model.local_pv_, reference[3], atol=1e-12)


def test_global_boxcar_degenerates_to_global_pca(data):
    X, coords = data
    model = GWPCA(
        n_components=3,
        kernel="boxcar",
        bandwidth=1e6,
        adaptive=False,
        scaling=True,
    ).fit(X, coords)
    processed = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)
    global_pca = PCA(n_components=3).fit(processed)
    expected = _canonicalize(global_pca.components_.T)
    for loading in model.loadings_:
        np.testing.assert_allclose(loading, expected, atol=1e-12)
    expected_ratio = global_pca.explained_variance_ratio_[:3] * 100
    np.testing.assert_allclose(
        model.local_pv_, np.repeat(expected_ratio[None, :], len(X), axis=0), atol=1e-12
    )


def test_dataframe_output_and_focal_scores(data):
    X, coords = data
    frame = pd.DataFrame(X, columns=["a", "b", "c", "d"])
    model = GWPCA(
        n_components=2,
        bandwidth=14,
        adaptive=True,
        compute_scores=True,
    ).fit(frame, pd.DataFrame(coords, columns=["x", "y"]))
    assert model.feature_names_ == ["a", "b", "c", "d"]
    assert model.focal_scores_.shape == (len(X), 2)
    assert len(model.scores_) == len(X)
    result = model.to_frame()
    assert list(result.columns) == [
        "Comp.1_PV",
        "Comp.2_PV",
        "local_CP",
        "win_var_PC1",
    ]
    assert set(result["win_var_PC1"]).issubset(frame.columns)


def test_transform_uses_fitted_local_centres_and_no_nearest_substitution(data):
    X, coords = data
    eval_coords = coords[[2, 8, 17]]
    model = GWPCA(
        n_components=2,
        bandwidth=2.0,
        adaptive=False,
        kernel="gaussian",
    ).fit(X, coords, eval_coords=eval_coords)
    rows = X[[2, 8, 17]]
    scores = model.transform(rows, eval_coords)
    processed = (rows - model.global_mean_) / model.global_scale_
    expected = np.einsum("ij,ijk->ik", processed - model.local_means_, model.loadings_)
    np.testing.assert_allclose(scores, expected, atol=1e-12)
    with pytest.raises(ValueError, match="match exactly"):
        model.transform(rows[:1], eval_coords[:1] + 0.01)


def test_cv_contributions_match_direct_formula(data):
    X, coords = data
    model = GWPCA(
        n_components=2,
        kernel="gaussian",
        bandwidth=2.2,
        adaptive=False,
    ).fit(X, coords, compute_cv=True)
    processed = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)
    expected = []
    distances = np.sqrt(((coords[:, None] - coords[None, :]) ** 2).sum(axis=2))
    for i in range(len(X)):
        weights = _kernel(distances[i], 2.2, "gaussian")
        weights[i] = 0
        use = weights > 0
        w = weights[use]
        local = processed[use]
        mean = (local * w[:, None]).sum(axis=0) / w.sum()
        _, _, vt = np.linalg.svd(
            (local - mean) * np.sqrt(w)[:, None], full_matrices=False
        )
        v = vt.T[:, :2]
        residual = processed[i] - processed[i] @ (v @ v.T)
        expected.append(np.sum(residual) ** 2)
    np.testing.assert_allclose(model.cv_scores_, expected, atol=1e-12)


def test_adaptive_bandwidth_selection_matches_gwmodel_gold(data):
    X, coords = data
    model = GWPCA(n_components=1, kernel="boxcar", bandwidth="cv", adaptive=True)
    selected = model.select_bandwidth(X[:12], coords[:12])
    processed = (X[:12] - X[:12].mean(axis=0)) / X[:12].std(axis=0, ddof=1)

    def score(candidate):
        return model._cv_contributions(processed, coords[:12], int(candidate)).sum()

    expected = model._gwmodel_golden_search(score, 2.0, 12.0, adaptive=True)
    assert selected == expected


def test_dublin_public_bandwidth_benchmark_is_131():
    geopandas = pytest.importorskip("geopandas")
    from pathlib import Path

    data_path = (
        Path(__file__).parents[1]
        / "src"
        / "pygwrx"
        / "data"
        / "DubVoter"
        / "Dub.voter.shp"
    )
    voter = geopandas.read_file(data_path)
    variables = [
        "DiffAdd",
        "LARent",
        "SC1",
        "Unempl",
        "LowEduc",
        "Age18_24",
        "Age25_44",
        "Age45_64",
    ]
    model = GWPCA(
        n_components=3,
        kernel="bisquare",
        bandwidth="cv",
        adaptive=True,
        scaling=True,
    )
    selected = model.select_bandwidth(voter[variables], voter[["X", "Y"]])
    assert selected == 131


def test_known_spatial_rotation_changes_winning_variable():
    rng = np.random.default_rng(7)
    x_coord = np.linspace(-3, 3, 80)
    coords = np.column_stack([x_coord, np.zeros_like(x_coord)])
    z1 = rng.normal(size=80)
    z2 = rng.normal(size=80)
    left = x_coord < 0
    a = np.where(left, 3 * z1 + 0.1 * z2, z2)
    b = np.where(left, z1, 3 * z2 + 0.1 * z1)
    c = rng.normal(scale=0.2, size=80)
    X = np.column_stack([a, b, c])
    model = GWPCA(
        n_components=1,
        kernel="gaussian",
        bandwidth=0.8,
        adaptive=False,
        scaling=False,
    ).fit(X, coords)
    winners = model.get_winning_variable(0)
    assert np.mean(winners[left] == 0) > 0.7
    assert np.mean(winners[~left] == 1) > 0.7


def test_failed_refit_clears_state(data):
    X, coords = data
    model = GWPCA(bandwidth=12, adaptive=True).fit(X, coords)
    bad = X.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        model.fit(bad, coords)
    assert model._is_fitted is False
    assert model.loadings_ is None


def test_constant_variable_rejected_when_scaling(data):
    X, coords = data
    constant = np.column_stack([X[:, :2], np.ones(len(X))])
    with pytest.raises(ValueError, match="positive finite"):
        GWPCA(n_components=2, bandwidth=12, adaptive=True).fit(constant, coords)
    model = GWPCA(
        n_components=2,
        bandwidth=12,
        adaptive=True,
        scaling=False,
    ).fit(constant, coords)
    assert np.all(np.isfinite(model.local_pv_))


def test_input_and_bandwidth_boundaries(data):
    X, coords = data
    with pytest.raises(ValueError, match="at least two variables"):
        GWPCA(n_components=1, bandwidth=10, adaptive=True).fit(X[:, :1], coords)
    with pytest.raises(ValueError, match="cannot exceed"):
        GWPCA(n_components=5, bandwidth=10, adaptive=True).fit(X, coords)
    with pytest.raises(TypeError, match="integer neighbour"):
        GWPCA(bandwidth=4.5, adaptive=True)
    with pytest.raises(ValueError, match="cannot exceed"):
        GWPCA(bandwidth=30, adaptive=True).fit(X, coords)
    with pytest.raises(ValueError, match="same number of rows"):
        GWPCA(bandwidth=10, adaptive=True).fit(X, coords[:-1])
