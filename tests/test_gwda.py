# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Numerical and engineering tests for standard GWDA."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)

from pygwrx.models.gwda import GWDA


def make_three_class_data(seed: int = 42, n_per_class: int = 18):
    rng = np.random.default_rng(seed)
    labels = np.repeat(np.array(["A", "B", "C"], dtype=object), n_per_class)
    centers = {"A": (-2.0, 0.0), "B": (2.0, 0.0), "C": (0.0, 2.7)}
    X = np.vstack(
        [
            rng.normal(centers[label], (0.55, 0.6), size=(n_per_class, 2))
            for label in ["A", "B", "C"]
        ]
    )
    # Interleave classes spatially so every broad local window contains all classes.
    coords = np.column_stack(
        [
            np.linspace(0.0, 12.0, labels.size),
            np.sin(np.linspace(0.0, 5.0, labels.size)),
        ]
    )
    order = np.arange(labels.size).reshape(3, n_per_class).T.reshape(-1)
    return X[order], labels[order], coords[order]


def independent_weighted_covariance(X: np.ndarray, weights: np.ndarray) -> np.ndarray:
    normalized = weights / weights.sum()
    mean = normalized @ X
    centered = X - mean
    return (centered.T * normalized) @ centered / (1.0 - np.sum(normalized**2))


def test_local_statistics_match_independent_gwmodel_translation():
    X, y, coords = make_three_class_data(n_per_class=12)
    model = GWDA(kernel="gaussian", bandwidth=100.0, adaptive=False)
    model.fit(X, y, coords, validate=False)

    distances = np.linalg.norm(coords - coords[0], axis=1)
    weights = np.exp(-0.5 * (distances / 100.0) ** 2)
    indices = np.flatnonzero(y == "A")
    expected_mean = np.average(X[indices], axis=0, weights=weights[indices])
    expected_cov = independent_weighted_covariance(X[indices], weights[indices])

    np.testing.assert_allclose(model.class_means_["A"][0], expected_mean, atol=1e-12)
    np.testing.assert_allclose(
        model.class_covariances_["A"][0], expected_cov, atol=1e-12
    )
    expected_prior = weights[indices].sum() / weights.sum()
    assert model.class_priors_["A"][0] == pytest.approx(expected_prior, abs=1e-12)


def test_global_wlda_degenerates_to_sklearn_lda_predictions():
    X, y, coords = make_three_class_data(n_per_class=30)
    train = np.arange(0, X.shape[0], 2)
    test = np.arange(1, X.shape[0], 2)
    model = GWDA(
        kernel="boxcar",
        bandwidth=1e6,
        adaptive=False,
        local_mean=False,
        local_cov=False,
        local_prior=False,
    ).fit(X[train], y[train], coords[train], validate=False)
    expected = LinearDiscriminantAnalysis().fit(X[train], y[train]).predict(X[test])
    np.testing.assert_array_equal(model.predict(X[test], coords[test]), expected)


def test_global_wqda_degenerates_to_sklearn_qda_predictions():
    X, y, coords = make_three_class_data(n_per_class=35)
    train = np.arange(0, X.shape[0], 2)
    test = np.arange(1, X.shape[0], 2)
    model = GWDA(
        kernel="boxcar",
        bandwidth=1e6,
        adaptive=False,
        quadratic=True,
        local_mean=False,
        local_cov=False,
        local_prior=False,
    ).fit(X[train], y[train], coords[train], validate=False)
    expected = QuadraticDiscriminantAnalysis().fit(X[train], y[train]).predict(X[test])
    agreement = np.mean(model.predict(X[test], coords[test]) == expected)
    assert agreement >= 0.95


def test_probabilities_entropy_and_confusion_are_well_formed():
    X, y, coords = make_three_class_data(n_per_class=18)
    model = GWDA(kernel="gaussian", bandwidth=8.0, adaptive=False).fit(X, y, coords)
    np.testing.assert_allclose(model.probabilities_.sum(axis=1), 1.0, atol=1e-12)
    assert np.all((model.entropy_ >= 0.0) & (model.entropy_ <= 1.0 + 1e-12))
    assert model.confusion_matrix_.shape == (4, 4)
    assert model.confusion_matrix_[-1, -1] == X.shape[0]
    assert model.correct_ratio_ == pytest.approx(np.mean(model.predictions_ == y))


def test_adaptive_bandwidth_is_a_neighbour_count():
    X, y, coords = make_three_class_data(n_per_class=16)
    model = GWDA(kernel="boxcar", bandwidth=X.shape[0], adaptive=True).fit(
        X, y, coords, validate=False
    )
    assert model.bandwidth_ == X.shape[0]
    expected = np.array([np.mean(y == label) for label in model.classes_])
    observed = np.array([model.class_priors_[label][0] for label in model.classes_])
    np.testing.assert_allclose(observed, expected, atol=1e-12)


def test_bandwidth_selection_returns_best_tested_accuracy():
    X, y, coords = make_three_class_data(n_per_class=10)
    model = GWDA(kernel="gaussian", bandwidth="cv", adaptive=True, regularization=1e-8)
    selected = model.select_bandwidth(X, y, coords, bounds=(12, 20))
    scores = dict(model.bandwidth_scores_)
    assert selected in scores
    assert scores[selected] == max(scores.values())
    assert selected == min(
        key for key, value in scores.items() if value == max(scores.values())
    )


def test_predict_does_not_overwrite_fitted_validation_state():
    X, y, coords = make_three_class_data(n_per_class=16)
    model = GWDA(kernel="gaussian", bandwidth=7.0, adaptive=False).fit(X, y, coords)
    stored_predictions = model.predictions_.copy()
    stored_probabilities = model.probabilities_.copy()
    stored_accuracy = model.correct_ratio_
    result = model.predict(X[:4], coords[:4] + 0.05)
    assert result.shape == (4,)
    np.testing.assert_array_equal(model.predictions_, stored_predictions)
    np.testing.assert_allclose(model.probabilities_, stored_probabilities)
    assert model.correct_ratio_ == stored_accuracy


def test_dataframe_input_preserves_feature_names_and_label_type():
    X, y, coords = make_three_class_data(n_per_class=15)
    frame = pd.DataFrame(X, columns=["income", "education"])
    coord_frame = pd.DataFrame(coords, columns=["x", "y"])
    model = GWDA(kernel="gaussian", bandwidth=8.0, adaptive=False).fit(
        frame, pd.Series(y), coord_frame
    )
    assert model.feature_names_in_ == ["income", "education"]
    predicted = model.predict(frame.iloc[:3], coord_frame.iloc[:3])
    assert predicted.dtype == y.dtype
    with pytest.raises(ValueError, match="columns"):
        model.predict(frame[["education", "income"]].iloc[:3], coord_frame.iloc[:3])


def test_prior_validation_and_fixed_priors():
    X, y, coords = make_three_class_data(n_per_class=15)
    with pytest.raises(ValueError, match="sum to one"):
        GWDA(bandwidth=8.0, adaptive=False, prior=[0.2, 0.2, 0.2]).fit(X, y, coords)
    model = GWDA(
        kernel="gaussian",
        bandwidth=8.0,
        adaptive=False,
        prior=[0.2, 0.3, 0.5],
    ).fit(X, y, coords, validate=False)
    for index, label in enumerate(model.classes_):
        np.testing.assert_allclose(model.class_priors_[label], [0.2, 0.3, 0.5][index])


def test_singular_covariance_requires_explicit_regularization():
    X, y, coords = make_three_class_data(n_per_class=14)
    X_singular = np.column_stack([X[:, 0], X[:, 0]])
    with pytest.raises(np.linalg.LinAlgError, match="regularization"):
        GWDA(kernel="gaussian", bandwidth=8.0, adaptive=False).fit(
            X_singular, y, coords, validate=False
        )
    fitted = GWDA(
        kernel="gaussian",
        bandwidth=8.0,
        adaptive=False,
        regularization=1e-6,
    ).fit(X_singular, y, coords, validate=False)
    assert fitted._is_fitted


def test_failed_refit_clears_previous_state():
    X, y, coords = make_three_class_data(n_per_class=15)
    model = GWDA(kernel="gaussian", bandwidth=8.0, adaptive=False).fit(X, y, coords)
    assert model._is_fitted
    with pytest.raises(ValueError):
        model.fit(X[:5], y[:5], coords[:5])
    assert not model._is_fitted
    assert model.predictions_ is None
    assert model.class_means_ is None
    assert model.class_covs_ is None


def test_known_spatial_regime_is_classified_with_local_model():
    rng = np.random.default_rng(7)
    n = 80
    x_coord = np.linspace(0.0, 10.0, n)
    coords = np.column_stack([x_coord, np.zeros(n)])
    signal = rng.normal(size=n)
    noise = rng.normal(scale=0.25, size=n)
    # The class relationship reverses across space, which a global linear rule cannot represent.
    y = np.where(
        np.where(x_coord < 5.0, signal, -signal) + noise > 0, "positive", "negative"
    )
    X = np.column_stack([signal, rng.normal(scale=0.4, size=n)])
    local = GWDA(
        kernel="gaussian",
        bandwidth=1.6,
        adaptive=False,
        regularization=1e-5,
    ).fit(X, y, coords)
    global_model = GWDA(
        kernel="boxcar",
        bandwidth=1e6,
        adaptive=False,
        local_mean=False,
        local_cov=False,
        local_prior=False,
        regularization=1e-5,
    ).fit(X, y, coords)
    assert local.correct_ratio_ >= global_model.correct_ratio_ + 0.15


def test_public_iris_case_matches_global_lda_degenerate_path():
    from sklearn.datasets import load_iris

    iris = load_iris()
    X = iris.data
    y = iris.target_names[iris.target]
    coords = X[:, :2]
    train = np.arange(X.shape[0]) % 3 != 0
    test = ~train
    model = GWDA(
        kernel="boxcar",
        bandwidth=1e6,
        adaptive=False,
        local_mean=False,
        local_cov=False,
        local_prior=False,
    ).fit(X[train], y[train], coords[train], validate=False)
    expected = LinearDiscriminantAnalysis().fit(X[train], y[train]).predict(X[test])
    predicted = model.predict(X[test], coords[test])
    np.testing.assert_array_equal(predicted, expected)
    assert np.mean(predicted == y[test]) >= 0.94


def test_full_local_wlda_cost_matches_independent_reference():
    X, y, coords = make_three_class_data(n_per_class=14)
    X_eval = X[[7]] + np.array([[0.08, -0.03]])
    coords_eval = coords[[7]] + np.array([[0.04, 0.02]])
    bandwidth = 5.5
    model = GWDA(kernel="gaussian", bandwidth=bandwidth, adaptive=False).fit(
        X, y, coords, X_pred=X_eval, coords_pred=coords_eval, validate=False
    )

    distances = np.linalg.norm(coords - coords_eval[0], axis=1)
    weights = np.exp(-0.5 * (distances / bandwidth) ** 2)
    means = []
    covariances = []
    priors = []
    counts = []
    for label in model.classes_:
        indices = np.flatnonzero(y == label)
        means.append(np.average(X[indices], axis=0, weights=weights[indices]))
        covariances.append(
            independent_weighted_covariance(X[indices], weights[indices])
        )
        priors.append(weights[indices].sum() / weights.sum())
        counts.append(indices.size)
    pooled = sum(
        count * covariance for count, covariance in zip(counts, covariances)
    ) / sum(counts)
    inverse = np.linalg.inv(pooled)
    _, logdet = np.linalg.slogdet(pooled)
    expected = []
    for mean, prior in zip(means, priors):
        difference = X_eval[0] - mean
        expected.append(
            0.5 * logdet + 0.5 * difference @ inverse @ difference - np.log(prior)
        )
    np.testing.assert_allclose(model.discriminant_scores_[0], expected, atol=1e-12)
