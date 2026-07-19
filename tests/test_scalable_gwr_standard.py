"""Published ScaGWR implementation tests."""

import inspect

import numpy as np
import pandas as pd
import pytest
from scipy.spatial import cKDTree

from pygwrx.models.scalable_gwr import ScalableGWR


def _data(seed=7, n=48):
    rng = np.random.default_rng(seed)
    coords = rng.uniform(-2.0, 2.0, size=(n, 2))
    X = rng.normal(size=(n, 2))
    beta0 = 1.2 + 0.3 * coords[:, 0]
    beta1 = 2.0 - 0.4 * coords[:, 1]
    beta2 = -0.8 + 0.2 * coords[:, 0]
    y = beta0 + beta1 * X[:, 0] + beta2 * X[:, 1] + rng.normal(0, 0.04, n)
    return X, y, coords


def _basis(distances, h0, kernel, degree):
    if kernel == "gaussian":
        base = np.exp(-np.square(distances / h0))
    else:
        base = np.exp(-distances / h0)
    result = np.ones((len(distances), degree + 1))
    numerator = 2.0 ** (degree / 2.0)
    for index in range(1, degree + 1):
        result[:, index] = base ** (numerator / (2.0**index))
    return result


def _basis_coefficients(scale, degree):
    powers = np.arange(1, degree + 2, dtype=float)
    values = scale**powers
    return values / values.sum()


def _explicit_coefficients(model, X, y, coords, eval_coords):
    design = np.column_stack((np.ones(len(X)), X))
    tree = cKDTree(coords)
    distances, indices = tree.query(eval_coords, k=model.bandwidth_)
    if np.ndim(distances) == 1:
        distances = distances[:, None]
        indices = indices[:, None]
    coefficients = _basis_coefficients(model.scale_, model.polynomial)
    output = []
    for d, idx in zip(distances, indices):
        local_weights = (
            _basis(d, model.base_bandwidth_, model.kernel, model.polynomial)
            @ coefficients
        )
        weights = np.full(len(X), model.penalty_)
        weights[idx] += local_weights
        system = design.T @ (weights[:, None] * design)
        rhs = design.T @ (weights * y)
        output.append(np.linalg.solve(system, rhs))
    return np.asarray(output)


def test_fixed_parameters_match_explicit_weighted_regressions():
    X, y, coords = _data()
    model = ScalableGWR(
        bandwidth=18,
        polynomial=4,
        kernel="gaussian",
        optimize_bandwidth=False,
        scale=1.7,
        penalty=0.06,
    ).fit(X, y, coords)
    expected = _explicit_coefficients(model, X, y, coords, coords)
    np.testing.assert_allclose(model.coefficients_, expected, rtol=1e-10, atol=1e-10)


def test_exponential_kernel_matches_explicit_reference():
    X, y, coords = _data(seed=8)
    model = ScalableGWR(
        bandwidth=16,
        polynomial=3,
        kernel="exponential",
        optimize_bandwidth=False,
        scale=0.8,
        penalty=0.03,
    ).fit(X, y, coords)
    expected = _explicit_coefficients(model, X, y, coords, coords)
    np.testing.assert_allclose(model.coefficients_, expected, rtol=1e-10, atol=1e-10)


def test_new_location_prediction_estimates_model_not_coefficient_interpolation():
    X, y, coords = _data(seed=9)
    model = ScalableGWR(
        bandwidth=15,
        optimize_bandwidth=False,
        scale=1.3,
        penalty=0.04,
    ).fit(X, y, coords)
    X0 = X[:5] + 0.1
    coords0 = coords[:5] + np.array([0.07, -0.04])
    result = model.predict_result(X0, coords0, return_standard_errors=True)
    expected_beta = _explicit_coefficients(model, X, y, coords, coords0)
    expected_prediction = np.sum(
        np.column_stack((np.ones(5), X0)) * expected_beta, axis=1
    )
    np.testing.assert_allclose(
        result.coefficients, expected_beta, rtol=1e-10, atol=1e-10
    )
    np.testing.assert_allclose(
        result.predictions, expected_prediction, rtol=1e-10, atol=1e-10
    )
    assert result.standard_errors.shape == expected_beta.shape
    assert np.all(np.isfinite(result.standard_errors))


def test_cv_optimization_is_finite_and_improves_objective():
    X, y, coords = _data(seed=10)
    model = ScalableGWR(
        bandwidth=17,
        criterion="cv",
        optimize_bandwidth=True,
        optimizer_maxiter=80,
    ).fit(X, y, coords)
    assert model.scale_ > 0
    assert model.penalty_ >= 0
    assert np.isfinite(model.cv_score_)
    assert model.optimization_result_ is not None
    assert np.isfinite(model.optimization_result_.fun)


def test_aicc_calibration_and_inference_are_finite():
    X, y, coords = _data(seed=11)
    model = ScalableGWR(
        bandwidth=18,
        criterion="aicc",
        optimize_bandwidth=True,
        optimizer_maxiter=60,
    ).fit(X, y, coords)
    assert np.isfinite(model.aicc_)
    assert np.isfinite(model.trace_S_)
    assert np.isfinite(model.trace_StS_)
    assert 0 < model.effective_n_params_ < len(y)
    assert model.standard_errors_.shape == model.coefficients_.shape
    assert np.all(np.isfinite(model.standard_errors_))
    assert np.all((model.p_values_ >= 0) & (model.p_values_ <= 1))


def test_large_penalty_shrinks_coefficients_toward_ols():
    X, y, coords = _data(seed=12)
    model = ScalableGWR(
        bandwidth=14,
        optimize_bandwidth=False,
        scale=1.0,
        penalty=1.0e7,
    ).fit(X, y, coords)
    design = np.column_stack((np.ones(len(X)), X))
    ols = np.linalg.lstsq(design, y, rcond=None)[0]
    np.testing.assert_allclose(
        model.coefficients_,
        np.tile(ols, (len(y), 1)),
        rtol=2e-5,
        atol=2e-5,
    )
    assert model.effective_n_params_ == pytest.approx(design.shape[1], rel=2e-5)
    assert model.trace_S_ == pytest.approx(design.shape[1], rel=2e-5)
    assert model.trace_StS_ == pytest.approx(design.shape[1], rel=2e-5)


def test_dataframe_metadata_and_prediction_column_order():
    X, y, coords = _data(seed=13)
    frame = pd.DataFrame(X, columns=["income", "density"])
    model = ScalableGWR(
        bandwidth=15,
        optimize_bandwidth=False,
        scale=1.0,
        penalty=0.05,
    ).fit(frame, pd.Series(y), pd.DataFrame(coords, columns=["x", "y"]))
    assert model.feature_names_in_ == ("income", "density")
    assert model.design_feature_names_ == ("Intercept", "income", "density")
    with pytest.raises(ValueError, match="columns"):
        model.predict(frame[["density", "income"]], coords)


def test_fit_failure_clears_previous_state():
    X, y, coords = _data(seed=14)
    model = ScalableGWR(
        bandwidth=15,
        optimize_bandwidth=False,
        scale=1.0,
        penalty=0.05,
    ).fit(X, y, coords)
    assert model._is_fitted
    bad = X.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        model.fit(bad, y, coords)
    assert not model._is_fitted
    assert model.coefficients_ is None


def test_prediction_does_not_mutate_training_state():
    X, y, coords = _data(seed=15)
    model = ScalableGWR(
        bandwidth=16,
        optimize_bandwidth=False,
        scale=1.1,
        penalty=0.05,
    ).fit(X, y, coords)
    coefficients = model.coefficients_.copy()
    fitted = model.fitted_values_.copy()
    _ = model.predict(X[:3], coords[:3] + 0.01)
    np.testing.assert_array_equal(model.coefficients_, coefficients)
    np.testing.assert_array_equal(model.fitted_values_, fitted)


def test_sampled_cv_keeps_all_observations_available():
    X, y, coords = _data(seed=16, n=70)
    model = ScalableGWR(
        bandwidth=20,
        sample_size=25,
        random_state=2,
        optimize_bandwidth=True,
        optimizer_maxiter=50,
    ).fit(X, y, coords)
    assert model.coefficients_.shape == (70, 3)
    assert model.X_train_.shape[0] == 70
    assert np.isfinite(model.cv_score_)


def test_no_full_pairwise_distance_matrix_is_constructed(monkeypatch):
    X, y, coords = _data(seed=17)
    import scipy.spatial.distance as distance

    def forbidden(*args, **kwargs):
        raise AssertionError("full pairwise distance matrix must not be used")

    monkeypatch.setattr(distance, "cdist", forbidden)
    model = ScalableGWR(
        bandwidth=15,
        optimize_bandwidth=False,
        scale=1.0,
        penalty=0.05,
    ).fit(X, y, coords)
    assert model.coefficients_.shape == (len(y), 3)


def test_parameter_and_kernel_validation():
    assert "adaptive" not in inspect.signature(ScalableGWR).parameters
    with pytest.raises(ValueError, match="continuous"):
        ScalableGWR(kernel="bisquare")
    with pytest.raises(ValueError, match="polynomial"):
        ScalableGWR(polynomial=0)
    with pytest.raises(ValueError, match="neighbour"):
        ScalableGWR(bandwidth=1)


def test_constant_predictor_is_rejected_explicitly():
    X, y, coords = _data(seed=18)
    X[:, 1] = 1.0
    model = ScalableGWR(
        bandwidth=15,
        optimize_bandwidth=False,
        scale=1.0,
        penalty=0.05,
    )
    with pytest.raises(ValueError, match="constant"):
        model.fit(X, y, coords)


def test_summary_returns_diagnostics_copy():
    X, y, coords = _data(seed=19)
    model = ScalableGWR(
        bandwidth=16,
        optimize_bandwidth=False,
        scale=1.0,
        penalty=0.05,
    ).fit(X, y, coords)
    summary = model.summary()
    assert isinstance(summary, str)
    assert "n_neighbors" in summary
    assert "aicc" in summary
    assert model.diagnostics_["rss"] >= 0


def test_training_frame_contains_coefficients_and_inference():
    X, y, coords = _data(seed=19)
    model = ScalableGWR(
        bandwidth=15,
        optimize_bandwidth=False,
        scale=1.0,
        penalty=0.05,
    ).fit(X, y, coords)
    frame = model.to_frame()
    assert len(frame) == len(y)
    assert {
        "observed",
        "fitted",
        "residual",
        "coef_Intercept",
        "se_Intercept",
    }.issubset(frame.columns)


def test_precompressed_cv_matches_explicit_leave_one_out_objective():
    X, y, coords = _data(seed=20, n=42)
    model = ScalableGWR(
        bandwidth=14,
        optimize_bandwidth=False,
        scale=1.4,
        penalty=0.07,
    ).fit(X, y, coords)
    design = np.column_stack((np.ones(len(X)), X))
    tree = cKDTree(coords)
    distances, indices = tree.query(coords, k=model.bandwidth_ + 2)
    cv_rss = 0.0
    basis_coef = _basis_coefficients(model.scale_, model.polynomial)
    for i in range(len(y)):
        keep = indices[i] != i
        idx = indices[i, keep][: model.bandwidth_]
        d = distances[i, keep][: model.bandwidth_]
        local = (
            _basis(d, model.base_bandwidth_, model.kernel, model.polynomial)
            @ basis_coef
        )
        weights = np.full(len(y), model.penalty_)
        weights[idx] += local
        beta = np.linalg.solve(
            design.T @ (weights[:, None] * design),
            design.T @ (weights * y),
        )
        cv_rss += (y[i] - design[i] @ beta) ** 2
    np.testing.assert_allclose(
        model.cv_score_**2 * len(y),
        cv_rss,
        rtol=1e-10,
        atol=1e-10,
    )
