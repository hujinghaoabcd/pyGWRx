# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Smoke tests for every model-specific plotting family.

Author:
    Jinghao Hu
"""

from __future__ import annotations

import warnings
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from pygwrx.models import (
    GRGWR,
    GTWR,
    GWDA,
    GWGLM,
    GWPCA,
    GWSS,
    LGGWR,
    MGTWR,
    RGWR,
    SGTWR,
    SGWR,
    STWR,
    BootstrapGWR,
    GWLasso,
    MixedGWR,
    ScalableGWR,
)
from pygwrx.plotting import (
    compare_model_diagnostics,
    plot_bootstrap_bandwidths,
    plot_bootstrap_pvalues,
    plot_grgwr_coefficient_surface,
    plot_grgwr_convergence,
    plot_grgwr_regime_sizes,
    plot_grgwr_regimes,
    plot_gwda_classification,
    plot_gwda_confusion_matrix,
    plot_gwglm_residuals,
    plot_gwlasso_active_map,
    plot_gwlasso_alpha,
    plot_gwlasso_selection_frequency,
    plot_gwpca_explained_variance,
    plot_gwpca_loading,
    plot_gwss_statistic,
    plot_lggwr_latent_geometry,
    plot_lggwr_metric_matrix,
    plot_lggwr_neighbourhood_comparison,
    plot_lggwr_training,
    plot_mgtwr_scales,
    plot_mixed_gwr_coefficients,
    plot_rgwr_convergence,
    plot_rgwr_weights,
    plot_scalable_gwr_kernel,
    plot_selection_history,
    plot_temporal_bandwidths,
    plot_temporal_coefficient_slices,
    plot_temporal_residuals,
    plot_temporal_trajectory,
    plot_weight_decomposition,
    plot_weight_profiles,
)


def _assert_plot(result):
    fig, axes = result
    assert isinstance(fig, Figure)
    array = np.asarray(axes, dtype=object).reshape(-1)
    assert all(isinstance(axis, Axes) for axis in array)
    fig.canvas.draw()
    plt.close(fig)


@pytest.fixture(scope="module")
def fitted_models():
    rng = np.random.default_rng(4)
    n = 36
    coords = rng.uniform(0.0, 5.0, size=(n, 2))
    X = rng.normal(size=(n, 3))
    y = 1.0 + 1.2 * X[:, 0] - 0.7 * X[:, 1] + 0.2 * coords[:, 0]
    y += rng.normal(0.0, 0.15, size=n)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        rgwr = RGWR(bandwidth=20, adaptive=True, max_iter=3).fit(X, y, coords)
    models = {
        "rgwr": rgwr,
        "gwglm": GWGLM(family="gaussian", bandwidth=20, adaptive=True).fit(
            X, y, coords
        ),
        "gwlasso": GWLasso(bandwidth=20, adaptive=True, alpha=0.03, max_iter=500).fit(
            X, y, coords
        ),
        "mixed": MixedGWR(
            bandwidth=20,
            adaptive=True,
            local_vars=[0, 1],
            global_vars=[2],
        ).fit(X, y, coords),
        "bootstrap": BootstrapGWR(
            bandwidth=20,
            adaptive=True,
            n_bootstrap=5,
            reselect_bandwidth=False,
            store_local_bootstrap=True,
            random_state=1,
        ).fit(X, y, coords),
        "gwss": GWSS(bandwidth=20, adaptive=True, quantile=True).fit(
            pd.DataFrame(X, columns=["a", "b", "c"]), coords
        ),
        "gwpca": GWPCA(n_components=2, bandwidth=20, adaptive=True).fit(
            pd.DataFrame(X, columns=["a", "b", "c"]), coords
        ),
        "gwda": GWDA(bandwidth=20, adaptive=True).fit(
            X, (X[:, 0] > 0.0).astype(int), coords
        ),
        "sgwr": SGWR(
            bandwidth=20,
            adaptive=True,
            alpha=0.5,
            store_weights=True,
        ).fit(X, y, coords),
        "gtwr": GTWR(bandwidth=20, adaptive=True, lambda_st=0.3).fit(
            X, y, coords, np.repeat(np.arange(4), 9)
        ),
        "mgtwr": MGTWR(
            bandwidths=[20, 20, 20],
            taus=[1.0, 1.0, 1.0],
            adaptive=True,
            calculate_inference=False,
        ).fit(X[:, :2], y, coords, np.repeat(np.arange(4), 9)),
        "scalable": ScalableGWR(
            bandwidth=20,
            optimize_bandwidth=False,
        ).fit(X[:, :2], y, coords),
        "sgtwr": SGTWR(
            spatial_bandwidth=20,
            temporal_bandwidth=2.0,
            adaptive=True,
            alpha=0.5,
            store_weights=True,
        ).fit(X, y, coords, np.repeat(np.arange(4), 9)),
        "lggwr": LGGWR(
            latent_dim=2,
            bandwidth=2.5,
            adaptive=False,
            max_iter=4,
            select_bandwidth=False,
            random_state=1,
        ).fit(X[:, :2], y, coords, attributes=X[:, 2:]),
    }

    models["stwr"] = STWR(
        spatial_bandwidth=8,
        adaptive=True,
        alpha=0.3,
        theta=0.0,
        tick_nums=2,
        store_weights=True,
    ).fit(
        [X[:12, :2], X[12:24, :2], X[24:, :2]],
        [y[:12], y[12:24], y[24:]],
        [coords[:12], coords[12:24], coords[24:]],
        [0.0, 1.0, 1.0],
    )

    n_regime = 80
    regime_coords = rng.uniform(0.0, 10.0, size=(n_regime, 2))
    regime_X = rng.normal(size=(n_regime, 2))
    truth = regime_coords[:, 0] > 5.0
    regime_y = np.where(truth, -2.0, 2.0) * regime_X[:, 0]
    regime_y += regime_X[:, 1] + rng.normal(0.0, 0.1, n_regime)
    models["grgwr"] = GRGWR(
        n_regimes=2,
        bandwidth=20,
        max_iter=3,
        random_state=1,
    ).fit(regime_X, regime_y, regime_coords)
    return models


def test_robust_glm_regularized_and_bootstrap_plots(fitted_models):
    m = fitted_models
    for result in (
        plot_rgwr_weights(m["rgwr"]),
        plot_rgwr_convergence(m["rgwr"]),
        plot_gwglm_residuals(m["gwglm"]),
        plot_gwlasso_selection_frequency(m["gwlasso"]),
        plot_gwlasso_active_map(m["gwlasso"], 0),
        plot_gwlasso_alpha(m["gwlasso"]),
        plot_mixed_gwr_coefficients(m["mixed"]),
        plot_bootstrap_pvalues(m["bootstrap"], 0),
        plot_bootstrap_bandwidths(m["bootstrap"]),
        plot_scalable_gwr_kernel(m["scalable"]),
    ):
        _assert_plot(result)


def test_multivariate_and_classification_plots(fitted_models):
    m = fitted_models
    for result in (
        plot_gwss_statistic(m["gwss"], "mean", "a"),
        plot_gwss_statistic(m["gwss"], "correlation", "a", second_feature="b"),
        plot_gwpca_explained_variance(m["gwpca"], 0),
        plot_gwpca_loading(m["gwpca"], "a", 0),
        plot_gwda_classification(m["gwda"]),
        plot_gwda_classification(m["gwda"], confidence=True),
        plot_gwda_confusion_matrix(m["gwda"]),
    ):
        _assert_plot(result)


def test_spatiotemporal_and_weight_plots(fitted_models):
    m = fitted_models
    for result in (
        plot_temporal_coefficient_slices(m["gtwr"], 0),
        plot_temporal_trajectory(m["gtwr"], 0),
        plot_temporal_residuals(m["gtwr"]),
        plot_temporal_bandwidths(m["sgtwr"]),
        plot_mgtwr_scales(m["mgtwr"]),
        plot_weight_decomposition(m["sgwr"], 0),
        plot_weight_profiles(m["sgwr"], 0),
        plot_weight_decomposition(m["stwr"], 0),
        plot_weight_decomposition(m["sgtwr"], 0),
    ):
        _assert_plot(result)

    search_model = SimpleNamespace(
        _is_fitted=True,
        diagnostics_={"aicc": 1.0},
        selection_history_=[{"aicc": 3.0}, {"aicc": 1.0}, {"aicc": 2.0}],
    )
    _assert_plot(plot_selection_history(search_model))


def test_original_model_and_comparison_plots(fitted_models):
    m = fitted_models
    for result in (
        plot_lggwr_latent_geometry(m["lggwr"]),
        plot_lggwr_metric_matrix(m["lggwr"]),
        plot_lggwr_training(m["lggwr"]),
        plot_lggwr_neighbourhood_comparison(m["lggwr"], 0),
        plot_grgwr_regimes(m["grgwr"]),
        plot_grgwr_convergence(m["grgwr"]),
        plot_grgwr_regime_sizes(m["grgwr"]),
        plot_grgwr_coefficient_surface(m["grgwr"], 0),
        compare_model_diagnostics([m["rgwr"], m["gwglm"]]),
    ):
        _assert_plot(result)
