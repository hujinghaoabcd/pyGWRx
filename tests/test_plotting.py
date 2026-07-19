# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Tests for the model-aware visualization layer.

Author:
    Jinghao Hu
"""

from __future__ import annotations

__author__ = "Jinghao Hu"
__license__ = "MIT"

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from pygwrx.models import GWR, LCRGWR, MGWR
from pygwrx.plotting import (
    compare_coefficient_surfaces,
    plot_bandwidth,
    plot_bandwidth_selection,
    plot_coefficient_map,
    plot_coefficient_surface,
    plot_coefficient_variability,
    plot_diagnostic_panel,
    plot_kernel_weights,
    plot_local_coefficients,
    plot_local_collinearity,
    plot_local_diagnostic_map,
    plot_local_diagnostics,
    plot_local_r2,
    plot_mgwr_bandwidths,
    plot_model_significance_map,
    plot_multiple_coefficients,
    plot_observed_vs_predicted,
    plot_qq,
    plot_residual_histogram,
    plot_residuals,
    plot_significance_map,
    plot_spatial_residuals,
)


def _data():
    rng = np.random.default_rng(2026)
    n = 32
    coords = rng.uniform(0.0, 10.0, size=(n, 2))
    X = rng.normal(size=(n, 2))
    y = 1.0 + (1.2 + 0.08 * coords[:, 0]) * X[:, 0] - 0.7 * X[:, 1]
    y += rng.normal(0.0, 0.18, size=n)
    return pd.DataFrame(X, columns=["income", "rurality"]), y, coords


@pytest.fixture(scope="module")
def fitted_models():
    X, y, coords = _data()
    gwr = GWR(kernel="bisquare", bandwidth=18, adaptive=True).fit(X, y, coords)
    mgwr = MGWR(
        kernel="bisquare",
        bandwidths=[18, 20, 22],
        init_bandwidth=20,
        adaptive=True,
        max_iter=5,
        tol=1.0,
    ).fit(X, y, coords)
    lcr = LCRGWR(
        kernel="bisquare",
        bandwidth=18,
        adaptive=True,
        cn_thresh=10.0,
    ).fit(X, y, coords)
    return X, y, coords, gwr, mgwr, lcr


def _assert_plot(result):
    fig, axes = result
    assert isinstance(fig, Figure)
    if isinstance(axes, np.ndarray):
        assert all(isinstance(axis, Axes) for axis in axes.flat)
    else:
        assert isinstance(axes, Axes)
    fig.canvas.draw()
    plt.close(fig)


def test_model_aware_coefficient_and_significance_maps(fitted_models):
    _, _, _, gwr, mgwr, _ = fitted_models
    _assert_plot(plot_coefficient_map(gwr, "income", theme="paper"))
    _assert_plot(
        plot_coefficient_map(
            mgwr,
            "income",
            significance="adjusted",
            theme="paper",
        )
    )
    _assert_plot(plot_significance_map(gwr, "income", correction="bh"))
    _assert_plot(plot_model_significance_map(gwr, "income", correction="raw"))


def test_model_aware_maps_support_geodataframe(fitted_models):
    gpd = pytest.importorskip("geopandas")
    from shapely.geometry import Point

    _, _, coords, gwr, _, _ = fitted_models
    geometry = gpd.GeoDataFrame(
        {"row": np.arange(coords.shape[0])},
        geometry=[Point(x, y) for x, y in coords],
        crs="EPSG:3857",
    )
    _assert_plot(plot_coefficient_map(gwr, "rurality", geometry=geometry))
    _assert_plot(
        plot_local_diagnostic_map(gwr, "standardized_residual", geometry=geometry)
    )


def test_local_diagnostic_maps(fitted_models):
    _, _, _, gwr, mgwr, _ = fitted_models
    for metric in (
        "local_r2",
        "residual",
        "standardized_residual",
        "influence",
        "cooks_distance",
    ):
        model = gwr if metric == "local_r2" else mgwr
        _assert_plot(plot_local_diagnostic_map(model, metric))


def test_collinearity_maps(fitted_models):
    _, _, _, gwr, _, lcr = fitted_models
    _assert_plot(plot_local_collinearity(gwr, "condition_number"))
    for metric in (
        "condition_number",
        "compensated_condition_number",
        "penalized_system_condition_number",
        "local_lambda",
    ):
        _assert_plot(plot_local_collinearity(lcr, metric))


def test_bandwidth_and_model_comparison_plots(fitted_models):
    _, _, _, gwr, mgwr, _ = fitted_models
    _assert_plot(plot_mgwr_bandwidths(mgwr))
    _assert_plot(plot_kernel_weights(gwr, focus=3))
    _assert_plot(compare_coefficient_surfaces([gwr, mgwr], "income"))
    _assert_plot(
        plot_bandwidth_selection(
            [10, 15, 20, 25],
            [14.0, 9.0, 7.5, 8.2],
            20,
            criterion="AICc",
        )
    )


def test_array_diagnostic_plots(fitted_models):
    _, y, coords, gwr, _, _ = fitted_models
    _assert_plot(plot_residuals(gwr.fitted_values_, gwr.residuals_))
    _assert_plot(plot_residual_histogram(gwr.residuals_))
    _assert_plot(plot_qq(gwr.residuals_))
    _assert_plot(plot_spatial_residuals(coords, gwr.residuals_))
    _assert_plot(plot_observed_vs_predicted(y, gwr.fitted_values_))
    _assert_plot(
        plot_coefficient_variability(gwr.coef_, feature_names=["income", "rurality"])
    )
    _assert_plot(plot_diagnostic_panel(y, gwr.fitted_values_, gwr.residuals_, coords))
    _assert_plot(plot_diagnostic_panel(gwr))
    _assert_plot(
        plot_local_diagnostics(
            coords,
            {
                "local_r2": gwr.local_r2_,
                "influence": gwr.influence_,
                "cooks_distance": gwr.cooks_distance_,
            },
        )
    )


def test_legacy_array_maps_are_implemented(fitted_models):
    _, _, coords, gwr, _, _ = fitted_models
    _assert_plot(
        plot_local_coefficients(
            coords,
            gwr.coef_,
            feature_idx=0,
            feature_name="income",
        )
    )
    _assert_plot(
        plot_coefficient_surface(
            coords,
            gwr.coef_,
            feature_idx=0,
            interpolation="nearest",
        )
    )
    p_values = np.full_like(gwr.coef_, 0.02)
    _assert_plot(
        plot_significance_map(
            coords,
            p_values,
            feature_idx=0,
            coefficients=gwr.coef_,
        )
    )
    _assert_plot(plot_local_r2(coords, gwr.local_r2_))
    _assert_plot(plot_bandwidth(coords, 2.0, sample_locations=coords[:3]))
    _assert_plot(
        plot_multiple_coefficients(
            coords,
            gwr.coef_,
            feature_names=["income", "rurality"],
            shared_scale=True,
        )
    )


def test_invalid_inputs_raise_clear_errors(fitted_models):
    _, _, coords, gwr, _, _ = fitted_models
    with pytest.raises(ValueError, match="Unknown feature"):
        plot_coefficient_map(gwr, "missing")
    with pytest.raises(ValueError, match="same length"):
        plot_residuals([1, 2], [1])
    with pytest.raises(ValueError, match="not fitted"):
        compare_coefficient_surfaces([gwr, GWR()], "income")
    with pytest.raises(ValueError, match="strictly between"):
        plot_model_significance_map(gwr, "income", alpha=1.0)
    with pytest.raises(ValueError, match="at least two columns"):
        plot_local_r2(np.ones((coords.shape[0], 1)), gwr.local_r2_)


def test_plotting_functions_do_not_call_show(monkeypatch, fitted_models):
    _, _, _, gwr, _, _ = fitted_models

    def fail_show(*args, **kwargs):
        raise AssertionError("plotting helpers must not call plt.show()")

    monkeypatch.setattr(plt, "show", fail_show)
    _assert_plot(plot_coefficient_map(gwr, "income"))
    _assert_plot(
        plot_diagnostic_panel(gwr.y_train_, gwr.fitted_values_, gwr.residuals_)
    )
