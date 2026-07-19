# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Small fitted-model factories used by plotting examples."""

from __future__ import annotations

import warnings
from types import SimpleNamespace

from _common import (
    classification_data,
    latent_regression,
    mixed_regression,
    regime_regression,
    spatial_regression,
    stwr_stages,
    temporal_regression,
)

from pygwrx import (
    GRGWR,
    GTWR,
    GWDA,
    GWGLM,
    GWPCA,
    GWR,
    GWSS,
    LCRGWR,
    LGGWR,
    MGTWR,
    MGWR,
    RGWR,
    SGTWR,
    SGWR,
    STWR,
    BootstrapGWR,
    GWLasso,
    MixedGWR,
    ScalableGWR,
)


def surface_models():
    X, y, coords = spatial_regression(n=36, p=2, seed=101)
    gwr = GWR(kernel="bisquare", bandwidth=20, adaptive=True).fit(X, y, coords)
    mgwr = MGWR(
        kernel="bisquare",
        bandwidths=[20, 22, 24],
        adaptive=True,
        max_iter=5,
        tol=1.0,
    ).fit(X, y, coords)
    lcr = LCRGWR(
        kernel="bisquare",
        bandwidth=20,
        adaptive=True,
        cn_thresh=10.0,
    ).fit(X, y, coords)
    return X, y, coords, gwr, mgwr, lcr


def regularized_models():
    X, y, coords = spatial_regression(n=36, p=3, seed=103)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rgwr = RGWR(bandwidth=20, adaptive=True, max_iter=3).fit(X, y, coords)
    gwglm = GWGLM(family="gaussian", bandwidth=20, adaptive=True).fit(X, y, coords)
    gwlasso = GWLasso(bandwidth=20, adaptive=True, alpha=0.04, max_iter=600).fit(
        X, y, coords
    )
    Xm, ym, cm = mixed_regression(n=48, seed=107)
    mixed = MixedGWR(
        bandwidth=22,
        adaptive=True,
        global_vars=["global_x"],
        local_vars=["local_x"],
    ).fit(Xm, ym, cm, compute_enp=False)
    bootstrap = BootstrapGWR(
        bandwidth=20,
        adaptive=True,
        n_bootstrap=5,
        reselect_bandwidth=False,
        store_local_bootstrap=True,
        random_state=1,
    ).fit(X.iloc[:, :2], y, coords)
    scalable = ScalableGWR(
        bandwidth=20,
        optimize_bandwidth=False,
    ).fit(X.iloc[:, :2], y, coords)
    return X, y, coords, rgwr, gwglm, gwlasso, mixed, bootstrap, scalable


def multivariate_models():
    X, _, coords = spatial_regression(n=36, p=3, seed=109)
    gwss = GWSS(bandwidth=20, adaptive=True, quantile=True).fit(X, coords)
    gwpca = GWPCA(n_components=2, bandwidth=20, adaptive=True).fit(X, coords)
    Xc, yc, cc = classification_data(n=44, seed=113)
    gwda = GWDA(bandwidth=22, adaptive=True).fit(Xc, yc, cc)
    return X, coords, gwss, gwpca, Xc, yc, cc, gwda


def temporal_models():
    X, y, coords, times = temporal_regression(n=36, p=2, seed=127)
    gtwr = GTWR(bandwidth=20, adaptive=True, lambda_st=0.3).fit(X, y, coords, times)
    mgtwr = MGTWR(
        bandwidths=[20, 20, 20],
        taus=[1.0, 1.0, 1.0],
        adaptive=True,
        calculate_inference=False,
    ).fit(X, y, coords, times)
    sgtwr = SGTWR(
        spatial_bandwidth=20,
        temporal_bandwidth=2.0,
        adaptive=True,
        alpha=0.5,
        store_weights=True,
    ).fit(X, y, coords, times)
    sgwr = SGWR(
        bandwidth=20,
        adaptive=True,
        alpha=0.5,
        store_weights=True,
    ).fit(X, y, coords)
    Xs, ys, cs, intervals = stwr_stages(n_per_stage=12, seed=131)
    stwr = STWR(
        spatial_bandwidth=8,
        adaptive=True,
        alpha=0.3,
        theta=0.0,
        tick_nums=2,
        store_weights=True,
    ).fit(Xs, ys, cs, intervals)
    search_model = SimpleNamespace(
        _is_fitted=True,
        diagnostics_={"aicc": 1.0},
        selection_history_=[{"aicc": 3.0}, {"aicc": 1.0}, {"aicc": 2.0}],
    )
    return X, y, coords, times, gtwr, mgtwr, sgtwr, sgwr, stwr, search_model


def original_models():
    X, y, coords, attrs = latent_regression(n=42, seed=137)
    lggwr = LGGWR(
        latent_dim=2,
        bandwidth=2.5,
        adaptive=False,
        max_iter=4,
        select_bandwidth=False,
        random_state=1,
    ).fit(X, y, coords, attrs)
    Xr, yr, cr, _ = regime_regression(n=52, seed=139)
    grgwr = GRGWR(
        n_regimes=2,
        bandwidth=18,
        max_iter=2,
        random_state=1,
    ).fit(Xr, yr, cr)
    return lggwr, grgwr
