# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Tests for the unified diagnostic extraction layer.

Author:
    Jinghao Hu
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pygwrx.diagnostics import (
    diagnostics_frame,
    focus_weight_components,
    influence_thresholds,
    local_diagnostic_frame,
    model_diagnostic_summary,
    parameter_significance,
    parameter_trajectory,
    temporal_groups,
    weight_components,
)
from pygwrx.models import GTWR, GWR, MGTWR, SGWR, ScalableGWR


def _data(seed: int = 31, n: int = 36):
    rng = np.random.default_rng(seed)
    coords = rng.uniform(0.0, 5.0, size=(n, 2))
    X = pd.DataFrame(rng.normal(size=(n, 2)), columns=["income", "rurality"])
    y = 1.0 + 1.4 * X["income"].to_numpy() - 0.6 * X["rurality"].to_numpy()
    y += 0.15 * coords[:, 0] + rng.normal(0.0, 0.12, size=n)
    return X, y, coords


def test_global_and_local_diagnostic_views_are_consistent():
    X, y, coords = _data()
    model = GWR(kernel="bisquare", bandwidth=20, adaptive=True).fit(X, y, coords)
    summary = model_diagnostic_summary(model)
    assert summary.model_name == "GWR"
    assert summary.n_samples == len(y)
    assert {"r2", "rmse", "aicc"}.issubset(summary.metrics)

    frame = local_diagnostic_frame(model)
    assert frame.shape[0] == len(y)
    assert {"observed", "fitted", "residual", "influence"}.issubset(frame)
    thresholds = influence_thresholds(model)
    assert thresholds.leverage is not None
    assert thresholds.cooks_distance == 4.0 / len(y)


def test_model_comparison_and_parameter_significance():
    X, y, coords = _data()
    first = GWR(kernel="bisquare", bandwidth=18, adaptive=True).fit(X, y, coords)
    second = GWR(kernel="gaussian", bandwidth=22, adaptive=True).fit(X, y, coords)
    frame = diagnostics_frame([first, second], labels=["bisquare", "gaussian"])
    assert list(frame.index) == ["bisquare", "gaussian"]
    assert np.all(np.isfinite(frame["rmse"]))

    inference = parameter_significance(first, "income", correction="bh")
    assert inference.shape[0] == len(y)
    assert set(inference["category"].unique()).issubset({-1, 0, 1})


def test_weight_and_temporal_views_use_fitted_model_state():
    X, y, coords = _data()
    sgwr = SGWR(
        bandwidth=20,
        adaptive=True,
        alpha=0.45,
        store_weights=True,
    ).fit(X, y, coords)
    components = weight_components(sgwr)
    assert {"spatial", "similarity", "combined"}.issubset(components.components)
    rows = focus_weight_components(sgwr, 3)
    assert all(values.shape == (len(y),) for values in rows.values())

    times = np.repeat(np.arange(4), len(y) // 4)
    gtwr = GTWR(
        bandwidth=20,
        adaptive=True,
        lambda_st=0.3,
    ).fit(X, y, coords, times)
    groups = temporal_groups(gtwr)
    assert len(groups.values) == 4
    trajectory = parameter_trajectory(gtwr, "income")
    assert trajectory.shape[0] == 4


def test_full_matrix_inference_for_mgtwr_and_scalable_gwr():
    X, y, coords = _data()
    times = np.repeat(np.arange(4), len(y) // 4)
    mgtwr = MGTWR(
        bandwidths=[20, 20, 20],
        taus=[1.0, 1.0, 1.0],
        adaptive=True,
        calculate_inference=True,
    ).fit(X, y, coords, times)
    mgtwr_view = parameter_significance(mgtwr, "income", correction="bh")
    assert mgtwr_view.shape[0] == len(y)
    assert model_diagnostic_summary(mgtwr).metrics["r2"] > 0.0

    scalable = ScalableGWR(
        bandwidth=20,
        optimize_bandwidth=False,
    ).fit(X, y, coords)
    scalable_view = parameter_significance(scalable, "income", correction="bh")
    assert scalable_view.shape[0] == len(y)
    assert model_diagnostic_summary(scalable).metrics["cv"] > 0.0
