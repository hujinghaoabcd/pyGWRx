# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Architecture and numerical contracts for the C1 private GWR engine."""

from __future__ import annotations

import ast
import inspect

import numpy as np
import pytest

import pygwrx
import pygwrx.models as models
import pygwrx.models.gwr as gwr_module
from pygwrx.core.bandwidth import get_bandwidth_selector
from pygwrx.core.kernels import gaussian_kernel
from pygwrx.models import _gwr_engine
from pygwrx.models._gwr_engine import _get_gwr_bandwidth_selector


def _selector_data(n_samples: int = 18) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(20260830)
    coords = rng.uniform(0.0, 2.0, size=(n_samples, 2))
    x = rng.normal(size=n_samples)
    X = np.column_stack([np.ones(n_samples), x])
    y = (
        1.5
        + 0.7 * x
        + 0.08 * np.sin(coords[:, 0] * 2.0)
        + rng.normal(0.0, 0.03, size=n_samples)
    )
    return X, y, coords


def test_private_engine_is_not_part_of_public_exports() -> None:
    assert "_gwr_engine" not in getattr(models, "__all__", ())
    assert "_GWRBandwidthSelector" not in getattr(models, "__all__", ())
    assert "_GWRBandwidthSelector" not in getattr(pygwrx, "__all__", ())
    assert not hasattr(pygwrx, "_GWRBandwidthSelector")


def test_gwr_production_module_does_not_own_bandwidth_objective() -> None:
    tree = ast.parse(inspect.getsource(gwr_module))
    imported_from_core_bandwidth = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module == "pygwrx.core.bandwidth"
    ]
    assert imported_from_core_bandwidth == []

    source = inspect.getsource(gwr_module.GWR._resolve_bandwidth)
    assert "_get_gwr_bandwidth_selector" in source
    assert "get_bandwidth_selector" not in source.replace(
        "_get_gwr_bandwidth_selector", ""
    )


def test_engine_keeps_geometry_weights_and_rank_policy_injected() -> None:
    signature = inspect.signature(_gwr_engine._fit_gwr_training_locations)
    assert "distance_rows" in signature.parameters
    assert "weights_from_distances" in signature.parameters
    assert "rank_policy" in signature.parameters

    source = inspect.getsource(_gwr_engine._fit_gwr_training_locations)
    assert "compute_distance_matrix" not in source
    assert "adaptive_bandwidth_weights" not in source
    assert "kernel_func" not in source


@pytest.mark.parametrize("criterion", ["cv", "aic", "aicc", "bic"])
def test_private_fixed_grid_selector_matches_public_compatibility_facade(
    criterion: str,
) -> None:
    X, y, coords = _selector_data()
    kwargs = {
        "n_intervals": 7,
        "optimization_method": "grid",
        "adaptive": False,
        "verbose": False,
    }
    public = get_bandwidth_selector(criterion, **kwargs)
    private = _get_gwr_bandwidth_selector(
        criterion,
        adaptive=False,
        verbose=False,
        optimization_method="grid",
    )
    private.n_intervals = 7

    public_selected = public.select(
        X,
        y,
        coords,
        gaussian_kernel,
        bandwidth_range=(0.25, 2.5),
    )
    private_selected = private.select(
        X,
        y,
        coords,
        gaussian_kernel,
        bandwidth_range=(0.25, 2.5),
    )

    assert private_selected == public_selected
    assert private.search_range_ == public.search_range_
    assert private.best_score_ == public.best_score_
    assert private.search_trace_ == public.search_trace_


@pytest.mark.parametrize("criterion", ["cv", "aic", "aicc", "bic"])
def test_private_adaptive_selector_matches_public_exhaustive_trace(
    criterion: str,
) -> None:
    X, y, coords = _selector_data(16)
    public = get_bandwidth_selector(
        criterion,
        adaptive=True,
        optimization_method="brent",
        verbose=False,
    )
    private = _get_gwr_bandwidth_selector(
        criterion,
        adaptive=True,
        optimization_method="brent",
        verbose=False,
    )

    public_selected = public.select(
        X,
        y,
        coords,
        gaussian_kernel,
        bandwidth_range=(5, 8),
    )
    private_selected = private.select(
        X,
        y,
        coords,
        gaussian_kernel,
        bandwidth_range=(5, 8),
    )

    assert private_selected == public_selected
    assert private.search_range_ == public.search_range_ == (5, 8)
    assert private.best_score_ == public.best_score_
    assert private.search_trace_ == public.search_trace_
    assert tuple(candidate for candidate, _ in private.search_trace_) == (5, 6, 7, 8)


def test_private_engine_defines_no_estimator_base_class() -> None:
    tree = ast.parse(inspect.getsource(_gwr_engine))
    class_names = {
        node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)
    }
    assert "BaseGWR" not in class_names
    assert "GWR" not in class_names
