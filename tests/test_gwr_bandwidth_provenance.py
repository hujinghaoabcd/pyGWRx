"""Regression tests for GWR bandwidth-search range and provenance."""

from __future__ import annotations

import numpy as np
import pytest

from pygwrx import GWR
from pygwrx.core.bandwidth import CrossValidationSelector, _automatic_bandwidth_range
from pygwrx.core.kernels import bisquare_kernel
from pygwrx.core.utils import compute_distance_matrix


def test_automatic_fixed_range_uses_full_observed_distance_scale() -> None:
    coords = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [100.0, 0.0],
        ]
    )
    distances = compute_distance_matrix(coords, coords)

    lower, upper = _automatic_bandwidth_range(
        distances,
        adaptive=False,
        n_samples=coords.shape[0],
        n_features=2,
    )

    assert lower == pytest.approx(0.5)
    assert upper == pytest.approx(200.0)
    assert upper > np.max(distances)


def test_fixed_compact_search_keeps_isolated_location_in_domain() -> None:
    cluster = np.arange(50, dtype=float)
    x_coord = np.concatenate([cluster, [1000.0]])
    coords = np.column_stack([x_coord, np.zeros_like(x_coord)])
    x = x_coord / 1000.0
    X = np.column_stack([np.ones_like(x), x])
    y = 1.0 + 2.0 * x + 0.05 * np.sin(x_coord / 7.0)

    selector = CrossValidationSelector(
        n_intervals=9,
        optimization_method="grid",
        adaptive=False,
    )
    selected = selector.select(X, y, coords, bisquare_kernel)

    assert np.isfinite(selected)
    assert selector.search_range_ is not None
    assert float(selector.search_range_[1]) > float(np.max(compute_distance_matrix(coords)))
    assert any(np.isfinite(score) for _, score in selector.search_trace_)


def test_gwr_retains_adaptive_bandwidth_search_provenance(synthetic) -> None:
    model = GWR(
        kernel="bisquare",
        bandwidth="aicc",
        adaptive=True,
        bandwidth_range=(8, 12),
        optimization_method="brent",
    ).fit(
        synthetic["X"],
        synthetic["y"],
        synthetic["coords"],
        compute_hat_matrix=False,
        compute_local_r2=False,
        compute_inference=False,
    )

    search = model.bandwidth_search_
    assert search is not None
    assert search["criterion"] == "aicc"
    assert search["adaptive"] is True
    assert search["optimization_method"] == "exhaustive_integer"
    assert search["search_range"] == (8, 12)
    assert search["selected"] == model.bandwidth_
    assert np.isfinite(float(search["best_score"]))
    assert tuple(k for k, _ in search["trace"]) == tuple(range(8, 13))
    assert isinstance(search["boundary_solution"], bool)


def test_gwr_retains_fixed_bandwidth_search_provenance(synthetic) -> None:
    model = GWR(
        kernel="gaussian",
        bandwidth="cv",
        adaptive=False,
        optimization_method="grid",
    ).fit(
        synthetic["X"],
        synthetic["y"],
        synthetic["coords"],
        compute_hat_matrix=False,
        compute_local_r2=False,
        compute_inference=False,
    )

    search = model.bandwidth_search_
    assert search is not None
    assert search["criterion"] == "cv"
    assert search["adaptive"] is False
    assert search["optimization_method"] == "grid"
    assert search["selected"] == pytest.approx(model.bandwidth_)
    assert np.isfinite(float(search["best_score"]))
    assert search["search_range"] is not None
    assert len(search["trace"]) == 20

    lower, upper = search["search_range"]
    distances = compute_distance_matrix(synthetic["coords"])
    positive = distances[distances > 0.0]
    assert lower == pytest.approx(0.5 * float(np.min(positive)))
    assert upper == pytest.approx(2.0 * float(np.max(positive)))


def test_numeric_bandwidth_has_no_search_provenance(synthetic) -> None:
    model = GWR(kernel="gaussian", bandwidth=4.0, adaptive=False).fit(
        synthetic["X"],
        synthetic["y"],
        synthetic["coords"],
        compute_hat_matrix=False,
        compute_local_r2=False,
        compute_inference=False,
    )

    assert model.bandwidth_search_ is None


def test_summary_reports_automatic_bandwidth_search(synthetic) -> None:
    model = GWR(
        kernel="bisquare",
        bandwidth="aicc",
        adaptive=True,
        bandwidth_range=(8, 12),
    ).fit(
        synthetic["X"],
        synthetic["y"],
        synthetic["coords"],
        compute_hat_matrix=False,
        compute_local_r2=False,
        compute_inference=False,
    )

    summary = model.summary()
    assert "Bandwidth criterion: aicc" in summary
    assert "Bandwidth search method: exhaustive_integer" in summary
    assert "Bandwidth search range: (8, 12)" in summary
    assert "Bandwidth boundary solution:" in summary
