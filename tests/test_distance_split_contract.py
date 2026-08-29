# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Architecture contracts for the B3 distance-module split.

Author:
    Jinghao Hu
"""

from __future__ import annotations

import numpy as np
import pytest

import pygwrx
import pygwrx.core as core
from pygwrx.core import distance, utils


PUBLIC_DISTANCE_NAMES = (
    "euclidean_distance",
    "manhattan_distance",
    "chebyshev_distance",
    "minkowski_distance",
    "haversine_distance",
    "compute_distance_matrix",
    "DistanceCache",
    "chunked_computation",
)


def test_distance_module_is_canonical_and_legacy_utils_reexports() -> None:
    """Existing public/legacy paths must resolve to the canonical distance owner."""
    for name in PUBLIC_DISTANCE_NAMES:
        canonical = getattr(distance, name)
        assert getattr(utils, name) is canonical
        assert getattr(core, name) is canonical

    assert distance.compute_distance_matrix.__module__ == "pygwrx.core.distance"
    assert distance.DistanceCache.__module__ == "pygwrx.core.distance"
    assert distance.chunked_computation.__module__ == "pygwrx.core.distance"


def test_distance_metric_spec_stays_private_first() -> None:
    """The new architecture value object is intentionally not a public core API."""
    assert hasattr(distance, "DistanceMetricSpec")
    assert "DistanceMetricSpec" not in distance.__all__
    assert "DistanceMetricSpec" not in core.__all__
    assert not hasattr(core, "DistanceMetricSpec")
    assert not hasattr(pygwrx, "DistanceMetricSpec")


def test_distance_metric_spec_normalizes_aliases_and_freezes_params() -> None:
    """Canonicalize ordinary metric names and parameters into one internal form."""
    cityblock = distance.DistanceMetricSpec(" cityblock ")
    assert cityblock.name == "manhattan"
    assert dict(cityblock.params) == {}

    params = {"p": 3.0}
    minkowski = distance.DistanceMetricSpec("minkowski", params)
    params["p"] = 4.0
    assert minkowski.name == "minkowski"
    assert dict(minkowski.params) == {"p": 3.0}

    with pytest.raises(TypeError, match="Unexpected parameter"):
        distance.DistanceMetricSpec("euclidean", {"p": 2.0})


def test_metric_params_flow_through_bounded_distance_blocks() -> None:
    """Private metric parameters must not disable bounded row/block execution."""
    coords = np.array([[0.0, 0.0], [1.0, 2.0], [4.0, 2.0]])

    blocks = list(
        distance._iter_distance_blocks(
            coords,
            distance_metric="minkowski",
            metric_params={"p": 1.0},
            block_rows=2,
        )
    )
    streamed = np.vstack(blocks)
    expected = distance.compute_distance_matrix(coords, metric="manhattan")

    assert [block.shape[0] for block in blocks] == [2, 1]
    np.testing.assert_allclose(streamed, expected, rtol=0.0, atol=0.0)


def test_distance_spec_does_not_accept_model_specific_geometry_parameters() -> None:
    """GTWR/model geometry must remain outside the ordinary metric specification."""
    for parameter in ("lambda", "tau", "ksi", "causal"):
        with pytest.raises(TypeError, match="Unexpected parameter"):
            distance.DistanceMetricSpec("euclidean", {parameter: 1.0})
