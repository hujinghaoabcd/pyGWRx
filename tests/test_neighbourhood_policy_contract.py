# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Architecture contracts for explicit B5 neighbourhood semantics.

Author:
    Jinghao Hu
"""

from __future__ import annotations

import numpy as np
import pytest

import pygwrx
import pygwrx.core as core
from pygwrx.core import weights as weights_module
from pygwrx.core.kernels import bisquare_kernel
from pygwrx.core.solver import adaptive_bandwidth_weights
from pygwrx.core.weights import (
    DISTANCE_THRESHOLD_INCLUSIVE_POLICY,
    STABLE_RANK_KERNEL_BOUNDARY_POLICY,
    AdaptiveBandwidth,
    FixedBandwidth,
    NeighbourhoodPolicy,
    exclude_focal_for_loocv,
    normalize_bandwidth,
    weights_from_distances,
)
from pygwrx.models.gwda import GWDA
from pygwrx.models.gwpca import GWPCA
from pygwrx.models.gwss import GWSS
from pygwrx.models.mgwr import MGWR


def test_weights_spine_is_private_first() -> None:
    """Keep B5 policy/spec objects internal until the 0.2 API decision."""
    assert weights_module.__all__ == ()
    for name in (
        "NeighbourhoodPolicy",
        "FixedBandwidth",
        "AdaptiveBandwidth",
        "DISTANCE_THRESHOLD_INCLUSIVE_POLICY",
        "STABLE_RANK_KERNEL_BOUNDARY_POLICY",
    ):
        assert not hasattr(core, name)
        assert not hasattr(pygwrx, name)


def test_bandwidth_specs_are_explicit_and_validated() -> None:
    """Normalize legacy numeric bandwidths without hiding adaptive semantics."""
    fixed = normalize_bandwidth(2.5, adaptive=False)
    assert fixed == FixedBandwidth(2.5)

    adaptive = normalize_bandwidth(
        4,
        adaptive=True,
        neighbourhood_policy=DISTANCE_THRESHOLD_INCLUSIVE_POLICY,
    )
    assert adaptive == AdaptiveBandwidth(4, DISTANCE_THRESHOLD_INCLUSIVE_POLICY)

    with pytest.raises(ValueError, match="explicit neighbourhood_policy"):
        normalize_bandwidth(4, adaptive=True)
    with pytest.raises(ValueError, match="applies only to adaptive"):
        normalize_bandwidth(
            2.5,
            adaptive=False,
            neighbourhood_policy=DISTANCE_THRESHOLD_INCLUSIVE_POLICY,
        )
    with pytest.raises(TypeError, match="positive integer"):
        AdaptiveBandwidth(3.5, DISTANCE_THRESHOLD_INCLUSIVE_POLICY)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="greater than zero"):
        FixedBandwidth(0.0)


def test_two_frozen_adaptive_policy_families_are_semantically_distinct() -> None:
    """Do not collapse inclusive distance-threshold and stable-rank semantics."""
    distances = np.array([0.0, 1.0, 2.0, 2.0, 4.0])
    threshold = weights_from_distances(
        distances,
        AdaptiveBandwidth(3, DISTANCE_THRESHOLD_INCLUSIVE_POLICY),
        "bisquare",
    )
    stable_rank = weights_from_distances(
        distances,
        AdaptiveBandwidth(3, STABLE_RANK_KERNEL_BOUNDARY_POLICY),
        "bisquare",
    )

    # The GWR-style policy advances the k-th distance, so every observation tied
    # on that distance remains just inside a compact-kernel boundary.
    assert np.count_nonzero(threshold > 0.0) == 4
    assert threshold[2] > 0.0
    assert threshold[3] > 0.0

    # The GWmodel stable-rank policy passes the exact k-th distance to bisquare,
    # so observations exactly on the boundary receive zero weight.
    assert np.count_nonzero(stable_rank > 0.0) == 2
    assert stable_rank[2] == 0.0
    assert stable_rank[3] == 0.0


def test_distance_threshold_policy_matches_existing_solver_semantics() -> None:
    """Freeze GWR-style boundary and duplicate-coordinate behavior."""
    for distances, k in (
        (np.array([0.0, 1.0, 2.0, 2.0, 4.0]), 3),
        (np.array([0.0, 0.0, 0.0, 3.0, 5.0]), 2),
    ):
        historical_scale = adaptive_bandwidth_weights(distances, k)
        expected = bisquare_kernel(distances, historical_scale)
        actual = weights_from_distances(
            distances,
            AdaptiveBandwidth(k, DISTANCE_THRESHOLD_INCLUSIVE_POLICY),
            "bisquare",
        )
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)


def test_distance_threshold_policy_matches_existing_mgwr_vectorized_path() -> None:
    """Freeze MGWR's current inclusive k-th distance mechanism before migration."""
    distances = np.array(
        [
            [0.0, 1.0, 2.0, 2.0, 5.0],
            [1.0, 0.0, 1.0, 3.0, 4.0],
            [0.0, 0.0, 0.0, 2.0, 6.0],
        ]
    )
    model = MGWR(kernel="bisquare", bandwidths=3, adaptive=True)
    expected = model._adaptive_weight_matrix(distances.copy(), 3)
    actual = np.vstack(
        [
            weights_from_distances(
                row,
                AdaptiveBandwidth(3, DISTANCE_THRESHOLD_INCLUSIVE_POLICY),
                "bisquare",
            )
            for row in distances
        ]
    )
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)


@pytest.mark.parametrize(
    "kernel,distances,k",
    [
        ("bisquare", np.array([0.0, 1.0, 1.0, 2.0, 4.0]), 3),
        ("bisquare", np.array([0.0, 0.0, 0.0, 2.0, 4.0]), 2),
        ("boxcar", np.array([0.0, 1.0, 1.0, 1.0, 4.0]), 3),
    ],
)
def test_stable_rank_policy_matches_gwmodel_family(
    kernel: str,
    distances: np.ndarray,
    k: int,
) -> None:
    """Freeze GWPCA/GWDA/GWSS stable-rank, tie, and zero-distance behavior."""
    models = (
        GWPCA(kernel=kernel, bandwidth=k, adaptive=True),
        GWDA(kernel=kernel, bandwidth=k, adaptive=True),
        GWSS(kernel=kernel, bandwidth=k, adaptive=True),
    )
    actual = weights_from_distances(
        distances,
        AdaptiveBandwidth(k, STABLE_RANK_KERNEL_BOUNDARY_POLICY),
        kernel,
    )
    for model in models:
        expected = model._weights(distances, k)
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)


def test_stable_rank_boxcar_uses_input_order_to_break_boundary_ties() -> None:
    """Exact top-k boxcar support must not expand to every tied-distance row."""
    distances = np.array([0.0, 1.0, 1.0, 1.0, 3.0])
    weights = weights_from_distances(
        distances,
        AdaptiveBandwidth(3, STABLE_RANK_KERNEL_BOUNDARY_POLICY),
        "boxcar",
    )
    np.testing.assert_array_equal(weights, np.array([1.0, 1.0, 1.0, 0.0, 0.0]))


def test_loocv_policy_records_post_construction_focal_exclusion() -> None:
    """Freeze the current convention: focal counts toward k, then its weight is zeroed."""
    for policy in (
        DISTANCE_THRESHOLD_INCLUSIVE_POLICY,
        STABLE_RANK_KERNEL_BOUNDARY_POLICY,
    ):
        assert policy.focal_observation_counts is True
        assert policy.loocv_focal_exclusion == "after_weight_construction"
        original = np.array([1.0, 0.8, 0.4, 0.0])
        excluded = exclude_focal_for_loocv(original, 0, policy=policy)
        np.testing.assert_array_equal(excluded, np.array([0.0, 0.8, 0.4, 0.0]))
        np.testing.assert_array_equal(original, np.array([1.0, 0.8, 0.4, 0.0]))


def test_policy_fields_reject_unknown_semantics() -> None:
    """Prevent silent typo-driven creation of a third neighbourhood convention."""
    with pytest.raises(ValueError, match="Unsupported boundary_rule"):
        NeighbourhoodPolicy(
            focal_observation_counts=True,
            boundary_rule="expand",  # type: ignore[arg-type]
            zero_distance_rule="smallest_positive",
            tie_rule="distance_threshold",
            loocv_focal_exclusion="after_weight_construction",
        )
