# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Private-first bandwidth and neighbourhood semantics for pyGWRx.

This module separates the mathematical meaning of fixed/adaptive bandwidths
from model execution strategy. It deliberately represents neighbourhood
boundary, duplicate-distance, tie, focal-observation, and leave-one-out rules
explicitly so future model migrations cannot silently substitute one model
family's adaptive semantics for another's.

Weight storage, bandwidth search objectives, distance calculation, solvers, and
model-specific geometry do not belong here.

Author:
    Jinghao Hu
"""


from __future__ import annotations

__author__ = "Jinghao Hu"
__license__ = "MIT"

from dataclasses import dataclass
from numbers import Integral, Real
from typing import Literal, TypeAlias, Union

import numpy as np

from pygwrx.core.kernels import KernelLike, get_kernel_function

__all__: tuple[str, ...] = ()

BoundaryRule: TypeAlias = Literal["include_kth", "kernel_boundary"]
ZeroDistanceRule: TypeAlias = Literal["smallest_positive", "stable_top_k"]
TieRule: TypeAlias = Literal["distance_threshold", "stable_rank"]
LoocvFocalExclusion: TypeAlias = Literal[
    "after_weight_construction",
    "before_neighbour_selection",
]

_ALLOWED_BOUNDARY_RULES = {"include_kth", "kernel_boundary"}
_ALLOWED_ZERO_DISTANCE_RULES = {"smallest_positive", "stable_top_k"}
_ALLOWED_TIE_RULES = {"distance_threshold", "stable_rank"}
_ALLOWED_LOOCV_RULES = {"after_weight_construction", "before_neighbour_selection"}


@dataclass(frozen=True)
class NeighbourhoodPolicy:
    """Describe adaptive-neighbourhood semantics independently of a model class.

    Args:
        focal_observation_counts: Whether a zero-distance focal observation counts
            toward the requested neighbour order when it is present in the candidate
            distance row.
        boundary_rule: ``"include_kth"`` advances a positive k-th distance by one
            representable float before evaluating the kernel. ``"kernel_boundary"``
            passes the k-th distance to the kernel unchanged.
        zero_distance_rule: Fallback when the selected k-th distance is zero.
            ``"smallest_positive"`` uses the smallest positive distance;
            ``"stable_top_k"`` selects exactly the first k observations under a
            stable distance ranking and assigns them unit weight.
        tie_rule: ``"distance_threshold"`` defines the neighbourhood by distance
            scale and therefore may include more than k observations at a tied
            boundary. ``"stable_rank"`` preserves input order when an exact top-k
            selection is required.
        loocv_focal_exclusion: Stage at which leave-one-out calibration removes
            the focal observation.

    Notes:
        The policy is descriptive as well as executable. In particular,
        ``focal_observation_counts`` records an important model-family contract;
        callers remain responsible for supplying the distance candidates implied
        by that contract when constructing non-standard focal-excluding policies.
    """

    focal_observation_counts: bool
    boundary_rule: BoundaryRule
    zero_distance_rule: ZeroDistanceRule
    tie_rule: TieRule
    loocv_focal_exclusion: LoocvFocalExclusion

    def __post_init__(self) -> None:
        if not isinstance(self.focal_observation_counts, bool):
            raise TypeError("focal_observation_counts must be boolean.")
        if self.boundary_rule not in _ALLOWED_BOUNDARY_RULES:
            raise ValueError(f"Unsupported boundary_rule: {self.boundary_rule!r}.")
        if self.zero_distance_rule not in _ALLOWED_ZERO_DISTANCE_RULES:
            raise ValueError(
                f"Unsupported zero_distance_rule: {self.zero_distance_rule!r}."
            )
        if self.tie_rule not in _ALLOWED_TIE_RULES:
            raise ValueError(f"Unsupported tie_rule: {self.tie_rule!r}.")
        if self.loocv_focal_exclusion not in _ALLOWED_LOOCV_RULES:
            raise ValueError(
                "Unsupported loocv_focal_exclusion: "
                f"{self.loocv_focal_exclusion!r}."
            )


@dataclass(frozen=True)
class FixedBandwidth:
    """Internal normalized fixed-distance bandwidth."""

    value: float

    def __post_init__(self) -> None:
        if isinstance(self.value, (bool, np.bool_)) or not isinstance(self.value, Real):
            raise TypeError("Fixed bandwidth must be a positive real scalar.")
        value = float(self.value)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError("Fixed bandwidth must be finite and greater than zero.")
        object.__setattr__(self, "value", value)


@dataclass(frozen=True)
class AdaptiveBandwidth:
    """Internal normalized adaptive neighbour-order bandwidth."""

    k: int
    neighbourhood_policy: NeighbourhoodPolicy

    def __post_init__(self) -> None:
        if isinstance(self.k, (bool, np.bool_)) or not isinstance(self.k, Integral):
            raise TypeError("Adaptive bandwidth k must be a positive integer.")
        k = int(self.k)
        if k < 1:
            raise ValueError("Adaptive bandwidth k must be at least 1.")
        if not isinstance(self.neighbourhood_policy, NeighbourhoodPolicy):
            raise TypeError("neighbourhood_policy must be a NeighbourhoodPolicy.")
        object.__setattr__(self, "k", k)


BandwidthSpec: TypeAlias = Union[FixedBandwidth, AdaptiveBandwidth]


# Standard GWR/GTWR-style adaptive semantics currently used by the shared solver
# and by MGWR's vectorized adaptive-weight path.
DISTANCE_THRESHOLD_INCLUSIVE_POLICY = NeighbourhoodPolicy(
    focal_observation_counts=True,
    boundary_rule="include_kth",
    zero_distance_rule="smallest_positive",
    tie_rule="distance_threshold",
    loocv_focal_exclusion="after_weight_construction",
)

# GWmodel-compatible stable-rank semantics currently used by GWPCA, GWDA, and
# GWSS. Compact kernels receive the exact k-th distance, while boxcar and a
# zero-distance k-th rank use an exact stable top-k membership mask.
STABLE_RANK_KERNEL_BOUNDARY_POLICY = NeighbourhoodPolicy(
    focal_observation_counts=True,
    boundary_rule="kernel_boundary",
    zero_distance_rule="stable_top_k",
    tie_rule="stable_rank",
    loocv_focal_exclusion="after_weight_construction",
)


def normalize_bandwidth(
    bandwidth: float | int,
    *,
    adaptive: bool,
    neighbourhood_policy: NeighbourhoodPolicy | None = None,
) -> BandwidthSpec:
    """Normalize the legacy number-plus-adaptive form into an internal spec."""
    if not isinstance(adaptive, (bool, np.bool_)):
        raise TypeError("adaptive must be boolean.")
    if bool(adaptive):
        if neighbourhood_policy is None:
            raise ValueError(
                "Adaptive bandwidths require an explicit neighbourhood_policy."
            )
        if isinstance(bandwidth, (bool, np.bool_)):
            raise TypeError("Adaptive bandwidth must be an integer neighbour count.")
        numeric = float(bandwidth)
        if not np.isfinite(numeric) or not numeric.is_integer():
            raise ValueError(
                "Adaptive bandwidth must be a finite integer neighbour count."
            )
        return AdaptiveBandwidth(int(numeric), neighbourhood_policy)
    if neighbourhood_policy is not None:
        raise ValueError("neighbourhood_policy applies only to adaptive bandwidths.")
    return FixedBandwidth(bandwidth)


def _validate_distance_row(distances: np.ndarray) -> np.ndarray:
    """Validate one non-negative finite distance row."""
    try:
        values = np.asarray(distances, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError("distances must contain numeric values.") from exc
    if values.ndim != 1:
        raise ValueError("distances must be one-dimensional.")
    if values.size == 0:
        raise ValueError("distances must contain at least one value.")
    if not np.all(np.isfinite(values)):
        raise ValueError("distances contain NaN or infinite values.")
    if np.any(values < 0.0):
        raise ValueError("distances must be non-negative.")
    return values


def _stable_top_k_weights(distances: np.ndarray, k: int) -> np.ndarray:
    """Return exact top-k unit weights with stable input-order tie breaking."""
    order = np.argsort(distances, kind="stable")
    weights = np.zeros_like(distances, dtype=float)
    weights[order[:k]] = 1.0
    return weights


def _adaptive_distance_scale(
    distances: np.ndarray,
    bandwidth: AdaptiveBandwidth,
) -> float | None:
    """Resolve the adaptive distance scale, or ``None`` for exact stable top-k."""
    k = bandwidth.k
    if k > distances.size:
        raise ValueError(
            f"Adaptive bandwidth k must not exceed {distances.size}; got {k}."
        )

    policy = bandwidth.neighbourhood_policy
    if policy.tie_rule == "stable_rank":
        order = np.argsort(distances, kind="stable")
        selected = float(distances[order[k - 1]])
    else:
        selected = float(np.partition(distances, k - 1)[k - 1])

    if selected <= 0.0:
        if policy.zero_distance_rule == "stable_top_k":
            return None
        positive = distances[distances > 0.0]
        if positive.size == 0:
            raise ValueError(
                "Adaptive bandwidth is undefined because all distances are zero."
            )
        selected = float(np.min(positive))

    if policy.boundary_rule == "include_kth":
        selected = float(np.nextafter(selected, np.inf))
    return selected


def _validate_weight_row(
    weights: np.ndarray,
    expected_shape: tuple[int, ...],
) -> np.ndarray:
    """Validate one kernel-generated observation-weight row."""
    values = np.asarray(weights, dtype=float)
    if values.shape != expected_shape:
        raise ValueError(
            "Kernel must return a weight row with the same shape as distances; "
            f"expected {expected_shape}, got {values.shape}."
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("Kernel returned NaN or infinite weights.")
    if np.any(values < 0.0):
        raise ValueError("Kernel returned negative weights.")
    return values


def weights_from_distances(
    distances: np.ndarray,
    bandwidth: BandwidthSpec,
    kernel: KernelLike,
) -> np.ndarray:
    """Construct one weight row under an explicit bandwidth/neighbourhood policy.

    This function does not calculate distances, store dense weight matrices, perform
    leave-one-out exclusion, or choose a bandwidth. Those responsibilities remain
    separate from neighbourhood semantics.
    """
    values = _validate_distance_row(distances)
    kernel_func = get_kernel_function(kernel)

    if isinstance(bandwidth, FixedBandwidth):
        return _validate_weight_row(
            kernel_func(values, bandwidth.value),
            values.shape,
        )
    if not isinstance(bandwidth, AdaptiveBandwidth):
        raise TypeError("bandwidth must be FixedBandwidth or AdaptiveBandwidth.")

    policy = bandwidth.neighbourhood_policy
    if bandwidth.k > values.size:
        raise ValueError(
            f"Adaptive bandwidth k must not exceed {values.size}; got {bandwidth.k}."
        )

    kernel_name = kernel.strip().lower() if isinstance(kernel, str) else None
    if policy.tie_rule == "stable_rank" and kernel_name == "boxcar":
        return _stable_top_k_weights(values, bandwidth.k)

    distance_scale = _adaptive_distance_scale(values, bandwidth)
    if distance_scale is None:
        return _stable_top_k_weights(values, bandwidth.k)

    return _validate_weight_row(
        kernel_func(values, distance_scale),
        values.shape,
    )


def exclude_focal_for_loocv(
    weights: np.ndarray,
    focal_index: int,
    *,
    policy: NeighbourhoodPolicy,
) -> np.ndarray:
    """Apply the current post-construction leave-one-out focal exclusion rule."""
    values = np.asarray(weights, dtype=float)
    if values.ndim != 1:
        raise ValueError("weights must be one-dimensional.")
    if not isinstance(focal_index, Integral) or isinstance(
        focal_index, (bool, np.bool_)
    ):
        raise TypeError("focal_index must be an integer.")
    index = int(focal_index)
    if index < 0 or index >= values.size:
        raise IndexError("focal_index is outside the weight row.")
    if policy.loocv_focal_exclusion != "after_weight_construction":
        raise ValueError(
            "This helper applies only to policies that exclude the focal observation "
            "after weight construction."
        )
    result = values.copy()
    result[index] = 0.0
    return result
