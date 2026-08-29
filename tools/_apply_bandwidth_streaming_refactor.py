from pathlib import Path
import re

path = Path("src/pygwrx/core/bandwidth.py")
text = path.read_text(encoding="utf-8")

old = "from typing import Callable, Optional, Tuple, Union"
new = "from typing import Callable, Iterator, Optional, Tuple, Union"
assert text.count(old) == 1
text = text.replace(old, new)

old = "from pygwrx.core.solver import weighted_least_squares\n"
new = (
    "from pygwrx.core.solver import weighted_least_squares\n"
    "from pygwrx.core.utils import compute_distance_matrix\n"
)
assert text.count(old) == 1
text = text.replace(old, new)

marker = "_RIDGE = 0.0\n"
helpers = '''_RIDGE = 0.0
_DISTANCE_BLOCK_ROWS = 128


def _iter_distance_rows(
    coords: np.ndarray,
    *,
    distance_metric: str,
) -> Iterator[np.ndarray]:
    """Yield coordinate-to-coordinate distance rows from bounded-size blocks."""
    n_samples = coords.shape[0]
    for start in range(0, n_samples, _DISTANCE_BLOCK_ROWS):
        stop = min(start + _DISTANCE_BLOCK_ROWS, n_samples)
        block = np.asarray(
            compute_distance_matrix(
                coords[start:stop],
                coords,
                metric=distance_metric,
            ),
            dtype=float,
        )
        expected_shape = (stop - start, n_samples)
        if block.shape != expected_shape:
            raise ValueError("The computed distance block has an invalid shape.")
        if not np.all(np.isfinite(block)) or np.any(block < 0):
            raise ValueError("The computed distance block contains invalid distances.")
        for distance_row in block:
            yield distance_row


def _positive_pairwise_distance_extrema(
    coords: np.ndarray,
    *,
    distance_metric: str,
) -> tuple[float, float]:
    """Return min/max positive unique pairwise distances without full materialization."""
    minimum_positive = np.inf
    maximum_positive = -np.inf

    for row_index, distance_row in enumerate(
        _iter_distance_rows(coords, distance_metric=distance_metric)
    ):
        unique_tail = distance_row[row_index + 1 :]
        positive = unique_tail[unique_tail > 0.0]
        if positive.size == 0:
            continue
        minimum_positive = min(minimum_positive, float(np.min(positive)))
        maximum_positive = max(maximum_positive, float(np.max(positive)))

    if not np.isfinite(minimum_positive) or not np.isfinite(maximum_positive):
        raise ValueError(
            "Cannot select a fixed bandwidth because all pairwise coordinate distances "
            "are zero. Use distinct coordinates or an adaptive specification "
            "with valid non-zero neighbour distances."
        )
    return float(minimum_positive), float(maximum_positive)
'''
assert text.count(marker) == 1
text = text.replace(marker, helpers, 1)

pattern = re.compile(
    r"def _automatic_bandwidth_range\(.*?\n\n\ndef _normalize_candidate\(",
    re.S,
)
replacement = '''def _automatic_bandwidth_range(
    coords: np.ndarray,
    *,
    distance_metric: str,
    adaptive: bool,
    n_samples: int,
    n_features: int,
) -> tuple[Bandwidth, Bandwidth]:
    """Derive a valid search interval without retaining a full distance matrix.

    Adaptive bandwidths use an integer neighbour-order domain and do not require
    pairwise distances to establish the search range. Fixed-distance searches scan
    bounded distance blocks only to recover the minimum and maximum positive unique
    pairwise distances used by the existing search-range policy.
    """
    if adaptive:
        lower = max(n_features + 1, 2, int(np.ceil(0.05 * n_samples)))
        upper = n_samples
        if lower > upper:
            raise ValueError(
                "Adaptive bandwidth selection is not possible: the sample size is too "
                "small for the number of design-matrix columns."
            )
        return lower, upper

    minimum_positive, maximum_positive = _positive_pairwise_distance_extrema(
        coords,
        distance_metric=distance_metric,
    )

    lower = max(np.nextafter(0.0, 1.0), 0.5 * minimum_positive)
    upper = 2.0 * maximum_positive
    if not np.isfinite(upper):
        upper = float(np.nextafter(maximum_positive, np.inf))
    if upper <= lower:
        upper = float(np.nextafter(lower, np.inf))

    return float(lower), float(upper)


def _normalize_candidate('''
text, count = pattern.subn(replacement, text, count=1)
assert count == 1

pattern = re.compile(r"    def _prepare\(.*?\n    def _search\(", re.S)
replacement = '''    def _prepare(
        self,
        X: np.ndarray,
        y: np.ndarray,
        coords: np.ndarray,
        kernel_func: KernelFunction,
        bandwidth_range: BandwidthRange,
        distance_metric: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, Bandwidth, Bandwidth]:
        if not isinstance(distance_metric, str):
            raise TypeError("distance_metric must be a string.")

        X_arr, y_arr, coords_arr = _validate_selector_inputs(X, y, coords, kernel_func)

        legacy_auto_range = False
        if (
            bandwidth_range is not None
            and isinstance(bandwidth_range, (tuple, list))
            and len(bandwidth_range) == 2
        ):
            try:
                range_lower = float(bandwidth_range[0])
                range_upper = float(bandwidth_range[1])
            except (TypeError, ValueError):
                range_lower = np.nan
                range_upper = np.nan

            if self.adaptive:
                legacy_auto_range = (
                    X_arr.shape[0] < 20
                    and range_lower >= 20
                    and np.isclose(range_upper, X_arr.shape[0])
                )
            else:
                legacy_auto_range = (
                    np.isclose(range_lower, 1.0) and 0 < range_upper < 1.0
                )

        if bandwidth_range is None or legacy_auto_range:
            lower, upper = _automatic_bandwidth_range(
                coords_arr,
                distance_metric=distance_metric,
                adaptive=self.adaptive,
                n_samples=X_arr.shape[0],
                n_features=X_arr.shape[1],
            )
        else:
            lower, upper = _validate_bandwidth_range(
                bandwidth_range,
                adaptive=self.adaptive,
                n_samples=X_arr.shape[0],
                n_features=X_arr.shape[1],
            )

        return X_arr, y_arr, coords_arr, lower, upper

    def _search('''
text, count = pattern.subn(replacement, text, count=1)
assert count == 1

old = "X_arr, y_arr, _, distances, lower, upper = self._prepare("
assert text.count(old) == 3
text = text.replace(old, "X_arr, y_arr, coords_arr, lower, upper = self._prepare(")

old = "            for i, dists in enumerate(distances):"
new = '''            for i, dists in enumerate(
                _iter_distance_rows(coords_arr, distance_metric=distance_metric)
            ):'''
assert text.count(old) == 3
text = text.replace(old, new)

path.write_text(text, encoding="utf-8")

test_path = Path("tests/test_bandwidth_distance_streaming.py")
test_path.write_text('''# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Regression tests for bounded-memory bandwidth distance evaluation."""

import importlib

import numpy as np
import pytest

from pygwrx.core.bandwidth import AICSelector, BICSelector, CrossValidationSelector
from pygwrx.core.kernels import gaussian_kernel

bandwidth_module = importlib.import_module("pygwrx.core.bandwidth")


def _make_data(n_samples: int = 132):
    rng = np.random.default_rng(20260829)
    coords = rng.uniform(0.0, 1.0, size=(n_samples, 2))
    X = np.column_stack([np.ones(n_samples), rng.normal(size=n_samples)])
    y = 1.5 + 0.8 * X[:, 1] + rng.normal(0.0, 0.08, size=n_samples)
    return X, y, coords


def _track_distance_blocks(monkeypatch, n_samples: int):
    original = bandwidth_module.compute_distance_matrix
    calls = []

    def tracked(left, right, metric="euclidean"):
        left_arr = np.asarray(left)
        right_arr = np.asarray(right)
        calls.append((left_arr.shape[0], right_arr.shape[0]))
        if left_arr.shape[0] == n_samples and right_arr.shape[0] == n_samples:
            raise AssertionError("bandwidth selection requested a full n x n distance matrix")
        return original(left, right, metric=metric)

    monkeypatch.setattr(bandwidth_module, "compute_distance_matrix", tracked)
    return calls


@pytest.mark.parametrize(
    "selector",
    [
        CrossValidationSelector(n_intervals=2, optimization_method="grid"),
        AICSelector(n_intervals=2, optimization_method="grid"),
        BICSelector(n_intervals=2, optimization_method="grid"),
    ],
)
def test_fixed_bandwidth_objectives_use_bounded_distance_blocks(monkeypatch, selector):
    X, y, coords = _make_data()
    calls = _track_distance_blocks(monkeypatch, coords.shape[0])
    selected = selector.select(
        X,
        y,
        coords,
        gaussian_kernel,
        bandwidth_range=(0.25, 1.25),
    )
    assert np.isfinite(float(selected))
    assert calls
    assert max(left_rows for left_rows, _ in calls) <= 128
    assert {right_rows for _, right_rows in calls} == {coords.shape[0]}


def test_fixed_automatic_range_uses_bounded_distance_scan(monkeypatch):
    X, y, coords = _make_data()
    calls = _track_distance_blocks(monkeypatch, coords.shape[0])
    selector = CrossValidationSelector(n_intervals=2, optimization_method="grid")
    selected = selector.select(X, y, coords, gaussian_kernel)
    assert np.isfinite(float(selected))
    assert selector.search_range_ is not None
    assert calls
    assert max(left_rows for left_rows, _ in calls) <= 128


def test_adaptive_bandwidth_objective_uses_bounded_distance_blocks(monkeypatch):
    X, y, coords = _make_data()
    calls = _track_distance_blocks(monkeypatch, coords.shape[0])
    selector = CrossValidationSelector(adaptive=True)
    selected = selector.select(
        X,
        y,
        coords,
        gaussian_kernel,
        bandwidth_range=(8, 10),
    )
    assert int(selected) in {8, 9, 10}
    assert calls
    assert max(left_rows for left_rows, _ in calls) <= 128
''', encoding="utf-8")
