# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""A5 contract tests for GTWR user-facing numerical warnings."""

from __future__ import annotations

import warnings

import numpy as np

from pygwrx import GTWR


def test_gtwr_sparse_local_warning_describes_minimum_norm_unpenalized_solution():
    """Sparse local fits must describe the rank-aware solver truthfully."""
    design = np.array(
        [
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
        ]
    )
    response = np.array([1.0, 2.0, 3.0])
    distances = np.array(
        [
            [0.0, 10.0, 10.0],
            [10.0, 0.0, 10.0],
            [10.0, 10.0, 0.0],
        ]
    )

    model = GTWR(kernel="bisquare", bandwidth=1.0, lambda_st=1.0)
    model.y_train_ = response.copy()
    model.bandwidth_ = 1.0

    def focal_only_kernel(distance_row: np.ndarray, bandwidth: float) -> np.ndarray:
        return np.where(distance_row < bandwidth, 1.0, 0.0)

    model.kernel_func_ = focal_only_kernel

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        local_fit = model._fit_training_locations(
            design,
            distances,
            store_hat_matrix=False,
            compute_inference=False,
        )

    runtime_messages = [
        str(item.message)
        for item in caught
        if issubclass(item.category, RuntimeWarning)
    ]
    assert len(runtime_messages) == design.shape[0]
    assert all(
        "rank-aware WLS solver returns a minimum-norm unpenalized local solution"
        in message
        for message in runtime_messages
    )
    assert all("ridge" not in message.lower() for message in runtime_messages)

    expected = np.vstack(
        [np.linalg.pinv(design[[index]]) @ response[[index]] for index in range(3)]
    )
    np.testing.assert_allclose(local_fit.params, expected, atol=1e-12, rtol=0.0)
