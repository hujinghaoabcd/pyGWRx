# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Architecture and statistical contracts for the B8 diagnostics split."""

from __future__ import annotations

import numpy as np
import pytest

import pygwrx.core as core
from pygwrx.core import gaussian_diagnostics, metrics

GAUSSIAN_DIAGNOSTIC_NAMES = (
    "compute_adjusted_r_squared",
    "compute_aic",
    "compute_aicc",
    "compute_bic",
    "compute_local_r_squared",
    "compute_effective_parameters",
    "compute_diagnostics",
    "compute_trace_statistics",
    "compute_edf",
    "compute_enp",
)


def test_gaussian_diagnostics_module_is_canonical_and_metrics_reexports() -> None:
    """Legacy/public paths must resolve to the canonical Gaussian owner."""
    for name in GAUSSIAN_DIAGNOSTIC_NAMES:
        canonical = getattr(gaussian_diagnostics, name)
        assert getattr(metrics, name) is canonical
        assert getattr(core, name) is canonical

    assert metrics.compute_r_squared.__module__ == "pygwrx.core.metrics"
    assert gaussian_diagnostics.compute_aic.__module__ == (
        "pygwrx.core.gaussian_diagnostics"
    )
    assert gaussian_diagnostics.compute_diagnostics.__module__ == (
        "pygwrx.core.gaussian_diagnostics"
    )
    assert "compute_r_squared" not in gaussian_diagnostics.__all__


def test_trace_edf_enp_conventions_and_diagnostic_keys_are_frozen() -> None:
    """B8 must move diagnostics without changing smoother-statistic semantics."""
    y_true = np.asarray([1.0, 2.0, 3.5, 5.0])
    y_pred = np.asarray([1.1, 1.8, 3.4, 4.8])
    hat_matrix = np.asarray(
        [
            [0.70, 0.20, 0.05, 0.05],
            [0.15, 0.65, 0.15, 0.05],
            [0.05, 0.15, 0.65, 0.15],
            [0.05, 0.05, 0.20, 0.70],
        ]
    )

    trace_s = float(np.trace(hat_matrix))
    trace_sts = float(np.sum(hat_matrix * hat_matrix))
    expected_enp_v2 = 2.0 * trace_s - trace_sts
    expected_edf_v2 = y_true.size - expected_enp_v2

    trace_stats = gaussian_diagnostics.compute_trace_statistics(hat_matrix)
    assert trace_stats == pytest.approx(
        {"trace_S": trace_s, "trace_StS": trace_sts}
    )
    assert gaussian_diagnostics.compute_enp(trace_s, trace_sts) == pytest.approx(
        expected_enp_v2
    )
    assert gaussian_diagnostics.compute_edf(
        y_true.size, trace_s, trace_sts
    ) == pytest.approx(expected_edf_v2)

    diagnostics = gaussian_diagnostics.compute_diagnostics(
        y_true,
        y_pred,
        hat_matrix=hat_matrix,
        compute_gwr_stats=True,
    )
    assert set(diagnostics) == {
        "r2",
        "rss",
        "rmse",
        "mae",
        "effective_params",
        "adj_r2",
        "aic",
        "aicc",
        "bic",
        "trace_S",
        "trace_StS",
        "enp_v1",
        "edf_v1",
        "enp_v2",
        "edf_v2",
        "enp",
        "edf",
    }
    assert diagnostics["effective_params"] == pytest.approx(trace_s)
    assert diagnostics["enp_v1"] == pytest.approx(trace_s)
    assert diagnostics["enp_v2"] == pytest.approx(expected_enp_v2)
    assert diagnostics["enp"] == pytest.approx(expected_enp_v2)
    assert diagnostics["edf_v2"] == pytest.approx(expected_edf_v2)
    assert diagnostics["edf"] == pytest.approx(expected_edf_v2)
