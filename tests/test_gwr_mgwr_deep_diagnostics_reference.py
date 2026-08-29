# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Blocking mgwr references for deep standard-GWR diagnostics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from pygwrx import GWR

pytestmark = pytest.mark.reference

DATA_DIR = Path(__file__).parent / "reference_data" / "gwr"
DEEP_DIAGNOSTIC_ATOL = 1e-6
DEEP_DIAGNOSTIC_RTOL = 1e-6


def _load_json(name: str) -> dict[str, Any]:
    return json.loads((DATA_DIR / name).read_text(encoding="utf-8"))


def _vector(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=float).reshape(-1)


@pytest.fixture(scope="module")
def gwr_data() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    frame = pd.read_csv(DATA_DIR / "input.csv")
    return frame[["x1", "x2"]], frame["response"], frame[["x", "ycoord"]]


@pytest.fixture(scope="module")
def mgwr_reference() -> dict[str, Any]:
    reference = _load_json("mgwr_2.2.1.json")
    assert reference["reference_version"] == "2.2.1"
    return reference


@pytest.mark.parametrize(
    ("case", "kernel", "bandwidth", "adaptive", "sigma2_v1"),
    [
        ("fixed_gaussian_v2", "gaussian", 55.0, False, False),
        ("fixed_bisquare_v2", "bisquare", 70.0, False, False),
        ("adaptive_gaussian_v2", "gaussian", 20, True, False),
        ("adaptive_bisquare_v2", "bisquare", 20, True, False),
        ("fixed_gaussian_v1", "gaussian", 55.0, False, True),
    ],
)
def test_gwr_deep_diagnostics_match_mgwr(
    gwr_data: tuple[pd.DataFrame, pd.Series, pd.DataFrame],
    mgwr_reference: dict[str, Any],
    case: str,
    kernel: str,
    bandwidth: float | int,
    adaptive: bool,
    sigma2_v1: bool,
) -> None:
    """Lock smoother/influence diagnostics that were previously report-only."""
    X, y, coords = gwr_data
    model = GWR(
        kernel=kernel,
        bandwidth=bandwidth,
        adaptive=adaptive,
        sigma2_v1=sigma2_v1,
        fit_intercept=True,
        distance_metric="euclidean",
    ).fit(
        X,
        y,
        coords,
        compute_hat_matrix=True,
        compute_local_r2=False,
        compute_inference=True,
    )
    expected = mgwr_reference["cases"][case]

    np.testing.assert_allclose(
        model.influence_,
        _vector(expected["influence"]),
        rtol=DEEP_DIAGNOSTIC_RTOL,
        atol=DEEP_DIAGNOSTIC_ATOL,
    )
    np.testing.assert_allclose(
        model.hat_matrix_,
        np.asarray(expected["hat_matrix"], dtype=float),
        rtol=DEEP_DIAGNOSTIC_RTOL,
        atol=DEEP_DIAGNOSTIC_ATOL,
    )
    np.testing.assert_allclose(
        model.standardized_residuals_,
        _vector(expected["standardized_residuals"]),
        rtol=DEEP_DIAGNOSTIC_RTOL,
        atol=DEEP_DIAGNOSTIC_ATOL,
    )
    np.testing.assert_allclose(
        model.cooks_distance_,
        _vector(expected["cooks_distance"]),
        rtol=DEEP_DIAGNOSTIC_RTOL,
        atol=DEEP_DIAGNOSTIC_ATOL,
    )

    # These identities ensure the archived smoother is not merely shape-compatible.
    np.testing.assert_allclose(
        np.diag(model.hat_matrix_), model.influence_, rtol=0.0, atol=1e-12
    )
    np.testing.assert_allclose(
        model.hat_matrix_ @ model.y_train_,
        model.fitted_values_,
        rtol=0.0,
        atol=1e-8,
    )
