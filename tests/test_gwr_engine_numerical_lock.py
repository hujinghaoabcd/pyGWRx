# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""C2 blocking numerical locks for the extracted standard-GWR engine.

These tests intentionally exercise the private engine directly against the
already frozen external ``mgwr 2.2.1`` artifacts.  Public ``GWR`` reference
tests remain authoritative as well; this layer prevents a future public-shell
adapter from hiding drift inside the engine extracted during C1.

Author:
    Jinghao Hu
"""

from __future__ import annotations

__author__ = "Jinghao Hu"
__license__ = "MIT"

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from pygwrx.core.distance import _iter_distance_rows
from pygwrx.core.kernels import get_kernel_function
from pygwrx.core.utils import add_intercept
from pygwrx.models._gwr_engine import (
    _collect_gwr_inference,
    _compute_gwr_local_r2,
    _fit_gwr_prediction_locations,
    _fit_gwr_training_locations,
    _gwr_spatial_weights,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(__file__).parent / "reference_data" / "gwr"
CORE_ATOL = 2e-7
INFERENCE_ATOL = 1e-5
DEEP_DIAGNOSTIC_ATOL = 1e-6
PREDICTION_ATOL = 1e-8

_REQUIRED_GATE_TOKENS = {
    "deep hat/influence/Cook's D": (
        "tests/test_gwr_mgwr_deep_diagnostics_reference.py",
        "test_gwr_deep_diagnostics_match_mgwr",
    ),
    "rank-deficient behavior": (
        "tests/test_gwr_rank_inference.py",
        "test_rank_deficient_calibration_keeps_coefficients_but_masks_inference",
    ),
    "prediction": (
        "tests/test_gwr_external_references.py",
        "test_gwr_new_location_prediction_matches_external_references",
    ),
    "streaming": (
        "tests/test_gwr_distance_streaming.py",
        "test_numeric_bandwidth_fit_streams_calibration_and_local_r2_distances",
    ),
    "bandwidth provenance": (
        "tests/test_gwr_bandwidth_provenance.py",
        "test_gwr_retains_adaptive_bandwidth_search_provenance",
    ),
    "failed-refit atomicity": (
        "tests/test_estimator_fitted_state_atomicity.py",
        '"GWR": _FitCase(',
    ),
}


def _load_json(name: str) -> dict[str, Any]:
    return json.loads((DATA_DIR / name).read_text(encoding="utf-8"))


def _vector(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=float).reshape(-1)


def _training_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frame = pd.read_csv(DATA_DIR / "input.csv")
    X_design = add_intercept(frame[["x1", "x2"]].to_numpy(dtype=float))
    y = frame["response"].to_numpy(dtype=float)
    coords = frame[["x", "ycoord"]].to_numpy(dtype=float)
    return X_design, y, coords


def _engine_callbacks(
    coords_train: np.ndarray,
    *,
    kernel: str,
    bandwidth: float | int,
    adaptive: bool,
):
    kernel_func = get_kernel_function(kernel)

    def distance_rows(target_coords: np.ndarray):
        return _iter_distance_rows(
            np.asarray(target_coords, dtype=float),
            coords_train,
            distance_metric="euclidean",
        )

    def weights_from_distances(distances: np.ndarray) -> np.ndarray:
        return _gwr_spatial_weights(
            distances,
            bandwidth=bandwidth,
            adaptive=adaptive,
            kernel_func=kernel_func,
        )

    def rank_policy(
        rank_deficient: np.ndarray,
        *,
        context: str,
        n_parameters: int,
    ) -> None:
        assert not np.any(rank_deficient), (
            f"frozen reference unexpectedly became rank deficient in {context} "
            f"for {n_parameters} parameters"
        )

    return distance_rows, weights_from_distances, rank_policy


@pytest.mark.reference
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
def test_private_gwr_engine_calibration_matches_frozen_mgwr(
    case: str,
    kernel: str,
    bandwidth: float | int,
    adaptive: bool,
    sigma2_v1: bool,
) -> None:
    """Lock the C1 engine itself, not only the public estimator shell."""
    X_design, y, coords = _training_arrays()
    expected = _load_json("mgwr_2.2.1.json")["cases"][case]
    distance_rows, weights_from_distances, rank_policy = _engine_callbacks(
        coords,
        kernel=kernel,
        bandwidth=bandwidth,
        adaptive=adaptive,
    )

    local_fit = _fit_gwr_training_locations(
        X_design,
        y,
        coords,
        distance_rows=distance_rows,
        weights_from_distances=weights_from_distances,
        rank_policy=rank_policy,
        store_hat_matrix=True,
        compute_inference=True,
    )
    assert local_fit.hat_matrix is not None
    assert local_fit.covariance_factors is not None

    residuals = y - local_fit.fitted_values
    local_r2 = _compute_gwr_local_r2(
        y,
        residuals,
        local_fit.distances,
        weights_from_distances=weights_from_distances,
    )
    inference = _collect_gwr_inference(
        residuals,
        local_fit.influence,
        local_fit.params[:, 1:],
        local_fit.params[:, 0],
        local_fit.covariance_factors,
        n_samples=y.size,
        fit_intercept=True,
        sigma2_v1=sigma2_v1,
        trace_S=local_fit.trace_S,
        trace_StS=local_fit.trace_StS,
    )

    np.testing.assert_allclose(
        local_fit.params,
        np.asarray(expected["params"], dtype=float),
        rtol=1e-7,
        atol=CORE_ATOL,
    )
    np.testing.assert_allclose(
        local_fit.fitted_values,
        _vector(expected["predy"]),
        rtol=1e-7,
        atol=CORE_ATOL,
    )
    np.testing.assert_allclose(
        residuals,
        _vector(expected["residuals"]),
        rtol=1e-7,
        atol=CORE_ATOL,
    )
    np.testing.assert_allclose(
        local_r2,
        _vector(expected["local_r2"]),
        rtol=1e-7,
        atol=CORE_ATOL,
    )
    np.testing.assert_allclose(
        local_fit.influence,
        _vector(expected["influence"]),
        rtol=1e-6,
        atol=DEEP_DIAGNOSTIC_ATOL,
    )
    np.testing.assert_allclose(
        local_fit.hat_matrix,
        np.asarray(expected["hat_matrix"], dtype=float),
        rtol=1e-6,
        atol=DEEP_DIAGNOSTIC_ATOL,
    )
    np.testing.assert_allclose(
        inference.standardized_residuals,
        _vector(expected["standardized_residuals"]),
        rtol=1e-6,
        atol=DEEP_DIAGNOSTIC_ATOL,
    )
    np.testing.assert_allclose(
        inference.cooks_distance,
        _vector(expected["cooks_distance"]),
        rtol=1e-6,
        atol=DEEP_DIAGNOSTIC_ATOL,
    )
    np.testing.assert_allclose(
        inference.parameter_standard_errors,
        np.asarray(expected["bse"], dtype=float),
        rtol=1e-6,
        atol=INFERENCE_ATOL,
    )
    np.testing.assert_allclose(
        inference.parameter_t_values,
        np.asarray(expected["tvalues"], dtype=float),
        rtol=1e-6,
        atol=INFERENCE_ATOL,
    )
    assert inference.sigma2 == pytest.approx(
        expected["diagnostics"]["sigma2"],
        rel=1e-6,
        abs=INFERENCE_ATOL,
    )


@pytest.mark.reference
def test_private_gwr_engine_prediction_matches_frozen_mgwr() -> None:
    """Lock independent-location prediction through the private engine path."""
    X_design, y, coords = _training_arrays()
    prediction = pd.read_csv(DATA_DIR / "prediction.csv")
    X_new = prediction[["x1", "x2"]].to_numpy(dtype=float)
    coords_new = prediction[["x", "ycoord"]].to_numpy(dtype=float)
    expected = _load_json("mgwr_2.2.1.json")["fixed_gaussian_prediction"]
    distance_rows, weights_from_distances, rank_policy = _engine_callbacks(
        coords,
        kernel="gaussian",
        bandwidth=55.0,
        adaptive=False,
    )

    local_fit = _fit_gwr_prediction_locations(
        X_design,
        y,
        coords_new,
        distance_rows=distance_rows,
        weights_from_distances=weights_from_distances,
        rank_policy=rank_policy,
        compute_inference=False,
    )
    predictions = np.einsum(
        "ij,ij->i",
        X_new,
        local_fit.full_params[:, 1:],
    ) + local_fit.full_params[:, 0]

    np.testing.assert_allclose(
        local_fit.full_params,
        np.asarray(expected["params"], dtype=float),
        rtol=1e-8,
        atol=PREDICTION_ATOL,
    )
    np.testing.assert_allclose(
        predictions,
        _vector(expected["predictions"]),
        rtol=1e-8,
        atol=PREDICTION_ATOL,
    )


def test_c2_required_gwr_gate_inventory_and_reference_floor() -> None:
    """Prevent required C2 gates from being silently removed after the lock."""
    for label, (relative_path, token) in _REQUIRED_GATE_TOKENS.items():
        path = REPO_ROOT / relative_path
        assert path.is_file(), f"missing C2 {label} gate: {relative_path}"
        assert token in path.read_text(encoding="utf-8"), (
            f"missing C2 {label} contract token {token!r} in {relative_path}"
        )

    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "-m", "reference"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    reference_nodes = [
        line
        for line in completed.stdout.splitlines()
        if line.lstrip().startswith("tests") and "::" in line
    ]
    assert len(reference_nodes) >= 50, (
        f"C2 requires at least 50 blocking external-reference tests; "
        f"collected {len(reference_nodes)}"
    )
