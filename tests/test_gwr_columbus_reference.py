"""Real-data external GWR validation on the Columbus benchmark."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from pygwrx import GWR
from pygwrx.core.bandwidth import _fit_local_model, _kernel_weights
from pygwrx.core.kernels import get_kernel_function
from pygwrx.core.utils import add_intercept, compute_distance_matrix

pytestmark = pytest.mark.reference

ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = ROOT / "src" / "pygwrx" / "data" / "Columbus" / "columbus.csv"
REFERENCE_DIR = (
    ROOT / "tests" / "reference_data" / "gwr" / "real_columbus" / "frozen"
)
HOLDOUT_ROWS = (0, 10, 20, 30, 40)

SPECS: dict[str, dict[str, Any]] = {
    "fixed_gaussian_v2": {
        "kernel": "gaussian",
        "bandwidth": 10.0,
        "adaptive": False,
        "sigma2_v1": False,
    },
    "fixed_bisquare_v2": {
        "kernel": "bisquare",
        "bandwidth": 15.0,
        "adaptive": False,
        "sigma2_v1": False,
    },
    "adaptive_gaussian_v2": {
        "kernel": "gaussian",
        "bandwidth": 24,
        "adaptive": True,
        "sigma2_v1": False,
    },
    "adaptive_bisquare_v2": {
        "kernel": "bisquare",
        "bandwidth": 24,
        "adaptive": True,
        "sigma2_v1": False,
    },
}


def _load(name: str) -> dict[str, Any]:
    return json.loads((REFERENCE_DIR / name).read_text(encoding="utf-8"))


def _array(value: Any) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 2 and 1 in arr.shape:
        return arr.reshape(-1)
    return arr


def _full_params(model: GWR) -> np.ndarray:
    return np.column_stack([model.intercept_, model.coef_])


def _fit(
    X: pd.DataFrame,
    y: pd.Series,
    coords: pd.DataFrame,
    **spec: Any,
) -> GWR:
    return GWR(
        **spec,
        fit_intercept=True,
        distance_metric="euclidean",
    ).fit(
        X,
        y,
        coords,
        compute_hat_matrix=True,
        compute_local_r2=True,
        compute_inference=True,
    )


@pytest.fixture(scope="module")
def columbus_frame() -> pd.DataFrame:
    return pd.read_csv(SOURCE_PATH)


@pytest.fixture(scope="module")
def calibration_fits(columbus_frame: pd.DataFrame) -> dict[str, GWR]:
    X = columbus_frame[["INC", "HOVAL"]]
    y = columbus_frame["CRIME"]
    coords = columbus_frame[["X", "Y"]]
    return {name: _fit(X, y, coords, **spec) for name, spec in SPECS.items()}


def test_columbus_fixed_calibration_matches_three_external_implementations(
    calibration_fits: dict[str, GWR],
) -> None:
    mgwr = _load("mgwr.json")
    gwmodel = _load("GWmodel.json")
    spgwr = _load("spgwr.json")

    for py_case, sp_case in [
        ("fixed_gaussian_v2", "fixed_gaussian"),
        ("fixed_bisquare_v2", "fixed_bisquare"),
    ]:
        model = calibration_fits[py_case]
        params = _full_params(model)
        fitted = _array(model.fitted_values_)

        np.testing.assert_allclose(
            params,
            _array(mgwr["cases"][py_case]["params"]),
            rtol=0.0,
            atol=3e-6,
        )
        np.testing.assert_allclose(
            fitted,
            _array(mgwr["cases"][py_case]["predy"]),
            rtol=0.0,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            params,
            _array(gwmodel["cases"][py_case]["params"]),
            rtol=0.0,
            atol=3e-6,
        )
        np.testing.assert_allclose(
            fitted,
            _array(gwmodel["cases"][py_case]["predy"]),
            rtol=0.0,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            params,
            _array(spgwr["cases"][sp_case]["params"]),
            rtol=0.0,
            atol=3e-6,
        )
        np.testing.assert_allclose(
            fitted,
            _array(spgwr["cases"][sp_case]["predy"]),
            rtol=0.0,
            atol=1e-6,
        )


def test_columbus_adaptive_calibration_matches_mgwr_and_gwmodel(
    calibration_fits: dict[str, GWR],
) -> None:
    mgwr = _load("mgwr.json")
    gwmodel = _load("GWmodel.json")

    for case in ["adaptive_gaussian_v2", "adaptive_bisquare_v2"]:
        model = calibration_fits[case]
        params = _full_params(model)
        fitted = _array(model.fitted_values_)

        np.testing.assert_allclose(
            params,
            _array(mgwr["cases"][case]["params"]),
            rtol=0.0,
            atol=2e-5,
        )
        np.testing.assert_allclose(
            fitted,
            _array(mgwr["cases"][case]["predy"]),
            rtol=0.0,
            atol=5e-6,
        )
        np.testing.assert_allclose(
            params,
            _array(gwmodel["cases"][case]["params"]),
            rtol=0.0,
            atol=3e-6,
        )
        np.testing.assert_allclose(
            fitted,
            _array(gwmodel["cases"][case]["predy"]),
            rtol=0.0,
            atol=1e-6,
        )


def test_columbus_holdout_prediction_matches_three_external_implementations(
    columbus_frame: pd.DataFrame,
) -> None:
    holdout = columbus_frame.iloc[list(HOLDOUT_ROWS)]
    training = columbus_frame.drop(columbus_frame.index[list(HOLDOUT_ROWS)])
    model = _fit(
        training[["INC", "HOVAL"]],
        training["CRIME"],
        training[["X", "Y"]],
        kernel="gaussian",
        bandwidth=10.0,
        adaptive=False,
        sigma2_v1=False,
    )
    result = model.predict_result(
        holdout[["INC", "HOVAL"]],
        holdout[["X", "Y"]],
    )
    params = np.column_stack([result.intercept, result.coef])

    for name in ["mgwr.json", "GWmodel.json", "spgwr.json"]:
        ref = _load(name)["held_out_fixed_gaussian_prediction"]
        np.testing.assert_allclose(
            params,
            _array(ref["params"]),
            rtol=0.0,
            atol=5e-7,
        )
        np.testing.assert_allclose(
            result.predictions,
            _array(ref["predictions"]),
            rtol=0.0,
            atol=3e-7,
        )


def _cv_sse(
    X_design: np.ndarray,
    y: np.ndarray,
    distances: np.ndarray,
    k: int,
) -> float:
    kernel = get_kernel_function("bisquare")
    squared_error = 0.0
    for i, dists in enumerate(distances):
        weights = _kernel_weights(
            dists,
            k,
            adaptive=True,
            kernel_func=kernel,
        ).copy()
        weights[i] = 0.0
        beta, _ = _fit_local_model(X_design, y, weights)
        residual = float(y[i] - X_design[i] @ beta)
        squared_error += residual * residual
    return float(squared_error)


def test_columbus_adaptive_bandwidth_argmins_are_stable(
    columbus_frame: pd.DataFrame,
) -> None:
    X_frame = columbus_frame[["INC", "HOVAL"]]
    y_series = columbus_frame["CRIME"]
    coords_frame = columbus_frame[["X", "Y"]]
    X_design = add_intercept(X_frame.to_numpy(dtype=float))
    y = y_series.to_numpy(dtype=float)
    coords = coords_frame.to_numpy(dtype=float)
    distances = np.asarray(
        compute_distance_matrix(coords, coords, metric="euclidean"),
        dtype=float,
    )

    cv_sse: dict[int, float] = {}
    aicc: dict[int, float] = {}
    for k in range(5, 50):
        cv_sse[k] = _cv_sse(X_design, y, distances, k)
        model = GWR(
            kernel="bisquare",
            bandwidth=k,
            adaptive=True,
            sigma2_v1=False,
            fit_intercept=True,
            distance_metric="euclidean",
        ).fit(
            X_frame,
            y_series,
            coords_frame,
            compute_hat_matrix=False,
            compute_local_r2=False,
            compute_inference=False,
        )
        aicc[k] = float((model.diagnostics_ or {})["aicc"])

    assert min(cv_sse, key=cv_sse.get) == 11
    assert min(aicc, key=aicc.get) == 24


def test_columbus_near_saturated_boundary_is_preserved() -> None:
    summary = _load("bandwidth_summary.json")
    boundary = summary["near_saturated_boundary"]
    assert boundary["k"] == 4
    assert boundary["pygwrx_trace_S"] > 48.99
    assert summary["criteria"]["cv_sse"]["pygwrx_raw_argmin"] == 11
    assert summary["criteria"]["aicc"]["pygwrx_k_ge_5_argmin"] == 24


def test_columbus_reference_provenance_and_adaptive_spgwr_boundary() -> None:
    assert _load("mgwr.json")["reference_version"] == "2.2.1"
    assert _load("GWmodel.json")["reference_version"] == "2.4.1"

    spgwr = _load("spgwr.json")
    assert spgwr["reference_version"]
    notes = str(spgwr.get("notes", "")).lower()
    assert "q" in notes or "proportion" in notes
