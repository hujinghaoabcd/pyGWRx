# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Independent numerical-reference tests for the basic GWR implementation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from pygwrx import GWR
from pygwrx.core.bandwidth import AICSelector, BICSelector, CrossValidationSelector
from pygwrx.core.kernels import get_kernel_function
from pygwrx.core.utils import add_intercept

pytestmark = pytest.mark.reference

DATA_DIR = Path(__file__).parent / "reference_data" / "gwr"

CORE_ATOL = 2e-7
INFERENCE_ATOL = 1e-5
DIAGNOSTIC_ATOL = 1e-5
PREDICTION_ATOL = 1e-8


def _load_json(name: str) -> dict[str, Any]:
    return json.loads((DATA_DIR / name).read_text(encoding="utf-8"))


def _vector(value: Any) -> np.ndarray:
    """Normalize reference vectors stored as either (n,) or (n, 1)."""
    return np.asarray(value, dtype=float).reshape(-1)


@pytest.fixture(scope="module")
def gwr_data() -> dict[str, Any]:
    frame = pd.read_csv(DATA_DIR / "input.csv")
    prediction = pd.read_csv(DATA_DIR / "prediction.csv")
    return {
        "X": frame[["x1", "x2"]],
        "y": frame["response"],
        "coords": frame[["x", "ycoord"]],
        "X_new": prediction[["x1", "x2"]],
        "coords_new": prediction[["x", "ycoord"]],
    }


@pytest.fixture(scope="module")
def references() -> dict[str, dict[str, Any]]:
    return {
        "mgwr": _load_json("mgwr_2.2.1.json"),
        "GWmodel": _load_json("GWmodel_reference.json"),
        "spgwr": _load_json("spgwr_reference.json"),
    }


@pytest.fixture(scope="module")
def fitted_models(gwr_data: dict[str, Any]) -> dict[str, GWR]:
    specs = {
        "fixed_gaussian_v2": {
            "kernel": "gaussian",
            "bandwidth": 55.0,
            "adaptive": False,
            "sigma2_v1": False,
        },
        "fixed_bisquare_v2": {
            "kernel": "bisquare",
            "bandwidth": 70.0,
            "adaptive": False,
            "sigma2_v1": False,
        },
        "adaptive_gaussian_v2": {
            "kernel": "gaussian",
            "bandwidth": 20,
            "adaptive": True,
            "sigma2_v1": False,
        },
        "adaptive_bisquare_v2": {
            "kernel": "bisquare",
            "bandwidth": 20,
            "adaptive": True,
            "sigma2_v1": False,
        },
        "fixed_gaussian_v1": {
            "kernel": "gaussian",
            "bandwidth": 55.0,
            "adaptive": False,
            "sigma2_v1": True,
        },
    }
    return {
        name: GWR(
            kernel=spec["kernel"],
            bandwidth=spec["bandwidth"],
            adaptive=spec["adaptive"],
            sigma2_v1=spec["sigma2_v1"],
            fit_intercept=True,
            distance_metric="euclidean",
        ).fit(
            gwr_data["X"],
            gwr_data["y"],
            gwr_data["coords"],
            compute_hat_matrix=True,
            compute_local_r2=True,
            compute_inference=True,
        )
        for name, spec in specs.items()
    }


def _params(model: GWR) -> np.ndarray:
    return np.column_stack([model.intercept_, model.coef_])


def _standard_errors(model: GWR) -> np.ndarray:
    return np.column_stack([model.intercept_se_, model.coef_se_])


def _t_values(model: GWR) -> np.ndarray:
    return np.column_stack([model.intercept_t_, model.coef_t_])


@pytest.mark.parametrize(
    ("implementation", "py_case", "reference_case"),
    [
        ("mgwr", "fixed_gaussian_v2", "fixed_gaussian_v2"),
        ("mgwr", "fixed_bisquare_v2", "fixed_bisquare_v2"),
        ("mgwr", "adaptive_gaussian_v2", "adaptive_gaussian_v2"),
        ("mgwr", "adaptive_bisquare_v2", "adaptive_bisquare_v2"),
        ("mgwr", "fixed_gaussian_v1", "fixed_gaussian_v1"),
        ("GWmodel", "fixed_gaussian_v2", "fixed_gaussian_v2"),
        ("GWmodel", "fixed_bisquare_v2", "fixed_bisquare_v2"),
        ("GWmodel", "adaptive_gaussian_v2", "adaptive_gaussian_v2"),
        ("GWmodel", "adaptive_bisquare_v2", "adaptive_bisquare_v2"),
        ("spgwr", "fixed_gaussian_v2", "fixed_gaussian"),
        ("spgwr", "fixed_bisquare_v2", "fixed_bisquare"),
    ],
)
def test_gwr_calibration_matches_external_references(
    fitted_models: dict[str, GWR],
    references: dict[str, dict[str, Any]],
    implementation: str,
    py_case: str,
    reference_case: str,
) -> None:
    model = fitted_models[py_case]
    reference = references[implementation]["cases"][reference_case]

    np.testing.assert_allclose(
        _params(model), reference["params"], rtol=1e-7, atol=CORE_ATOL
    )
    np.testing.assert_allclose(
        model.fitted_values_, _vector(reference["predy"]), rtol=1e-7, atol=CORE_ATOL
    )
    np.testing.assert_allclose(
        model.residuals_,
        _vector(reference["residuals"]),
        rtol=1e-7,
        atol=CORE_ATOL,
    )

    # GWmodel's Local_R2 uses a different convention and is intentionally excluded.
    if implementation != "GWmodel":
        np.testing.assert_allclose(
            model.local_r2_,
            _vector(reference["local_r2"]),
            rtol=1e-7,
            atol=CORE_ATOL,
        )


@pytest.mark.parametrize(
    ("implementation", "case"),
    [
        ("mgwr", "fixed_gaussian_v2"),
        ("mgwr", "fixed_bisquare_v2"),
        ("mgwr", "adaptive_gaussian_v2"),
        ("mgwr", "adaptive_bisquare_v2"),
        ("mgwr", "fixed_gaussian_v1"),
        ("GWmodel", "fixed_gaussian_v2"),
        ("GWmodel", "fixed_bisquare_v2"),
        ("GWmodel", "adaptive_gaussian_v2"),
        ("GWmodel", "adaptive_bisquare_v2"),
    ],
)
def test_gwr_inference_matches_external_references(
    fitted_models: dict[str, GWR],
    references: dict[str, dict[str, Any]],
    implementation: str,
    case: str,
) -> None:
    model = fitted_models[case]
    reference = references[implementation]["cases"][case]

    np.testing.assert_allclose(
        _standard_errors(model),
        reference["bse"],
        rtol=1e-6,
        atol=INFERENCE_ATOL,
    )
    np.testing.assert_allclose(
        _t_values(model),
        reference["tvalues"],
        rtol=1e-6,
        atol=INFERENCE_ATOL,
    )


@pytest.mark.parametrize(
    "case",
    [
        "fixed_gaussian_v2",
        "fixed_bisquare_v2",
        "adaptive_gaussian_v2",
        "adaptive_bisquare_v2",
    ],
)
def test_gwr_mgwr_diagnostics_match(
    fitted_models: dict[str, GWR],
    references: dict[str, dict[str, Any]],
    case: str,
) -> None:
    model = fitted_models[case]
    expected = references["mgwr"]["cases"][case]["diagnostics"]
    actual = model.diagnostics_
    assert actual is not None

    for metric in ("r2", "adj_r2", "aic", "aicc", "bic", "trace_S", "trace_StS"):
        assert actual[metric] == pytest.approx(
            expected[metric], rel=1e-6, abs=DIAGNOSTIC_ATOL
        )
    assert model.sigma2_ == pytest.approx(
        expected["sigma2"], rel=1e-6, abs=DIAGNOSTIC_ATOL
    )


@pytest.mark.parametrize(
    "case",
    [
        "fixed_gaussian_v2",
        "fixed_bisquare_v2",
        "adaptive_gaussian_v2",
        "adaptive_bisquare_v2",
    ],
)
def test_gwr_gwmodel_aicc_and_effective_parameters_match(
    fitted_models: dict[str, GWR],
    references: dict[str, dict[str, Any]],
    case: str,
) -> None:
    model = fitted_models[case]
    expected = references["GWmodel"]["cases"][case]["diagnostics"]
    actual = model.diagnostics_
    assert actual is not None

    mapping = {
        "r2": "gw.R2",
        "adj_r2": "gwR2.adj",
        "aicc": "AICc",
        "enp_v2": "enp",
        "edf_v2": "edf",
    }
    for actual_key, reference_key in mapping.items():
        assert actual[actual_key] == pytest.approx(
            expected[reference_key], rel=1e-6, abs=DIAGNOSTIC_ATOL
        )


def test_gwr_sigma2_v1_matches_mgwr_but_adjusted_r2_is_not_forced_equal(
    fitted_models: dict[str, GWR],
    references: dict[str, dict[str, Any]],
) -> None:
    model = fitted_models["fixed_gaussian_v1"]
    expected = references["mgwr"]["cases"]["fixed_gaussian_v1"]["diagnostics"]
    actual = model.diagnostics_
    assert actual is not None

    assert model.sigma2_ == pytest.approx(
        expected["sigma2"], rel=1e-6, abs=DIAGNOSTIC_ATOL
    )
    # mgwr changes its ENP convention when sigma2_v1=True; PyGWRx keeps the
    # diagnostics EDF convention explicit and stable. Preserve that distinction.
    assert abs(actual["adj_r2"] - expected["adj_r2"]) > 1e-5


@pytest.mark.parametrize("implementation", ["mgwr", "GWmodel", "spgwr"])
def test_gwr_new_location_prediction_matches_external_references(
    fitted_models: dict[str, GWR],
    references: dict[str, dict[str, Any]],
    gwr_data: dict[str, Any],
    implementation: str,
) -> None:
    result = fitted_models["fixed_gaussian_v2"].predict_result(
        gwr_data["X_new"], gwr_data["coords_new"]
    )
    expected = references[implementation]["fixed_gaussian_prediction"]

    np.testing.assert_allclose(
        np.column_stack([result.intercept, result.coef]),
        expected["params"],
        rtol=1e-8,
        atol=PREDICTION_ATOL,
    )
    np.testing.assert_allclose(
        result.predictions,
        _vector(expected["predictions"]),
        rtol=1e-8,
        atol=PREDICTION_ATOL,
    )


def test_gwr_spgwr_fixed_tricube_matches(
    gwr_data: dict[str, Any],
    references: dict[str, dict[str, Any]],
) -> None:
    model = GWR(
        kernel="tricube",
        bandwidth=70.0,
        adaptive=False,
        sigma2_v1=False,
    ).fit(gwr_data["X"], gwr_data["y"], gwr_data["coords"])
    reference = references["spgwr"]["cases"]["fixed_tricube"]

    np.testing.assert_allclose(
        _params(model), reference["params"], rtol=1e-7, atol=CORE_ATOL
    )
    np.testing.assert_allclose(
        model.fitted_values_,
        _vector(reference["predy"]),
        rtol=1e-7,
        atol=CORE_ATOL,
    )
    np.testing.assert_allclose(
        model.residuals_,
        _vector(reference["residuals"]),
        rtol=1e-7,
        atol=CORE_ATOL,
    )
    np.testing.assert_allclose(
        model.local_r2_,
        _vector(reference["local_r2"]),
        rtol=1e-7,
        atol=CORE_ATOL,
    )


def test_controlled_adaptive_bandwidth_argmins_match_external_references(
    gwr_data: dict[str, Any],
) -> None:
    X_design = add_intercept(gwr_data["X"].to_numpy(dtype=float))
    y = gwr_data["y"].to_numpy(dtype=float)
    coords = gwr_data["coords"].to_numpy(dtype=float)
    kernel = get_kernel_function("bisquare")

    cv = CrossValidationSelector(
        n_intervals=100,
        adaptive=True,
        optimization_method="grid",
    ).select(X_design, y, coords, kernel, bandwidth_range=(6, 40))
    aic = AICSelector(
        n_intervals=100,
        corrected=False,
        adaptive=True,
        optimization_method="grid",
    ).select(X_design, y, coords, kernel, bandwidth_range=(5, 40))
    aicc = AICSelector(
        n_intervals=100,
        corrected=True,
        adaptive=True,
        optimization_method="grid",
    ).select(X_design, y, coords, kernel, bandwidth_range=(5, 40))
    bic = BICSelector(
        n_intervals=100,
        adaptive=True,
        optimization_method="grid",
    ).select(X_design, y, coords, kernel, bandwidth_range=(5, 40))

    # Shared-candidate external validation gives these same minima:
    # CV: PyGWRx = mgwr = GWmodel = 15
    # AIC: PyGWRx = mgwr = 5
    # AICc: PyGWRx = mgwr = GWmodel = 22
    # BIC: PyGWRx = mgwr = 5
    assert cv == 15
    assert aic == 5
    assert aicc == 22
    assert bic == 5


def test_aicc_rejects_saturated_k4_boundary(gwr_data: dict[str, Any]) -> None:
    model = GWR(
        kernel="bisquare",
        bandwidth=4,
        adaptive=True,
        sigma2_v1=False,
    ).fit(
        gwr_data["X"],
        gwr_data["y"],
        gwr_data["coords"],
        compute_local_r2=False,
        compute_inference=False,
    )
    diagnostics = model.diagnostics_
    assert diagnostics is not None
    assert np.isinf(diagnostics["aicc"])
