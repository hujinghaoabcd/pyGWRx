"""Generate frozen GWR references with the independently maintained mgwr package."""

from __future__ import annotations

import json
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from mgwr.gwr import GWR as ReferenceGWR
from mgwr.sel_bw import Sel_BW

ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "tests" / "reference_data" / "gwr"
INPUT_PATH = DATA_DIR / "input.csv"
PREDICTION_PATH = DATA_DIR / "prediction.csv"
OUTPUT_PATH = DATA_DIR / "mgwr_2.2.1.json"

CASES = (
    {
        "name": "fixed_gaussian_v2",
        "kernel": "gaussian",
        "bandwidth": 55.0,
        "fixed": True,
        "sigma2_v1": False,
    },
    {
        "name": "fixed_bisquare_v2",
        "kernel": "bisquare",
        "bandwidth": 70.0,
        "fixed": True,
        "sigma2_v1": False,
    },
    {
        "name": "adaptive_gaussian_v2",
        "kernel": "gaussian",
        "bandwidth": 20,
        "fixed": False,
        "sigma2_v1": False,
    },
    {
        "name": "adaptive_bisquare_v2",
        "kernel": "bisquare",
        "bandwidth": 20,
        "fixed": False,
        "sigma2_v1": False,
    },
    {
        "name": "fixed_gaussian_v1",
        "kernel": "gaussian",
        "bandwidth": 55.0,
        "fixed": True,
        "sigma2_v1": True,
    },
)


def _array(value: Any) -> list[Any] | None:
    if value is None:
        return None
    return np.asarray(value, dtype=float).tolist()


def _scalar(value: Any) -> float | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=float)
    if array.size != 1:
        raise ValueError(f"Expected a scalar, got shape {array.shape}.")
    return float(array.reshape(-1)[0])


def _optional_attr(obj: Any, *names: str) -> Any:
    for name in names:
        if hasattr(obj, name):
            try:
                return getattr(obj, name)
            except (AttributeError, NotImplementedError):
                continue
    return None


def _fit_case(
    case: dict[str, Any],
    coords: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
) -> tuple[ReferenceGWR, Any, dict[str, Any]]:
    model = ReferenceGWR(
        coords,
        y,
        X,
        case["bandwidth"],
        fixed=case["fixed"],
        kernel=case["kernel"],
        constant=True,
        spherical=False,
        sigma2_v1=case["sigma2_v1"],
        hat_matrix=True,
    )
    result = model.fit()
    payload = {
        "config": case,
        "params": _array(result.params),
        "predy": _array(result.predy),
        "residuals": _array(result.resid_response),
        "local_r2": _array(result.localR2),
        "bse": _array(result.bse),
        "tvalues": _array(result.tvalues),
        "influence": _array(result.influ),
        "standardized_residuals": _array(result.std_res),
        "cooks_distance": _array(_optional_attr(result, "CooksD", "cooksD")),
        "hat_matrix": _array(_optional_attr(result, "S")),
        "diagnostics": {
            "trace_S": _scalar(_optional_attr(result, "tr_S")),
            "trace_StS": _scalar(_optional_attr(result, "tr_STS", "tr_StS")),
            "r2": _scalar(result.R2),
            "adj_r2": _scalar(result.adj_R2),
            "aic": _scalar(result.aic),
            "aicc": _scalar(result.aicc),
            "bic": _scalar(result.bic),
            "sigma2": _scalar(result.sigma2),
        },
    }
    return model, result, payload


def _prediction_payload(
    model: ReferenceGWR,
    result: Any,
    prediction_frame: pd.DataFrame,
) -> dict[str, Any]:
    points = prediction_frame[["x", "ycoord"]].to_numpy(dtype=float)
    predictors = prediction_frame[["x1", "x2"]].to_numpy(dtype=float)
    prediction_result = model.predict(
        points,
        predictors,
        exog_scale=result.scale,
        exog_resid=result.resid_response,
    )
    params = np.asarray(prediction_result.params, dtype=float)
    design = np.column_stack([np.ones(predictors.shape[0]), predictors])
    predictions = np.einsum("ij,ij->i", design, params)
    return {
        "coords": points.tolist(),
        "X": predictors.tolist(),
        "params": params.tolist(),
        "predictions": predictions.tolist(),
        "bse": _array(_optional_attr(prediction_result, "bse")),
        "tvalues": _array(_optional_attr(prediction_result, "tvalues")),
    }


def _selected_bandwidths(
    coords: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
) -> dict[str, float]:
    selected: dict[str, float] = {}
    for criterion in ("CV", "AIC", "AICc", "BIC"):
        selector = Sel_BW(
            coords,
            y,
            X,
            fixed=False,
            kernel="bisquare",
            constant=True,
            spherical=False,
        )
        selected[criterion.lower()] = float(
            selector.search(criterion=criterion, bw_min=8, bw_max=32)
        )
    return selected


def main() -> None:
    frame = pd.read_csv(INPUT_PATH)
    prediction_frame = pd.read_csv(PREDICTION_PATH)
    coords = frame[["x", "ycoord"]].to_numpy(dtype=float)
    X = frame[["x1", "x2"]].to_numpy(dtype=float)
    y = frame[["response"]].to_numpy(dtype=float)

    cases: dict[str, Any] = {}
    prediction_source: tuple[ReferenceGWR, Any] | None = None
    for case in CASES:
        model, result, payload = _fit_case(case, coords, X, y)
        cases[case["name"]] = payload
        if case["name"] == "fixed_gaussian_v2":
            prediction_source = (model, result)

    if prediction_source is None:
        raise RuntimeError("The fixed Gaussian prediction reference was not generated.")

    payload = {
        "generator": "tools/reference/gwr/generate_mgwr.py",
        "reference_package": "mgwr",
        "reference_version": version("mgwr"),
        "n_samples": int(frame.shape[0]),
        "features": ["x1", "x2"],
        "cases": cases,
        "adaptive_bisquare_bandwidth_selection": _selected_bandwidths(coords, X, y),
        "fixed_gaussian_prediction": _prediction_payload(
            prediction_source[0], prediction_source[1], prediction_frame
        ),
    }
    OUTPUT_PATH.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {OUTPUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
