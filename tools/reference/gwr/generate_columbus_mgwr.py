#!/usr/bin/env python3
"""Generate mgwr references for the real Columbus GWR validation dataset."""

from __future__ import annotations

import json
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from mgwr.gwr import GWR as ReferenceGWR

ROOT = Path(__file__).resolve().parents[3]
SOURCE_PATH = ROOT / "src" / "pygwrx" / "data" / "Columbus" / "columbus.csv"
OUTPUT_DIR = ROOT / "tests" / "reference_data" / "gwr" / "real_columbus"
OUTPUT_PATH = OUTPUT_DIR / "mgwr_2.2.1.json"

# These cases intentionally cover both fixed/adaptive and Gaussian/bisquare
# semantics on the same real dataset.  The adaptive k=24 case matches the
# public Columbus example in pyGWRx.
CASES = (
    {
        "name": "fixed_gaussian_v2",
        "kernel": "gaussian",
        "bandwidth": 10.0,
        "fixed": True,
        "sigma2_v1": False,
    },
    {
        "name": "fixed_bisquare_v2",
        "kernel": "bisquare",
        "bandwidth": 15.0,
        "fixed": True,
        "sigma2_v1": False,
    },
    {
        "name": "adaptive_gaussian_v2",
        "kernel": "gaussian",
        "bandwidth": 24,
        "fixed": False,
        "sigma2_v1": False,
    },
    {
        "name": "adaptive_bisquare_v2",
        "kernel": "bisquare",
        "bandwidth": 24,
        "fixed": False,
        "sigma2_v1": False,
    },
    {
        "name": "fixed_gaussian_v1",
        "kernel": "gaussian",
        "bandwidth": 10.0,
        "fixed": True,
        "sigma2_v1": True,
    },
)

# Five geographically dispersed rows, selected deterministically from the
# published 49-row ordering.  They are excluded from calibration before
# prediction, so this is a genuine held-out real-data prediction check.
HOLDOUT_ROWS = (0, 10, 20, 30, 40)


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
        n_jobs=1,
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


def _holdout_prediction(frame: pd.DataFrame) -> dict[str, Any]:
    holdout = frame.iloc[list(HOLDOUT_ROWS)].copy()
    training = frame.drop(frame.index[list(HOLDOUT_ROWS)]).copy()

    train_coords = training[["X", "Y"]].to_numpy(dtype=float)
    train_X = training[["INC", "HOVAL"]].to_numpy(dtype=float)
    train_y = training[["CRIME"]].to_numpy(dtype=float)
    holdout_coords = holdout[["X", "Y"]].to_numpy(dtype=float)
    holdout_X = holdout[["INC", "HOVAL"]].to_numpy(dtype=float)

    model = ReferenceGWR(
        train_coords,
        train_y,
        train_X,
        10.0,
        fixed=True,
        kernel="gaussian",
        constant=True,
        spherical=False,
        sigma2_v1=False,
        hat_matrix=True,
        n_jobs=1,
    )
    result = model.fit()
    prediction_result = model.predict(
        holdout_coords,
        holdout_X,
        exog_scale=result.scale,
        exog_resid=result.resid_response,
    )
    params = np.asarray(prediction_result.params, dtype=float)
    design = np.column_stack([np.ones(holdout_X.shape[0]), holdout_X])
    predictions = np.einsum("ij,ij->i", design, params)

    return {
        "config": {
            "kernel": "gaussian",
            "bandwidth": 10.0,
            "fixed": True,
            "sigma2_v1": False,
        },
        "holdout_rows_zero_based": list(HOLDOUT_ROWS),
        "holdout_polyid": holdout["POLYID"].astype(int).tolist(),
        "actual_response": holdout["CRIME"].astype(float).tolist(),
        "coords": holdout_coords.tolist(),
        "X": holdout_X.tolist(),
        "params": params.tolist(),
        "predictions": predictions.tolist(),
        "bse": _array(_optional_attr(prediction_result, "bse")),
        "tvalues": _array(_optional_attr(prediction_result, "tvalues")),
        "n_training": int(training.shape[0]),
        "n_holdout": int(holdout.shape[0]),
    }


def main() -> None:
    frame = pd.read_csv(SOURCE_PATH)
    coords = frame[["X", "Y"]].to_numpy(dtype=float)
    X = frame[["INC", "HOVAL"]].to_numpy(dtype=float)
    y = frame[["CRIME"]].to_numpy(dtype=float)

    cases: dict[str, Any] = {}
    for case in CASES:
        _, _, payload = _fit_case(case, coords, X, y)
        cases[case["name"]] = payload

    payload = {
        "generator": "tools/reference/gwr/generate_columbus_mgwr.py",
        "reference_package": "mgwr",
        "reference_version": version("mgwr"),
        "dataset": "Columbus (OH) neighborhood crime",
        "dataset_source": "src/pygwrx/data/Columbus/columbus.csv",
        "formula": "CRIME ~ INC + HOVAL",
        "n_samples": int(frame.shape[0]),
        "features": ["INC", "HOVAL"],
        "response": "CRIME",
        "coords": ["X", "Y"],
        "cases": cases,
        "held_out_fixed_gaussian_prediction": _holdout_prediction(frame),
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {OUTPUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
