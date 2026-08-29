#!/usr/bin/env python3
"""Generate controlled adaptive GWR bandwidth curves for Columbus real data."""

from __future__ import annotations

import json
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from mgwr.diagnostics import get_AIC, get_AICc, get_BIC, get_CV
from mgwr.gwr import GWR as ReferenceGWR

from pygwrx import GWR
from pygwrx.core.bandwidth import _fit_local_model, _kernel_weights
from pygwrx.core.kernels import get_kernel_function
from pygwrx.core.utils import add_intercept, compute_distance_matrix

ROOT = Path(__file__).resolve().parents[3]
SOURCE_PATH = ROOT / "src" / "pygwrx" / "data" / "Columbus" / "columbus.csv"
OUTPUT_DIR = ROOT / "tests" / "reference_data" / "gwr" / "real_columbus"
PYGWRX_OUTPUT = OUTPUT_DIR / "pygwrx_bandwidth_curve.json"
MGWR_OUTPUT = OUTPUT_DIR / "mgwr_bandwidth_curve.json"

# n=49, with three fitted design columns (intercept + INC + HOVAL).
# Preserve the complete mathematically admissible neighbour-order domain.  Low-k
# candidates that produce singular/saturated fits are retained as error/null
# points rather than silently clipped from the archive.
CANDIDATES = tuple(range(4, 50))


def _finite_or_none(value: Any) -> float | None:
    try:
        out = float(np.asarray(value, dtype=float).reshape(-1)[0])
    except (TypeError, ValueError, IndexError):
        return None
    return out if np.isfinite(out) else None


def _pygwrx_cv_sse(
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


def _pygwrx_curve(
    X_frame: pd.DataFrame,
    y_series: pd.Series,
    coords_frame: pd.DataFrame,
) -> dict[str, Any]:
    X = X_frame.to_numpy(dtype=float)
    y = y_series.to_numpy(dtype=float)
    coords = coords_frame.to_numpy(dtype=float)
    X_design = add_intercept(X)
    distances = np.asarray(
        compute_distance_matrix(coords, coords, metric="euclidean"), dtype=float
    )

    points: list[dict[str, Any]] = []
    for k in CANDIDATES:
        try:
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
            diagnostics = model.diagnostics_ or {}
            cv_sse = _pygwrx_cv_sse(X_design, y, distances, k)
            points.append(
                {
                    "k": k,
                    "cv_sse": cv_sse,
                    "cv_mean": cv_sse / y.size,
                    "aic": _finite_or_none(diagnostics.get("aic")),
                    "aicc": _finite_or_none(diagnostics.get("aicc")),
                    "bic": _finite_or_none(diagnostics.get("bic")),
                    "trace_S": _finite_or_none(diagnostics.get("trace_S")),
                    "status": "ok",
                }
            )
        except Exception as exc:  # pragma: no cover - generator diagnostic path
            points.append({"k": k, "status": "error", "error": repr(exc)})

    return {
        "implementation": "pyGWRx",
        "dataset": "Columbus (OH) neighborhood crime",
        "formula": "CRIME ~ INC + HOVAL",
        "candidate_semantics": "adaptive integer neighbour-order bandwidth",
        "kernel": "bisquare",
        "candidate_min": min(CANDIDATES),
        "candidate_max": max(CANDIDATES),
        "n_samples": int(y.size),
        "points": points,
    }


def _mgwr_curve(
    X: np.ndarray,
    y: np.ndarray,
    coords: np.ndarray,
) -> dict[str, Any]:
    points: list[dict[str, Any]] = []
    n = int(y.shape[0])
    for k in CANDIDATES:
        try:
            result = ReferenceGWR(
                coords,
                y,
                X,
                k,
                fixed=False,
                kernel="bisquare",
                constant=True,
                spherical=False,
                sigma2_v1=False,
                hat_matrix=False,
                n_jobs=1,
            ).fit(lite=True)
            cv_mean = _finite_or_none(get_CV(result))
            points.append(
                {
                    "k": k,
                    "cv_mean": cv_mean,
                    "cv_sse": None if cv_mean is None else cv_mean * n,
                    "aic": _finite_or_none(get_AIC(result)),
                    "aicc": _finite_or_none(get_AICc(result)),
                    "bic": _finite_or_none(get_BIC(result)),
                    "trace_S": _finite_or_none(getattr(result, "tr_S", None)),
                    "status": "ok",
                }
            )
        except Exception as exc:  # pragma: no cover - generator diagnostic path
            points.append({"k": k, "status": "error", "error": repr(exc)})

    return {
        "implementation": "mgwr",
        "reference_version": version("mgwr"),
        "dataset": "Columbus (OH) neighborhood crime",
        "formula": "CRIME ~ INC + HOVAL",
        "candidate_semantics": "adaptive integer nearest-neighbour bandwidth",
        "kernel": "bisquare",
        "candidate_min": min(CANDIDATES),
        "candidate_max": max(CANDIDATES),
        "n_samples": n,
        "cv_raw_semantics": "mean squared leave-one-out error",
        "cv_sse_normalization": "cv_mean * n_samples",
        "points": points,
    }


def main() -> None:
    frame = pd.read_csv(SOURCE_PATH)
    X_frame = frame[["INC", "HOVAL"]]
    y_series = frame["CRIME"]
    coords_frame = frame[["X", "Y"]]

    pygwrx_payload = _pygwrx_curve(X_frame, y_series, coords_frame)
    mgwr_payload = _mgwr_curve(
        X_frame.to_numpy(dtype=float),
        y_series.to_numpy(dtype=float).reshape(-1, 1),
        coords_frame.to_numpy(dtype=float),
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PYGWRX_OUTPUT.write_text(
        json.dumps(pygwrx_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    MGWR_OUTPUT.write_text(
        json.dumps(mgwr_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {PYGWRX_OUTPUT.relative_to(ROOT)}")
    print(f"wrote {MGWR_OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
