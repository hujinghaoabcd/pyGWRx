#!/usr/bin/env python3
"""Compare PyGWRx GWR against frozen mgwr, GWmodel, and spgwr references."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pygwrx import GWR

ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "tests" / "reference_data" / "gwr"
OUT_DIR = ROOT / "validation_results" / "gwr"


def _load_json(name: str) -> dict[str, Any]:
    return json.loads((DATA_DIR / name).read_text(encoding="utf-8"))


def _array(value: Any) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 2 and 1 in arr.shape:
        arr = arr.reshape(-1)
    return arr


def _compare(actual: Any, reference: Any) -> dict[str, float]:
    a = _array(actual)
    r = _array(reference)
    if a.shape != r.shape:
        raise ValueError(f"shape mismatch: {a.shape} != {r.shape}")
    delta = np.abs(a - r)
    denom = np.maximum(np.abs(r), np.finfo(float).eps)
    rel = delta / denom
    return {
        "max_abs_diff": float(np.max(delta)),
        "mean_abs_diff": float(np.mean(delta)),
        "rmse": float(np.sqrt(np.mean((a - r) ** 2))),
        "max_rel_diff": float(np.max(rel)),
    }


def _full_params(model: GWR) -> np.ndarray:
    return np.column_stack([model.intercept_, model.coef_])


def _full_se(model: GWR) -> np.ndarray:
    return np.column_stack([model.intercept_se_, model.coef_se_])


def _full_t(model: GWR) -> np.ndarray:
    return np.column_stack([model.intercept_t_, model.coef_t_])


def _fit_case(
    X: pd.DataFrame,
    y: pd.Series,
    coords: pd.DataFrame,
    *,
    kernel: str,
    bandwidth: float | int,
    adaptive: bool,
    sigma2_v1: bool,
) -> GWR:
    return GWR(
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
        compute_local_r2=True,
        compute_inference=True,
    )


def _record(
    rows: list[dict[str, Any]],
    *,
    reference: str,
    version: str,
    case: str,
    metric: str,
    actual: Any,
    expected: Any,
    interpretation: str = "strict",
) -> None:
    stats = _compare(actual, expected)
    rows.append(
        {
            "reference": reference,
            "reference_version": version,
            "case": case,
            "metric": metric,
            "interpretation": interpretation,
            **stats,
        }
    )


def main() -> None:
    frame = pd.read_csv(DATA_DIR / "input.csv")
    pred_frame = pd.read_csv(DATA_DIR / "prediction.csv")
    X = frame[["x1", "x2"]]
    y = frame["response"]
    coords = frame[["x", "ycoord"]]
    X_new = pred_frame[["x1", "x2"]]
    coords_new = pred_frame[["x", "ycoord"]]

    mgwr = _load_json("mgwr_2.2.1.json")
    gwmodel = _load_json("GWmodel_reference.json")
    spgwr = _load_json("spgwr_reference.json")

    specs = {
        "fixed_gaussian_v2": dict(
            kernel="gaussian", bandwidth=55.0, adaptive=False, sigma2_v1=False
        ),
        "fixed_bisquare_v2": dict(
            kernel="bisquare", bandwidth=70.0, adaptive=False, sigma2_v1=False
        ),
        "adaptive_gaussian_v2": dict(
            kernel="gaussian", bandwidth=20, adaptive=True, sigma2_v1=False
        ),
        "adaptive_bisquare_v2": dict(
            kernel="bisquare", bandwidth=20, adaptive=True, sigma2_v1=False
        ),
        "fixed_gaussian_v1": dict(
            kernel="gaussian", bandwidth=55.0, adaptive=False, sigma2_v1=True
        ),
    }
    fits = {name: _fit_case(X, y, coords, **spec) for name, spec in specs.items()}
    rows: list[dict[str, Any]] = []

    for case in specs:
        ref = mgwr["cases"][case]
        model = fits[case]
        metrics = [
            ("params", _full_params(model)),
            ("fitted", model.fitted_values_),
            ("residuals", model.residuals_),
            ("local_r2", model.local_r2_),
            ("standard_errors", _full_se(model)),
            ("t_values", _full_t(model)),
            ("influence", model.influence_),
            ("hat_matrix", model.hat_matrix_),
            ("standardized_residuals", model.standardized_residuals_),
            ("cooks_distance", model.cooks_distance_),
        ]
        for metric, actual in metrics:
            ref_key = {
                "fitted": "predy",
                "standard_errors": "bse",
                "t_values": "tvalues",
            }.get(metric, metric)
            _record(
                rows,
                reference="mgwr",
                version=mgwr["reference_version"],
                case=case,
                metric=metric,
                actual=actual,
                expected=ref[ref_key],
            )
        md = model.diagnostics_
        rd = ref["diagnostics"]
        for metric in ["r2", "adj_r2", "aic", "aicc", "bic", "trace_S", "trace_StS"]:
            _record(
                rows,
                reference="mgwr",
                version=mgwr["reference_version"],
                case=case,
                metric=metric,
                actual=[md[metric]],
                expected=[rd[metric]],
            )
        _record(
            rows,
            reference="mgwr",
            version=mgwr["reference_version"],
            case=case,
            metric="sigma2",
            actual=[model.sigma2_],
            expected=[rd["sigma2"]],
        )

    for case in [
        "fixed_gaussian_v2",
        "fixed_bisquare_v2",
        "adaptive_gaussian_v2",
        "adaptive_bisquare_v2",
    ]:
        ref = gwmodel["cases"][case]
        model = fits[case]
        for metric, actual, expected in [
            ("params", _full_params(model), ref["params"]),
            ("fitted", model.fitted_values_, ref["predy"]),
            ("residuals", model.residuals_, ref["residuals"]),
            ("standard_errors", _full_se(model), ref["bse"]),
            ("t_values", _full_t(model), ref["tvalues"]),
            ("local_r2", model.local_r2_, ref["local_r2"]),
        ]:
            _record(
                rows,
                reference="GWmodel",
                version=gwmodel["reference_version"],
                case=case,
                metric=metric,
                actual=actual,
                expected=expected,
                interpretation=("different_definition" if metric == "local_r2" else "strict"),
            )
        md = model.diagnostics_
        rd = ref["diagnostics"]
        mapping = {
            "r2": "gw.R2",
            "adj_r2": "gwR2.adj",
            "aicc": "AICc",
            "enp_v2": "enp",
            "edf_v2": "edf",
        }
        for metric, ref_metric in mapping.items():
            _record(
                rows,
                reference="GWmodel",
                version=gwmodel["reference_version"],
                case=case,
                metric=metric,
                actual=[md[metric]],
                expected=[rd[ref_metric]],
            )
        for metric, ref_metric in [("aic", "AIC"), ("bic", "BIC")]:
            _record(
                rows,
                reference="GWmodel",
                version=gwmodel["reference_version"],
                case=case,
                metric=metric,
                actual=[md[metric]],
                expected=[rd[ref_metric]],
                interpretation="different_definition",
            )

    sp_pairs = [
        ("fixed_gaussian_v2", "fixed_gaussian", "strict"),
        ("fixed_bisquare_v2", "fixed_bisquare", "strict"),
        ("adaptive_gaussian_v2", "adaptive_gaussian", "different_adaptive_semantics"),
        ("adaptive_bisquare_v2", "adaptive_bisquare", "different_adaptive_semantics"),
    ]
    for py_case, ref_case, interpretation in sp_pairs:
        model = fits[py_case]
        ref = spgwr["cases"][ref_case]
        for metric, actual, expected in [
            ("params", _full_params(model), ref["params"]),
            ("fitted", model.fitted_values_, ref["predy"]),
            ("residuals", model.residuals_, ref["residuals"]),
            ("local_r2", model.local_r2_, ref["local_r2"]),
        ]:
            _record(
                rows,
                reference="spgwr",
                version=spgwr["reference_version"],
                case=py_case,
                metric=metric,
                actual=actual,
                expected=expected,
                interpretation=interpretation,
            )

    pred_model = fits["fixed_gaussian_v2"]
    pred = pred_model.predict_result(X_new, coords_new)
    pred_params = np.column_stack([pred.intercept, pred.coef])
    for reference, payload in [("mgwr", mgwr), ("GWmodel", gwmodel), ("spgwr", spgwr)]:
        ref = payload["fixed_gaussian_prediction"]
        _record(
            rows,
            reference=reference,
            version=payload["reference_version"],
            case="fixed_gaussian_prediction",
            metric="params",
            actual=pred_params,
            expected=ref["params"],
        )
        _record(
            rows,
            reference=reference,
            version=payload["reference_version"],
            case="fixed_gaussian_prediction",
            metric="predictions",
            actual=pred.predictions,
            expected=ref["predictions"],
        )

    bw_rows = []
    py_bw = {}
    for criterion in ["cv", "aic", "aicc", "bic"]:
        model = GWR(
            kernel="bisquare",
            bandwidth=criterion,
            adaptive=True,
            bandwidth_range=(5, 35),
            optimization_method="grid",
        ).fit(
            X,
            y,
            coords,
            compute_hat_matrix=False,
            compute_local_r2=False,
            compute_inference=False,
        )
        py_bw[criterion] = int(model.bandwidth_)
    for criterion, value in py_bw.items():
        bw_rows.append(
            {
                "implementation": "pyGWRx",
                "criterion": criterion,
                "bandwidth": value,
                "bandwidth_type": "adaptive_neighbours",
            }
        )
    for criterion, value in mgwr["adaptive_bisquare_bandwidth_selection"].items():
        bw_rows.append(
            {
                "implementation": "mgwr",
                "criterion": criterion,
                "bandwidth": value,
                "bandwidth_type": "adaptive_neighbours",
            }
        )
    for criterion, value in gwmodel["adaptive_bisquare_bandwidth_selection"].items():
        bw_rows.append(
            {
                "implementation": "GWmodel",
                "criterion": criterion,
                "bandwidth": value,
                "bandwidth_type": "adaptive_neighbours",
            }
        )
    for criterion, value in spgwr["fixed_bandwidth_selection"].items():
        bw_rows.append(
            {
                "implementation": "spgwr",
                "criterion": criterion,
                "bandwidth": value,
                "bandwidth_type": "fixed_distance",
            }
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "reference",
        "reference_version",
        "case",
        "metric",
        "interpretation",
        "max_abs_diff",
        "mean_abs_diff",
        "rmse",
        "max_rel_diff",
    ]
    with (OUT_DIR / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    (OUT_DIR / "comparison.json").write_text(
        json.dumps(rows, indent=2) + "\n", encoding="utf-8"
    )
    with (OUT_DIR / "bandwidth_selection.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["implementation", "criterion", "bandwidth", "bandwidth_type"],
        )
        writer.writeheader()
        writer.writerows(bw_rows)

    strict = [row for row in rows if row["interpretation"] == "strict"]
    by_ref: dict[str, list[dict[str, Any]]] = {}
    for row in strict:
        by_ref.setdefault(row["reference"], []).append(row)

    lines = [
        "# GWR External Reference Validation",
        "",
        "Generated from the deterministic 40-point calibration fixture and 5 independent prediction locations.",
        "",
        "## Strict like-for-like comparisons",
        "",
        "| Reference | Checks | Worst max absolute difference | Metric/case |",
        "|---|---:|---:|---|",
    ]
    for reference, group in by_ref.items():
        worst = max(group, key=lambda row: row["max_abs_diff"])
        lines.append(
            f"| {reference} | {len(group)} | {worst['max_abs_diff']:.6e} | "
            f"{worst['case']} / {worst['metric']} |"
        )
    lines.extend(
        [
            "",
            "## Known semantic differences",
            "",
            "- GWmodel Local_R2 is not numerically identical to the mgwr/spgwr/PyGWRx local weighted R² convention and is reported separately.",
            "- GWmodel AIC and BIC labels use formulas that differ from the RSS/trace(S) formulas used by PyGWRx/mgwr; AICc is directly comparable and is tested strictly.",
            "- spgwr adaptive bandwidth is supplied as a sample proportion and resolves local radii differently from integer-k adaptive bandwidths; adaptive results are therefore semantic cross-checks, not strict equality tests.",
            "- Bandwidth-selection outputs are retained in bandwidth_selection.csv rather than forced into a universal equality assertion.",
            "",
        ]
    )
    report = OUT_DIR / "gwr_validation_report.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print(report.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
