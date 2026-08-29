#!/usr/bin/env python3
"""Compare pyGWRx GWR with external implementations on the real Columbus dataset."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pygwrx import GWR

ROOT = Path(__file__).resolve().parents[3]
SOURCE_PATH = ROOT / "src" / "pygwrx" / "data" / "Columbus" / "columbus.csv"
DATA_DIR = ROOT / "tests" / "reference_data" / "gwr" / "real_columbus"
OUT_DIR = ROOT / "validation_results" / "gwr" / "real_columbus"
HOLDOUT_ROWS = (0, 10, 20, 30, 40)

SPECS: dict[str, dict[str, Any]] = {
    "fixed_gaussian_v2": dict(
        kernel="gaussian", bandwidth=10.0, adaptive=False, sigma2_v1=False
    ),
    "fixed_bisquare_v2": dict(
        kernel="bisquare", bandwidth=15.0, adaptive=False, sigma2_v1=False
    ),
    "adaptive_gaussian_v2": dict(
        kernel="gaussian", bandwidth=24, adaptive=True, sigma2_v1=False
    ),
    "adaptive_bisquare_v2": dict(
        kernel="bisquare", bandwidth=24, adaptive=True, sigma2_v1=False
    ),
    "fixed_gaussian_v1": dict(
        kernel="gaussian", bandwidth=10.0, adaptive=False, sigma2_v1=True
    ),
}


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _array(value: Any) -> np.ndarray:
    if value is None:
        raise ValueError("Reference value is null.")
    raw = np.asarray(value, dtype=object)

    def convert(item: Any) -> float:
        if isinstance(item, str) and item.upper() == "NA":
            return np.nan
        return float(item)

    arr = np.vectorize(convert, otypes=[float])(raw)
    if arr.ndim == 2 and 1 in arr.shape:
        arr = arr.reshape(-1)
    return arr


def _compare(actual: Any, reference: Any) -> dict[str, float]:
    a = _array(actual)
    r = _array(reference)
    if a.shape != r.shape:
        raise ValueError(f"shape mismatch: {a.shape} != {r.shape}")
    mask = np.isfinite(a) & np.isfinite(r)
    if not np.any(mask):
        raise ValueError("No finite overlapping values for comparison.")
    delta = np.abs(a[mask] - r[mask])
    denom = np.maximum(np.abs(r[mask]), np.finfo(float).eps)
    rel = delta / denom
    return {
        "n_compared": int(mask.sum()),
        "max_abs_diff": float(np.max(delta)),
        "mean_abs_diff": float(np.mean(delta)),
        "rmse": float(np.sqrt(np.mean((a[mask] - r[mask]) ** 2))),
        "max_rel_diff": float(np.max(rel)),
    }


def _full_params(model: GWR) -> np.ndarray:
    return np.column_stack([model.intercept_, model.coef_])


def _full_se(model: GWR) -> np.ndarray:
    return np.column_stack([model.intercept_se_, model.coef_se_])


def _full_t(model: GWR) -> np.ndarray:
    return np.column_stack([model.intercept_t_, model.coef_t_])


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
    rows.append(
        {
            "reference": reference,
            "reference_version": version,
            "case": case,
            "metric": metric,
            "interpretation": interpretation,
            **_compare(actual, expected),
        }
    )


def _calibration_comparisons(
    frame: pd.DataFrame,
    mgwr: dict[str, Any],
    gwmodel: dict[str, Any],
    spgwr: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, GWR]]:
    X = frame[["INC", "HOVAL"]]
    y = frame["CRIME"]
    coords = frame[["X", "Y"]]
    fits = {name: _fit(X, y, coords, **spec) for name, spec in SPECS.items()}
    rows: list[dict[str, Any]] = []

    for case, model in fits.items():
        ref = mgwr["cases"][case]
        for metric, actual, ref_key in [
            ("params", _full_params(model), "params"),
            ("fitted", model.fitted_values_, "predy"),
            ("residuals", model.residuals_, "residuals"),
            ("local_r2", model.local_r2_, "local_r2"),
            ("standard_errors", _full_se(model), "bse"),
            ("t_values", _full_t(model), "tvalues"),
            ("influence", model.influence_, "influence"),
            ("hat_matrix", model.hat_matrix_, "hat_matrix"),
            (
                "standardized_residuals",
                model.standardized_residuals_,
                "standardized_residuals",
            ),
            ("cooks_distance", model.cooks_distance_, "cooks_distance"),
        ]:
            _record(
                rows,
                reference="mgwr",
                version=mgwr["reference_version"],
                case=case,
                metric=metric,
                actual=actual,
                expected=ref[ref_key],
            )

        md = model.diagnostics_ or {}
        rd = ref["diagnostics"]
        for metric in [
            "r2",
            "adj_r2",
            "aic",
            "aicc",
            "bic",
            "trace_S",
            "trace_StS",
        ]:
            _record(
                rows,
                reference="mgwr",
                version=mgwr["reference_version"],
                case=case,
                metric=metric,
                actual=[md[metric]],
                expected=[rd[metric]],
                interpretation=(
                    "different_enp_convention"
                    if case == "fixed_gaussian_v1" and metric == "adj_r2"
                    else "strict"
                ),
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
        model = fits[case]
        ref = gwmodel["cases"][case]
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
                interpretation=(
                    "different_definition" if metric == "local_r2" else "strict"
                ),
            )

        md = model.diagnostics_ or {}
        rd = ref["diagnostics"]
        for metric, ref_metric in {
            "r2": "gw.R2",
            "adj_r2": "gwR2.adj",
            "aicc": "AICc",
            "enp_v2": "enp",
            "edf_v2": "edf",
        }.items():
            if ref_metric in rd:
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
            if ref_metric in rd:
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
        (
            "adaptive_gaussian_v2",
            "adaptive_gaussian",
            "different_adaptive_semantics",
        ),
        (
            "adaptive_bisquare_v2",
            "adaptive_bisquare",
            "different_adaptive_semantics",
        ),
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

    return rows, fits


def _holdout_comparisons(
    frame: pd.DataFrame,
    references: list[tuple[str, dict[str, Any]]],
) -> list[dict[str, Any]]:
    holdout = frame.iloc[list(HOLDOUT_ROWS)].copy()
    training = frame.drop(frame.index[list(HOLDOUT_ROWS)]).copy()
    X_train = training[["INC", "HOVAL"]]
    y_train = training["CRIME"]
    coords_train = training[["X", "Y"]]
    X_holdout = holdout[["INC", "HOVAL"]]
    coords_holdout = holdout[["X", "Y"]]

    model = _fit(
        X_train,
        y_train,
        coords_train,
        kernel="gaussian",
        bandwidth=10.0,
        adaptive=False,
        sigma2_v1=False,
    )
    result = model.predict_result(X_holdout, coords_holdout)
    actual_params = np.column_stack([result.intercept, result.coef])

    rows: list[dict[str, Any]] = []
    for name, payload in references:
        ref = payload["held_out_fixed_gaussian_prediction"]
        if ref.get("error"):
            continue
        _record(
            rows,
            reference=name,
            version=payload["reference_version"],
            case="held_out_fixed_gaussian_prediction",
            metric="params",
            actual=actual_params,
            expected=ref["params"],
        )
        _record(
            rows,
            reference=name,
            version=payload["reference_version"],
            case="held_out_fixed_gaussian_prediction",
            metric="predictions",
            actual=result.predictions,
            expected=ref["predictions"],
        )
    return rows


def _point_map(payload: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {int(point["k"]): point for point in payload["points"]}


def _curve_rows(
    pygwrx: dict[str, Any],
    mgwr: dict[str, Any],
    gwmodel: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    py = _point_map(pygwrx)
    mg = _point_map(mgwr)
    gw_payload = gwmodel["adaptive_bisquare_bandwidth_curve"]
    gw = _point_map(gw_payload)

    comparison_rows: list[dict[str, Any]] = []
    argmin_rows: list[dict[str, Any]] = []

    criterion_specs = [
        ("cv_sse", ["mgwr", "GWmodel"], "strict"),
        ("aic", ["mgwr"], "strict"),
        ("aicc", ["mgwr", "GWmodel"], "strict"),
        ("bic", ["mgwr"], "strict"),
        ("bic", ["GWmodel"], "different_definition"),
    ]
    sources = {"pyGWRx": py, "mgwr": mg, "GWmodel": gw}

    for criterion, reference_names, interpretation in criterion_specs:
        for reference in reference_names:
            ks: list[int] = []
            py_values: list[float] = []
            ref_values: list[float] = []
            for k in sorted(set(py) & set(sources[reference])):
                p = py[k].get(criterion)
                r = sources[reference][k].get(criterion)
                if p is None or r is None:
                    continue
                try:
                    pf = float(p)
                    rf = float(r)
                except (TypeError, ValueError):
                    continue
                if np.isfinite(pf) and np.isfinite(rf):
                    ks.append(k)
                    py_values.append(pf)
                    ref_values.append(rf)
            if not ks:
                continue
            stats = _compare(py_values, ref_values)
            comparison_rows.append(
                {
                    "criterion": criterion,
                    "reference": reference,
                    "reference_version": (
                        mgwr.get("reference_version")
                        if reference == "mgwr"
                        else gwmodel.get("reference_version")
                    ),
                    "interpretation": interpretation,
                    "common_k_min": min(ks),
                    "common_k_max": max(ks),
                    "n_candidates": len(ks),
                    **stats,
                }
            )
            py_argmin = ks[int(np.argmin(py_values))]
            ref_argmin = ks[int(np.argmin(ref_values))]
            argmin_rows.append(
                {
                    "criterion": criterion,
                    "reference": reference,
                    "interpretation": interpretation,
                    "common_k_min": min(ks),
                    "common_k_max": max(ks),
                    "n_candidates": len(ks),
                    "pygwrx_argmin": py_argmin,
                    "reference_argmin": ref_argmin,
                    "match": py_argmin == ref_argmin,
                }
            )
    return comparison_rows, argmin_rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    frame = pd.read_csv(SOURCE_PATH)
    mgwr = _load(DATA_DIR / "mgwr_2.2.1.json")
    gwmodel = _load(DATA_DIR / "GWmodel_reference.json")
    spgwr = _load(DATA_DIR / "spgwr_reference.json")
    pygwrx_curve = _load(DATA_DIR / "pygwrx_bandwidth_curve.json")
    mgwr_curve = _load(DATA_DIR / "mgwr_bandwidth_curve.json")

    rows, _ = _calibration_comparisons(frame, mgwr, gwmodel, spgwr)
    rows.extend(
        _holdout_comparisons(
            frame,
            [("mgwr", mgwr), ("GWmodel", gwmodel), ("spgwr", spgwr)],
        )
    )
    curve_rows, argmin_rows = _curve_rows(pygwrx_curve, mgwr_curve, gwmodel)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _write_csv(OUT_DIR / "comparison.csv", rows)
    _write_csv(OUT_DIR / "bandwidth_curve_comparison.csv", curve_rows)
    _write_csv(OUT_DIR / "bandwidth_curve_argmins.csv", argmin_rows)
    (OUT_DIR / "comparison.json").write_text(
        json.dumps(rows, indent=2) + "\n", encoding="utf-8"
    )
    (OUT_DIR / "bandwidth_curve_comparison.json").write_text(
        json.dumps(
            {"curve_comparisons": curve_rows, "argmins": argmin_rows}, indent=2
        )
        + "\n",
        encoding="utf-8",
    )

    strict = [row for row in rows if row["interpretation"] == "strict"]
    by_reference: dict[str, list[dict[str, Any]]] = {}
    for row in strict:
        by_reference.setdefault(row["reference"], []).append(row)

    lines = [
        "# Columbus Real-Data GWR External Validation",
        "",
        "Dataset: 49 Columbus, Ohio neighbourhoods; response `CRIME`; predictors `INC` and `HOVAL`; coordinates `X`, `Y`.",
        "",
        "The calibration checks fixed/adaptive Gaussian and bisquare GWR. Five geographically dispersed observations (zero-based rows 0, 10, 20, 30, and 40) are additionally held out for genuine out-of-sample local calibration and prediction.",
        "",
        "## Strict like-for-like comparisons",
        "",
        "| Reference | Version | Checks | Worst max absolute difference | Metric/case |",
        "|---|---|---:|---:|---|",
    ]
    versions = {
        "mgwr": mgwr["reference_version"],
        "GWmodel": gwmodel["reference_version"],
        "spgwr": spgwr["reference_version"],
    }
    for reference, group in by_reference.items():
        worst = max(group, key=lambda row: row["max_abs_diff"])
        lines.append(
            f"| {reference} | {versions[reference]} | {len(group)} | "
            f"{worst['max_abs_diff']:.6e} | {worst['case']} / {worst['metric']} |"
        )

    lines.extend(
        [
            "",
            "## Controlled adaptive-bisquare bandwidth curves",
            "",
            "| Criterion | Reference | Interpretation | Common k | Max absolute difference | pyGWRx argmin | Reference argmin |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
    )
    argmins = {(r["criterion"], r["reference"]): r for r in argmin_rows}
    for row in curve_rows:
        arg = argmins[(row["criterion"], row["reference"])]
        lines.append(
            f"| {row['criterion']} | {row['reference']} | {row['interpretation']} | "
            f"{row['common_k_min']}–{row['common_k_max']} | {row['max_abs_diff']:.6e} | "
            f"{arg['pygwrx_argmin']} | {arg['reference_argmin']} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation notes",
            "",
            "- Fixed-bandwidth spgwr comparisons are strict because the kernel and bandwidth semantics align exactly.",
            "- Adaptive spgwr results are archived as semantic cross-checks because spgwr specifies adaptive bandwidth as a sample proportion rather than an integer neighbour order.",
            "- GWmodel `Local_R2` is archived as a definition difference; fixed cases show that mgwr/spgwr use the same local-R² convention as pyGWRx.",
            "- GWmodel AIC/BIC conventions differ from pyGWRx/mgwr, so only like-for-like criteria are strict.",
            "- Bandwidth curves compare the same integer candidates rather than each package's default optimizer, separating criterion mathematics from search-strategy effects.",
            "",
        ]
    )
    report = OUT_DIR / "gwr_columbus_validation_report.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print(report.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
