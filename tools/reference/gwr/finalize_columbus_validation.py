#!/usr/bin/env python3
"""Finalize compact, versionable Columbus GWR validation fixtures and reports."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = ROOT / "tests" / "reference_data" / "gwr" / "real_columbus"
FROZEN_DIR = RAW_DIR / "frozen"
RESULT_DIR = ROOT / "validation_results" / "gwr" / "real_columbus"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _compact_case(case: dict[str, Any], *, keep_inference: bool = True) -> dict[str, Any]:
    keys = ["config", "params", "predy", "residuals", "local_r2", "diagnostics"]
    if keep_inference:
        keys.extend(
            [
                "bse",
                "tvalues",
                "influence",
                "standardized_residuals",
                "cooks_distance",
            ]
        )
    return {key: case.get(key) for key in keys if key in case and case.get(key) is not None}


def _compact_prediction(prediction: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "config",
        "holdout_rows_zero_based",
        "holdout_polyid",
        "actual_response",
        "coords",
        "X",
        "params",
        "predictions",
        "bse",
        "tvalues",
        "n_training",
        "n_holdout",
        "error",
    ]
    return {
        key: prediction.get(key)
        for key in keys
        if key in prediction and prediction.get(key) is not None
    }


def _compact_reference(payload: dict[str, Any], implementation: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        key: payload[key]
        for key in [
            "generator",
            "reference_package",
            "reference_version",
            "dataset",
            "dataset_source",
            "formula",
            "n_samples",
            "features",
            "response",
            "coords",
            "notes",
        ]
        if key in payload
    }
    result["cases"] = {
        name: _compact_case(case, keep_inference=implementation != "spgwr")
        for name, case in payload.get("cases", {}).items()
    }
    prediction = payload.get("held_out_fixed_gaussian_prediction")
    if prediction is not None:
        result["held_out_fixed_gaussian_prediction"] = _compact_prediction(prediction)
    if implementation == "GWmodel" and "adaptive_bisquare_bandwidth_curve" in payload:
        result["adaptive_bisquare_bandwidth_curve"] = payload[
            "adaptive_bisquare_bandwidth_curve"
        ]
    return result


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _point_map(payload: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {int(point["k"]): point for point in payload["points"]}


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _criterion_summary(
    pygwrx_curve: dict[str, Any],
    mgwr_curve: dict[str, Any],
    gwmodel: dict[str, Any],
) -> dict[str, Any]:
    py = _point_map(pygwrx_curve)
    mg = _point_map(mgwr_curve)
    gw = _point_map(gwmodel["adaptive_bisquare_bandwidth_curve"])
    sources = {"mgwr": mg, "GWmodel": gw}
    specs = {
        "cv_sse": ["mgwr", "GWmodel"],
        "aic": ["mgwr"],
        "aicc": ["mgwr", "GWmodel"],
        "bic": ["mgwr", "GWmodel"],
    }
    summary: dict[str, Any] = {
        "raw_candidate_domain": [4, 49],
        "near_saturated_boundary": {
            "k": 4,
            "reason": "trace(S) is approximately n; AICc is non-finite in pyGWRx and the point is retained only for transparency",
            "pygwrx_trace_S": py[4].get("trace_S"),
        },
        "criteria": {},
    }
    for criterion, refs in specs.items():
        criterion_result: dict[str, Any] = {}
        py_finite = [(k, _finite(point.get(criterion))) for k, point in py.items()]
        py_finite = [(k, value) for k, value in py_finite if value is not None]
        criterion_result["pygwrx_raw_argmin"] = min(py_finite, key=lambda item: item[1])[0]
        stable_py = [(k, value) for k, value in py_finite if k >= 5]
        criterion_result["pygwrx_k_ge_5_argmin"] = min(stable_py, key=lambda item: item[1])[0]
        criterion_result["references"] = {}
        for ref_name in refs:
            ref = sources[ref_name]
            common: list[tuple[int, float, float]] = []
            for k in sorted(set(py) & set(ref)):
                p_value = _finite(py[k].get(criterion))
                r_value = _finite(ref[k].get(criterion))
                if p_value is not None and r_value is not None:
                    common.append((k, p_value, r_value))
            if not common:
                continue
            raw_py = min(common, key=lambda item: item[1])[0]
            raw_ref = min(common, key=lambda item: item[2])[0]
            stable = [item for item in common if item[0] >= 5]
            stable_py_argmin = min(stable, key=lambda item: item[1])[0]
            stable_ref_argmin = min(stable, key=lambda item: item[2])[0]
            delta = np.asarray([abs(p - r) for _, p, r in stable], dtype=float)
            ref_values = np.asarray([abs(r) for _, _, r in stable], dtype=float)
            rel = delta / np.maximum(ref_values, np.finfo(float).eps)
            criterion_result["references"][ref_name] = {
                "common_raw_domain": [min(k for k, _, _ in common), max(k for k, _, _ in common)],
                "raw_argmin": {"pygwrx": raw_py, "reference": raw_ref},
                "k_ge_5_argmin": {
                    "pygwrx": stable_py_argmin,
                    "reference": stable_ref_argmin,
                },
                "k_ge_5_max_abs_diff": float(delta.max()),
                "k_ge_5_max_rel_diff": float(rel.max()),
                "interpretation": (
                    "different_definition"
                    if criterion == "bic" and ref_name == "GWmodel"
                    else "strict"
                ),
            }
        summary["criteria"][criterion] = criterion_result
    return summary


def main() -> None:
    mgwr = _load(RAW_DIR / "mgwr_2.2.1.json")
    gwmodel = _load(RAW_DIR / "GWmodel_reference.json")
    spgwr = _load(RAW_DIR / "spgwr_reference.json")
    pygwrx_curve = _load(RAW_DIR / "pygwrx_bandwidth_curve.json")
    mgwr_curve = _load(RAW_DIR / "mgwr_bandwidth_curve.json")

    FROZEN_DIR.mkdir(parents=True, exist_ok=True)
    _write(FROZEN_DIR / "mgwr.json", _compact_reference(mgwr, "mgwr"))
    _write(FROZEN_DIR / "GWmodel.json", _compact_reference(gwmodel, "GWmodel"))
    _write(FROZEN_DIR / "spgwr.json", _compact_reference(spgwr, "spgwr"))
    _write(FROZEN_DIR / "pygwrx_bandwidth_curve.json", pygwrx_curve)
    _write(FROZEN_DIR / "mgwr_bandwidth_curve.json", mgwr_curve)

    summary = _criterion_summary(pygwrx_curve, mgwr_curve, gwmodel)
    _write(FROZEN_DIR / "bandwidth_summary.json", summary)

    comparisons = _read_csv(RESULT_DIR / "comparison.csv")
    strict = [row for row in comparisons if row["interpretation"] == "strict"]
    versions = {
        "mgwr": mgwr["reference_version"],
        "GWmodel": gwmodel["reference_version"],
        "spgwr": spgwr["reference_version"],
    }

    lines = [
        "# Columbus Real-Data GWR External Validation",
        "",
        "Dataset: 49 Columbus, Ohio neighbourhoods; model `CRIME ~ INC + HOVAL`; coordinates `X`, `Y`.",
        "",
        "Four calibration configurations are checked: fixed/adaptive × Gaussian/bisquare. Five geographically dispersed neighbourhoods (zero-based rows 0, 10, 20, 30, 40) are withheld and predicted from a 44-neighbourhood training fit.",
        "",
        "## Strict numerical comparisons",
        "",
        "| Reference | Version | Strict checks | Worst max absolute difference | Worst case/metric |",
        "|---|---|---:|---:|---|",
    ]
    for reference in ("mgwr", "GWmodel", "spgwr"):
        group = [row for row in strict if row["reference"] == reference]
        worst = max(group, key=lambda row: float(row["max_abs_diff"]))
        lines.append(
            f"| {reference} | {versions[reference]} | {len(group)} | "
            f"{float(worst['max_abs_diff']):.6e} | {worst['case']} / {worst['metric']} |"
        )

    lines.extend(
        [
            "",
            "## Controlled adaptive-bisquare bandwidth validation",
            "",
            "All integer candidates `k=4..49` are archived. `k=4` is retained as a transparent near-saturated boundary (`trace(S) ≈ n`); pyGWRx correctly reports non-finite AICc there. The table therefore reports both raw argmins and the `k>=5` diagnostic summary instead of silently deleting the boundary point.",
            "",
            "| Criterion | Reference | Raw argmin py/ref | k>=5 argmin py/ref | k>=5 max abs diff | k>=5 max rel diff | Interpretation |",
            "|---|---|---:|---:|---:|---:|---|",
        ]
    )
    for criterion, payload in summary["criteria"].items():
        for reference, item in payload["references"].items():
            raw = item["raw_argmin"]
            stable = item["k_ge_5_argmin"]
            lines.append(
                f"| {criterion} | {reference} | {raw['pygwrx']}/{raw['reference']} | "
                f"{stable['pygwrx']}/{stable['reference']} | "
                f"{item['k_ge_5_max_abs_diff']:.6e} | "
                f"{item['k_ge_5_max_rel_diff']:.6e} | {item['interpretation']} |"
            )

    lines.extend(
        [
            "",
            "## Semantic boundaries",
            "",
            "- Fixed Gaussian/bisquare comparisons against all three packages are strict like-for-like checks.",
            "- On real data, pyGWRx adaptive neighbourhoods are numerically closest to GWmodel; mgwr remains very close, while spgwr uses a sample-proportion `q` and is therefore archived only as an adaptive semantic cross-check.",
            "- GWmodel `Local_R2` and AIC/BIC conventions are not forced to equal pyGWRx where definitions differ.",
            "- mgwr `sigma2_v1=True` adjusted-R² uses a different ENP convention; the distinction is explicitly preserved.",
            "- Raw full external outputs are reproducible with the generator scripts; compact frozen fixtures omit large hat matrices to keep the repository lean.",
            "",
        ]
    )
    report = RESULT_DIR / "gwr_columbus_validation_report.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print(report.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
