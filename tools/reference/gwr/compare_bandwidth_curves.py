#!/usr/bin/env python3
"""Compare controlled GWR bandwidth criterion curves across implementations."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "tests" / "reference_data" / "gwr"
OUT_DIR = ROOT / "validation_results" / "gwr"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SOURCES = {
    "pyGWRx": "pygwrx_bandwidth_curve.json",
    "mgwr": "mgwr_bandwidth_curve.json",
    "GWmodel": "GWmodel_bandwidth_curve.json",
    "spgwr": "spgwr_bandwidth_curve.json",
}


def _load(name: str) -> dict[str, Any]:
    return json.loads((DATA_DIR / name).read_text(encoding="utf-8"))


def _point_map(payload: dict[str, Any]) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for point in payload["points"]:
        key = point.get("k", point.get("k_equivalent"))
        result[int(key)] = point
    return result


def _number(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _argmin(points: dict[int, dict[str, Any]], metric: str) -> int | None:
    values = [(k, _number(point.get(metric))) for k, point in points.items()]
    finite = [(k, value) for k, value in values if value is not None]
    if not finite:
        return None
    return min(finite, key=lambda item: (item[1], item[0]))[0]


def _curve_stats(
    left: dict[int, dict[str, Any]],
    right: dict[int, dict[str, Any]],
    metric: str,
) -> dict[str, Any]:
    pairs: list[tuple[int, float, float]] = []
    for k in sorted(set(left) & set(right)):
        lv = _number(left[k].get(metric))
        rv = _number(right[k].get(metric))
        if lv is not None and rv is not None:
            pairs.append((k, lv, rv))
    if not pairs:
        return {
            "n_compared": 0,
            "max_abs_diff": None,
            "mean_abs_diff": None,
            "rmse": None,
            "max_shape_diff": None,
        }

    lvals = np.asarray([item[1] for item in pairs], dtype=float)
    rvals = np.asarray([item[2] for item in pairs], dtype=float)
    diff = lvals - rvals
    lshape = lvals - np.min(lvals)
    rshape = rvals - np.min(rvals)
    return {
        "n_compared": len(pairs),
        "max_abs_diff": float(np.max(np.abs(diff))),
        "mean_abs_diff": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff**2))),
        "max_shape_diff": float(np.max(np.abs(lshape - rshape))),
    }


def main() -> None:
    payloads = {name: _load(filename) for name, filename in SOURCES.items()}
    points = {name: _point_map(payload) for name, payload in payloads.items()}
    ks = list(range(4, 41))

    curve_rows: list[dict[str, Any]] = []
    for k in ks:
        row: dict[str, Any] = {"k": k}
        for implementation in ("pyGWRx", "mgwr", "GWmodel"):
            point = points[implementation].get(k, {})
            for metric in ("cv_sse", "aic", "aicc", "bic", "trace_S"):
                row[f"{implementation}_{metric}"] = point.get(metric)
        sp_point = points["spgwr"].get(k, {})
        row["spgwr_q"] = sp_point.get("q")
        row["spgwr_cv_sse"] = sp_point.get("cv_sse")
        row["spgwr_aicc_like"] = sp_point.get("aicc_like")
        curve_rows.append(row)

    curve_path = OUT_DIR / "adaptive_bandwidth_criterion_curves.csv"
    with curve_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(curve_rows[0]))
        writer.writeheader()
        writer.writerows(curve_rows)

    comparisons = [
        # Same integer-k semantics and same published criterion family: strict.
        ("pyGWRx", "mgwr", "cv_sse", "strict"),
        ("pyGWRx", "mgwr", "aic", "strict"),
        ("pyGWRx", "mgwr", "aicc", "strict"),
        ("pyGWRx", "mgwr", "bic", "strict"),
        ("pyGWRx", "GWmodel", "cv_sse", "strict"),
        ("pyGWRx", "GWmodel", "aicc", "strict"),
        # GWmodel BIC is archived to diagnose formula differences rather than assumed equal.
        ("pyGWRx", "GWmodel", "bic", "definition_check"),
        ("mgwr", "GWmodel", "cv_sse", "strict"),
        ("mgwr", "GWmodel", "aicc", "strict"),
        ("mgwr", "GWmodel", "bic", "definition_check"),
        # spgwr uses continuous adaptive q, so these are shape/argmin sensitivity checks only.
        ("pyGWRx", "spgwr", "cv_sse", "different_adaptive_semantics"),
    ]

    summary_rows: list[dict[str, Any]] = []
    for left_name, right_name, metric, interpretation in comparisons:
        stats = _curve_stats(points[left_name], points[right_name], metric)
        left_argmin = _argmin(points[left_name], metric)
        right_argmin = _argmin(points[right_name], metric)
        summary_rows.append(
            {
                "left": left_name,
                "right": right_name,
                "metric": metric,
                "interpretation": interpretation,
                **stats,
                "left_argmin_k": left_argmin,
                "right_argmin_k": right_argmin,
                "argmin_match": (
                    None
                    if left_argmin is None or right_argmin is None
                    else left_argmin == right_argmin
                ),
            }
        )

    # Add standalone argmins so every supported criterion has an auditable optimum.
    argmins: list[dict[str, Any]] = []
    for implementation in ("pyGWRx", "mgwr", "GWmodel"):
        for metric in ("cv_sse", "aic", "aicc", "bic"):
            optimum = _argmin(points[implementation], metric)
            if optimum is not None:
                argmins.append(
                    {
                        "implementation": implementation,
                        "metric": metric,
                        "argmin_k": optimum,
                    }
                )
    for metric in ("cv_sse", "aicc_like"):
        optimum = _argmin(points["spgwr"], metric)
        if optimum is not None:
            argmins.append(
                {
                    "implementation": "spgwr",
                    "metric": metric,
                    "argmin_k": optimum,
                    "note": "k is only q*n equivalence; spgwr optimizes continuous q",
                }
            )

    summary_path = OUT_DIR / "bandwidth_curve_comparison.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    argmin_path = OUT_DIR / "bandwidth_curve_argmins.csv"
    fieldnames = ["implementation", "metric", "argmin_k", "note"]
    with argmin_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in argmins:
            writer.writerow({key: row.get(key) for key in fieldnames})

    json_path = OUT_DIR / "bandwidth_curve_comparison.json"
    json_path.write_text(
        json.dumps(
            {
                "candidate_domain": [4, 40],
                "kernel": "bisquare",
                "comparisons": summary_rows,
                "argmins": argmins,
                "source_metadata": {
                    name: {
                        key: payload.get(key)
                        for key in (
                            "implementation",
                            "reference_version",
                            "candidate_semantics",
                            "n_samples",
                        )
                        if payload.get(key) is not None
                    }
                    for name, payload in payloads.items()
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    report_lines = [
        "# Controlled GWR Bandwidth-Criterion Validation",
        "",
        "All integer adaptive candidates `k=4..40` are evaluated explicitly. This removes",
        "optimizer stopping rules and default search ranges from the numerical comparison.",
        "mgwr CV is converted from mean squared LOO error to SSE by multiplying by n=40.",
        "spgwr remains a semantic cross-check because its adaptive parameter is a continuous",
        "sample proportion q rather than an integer neighbour-order bandwidth.",
        "",
        "## Pairwise curve comparisons",
        "",
        "| Left | Right | Metric | Interpretation | n | Max abs diff | RMSE | Argmin left | Argmin right | Match |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary_rows:
        def fmt(value: Any) -> str:
            if value is None:
                return "—"
            if isinstance(value, bool):
                return "yes" if value else "no"
            if isinstance(value, float):
                return f"{value:.6e}"
            return str(value)

        report_lines.append(
            "| {left} | {right} | {metric} | {interpretation} | {n_compared} | "
            "{max_abs_diff} | {rmse} | {left_argmin_k} | {right_argmin_k} | {argmin_match} |".format(
                **{key: fmt(value) for key, value in row.items()}
            )
        )

    report_lines.extend(
        [
            "",
            "## Criterion minima on the shared candidate domain",
            "",
            "| Implementation | Criterion | Argmin k | Note |",
            "|---|---|---:|---|",
        ]
    )
    for row in argmins:
        report_lines.append(
            f"| {row['implementation']} | {row['metric']} | {row['argmin_k']} | {row.get('note', '')} |"
        )

    report_lines.extend(
        [
            "",
            "## Interpretation rules",
            "",
            "- `strict`: same integer-k bandwidth semantics and directly comparable criterion definition.",
            "- `definition_check`: values are archived, but equality is not assumed until formulas are matched.",
            "- `different_adaptive_semantics`: spgwr q=k/n is a sensitivity comparison, not an equality test.",
            "",
        ]
    )
    (OUT_DIR / "bandwidth_curve_report.md").write_text(
        "\n".join(report_lines), encoding="utf-8"
    )

    print(f"wrote {curve_path.relative_to(ROOT)}")
    print(f"wrote {summary_path.relative_to(ROOT)}")
    print(f"wrote {argmin_path.relative_to(ROOT)}")
    print(f"wrote {(OUT_DIR / 'bandwidth_curve_report.md').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
