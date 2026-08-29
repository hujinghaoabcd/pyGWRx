#!/usr/bin/env python3
"""Compare controlled GWR bandwidth criterion curves across implementations."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

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


def _finite_ks(
    points: dict[int, dict[str, Any]], metric: str, candidates: Iterable[int]
) -> set[int]:
    return {
        int(k)
        for k in candidates
        if k in points and _number(points[k].get(metric)) is not None
    }


def _nonsaturated_ks(
    points: dict[int, dict[str, Any]], candidates: Iterable[int], *, n_samples: int
) -> set[int]:
    """Exclude essentially interpolating smoothers with trace(S) >= n-1.

    This is only a validation-domain guard. It does not modify model fitting or the
    public bandwidth selector. The excluded points remain in the raw curve CSV.
    """
    valid: set[int] = set()
    for k in candidates:
        point = points.get(int(k), {})
        trace_s = _number(point.get("trace_S"))
        if trace_s is not None and trace_s < n_samples - 1.0:
            valid.add(int(k))
    return valid


def _argmin(
    points: dict[int, dict[str, Any]], metric: str, allowed_ks: Iterable[int]
) -> int | None:
    finite: list[tuple[int, float]] = []
    for k in sorted(set(int(value) for value in allowed_ks)):
        if k not in points:
            continue
        value = _number(points[k].get(metric))
        if value is not None:
            finite.append((k, value))
    if not finite:
        return None
    return min(finite, key=lambda item: (item[1], item[0]))[0]


def _curve_stats(
    left: dict[int, dict[str, Any]],
    right: dict[int, dict[str, Any]],
    metric: str,
    allowed_ks: Iterable[int],
) -> dict[str, Any]:
    pairs: list[tuple[int, float, float]] = []
    for k in sorted(set(int(value) for value in allowed_ks)):
        if k not in left or k not in right:
            continue
        lv = _number(left[k].get(metric))
        rv = _number(right[k].get(metric))
        if lv is not None and rv is not None:
            pairs.append((k, lv, rv))
    if not pairs:
        return {
            "n_compared": 0,
            "first_k": None,
            "last_k": None,
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
        "first_k": pairs[0][0],
        "last_k": pairs[-1][0],
        "max_abs_diff": float(np.max(np.abs(diff))),
        "mean_abs_diff": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff**2))),
        "max_shape_diff": float(np.max(np.abs(lshape - rshape))),
    }


def main() -> None:
    payloads = {name: _load(filename) for name, filename in SOURCES.items()}
    points = {name: _point_map(payload) for name, payload in payloads.items()}
    ks = set(range(4, 41))
    n_samples = int(payloads["pyGWRx"]["n_samples"])

    # Raw k=4..40 values remain archived. Strict comparisons use domains where the
    # criterion is mathematically meaningful and independently estimable.
    cv_domain = set.intersection(
        *(
            _finite_ks(points[name], "cv_sse", ks)
            for name in ("pyGWRx", "mgwr", "GWmodel")
        )
    )

    # AIC/BIC can numerically reward a saturated interpolation at k=4. Exclude only
    # this near-saturated boundary using trace(S) < n-1; no arbitrary 20-neighbour
    # lower bound is imposed.
    nonsaturated = _nonsaturated_ks(points["pyGWRx"], ks, n_samples=n_samples)
    nonsaturated &= _nonsaturated_ks(points["mgwr"], ks, n_samples=n_samples)
    aic_domain = (
        nonsaturated
        & _finite_ks(points["pyGWRx"], "aic", ks)
        & _finite_ks(points["mgwr"], "aic", ks)
    )
    bic_domain = (
        nonsaturated
        & _finite_ks(points["pyGWRx"], "bic", ks)
        & _finite_ks(points["mgwr"], "bic", ks)
    )

    # AICc requires n - 2 - trace(S) > 0. PyGWRx already returns infinity when this
    # fails. Requiring finite values from all three packages excludes the misleading
    # finite k=4 values returned by mgwr/GWmodel on this saturated fixture.
    aicc_domain = set.intersection(
        *(
            _finite_ks(points[name], "aicc", ks)
            for name in ("pyGWRx", "mgwr", "GWmodel")
        )
    )

    # spgwr q=k/n is a semantic sensitivity curve, never an integer-k equality test.
    spgwr_cv_domain = _finite_ks(points["pyGWRx"], "cv_sse", ks) & _finite_ks(
        points["spgwr"], "cv_sse", ks
    )

    domains = {
        "cv_common_finite": cv_domain,
        "aic_nonsaturated": aic_domain,
        "aicc_common_valid": aicc_domain,
        "bic_nonsaturated": bic_domain,
        "spgwr_semantic": spgwr_cv_domain,
    }

    curve_rows: list[dict[str, Any]] = []
    for k in sorted(ks):
        row: dict[str, Any] = {
            "k": k,
            "valid_cv_strict": k in cv_domain,
            "valid_aic_strict": k in aic_domain,
            "valid_aicc_strict": k in aicc_domain,
            "valid_bic_strict": k in bic_domain,
        }
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
        ("pyGWRx", "mgwr", "cv_sse", "strict", "cv_common_finite"),
        ("pyGWRx", "GWmodel", "cv_sse", "strict", "cv_common_finite"),
        ("mgwr", "GWmodel", "cv_sse", "strict", "cv_common_finite"),
        ("pyGWRx", "mgwr", "aic", "strict", "aic_nonsaturated"),
        ("pyGWRx", "mgwr", "aicc", "strict", "aicc_common_valid"),
        ("pyGWRx", "GWmodel", "aicc", "strict", "aicc_common_valid"),
        ("mgwr", "GWmodel", "aicc", "strict", "aicc_common_valid"),
        ("pyGWRx", "mgwr", "bic", "strict", "bic_nonsaturated"),
        ("pyGWRx", "GWmodel", "bic", "definition_check", "bic_nonsaturated"),
        ("mgwr", "GWmodel", "bic", "definition_check", "bic_nonsaturated"),
        (
            "pyGWRx",
            "spgwr",
            "cv_sse",
            "different_adaptive_semantics",
            "spgwr_semantic",
        ),
    ]

    summary_rows: list[dict[str, Any]] = []
    for left_name, right_name, metric, interpretation, domain_name in comparisons:
        domain = domains[domain_name]
        stats = _curve_stats(points[left_name], points[right_name], metric, domain)
        left_argmin = _argmin(points[left_name], metric, domain)
        right_argmin = _argmin(points[right_name], metric, domain)
        summary_rows.append(
            {
                "left": left_name,
                "right": right_name,
                "metric": metric,
                "interpretation": interpretation,
                "validation_domain": domain_name,
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

    argmin_specs = [
        ("pyGWRx", "cv_sse", "cv_common_finite", "strict three-way CV domain"),
        ("mgwr", "cv_sse", "cv_common_finite", "strict three-way CV domain"),
        ("GWmodel", "cv_sse", "cv_common_finite", "strict three-way CV domain"),
        ("pyGWRx", "aic", "aic_nonsaturated", "saturated k=4 excluded"),
        ("mgwr", "aic", "aic_nonsaturated", "saturated k=4 excluded"),
        ("pyGWRx", "aicc", "aicc_common_valid", "requires finite valid AICc"),
        ("mgwr", "aicc", "aicc_common_valid", "requires finite valid AICc"),
        ("GWmodel", "aicc", "aicc_common_valid", "requires finite valid AICc"),
        ("pyGWRx", "bic", "bic_nonsaturated", "saturated k=4 excluded"),
        ("mgwr", "bic", "bic_nonsaturated", "saturated k=4 excluded"),
        (
            "GWmodel",
            "bic",
            "bic_nonsaturated",
            "different BIC formula; diagnostic only",
        ),
    ]
    argmins: list[dict[str, Any]] = []
    for implementation, metric, domain_name, note in argmin_specs:
        optimum = _argmin(points[implementation], metric, domains[domain_name])
        if optimum is not None:
            argmins.append(
                {
                    "implementation": implementation,
                    "metric": metric,
                    "argmin_k": optimum,
                    "validation_domain": domain_name,
                    "note": note,
                }
            )
    for metric in ("cv_sse", "aicc_like"):
        optimum = _argmin(points["spgwr"], metric, spgwr_cv_domain)
        if optimum is not None:
            argmins.append(
                {
                    "implementation": "spgwr",
                    "metric": metric,
                    "argmin_k": optimum,
                    "validation_domain": "spgwr_semantic",
                    "note": "k is only q*n equivalence; spgwr optimizes continuous q",
                }
            )

    summary_path = OUT_DIR / "bandwidth_curve_comparison.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    argmin_path = OUT_DIR / "bandwidth_curve_argmins.csv"
    fieldnames = [
        "implementation",
        "metric",
        "argmin_k",
        "validation_domain",
        "note",
    ]
    with argmin_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(argmins)

    domain_metadata = {
        name: {
            "k_values": sorted(domain),
            "first_k": min(domain) if domain else None,
            "last_k": max(domain) if domain else None,
            "n_candidates": len(domain),
        }
        for name, domain in domains.items()
    }

    json_path = OUT_DIR / "bandwidth_curve_comparison.json"
    json_path.write_text(
        json.dumps(
            {
                "raw_candidate_domain": [4, 40],
                "kernel": "bisquare",
                "validation_domains": domain_metadata,
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

    def fmt(value: Any) -> str:
        if value is None:
            return "—"
        if isinstance(value, bool):
            return "yes" if value else "no"
        if isinstance(value, float):
            return f"{value:.6e}"
        return str(value)

    report_lines = [
        "# Controlled GWR Bandwidth-Criterion Validation",
        "",
        "All raw integer adaptive candidates `k=4..40` are archived. Strict comparisons",
        "use criterion-specific validity domains so saturated or non-estimable boundary",
        "candidates do not masquerade as optimal bandwidths.",
        "",
        "- CV strict domain: candidates with finite CV from PyGWRx, mgwr, and GWmodel.",
        "- AIC/BIC strict domain: k values below the near-saturated trace(S) boundary are excluded.",
        "- AICc strict domain: candidates must be finite in all three implementations; this",
        "  excludes k=4 where `n - 2 - trace(S) <= 0` and AICc is mathematically invalid.",
        "- mgwr CV is converted from mean squared LOO error to SSE by multiplying by n=40.",
        "- spgwr remains a semantic cross-check because its adaptive parameter is a continuous",
        "  sample proportion q rather than an integer neighbour-order bandwidth.",
        "",
        "## Validation domains",
        "",
        "| Domain | First k | Last k | Candidates |",
        "|---|---:|---:|---:|",
    ]
    for name, metadata in domain_metadata.items():
        report_lines.append(
            f"| {name} | {metadata['first_k']} | {metadata['last_k']} | {metadata['n_candidates']} |"
        )

    report_lines.extend(
        [
            "",
            "## Pairwise curve comparisons",
            "",
            "| Left | Right | Metric | Interpretation | Domain | n | Max abs diff | RMSE | Argmin left | Argmin right | Match |",
            "|---|---|---|---|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in summary_rows:
        report_lines.append(
            "| {left} | {right} | {metric} | {interpretation} | {validation_domain} | {n_compared} | "
            "{max_abs_diff} | {rmse} | {left_argmin_k} | {right_argmin_k} | {argmin_match} |".format(
                **{key: fmt(value) for key, value in row.items()}
            )
        )

    report_lines.extend(
        [
            "",
            "## Criterion minima on the controlled validation domains",
            "",
            "| Implementation | Criterion | Argmin k | Domain | Note |",
            "|---|---|---:|---|---|",
        ]
    )
    for row in argmins:
        report_lines.append(
            f"| {row['implementation']} | {row['metric']} | {row['argmin_k']} | "
            f"{row['validation_domain']} | {row['note']} |"
        )

    report_lines.extend(
        [
            "",
            "## Low-bandwidth boundary finding",
            "",
            "`k=4` is an essentially saturated smoother for this fixture (`trace(S)≈40` with n=40).",
            "PyGWRx correctly returns infinite/invalid AICc there, while mgwr and GWmodel return",
            "finite negative values. Those values are retained in the raw curve archive but are not",
            "allowed to determine the validated AICc optimum. GWmodel also returns no finite CV at",
            "k=4 or k=5; therefore the strict three-way CV domain begins at k=6.",
            "",
            "## Interpretation rules",
            "",
            "- `strict`: same integer-k bandwidth semantics and directly comparable criterion definition.",
            "- `definition_check`: values are archived, but equality is not assumed until formulas are matched.",
            "- `different_adaptive_semantics`: spgwr q=k/n is a sensitivity comparison, not an equality test.",
            "",
        ]
    )
    report_path = OUT_DIR / "bandwidth_curve_report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    print(f"wrote {curve_path.relative_to(ROOT)}")
    print(f"wrote {summary_path.relative_to(ROOT)}")
    print(f"wrote {argmin_path.relative_to(ROOT)}")
    print(f"wrote {report_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
