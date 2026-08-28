# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT
"""Generate grouped public API pages and the example-coverage manifest."""

from __future__ import annotations

import ast
import csv
import inspect
import json
import shutil
from pathlib import Path
from typing import Any

import pygwrx.core as core
import pygwrx.diagnostics as diagnostics
import pygwrx.io as io
import pygwrx.models as models
import pygwrx.plotting as plotting

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = "https://github.com/hujinghaoabcd/pyGWRx/blob/main"
EXAMPLES = ROOT / "examples"
DOCS_API = ROOT / "docs" / "api"

PUBLIC_MODULES: dict[str, Any] = {
    "models": models,
    "core": core,
    "diagnostics": diagnostics,
    "plotting": plotting,
    "io": io,
}

GROUPS: dict[str, list[tuple[str, str, list[str]]]] = {
    "models": [
        ("gwr", "GWR", ["GWR", "GWRPredictionResult"]),
        ("mgwr", "MGWR", ["MGWR"]),
        ("rgwr", "RGWR", ["RGWR"]),
        ("stwr", "STWR", ["STWR", "STWRPredictionResult"]),
        ("gtwr", "GTWR", ["GTWR", "GTWRPredictionResult"]),
        ("gwglm", "GWGLM", ["GWGLM", "GWGLMPredictionResult"]),
        ("gw-lasso", "GWLasso", ["GWLasso"]),
        ("mixed-gwr", "MixedGWR", ["MixedGWR"]),
        ("gwpca", "GWPCA", ["GWPCA"]),
        ("gwda", "GWDA", ["GWDA"]),
        ("gwss", "GWSS", ["GWSS"]),
        ("scalable-gwr", "ScalableGWR", ["ScalableGWR"]),
        ("lcr-gwr", "LCRGWR", ["LCRGWR"]),
        ("bootstrap-gwr", "BootstrapGWR", ["BootstrapGWR"]),
        ("sgwr", "SGWR", ["SGWR"]),
        ("sgtwr", "SGTWR", ["SGTWR", "SGTWRPredictionResult"]),
        ("mgtwr", "MGTWR", ["MGTWR"]),
        ("lg-gwr", "LGGWR", ["LGGWR", "LGGWRPredictionResult"]),
        ("gr-gwr", "GRGWR", ["GRGWR", "GRGWRPredictionResult"]),
    ],
    "core": [
        (
            "base",
            "Base classes",
            [
                "BaseSpatialEstimator",
                "BaseSpatialRegressor",
                "SpatiotemporalMixin",
                "MultiscaleMixin",
                "BaseSpatiotemporalRegressor",
                "BaseMultiscaleRegressor",
                "BaseSpatialClassifier",
                "BaseSpatialTransformer",
                "BaseSpatialStatistics",
                "BaseSpatialInference",
            ],
        ),
        (
            "kernels",
            "Kernels",
            [
                "gaussian_kernel",
                "bisquare_kernel",
                "exponential_kernel",
                "tricube_kernel",
                "boxcar_kernel",
                "get_kernel_function",
            ],
        ),
        (
            "bandwidth",
            "Bandwidth selection",
            [
                "BandwidthSelector",
                "CrossValidationSelector",
                "AICSelector",
                "BICSelector",
                "get_bandwidth_selector",
            ],
        ),
        (
            "optimization",
            "Optimization",
            [
                "OptimizationResult",
                "GoldenSectionSearch",
                "BrentSearch",
            ],
        ),
        (
            "solver",
            "Local solvers",
            [
                "weighted_least_squares",
                "local_regression",
                "compute_hat_matrix",
                "adaptive_bandwidth_weights",
            ],
        ),
        (
            "metrics",
            "Metrics",
            [
                "compute_r_squared",
                "compute_adjusted_r_squared",
                "compute_aic",
                "compute_aicc",
                "compute_bic",
                "compute_local_r_squared",
                "compute_effective_parameters",
                "compute_diagnostics",
                "compute_trace_statistics",
                "compute_edf",
                "compute_enp",
            ],
        ),
        (
            "utils",
            "Distances and validation",
            [
                "euclidean_distance",
                "manhattan_distance",
                "chebyshev_distance",
                "minkowski_distance",
                "haversine_distance",
                "compute_distance_matrix",
                "DistanceCache",
                "validate_coords",
                "validate_data",
                "add_intercept",
                "extract_geopandas_coords",
                "chunked_computation",
            ],
        ),
    ],
    "diagnostics": [
        (
            "model",
            "Model summaries",
            [
                "DiagnosticSummary",
                "diagnostics_frame",
                "model_diagnostic_summary",
            ],
        ),
        (
            "residuals",
            "Residuals and influence",
            [
                "InfluenceThresholds",
                "influence_thresholds",
                "local_diagnostic_frame",
            ],
        ),
        ("collinearity", "Local collinearity", ["LocalCollinearityDiagnostics"]),
        (
            "inference",
            "Parameter inference",
            [
                "ParameterInference",
                "adjust_pvalues",
                "feature_names",
                "parameter_inference",
                "parameter_significance",
            ],
        ),
        (
            "temporal",
            "Temporal diagnostics",
            [
                "TemporalGroups",
                "model_times",
                "parameter_trajectory",
                "temporal_groups",
                "temporal_parameter_frame",
            ],
        ),
        (
            "weights",
            "Weight diagnostics",
            [
                "WeightComponents",
                "focus_weight_components",
                "weight_components",
            ],
        ),
        (
            "regimes",
            "Regime diagnostics",
            [
                "boundary_frame",
                "regime_frame",
                "regime_summary",
            ],
        ),
    ],
    "plotting": [
        (
            "surfaces",
            "Coefficient and diagnostic surfaces",
            [
                "plot_coefficient_map",
                "plot_significance_map",
                "plot_model_significance_map",
                "plot_local_diagnostic_map",
                "plot_local_collinearity",
            ],
        ),
        (
            "array-maps",
            "Array-based maps",
            [
                "plot_array_significance_map",
                "plot_local_coefficients",
                "plot_coefficient_surface",
                "plot_local_r2",
                "plot_bandwidth",
                "create_choropleth",
                "plot_multiple_coefficients",
            ],
        ),
        (
            "comparison",
            "Model comparison",
            [
                "compare_coefficient_surfaces",
                "compare_model_diagnostics",
            ],
        ),
        (
            "bandwidth",
            "Bandwidth plots",
            [
                "plot_kernel_weights",
                "plot_mgwr_bandwidths",
            ],
        ),
        (
            "diagnostics",
            "Regression diagnostics",
            [
                "plot_residuals",
                "plot_residual_histogram",
                "plot_qq",
                "plot_spatial_residuals",
                "plot_observed_vs_predicted",
                "plot_bandwidth_selection",
                "plot_coefficient_variability",
                "plot_diagnostic_panel",
                "plot_local_diagnostics",
            ],
        ),
        (
            "robust",
            "Robust and GLM plots",
            [
                "plot_rgwr_weights",
                "plot_rgwr_convergence",
                "plot_gwglm_residuals",
            ],
        ),
        (
            "regularization",
            "Regularization and mixed-model plots",
            [
                "plot_gwlasso_selection_frequency",
                "plot_gwlasso_active_map",
                "plot_gwlasso_alpha",
                "plot_mixed_gwr_coefficients",
            ],
        ),
        (
            "bootstrap",
            "Bootstrap plots",
            [
                "plot_bootstrap_pvalues",
                "plot_bootstrap_bandwidths",
            ],
        ),
        (
            "multivariate",
            "Multivariate and classification plots",
            [
                "plot_gwss_statistic",
                "plot_gwpca_explained_variance",
                "plot_gwpca_loading",
                "plot_gwda_classification",
                "plot_gwda_confusion_matrix",
            ],
        ),
        ("scalable", "Scalable GWR plots", ["plot_scalable_gwr_kernel"]),
        (
            "temporal",
            "Temporal plots",
            [
                "plot_temporal_coefficient_slices",
                "plot_mgtwr_scales",
                "plot_temporal_trajectory",
                "plot_temporal_residuals",
                "plot_temporal_bandwidths",
            ],
        ),
        (
            "decomposition",
            "Weight decomposition",
            [
                "plot_weight_decomposition",
                "plot_weight_profiles",
                "plot_selection_history",
            ],
        ),
        (
            "geometry",
            "LGGWR geometry",
            [
                "plot_lggwr_latent_geometry",
                "plot_lggwr_metric_matrix",
                "plot_lggwr_training",
                "plot_lggwr_neighbourhood_comparison",
            ],
        ),
        (
            "regimes",
            "GRGWR regimes",
            [
                "plot_grgwr_regimes",
                "plot_grgwr_convergence",
                "plot_grgwr_regime_sizes",
                "plot_grgwr_coefficient_surface",
            ],
        ),
    ],
    "io": [
        (
            "data",
            "Data conversion and persistence",
            [
                "load_data",
                "to_geodataframe",
                "from_geodataframe",
                "save_results",
            ],
        ),
        (
            "datasets",
            "Dataset registry",
            [
                "load_dataset",
                "load_dublin_voter",
                "load_hiv",
                "load_crime",
                "load_housing",
                "load_columbus",
                "load_ewhp",
                "load_georgia",
                "get_dublin_voter",
                "load_dubvoter",
                "get_dubvoter",
                "get_dataset_info",
                "list_datasets",
            ],
        ),
    ],
}

NAMESPACE_TITLES = {
    "models": "Models and result objects",
    "core": "Core numerical API",
    "diagnostics": "Diagnostics API",
    "plotting": "Plotting API",
    "io": "Input and output API",
}


def _example_imports() -> dict[tuple[str, str], str]:
    """Map every public symbol to its most specific maintained example."""
    candidates: dict[tuple[str, str], list[str]] = {}
    for path in sorted(EXAMPLES.rglob("*.py")):
        if path.name.startswith("_") or path.name in {
            "run_all.py",
            "validate_coverage.py",
        }:
            continue
        relative = path.relative_to(ROOT).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or not node.module:
                continue
            if node.module == "pygwrx":
                for alias in node.names:
                    if alias.name in models.__all__:
                        candidates.setdefault(("models", alias.name), []).append(
                            relative
                        )
            for namespace in ("core", "diagnostics", "plotting", "io"):
                if node.module == f"pygwrx.{namespace}":
                    for alias in node.names:
                        if alias.name in PUBLIC_MODULES[namespace].__all__:
                            candidates.setdefault((namespace, alias.name), []).append(
                                relative
                            )

    def rank(namespace: str, path: str) -> tuple[int, str]:
        preferred = f"examples/{namespace}/"
        if path.startswith(preferred):
            return (0, path)
        if namespace == "models" and path.startswith("examples/models/"):
            return (0, path)
        if path.startswith("examples/workflows/"):
            return (1, path)
        return (2, path)

    return {
        key: min(paths, key=lambda path: rank(key[0], path))
        for key, paths in candidates.items()
    }


def _kind(value: Any) -> str:
    if inspect.isclass(value):
        return "class"
    if inspect.isfunction(value) or inspect.isbuiltin(value):
        return "function"
    return "object"


def _directive(namespace: str, name: str) -> str:
    return f"::: pygwrx.{namespace}.{name}\n"


def _summary(value: Any) -> str:
    """Return the first useful paragraph from an object's docstring."""
    doc = inspect.getdoc(value) or "No summary is available."
    paragraphs = [
        part.strip().replace("\n", " ") for part in doc.split("\n\n") if part.strip()
    ]
    return paragraphs[0] if paragraphs else "No summary is available."


def _signature(value: Any) -> str:
    """Return a display signature without failing on extension objects."""
    try:
        return str(inspect.signature(value))
    except (TypeError, ValueError):
        return "(signature unavailable)"


def _example_source(relative_path: str) -> str:
    """Read a maintained example for inline API documentation."""
    if not relative_path:
        return ""
    return (ROOT / relative_path).read_text(encoding="utf-8").rstrip()


def _guide_link(namespace: str) -> str:
    links = {
        "models": "../../models/index.md",
        "core": "../../guides/core-numerics.md",
        "diagnostics": "../../guides/diagnostics.md",
        "plotting": "../../guides/visualization.md",
        "io": "../../guides/geospatial-io.md",
    }
    return links[namespace]


def generate() -> None:
    if DOCS_API.exists():
        shutil.rmtree(DOCS_API)
    DOCS_API.mkdir(parents=True)

    coverage = _example_imports()
    manifest: dict[str, dict[str, str]] = {}
    rows: list[dict[str, str]] = []

    for namespace, module in PUBLIC_MODULES.items():
        manifest[namespace] = {}
        group_dir = DOCS_API / namespace
        group_dir.mkdir()
        namespace_index = [
            f"# {NAMESPACE_TITLES[namespace]}",
            "",
            f"Public symbols: **{len(module.__all__)}**.",
            "",
            "| Group | Symbols | Reference |",
            "|---|---:|---|",
        ]
        grouped_names: list[str] = []
        for slug, title, names in GROUPS[namespace]:
            grouped_names.extend(names)
            namespace_index.append(f"| {title} | {len(names)} | [{title}]({slug}.md) |")
            page = [
                f"# {title}",
                "",
                f"This page documents **{len(names)}** public symbols. Each entry includes its purpose, import path, full API docstring, and the maintained example that exercises it.",
                "",
                f"[Conceptual guide]({_guide_link(namespace)}){{ .md-button }}",
                "",
            ]
            page_examples: list[str] = []
            for name in names:
                if name not in module.__all__:
                    raise SystemExit(
                        f"Configured API symbol is not public: {namespace}.{name}"
                    )
                example = coverage.get((namespace, name), "")
                manifest[namespace][name] = example
                value = getattr(module, name)
                kind = _kind(value)
                summary = _summary(value)
                signature = _signature(value).replace("|", "\\|")
                if example and example not in page_examples:
                    page_examples.append(example)
                page.extend(
                    [
                        f"## `{name}`",
                        "",
                        summary,
                        "",
                        "| Property | Value |",
                        "|---|---|",
                        f"| Type | `{kind}` |",
                        f"| Import | `from pygwrx.{namespace} import {name}` |",
                        f"| Signature | `{name}{signature}` |",
                        (
                            f"| Maintained example | [`{example}`]({REPOSITORY}/{example}) |"
                            if example
                            else "| Maintained example | **Missing** |"
                        ),
                        "",
                        _directive(namespace, name),
                        "",
                    ]
                )
                rows.append(
                    {
                        "namespace": namespace,
                        "symbol": name,
                        "kind": kind,
                        "summary": summary,
                        "example": example,
                    }
                )
            page.extend(["## Runnable examples used on this page", ""])
            for example in page_examples:
                page.extend(
                    [
                        f'??? example "`{example}`"',
                        "",
                        "    ```python",
                        *[
                            f"    {line}"
                            for line in _example_source(example).splitlines()
                        ],
                        "    ```",
                        "",
                    ]
                )
            (group_dir / f"{slug}.md").write_text("\n".join(page), encoding="utf-8")
        if set(grouped_names) != set(module.__all__):
            missing = sorted(set(module.__all__) - set(grouped_names))
            stale = sorted(set(grouped_names) - set(module.__all__))
            raise SystemExit(
                f"API grouping mismatch for {namespace}: missing={missing}, stale={stale}"
            )
        (group_dir / "index.md").write_text(
            "\n".join(namespace_index) + "\n", encoding="utf-8"
        )

    total = sum(len(module.__all__) for module in PUBLIC_MODULES.values())
    missing_rows = [row for row in rows if not row["example"]]
    index_lines = [
        "# API reference",
        "",
        "The API reference is generated from the public `__all__` contracts. Each symbol links to a runnable example.",
        "",
        "| Namespace | Public symbols | Reference |",
        "|---|---:|---|",
        f"| `pygwrx.models` | {len(models.__all__)} | [Models](models/index.md) |",
        f"| `pygwrx.core` | {len(core.__all__)} | [Core](core/index.md) |",
        f"| `pygwrx.diagnostics` | {len(diagnostics.__all__)} | [Diagnostics](diagnostics/index.md) |",
        f"| `pygwrx.plotting` | {len(plotting.__all__)} | [Plotting](plotting/index.md) |",
        f"| `pygwrx.io` | {len(io.__all__)} | [I/O](io/index.md) |",
        "",
        f"Coverage status: **{total - len(missing_rows)}/{total}**.",
        "",
        "See the [public API and example inventory](public-api-inventory.md) for the complete mapping.",
    ]
    (DOCS_API / "index.md").write_text("\n".join(index_lines) + "\n", encoding="utf-8")

    inventory = [
        "# Public API and example inventory",
        "",
        "This file is generated. Do not edit it manually.",
        "",
        "| Namespace | Symbol | Kind | Purpose | Runnable example |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        example = row["example"]
        link = f"[`{example}`]({REPOSITORY}/{example})" if example else "**Missing**"
        summary = row["summary"].replace("|", "\\|")
        inventory.append(
            f"| `pygwrx.{row['namespace']}` | `{row['symbol']}` | {row['kind']} | {summary} | {link} |"
        )
    (DOCS_API / "public-api-inventory.md").write_text(
        "\n".join(inventory) + "\n", encoding="utf-8"
    )

    (EXAMPLES / "API_COVERAGE.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (EXAMPLES / "API_COVERAGE.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["namespace", "symbol", "kind", "summary", "example"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)

    if missing_rows:
        names = ", ".join(f"{row['namespace']}.{row['symbol']}" for row in missing_rows)
        raise SystemExit(f"Public symbols without examples: {names}")
    print(f"Generated grouped API documentation and {total}/{total} coverage entries.")


if __name__ == "__main__":
    generate()
