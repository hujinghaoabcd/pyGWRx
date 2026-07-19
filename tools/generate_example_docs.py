# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Generate detailed documentation pages from maintained runnable examples."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
DOCS = ROOT / "docs" / "examples"
REPOSITORY = "https://github.com/hujinghaoabcd/pyGWRx/blob/main"

CATEGORY_INFO = {
    "models": {
        "title": "Model examples",
        "purpose": "One isolated, runnable script for every supported public model.",
        "inspect": "Inspect fitted attributes, the model-specific result contract, diagnostics, and any documented prediction limitation.",
    },
    "core": {
        "title": "Core numerical examples",
        "purpose": "Kernels, distances, validation, solvers, metrics, optimisation, bandwidth selection, and shared base classes.",
        "inspect": "Compare shapes, numerical return values, convergence objects, selected bandwidths, and error-handling behaviour.",
    },
    "diagnostics": {
        "title": "Diagnostics examples",
        "purpose": "Global summaries, local inference, collinearity, influence, residual, temporal, weight, and regime diagnostics.",
        "inspect": "Check that the reported statistic matches the fitted model contract and interpret thresholds together with spatial context.",
    },
    "plotting": {
        "title": "Plotting examples",
        "purpose": "Every public plotting function, including model-aware, array-compatible, temporal, robust, multivariate, and research-model plots.",
        "inspect": "Review the generated files under examples/output, axis labels, mapped quantities, significance masks, and model-specific annotations.",
    },
    "io": {
        "title": "I/O examples",
        "purpose": "Bundled datasets, DataFrame/array conversion, GeoDataFrame round trips, and supported persistence helpers.",
        "inspect": "Verify column names, dtypes, coordinate order, CRS preservation, index alignment, and round-trip equality.",
    },
    "workflows": {
        "title": "Workflow examples",
        "purpose": "Multi-step analyses that combine data preparation, fitting, diagnostics, comparison, prediction, and export.",
        "inspect": "Follow the order of operations and note where validation, interpretation, and capability checks occur before export.",
    },
}

MODEL_GUIDES = {
    "01_gwr.py": "gwr.md",
    "02_mgwr.py": "mgwr.md",
    "03_rgwr.py": "rgwr.md",
    "04_stwr.py": "stwr.md",
    "05_gtwr.py": "gtwr.md",
    "06_gwglm.py": "gwglm.md",
    "07_gw_lasso.py": "gw-lasso.md",
    "08_mixed_gwr.py": "mixed-gwr.md",
    "09_gwpca.py": "gwpca.md",
    "10_gwda.py": "gwda.md",
    "11_gwss.py": "gwss.md",
    "12_scalable_gwr.py": "scalable-gwr.md",
    "13_lcr_gwr.py": "lcr-gwr.md",
    "14_bootstrap_gwr.py": "bootstrap-gwr.md",
    "15_sgwr.py": "sgwr.md",
    "16_sgtwr.py": "sgtwr.md",
    "17_mgtwr.py": "mgtwr.md",
    "18_lg_gwr.py": "lg-gwr.md",
    "19_gr_gwr.py": "gr-gwr.md",
}


def docstring(tree: ast.Module) -> str:
    """Return the module docstring as a one-line purpose statement."""
    return (ast.get_docstring(tree) or "Runnable pyGWRx example.").replace("\n", " ")


def imported_symbols(tree: ast.Module) -> list[str]:
    """Return public pyGWRx imports in source order without duplicates."""
    names: list[str] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.ImportFrom)
            and node.module
            and node.module.startswith("pygwrx")
        ):
            for alias in node.names:
                if alias.name != "*" and alias.name not in names:
                    names.append(alias.name)
    return names


def requirements(source: str, category: str) -> str:
    """Infer optional extras beyond the complete base installation."""
    extras: list[str] = []
    if any(name in source for name in ("GWLasso", "GWPCA", "GRGWR")):
        extras.append("ml")
    extras = list(dict.fromkeys(extras))
    if not extras:
        return "base installation"
    joined = ",".join(extras)
    return f'`pip install -e ".[{joined}]"`'


def render_category(category: str) -> None:
    """Generate one detailed catalog page for an example category."""
    info = CATEGORY_INFO[category]
    files = sorted((EXAMPLES / category).glob("*.py"))
    files = [path for path in files if not path.name.startswith("_")]
    lines = [
        f'# {info["title"]}',
        "",
        info["purpose"],
        "",
        f"This page embeds **{len(files)}** maintained scripts. The code shown here is read directly from `examples/{category}/`, so the documentation and executable source cannot silently diverge.",
        "",
        '!!! tip "How to use this catalog"',
        "    Read the purpose and APIs first, run the exact command, then use the inspection note to decide which output matters. For conceptual interpretation, follow the linked model or function guide rather than reading code alone.",
        "",
    ]
    for path in files:
        source = path.read_text(encoding="utf-8").rstrip()
        tree = ast.parse(source, filename=str(path))
        purpose = docstring(tree)
        symbols = imported_symbols(tree)
        relative = path.relative_to(ROOT).as_posix()
        lines.extend(
            [
                f"## `{path.name}`",
                "",
                f"**Purpose.** {purpose}",
                "",
                f"**Public APIs exercised.** {', '.join(f'`{name}`' for name in symbols) if symbols else 'No direct public import; this script composes shared example helpers.'}",
                "",
                f"**Environment.** {requirements(source, category)}.",
                "",
                f"**Run.** `python {relative}`",
                "",
                f"**What to inspect.** {info['inspect']}",
                "",
            ]
        )
        if category == "models" and path.name in MODEL_GUIDES:
            lines.extend(
                [
                    f"[Detailed model guide](../models/{MODEL_GUIDES[path.name]}){{ .md-button .md-button--primary }}",
                    f"[Chinese guide](../zh/models/{MODEL_GUIDES[path.name]}){{ .md-button }}",
                    f"[Open source]({REPOSITORY}/{relative}){{ .md-button }}",
                    "",
                ]
            )
        else:
            lines.extend(
                [
                    f"[Open source]({REPOSITORY}/{relative}){{ .md-button }}",
                    "",
                ]
            )
        lines.extend(["```python", source, "```", ""])
    (DOCS / f"{category}.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Generate all example catalog pages."""
    DOCS.mkdir(parents=True, exist_ok=True)
    for category in CATEGORY_INFO:
        render_category(category)
    print("Generated detailed documentation for 45 maintained examples.")


if __name__ == "__main__":
    main()
