# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT
"""Fail when a public API symbol lacks a runnable example mapping."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import pygwrx.core as core
import pygwrx.diagnostics as diagnostics
import pygwrx.io as io
import pygwrx.models as models
import pygwrx.plotting as plotting

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "examples" / "API_COVERAGE.json"
PUBLIC_MODULES: dict[str, Any] = {
    "models": models,
    "core": core,
    "diagnostics": diagnostics,
    "plotting": plotting,
    "io": io,
}


def _imports_symbol(path: Path, namespace: str, symbol: str) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    accepted_modules = {f"pygwrx.{namespace}"}
    if namespace == "models":
        accepted_modules.add("pygwrx")
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module in accepted_modules:
            if any(alias.name == symbol for alias in node.names):
                return True
    return False


def main() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    errors: list[str] = []
    total = 0
    for namespace, module in PUBLIC_MODULES.items():
        expected = set(module.__all__)
        recorded = set(manifest.get(namespace, {}))
        for name in sorted(expected - recorded):
            errors.append(f"missing manifest entry: {namespace}.{name}")
        for name in sorted(recorded - expected):
            errors.append(f"stale manifest entry: {namespace}.{name}")
        for name in sorted(expected & recorded):
            total += 1
            relative = manifest[namespace][name]
            path = ROOT / relative
            if not path.is_file():
                errors.append(f"missing example file: {namespace}.{name} -> {relative}")
            elif not _imports_symbol(path, namespace, name):
                errors.append(
                    f"example does not import symbol: {namespace}.{name} -> {relative}"
                )
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"Public API example coverage verified: {total}/{total} symbols.")


if __name__ == "__main__":
    main()
