# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Finalize and validate the release CycloneDX SBOM."""

from __future__ import annotations

import argparse
import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any


def _component_name(component: dict[str, Any]) -> str:
    return str(component.get("name", "")).strip().lower()


def main() -> None:
    """Add exact pyGWRx identity and reject removed dependencies."""
    parser = argparse.ArgumentParser()
    parser.add_argument("sbom", type=Path)
    args = parser.parse_args()

    try:
        package_version = version("pyGWRx")
    except PackageNotFoundError as exc:
        raise RuntimeError(
            "Run finalize_sbom.py with a Python environment containing pyGWRx."
        ) from exc

    bom = json.loads(args.sbom.read_text(encoding="utf-8"))
    metadata = bom.setdefault("metadata", {})
    root = metadata.setdefault("component", {})
    root["name"] = "pyGWRx"
    root["type"] = "library"
    root["version"] = package_version
    root["purl"] = f"pkg:pypi/pygwrx@{package_version}"

    components = bom.get("components", [])
    all_components = [root, *components]
    removed = [item for item in all_components if _component_name(item) == "mgtwr"]
    if removed:
        raise RuntimeError("The removed external mgtwr package is present in the SBOM.")

    args.sbom.write_text(
        json.dumps(bom, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"Finalized {args.sbom} for pyGWRx {package_version}; "
        f"validated {len(components)} dependency components."
    )


if __name__ == "__main__":
    main()
