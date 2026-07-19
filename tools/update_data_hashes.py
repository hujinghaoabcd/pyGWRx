# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Regenerate the bundled-data SHA-256 manifest deterministically.

Author:
    Jinghao Hu
"""

from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "src" / "pygwrx" / "data"
MANIFEST = ROOT / "DATA_HASHES.sha256"
_TEXT_DATA_SUFFIXES = {".cpg", ".csv", ".md", ".prj"}


def _canonical_integrity_bytes(path: Path) -> bytes:
    """Return platform-independent bytes for the integrity manifest."""
    data = path.read_bytes()
    if path.suffix.lower() in _TEXT_DATA_SUFFIXES:
        return data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return data


def main() -> None:
    """Hash every bundled-data file and rewrite the repository manifest."""
    lines = []
    for path in sorted(
        candidate for candidate in DATA_ROOT.rglob("*") if candidate.is_file()
    ):
        digest = hashlib.sha256(_canonical_integrity_bytes(path)).hexdigest()
        relative = path.relative_to(ROOT).as_posix()
        lines.append(f"{digest}  {relative}")
    MANIFEST.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(lines)} hashes to {MANIFEST.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
