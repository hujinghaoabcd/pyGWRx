# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Verify bundled-data hashes and pinned FastSGWR Git blob identities.

The check is intentionally offline. It verifies local release content against the
committed canonical-content record rather than downloading mutable upstream
resources.

Author:
    Jinghao Hu
"""

from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "DATA_HASHES.sha256"
_TEXT_DATA_SUFFIXES = {".cpg", ".csv", ".md", ".prj"}

FASTSGWR_BLOBS = {
    "src/pygwrx/data/Crime/Crime.csv": "ac8ac10e020232a5293e7984c9e90ac440f91414",
    "src/pygwrx/data/HIV/HIV.csv": "cbe28a992be30dab5f7913f277d87672d5865d13",
    "src/pygwrx/data/Housing/Housing.csv": "35f4a3e7f8fea05d8f34a0c2bd03312afe74559e",
}


def _canonical_integrity_bytes(path: Path) -> bytes:
    """Return LF-normalized text bytes and exact binary bytes."""
    data = path.read_bytes()
    if path.suffix.lower() in _TEXT_DATA_SUFFIXES:
        return data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return data


def _canonical_crlf_bytes(path: Path) -> bytes:
    """Restore canonical CRLF bytes for pinned FastSGWR comparison."""
    normalized = _canonical_integrity_bytes(path)
    return normalized.replace(b"\n", b"\r\n")


def _git_blob_sha1(data: bytes) -> str:
    """Return the Git blob object identifier for raw file bytes."""
    header = f"blob {len(data)}\0".encode("ascii")
    return hashlib.sha1(header + data, usedforsecurity=False).hexdigest()


def main() -> None:
    """Validate the SHA-256 manifest and exact pinned CSV blob identities."""
    errors = []
    listed_paths = set()
    for line_number, raw_line in enumerate(
        MANIFEST.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line:
            continue
        try:
            expected, relative = line.split("  ", 1)
        except ValueError:
            errors.append(f"line {line_number}: malformed manifest entry")
            continue
        path = ROOT / relative
        listed_paths.add(relative)
        if not path.is_file():
            errors.append(f"missing file: {relative}")
            continue
        actual = hashlib.sha256(_canonical_integrity_bytes(path)).hexdigest()
        if actual != expected:
            errors.append(
                f"SHA-256 mismatch for {relative}: expected {expected}, got {actual}"
            )

    actual_paths = {
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "src" / "pygwrx" / "data").rglob("*")
        if path.is_file()
    }
    for relative in sorted(actual_paths - listed_paths):
        errors.append(f"unlisted bundled-data file: {relative}")
    for relative in sorted(listed_paths - actual_paths):
        errors.append(f"manifest entry outside current bundled data: {relative}")

    for relative, expected in FASTSGWR_BLOBS.items():
        path = ROOT / relative
        actual = _git_blob_sha1(_canonical_crlf_bytes(path))
        if actual != expected:
            errors.append(
                f"Git blob mismatch for {relative}: expected {expected}, got {actual}"
            )

    if errors:
        raise SystemExit("Bundled-data verification failed:\n- " + "\n- ".join(errors))

    print(
        f"Verified {len(listed_paths)} SHA-256 entries and "
        f"{len(FASTSGWR_BLOBS)} pinned Git blobs."
    )


if __name__ == "__main__":
    main()
