# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Verify release archive contents and forbidden dependency metadata."""

from __future__ import annotations

import argparse
import tarfile
import zipfile
from pathlib import Path

REQUIRED_WHEEL_SUFFIXES = (
    "pygwrx/py.typed",
    "pygwrx/data/LICENSES.md",
    "pygwrx/data/Columbus/columbus.csv",
)
REQUIRED_SDIST_SUFFIXES = (
    "pyproject.toml",
    "SECURITY.md",
    "DATA_LICENSES.md",
    "DATA_PROVENANCE.md",
    "DATA_HASHES.sha256",
    "THIRD_PARTY_NOTICES.md",
    "pyGWRx_0.1.2_release_notes.md",
    "MODIFICATION_AND_VALIDATION_REPORT.md",
    "HANDOFF_NEXT_CONVERSATION.md",
    "src/pygwrx/py.typed",
    "src/pygwrx/data/LICENSES.md",
    "tests/test_mgtwr.py",
    "tests/reference_data/mgtwr_fixed_gaussian_reference.json",
    "tools/smoke_installed_distribution.py",
    ".pre-commit-config.yaml",
)
FORBIDDEN_METADATA = ("Requires-Dist: mgtwr", "Provides-Extra: mgtwr")


def _contains_suffix(names: list[str], suffix: str) -> bool:
    return any(name.endswith(suffix) for name in names)


def verify_wheel(path: Path) -> None:
    """Verify one wheel's files and metadata."""
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        for suffix in REQUIRED_WHEEL_SUFFIXES:
            if not _contains_suffix(names, suffix):
                raise RuntimeError(f"{path.name} is missing {suffix!r}.")
        metadata_names = [
            name for name in names if name.endswith(".dist-info/METADATA")
        ]
        if len(metadata_names) != 1:
            raise RuntimeError(f"{path.name} must contain exactly one METADATA file.")
        metadata = archive.read(metadata_names[0]).decode("utf-8")
        for token in FORBIDDEN_METADATA:
            if token in metadata:
                raise RuntimeError(f"{path.name} contains forbidden metadata: {token}")


def verify_sdist(path: Path) -> None:
    """Verify one source distribution's files and metadata."""
    with tarfile.open(path, "r:gz") as archive:
        members = archive.getmembers()
        names = [member.name for member in members]
        for suffix in REQUIRED_SDIST_SUFFIXES:
            if not _contains_suffix(names, suffix):
                raise RuntimeError(f"{path.name} is missing {suffix!r}.")
        pkg_info = [
            member
            for member in members
            if member.name.endswith("/PKG-INFO") and len(Path(member.name).parts) == 2
        ]
        if len(pkg_info) != 1:
            raise RuntimeError(f"{path.name} must contain exactly one root PKG-INFO.")
        extracted = archive.extractfile(pkg_info[0])
        if extracted is None:
            raise RuntimeError(f"Unable to read PKG-INFO from {path.name}.")
        metadata = extracted.read().decode("utf-8")
        for token in FORBIDDEN_METADATA:
            if token in metadata:
                raise RuntimeError(f"{path.name} contains forbidden metadata: {token}")


def main() -> None:
    """Validate exactly one wheel and one source distribution."""
    parser = argparse.ArgumentParser()
    parser.add_argument("dist_dir", type=Path)
    args = parser.parse_args()
    wheels = sorted(args.dist_dir.glob("*.whl"))
    sdists = sorted(args.dist_dir.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        raise RuntimeError("Expected exactly one wheel and one .tar.gz sdist.")
    verify_wheel(wheels[0])
    verify_sdist(sdists[0])
    print(f"Verified {wheels[0].name} and {sdists[0].name}.")


if __name__ == "__main__":
    main()
