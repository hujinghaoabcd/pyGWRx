from __future__ import annotations

import hashlib
from pathlib import Path

import pygwrx
from pygwrx.io import get_dataset_info, list_datasets, load_dataset

ROOT = Path(__file__).resolve().parents[1]


def _git_blob_sha1(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(
        f"blob {len(data)}\0".encode("ascii") + data,
        usedforsecurity=False,
    ).hexdigest()


def test_every_dataset_exposes_exact_provenance_metadata():
    required = {
        "license",
        "source_url",
        "source_version",
        "source_revision",
        "source_path",
        "evidence_date",
        "processing",
        "integrity",
    }
    for name in list_datasets(verbose=False):
        info = get_dataset_info(name)
        assert required <= info.keys()
        assert info["source_version"]
        assert info["source_revision"]
        assert info["source_path"]
        assert info["evidence_date"] == "2026-07-19"

        loaded = load_dataset(name, return_type="dict")
        for key in required:
            assert loaded[key] == info[key]


def test_fastsgwr_csv_bytes_match_pinned_git_blobs():
    expected = {
        "Crime/Crime.csv": "ac8ac10e020232a5293e7984c9e90ac440f91414",
        "HIV/HIV.csv": "cbe28a992be30dab5f7913f277d87672d5865d13",
        "Housing/Housing.csv": "35f4a3e7f8fea05d8f34a0c2bd03312afe74559e",
    }
    data_root = Path(pygwrx.__file__).resolve().parent / "data"
    for relative, blob_sha in expected.items():
        assert _git_blob_sha1(data_root / relative) == blob_sha


def test_data_hash_manifest_covers_and_verifies_every_bundled_file():
    manifest = ROOT / "DATA_HASHES.sha256"
    listed = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        listed[relative] = digest

    data_root = ROOT / "src" / "pygwrx" / "data"
    actual_paths = {
        path.relative_to(ROOT).as_posix()
        for path in data_root.rglob("*")
        if path.is_file()
    }
    assert set(listed) == actual_paths
    for relative, expected in listed.items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected
