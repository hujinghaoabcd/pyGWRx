# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT
"""Freeze the Tier-A public estimator API/capability surface during refactoring."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pygwrx
import pygwrx.models as models

ROOT = Path(__file__).resolve().parents[1]
ESTIMATOR_NAMES = (
    "GWR",
    "MGWR",
    "RGWR",
    "STWR",
    "GTWR",
    "GWGLM",
    "GWLasso",
    "MixedGWR",
    "GWPCA",
    "GWDA",
    "GWSS",
    "ScalableGWR",
    "LCRGWR",
    "BootstrapGWR",
    "SGWR",
    "SGTWR",
    "MGTWR",
    "LGGWR",
    "GRGWR",
)


def test_all_19_estimators_are_public_from_root_and_models() -> None:
    assert len(ESTIMATOR_NAMES) == 19
    for name in ESTIMATOR_NAMES:
        assert name in pygwrx.__all__
        assert name in models.__all__
        assert getattr(pygwrx, name) is getattr(models, name)


def test_frozen_estimator_contract_matches_runtime_api() -> None:
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "architecture" / "generate_estimator_contract.py"),
            "--check",
        ],
        cwd=ROOT,
        check=True,
    )
