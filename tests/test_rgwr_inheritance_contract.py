# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""D1 contracts for removing RGWR's concrete GWR inheritance."""

from __future__ import annotations

__author__ = "Jinghao Hu"
__license__ = "MIT"

import ast
from pathlib import Path

import numpy as np

from pygwrx import GWR, RGWR
from pygwrx.core.base import BaseSpatialRegressor
from pygwrx.models.gwr import GWRPredictionResult

REPO_ROOT = Path(__file__).resolve().parents[1]
RGWR_SOURCE = REPO_ROOT / "src" / "pygwrx" / "models" / "rgwr.py"


def test_rgwr_no_longer_inherits_concrete_gwr() -> None:
    assert RGWR.__bases__ == (BaseSpatialRegressor,)
    assert GWR not in RGWR.__mro__
    assert not issubclass(RGWR, GWR)


def test_rgwr_source_uses_gwr_only_for_public_result_type() -> None:
    tree = ast.parse(RGWR_SOURCE.read_text(encoding="utf-8"))
    gwr_imports = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == "pygwrx.models.gwr"
    ]
    assert len(gwr_imports) == 1
    assert {alias.name for alias in gwr_imports[0].names} == {"GWRPredictionResult"}


def test_rgwr_reuses_private_gwr_engine() -> None:
    tree = ast.parse(RGWR_SOURCE.read_text(encoding="utf-8"))
    engine_imports = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module == "pygwrx.models._gwr_engine"
    ]
    assert len(engine_imports) == 1
    imported = {alias.name for alias in engine_imports[0].names}
    assert "_fit_gwr_training_locations" in imported
    assert "_fit_gwr_prediction_locations" in imported
    assert "_collect_gwr_inference" in imported


def test_rgwr_prediction_result_contract_survives_composition() -> None:
    rng = np.random.default_rng(731)
    coords = rng.uniform(size=(28, 2))
    X = rng.normal(size=(28, 2))
    y = 0.8 + 1.3 * X[:, 0] - 0.5 * X[:, 1] + rng.normal(scale=0.08, size=28)

    model = RGWR(
        kernel="bisquare",
        bandwidth=18,
        adaptive=True,
        method="automatic",
        cut1=20.0,
        cut2=30.0,
        tol=1.0e-10,
    ).fit(X, y, coords, compute_local_r2=False)
    result = model.predict_result(X[:3], coords[:3])

    assert isinstance(result, GWRPredictionResult)
    assert result.predictions.shape == (3,)
    assert result.coef.shape == (3, 2)
    assert result.local_rank is not None
