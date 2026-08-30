# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""C3 contract tests for the non-growing protected GWR surface."""

from __future__ import annotations

__author__ = "Jinghao Hu"
__license__ = "MIT"

import ast
import inspect
import json
from pathlib import Path

from pygwrx import GWR

REPO_ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = REPO_ROOT / "architecture_contracts" / "gwr_protected_surface.json"
GWR_SOURCE_PATH = REPO_ROOT / "src" / "pygwrx" / "models" / "gwr.py"

_FROZEN_C3_MAXIMUM = {
    "_compute_local_r2",
    "_compute_local_r2_from_distance_rows",
    "_compute_local_r2_from_distances",
    "_fit_training_locations",
    "_iter_distance_rows",
    "_prediction_parameters",
    "_reset_fit_state",
    "_reset_inference_state",
    "_resolve_bandwidth",
    "_set_inference_results",
    "_warn_rank_deficiency",
    "_weights_from_distances",
}


def _load_contract() -> dict[str, object]:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def _source_declared_protected_methods() -> set[str]:
    tree = ast.parse(GWR_SOURCE_PATH.read_text(encoding="utf-8"))
    gwr_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "GWR"
    )
    return {
        node.name
        for node in gwr_class.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("_")
        and not node.name.startswith("__")
    }


def _runtime_declared_protected_methods() -> set[str]:
    names: set[str] = set()
    for name, value in GWR.__dict__.items():
        if not name.startswith("_") or name.startswith("__"):
            continue
        if isinstance(value, (staticmethod, classmethod)):
            value = value.__func__
        if inspect.isfunction(value):
            names.add(name)
    return names


def test_c3_contract_is_nonexpanding_and_removal_safe() -> None:
    contract = _load_contract()
    policy = contract["policy"]

    assert contract["phase"] == "C3"
    assert contract["target"] == "pygwrx.models.gwr.GWR"
    assert isinstance(policy, dict)
    assert policy["protected_method_additions_allowed"] is False
    assert policy["protected_method_removals_allowed"] is True
    assert policy["inherited_methods_in_scope"] is False
    assert policy["dunder_methods_in_scope"] is False
    assert policy["preferred_implementation_owner"] == "pygwrx.models._gwr_engine"


def test_frozen_maximum_matches_c3_baseline() -> None:
    contract = _load_contract()
    frozen = set(contract["frozen_maximum_protected_methods"])

    assert frozen == _FROZEN_C3_MAXIMUM
    assert len(frozen) == 12


def test_gwr_declared_protected_surface_cannot_grow() -> None:
    source_methods = _source_declared_protected_methods()
    runtime_methods = _runtime_declared_protected_methods()

    assert source_methods == runtime_methods
    unexpected = source_methods - _FROZEN_C3_MAXIMUM
    assert not unexpected, (
        "C3 forbids adding new protected implementation methods to GWR; "
        f"move new helper logic to the private GWR engine or model-owned module: "
        f"{sorted(unexpected)}"
    )


def test_c3_contract_allows_later_removal_but_not_replacement() -> None:
    current = _source_declared_protected_methods()

    # D1/D2/D3 may shrink the transitional surface. The C3 maximum deliberately
    # remains unchanged so removed hooks cannot later be replaced by new ones.
    assert current <= _FROZEN_C3_MAXIMUM
    assert (REPO_ROOT / "src" / "pygwrx" / "models" / "_gwr_engine.py").is_file()
