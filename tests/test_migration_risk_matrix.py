"""Schema/coverage tests for the A3 migration-risk planning artifact."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).parents[1]
CONTRACT_DIR = ROOT / "architecture_contracts"


def _load(name: str):
    return json.loads((CONTRACT_DIR / name).read_text(encoding="utf-8"))


def test_migration_risk_matrix_covers_exactly_the_a1_estimators():
    api_contract = _load("estimators.json")
    risk_contract = _load("migration_risks.json")

    assert risk_contract["schema_version"] == 1
    assert risk_contract["baseline_main_sha"] == (
        "b655688f7201aaa9677fe153f2cbc15e6e63afb6"
    )
    assert set(risk_contract["estimators"]) == set(api_contract["estimators"])
    assert len(risk_contract["estimators"]) == api_contract["estimator_count"] == 19


def test_every_estimator_has_actionable_risk_metadata():
    contract = _load("migration_risks.json")
    allowed_roles = {"regressor", "classifier", "transformer", "statistics", "inference"}
    allowed_risks = {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
    required = {
        "role",
        "source",
        "current_base",
        "risk",
        "phase",
        "dependencies",
        "numerical_gate",
        "execution",
        "specialized",
        "debts",
        "prerequisites",
        "rationale",
    }

    for name, entry in contract["estimators"].items():
        assert required <= entry.keys(), name
        assert entry["role"] in allowed_roles, name
        assert entry["risk"] in allowed_risks, name
        assert entry["source"].startswith("src/pygwrx/models/"), name
        assert entry["source"].endswith(".py"), name
        assert isinstance(entry["dependencies"], list), name
        assert isinstance(entry["specialized"], list) and entry["specialized"], name
        assert isinstance(entry["debts"], list) and entry["debts"], name
        assert isinstance(entry["prerequisites"], list) and entry["prerequisites"], name
        assert entry["numerical_gate"].strip(), name
        assert entry["execution"].strip(), name
        assert entry["rationale"].strip(), name


def test_a3_is_not_a_current_mro_compatibility_freeze():
    contract = _load("migration_risks.json")
    serialized = json.dumps(contract, sort_keys=True)

    # A3 records current_base for planning evidence only. It must not grow an
    # expected-MRO or inheritance-lock field that would conflict with the final
    # architecture decision to remove concrete public-estimator inheritance.
    assert "expected_mro" not in serialized
    assert "mro_contract" not in serialized
    assert "inheritance_lock" not in serialized


def test_human_readable_matrix_mentions_all_estimators():
    contract = _load("migration_risks.json")
    document = (CONTRACT_DIR / "MIGRATION_RISK_MATRIX.md").read_text(encoding="utf-8")

    for name in contract["estimators"]:
        assert name in document


def test_known_concrete_inheritance_debts_are_explicit():
    contract = _load("migration_risks.json")["estimators"]

    expected = {
        "RGWR": "inherits public GWR",
        "LCRGWR": "inherits public GWR",
        "GWGLM": "inherits public GWR",
        "MGTWR": "inherits public MGWR",
    }
    for name, dependency in expected.items():
        assert dependency in contract[name]["dependencies"]
        assert contract[name]["risk"] in {"HIGH", "CRITICAL"}


def test_special_execution_models_remain_visible_in_planning_contract():
    contract = _load("migration_risks.json")["estimators"]

    assert "streamed" in contract["GWR"]["execution"]
    assert "dense" in contract["MGWR"]["execution"]
    assert "cKDTree" in contract["ScalableGWR"]["execution"]
    assert "n×n" in contract["SGWR"]["execution"]
    assert "learned latent geometry" in contract["LGGWR"]["execution"]
