# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT
"""Generate the frozen public-estimator API/capability contract for refactors."""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
from pathlib import Path
from typing import Any

import pygwrx
import pygwrx.models as models

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "architecture_contracts" / "estimators.json"

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

CAPABILITY_METHODS = (
    "fit",
    "predict",
    "predict_result",
    "predict_proba",
    "transform",
    "score",
    "summary",
    "to_frame",
    "results_frame",
    "to_geodataframe",
    "select_bandwidth",
)

HEAVY_OUTPUT_CONTROLS = {
    "compute_hat_matrix",
    "compute_hat_matrix_flag",
    "compute_inference",
    "compute_local_r2",
    "compute_scores",
    "store_weights",
    "store_local_bootstrap",
}

KNOWN_DEPRECATIONS: dict[str, list[str]] = {
    "GWR": ["fit(compute_hat_matrix_flag=...) is a legacy compatibility alias"],
    "LGGWR": ["orthogonal_constraint is deprecated; use scale_constraint"],
}

RESULT_FRAME_SCHEMAS: dict[str, dict[str, list[str]]] = {
    "GWRPredictionResult": {
        "always": ["coord_0", "coord_1", "prediction", "intercept"],
        "optional": [
            "intercept_se",
            "intercept_t",
            "local_rank",
            "local_condition_number",
            "rank_deficient",
        ],
        "per_feature": ["coef_{feature}", "se_{feature}", "t_{feature}"],
    },
    "GTWRPredictionResult": {
        "always": ["coord_0", "coord_1", "time", "prediction", "intercept"],
        "optional": ["intercept_se", "intercept_t"],
        "per_feature": ["coef_{feature}", "se_{feature}", "t_{feature}"],
    },
    "GWGLMPredictionResult": {
        "always": [
            "coord_0",
            "coord_1",
            "prediction",
            "linear_predictor",
            "intercept",
        ],
        "optional": ["exposure", "intercept_se", "intercept_z"],
        "per_feature": ["coef_{feature}", "se_{feature}", "z_{feature}"],
    },
    "STWRPredictionResult": {
        "always": [
            "coord_0",
            "coord_1",
            "prediction",
            "reference_y",
            "intercept",
        ],
        "optional": [],
        "per_feature": ["coef_{feature}"],
    },
    "SGTWRPredictionResult": {
        "always": ["coord_0", "coord_1", "time", "prediction", "intercept"],
        "optional": [],
        "per_feature": ["coef_{feature}"],
    },
    "LGGWRPredictionResult": {
        "always": ["coord_0", "coord_1", "prediction", "intercept"],
        "optional": [],
        "per_feature": ["coef_{feature}"],
        "repeated": ["latent_{index}"],
    },
    "GRGWRPredictionResult": {
        "always": ["coord_0", "coord_1", "prediction", "regime", "intercept"],
        "optional": [],
        "per_feature": ["coef_{feature}"],
    },
}


def _default_repr(value: Any) -> str:
    if value is inspect.Signature.empty:
        return "<required>"
    return repr(value)


def _signature_contract(value: Any) -> list[dict[str, str]] | None:
    if value is None:
        return None
    try:
        signature = inspect.signature(value)
    except (TypeError, ValueError):
        return None
    return [
        {
            "name": parameter.name,
            "kind": parameter.kind.name,
            "default": _default_repr(parameter.default),
        }
        for parameter in signature.parameters.values()
    ]


def _public_fitted_state(instance: object) -> list[str]:
    """Return public trailing-underscore attributes initialized by the estimator."""
    return sorted(
        name
        for name in vars(instance)
        if name.endswith("_") and not name.startswith("_")
    )


def _method_contract(cls: type, name: str) -> dict[str, Any] | None:
    method = getattr(cls, name, None)
    if method is None or not callable(method):
        return None
    return {"parameters": _signature_contract(method)}


def _heavy_controls(cls: type) -> list[dict[str, str]]:
    controls: list[dict[str, str]] = []
    callables: list[tuple[str, Any]] = [("constructor", cls)]
    for name in CAPABILITY_METHODS:
        value = getattr(cls, name, None)
        if callable(value):
            callables.append((name, value))
    for location, value in callables:
        try:
            parameters = inspect.signature(value).parameters
        except (TypeError, ValueError):
            continue
        for parameter in parameters:
            if parameter in HEAVY_OUTPUT_CONTROLS:
                controls.append({"location": location, "parameter": parameter})
    return sorted(controls, key=lambda item: (item["location"], item["parameter"]))


def _prediction_result(cls: type) -> dict[str, Any] | None:
    method = getattr(cls, "predict_result", None)
    if method is None:
        return None
    annotation = inspect.signature(method).return_annotation
    if annotation is inspect.Signature.empty:
        return None
    name = (
        annotation
        if isinstance(annotation, str)
        else getattr(annotation, "__name__", str(annotation))
    )
    name = str(name).strip("'\"")
    module = importlib.import_module(cls.__module__)
    result_cls = getattr(module, name, None)
    if result_cls is None or not inspect.isclass(result_cls):
        return {"name": name, "constructor": None, "frame_schema": None}
    return {
        "name": name,
        "constructor": _signature_contract(result_cls),
        "frame_schema": RESULT_FRAME_SCHEMAS.get(name),
    }


def generate_contract() -> dict[str, Any]:
    estimators: dict[str, Any] = {}
    for name in ESTIMATOR_NAMES:
        cls = getattr(models, name)
        root_cls = getattr(pygwrx, name)
        if root_cls is not cls:
            raise RuntimeError(
                f"pygwrx.{name} and pygwrx.models.{name} are not identical"
            )
        instance = cls()
        methods = {
            method_name: contract
            for method_name in CAPABILITY_METHODS
            if (contract := _method_contract(cls, method_name)) is not None
        }
        estimators[name] = {
            "public_module": cls.__module__,
            "root_import": f"pygwrx.{name}",
            "models_import": f"pygwrx.models.{name}",
            "constructor": _signature_contract(cls),
            "methods": methods,
            "initialized_public_fitted_state": _public_fitted_state(instance),
            "prediction_result": _prediction_result(cls),
            "heavy_output_controls": _heavy_controls(cls),
            "deprecations": KNOWN_DEPRECATIONS.get(name, []),
        }
    return {
        "schema_version": 1,
        "purpose": "Tier-A public estimator API/capability freeze for the pyGWRx 0.2 architecture refactor",
        "estimator_count": len(ESTIMATOR_NAMES),
        "estimators": estimators,
        "non_contractual": [
            "MRO and concrete parent classes",
            "private attributes and private helper signatures",
            "internal engine/provider/policy implementation",
        ],
    }


def render_contract() -> str:
    return json.dumps(generate_contract(), indent=2, sort_keys=True) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    text = render_contract()
    if args.check:
        if not args.output.exists():
            print(text)
            raise SystemExit(
                "Estimator contract file is missing; generated contract printed above."
            )
        expected = args.output.read_text(encoding="utf-8")
        if expected != text:
            print("--- GENERATED ESTIMATOR CONTRACT ---")
            print(text)
            raise SystemExit(
                "Estimator architecture contract changed. Regenerate intentionally with "
                "`python tools/architecture/generate_estimator_contract.py` and review the diff."
            )
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
