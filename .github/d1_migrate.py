from __future__ import annotations

import ast
from pathlib import Path

RGWR_PATH = Path("src/pygwrx/models/rgwr.py")
GWR_PATH = Path("src/pygwrx/models/gwr.py")
TEST_PATH = Path("tests/test_rgwr_inheritance_contract.py")

rgwr = RGWR_PATH.read_text(encoding="utf-8")
gwr = GWR_PATH.read_text(encoding="utf-8")

rgwr = rgwr.replace(
    "The estimator reuses the validated Gaussian\n"
    "GWR calibration, inference, prediction, and result interfaces from\n"
    ":class:`pygwrx.models.GWR`.",
    "The estimator reuses the validated private standard-GWR execution engine\n"
    "for calibration, inference, and prediction while owning its estimator\n"
    "lifecycle and robust result interface directly.",
)
rgwr = rgwr.replace(
    "from typing import Callable, List, Optional, Tuple, Union",
    "from typing import Callable, Dict, Iterable, Iterator, List, Optional, Tuple, Union",
)

old_imports = """from pygwrx.core.metrics import compute_diagnostics
from pygwrx.core.utils import add_intercept
from pygwrx.models.gwr import GWR
"""
new_imports = """from pygwrx.core.base import BaseSpatialRegressor
from pygwrx.core.kernels import get_kernel_function
from pygwrx.core.metrics import compute_diagnostics
from pygwrx.core.solver import _weighted_least_squares_details
from pygwrx.core.utils import _iter_distance_rows as _iter_core_distance_rows
from pygwrx.core.utils import add_intercept, validate_coords
from pygwrx.models._gwr_engine import (
    _collect_gwr_inference,
    _compute_gwr_local_r2,
    _fit_gwr_prediction_locations,
    _fit_gwr_training_locations,
    _get_gwr_bandwidth_selector,
    _gwr_spatial_weights,
    _GWRLocalFitResult,
)
from pygwrx.models.gwr import GWRPredictionResult
"""
assert rgwr.count(old_imports) == 1
rgwr = rgwr.replace(old_imports, new_imports)

assert rgwr.count("class RGWR(GWR):") == 1
rgwr = rgwr.replace("class RGWR(GWR):", "class RGWR(BaseSpatialRegressor):")

old_init = """        super().__init__(
            kernel=kernel,
            bandwidth=bandwidth,
            bandwidth_method=bandwidth_method,
            adaptive=adaptive,
            bandwidth_range=bandwidth_range,
            optimization_method=optimization_method,
            fit_intercept=fit_intercept,
            distance_metric=distance_metric,
            sigma2_v1=sigma2_v1,
            verbose=verbose,
        )
"""
new_init = """        if not isinstance(sigma2_v1, (bool, np.bool_)):
            raise TypeError("sigma2_v1 must be boolean.")
        if isinstance(bandwidth, str) and bandwidth.strip().lower() == "adaptive":
            raise ValueError(
                "GWR uses adaptive=True to request a nearest-neighbour bandwidth; "
                "bandwidth must be numeric, None, or one of 'cv', 'aic', 'aicc', 'bic'."
            )
        super().__init__(
            kernel=kernel,
            bandwidth=bandwidth,
            bandwidth_method=bandwidth_method,
            adaptive=adaptive,
            bandwidth_range=bandwidth_range,
            optimization_method=optimization_method,
            fit_intercept=fit_intercept,
            distance_metric=distance_metric,
            verbose=verbose,
        )
        self.sigma2_v1 = bool(sigma2_v1)
        self.S_matrix_: Optional[np.ndarray] = None
        self.bandwidth_search_: Optional[Dict[str, object]] = None
        self._reset_inference_state()
"""
assert rgwr.count(old_init) == 1
rgwr = rgwr.replace(old_init, new_init)

gwr_lines = gwr.splitlines(keepends=True)
gwr_tree = ast.parse(gwr)
gwr_class = next(
    node
    for node in gwr_tree.body
    if isinstance(node, ast.ClassDef) and node.name == "GWR"
)


def extract_method(name: str) -> str:
    node = next(
        item
        for item in gwr_class.body
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
        and item.name == name
    )
    first = min(
        [node.lineno] + [decorator.lineno for decorator in node.decorator_list]
    )
    return "".join(gwr_lines[first - 1 : node.end_lineno]) + "\n"


support_names = [
    "_reset_inference_state",
    "_resolve_bandwidth",
    "_iter_distance_rows",
    "_warn_rank_deficiency",
    "_fit_training_locations",
    "_compute_local_r2_from_distance_rows",
    "_compute_local_r2_from_distances",
    "_compute_local_r2",
    "_set_inference_results",
]
support = "".join(extract_method(name) for name in support_names)

initial_fit = extract_method("fit")
initial_fit = initial_fit.replace("    def fit(", "    def _fit_initial_gwr(", 1)
initial_fit = initial_fit.replace(') -> "GWR":', ') -> "RGWR":', 1)
support += initial_fit

for name in (
    "predict",
    "_prediction_parameters",
    "predict_result",
    "get_local_parameters",
    "get_local_coefficients",
):
    support += extract_method(name)

gwr_frame = extract_method("to_frame")
gwr_frame = gwr_frame.replace("    def to_frame(", "    def _gwr_to_frame(", 1)
support += gwr_frame

gwr_summary = extract_method("summary")
gwr_summary = gwr_summary.replace("    def summary(", "    def _gwr_summary(", 1)
support += gwr_summary

reset_marker = """    def _reset_fit_state(self) -> None:
        super()._reset_fit_state()
        self._reset_robust_state()
"""
assert rgwr.count(reset_marker) == 1
new_reset = """    def _reset_fit_state(self) -> None:
        self._mark_unfitted()
        self._reset_regression_state()
        self._reset_gwr_state()
        self._reset_inference_state()
        self.S_matrix_ = None
        self.bandwidth_search_ = None
        self.n_samples_ = None
        self.n_features_in_ = None
        self.feature_names_in_ = None
        self._reset_robust_state()
"""
rgwr = rgwr.replace(reset_marker, support + new_reset)

old_weight = "        spatial_weights = super()._weights_from_distances(distances)"
new_weight = """        if self.bandwidth_ is None or self.kernel_func_ is None:
            raise RuntimeError("The fitted bandwidth and kernel are unavailable.")
        spatial_weights = _gwr_spatial_weights(
            distances,
            bandwidth=self.bandwidth_,
            adaptive=self.adaptive,
            kernel_func=self.kernel_func_,
        )"""
assert rgwr.count(old_weight) == 1
rgwr = rgwr.replace(old_weight, new_weight)

assert rgwr.count("            super().fit(\n") == 1
rgwr = rgwr.replace("            super().fit(\n", "            self._fit_initial_gwr(\n", 1)
assert rgwr.count("        frame = super().to_frame()") == 2
# The first occurrence belongs to the copied GWR frame helper and must keep
# BaseSpatialRegressor dispatch. The second is RGWR's robust extension.
pos = rgwr.rfind("        frame = super().to_frame()")
rgwr = rgwr[:pos] + "        frame = self._gwr_to_frame()" + rgwr[pos + len("        frame = super().to_frame()") :]
assert rgwr.count("        base_summary = super().summary()") == 1
rgwr = rgwr.replace(
    "        base_summary = super().summary()",
    "        base_summary = self._gwr_summary()",
    1,
)

assert "class RGWR(GWR)" not in rgwr
assert "from pygwrx.models.gwr import GWR\n" not in rgwr
assert "super().fit(" not in rgwr
ast.parse(rgwr)
RGWR_PATH.write_text(rgwr, encoding="utf-8")

TEST_PATH.write_text(
    '''# SPDX-FileCopyrightText: 2026 Jinghao Hu
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
''',
    encoding="utf-8",
)
ast.parse(TEST_PATH.read_text(encoding="utf-8"))
