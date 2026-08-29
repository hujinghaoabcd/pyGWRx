"""Cross-model fitted-state atomicity contract.

This safety-freeze test links the A1 public estimator inventory to A2 lifecycle
semantics.  Every public estimator must discard fitted results when a later
``fit`` attempt fails; an object must never expose a mixture of results from a
previous successful fit and a partially completed failed refit.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import pytest

import pygwrx

_CONTRACT_PATH = (
    Path(__file__).resolve().parents[1] / "architecture_contracts" / "estimators.json"
)
_CONTRACT = json.loads(_CONTRACT_PATH.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class _FitCase:
    factory: Callable[[], Any]
    success: Callable[[Any], None]
    failure: Callable[[Any], None]


def _regression_data(seed: int = 20260829, n: int = 36):
    rng = np.random.default_rng(seed)
    coords = rng.uniform(0.0, 10.0, size=(n, 2))
    times = np.linspace(0.0, 4.0, n)
    X = rng.normal(size=(n, 3))
    y = 1.1 + 1.4 * X[:, 0] - 0.7 * X[:, 1] + 0.35 * X[:, 2]
    y += 0.08 * coords[:, 0] + rng.normal(0.0, 0.08, size=n)
    attrs = rng.normal(size=(n, 2))
    return X, y, coords, times, attrs


def _classification_data(seed: int = 117, n_per_class: int = 12):
    rng = np.random.default_rng(seed)
    labels = np.repeat(np.asarray(["A", "B", "C"], dtype=object), n_per_class)
    X = np.vstack(
        [
            rng.normal(loc=center, scale=0.45, size=(n_per_class, 2))
            for center in ((-1.8, 0.0), (1.8, 0.0), (0.0, 2.4))
        ]
    )
    coords = np.column_stack(
        [np.linspace(0.0, 8.0, labels.size), np.sin(np.linspace(0.0, 4.0, labels.size))]
    )
    order = np.arange(labels.size).reshape(3, n_per_class).T.reshape(-1)
    return X[order], labels[order], coords[order]


def _stage_data(seed: int = 12, n: int = 12):
    rng = np.random.default_rng(seed)
    coords = np.column_stack([np.linspace(0.0, 5.0, n), np.zeros(n)])
    X_list = []
    y_list = []
    coords_list = []
    for stage in range(3):
        X = rng.normal(size=(n, 2))
        y = 1.0 + 0.15 * stage + 1.2 * X[:, 0] - 0.5 * X[:, 1]
        X_list.append(X)
        y_list.append(y)
        coords_list.append(coords.copy())
    return X_list, y_list, coords_list, [0.0, 1.0, 1.0]


def _snapshot(value: Any) -> Any:
    return copy.deepcopy(value)


def _assert_value_matches_baseline(actual: Any, expected: Any, *, label: str) -> None:
    if isinstance(expected, np.ndarray):
        assert isinstance(actual, np.ndarray), label
        np.testing.assert_equal(actual, expected, err_msg=label)
        return
    if isinstance(expected, pd.DataFrame):
        pd.testing.assert_frame_equal(actual, expected, obj=label)
        return
    if isinstance(expected, pd.Series):
        pd.testing.assert_series_equal(actual, expected, obj=label)
        return
    if isinstance(expected, dict):
        assert isinstance(actual, dict) and actual.keys() == expected.keys(), label
        for key in expected:
            _assert_value_matches_baseline(
                actual[key], expected[key], label=f"{label}[{key!r}]"
            )
        return
    if isinstance(expected, (list, tuple)):
        assert type(actual) is type(expected) and len(actual) == len(expected), label
        for index, (actual_item, expected_item) in enumerate(zip(actual, expected)):
            _assert_value_matches_baseline(
                actual_item, expected_item, label=f"{label}[{index}]"
            )
        return
    if isinstance(expected, (float, np.floating)) and np.isnan(expected):
        assert isinstance(actual, (float, np.floating)) and np.isnan(actual), label
        return
    assert actual == expected, label


def _bad_nan_refit(
    model: Any, X: np.ndarray, y: np.ndarray, coords: np.ndarray
) -> None:
    bad = X.copy()
    bad[0, 0] = np.nan
    model.fit(bad, y, coords)


def _bad_nan_unsupervised_refit(model: Any, X: np.ndarray, coords: np.ndarray) -> None:
    bad = X.copy()
    bad[0, 0] = np.nan
    model.fit(bad, coords)


def _build_cases() -> dict[str, _FitCase]:
    X, y, coords, times, attrs = _regression_data()
    X_class, y_class, coords_class = _classification_data()
    X_list, y_list, coords_list, intervals = _stage_data()

    mgwr_bandwidths = [18] * (X.shape[1] + 1)

    return {
        "GWR": _FitCase(
            lambda: pygwrx.GWR(kernel="gaussian", bandwidth=4.0),
            lambda model: model.fit(X, y, coords, compute_local_r2=False),
            lambda model: model.fit(X, y, coords, compute_hat_matrix="yes"),
        ),
        "MGWR": _FitCase(
            lambda: pygwrx.MGWR(
                bandwidths=mgwr_bandwidths,
                adaptive=True,
                init_bandwidth=18,
                max_iter=3,
                tol=1e-4,
                verbose=False,
            ),
            lambda model: model.fit(X, y, coords, compute_inference=False),
            lambda model: model.fit(X, y, coords, compute_hat_matrix="yes"),
        ),
        "RGWR": _FitCase(
            lambda: pygwrx.RGWR(
                kernel="gaussian",
                bandwidth=18,
                adaptive=True,
                max_iter=2,
                verbose=False,
            ),
            lambda model: model.fit(X, y, coords, compute_local_r2=False),
            lambda model: model.fit(X, y, coords, compute_hat_matrix="yes"),
        ),
        "GTWR": _FitCase(
            lambda: pygwrx.GTWR(
                kernel="gaussian",
                bandwidth=4.0,
                lambda_st=0.5,
                causal=False,
            ),
            lambda model: model.fit(
                X, y, coords, times, compute_local_r2=False, compute_inference=False
            ),
            lambda model: model.fit(X[:-1], y, coords[:-1], times[:-1]),
        ),
        "GWGLM": _FitCase(
            lambda: pygwrx.GWGLM(family="gaussian", bandwidth=4.0),
            lambda model: model.fit(X, y, coords),
            lambda model: model.fit(X, y[:-1], coords),
        ),
        "GWLasso": _FitCase(
            lambda: pygwrx.GWLasso(
                bandwidth=18, adaptive=True, alpha=0.05, max_iter=3000
            ),
            lambda model: model.fit(X, y, coords),
            lambda model: _bad_nan_refit(model, X, y, coords),
        ),
        "MixedGWR": _FitCase(
            lambda: pygwrx.MixedGWR(
                bandwidth=18,
                adaptive=True,
                global_vars=[0],
                local_vars=[1, 2],
                intercept_fixed=True,
                verbose=False,
            ),
            lambda model: model.fit(X, y, coords, compute_enp=False),
            lambda model: model.fit(X, y[:-1], coords, compute_enp=False),
        ),
        "LCRGWR": _FitCase(
            lambda: pygwrx.LCRGWR(
                bandwidth=18,
                adaptive=True,
                lambda_adjust=False,
                verbose=False,
            ),
            lambda model: model.fit(
                X, y, coords, compute_local_r2=False, compute_cv=False
            ),
            lambda model: model.fit(X, y[:-1], coords),
        ),
        "ScalableGWR": _FitCase(
            lambda: pygwrx.ScalableGWR(
                bandwidth=14,
                optimize_bandwidth=False,
                scale=1.0,
                penalty=0.05,
                verbose=False,
            ),
            lambda model: model.fit(X, y, coords),
            lambda model: _bad_nan_refit(model, X, y, coords),
        ),
        "SGWR": _FitCase(
            lambda: pygwrx.SGWR(bandwidth=18, adaptive=True, alpha=0.5),
            lambda model: model.fit(X, y, coords),
            lambda model: model.fit(X, y[:-1], coords),
        ),
        "SGTWR": _FitCase(
            lambda: pygwrx.SGTWR(
                spatial_bandwidth=18,
                temporal_bandwidth=2.0,
                adaptive=True,
                alpha=0.5,
                ridge=1e-8,
            ),
            lambda model: model.fit(X, y, coords, times),
            lambda model: model.fit(X[:-1], y, coords[:-1], times[:-1]),
        ),
        "STWR": _FitCase(
            lambda: pygwrx.STWR(
                spatial_bandwidth=5.0,
                adaptive=False,
                kernel="gaussian",
                tick_nums=2,
                ridge=1e-8,
            ),
            lambda model: model.fit(X_list, y_list, coords_list, intervals),
            lambda model: model.fit(
                X_list,
                [*y_list[:-1], y_list[-1][:-1]],
                coords_list,
                intervals,
            ),
        ),
        "MGTWR": _FitCase(
            lambda: pygwrx.MGTWR(
                bandwidths=4.0,
                taus=0.7,
                kernel="gaussian",
                adaptive=False,
                calculate_inference=False,
            ),
            lambda model: model.fit(X, y, coords, times),
            lambda model: model.fit(X[:-1], y, coords, times),
        ),
        "LGGWR": _FitCase(
            lambda: pygwrx.LGGWR(
                max_iter=1,
                learning_rate=0.0,
                select_bandwidth=False,
                bandwidth_updates=0,
                random_state=0,
                verbose=False,
            ),
            lambda model: model.fit(X, y, coords, attrs),
            lambda model: model.fit(X[:-1], y, coords, attrs),
        ),
        "GRGWR": _FitCase(
            lambda: pygwrx.GRGWR(
                n_regimes=2,
                bandwidth=14,
                max_iter=0,
                random_state=0,
                verbose=False,
            ),
            lambda model: model.fit(X, y, coords),
            lambda model: model.fit(X[:-1], y, coords),
        ),
        "GWPCA": _FitCase(
            lambda: pygwrx.GWPCA(
                n_components=2, bandwidth=18, adaptive=True, verbose=False
            ),
            lambda model: model.fit(X, coords),
            lambda model: _bad_nan_unsupervised_refit(model, X, coords),
        ),
        "GWDA": _FitCase(
            lambda: pygwrx.GWDA(
                kernel="gaussian",
                bandwidth=6.0,
                adaptive=False,
                regularization=1e-6,
            ),
            lambda model: model.fit(X_class, y_class, coords_class),
            lambda model: model.fit(X_class, y_class[:-1], coords_class),
        ),
        "GWSS": _FitCase(
            lambda: pygwrx.GWSS(bandwidth=4.0, verbose=False),
            lambda model: model.fit(X, coords),
            lambda model: _bad_nan_unsupervised_refit(model, X, coords),
        ),
        "BootstrapGWR": _FitCase(
            lambda: pygwrx.BootstrapGWR(
                bandwidth=18,
                adaptive=True,
                kernel="gaussian",
                n_bootstrap=2,
                reselect_bandwidth=False,
                random_state=0,
                verbose=False,
            ),
            lambda model: model.fit(X, y, coords),
            lambda model: model.fit(X, y[:-1], coords),
        ),
    }


_CASES = _build_cases()
_EXPECTED_ESTIMATORS = set(_CONTRACT["estimators"])


def test_atomicity_registry_covers_every_a1_public_estimator():
    assert set(_CASES) == _EXPECTED_ESTIMATORS
    assert len(_CASES) == _CONTRACT["estimator_count"] == 19


@pytest.mark.parametrize("name", sorted(_CASES))
def test_failed_refit_restores_constructor_fitted_state(name: str):
    case = _CASES[name]
    model = case.factory()
    tracked = _CONTRACT["estimators"][name]["initialized_public_fitted_state"]
    baseline = {
        attribute: _snapshot(getattr(model, attribute)) for attribute in tracked
    }

    with np.errstate(all="ignore"):
        case.success(model)

    assert any(
        not _values_equal(getattr(model, attribute), baseline[attribute])
        for attribute in tracked
    ), f"{name} success fit did not change any tracked fitted-state attribute"

    with pytest.raises((ValueError, TypeError, RuntimeError, np.linalg.LinAlgError)):
        case.failure(model)

    if hasattr(model, "_is_fitted"):
        assert (
            model._is_fitted is False
        ), f"{name} kept _is_fitted=True after failed refit"
    if hasattr(type(model), "is_fitted_"):
        assert (
            model.is_fitted_ is False
        ), f"{name} kept is_fitted_=True after failed refit"

    for attribute in tracked:
        assert hasattr(
            model, attribute
        ), f"{name}.{attribute} disappeared after failed refit"
        _assert_value_matches_baseline(
            getattr(model, attribute),
            baseline[attribute],
            label=f"{name}.{attribute}",
        )


def _values_equal(actual: Any, expected: Any) -> bool:
    try:
        _assert_value_matches_baseline(actual, expected, label="value")
    except (AssertionError, ValueError, TypeError):
        return False
    return True
