# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Local parameter inference and multiple-testing correction.

Author:
    Jinghao Hu
"""

from __future__ import annotations

__author__ = "Jinghao Hu"
__license__ = "MIT"

from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.stats import t as student_t

from ._utils import require_fitted

FeatureLike = Union[str, int]


@dataclass(frozen=True)
class ParameterInference:
    """One local coefficient surface with inference arrays."""

    values: np.ndarray
    statistic: Optional[np.ndarray]
    standard_error: Optional[np.ndarray]
    label: str
    parameter_index: int
    distribution: str


def feature_names(model: Any) -> Tuple[str, ...]:
    """Return stable predictor names for a fitted model."""
    n = int(getattr(model, "n_features_in_", 0) or 0)
    if n == 0:
        coef = getattr(model, "coef_", None)
        if coef is not None:
            array = np.asarray(coef)
            if array.ndim == 2:
                n = int(array.shape[1])
    names = getattr(model, "feature_names_in_", None)
    if names is None:
        names = getattr(model, "feature_names_", None)
    if names is not None and len(names) == n:
        return tuple(str(name) for name in names)
    return tuple(f"x{i}" for i in range(n))


def _full_parameter_column(
    model: Any,
    names: Sequence[str],
    parameter_index: int,
    n_parameters: int,
) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """Return one column from the first compatible full-parameter matrix."""
    for name in names:
        value = getattr(model, name, None)
        if value is None:
            continue
        array = np.asarray(value, dtype=float)
        if array.ndim == 1 and n_parameters == 1:
            return array.reshape(-1), name
        if array.ndim == 2 and array.shape[1] == n_parameters:
            return array[:, parameter_index], name
    return None, None


def parameter_inference(model: Any, feature: FeatureLike) -> ParameterInference:
    """Extract a coefficient, test statistic, and standard error surface.

    The extractor understands both split attributes such as ``coef_t_`` and
    full parameter matrices such as ``parameter_t_values_`` or ``t_values_``.
    This keeps inference consistent across GWR, GWGLM, MGTWR, ScalableGWR,
    SGWR, STWR, and SGTWR.
    """
    require_fitted(model)
    coef = np.asarray(getattr(model, "coef_", None), dtype=float)
    if coef.ndim != 2:
        raise ValueError(
            f"{model.__class__.__name__} does not expose a local coef_ matrix."
        )
    names = feature_names(model)
    fit_intercept = bool(getattr(model, "fit_intercept", True))
    n_parameters = coef.shape[1] + int(fit_intercept)
    is_intercept = isinstance(feature, str) and feature.strip().lower() in {
        "intercept",
        "constant",
    }

    if is_intercept:
        if not fit_intercept:
            raise ValueError("The fitted model does not include an intercept.")
        values = np.asarray(getattr(model, "intercept_", None), dtype=float).reshape(-1)
        parameter_index = 0
        label = "Intercept"
        direct_statistic = getattr(model, "intercept_z_", None)
        distribution = "normal" if direct_statistic is not None else "t"
        if direct_statistic is None:
            direct_statistic = getattr(model, "intercept_t_", None)
        direct_se = getattr(model, "intercept_se_", None)
    else:
        if isinstance(feature, str):
            if feature not in names:
                raise ValueError(
                    f"Unknown feature {feature!r}. Choose from {list(names)}."
                )
            index = names.index(feature)
        elif isinstance(feature, (int, np.integer)) and not isinstance(
            feature, (bool, np.bool_)
        ):
            index = int(feature)
            if index < 0 or index >= coef.shape[1]:
                raise IndexError(
                    "feature index is outside the fitted coefficient range."
                )
        else:
            raise TypeError("feature must be a name, integer index, or 'intercept'.")
        values = coef[:, index]
        parameter_index = index + int(fit_intercept)
        label = names[index]
        direct_statistic = getattr(model, "coef_z_", None)
        distribution = "normal" if direct_statistic is not None else "t"
        if direct_statistic is None:
            direct_statistic = getattr(model, "coef_t_", None)
        if direct_statistic is not None:
            direct_statistic = np.asarray(direct_statistic, dtype=float)[:, index]
        direct_se = getattr(model, "coef_se_", None)
        if direct_se is not None:
            direct_se = np.asarray(direct_se, dtype=float)[:, index]

    statistic = (
        None
        if direct_statistic is None
        else np.asarray(direct_statistic, dtype=float).reshape(-1)
    )
    standard_error = (
        None if direct_se is None else np.asarray(direct_se, dtype=float).reshape(-1)
    )

    if statistic is None:
        statistic, source = _full_parameter_column(
            model,
            ("parameter_z_values_", "parameter_t_values_", "t_values_"),
            parameter_index,
            n_parameters,
        )
        if source == "parameter_z_values_":
            distribution = "normal"
    if standard_error is None:
        standard_error, _ = _full_parameter_column(
            model,
            ("parameter_standard_errors_", "standard_errors_"),
            parameter_index,
            n_parameters,
        )

    if statistic is not None and statistic.size != values.size:
        raise ValueError("Local test-statistic length does not match coefficients.")
    if standard_error is not None and standard_error.size != values.size:
        raise ValueError("Local standard-error length does not match coefficients.")
    return ParameterInference(
        values=np.asarray(values, dtype=float).reshape(-1),
        statistic=statistic,
        standard_error=standard_error,
        label=label,
        parameter_index=parameter_index,
        distribution=distribution,
    )


def adjust_pvalues(p_values: Sequence[float], method: str = "bh") -> np.ndarray:
    """Adjust p values using Bonferroni, BH, or BY correction."""
    p = np.asarray(p_values, dtype=float).reshape(-1)
    if p.size == 0 or not np.all(np.isfinite(p)) or np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("p_values must be finite values in [0, 1].")
    token = str(method).strip().lower().replace("-", "_")
    n = p.size
    if token in {"none", "raw"}:
        return p.copy()
    if token == "bonferroni":
        return np.minimum(p * n, 1.0)
    if token not in {"bh", "by"}:
        raise ValueError("method must be 'raw', 'bonferroni', 'bh', or 'by'.")
    order = np.argsort(p)
    ranked = p[order]
    ranks = np.arange(1, n + 1, dtype=float)
    multiplier = n / ranks
    if token == "by":
        multiplier *= np.sum(1.0 / ranks)
    adjusted = np.minimum.accumulate((ranked * multiplier)[::-1])[::-1]
    output = np.empty(n, dtype=float)
    output[order] = np.minimum(adjusted, 1.0)
    return output


def parameter_significance(
    model: Any,
    feature: FeatureLike,
    *,
    alpha: float = 0.05,
    correction: str = "adjusted",
) -> pd.DataFrame:
    """Return coefficient values, p values, and significance categories."""
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError("alpha must lie strictly between 0 and 1.")
    view = parameter_inference(model, feature)
    if view.statistic is None:
        raise ValueError("The fitted model does not expose local test statistics.")
    statistic = np.asarray(view.statistic, dtype=float)
    finite = np.isfinite(statistic)
    if view.distribution == "normal":
        p = 2.0 * norm.sf(np.abs(statistic))
    else:
        n = statistic.size
        trace_s = float(
            (getattr(model, "diagnostics_", None) or {}).get("trace_S", 0.0)
        )
        df = max(int(np.floor(n - trace_s)), 1)
        p = 2.0 * student_t.sf(np.abs(statistic), df)

    token = str(correction).strip().lower().replace("-", "_")
    critical = None
    if token in {"adjusted", "gwr_adjusted", "mgwr_adjusted"}:
        adjusted = getattr(model, "adjusted_alpha_by_variable_", None)
        local_alpha = float(alpha)
        if adjusted is not None:
            array = np.asarray(adjusted, dtype=float)
            standards = np.asarray([0.10, 0.05, 0.01])
            nearest = int(np.argmin(np.abs(standards - float(alpha))))
            if (
                array.ndim == 2
                and view.parameter_index < array.shape[0]
                and np.isclose(standards[nearest], alpha)
            ):
                local_alpha = float(array[view.parameter_index, nearest])
        else:
            diagnostics = getattr(model, "diagnostics_", None) or {}
            trace_s = float(diagnostics.get("trace_S", np.nan))
            p_count = np.asarray(model.coef_).shape[1] + int(
                bool(getattr(model, "fit_intercept", True))
            )
            if np.isfinite(trace_s) and trace_s > 0.0:
                local_alpha = min(float(alpha), float(alpha) * p_count / trace_s)
        significant = finite & (p <= local_alpha)
        adjusted_p = p
        critical = local_alpha
    else:
        adjusted_p = adjust_pvalues(np.where(finite, p, 1.0), method=token)
        significant = finite & (adjusted_p <= float(alpha))
        critical = float(alpha)

    category = np.zeros(statistic.size, dtype=int)
    category[significant & (view.values < 0.0)] = -1
    category[significant & (view.values > 0.0)] = 1
    return pd.DataFrame(
        {
            "coefficient": view.values,
            "statistic": statistic,
            "p_value": p,
            "adjusted_p_value": adjusted_p,
            "significant": significant,
            "category": category,
            "threshold": critical,
        }
    )
