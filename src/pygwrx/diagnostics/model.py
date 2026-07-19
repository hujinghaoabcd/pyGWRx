# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Unified model-level diagnostic summaries.

Author:
    Jinghao Hu
"""

from __future__ import annotations

__author__ = "Jinghao Hu"
__license__ = "MIT"

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ._utils import fitted_values, require_fitted, training_response


@dataclass(frozen=True)
class DiagnosticSummary:
    """Normalized global diagnostics for one fitted estimator."""

    model_name: str
    n_samples: Optional[int]
    n_features: Optional[int]
    family: Optional[str]
    metrics: Mapping[str, float]
    conditional_metrics: Tuple[str, ...] = ()

    def to_series(self, name: Optional[str] = None) -> pd.Series:
        """Return a labeled :class:`pandas.Series`."""
        values: Dict[str, object] = {
            "model": self.model_name,
            "n_samples": self.n_samples,
            "n_features": self.n_features,
            "family": self.family,
        }
        values.update(self.metrics)
        return pd.Series(values, name=name or self.model_name)


_ALIAS_GROUPS = {
    "r2": ("r2", "R2"),
    "adj_r2": ("adj_r2", "adjusted_r2", "adj_R2"),
    "rmse": ("rmse",),
    "mae": ("mae",),
    "rss": ("rss", "resid_ss"),
    "aic": ("aic",),
    "aicc": ("aicc", "AICc", "conditional_aicc"),
    "bic": ("bic",),
    "trace_S": ("trace_S", "tr_S", "effective_params"),
    "trace_StS": ("trace_StS", "tr_StS"),
    "enp": ("enp", "ENP", "conditional_enp", "effective_params"),
    "edf": ("edf", "residual_df"),
    "deviance": ("deviance",),
    "percent_deviance": ("percent_deviance",),
    "log_likelihood": ("log_likelihood", "llf"),
    "cv": ("cv", "cv_score", "bandwidth_cv_score"),
}


def _finite_scalar(value: Any) -> Optional[float]:
    try:
        array = np.asarray(value)
        if array.ndim != 0:
            return None
        number = float(array)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) or np.isinf(number) else None


def model_diagnostic_summary(model: Any) -> DiagnosticSummary:
    """Normalize global diagnostics exposed by any supported fitted model."""
    require_fitted(model)
    raw: Dict[str, Any] = dict(getattr(model, "diagnostics_", None) or {})
    attribute_aliases = {
        "r2_": "r2",
        "adjusted_r2_": "adjusted_r2",
        "rss_": "rss",
        "aic_": "aic",
        "aicc_": "aicc",
        "bic_": "bic",
        "trace_S_": "trace_S",
        "trace_StS_": "trace_StS",
        "effective_params_": "effective_params",
        "effective_n_params_": "effective_params",
        "effective_df_": "edf",
        "cv_score_": "cv_score",
        "bandwidth_cv_score_": "bandwidth_cv_score",
        "alpha_score_": "alpha_score",
        "sigma2_": "sigma2",
        "sigma_": "sigma",
    }
    for attribute, alias in attribute_aliases.items():
        value = getattr(model, attribute, None)
        if value is not None:
            raw.setdefault(alias, value)

    y = training_response(model)
    fitted = fitted_values(model)
    if y is not None and fitted is not None and y.size == fitted.size:
        residual = y - fitted
        raw.setdefault("rss", float(np.dot(residual, residual)))
        raw.setdefault("rmse", float(np.sqrt(np.mean(residual**2))))
        raw.setdefault("mae", float(np.mean(np.abs(residual))))

    metrics: Dict[str, float] = {}
    for canonical, aliases in _ALIAS_GROUPS.items():
        for alias in aliases:
            if alias in raw:
                value = _finite_scalar(raw[alias])
                if value is not None:
                    metrics[canonical] = value
                    break

    # Preserve additional scalar diagnostics without duplicating aliases.
    consumed = {alias for aliases in _ALIAS_GROUPS.values() for alias in aliases}
    for key, value in raw.items():
        if key in consumed or key in metrics:
            continue
        scalar = _finite_scalar(value)
        if scalar is not None:
            metrics[str(key)] = scalar

    n_samples = getattr(model, "n_samples_", None)
    if n_samples is None and y is not None:
        n_samples = int(y.size)
    n_features = getattr(model, "n_features_in_", None)
    family = getattr(model, "family", None)
    if family is not None and not isinstance(family, str):
        family = family.__class__.__name__

    conditional = tuple(
        name for name in ("aicc", "enp") if f"conditional_{name}" in raw
    )
    return DiagnosticSummary(
        model_name=model.__class__.__name__,
        n_samples=None if n_samples is None else int(n_samples),
        n_features=None if n_features is None else int(n_features),
        family=None if family is None else str(family),
        metrics=metrics,
        conditional_metrics=conditional,
    )


def diagnostics_frame(
    models: Iterable[Any], labels: Optional[Sequence[str]] = None
) -> pd.DataFrame:
    """Return one row of normalized global diagnostics per model."""
    model_list = list(models)
    if not model_list:
        raise ValueError("models must contain at least one fitted estimator.")
    if labels is not None and len(labels) != len(model_list):
        raise ValueError("labels must contain one entry per model.")
    records = []
    for index, model in enumerate(model_list):
        summary = model_diagnostic_summary(model)
        record = summary.to_series().to_dict()
        record["label"] = labels[index] if labels is not None else summary.model_name
        records.append(record)
    frame = pd.DataFrame(records).set_index("label")
    preferred = [
        "model",
        "n_samples",
        "n_features",
        "family",
        "r2",
        "adj_r2",
        "rmse",
        "mae",
        "aic",
        "aicc",
        "bic",
        "enp",
        "edf",
    ]
    columns = [name for name in preferred if name in frame.columns]
    columns.extend(name for name in frame.columns if name not in columns)
    return frame.loc[:, columns]
