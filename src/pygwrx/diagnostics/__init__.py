# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Unified diagnostics for fitted pyGWRx models.

The package separates statistical extraction from visualization. It provides
model-level summaries, row-wise residual and influence tables, local parameter
inference, weight decomposition, time-indexed views, regime diagnostics, and
local collinearity analysis.

Author:
    Jinghao Hu
"""

from __future__ import annotations

__author__ = "Jinghao Hu"
__license__ = "MIT"

from .collinearity import LocalCollinearityDiagnostics
from .inference import (
    ParameterInference,
    adjust_pvalues,
    feature_names,
    parameter_inference,
    parameter_significance,
)
from .model import DiagnosticSummary, diagnostics_frame, model_diagnostic_summary
from .regimes import boundary_frame, regime_frame, regime_summary
from .residuals import InfluenceThresholds, influence_thresholds, local_diagnostic_frame
from .temporal import (
    TemporalGroups,
    model_times,
    parameter_trajectory,
    temporal_groups,
    temporal_parameter_frame,
)
from .weights import WeightComponents, focus_weight_components, weight_components

__all__ = [
    "DiagnosticSummary",
    "InfluenceThresholds",
    "LocalCollinearityDiagnostics",
    "ParameterInference",
    "TemporalGroups",
    "WeightComponents",
    "adjust_pvalues",
    "boundary_frame",
    "diagnostics_frame",
    "feature_names",
    "focus_weight_components",
    "influence_thresholds",
    "local_diagnostic_frame",
    "model_diagnostic_summary",
    "model_times",
    "parameter_inference",
    "parameter_significance",
    "parameter_trajectory",
    "regime_frame",
    "regime_summary",
    "temporal_groups",
    "temporal_parameter_frame",
    "weight_components",
]
