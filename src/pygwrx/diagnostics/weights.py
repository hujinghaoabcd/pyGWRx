# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Extraction and validation of fitted spatial and spatiotemporal weights.

Author:
    Jinghao Hu
"""

from __future__ import annotations

__author__ = "Jinghao Hu"
__license__ = "MIT"

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

import numpy as np

from ._utils import require_fitted


@dataclass(frozen=True)
class WeightComponents:
    """Named weight matrices exposed by a fitted model."""

    components: Mapping[str, np.ndarray]
    combined_name: Optional[str]

    @property
    def combined(self) -> Optional[np.ndarray]:
        """Return the combined matrix when one is available."""
        if self.combined_name is None:
            return None
        return self.components[self.combined_name]


def weight_components(model: Any) -> WeightComponents:
    """Collect stored weight matrices using stable semantic names."""
    require_fitted(model)
    candidates = (
        ("spatial", "spatial_weights_"),
        ("temporal", "temporal_weights_"),
        ("spatiotemporal", "spatiotemporal_weights_"),
        ("similarity", "similarity_weights_"),
        ("combined", "combined_weights_"),
        ("weights", "weights_"),
    )
    components: Dict[str, np.ndarray] = {}
    for label, attribute in candidates:
        value = getattr(model, attribute, None)
        if value is None:
            continue
        array = np.asarray(value, dtype=float)
        if array.ndim != 2 or 0 in array.shape:
            continue
        if not np.all(np.isfinite(array)) or np.any(array < 0.0):
            raise ValueError(f"{attribute} must be a finite non-negative matrix.")
        components[label] = array
    if not components:
        raise ValueError(
            f"{model.__class__.__name__} does not store weight matrices. Refit with store_weights=True when supported."
        )
    combined_name = None
    for name in ("combined", "weights", "spatiotemporal", "spatial"):
        if name in components:
            combined_name = name
            break
    return WeightComponents(components=components, combined_name=combined_name)


def focus_weight_components(model: Any, focus: int) -> Dict[str, np.ndarray]:
    """Return one row from every stored weight component."""
    collection = weight_components(model)
    if not isinstance(focus, (int, np.integer)) or isinstance(focus, (bool, np.bool_)):
        raise TypeError("focus must be an integer row index.")
    index = int(focus)
    row_count = next(iter(collection.components.values())).shape[0]
    if index < 0 or index >= row_count:
        raise IndexError(f"focus must be in [0, {row_count - 1}].")
    output = {}
    for name, matrix in collection.components.items():
        if matrix.shape[0] != row_count:
            raise ValueError("Stored weight matrices have inconsistent row counts.")
        output[name] = matrix[index].copy()
    return output
