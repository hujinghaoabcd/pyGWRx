# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Private structural protocols for the pyGWRx architecture spine.

These protocols describe narrow capabilities consumed by future private engines,
diagnostics adapters, and plotting adapters. They intentionally contain no
implementation, estimator mathematics, validation policy, or execution policy.

This module is private and must not be re-exported from :mod:`pygwrx.core` or
the package root.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

import numpy as np

__all__: tuple[str, ...] = ()


class FittedLifecycleProtocol(Protocol):
    """Minimal fitted-state lifecycle exposed by estimator-like objects."""

    @property
    def is_fitted_(self) -> bool:
        """Whether fitting completed successfully."""
        ...

    def _mark_fitted(self) -> None:
        """Mark a successfully fitted object as fitted."""
        ...

    def _mark_unfitted(self) -> None:
        """Clear the fitted lifecycle marker before or after a failed fit."""
        ...

    def _check_is_fitted(self) -> None:
        """Raise when fitted state is unavailable."""
        ...


class RegressionSurfaceProtocol(Protocol):
    """Normalized row-aligned regression surface for downstream consumers."""

    @property
    def coordinates(self) -> np.ndarray:
        """Coordinates corresponding to rows in the fitted surface."""
        ...

    @property
    def response(self) -> np.ndarray | None:
        """Observed response values when the view represents fitted data."""
        ...

    @property
    def fitted_values(self) -> np.ndarray:
        """Row-aligned fitted or predicted values."""
        ...

    @property
    def residuals(self) -> np.ndarray | None:
        """Row-aligned residuals when an observed response is available."""
        ...


class ParameterInferenceProtocol(Protocol):
    """One local parameter surface and its optional inference quantities."""

    @property
    def values(self) -> np.ndarray:
        """Row-aligned local parameter values."""
        ...

    @property
    def statistic(self) -> np.ndarray | None:
        """Row-aligned test statistics when available."""
        ...

    @property
    def standard_error(self) -> np.ndarray | None:
        """Row-aligned standard errors when available."""
        ...

    @property
    def label(self) -> str:
        """Human-readable parameter label."""
        ...

    @property
    def parameter_index(self) -> int:
        """Index of the parameter in the model parameterization."""
        ...

    @property
    def distribution(self) -> str:
        """Reference distribution used by the inference statistic."""
        ...


class TemporalViewProtocol(Protocol):
    """Row-aligned temporal capability for spatiotemporal results."""

    @property
    def times(self) -> np.ndarray:
        """One time value per row in the associated result surface."""
        ...


class MultiscaleViewProtocol(Protocol):
    """Resolved parameter-specific scales for multiscale estimators."""

    @property
    def bandwidths(self) -> np.ndarray:
        """One resolved bandwidth or scale per model parameter."""
        ...


class WeightProviderProtocol(Protocol):
    """On-demand access to model-defined weight rows.

    The protocol deliberately does not define how distance, neighbourhood,
    kernels, caching, or model-specific geometry produce the weights.
    """

    def weight_row(self, target_index: int) -> np.ndarray:
        """Return the model-consistent weight row for one target."""
        ...


class StoredWeightComponentsProtocol(Protocol):
    """Named weight matrices retained by a fitted model or result view."""

    @property
    def components(self) -> Mapping[str, np.ndarray]:
        """Stored weight matrices keyed by stable semantic names."""
        ...

    @property
    def combined_name(self) -> str | None:
        """Name of the effective combined matrix when one is stored."""
        ...
