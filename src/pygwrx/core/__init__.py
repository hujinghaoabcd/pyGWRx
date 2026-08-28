# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Public interface for pyGWRx core components.

This module exposes the supported base classes, kernels, bandwidth selectors, numerical solvers, diagnostics, and utility functions used to implement geographically weighted models.

Author:
    Jinghao Hu
"""

from __future__ import annotations

__author__ = "Jinghao Hu"
__license__ = "MIT"

# Bandwidth selection
from pygwrx.core.bandwidth import (
    AICSelector,
    BandwidthSelector,
    BICSelector,
    CrossValidationSelector,
    get_bandwidth_selector,
)

# Base estimator hierarchy
from pygwrx.core.base import (
    BaseMultiscaleRegressor,
    BaseSpatialClassifier,
    BaseSpatialEstimator,
    BaseSpatialInference,
    BaseSpatialRegressor,
    BaseSpatialStatistics,
    BaseSpatialTransformer,
    BaseSpatiotemporalRegressor,
    MultiscaleMixin,
    SpatiotemporalMixin,
)

# Spatial kernels
from pygwrx.core.kernels import (
    bisquare_kernel,
    boxcar_kernel,
    exponential_kernel,
    gaussian_kernel,
    get_kernel_function,
    tricube_kernel,
)

# Diagnostics and model metrics
from pygwrx.core.metrics import (
    compute_adjusted_r_squared,
    compute_aic,
    compute_aicc,
    compute_bic,
    compute_diagnostics,
    compute_edf,
    compute_effective_parameters,
    compute_enp,
    compute_local_r_squared,
    compute_r_squared,
    compute_trace_statistics,
)

# One-dimensional optimisation
from pygwrx.core.optimization import (
    BrentSearch,
    GoldenSectionSearch,
    OptimizationResult,
)

# Numerical solvers
from pygwrx.core.solver import (
    adaptive_bandwidth_weights,
    compute_hat_matrix,
    local_regression,
    weighted_least_squares,
)

# Distance, validation, and data helpers
from pygwrx.core.utils import (
    DistanceCache,
    add_intercept,
    chebyshev_distance,
    chunked_computation,
    compute_distance_matrix,
    euclidean_distance,
    extract_geopandas_coords,
    haversine_distance,
    manhattan_distance,
    minkowski_distance,
    validate_coords,
    validate_data,
)

__all__ = [
    # Base estimator hierarchy
    "BaseSpatialEstimator",
    "BaseSpatialRegressor",
    "SpatiotemporalMixin",
    "MultiscaleMixin",
    "BaseSpatiotemporalRegressor",
    "BaseMultiscaleRegressor",
    "BaseSpatialClassifier",
    "BaseSpatialTransformer",
    "BaseSpatialStatistics",
    "BaseSpatialInference",
    # Kernels
    "gaussian_kernel",
    "bisquare_kernel",
    "exponential_kernel",
    "tricube_kernel",
    "boxcar_kernel",
    "get_kernel_function",
    # Bandwidth selectors
    "BandwidthSelector",
    "CrossValidationSelector",
    "AICSelector",
    "BICSelector",
    "get_bandwidth_selector",
    # Optimisation
    "OptimizationResult",
    "GoldenSectionSearch",
    "BrentSearch",
    # Numerical solvers
    "weighted_least_squares",
    "local_regression",
    "compute_hat_matrix",
    "adaptive_bandwidth_weights",
    # Metrics and diagnostics
    "compute_r_squared",
    "compute_adjusted_r_squared",
    "compute_aic",
    "compute_aicc",
    "compute_bic",
    "compute_local_r_squared",
    "compute_effective_parameters",
    "compute_diagnostics",
    "compute_trace_statistics",
    "compute_edf",
    "compute_enp",
    # Distance and validation helpers
    "euclidean_distance",
    "manhattan_distance",
    "chebyshev_distance",
    "minkowski_distance",
    "haversine_distance",
    "compute_distance_matrix",
    "DistanceCache",
    "validate_coords",
    "validate_data",
    "add_intercept",
    "extract_geopandas_coords",
    "chunked_computation",
]
