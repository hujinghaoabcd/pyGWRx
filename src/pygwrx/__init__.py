# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Public package interface for pyGWRx.

This module exposes the primary models and package metadata used by end users.

Author:
    Jinghao Hu
"""

__author__ = "Jinghao Hu"
__license__ = "MIT"

__version__ = "0.1.2"

# Import commonly used utilities
from pygwrx.core.kernels import (
    bisquare_kernel,
    exponential_kernel,
    gaussian_kernel,
)
from pygwrx.models.bootstrap_gwr import BootstrapGWR
from pygwrx.models.glm_gwr import GWGLM, GWGLMPredictionResult
from pygwrx.models.grgwr import GRGWR, GRGWRPredictionResult
from pygwrx.models.gtwr import GTWR, GTWRPredictionResult
from pygwrx.models.gw_lasso import GWLasso
from pygwrx.models.gwda import GWDA
from pygwrx.models.gwpca import GWPCA

# Import all models for convenient access
from pygwrx.models.gwr import GWR, GWRPredictionResult
from pygwrx.models.gwss import GWSS
from pygwrx.models.lcr_gwr import LCRGWR
from pygwrx.models.lg_gwr import LGGWR, LGGWRPredictionResult
from pygwrx.models.mgtwr import MGTWR
from pygwrx.models.mgwr import MGWR
from pygwrx.models.mixed_gwr import MixedGWR
from pygwrx.models.rgwr import RGWR
from pygwrx.models.scalable_gwr import ScalableGWR
from pygwrx.models.sgtwr import SGTWR, SGTWRPredictionResult
from pygwrx.models.sgwr import SGWR
from pygwrx.models.stwr import STWR, STWRPredictionResult

__all__ = [
    # Basic and Classic Models
    "GWR",
    "GWRPredictionResult",
    "MGWR",
    "RGWR",
    "STWR",
    "STWRPredictionResult",
    "GTWR",
    "GTWRPredictionResult",
    # Generalized Models
    "GWGLM",
    "GWGLMPredictionResult",
    "GWLasso",
    "MixedGWR",
    # Multivariate Analysis
    "GWPCA",
    "GWDA",
    "GWSS",
    # Advanced Models
    "ScalableGWR",
    "LCRGWR",
    "BootstrapGWR",
    "SGWR",
    "SGTWR",
    "SGTWRPredictionResult",
    "MGTWR",
    "LGGWR",
    "LGGWRPredictionResult",
    "GRGWR",
    "GRGWRPredictionResult",
    # Kernels
    "gaussian_kernel",
    "bisquare_kernel",
    "exponential_kernel",
    # Version
    "__version__",
]
