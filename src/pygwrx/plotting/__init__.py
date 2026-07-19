# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Publication-ready visualization for every supported pyGWRx model.

The preferred API accepts fitted estimators, obtains statistically meaningful
arrays through :mod:`pygwrx.diagnostics`, and returns Matplotlib figures without
calling ``plt.show()``. Historical array-based maps remain available for compatibility.

Author:
    Jinghao Hu
"""

from __future__ import annotations

__author__ = "Jinghao Hu"
__license__ = "MIT"

from pygwrx.plotting.bandwidth import plot_kernel_weights, plot_mgwr_bandwidths
from pygwrx.plotting.bootstrap import (
    plot_bootstrap_bandwidths,
    plot_bootstrap_pvalues,
)
from pygwrx.plotting.comparison import (
    compare_coefficient_surfaces,
    compare_model_diagnostics,
)
from pygwrx.plotting.decomposition import (
    plot_selection_history,
    plot_weight_decomposition,
    plot_weight_profiles,
)
from pygwrx.plotting.diagnostics import (
    plot_bandwidth_selection,
    plot_coefficient_variability,
    plot_diagnostic_panel,
    plot_local_diagnostics,
    plot_observed_vs_predicted,
    plot_qq,
    plot_residual_histogram,
    plot_residuals,
    plot_spatial_residuals,
)
from pygwrx.plotting.geometry import (
    plot_lggwr_latent_geometry,
    plot_lggwr_metric_matrix,
    plot_lggwr_neighbourhood_comparison,
    plot_lggwr_training,
)

# Historical array-based maps.
from pygwrx.plotting.map import (
    create_choropleth,
    plot_bandwidth,
    plot_coefficient_surface,
    plot_local_coefficients,
    plot_local_r2,
    plot_multiple_coefficients,
)
from pygwrx.plotting.map import plot_significance_map as plot_array_significance_map
from pygwrx.plotting.multivariate import (
    plot_gwda_classification,
    plot_gwda_confusion_matrix,
    plot_gwpca_explained_variance,
    plot_gwpca_loading,
    plot_gwss_statistic,
)
from pygwrx.plotting.regimes import (
    plot_grgwr_coefficient_surface,
    plot_grgwr_convergence,
    plot_grgwr_regime_sizes,
    plot_grgwr_regimes,
)
from pygwrx.plotting.regularization import (
    plot_gwlasso_active_map,
    plot_gwlasso_alpha,
    plot_gwlasso_selection_frequency,
    plot_mixed_gwr_coefficients,
)
from pygwrx.plotting.robust import (
    plot_gwglm_residuals,
    plot_rgwr_convergence,
    plot_rgwr_weights,
)
from pygwrx.plotting.scalable import plot_scalable_gwr_kernel
from pygwrx.plotting.surfaces import (
    plot_coefficient_map,
    plot_local_collinearity,
    plot_local_diagnostic_map,
)
from pygwrx.plotting.surfaces import (
    plot_significance_map as _plot_model_significance_map,
)
from pygwrx.plotting.temporal import (
    plot_mgtwr_scales,
    plot_temporal_bandwidths,
    plot_temporal_coefficient_slices,
    plot_temporal_residuals,
    plot_temporal_trajectory,
)


def plot_significance_map(first, *args, **kwargs):
    """Dispatch to model-aware or historical array-based significance mapping."""
    if hasattr(first, "coef_"):
        return _plot_model_significance_map(first, *args, **kwargs)
    return plot_array_significance_map(first, *args, **kwargs)


plot_model_significance_map = _plot_model_significance_map

__all__ = [
    # Universal model-aware surfaces and diagnostics.
    "plot_coefficient_map",
    "plot_significance_map",
    "plot_model_significance_map",
    "plot_array_significance_map",
    "plot_local_diagnostic_map",
    "plot_local_collinearity",
    "compare_coefficient_surfaces",
    "compare_model_diagnostics",
    "plot_kernel_weights",
    "plot_mgwr_bandwidths",
    "plot_residuals",
    "plot_residual_histogram",
    "plot_qq",
    "plot_spatial_residuals",
    "plot_observed_vs_predicted",
    "plot_bandwidth_selection",
    "plot_coefficient_variability",
    "plot_diagnostic_panel",
    "plot_local_diagnostics",
    # RGWR and GWGLM.
    "plot_rgwr_weights",
    "plot_rgwr_convergence",
    "plot_gwglm_residuals",
    # Regularized, mixed, and bootstrap models.
    "plot_gwlasso_selection_frequency",
    "plot_gwlasso_active_map",
    "plot_gwlasso_alpha",
    "plot_mixed_gwr_coefficients",
    "plot_bootstrap_pvalues",
    "plot_bootstrap_bandwidths",
    # Multivariate and classification models.
    "plot_gwss_statistic",
    "plot_gwpca_explained_variance",
    "plot_gwpca_loading",
    "plot_gwda_classification",
    "plot_gwda_confusion_matrix",
    # Scalable GWR.
    "plot_scalable_gwr_kernel",
    # Time and weight decomposition.
    "plot_temporal_coefficient_slices",
    "plot_mgtwr_scales",
    "plot_temporal_trajectory",
    "plot_temporal_residuals",
    "plot_temporal_bandwidths",
    "plot_weight_decomposition",
    "plot_weight_profiles",
    "plot_selection_history",
    # Original models.
    "plot_lggwr_latent_geometry",
    "plot_lggwr_metric_matrix",
    "plot_lggwr_training",
    "plot_lggwr_neighbourhood_comparison",
    "plot_grgwr_regimes",
    "plot_grgwr_convergence",
    "plot_grgwr_regime_sizes",
    "plot_grgwr_coefficient_surface",
    # Historical array-based maps.
    "plot_local_coefficients",
    "plot_coefficient_surface",
    "plot_local_r2",
    "plot_bandwidth",
    "create_choropleth",
    "plot_multiple_coefficients",
]
