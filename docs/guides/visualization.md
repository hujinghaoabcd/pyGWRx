# Visualization guide and gallery

pyGWRx plotting functions return Matplotlib figure/axes objects and do not call `plt.show()` automatically. This supports notebooks, batch reports, CI rendering, and publication pipelines.

## Plot by analytical question

| Question | Plot family |
|---|---|
| Where is an effect strong? | coefficient/surface maps |
| Where is an effect supported? | significance maps |
| Where does the model fit poorly? | local R², residual, observed-versus-predicted |
| Are estimates unstable? | collinearity and influence maps |
| What neighbourhood is used? | kernel profiles, temporal/similarity weight decomposition |
| Do variables operate at different scales? | MGWR/MGTWR scale plots |
| Are outliers or variables selected locally? | robust weights and GWLasso plots |
| How do latent geometry or regimes change structure? | LGGWR and GRGWR specialist plots |

!!! tip "Publication workflow"
    Use a projected coordinate system, consistent limits and colour scales across compared maps, explicit units, a significance/uncertainty layer, and vector output when possible.

## Surfaces and array inputs

```python
# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Model-aware coefficient maps plus all historical array-based maps."""

import matplotlib

matplotlib.use("Agg", force=True)
import geopandas as gpd
import numpy as np
from shapely.geometry import Point
from pygwrx.plotting import (
    create_choropleth,
    plot_array_significance_map,
    plot_bandwidth,
    plot_coefficient_map,
    plot_coefficient_surface,
    plot_local_coefficients,
    plot_local_diagnostic_map,
    plot_local_r2,
    plot_model_significance_map,
    plot_multiple_coefficients,
    plot_significance_map,
)
from _common import save_plot
from _models import surface_models

X, y, coords, gwr, _, _ = surface_models()
coords_array = coords.to_numpy()
p_values = np.full_like(gwr.coef_, 0.02)
plots = {
    "coefficient_map.png": plot_coefficient_map(gwr, "x1", theme="paper"),
    "model_significance.png": plot_model_significance_map(gwr, "x1", correction="raw"),
    "dispatch_model_significance.png": plot_significance_map(gwr, "x1"),
    "local_diagnostic.png": plot_local_diagnostic_map(gwr, "local_r2"),
    "array_significance.png": plot_array_significance_map(
        coords_array, p_values, feature_idx=0, coefficients=gwr.coef_
    ),
    "dispatch_array_significance.png": plot_significance_map(
        coords_array, p_values, feature_idx=0, coefficients=gwr.coef_
    ),
    "local_coefficients.png": plot_local_coefficients(coords_array, gwr.coef_, 0, "x1"),
    "coefficient_surface.png": plot_coefficient_surface(
        coords_array, gwr.coef_, 0, interpolation="nearest"
    ),
    "array_local_r2.png": plot_local_r2(coords_array, gwr.local_r2_),
    "bandwidth_map.png": plot_bandwidth(
        coords_array, 2.0, sample_locations=coords_array[:3]
    ),
    "multiple_coefficients.png": plot_multiple_coefficients(
        coords_array, gwr.coef_, feature_names=["x1", "x2"], shared_scale=True
    ),
}
for name, result in plots.items():
    print(save_plot(result, name))
gdf = gpd.GeoDataFrame(
    {"value": gwr.coef_[:, 0]},
    geometry=[Point(x, y) for x, y in coords_array],
    crs="EPSG:3857",
)
print(save_plot(create_choropleth(gdf, "value"), "choropleth.png"))
```

## Diagnostics and comparison

```python
# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""All general residual, bandwidth, comparison, and collinearity plots."""

import matplotlib

matplotlib.use("Agg", force=True)
from pygwrx.plotting import (
    compare_coefficient_surfaces,
    compare_model_diagnostics,
    plot_bandwidth_selection,
    plot_coefficient_variability,
    plot_diagnostic_panel,
    plot_kernel_weights,
    plot_local_collinearity,
    plot_local_diagnostics,
    plot_mgwr_bandwidths,
    plot_observed_vs_predicted,
    plot_qq,
    plot_residual_histogram,
    plot_residuals,
    plot_spatial_residuals,
)
from _common import save_plot
from _models import surface_models

X, y, coords, gwr, mgwr, lcr = surface_models()
plots = {
    "compare_surfaces.png": compare_coefficient_surfaces([gwr, mgwr], "x1"),
    "compare_diagnostics.png": compare_model_diagnostics([gwr, mgwr]),
    "kernel_weights.png": plot_kernel_weights(gwr, focus=3),
    "mgwr_bandwidths.png": plot_mgwr_bandwidths(mgwr),
    "residuals.png": plot_residuals(gwr.fitted_values_, gwr.residuals_),
    "residual_histogram.png": plot_residual_histogram(gwr.residuals_),
    "qq.png": plot_qq(gwr.residuals_),
    "spatial_residuals.png": plot_spatial_residuals(coords, gwr.residuals_),
    "observed_predicted.png": plot_observed_vs_predicted(y, gwr.fitted_values_),
    "bandwidth_selection.png": plot_bandwidth_selection(
        [10, 15, 20, 25], [14.0, 9.0, 7.5, 8.2], 20, criterion="AICc"
    ),
    "coefficient_variability.png": plot_coefficient_variability(
        gwr.coef_, feature_names=["x1", "x2"]
    ),
    "diagnostic_panel_arrays.png": plot_diagnostic_panel(
        y, gwr.fitted_values_, gwr.residuals_, coords
    ),
    "diagnostic_panel_model.png": plot_diagnostic_panel(gwr),
    "local_diagnostics.png": plot_local_diagnostics(
        coords, {"local_r2": gwr.local_r2_, "influence": gwr.influence_}
    ),
    "collinearity_gwr.png": plot_local_collinearity(gwr, "condition_number"),
    "collinearity_lcr.png": plot_local_collinearity(lcr, "local_lambda"),
}
for name, result in plots.items():
    print(save_plot(result, name))
```

## Robust, regularized, and bootstrap models

```python
# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""All robust, GLM, Lasso, mixed, bootstrap, and scalable plots."""

import matplotlib

matplotlib.use("Agg", force=True)
from pygwrx.plotting import (
    plot_bootstrap_bandwidths,
    plot_bootstrap_pvalues,
    plot_gwglm_residuals,
    plot_gwlasso_active_map,
    plot_gwlasso_alpha,
    plot_gwlasso_selection_frequency,
    plot_mixed_gwr_coefficients,
    plot_rgwr_convergence,
    plot_rgwr_weights,
    plot_scalable_gwr_kernel,
)
from _common import save_plot
from _models import regularized_models

X, y, coords, rgwr, gwglm, gwlasso, mixed, bootstrap, scalable = regularized_models()
plots = {
    "rgwr_weights.png": plot_rgwr_weights(rgwr),
    "rgwr_convergence.png": plot_rgwr_convergence(rgwr),
    "gwglm_residuals.png": plot_gwglm_residuals(gwglm),
    "gwlasso_frequency.png": plot_gwlasso_selection_frequency(gwlasso),
    "gwlasso_active.png": plot_gwlasso_active_map(gwlasso, "x1"),
    "gwlasso_alpha.png": plot_gwlasso_alpha(gwlasso),
    "mixed_coefficients.png": plot_mixed_gwr_coefficients(mixed),
    "bootstrap_pvalues.png": plot_bootstrap_pvalues(bootstrap, "x1"),
    "bootstrap_bandwidths.png": plot_bootstrap_bandwidths(bootstrap),
    "scalable_kernel.png": plot_scalable_gwr_kernel(scalable),
}
for name, result in plots.items():
    print(save_plot(result, name))
```

## Multivariate and classification models

```python
# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""All GWSS, GWPCA, and GWDA visualization functions."""

import matplotlib

matplotlib.use("Agg", force=True)
from pygwrx.plotting import (
    plot_gwda_classification,
    plot_gwda_confusion_matrix,
    plot_gwpca_explained_variance,
    plot_gwpca_loading,
    plot_gwss_statistic,
)
from _common import save_plot
from _models import multivariate_models

X, coords, gwss, gwpca, Xc, yc, cc, gwda = multivariate_models()
plots = {
    "gwss_mean.png": plot_gwss_statistic(gwss, "mean", "x1"),
    "gwss_correlation.png": plot_gwss_statistic(
        gwss, "correlation", "x1", second_feature="x2"
    ),
    "gwpca_variance.png": plot_gwpca_explained_variance(gwpca, 0),
    "gwpca_cumulative.png": plot_gwpca_explained_variance(gwpca, 0, cumulative=True),
    "gwpca_loading.png": plot_gwpca_loading(gwpca, "x1", 0),
    "gwda_classification.png": plot_gwda_classification(gwda),
    "gwda_confidence.png": plot_gwda_classification(gwda, confidence=True),
    "gwda_confusion.png": plot_gwda_confusion_matrix(gwda, normalize=True),
}
for name, result in plots.items():
    print(save_plot(result, name))
```

## Temporal and weight plots

```python
# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""All temporal, multiscale, weight decomposition, and selection-history plots."""

import matplotlib

matplotlib.use("Agg", force=True)
from pygwrx.plotting import (
    plot_mgtwr_scales,
    plot_selection_history,
    plot_temporal_bandwidths,
    plot_temporal_coefficient_slices,
    plot_temporal_residuals,
    plot_temporal_trajectory,
    plot_weight_decomposition,
    plot_weight_profiles,
)
from _common import save_plot
from _models import temporal_models

X, y, coords, times, gtwr, mgtwr, sgtwr, sgwr, stwr, search_model = temporal_models()
plots = {
    "temporal_slices.png": plot_temporal_coefficient_slices(gtwr, "x1"),
    "temporal_trajectory.png": plot_temporal_trajectory(gtwr, "x1"),
    "temporal_residuals.png": plot_temporal_residuals(gtwr),
    "temporal_bandwidths.png": plot_temporal_bandwidths(sgtwr),
    "mgtwr_scales.png": plot_mgtwr_scales(mgtwr),
    "sgwr_decomposition.png": plot_weight_decomposition(sgwr, 0),
    "sgwr_profiles.png": plot_weight_profiles(sgwr, 0, sort_by="combined"),
    "stwr_decomposition.png": plot_weight_decomposition(stwr, 0),
    "sgtwr_decomposition.png": plot_weight_decomposition(sgtwr, 0),
    "selection_history.png": plot_selection_history(search_model),
}
for name, result in plots.items():
    print(save_plot(result, name))
```

## LGGWR and GRGWR

```python
# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""All visualization functions for the two original research models."""

import matplotlib

matplotlib.use("Agg", force=True)
from pygwrx.plotting import (
    plot_grgwr_coefficient_surface,
    plot_grgwr_convergence,
    plot_grgwr_regime_sizes,
    plot_grgwr_regimes,
    plot_lggwr_latent_geometry,
    plot_lggwr_metric_matrix,
    plot_lggwr_neighbourhood_comparison,
    plot_lggwr_training,
)
from _common import save_plot
from _models import original_models

lggwr, grgwr = original_models()
plots = {
    "lggwr_geometry.png": plot_lggwr_latent_geometry(lggwr),
    "lggwr_metric.png": plot_lggwr_metric_matrix(lggwr),
    "lggwr_training.png": plot_lggwr_training(lggwr),
    "lggwr_neighbours.png": plot_lggwr_neighbourhood_comparison(lggwr, 0),
    "grgwr_regimes.png": plot_grgwr_regimes(grgwr),
    "grgwr_convergence.png": plot_grgwr_convergence(grgwr),
    "grgwr_sizes.png": plot_grgwr_regime_sizes(grgwr),
    "grgwr_surface.png": plot_grgwr_coefficient_surface(grgwr, "x1"),
}
for name, result in plots.items():
    print(save_plot(result, name))
```

## Figure gallery

<div class="figure-grid" markdown>

<figure markdown>
  ![01 Coefficient](../assets/figures/core/01_coefficient.png){ loading=lazy }
  <figcaption>01 Coefficient</figcaption>
</figure>

<figure markdown>
  ![02 Coefficient Significant](../assets/figures/core/02_coefficient_significant.png){ loading=lazy }
  <figcaption>02 Coefficient Significant</figcaption>
</figure>

<figure markdown>
  ![03 Significance Categories](../assets/figures/core/03_significance_categories.png){ loading=lazy }
  <figcaption>03 Significance Categories</figcaption>
</figure>

<figure markdown>
  ![04 Local R2](../assets/figures/core/04_local_r2.png){ loading=lazy }
  <figcaption>04 Local R2</figcaption>
</figure>

<figure markdown>
  ![05 Standardized Residual](../assets/figures/core/05_standardized_residual.png){ loading=lazy }
  <figcaption>05 Standardized Residual</figcaption>
</figure>

<figure markdown>
  ![06 Cooks Distance](../assets/figures/core/06_cooks_distance.png){ loading=lazy }
  <figcaption>06 Cooks Distance</figcaption>
</figure>

<figure markdown>
  ![07 Gwr Condition Number](../assets/figures/core/07_gwr_condition_number.png){ loading=lazy }
  <figcaption>07 Gwr Condition Number</figcaption>
</figure>

<figure markdown>
  ![08 Lcr Lambda](../assets/figures/core/08_lcr_lambda.png){ loading=lazy }
  <figcaption>08 Lcr Lambda</figcaption>
</figure>

<figure markdown>
  ![09 Mgwr Bandwidths](../assets/figures/core/09_mgwr_bandwidths.png){ loading=lazy }
  <figcaption>09 Mgwr Bandwidths</figcaption>
</figure>

<figure markdown>
  ![10 Kernel Weights](../assets/figures/core/10_kernel_weights.png){ loading=lazy }
  <figcaption>10 Kernel Weights</figcaption>
</figure>

<figure markdown>
  ![11 Gwr Mgwr Comparison](../assets/figures/core/11_gwr_mgwr_comparison.png){ loading=lazy }
  <figcaption>11 Gwr Mgwr Comparison</figcaption>
</figure>

<figure markdown>
  ![12 Diagnostic Panel](../assets/figures/core/12_diagnostic_panel.png){ loading=lazy }
  <figcaption>12 Diagnostic Panel</figcaption>
</figure>

<figure markdown>
  ![01 Rgwr Weights](../assets/figures/specialized/01_rgwr_weights.png){ loading=lazy }
  <figcaption>01 Rgwr Weights</figcaption>
</figure>

<figure markdown>
  ![02 Rgwr Convergence](../assets/figures/specialized/02_rgwr_convergence.png){ loading=lazy }
  <figcaption>02 Rgwr Convergence</figcaption>
</figure>

<figure markdown>
  ![03 Gwglm Residuals](../assets/figures/specialized/03_gwglm_residuals.png){ loading=lazy }
  <figcaption>03 Gwglm Residuals</figcaption>
</figure>

<figure markdown>
  ![04 Gwlasso Frequency](../assets/figures/specialized/04_gwlasso_frequency.png){ loading=lazy }
  <figcaption>04 Gwlasso Frequency</figcaption>
</figure>

<figure markdown>
  ![05 Gwlasso Active](../assets/figures/specialized/05_gwlasso_active.png){ loading=lazy }
  <figcaption>05 Gwlasso Active</figcaption>
</figure>

<figure markdown>
  ![06 Gwlasso Alpha](../assets/figures/specialized/06_gwlasso_alpha.png){ loading=lazy }
  <figcaption>06 Gwlasso Alpha</figcaption>
</figure>

<figure markdown>
  ![07 Mixed Coefficients](../assets/figures/specialized/07_mixed_coefficients.png){ loading=lazy }
  <figcaption>07 Mixed Coefficients</figcaption>
</figure>

<figure markdown>
  ![08 Bootstrap Pvalues](../assets/figures/specialized/08_bootstrap_pvalues.png){ loading=lazy }
  <figcaption>08 Bootstrap Pvalues</figcaption>
</figure>

<figure markdown>
  ![09 Bootstrap Bandwidths](../assets/figures/specialized/09_bootstrap_bandwidths.png){ loading=lazy }
  <figcaption>09 Bootstrap Bandwidths</figcaption>
</figure>

<figure markdown>
  ![10 Scalable Kernel](../assets/figures/specialized/10_scalable_kernel.png){ loading=lazy }
  <figcaption>10 Scalable Kernel</figcaption>
</figure>

<figure markdown>
  ![11 Gwss Mean](../assets/figures/specialized/11_gwss_mean.png){ loading=lazy }
  <figcaption>11 Gwss Mean</figcaption>
</figure>

<figure markdown>
  ![12 Gwss Correlation](../assets/figures/specialized/12_gwss_correlation.png){ loading=lazy }
  <figcaption>12 Gwss Correlation</figcaption>
</figure>

<figure markdown>
  ![13 Gwpca Variance](../assets/figures/specialized/13_gwpca_variance.png){ loading=lazy }
  <figcaption>13 Gwpca Variance</figcaption>
</figure>

<figure markdown>
  ![14 Gwpca Loading](../assets/figures/specialized/14_gwpca_loading.png){ loading=lazy }
  <figcaption>14 Gwpca Loading</figcaption>
</figure>

<figure markdown>
  ![15 Gwda Class](../assets/figures/specialized/15_gwda_class.png){ loading=lazy }
  <figcaption>15 Gwda Class</figcaption>
</figure>

<figure markdown>
  ![16 Gwda Confidence](../assets/figures/specialized/16_gwda_confidence.png){ loading=lazy }
  <figcaption>16 Gwda Confidence</figcaption>
</figure>

<figure markdown>
  ![17 Gwda Confusion](../assets/figures/specialized/17_gwda_confusion.png){ loading=lazy }
  <figcaption>17 Gwda Confusion</figcaption>
</figure>

<figure markdown>
  ![18 Gtwr Slices](../assets/figures/specialized/18_gtwr_slices.png){ loading=lazy }
  <figcaption>18 Gtwr Slices</figcaption>
</figure>

<figure markdown>
  ![19 Gtwr Trajectory](../assets/figures/specialized/19_gtwr_trajectory.png){ loading=lazy }
  <figcaption>19 Gtwr Trajectory</figcaption>
</figure>

<figure markdown>
  ![20 Gtwr Residuals](../assets/figures/specialized/20_gtwr_residuals.png){ loading=lazy }
  <figcaption>20 Gtwr Residuals</figcaption>
</figure>

<figure markdown>
  ![21 Mgtwr Scales](../assets/figures/specialized/21_mgtwr_scales.png){ loading=lazy }
  <figcaption>21 Mgtwr Scales</figcaption>
</figure>

<figure markdown>
  ![22 Sgtwr Scales](../assets/figures/specialized/22_sgtwr_scales.png){ loading=lazy }
  <figcaption>22 Sgtwr Scales</figcaption>
</figure>

<figure markdown>
  ![23 Sgwr Weights](../assets/figures/specialized/23_sgwr_weights.png){ loading=lazy }
  <figcaption>23 Sgwr Weights</figcaption>
</figure>

<figure markdown>
  ![24 Sgwr Profiles](../assets/figures/specialized/24_sgwr_profiles.png){ loading=lazy }
  <figcaption>24 Sgwr Profiles</figcaption>
</figure>

<figure markdown>
  ![25 Stwr Weights](../assets/figures/specialized/25_stwr_weights.png){ loading=lazy }
  <figcaption>25 Stwr Weights</figcaption>
</figure>

<figure markdown>
  ![26 Sgtwr Weights](../assets/figures/specialized/26_sgtwr_weights.png){ loading=lazy }
  <figcaption>26 Sgtwr Weights</figcaption>
</figure>

<figure markdown>
  ![27 Lggwr Latent](../assets/figures/specialized/27_lggwr_latent.png){ loading=lazy }
  <figcaption>27 Lggwr Latent</figcaption>
</figure>

<figure markdown>
  ![28 Lggwr Metric](../assets/figures/specialized/28_lggwr_metric.png){ loading=lazy }
  <figcaption>28 Lggwr Metric</figcaption>
</figure>

<figure markdown>
  ![29 Lggwr Training](../assets/figures/specialized/29_lggwr_training.png){ loading=lazy }
  <figcaption>29 Lggwr Training</figcaption>
</figure>

<figure markdown>
  ![30 Lggwr Neighbours](../assets/figures/specialized/30_lggwr_neighbours.png){ loading=lazy }
  <figcaption>30 Lggwr Neighbours</figcaption>
</figure>

<figure markdown>
  ![31 Grgwr Regimes](../assets/figures/specialized/31_grgwr_regimes.png){ loading=lazy }
  <figcaption>31 Grgwr Regimes</figcaption>
</figure>

<figure markdown>
  ![32 Grgwr Convergence](../assets/figures/specialized/32_grgwr_convergence.png){ loading=lazy }
  <figcaption>32 Grgwr Convergence</figcaption>
</figure>

<figure markdown>
  ![33 Grgwr Sizes](../assets/figures/specialized/33_grgwr_sizes.png){ loading=lazy }
  <figcaption>33 Grgwr Sizes</figcaption>
</figure>

<figure markdown>
  ![34 Grgwr Coefficient](../assets/figures/specialized/34_grgwr_coefficient.png){ loading=lazy }
  <figcaption>34 Grgwr Coefficient</figcaption>
</figure>

<figure markdown>
  ![35 Model Diagnostics](../assets/figures/specialized/35_model_diagnostics.png){ loading=lazy }
  <figcaption>35 Model Diagnostics</figcaption>
</figure>

</div>

See the [Plotting API](../api/plotting/index.md) for every function signature and mapped example.
