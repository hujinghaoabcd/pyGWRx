# Model handbook

pyGWRx exposes 19 supported public model classes. They share a consistent **fit → inspect → diagnose** style, but they do not all solve the same task and they do not all support independent-target prediction.

!!! warning "Choose by scientific question, not by model count"
    Begin with the simplest model that matches the response distribution, time structure, neighbourhood concept, and inference goal. More flexible models introduce more tuning parameters, identification risks, and validation requirements.

## Capability matrix

| Model | Family | Required data | New-location capability | Extra |
|---|---|---|---|---|
| [`GWR`](gwr.md) | Classic local regression | X, y, coordinates | Validated local re-calibration at new coordinates. | `base` |
| [`MGWR`](mgwr.md) | Multiscale local regression | X, y, coordinates | Independent-target prediction is intentionally unavailable in the current validated API. | `base` |
| [`RGWR`](rgwr.md) | Robust local regression | X, y, coordinates | Validated local prediction using the fitted robust calibration state. | `base` |
| [`STWR`](stwr.md) | Stage-based spatiotemporal regression | Lists of X, y, and coordinates by stage, plus time intervals | Prediction for the current/latest stage using the fitted historical-stage weighting structure. | `base` |
| [`GTWR`](gtwr.md) | Row-wise spatiotemporal regression | X, y, coordinates, and row-wise times | Validated at new space-time targets; causal filtering is available when configured. | `base` |
| [`GWGLM`](gwglm.md) | Generalized local regression | X, response, coordinates; optional exposure for Poisson | Validated for Gaussian means, binomial probabilities, and Poisson means. | `base` |
| [`GWLasso`](gw-lasso.md) | Locally regularized regression | X, y, coordinates | Validated local prediction with the learned local penalties and scaling state. | `ml` |
| [`MixedGWR`](mixed-gwr.md) | Semiparametric global-local regression | X, y, coordinates, and global/local variable assignments | Validated using global coefficients and re-estimated local components. | `base` |
| [`GWPCA`](gwpca.md) | Local multivariate transformation | Multivariate X and coordinates | Not a response predictor; `transform()` returns local component scores. | `ml` |
| [`GWDA`](gwda.md) | Local spatial classification | X, class labels, coordinates | Validated class labels and local class probabilities. | `base` |
| [`GWSS`](gwss.md) | Local descriptive statistics | Multivariate X and coordinates | Not applicable; this is a local-statistics estimator. | `base` |
| [`ScalableGWR`](scalable-gwr.md) | Approximate scalable local regression | X, y, coordinates | Validated using the fitted scalable kernel approximation. | `base` |
| [`LCRGWR`](lcr-gwr.md) | Collinearity-compensated local regression | X, y, coordinates | Validated local prediction with fitted or locally adjusted ridge terms. | `base` |
| [`BootstrapGWR`](bootstrap-gwr.md) | Spatial inference | X, y, coordinates | Not applicable; the estimator performs coefficient-variability inference. | `base` |
| [`SGWR`](sgwr.md) | Geography-plus-similarity regression | X, y, coordinates, and similarity-variable specification | Validated by recomputing geographic and attribute-similarity weights for targets. | `base` |
| [`SGTWR`](sgtwr.md) | Geography-time-similarity regression | X, y, coordinates, times, and similarity variables | Validated at target space-time points with optional causal filtering. | `base` |
| [`MGTWR`](mgtwr.md) | Multiscale spatiotemporal regression | X, y, coordinates, times; optional per-column bandwidths and taus | Independent-target prediction is intentionally unavailable in the current validated API. | `base` |
| [`LGGWR`](lg-gwr.md) | Original research model | X, y, coordinates, and contextual attributes | Validated using the learned geometry transform and target attributes. | `base` |
| [`GRGWR`](gr-gwr.md) | Original research model | X, y, coordinates, regime count, and connectivity settings | Validated using learned regime structure and target assignment logic. | `ml` |

## Recommended progression

1. **Continuous response:** begin with [`GWR`](gwr.md).
2. **Different spatial scales:** compare [`MGWR`](mgwr.md).
3. **Outliers or local collinearity:** consider [`RGWR`](rgwr.md) or [`LCRGWR`](lcr-gwr.md).
4. **Non-Gaussian response:** use [`GWGLM`](gwglm.md) or [`GWDA`](gwda.md).
5. **Time:** choose row-wise [`GTWR`](gtwr.md), stage-based [`STWR`](stwr.md), or multiscale [`MGTWR`](mgtwr.md).
6. **Alternative neighbourhoods:** use [`SGWR`](sgwr.md), [`SGTWR`](sgtwr.md), or research model [`LGGWR`](lg-gwr.md).
7. **Discrete spatial mechanisms:** evaluate research model [`GRGWR`](gr-gwr.md).
8. **Exploration/inference rather than prediction:** use [`GWSS`](gwss.md), [`GWPCA`](gwpca.md), or [`BootstrapGWR`](bootstrap-gwr.md).

## Shared interpretation rules

- A local coefficient is a conditional association under a chosen neighbourhood, not an automatic causal effect.
- Bandwidth, kernel, distance units, and local effective sample size are part of the model specification.
- Map uncertainty, local collinearity, influence, and residuals alongside coefficient surfaces.
- Use spatially blocked or temporally ordered validation when the intended use requires spatial or temporal transfer.
- Treat 0.x APIs and original research models as evolving; record the exact package version and configuration.

See [Choosing a model](../getting-started/choosing-a-model.md), [Diagnostics](../guides/diagnostics.md), and the [Chinese model handbook](../zh/models/index.md).
