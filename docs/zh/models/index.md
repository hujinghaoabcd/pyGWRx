# pyGWRx 模型手册

本手册详细说明 19 个正式公开模型，包括模型要解决的问题、数学形式、算法流程、pyGWRx 当前实现、适用场景、限制、推荐图件和完整可运行示例。

## 模型能力表

| 模型 | 类型 | 输入 | 新位置能力 |
|---|---|---|---|
| [`GWR`](gwr.md) | Classic local regression | X, y, coordinates | Validated local re-calibration at new coordinates. |
| [`MGWR`](mgwr.md) | Multiscale local regression | X, y, coordinates | Independent-target prediction is intentionally unavailable in the current validated API. |
| [`RGWR`](rgwr.md) | Robust local regression | X, y, coordinates | Validated local prediction using the fitted robust calibration state. |
| [`STWR`](stwr.md) | Stage-based spatiotemporal regression | Lists of X, y, and coordinates by stage, plus time intervals | Prediction for the current/latest stage using the fitted historical-stage weighting structure. |
| [`GTWR`](gtwr.md) | Row-wise spatiotemporal regression | X, y, coordinates, and row-wise times | Validated at new space-time targets; causal filtering is available when configured. |
| [`GWGLM`](gwglm.md) | Generalized local regression | X, response, coordinates; optional exposure for Poisson | Validated for Gaussian means, binomial probabilities, and Poisson means. |
| [`GWLasso`](gw-lasso.md) | Locally regularized regression | X, y, coordinates | Validated local prediction with the learned local penalties and scaling state. |
| [`MixedGWR`](mixed-gwr.md) | Semiparametric global-local regression | X, y, coordinates, and global/local variable assignments | Validated using global coefficients and re-estimated local components. |
| [`GWPCA`](gwpca.md) | Local multivariate transformation | Multivariate X and coordinates | Not a response predictor; `transform()` returns local component scores. |
| [`GWDA`](gwda.md) | Local spatial classification | X, class labels, coordinates | Validated class labels and local class probabilities. |
| [`GWSS`](gwss.md) | Local descriptive statistics | Multivariate X and coordinates | Not applicable; this is a local-statistics estimator. |
| [`ScalableGWR`](scalable-gwr.md) | Approximate scalable local regression | X, y, coordinates | Validated using the fitted scalable kernel approximation. |
| [`LCRGWR`](lcr-gwr.md) | Collinearity-compensated local regression | X, y, coordinates | Validated local prediction with fitted or locally adjusted ridge terms. |
| [`BootstrapGWR`](bootstrap-gwr.md) | Spatial inference | X, y, coordinates | Not applicable; the estimator performs coefficient-variability inference. |
| [`SGWR`](sgwr.md) | Geography-plus-similarity regression | X, y, coordinates, and similarity-variable specification | Validated by recomputing geographic and attribute-similarity weights for targets. |
| [`SGTWR`](sgtwr.md) | Geography-time-similarity regression | X, y, coordinates, times, and similarity variables | Validated at target space-time points with optional causal filtering. |
| [`MGTWR`](mgtwr.md) | Multiscale spatiotemporal regression | X, y, coordinates, times; optional per-column bandwidths and taus | 当前已验证 API 不提供独立目标位置预测；模型拟合与推断由 pyGWRx 内部实现。 |
| [`LGGWR`](lg-gwr.md) | Original research model | X, y, coordinates, and contextual attributes | Validated using the learned geometry transform and target attributes. |
| [`GRGWR`](gr-gwr.md) | Original research model | X, y, coordinates, regime count, and connectivity settings | Validated using learned regime structure and target assignment logic. |

## 使用原则

1. 先建立全局模型和标准 GWR 基线，再使用更复杂模型。
2. 固定带宽是距离，自适应带宽是近邻数，不能直接比较数值大小。
3. 局部系数不是自动的因果效应，必须结合不确定性、共线性和残差诊断。
4. 时空模型必须使用防止未来信息泄漏的验证方式。
5. LGGWR 和 GRGWR 是原创研究模型，应报告初始化、敏感性和当前验证边界。

[返回中文首页](../index.md) · [英文模型手册](../../models/index.md)
