# Choosing a model

Choose a model by the scientific task, response distribution, time structure, neighbourhood concept, and inferential goal. Do not start with the most flexible model.

## Step 1: what is the task?

| Task | Candidate models |
|---|---|
| Continuous-response local regression | GWR, MGWR, RGWR, LCRGWR, ScalableGWR, MixedGWR, GWLasso |
| Non-Gaussian local regression | GWGLM |
| Classification | GWDA |
| Local multivariate decomposition | GWPCA |
| Local descriptive statistics | GWSS |
| Coefficient non-stationarity inference | BootstrapGWR |
| Row-wise space-time regression | GTWR, SGTWR, MGTWR |
| Stage-based temporal regression | STWR |
| Geography plus attribute similarity | SGWR, SGTWR, LGGWR |
| Connected spatial regimes | GRGWR |

## Step 2: what mechanism needs to be added?

### One common spatial scale

Start with [GWR](../models/gwr.md). It is the reference model for continuous responses and the easiest local model to diagnose.

### Predictor-specific scales

Use [MGWR](../models/mgwr.md) when different variables plausibly operate at different spatial ranges. Remember that current independent-target prediction is unavailable.

### Outliers or local collinearity

- [RGWR](../models/rgwr.md): down-weights high-residual observations.
- [LCRGWR](../models/lcr-gwr.md): adds local ridge compensation where condition numbers are excessive.

Use data cleaning and substantive investigation before relying on either correction.

### Large sample size

Use [ScalableGWR](../models/scalable-gwr.md) when exact GWR is computationally prohibitive. Benchmark approximation accuracy on a manageable subset.

### Counts or binary outcomes

Use [GWGLM](../models/gwglm.md):

- Gaussian identity;
- Binomial logit;
- Poisson log with exposure support.

Use family-specific residuals and validation metrics.

### Global and local effects

Use [MixedGWR](../models/mixed-gwr.md) when theory supports a semiparametric partition. Automatic variable assignment is currently unavailable; specify global and local variables explicitly.

### Local sparse selection

Use [GWLasso](../models/gw-lasso.md) when the active predictor set may change across space. Inspect selection stability, not only one active-mask map.

### Time

| Data structure | Model | Main question |
|---|---|---|
| one time per row | GTWR | combined geographic-temporal neighbourhood |
| ordered snapshots/stages | STWR | history weighted by interval and process change |
| space + time + similarity | SGTWR | three notions of neighbourhood |
| coefficient-specific space-time scales | MGTWR | multiscale space-time effects |

Use future-safe validation. `MGTWR` currently exposes calibration-location results only.

### Functional similarity or learned geometry

- [SGWR](../models/sgwr.md): explicit convex combination of geographic and attribute-similarity weights.
- [SGTWR](../models/sgtwr.md): adds time.
- [LGGWR](../models/lg-gwr.md): learns latent geometry from coordinates and contextual attributes.

The similarity/attribute inputs must be defensible and available at prediction time.

### Spatial regimes

Use research model [GRGWR](../models/gr-gwr.md) when connected regions with abrupt mechanism changes are more plausible than a completely smooth surface. Test regime count, initialization, connectivity, and stability.

## Capability matrix

| Model | Primary task | New-location operation | Extra | Key caution |
|---|---|---|---|---|
| GWR | regression | `predict`, `predict_result` | base | one bandwidth for all coefficients |
| MGWR | multiscale regression | calibration only | base | expensive backfitting; no independent prediction |
| RGWR | robust regression | `predict`, `predict_result` | base | inspect robust weights and convergence |
| STWR | staged time regression | `predict`, `predict_result` | base | stage-list contract |
| GTWR | row-wise space-time regression | `predict`, `predict_result` | base | time scaling and leakage |
| GWGLM | Gaussian/binomial/Poisson | `predict`, `predict_result` | base | family and exposure semantics |
| GWLasso | sparse local regression | `predict` | `ml` | scaling and selection instability |
| MixedGWR | global + local effects | `predict` | base | explicit variable partition |
| GWPCA | local decomposition | `transform` | `ml` | loading sign/rotation and scaling |
| GWDA | local classification | `predict`, `predict_proba` | base | local class support |
| GWSS | local statistics | none | base | descriptive, not predictive |
| ScalableGWR | approximate regression | `predict`, `predict_result` | base | validate approximation |
| LCRGWR | ridge-compensated regression | `predict`, `predict_result` | base | penalty changes interpretation |
| BootstrapGWR | non-stationarity inference | none | base | Monte Carlo resolution and multiplicity |
| SGWR | geography + similarity | `predict`, `predict_result` | base | prevent outcome leakage |
| SGTWR | space + time + similarity | `predict`, `predict_result` | base | interacting scales and causal setting |
| MGTWR | multiscale space-time | calibration only | base | internal backfitting; no independent prediction |
| LGGWR | latent geometry | `predict`, `predict_result` | base | research identification and optimization |
| GRGWR | connected regimes | `predict`, `predict_result` | `ml` | regime-count and initialization sensitivity |

## A defensible comparison strategy

1. Define the intended prediction or inference task.
2. Use the same data, response coding, coordinate system, and validation split.
3. Compare a global baseline and standard GWR.
4. Add one mechanism at a time.
5. Compare predictive performance where supported.
6. Compare complexity, residual structure, uncertainty, and stability.
7. Prefer the simpler model when the specialized mechanism is not clearly supported.
