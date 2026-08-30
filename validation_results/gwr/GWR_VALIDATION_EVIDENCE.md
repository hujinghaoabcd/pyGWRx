# pyGWRx Standard GWR Validation Evidence

## 1. Purpose and validation claim

This report consolidates the numerical evidence used to freeze the standard `pygwrx.GWR` implementation. It is deliberately narrower than a general benchmarking report: the goal is to show that the implemented GWR equations, diagnostics, prediction path, and discrete adaptive-bandwidth criteria reproduce independent implementations when mathematical definitions are aligned, while explicitly preserving known semantic differences instead of forcing artificial equality.

The validation is built in three layers:

1. a deterministic synthetic calibration fixture with independent prediction locations;
2. controlled full adaptive-bandwidth criterion curves;
3. a real-data Columbus, Ohio validation with genuinely held-out locations.

The frozen external implementations are `mgwr 2.2.1`, `GWmodel` and `spgwr`; the R-package versions are recorded in the generated reference JSON and real-data comparison artifacts. All external outputs are generated outside the pyGWRx estimator and then frozen under `tests/reference_data/gwr/` for blocking CI.

## 2. Reproducibility architecture

The evidence is separated into independent generation, frozen reference data, pyGWRx comparison, and blocking tests.

| Layer | Purpose | Main artifacts |
|---|---|---|
| External generation | Produce outputs with independently maintained software | `tools/reference/gwr/generate_mgwr.py`, `generate_r_references.R`, Columbus generator scripts |
| Frozen reference | Preserve external numerical outputs used by CI | `tests/reference_data/gwr/*.json`, `tests/reference_data/gwr/real_columbus/frozen/` |
| Machine comparison | Quantify absolute/relative differences and preserve semantic labels | `validation_results/gwr/comparison.csv/json`, bandwidth and Columbus comparison tables |
| Blocking reference CI | Fail if aligned quantities drift outside tolerance | `tests/test_gwr_external_references.py`, `tests/test_gwr_mgwr_deep_diagnostics_reference.py`, `tests/test_gwr_engine_numerical_lock.py`, Columbus reference tests |

Normal CI consumes frozen outputs and therefore does not need an R installation. Regeneration remains independently reproducible from the scripts above.

## 3. Layer A — deterministic calibration and prediction fixture

### 3.1 Experimental design

The first layer uses a deterministic 40-observation calibration dataset with two predictors and planar coordinates. Five additional locations are held out from calibration and used only for target-location local recalibration and prediction.

The principal aligned cases are:

| Case | Kernel | Bandwidth | Adaptive | Variance convention |
|---|---|---:|---|---|
| fixed Gaussian v2 | Gaussian | 55.0 | no | v2 |
| fixed bisquare v2 | bisquare | 70.0 | no | v2 |
| adaptive Gaussian v2 | Gaussian | 20 neighbours | yes | v2 |
| adaptive bisquare v2 | bisquare | 20 neighbours | yes | v2 |
| fixed Gaussian v1 | Gaussian | 55.0 | no | v1 |

Like-for-like calibration checks include local parameters, fitted values, residuals, local R-squared where definitions match, coefficient standard errors, t statistics, effective-complexity diagnostics, variance estimates, and target-location predictions.

### 3.2 Strict cross-software results

The existing synthetic comparison contains the following strict like-for-like summary:

| Reference | Strict checks | Worst maximum absolute difference | Worst metric/case |
|---|---:|---:|---|
| mgwr 2.2.1 | 91 | 6.029953e-06 | adaptive bisquare v2 / AIC |
| GWmodel | 42 | 9.683034e-07 | adaptive bisquare v2 / t values |
| spgwr | 10 | 4.018937e-08 | fixed bisquare v2 / parameters |

These counts exclude quantities whose definitions are intentionally different.

### 3.3 Deep mgwr smoother and influence diagnostics

The comparison artifacts already contained full `mgwr` references for the smoother matrix and influence diagnostics. This evidence is now promoted from report-only comparison to blocking CI for five cases: the four v2 fixed/adaptive Gaussian/bisquare cases plus fixed-Gaussian v1.

For the four principal v2 cases, the observed maximum absolute differences are:

| Case | Influence | Full hat matrix S | Standardized residuals | Cook's distance |
|---|---:|---:|---:|---:|
| fixed Gaussian | 5.262777e-10 | 5.262777e-10 | 1.650877e-08 | 4.005574e-09 |
| fixed bisquare | 3.823084e-09 | 3.823084e-09 | 8.763863e-08 | 2.720939e-08 |
| adaptive Gaussian | 1.772487e-08 | 1.772487e-08 | 5.000563e-08 | 1.187200e-08 |
| adaptive bisquare | 1.217395e-07 | 1.217395e-07 | 3.366341e-07 | 5.878987e-08 |

The worst observed deep-diagnostic absolute difference among these cases is approximately `3.37e-7`, for adaptive-bisquare standardized residuals. Blocking CI uses `atol=1e-6` and `rtol=1e-6`, leaving room for normal cross-platform floating-point variation while remaining materially tighter than typical statistical reporting precision.

The new reference test also checks two internal identities:

- `diag(S)` equals the stored influence vector;
- `S @ y` reproduces the fitted values.

This means the archived smoother is validated as an operational linear smoother, not merely as an array with matching shape.

### 3.4 Independent target-location prediction

For fixed Gaussian GWR, pyGWRx target-location local parameters and predictions are checked against all three external implementations at five independent prediction locations. The prediction data do not participate in calibration.

This specifically validates the `predict_result()` path rather than inferring prediction correctness from in-sample fitted values.

## 4. Layer B — controlled adaptive-bandwidth criterion curves

### 4.1 Why a controlled curve is necessary

Bandwidth selectors can differ because of optimizer choice, default search bounds, treatment of invalid boundary candidates, or adaptive-bandwidth parameterization. Therefore selected bandwidths alone are not sufficient evidence.

The controlled experiment evaluates and archives every integer adaptive candidate from `k=4` through `k=40` and compares criterion curves on explicitly defined common validity domains.

### 4.2 Validated criterion minima

| Criterion | pyGWRx | mgwr | GWmodel | Interpretation |
|---|---:|---:|---:|---|
| CV SSE | 15 | 15 | 15 | strict common finite domain |
| AIC | 5 | 5 | — | strict against mgwr |
| AICc | 22 | 22 | 22 | strict common valid domain |
| BIC | 5 | 5 | 5 | GWmodel value definition differs; argmin shown diagnostically |

The complete selector search trace is blocking-tested, not only the final argmin.

### 4.3 Curve-level numerical agreement

Representative strict curve differences are:

| Pair / criterion | Candidates | Maximum absolute difference | RMSE | Argmin match |
|---|---:|---:|---:|---|
| pyGWRx vs mgwr / CV SSE | 35 | 8.502968e-05 | 1.446513e-05 | yes, k=15 |
| pyGWRx vs GWmodel / CV SSE | 35 | 8.519511e-05 | 1.448530e-05 | yes, k=15 |
| pyGWRx vs mgwr / AIC | 36 | 3.647979e-05 | 1.239228e-05 | yes, k=5 |
| pyGWRx vs mgwr / AICc | 36 | 3.985457e-04 | 6.831670e-05 | yes, k=22 |
| pyGWRx vs GWmodel / AICc | 36 | 6.313387e-04 | 1.080019e-04 | yes, k=22 |
| pyGWRx vs mgwr / BIC | 36 | 3.577190e-05 | 1.041383e-05 | yes, k=5 |

### 4.4 Saturated low-bandwidth boundary

`k=4` is effectively saturated for the 40-observation fixture, with `trace(S)` approximately equal to `n`. Under the pyGWRx AICc validity rule, the denominator becomes non-positive and AICc is therefore non-finite. The candidate is retained in the raw search trace as an invalid boundary point rather than silently deleted.

GWmodel also lacks finite CV values at `k=4` and `k=5`; consequently the strict three-way CV domain begins at `k=6`.

This boundary case is important because a selector that simply chooses the smallest numerical criterion without mathematical validity checks can return a meaningless near-interpolating solution.

## 5. Layer C — Columbus real-data validation

### 5.1 Experimental design

The real-data layer uses 49 Columbus, Ohio neighbourhoods and the standard model:

`CRIME ~ INC + HOVAL`

with coordinates `X`, `Y`.

Four configurations are evaluated: fixed/adaptive by Gaussian/bisquare. In addition, five geographically dispersed neighbourhoods at zero-based rows `0, 10, 20, 30, 40` are removed from calibration and predicted from the remaining 44 observations.

This makes the prediction exercise genuinely out of calibration sample.

### 5.2 Strict numerical results

| Reference | Version | Strict checks | Worst maximum absolute difference | Worst case/metric |
|---|---|---:|---:|---|
| mgwr | 2.2.1 | 91 | 1.529442e-05 | adaptive bisquare v2 / parameters |
| GWmodel | 2.4.1 | 42 | 2.109976e-06 | fixed bisquare v2 / parameters |
| spgwr | 0.6.37 | 10 | 2.109975e-06 | fixed bisquare v2 / parameters |

The real-data results therefore reproduce independently maintained implementations at approximately `1e-5` or better on the strict comparison set, while preserving package-specific definitions where required.

### 5.3 Real-data adaptive-bandwidth results

For adaptive bisquare GWR, all integer candidates `k=4..49` are archived.

| Criterion | Reference | Raw argmin py/ref | k>=5 argmin py/ref | k>=5 maximum absolute difference |
|---|---|---:|---:|---:|
| CV SSE | mgwr | 11 / 11 | 11 / 11 | 1.538905e+02 |
| CV SSE | GWmodel | 11 / 11 | 11 / 11 | 1.538911e+02 |
| AIC | mgwr | 4 / 4 | 5 / 5 | 3.614848e-05 |
| AICc | mgwr | 24 / 24 | 24 / 24 | 1.010642e-02 |
| AICc | GWmodel | 24 / 24 | 24 / 24 | 1.280206e-02 |
| BIC | mgwr | 4 / 4 | 5 / 5 | 1.966922e-05 |

The comparatively larger absolute CV-SSE difference occurs on a criterion whose numerical scale is much larger; the archived real-data comparison reports maximum relative differences of about `1.03e-3` for CV while retaining identical selected minima.

The `k=4` near-saturated boundary is again retained transparently. For AIC/AICc/BIC interpretation, the report therefore distinguishes raw minima from the diagnostically meaningful `k>=5` domain.

## 6. Semantic differences that are intentionally not treated as failures

| Package/quantity | Difference | Validation treatment |
|---|---|---|
| GWmodel `Local_R2` | Different local-R² convention | archived, not forced equal |
| GWmodel AIC/BIC labels | Formula differs from the RSS/trace(S) formulation used by pyGWRx/mgwr | AICc is strict where aligned; AIC/BIC differences documented |
| spgwr adaptive bandwidth | Continuous sample proportion `q`, not integer neighbour order `k` | semantic sensitivity check, not strict adaptive equality |
| mgwr with `sigma2_v1=True` adjusted R² | Different ENP convention | sigma² compared strictly; adjusted R² difference explicitly preserved |
| saturated/invalid low-k candidates | Criterion may be mathematically undefined | candidate retained in trace; invalid value cannot win validated selection |

These exclusions are part of the validation design. They prevent false claims of disagreement when two packages are calculating different quantities under the same label.

## 7. Blocking CI coverage after this evidence completion

The GWR external-reference suite contains **56 tests marked `reference`** after C2 adds six direct private-engine reference cases on top of the existing 50-test external/public-GWR suite. It runs as a dedicated blocking CI job in addition to the normal regression, platform, minimum-dependency, build, coverage, documentation, quality, and security checks.

The external-reference layer now blocks regressions in:

- direct C1 private-engine calibration, inference, smoother/influence and independent-location prediction paths;
- local parameter estimation;
- fitted values and residuals;
- local R-squared where definitions match;
- standard errors and t statistics;
- R², adjusted R² under aligned convention, AIC/AICc/BIC where definitions align;
- `trace(S)`, `trace(S'S)`, effective parameters and residual variance;
- full smoother matrix `S` against mgwr;
- leverage/influence `diag(S)`;
- standardized residuals;
- Cook's distance;
- independent target-location local parameters and predictions;
- controlled discrete adaptive-bandwidth criterion traces and selected minima;
- near-saturated invalid-boundary handling;
- synthetic and Columbus real-data cases.

## 8. What this evidence supports

The combined evidence supports the following bounded claim:

> For standard Gaussian-family GWR under the validated fixed/adaptive Gaussian, bisquare, and fixed-tricube configurations, pyGWRx reproduces independently maintained GWR implementations to small floating-point tolerances when mathematical definitions and bandwidth semantics are aligned. The agreement extends beyond coefficients to smoother structure, influence diagnostics, inference, independent-location prediction, and controlled bandwidth-selection criteria. Known cross-package definition differences are explicitly separated from numerical disagreement.

This does **not** claim that all GWR packages implement every diagnostic identically, nor that `spgwr`'s adaptive proportion is equivalent to integer-k neighbourhood selection. It also does not substitute external agreement for statistical appropriateness on a new empirical dataset.

## 9. Source artifacts

For audit or paper preparation, use the following sources rather than copying numbers from this narrative without provenance:

- `validation_results/gwr/comparison.csv` and `.json` — synthetic calibration/prediction differences;
- `validation_results/gwr/gwr_validation_report.md` — compact synthetic summary;
- `validation_results/gwr/adaptive_bandwidth_criterion_curves.csv` — raw controlled criterion values;
- `validation_results/gwr/bandwidth_curve_comparison.csv` and `.json` — curve-level differences;
- `validation_results/gwr/bandwidth_curve_report.md` — controlled bandwidth interpretation;
- `validation_results/gwr/real_columbus/comparison.csv` and `.json` — real-data comparisons;
- `validation_results/gwr/real_columbus/gwr_columbus_validation_report.md` — real-data narrative;
- `tests/test_gwr_engine_numerical_lock.py` — C2 direct private-engine blocking reference lock;
- `tests/reference_data/gwr/` — frozen external outputs consumed by CI;
- `tools/reference/gwr/` — independent reference-generation and comparison scripts.

The detailed machine-readable tables remain authoritative for exact values; this document provides the experimental design, interpretation rules, and consolidated evidence chain.
