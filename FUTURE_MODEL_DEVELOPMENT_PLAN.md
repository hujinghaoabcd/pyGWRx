# pyGWRx Future Model Development Plan

> Status: FUTURE / POST-ARCHITECTURE ROADMAP  
> Date: 2026-08-29  
> Priority: subordinate to `ARCHITECTURE_FINAL_DECISION.md`

This document records model families that are **not yet implemented in pyGWRx** but are plausible future additions after the 0.2 architecture refactor is stable. It is intentionally separated from the architecture execution plan: architecture work must not expand scope by implementing these models early.

## 1. Why this roadmap exists

Several role bases in the final architecture currently have only one direct estimator:

- `BaseSpatialClassifier` → `GWDA`
- `BaseSpatialTransformer` → `GWPCA`
- `BaseSpatialStatistics` → `GWSS`
- `BaseSpatialInference` → `BootstrapGWR`

This does **not** mean these roles are dead ends. Existing software and literature show additional geographically weighted methods that can naturally fit these roles. The role bases should therefore remain thin and extensible, but no shared algorithm should be invented until at least two real estimators need it.

## 2. Future classification family

Current pyGWRx:

- `GWDA`

Candidate future estimators, in recommended order:

### C-F1. Geographically Weighted Random Forest Classifier

A geographically weighted random forest classifier is already implemented in the current PySAL spatial machine-learning stack. It fits local random-forest models over geographically defined neighbourhoods and exposes local feature importance and classification outputs.

Recommended future pyGWRx role:

```text
BaseSpatialClassifier
├── GWDA
└── GWRandomForestClassifier
```

Reference implementation / design evidence:

- PySAL spatialml / gwlearn: `GWRandomForestClassifier`
- https://pysal.org/spml/v0.2.1/ensemble.html
- https://pysal.org/spml/v0.2.1/api.html

### C-F2. Geographically Weighted Gradient Boosting Classifier

PySAL also provides a geographically weighted gradient boosting classifier using the same neighbourhood/local-model idea.

Potential role:

```text
BaseSpatialClassifier
├── GWDA
├── GWRandomForestClassifier
└── GWGradientBoostingClassifier
```

Reference:

- https://pysal.org/spml/v0.2.1/ensemble.html

### C-F3. Geographically Weighted Logistic Classifier

PySAL exposes `GWLogisticRegression`. pyGWRx already has binomial functionality under `GWGLM`, so a future classification-facing wrapper should only be considered if it provides a materially clearer classifier API (`predict`, `predict_proba`, class labels) without duplicating statistical implementation.

Reference:

- https://pysal.org/spml/v0.2.1/api.html

**Architecture decision:** do not create a generic "GW machine-learning base" now. First add one validated classifier, then extract shared local-estimator infrastructure only if real duplication appears.

## 3. Future transformer / dimension-reduction family

Current pyGWRx:

- `GWPCA`

### T-F1. Robust GWPCA

GWmodel directly supports both basic and robust GWPCA through the `robust` option. This is the strongest and most natural next transformer after standard GWPCA.

Recommended future structure:

```text
BaseSpatialTransformer
├── GWPCA
└── RobustGWPCA
```

Implementation rule:

- standard GWPCA numerical behaviour must be frozen first;
- robust GWPCA should be validated directly against GWmodel;
- do not implement it merely as a boolean mode hidden inside a giant shared base if separate estimator semantics make the API clearer.

References:

- GWmodel `gwpca`: https://search.r-project.org/CRAN/refmans/GWmodel/html/gwpca.html
- GWmodel `bw.gwpca`: https://search.r-project.org/CRAN/refmans/GWmodel/html/bw.gwpca.html

### T-F2. Additional local component methods

Sparse/non-negative/local nonlinear component methods may be evaluated later, but they are **research backlog**, not committed roadmap items until a stable reference implementation and validation target are identified.

## 4. Future spatial-statistics family

Current pyGWRx:

- `GWSS`

`GWSS` already contains a broad group of local descriptive statistics (means, variances, covariance/correlation, quantiles, etc.), so this family currently has **lower priority for adding separate estimators**.

Possible future work should first extend or modularize validated GWSS capabilities rather than create artificial classes simply to populate the hierarchy.

**Architecture decision:** keep `BaseSpatialStatistics` thin. Do not add new statistics estimators unless they represent a genuinely different public object with its own fitted/result semantics.

## 5. Future inference / significance-testing family

Current pyGWRx:

- `BootstrapGWR` (currently focused on the MLR null-model path)

This family has the clearest expansion path.

### I-F1. GWR Monte Carlo spatial-variability test

GWmodel provides `gwr.montecarlo`, a randomisation test for significant spatial variability of GWR coefficients.

Recommended future structure:

```text
BaseSpatialInference
├── BootstrapGWR
└── GWRMonteCarlo
```

Reference:

- https://search.r-project.org/CRAN/refmans/GWmodel/html/gwr.montecarlo.html

### I-F2. GWPCA Monte Carlo test

GWmodel provides Monte Carlo tests for spatial variability of GWPCA eigenvalues.

Potential role:

```text
BaseSpatialInference
├── BootstrapGWR
├── GWRMonteCarlo
└── GWPCAMonteCarlo
```

Reference:

- https://search.r-project.org/CRAN/refmans/GWmodel/html/gwpca.montecarlo.1.html

### I-F3. GWSS Monte Carlo test

GWmodel includes `gwss.montecarlo`, so GWSS significance testing should live in the inference family rather than be forced into the descriptive-statistics estimator itself.

Reference index:

- https://search.r-project.org/CRAN/refmans/GWmodel/html/00Index.html

### I-F4. Complete BootstrapGWR null-model support

GWmodel's bootstrap GWR supports four null hypotheses:

- MLR
- ERR
- SMA
- LAG

pyGWRx should eventually evaluate adding the remaining ERR/SMA/LAG paths after direct reference validation. These should be treated as extensions of the bootstrap inference engine, not as reasons to enlarge `BaseSpatialInference` with statistical formulae.

Reference:

- https://search.r-project.org/CRAN/refmans/GWmodel/html/gwr.bootstrap.html

## 6. Other post-0.2 regression backlog

These are outside the one-model role-base question but are worth recording for later evaluation because GWmodel exposes them and pyGWRx may eventually want broader methodological coverage:

- heteroskedastic GWR (`gwr.hetero`);
- GWR model selection workflows;
- multiple-testing / adjusted local significance utilities (`gwr.t.adjust`);
- additional Minkowski-distance selection workflows.

Reference index:

- https://search.r-project.org/CRAN/refmans/GWmodel/html/00Index.html

These are **not** part of the 0.2 architecture refactor.

## 7. Priority order after architecture stabilization

Recommended order after the 0.2 architecture migration and numerical freeze are complete:

1. `GWRMonteCarlo`
2. complete `BootstrapGWR` ERR/SMA/LAG support
3. `RobustGWPCA`
4. `GWRandomForestClassifier`
5. `GWPCAMonteCarlo`
6. `GWSSMonteCarlo`
7. `GWGradientBoostingClassifier`
8. evaluate classifier-facing `GWLogisticRegression`
9. evaluate lower-priority statistics/regression backlog

Rationale:

- inference methods reuse already validated GWR/GWPCA/GWSS estimators and therefore are relatively natural additions;
- RobustGWPCA has a clear mature GWmodel reference;
- geographically weighted machine-learning classifiers are valuable but introduce a larger dependency/API/performance design surface and should come only after the estimator architecture is stable.

## 8. Rules for adding any future model

No future model may be added solely because a role base currently has one child.

Every new estimator must satisfy all of the following:

1. Clear published mathematical definition.
2. At least one credible independent implementation or reproducible reference target where feasible.
3. Model-specific numerical validation plan before merge.
4. No public estimator inherits another public estimator.
5. Shared infrastructure is extracted only after real duplication exists.
6. Model math remains in a private model engine or model module, not in role bases.
7. Distance/neighbourhood semantics are explicitly documented.
8. Dense memory behaviour is documented and benchmarked if relevant.
9. Public result types remain model-specific unless a real shared contract is demonstrated.
10. Addition must not weaken existing frozen GWR/reference tests.

## 9. Relationship to the current refactor

This roadmap is **future scope only**.

The immediate architecture sequence remains unchanged:

```text
A1  Public API & Capability Snapshot for the 19 current estimators
A2  fitted-state atomicity freeze
A3  migration risk matrix
A4  performance/memory baseline
A5  contract/message hygiene
B... core architecture migration
C... GWR engine extraction
D... remove concrete-estimator inheritance
...
J... 0.2.0 consolidation
```

Do not implement any candidate in this document before the architecture execution ledger permits future-model work.

## 10. One-sentence future direction

> After pyGWRx 0.2 stabilizes the architecture, expand from a GWR-centered regression package toward a broader geographically weighted modelling toolkit covering regression, classification, local dimension reduction, descriptive statistics, and formal spatial-variability inference—without rebuilding the architecture for each new method.
