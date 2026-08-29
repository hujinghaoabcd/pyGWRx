# A3 — 19-model Migration Risk Matrix

**Baseline main SHA:** `b655688f7201aaa9677fe153f2cbc15e6e63afb6`

This document is the human-readable view of `migration_risks.json`.

## What “risk” means

Migration risk is **not model quality**. It is the expected refactor hazard:

> blast radius × mathematical/execution complexity × current architectural coupling.

The matrix is a planning/control artifact. It records the current structure as evidence, but **does not freeze the current MRO**. The target architecture remains the one in `ARCHITECTURE_FINAL_DECISION.md`.

## Distribution

- CRITICAL: 10 — GWR, MGWR, GTWR, MGTWR, GWGLM, STWR, SGTWR, ScalableGWR, LGGWR, GRGWR
- HIGH: 8 — RGWR, LCRGWR, GWLasso, MixedGWR, SGWR, GWPCA, GWDA, GWSS
- MEDIUM: 1 — BootstrapGWR
- LOW: 0

## Matrix

| Estimator | Target role | Current base | Risk | Planned phase | Public-estimator dependency | Specialized math / execution |
|---|---|---|---|---|---|---|
| GRGWR | regressor | object | CRITICAL | G1+ | — | graph construction; contiguous regime discovery |
| GTWR | regressor | BaseSpatiotemporalRegressor | CRITICAL | G1+ | — | space-time metric; lambda/tau/ksi semantics |
| GWGLM | regressor | GWR | CRITICAL | D3 | inherits public GWR | Gaussian/Poisson/Binomial family logic; local IWLS |
| GWR | regressor | BaseSpatialRegressor | CRITICAL | C1/C2 | — | unpenalized rank-aware local WLS; hat/influence/inference |
| LGGWR | regressor | object | CRITICAL | G1+ | — | joint/separable learned metric; latent geometry optimization |
| MGTWR | regressor | MGWR | CRITICAL | D4 | inherits public MGWR | multiscale backfitting; space-time metric |
| MGWR | regressor | BaseMultiscaleRegressor | CRITICAL | G1+ | — | multiscale backfitting; parameter-specific bandwidth search |
| SGTWR | regressor | object | CRITICAL | D5 | uses public GTWR instance as time conversion utility | GTWR-like space-time weighting; attribute similarity |
| STWR | regressor | object | CRITICAL | G1+ | — | response-variation temporal effect; temporal window/weight construction |
| ScalableGWR | regressor | object | CRITICAL | G1+ | — | ScaGWR compressed estimator; continuous basis/penalty optimization |
| GWDA | classifier | object | HIGH | G1+/F3 | — | local means/covariances/priors; weighted LDA/QDA |
| GWLasso | regressor | BaseSpatialRegressor | HIGH | G1+ | — | L1 coordinate descent; local standardization |
| GWPCA | transformer | object | HIGH | G1+/F3 | — | weighted local PCA; component sign canonicalization |
| GWSS | statistics | object | HIGH | G1+/F3 | — | GWmodel weighted moments; weighted quantiles |
| LCRGWR | regressor | GWR | HIGH | D2 | inherits public GWR | local condition-number diagnostics; locally compensated ridge |
| MixedGWR | regressor | BaseSpatialRegressor | HIGH | G1+ | — | global/local coefficient partition; partial regression/backfitting |
| RGWR | regressor | GWR | HIGH | D1 | inherits public GWR | automatic robust residual reweighting; filtered outlier refit |
| SGWR | regressor | object | HIGH | G1+ | uses public GWR internally for pure-GWR bandwidth selection | geographic-similarity weight mixture; alpha optimization |
| BootstrapGWR | inference | object | MEDIUM | G1+/F3 | composes public GWR fits | parametric bootstrap; modified/localized nonstationarity statistics |

## Migration rules derived from the matrix

1. **GWR is the golden architecture sample, not a universal algorithm engine.** C1/C2 must pass the frozen external GWR reference suite before D-family inheritance removal begins.
2. **MGWR/MGTWR are cache/backfitting models.** Do not impose “stream everything” on them.
3. **GTWR/SGTWR/STWR do not share one temporal algorithm.** B4 may share parsing/normalization only; STWR response-variation semantics remain model-owned.
4. **LCRGWR, GWLasso and GWGLM must keep distinct solver semantics.** Generic WLS primitives may be shared only where algebra is identical.
5. **ScalableGWR keeps its kNN/compressed path.** A migration that introduces an n×n pairwise-distance requirement is a regression even if numerical outputs match on small tests.
6. **GWPCA/GWDA/GWSS share neighbourhood infrastructure only.** Their bandwidth objectives and reference formulas remain family-owned.
7. **BootstrapGWR is already composition-oriented.** It should migrate late enough to consume the stable GWR engine/protocol instead of being rewritten early.
8. **SGWR/SGTWR dense output storage is a later performance-policy concern.** Architecture PRs must not silently change default statistical outputs while fixing class structure.

## Critical estimators and mandatory gates

### GWR
- Gate: blocking multi-software external references plus deep smoother diagnostics.
- Main danger: widest blast radius; used as the C1/C2 architecture exemplar.
- Forbidden shortcut: extracting a “universal GWR base engine” and making other estimators subclass it publicly.

### MGWR / MGTWR
- Gate: model-specific backfitting and smoother tests plus A1/A2.
- Main danger: multiscale backfitting, repeated distance reuse, parameter-specific search.
- Forbidden shortcut: replacing their repeated cache strategy with ordinary GWR row streaming without benchmark and numerical proof.

### GTWR / SGTWR / STWR
- Gate: each family’s own model tests plus A1/A2.
- Main danger: semantically different temporal constructions.
- Forbidden shortcut: one common spatiotemporal estimator base with algorithmic distance/fit behavior.

### GWGLM
- Gate: family/IWLS tests and explicit audit of the internal ridge policy.
- Main danger: current GWR inheritance hides a different iterative estimator.
- Forbidden shortcut: routing IWLS through ordinary Gaussian WLS just to simplify architecture.

### ScalableGWR
- Gate: explicit local weighted-regression references and the “no full pairwise distance matrix” test.
- Main danger: losing the package’s scalable execution path during infrastructure unification.

### LGGWR / GRGWR
- Gate: learned-geometry / regime-topology model tests.
- Main danger: treating learned geometry or graph topology as if they were ordinary distance-metric configuration.

## Machine-readable contract

`architecture_contracts/migration_risks.json` is the canonical A3 data artifact. The accompanying test enforces:

- exactly the same 19 estimators as the A1 estimator contract;
- required risk/schema fields for every estimator;
- allowed target roles and risk values;
- a rationale, prerequisites and specialized-math declaration for every estimator;
- no accidental use of A3 as a current-MRO compatibility freeze.

A future architecture PR may update `current_base`, debts or planned phase **only when the code actually changes**. Lowering a risk level should include evidence in the PR description; it must never be used to weaken numerical or lifecycle gates.
