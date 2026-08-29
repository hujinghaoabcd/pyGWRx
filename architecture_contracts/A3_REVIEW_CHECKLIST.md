# A3 Review Checklist

A3 is planning/contract work only. It must not change estimator formulas, public signatures, numerical tolerances, or execution algorithms.

- [x] Exactly 19 estimators, matched to A1 `estimators.json`.
- [x] Each estimator has target role, current structural evidence, migration risk, planned phase, numerical gate, execution profile, specialized mathematics, prerequisites, debts, and rationale.
- [x] Current MRO is recorded as evidence only and is explicitly not frozen as compatibility API.
- [x] Known public-estimator inheritance debts are explicit: RGWR→GWR, LCRGWR→GWR, GWGLM→GWR, MGTWR→MGWR.
- [x] Public-estimator-as-utility dependencies are explicit where architecture work must remove them.
- [x] GWR streaming, MGWR cache reuse, ScalableGWR kNN/compressed execution, SGWR dense-weight behavior, LGGWR learned geometry, and GRGWR graph topology remain visible.
- [x] A1 public API and A2 fitted-state atomicity remain mandatory gates.
- [x] Cross-window handoff is now a repository-level mandatory workflow rule.

No risk level authorizes weakening a numerical/reference/lifecycle gate. A future PR that lowers risk should cite concrete evidence for the reduction.
