# Fitted-state atomicity contract

This file records the A2 safety-freeze contract for the 19 public estimators
captured by `architecture_contracts/estimators.json`.

## Contract

For every public estimator:

1. construction establishes a clean fitted-state baseline;
2. a successful `fit` may populate the estimator's fitted-state attributes;
3. if a later `fit` raises, the estimator must be unusable as fitted;
4. public fitted-state attributes initialized by the constructor must return to
   their constructor baseline rather than mixing old successful results with
   partial state from the failed refit;
5. shared training metadata (`n_samples_`, `n_features_in_`, and
   `feature_names_in_`) is part of fitted lifecycle state and must not remain
   stale after a failed refit.

The executable contract is
`tests/test_estimator_fitted_state_atomicity.py`, whose registry is required to
match all 19 estimators in the A1 machine-readable inventory.

## A2 characterization findings

The initial all-model characterization identified two classes of lifecycle
issues without changing any estimator formula:

- `MGWR` and `RGWR` validated per-fit options before clearing a previous fitted
  state, so an invalid fit option could leave the prior fitted model exposed;
- shared-base models including `GTWR`, `GWLasso`, `MGTWR`, and `MixedGWR`
  correctly became unfitted after a data-validation failure but retained stale
  feature/sample metadata.

A2 fixes these at the lifecycle layer:

- reset `MGWR` and `RGWR` before any per-fit validation that can raise;
- make `BaseSpatialEstimator._mark_unfitted()` clear the shared fitted metadata.

Model-specific numerical definitions, bandwidth semantics, solvers, reference
fixtures, and tolerances are unchanged.
